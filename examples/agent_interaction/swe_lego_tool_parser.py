"""Evidence-preserving tool parser adapter for SWE-Lego inference only."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from uni_agent.interaction.tool_parser import FunctionCallFormatError, HermesToolParser
from uni_agent.interaction.tool_schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema


class SWELegoToolParser(HermesToolParser):
    """Strict Hermes parser with narrow recovery for observed SWE-Lego output.

    Raw model text is never changed. Recovery metadata is retained so evidence
    can distinguish native valid calls from adapter-assisted calls.
    """

    _LEGACY_RE = re.compile(r"<(execute_bash|submit)>\s*(.*?)\s*</\1>", re.DOTALL)
    _FUNCTION_CALL_RE = re.compile(r"<functionCall>\s*(.*?)\s*</functionCall>", re.DOTALL)
    _PARAM_RE = re.compile(r"<parameter>\s*([a-z_]+)\"?>\s*(.*?)\s*</parameter>", re.DOTALL)
    _NAME_RE = re.compile(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']')
    _COMMAND_PREFIX_RE = re.compile(
        r'["\']arguments["\']\s*:\s*\{\s*["\']command["\']\s*:\s*(["\'])',
        re.DOTALL,
    )
    _CREATE_RE = re.compile(
        r"[\"' ]command[\"' ]\s*:\s*[\"' ]create[\"' ]\s*,\s*"
        r"[\"' ]file_text[\"' ]\s*:\s*[\"' ](.*)[\"' ]\s*,\s*"
        r"[\"' ]path[\"' ]\s*:\s*[\"' ]([^\"' ]+)[\"' ]\s*}\s*}\s*$",
        re.DOTALL,
    )

    def __init__(self):
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.last_event: dict[str, Any] | None = None

    @staticmethod
    def _response_id(model_output: str) -> str:
        return hashlib.sha256(model_output.encode("utf-8")).hexdigest()[:16]

    def _record(
        self,
        model_output: str,
        mode: str,
        calls: list[OpenAIFunctionToolCall],
        strict_error: str | None = None,
    ) -> None:
        event = {
            "response_id": self._response_id(model_output),
            "mode": mode,
            "strict_error": strict_error,
            "recovered_calls": [call.function.name for call in calls],
        }
        self.last_event = event
        if mode != "strict_hermes":
            self.events.append(event)

    def _parse_payload(
        self, name: str, arguments: dict[str, Any], valid_names: set[str]
    ) -> OpenAIFunctionToolCall:
        raw = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        return self._parse_single(raw, valid_names)

    def _recover_legacy(
        self, model_output: str, valid_names: set[str]
    ) -> list[OpenAIFunctionToolCall]:
        calls = []
        for name, body in self._LEGACY_RE.findall(model_output):
            if name not in valid_names:
                continue
            arguments = {} if name == "submit" else {"command": body.strip()}
            if name == "execute_bash" and not arguments["command"]:
                continue
            calls.append(self._parse_payload(name, arguments, valid_names))
        return calls

    @staticmethod
    def _strip_command_suffix(value: str, quote: str) -> str | None:
        value = value.rstrip()
        for suffix in (quote + "}}", "</span>}}"):
            if value.endswith(suffix):
                return value[: -len(suffix)]
        return None

    def _recover_function_call(
        self, model_output: str, valid_names: set[str]
    ) -> list[OpenAIFunctionToolCall]:
        calls = []
        for block in self._FUNCTION_CALL_RE.findall(model_output):
            params = {key: value.strip() for key, value in self._PARAM_RE.findall(block)}
            name = params.get("name", "")
            if name not in valid_names:
                continue
            if name == "submit":
                arguments = {}
            elif "arguments" in params:
                try:
                    arguments = json.loads(params["arguments"])
                except json.JSONDecodeError:
                    continue
            elif name == "execute_bash" and params.get("command"):
                arguments = {"command": params["command"]}
            else:
                continue
            if not isinstance(arguments, dict):
                continue
            calls.append(self._parse_payload(name, arguments, valid_names))
        return calls

    def _recover_hermes(
        self, model_output: str, valid_names: set[str]
    ) -> list[OpenAIFunctionToolCall]:
        calls = []
        matches = [m for m in self.tool_call_regex.findall(model_output) if m.strip()]
        for raw in matches:
            name_match = self._NAME_RE.search(raw)
            if not name_match:
                continue
            name = name_match.group(1)
            if name in {"view", "create", "str_replace", "insert", "undo_edit"} and "str_replace_editor" in valid_names:
                candidates = [raw]
                if raw.rstrip().endswith("}}}"):
                    candidates.append(raw.rstrip()[:-1])
                for candidate in candidates:
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    arguments = obj.get("arguments") or {key: value for key, value in obj.items() if key != "name"}
                    if not isinstance(arguments, dict):
                        continue
                    arguments["command"] = name
                    calls.append(self._parse_payload("str_replace_editor", arguments, valid_names))
                    break
                if calls:
                    continue

            if name not in valid_names and "execute_bash" in valid_names:
                if re.search(r"[\s/|&;]", name) and "arguments" not in raw:
                    calls.append(self._parse_payload("execute_bash", {"command": name}, valid_names))
                    continue

            if name == "execute_bash":
                prefix = self._COMMAND_PREFIX_RE.search(raw)
                if not prefix:
                    continue
                command = self._strip_command_suffix(raw[prefix.end() :], prefix.group(1))
                if command is None or not command.strip():
                    continue
                calls.append(self._parse_payload(name, {"command": command}, valid_names))
                continue

            if name == "str_replace_editor":
                create_match = self._CREATE_RE.search(raw)
                if create_match:
                    calls.append(
                        self._parse_payload(
                            name,
                            {
                                "command": "create",
                                "file_text": create_match.group(1),
                                "path": create_match.group(2),
                            },
                            valid_names,
                        )
                    )
                    continue

            # One observed editor error inserts a quote between the arguments
            # object and outer object: ...]}"} instead of ...]}}.
            if name == "str_replace_editor" and raw.rstrip().endswith('}"}'):
                repaired = raw.rstrip()[:-2] + "}"
                try:
                    calls.append(self._parse_single(repaired, valid_names))
                except FunctionCallFormatError:
                    continue
        return calls

    def extract_tool_calls(
        self, model_output: str, tools: list[OpenAIFunctionToolSchema]
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        self.last_event = None
        valid_names = {tool.function.name for tool in tools if tool.type == "function"}
        strict_error = None
        try:
            content, calls = super().extract_tool_calls(model_output, tools)
            if calls:
                self._record(model_output, "strict_hermes", calls)
                return content, calls
        except FunctionCallFormatError as exc:
            strict_error = f"{type(exc).__name__}: {exc}"

        calls = self._recover_function_call(model_output, valid_names)
        if calls:
            self._record(model_output, "legacy_function_call", calls, strict_error)
            return model_output[: model_output.find("<functionCall>")], calls

        calls = self._recover_legacy(model_output, valid_names)
        if calls:
            self._record(model_output, "legacy_xml", calls, strict_error)
            first = min(model_output.find(f"<{call.function.name}>") for call in calls)
            return model_output[:first], calls

        calls = self._recover_hermes(model_output, valid_names)
        if calls:
            self._record(model_output, "repaired_hermes", calls, strict_error)
            return model_output[: model_output.find(self.tool_call_start_token)], calls

        if strict_error is not None:
            raise FunctionCallFormatError(strict_error) from None
        self._record(model_output, "no_tool_call", [])
        return model_output, []
