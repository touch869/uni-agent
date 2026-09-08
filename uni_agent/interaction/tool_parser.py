import ast
import hashlib
import json
from typing import Any

import regex

from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)


class FunctionCallFormatError(Exception):
    pass


def deterministic_tool_call_id(
    *,
    step_idx: int,
    call_index: int,
    name: str,
    arguments: Any,
) -> str:
    """Return a stable OpenAI-compatible ID for one logical tool call."""
    payload = {
        "step_idx": int(step_idx),
        "call_index": int(call_index),
        "name": name,
        "arguments": arguments,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"call_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _strict_json_loads(value: str) -> Any:
    """Load standards-compliant JSON (Python otherwise accepts NaN/Infinity)."""
    return json.loads(value, parse_constant=_reject_json_constant)


def _next_non_whitespace(value: str, start: int) -> str | None:
    for char in value[start:]:
        if not char.isspace():
            return char
    return None


def _quote_can_close(value: str, quote_index: int) -> bool:
    cursor = quote_index + 1
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    if cursor == len(value) or value[cursor] in (":", "}", "]"):
        return True
    if value[cursor] != ",":
        return False

    cursor += 1
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    # A comma inside shell/Python text is normally followed by an identifier.
    # A JSON member or the next string array item starts with a quote.
    return cursor == len(value) or value[cursor] in ('"', "}", "]")


def _previous_non_whitespace(value: list[str]) -> str | None:
    for char in reversed(value):
        if not char.isspace():
            return char
    return None


def _repair_json_syntax(value: str) -> str | None:
    """Repair only locally unambiguous JSON-string mistakes.

    This deliberately is not a general JSON-repair parser. It handles the
    failures repeatedly produced by the SWE-Lego model: raw control characters
    or invalid backslashes inside strings, quotes inside a string whose next
    token cannot legally follow a closing quote, and one stray quote between
    closing containers. The caller must parse and schema-validate the result.
    """
    output: list[str] = []
    in_string = False
    containers: list[str] = []
    repairs = 0
    index = 0
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    control_escapes = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}

    while index < len(value):
        char = value[index]
        if not in_string:
            if char == '"':
                previous = _previous_non_whitespace(output)
                following = _next_non_whitespace(value, index + 1)
                if previous in ("}", "]") and (following in ("}", "]") or following is None):
                    repairs += 1
                    index += 1
                    continue
                in_string = True
            elif char in ("{", "["):
                containers.append("}" if char == "{" else "]")
            elif char in ("}", "]"):
                if not containers or containers[-1] != char:
                    return None
                containers.pop()
            output.append(char)
            index += 1
            continue

        if char == "\\":
            following = value[index + 1] if index + 1 < len(value) else None
            valid_unicode = (
                following == "u"
                and index + 5 < len(value)
                and all(candidate in "0123456789abcdefABCDEF" for candidate in value[index + 2 : index + 6])
            )
            if valid_unicode:
                output.extend(value[index : index + 6])
                index += 6
                continue
            if following in valid_escapes:
                output.extend((char, following))
                index += 2
                continue

            output.append("\\\\")
            repairs += 1
            index += 1
            continue

        if char == '"':
            if _quote_can_close(value, index):
                in_string = False
                output.append(char)
            else:
                output.append('\\"')
                repairs += 1
            index += 1
            continue

        if ord(char) < 0x20:
            escaped = control_escapes.get(char)
            if escaped is None:
                escaped = f"\\u{ord(char):04x}"
            output.append(escaped)
            repairs += 1
        else:
            output.append(char)
        index += 1

    if in_string:
        return None
    if containers:
        output.extend(reversed(containers))
        repairs += len(containers)
    if repairs == 0:
        return None
    return "".join(output)


def load_json_object(value: str, *, context: str) -> tuple[dict[str, Any], bool]:
    """Strictly load a JSON object, with one conservative repair attempt."""
    try:
        parsed = _strict_json_loads(value)
        repaired = False
    except (json.JSONDecodeError, ValueError) as original_error:
        repaired_value = _repair_json_syntax(value)
        if repaired_value is None:
            raise FunctionCallFormatError(f"Invalid {context} JSON: {original_error}.") from None
        try:
            parsed = _strict_json_loads(repaired_value)
        except (json.JSONDecodeError, ValueError):
            raise FunctionCallFormatError(f"Invalid {context} JSON: {original_error}.") from None
        repaired = True

    if not isinstance(parsed, dict):
        raise FunctionCallFormatError(f"Invalid {context}: expected a JSON object, got {type(parsed).__name__}.")
    return parsed, repaired


def validate_function_call(
    name: str,
    arguments: dict[str, Any],
    tools: list[OpenAIFunctionToolSchema],
) -> None:
    """Validate function arguments against the advertised tool schema."""
    tool = next(
        (candidate for candidate in tools if candidate.type == "function" and candidate.function.name == name),
        None,
    )
    if tool is None:
        valid_names = sorted(candidate.function.name for candidate in tools if candidate.type == "function")
        raise FunctionCallFormatError(
            f"Invalid action: function '{name}' is not defined in the tools list. Allowed functions: {valid_names}."
        )

    parameters = tool.function.parameters
    properties = parameters.properties
    missing = [parameter for parameter in parameters.required if parameter not in arguments]
    if missing:
        raise FunctionCallFormatError(f"Invalid arguments for '{name}': missing required parameter(s): {missing}.")

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise FunctionCallFormatError(
            f"Invalid arguments for '{name}': unknown parameter(s): {unknown}. "
            f"Allowed parameters: {sorted(properties)}."
        )

    type_checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) in (int, float),
        "boolean": lambda item: type(item) is bool,
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    for parameter, argument in arguments.items():
        property_schema = properties[parameter]
        expected_type = property_schema.type.lower()
        checker = type_checks.get(expected_type)
        if checker is not None and not checker(argument):
            raise FunctionCallFormatError(
                f"Invalid argument '{parameter}' for '{name}': expected {expected_type}, got {type(argument).__name__}."
            )
        if property_schema.enum is not None and argument not in property_schema.enum:
            raise FunctionCallFormatError(
                f"Invalid argument '{parameter}' for '{name}': value {argument!r} is not in {property_schema.enum}."
            )


# modified from qwen3 coder tool parser
class XMLToolParser:
    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_prefix: str = "<function="

        # Regex patterns
        self.tool_call_complete_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", regex.DOTALL)
        self.tool_call_function_regex = regex.compile(r"<function=(.*?)</function>|<function=(.*)$", regex.DOTALL)
        self.tool_call_parameter_regex = regex.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", regex.DOTALL
        )

    def _get_arguments_config(self, func_name: str, tools: list[OpenAIFunctionToolSchema]) -> dict:
        for config in tools:
            assert config.type == "function"
            if config.function.name == func_name:
                properties = config.function.parameters.properties
                return {k: v.model_dump() for k, v in properties.items()}
        raise FunctionCallFormatError(
            f"Invalid action: function '{func_name}' is not defined in the tools list.\n"
            f"Allowed functions should be one of: {[tool.function.name for tool in tools]}."
        )

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                raise FunctionCallFormatError(
                    f"Invalid action: parameter '{param_name}' is not defined "
                    f"in the parameters for function '{func_name}'.\n"
                    f"Allowed parameters for function '{func_name}': {list(param_config.keys())}."
                )
            return param_value

        if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
                return param_value
            except Exception:
                raise FunctionCallFormatError(
                    f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                    f"is not an integer in tool call '{func_name}'."
                ) from None
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                param_value = (
                    float_param_value if float_param_value - int(float_param_value) != 0 else int(float_param_value)
                )
                return param_value
            except Exception:
                raise FunctionCallFormatError(
                    f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                    f"is not a float in tool call '{func_name}'."
                ) from None
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value in ["true", "false"]:
                return param_value == "true"
            raise FunctionCallFormatError(
                f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                f"is not a boolean (`true` of `false`) in tool call '{func_name}'."
            )
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    pass
            try:
                param_value = ast.literal_eval(param_value)  # safer
                return param_value
            except Exception:
                raise FunctionCallFormatError(
                    f"Invalid action: value '{param_value}' of parameter '{param_name}' "
                    f"is not valid in tool call '{func_name}'."
                ) from None

    def _parse_xml_function_call(
        self,
        function_call_str: str,
        tools: list[OpenAIFunctionToolSchema],
        call_index: int,
    ) -> OpenAIFunctionToolCall:
        # Extract function name
        if ">" not in function_call_str:
            raise FunctionCallFormatError("Invalid function call format: Cannot find function name.")
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            if ">" not in match_text:
                raise FunctionCallFormatError(
                    f"Invalid function call format: Cannot find parameter name in tool call '{function_name}'."
                )
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)

        validate_function_call(function_name, param_dict, tools)
        function_call = OpenAIFunctionCallSchema(name=function_name, arguments=param_dict)
        tool_call = OpenAIFunctionToolCall(
            id=deterministic_tool_call_id(
                step_idx=0,
                call_index=call_index,
                name=function_name,
                arguments=param_dict,
            ),
            type="function",
            function=function_call,
        )
        return tool_call

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Return ``<function=...>`` bodies found inside ``<tool_call>`` blocks.
        Empty list = nothing recoverable (caller treats as "no tool calls").
        """
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]
        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))
        return [match[0] if match[0] else match[1] for match in raw_function_calls]

    def extract_tool_calls(
        self, model_output: str, tools: list[OpenAIFunctionToolSchema]
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse ``<tool_call>...</tool_call>`` blocks out of ``model_output``.

        Returns ``(content_before_marker, tool_calls)``. ``tool_calls`` is
        ``[]`` whenever nothing parseable comes out -- whether the marker
        was absent or present-but-unrecoverable. Callers decide what to
        do (single-shot raises, chat treats as turn-end). When a function
        name IS recovered but invalid (unknown name, bad arg type, ...)
        we still raise :class:`FunctionCallFormatError`.
        """
        if self.tool_call_start_token not in model_output:
            return model_output, []

        function_calls = self._get_function_calls(model_output)
        if not function_calls:
            return model_output, []
        tool_calls = [
            self._parse_xml_function_call(function_call_str, tools, call_index)
            for call_index, function_call_str in enumerate(function_calls)
        ]

        content_index = model_output.find(self.tool_call_start_token)
        content_index = content_index if content_index >= 0 else model_output.find(self.tool_call_prefix)
        content = model_output[:content_index]

        return content, tool_calls


class HermesToolParser:
    """Parser for the Hermes JSON tool-call format.
    Expected format::
        <tool_call>
        {"name": "<function-name>", "arguments": {...}}
        </tool_call>
    """

    _FORMAT_HINT = (
        'Expected format:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )

    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", regex.DOTALL)

    def extract_tool_calls(
        self, model_output: str, tools: list[OpenAIFunctionToolSchema]
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse ``<tool_call>{...}</tool_call>`` JSON blocks out of ``model_output``.

        Returns ``(content_before_marker, tool_calls)``. ``tool_calls`` is
        ``[]`` when nothing parseable comes out (no marker, or marker
        pair with empty body); callers decide. The "unclosed" case
        (``<tool_call>`` opened with partial JSON, no closing tag) DOES
        raise -- partial JSON is a clear formatting bug worth surfacing
        back to the model.
        """
        if self.tool_call_start_token not in model_output:
            return model_output, []
        if self.tool_call_end_token not in model_output:
            raise FunctionCallFormatError(
                f"Unclosed tool call: missing {self.tool_call_end_token}. {self._FORMAT_HINT}"
            )

        matches = [m for m in self.tool_call_regex.findall(model_output) if m.strip()]
        if not matches:
            return model_output, []

        tool_calls: list[OpenAIFunctionToolCall] = []
        for call_index, raw in enumerate(matches):
            tool_calls.append(self._parse_single(raw, tools, call_index))

        content_index = model_output.find(self.tool_call_start_token)
        return model_output[:content_index], tool_calls

    def _parse_single(
        self,
        raw: str,
        tools: list[OpenAIFunctionToolSchema],
        call_index: int = 0,
    ) -> OpenAIFunctionToolCall:
        try:
            obj, _ = load_json_object(raw, context="tool_call")
        except FunctionCallFormatError as error:
            raise FunctionCallFormatError(f"{error} {self._FORMAT_HINT}") from None
        if "name" not in obj:
            raise FunctionCallFormatError(f"Invalid tool_call: missing 'name' field. {self._FORMAT_HINT}")

        name = obj["name"]
        if not isinstance(name, str):
            raise FunctionCallFormatError(f"Invalid tool_call: 'name' must be a string, got {type(name).__name__}.")

        arguments: Any = obj.get("arguments", {})
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            # Some models double-encode arguments as a JSON string; accept that.
            try:
                arguments, _ = load_json_object(arguments, context=f"arguments for '{name}'")
            except FunctionCallFormatError:
                raise
        if not isinstance(arguments, dict):
            raise FunctionCallFormatError(
                f"Invalid arguments for '{name}': expected a JSON object, got {type(arguments).__name__}."
            )

        validate_function_call(name, arguments, tools)
        function_call = OpenAIFunctionCallSchema(name=name, arguments=arguments)
        return OpenAIFunctionToolCall(
            id=deterministic_tool_call_id(
                step_idx=0,
                call_index=call_index,
                name=name,
                arguments=arguments,
            ),
            type="function",
            function=function_call,
        )


_PARSER_REGISTRY: dict[str, type] = {
    "qwen3_coder": XMLToolParser,
    "hermes": HermesToolParser,
}


def get_tool_parser(name: str):
    """Instantiate a tool-call parser by registered name."""
    if name not in _PARSER_REGISTRY:
        raise ValueError(f"Unknown tool parser: {name!r}. Available parsers: {sorted(_PARSER_REGISTRY.keys())}.")
    return _PARSER_REGISTRY[name]()
