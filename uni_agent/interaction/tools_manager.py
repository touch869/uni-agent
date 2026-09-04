import json
import shlex

from pydantic import BaseModel, ConfigDict

from uni_agent.interaction.tool_parser import (
    FunctionCallFormatError,
    deterministic_tool_call_id,
    get_tool_parser,
    load_json_object,
    validate_function_call,
)
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)
from uni_agent.tools import ToolConfig


class ToolsManagerConfig(BaseModel):
    """Config for the tools list."""

    tools: list[ToolConfig]
    parser: str = "qwen3_coder"
    """Name of the registered tool-call parser. Built-in: "qwen3_coder", "hermes"."""
    model_config = ConfigDict(extra="ignore")


class ToolsManager:
    """Builds tool instances and OpenAI tool schemas from ToolsConfig."""

    def __init__(self, tools_manager_config: ToolsManagerConfig):
        self.tools_manager_config = tools_manager_config
        self.tools = [tc.get_tool() for tc in tools_manager_config.tools]
        self.tools_schemas = [t.get_tool_schema() for t in self.tools]
        self._tool_parser = get_tool_parser(tools_manager_config.parser)

    async def parse_action(
        self,
        model_output: str,
        step_idx: int = 0,
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse tool calls from raw text. Returns ``(content, tool_calls)``;
        ``tool_calls`` is ``[]`` when the text contains no tool-call
        marker (callers decide -- single-shot raises, chat_mode treats
        as turn-end). Markers that ARE present but malformed raise
        :class:`FunctionCallFormatError`.
        """
        tools = [OpenAIFunctionToolSchema(**schema) for schema in self.tools_schemas]
        content, tool_calls = self._tool_parser.extract_tool_calls(model_output, tools)
        return content, self._assign_deterministic_ids(tool_calls, step_idx)

    async def parse_structured_action(
        self,
        content: str,
        tool_calls_data: list[dict],
        step_idx: int = 0,
    ) -> tuple[str, list[OpenAIFunctionToolCall]]:
        """Parse OpenAI-style structured tool calls. May return an empty list
        (callers decide); unknown names / invalid JSON args raise
        :class:`FunctionCallFormatError`.
        """
        tool_calls = []
        tools = [OpenAIFunctionToolSchema(**schema) for schema in self.tools_schemas]
        for tool_call_data in tool_calls_data:
            function_data = tool_call_data["function"]
            function_name = function_data["name"]
            arguments = function_data.get("arguments", {})
            if isinstance(arguments, str):
                arguments, _ = load_json_object(arguments, context=f"arguments for '{function_name}'")
            if not isinstance(arguments, dict):
                raise FunctionCallFormatError(
                    f"Invalid arguments for '{function_name}': expected a JSON object, got {type(arguments).__name__}."
                )
            validate_function_call(function_name, arguments, tools)

            function_call = OpenAIFunctionCallSchema(name=function_name, arguments=arguments)
            tool_calls.append(
                OpenAIFunctionToolCall(
                    id=deterministic_tool_call_id(
                        step_idx=step_idx,
                        call_index=len(tool_calls),
                        name=function_name,
                        arguments=arguments,
                    ),
                    type=tool_call_data.get("type", "function"),
                    function=function_call,
                )
            )
        return content, tool_calls

    @staticmethod
    def normalize_structured_tool_call_ids(
        tool_calls_data: list[dict],
        step_idx: int,
    ) -> list[dict]:
        """Replace provider IDs before validation, including malformed calls.

        A rejected structured call is still written to the assistant message
        and echoed by the tool-format feedback. It therefore must not retain a
        provider-generated UUID, or the next prompt would differ across runs.
        """
        normalized = []
        for call_index, tool_call_data in enumerate(tool_calls_data):
            function_data = dict(tool_call_data.get("function") or {})
            name = function_data.get("name", "")
            arguments = function_data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments_for_id, _ = load_json_object(arguments, context=f"arguments for '{name}'")
                except FunctionCallFormatError:
                    arguments_for_id = arguments
            else:
                arguments_for_id = arguments
            normalized.append(
                {
                    **tool_call_data,
                    "id": deterministic_tool_call_id(
                        step_idx=step_idx,
                        call_index=call_index,
                        name=str(name),
                        arguments=arguments_for_id,
                    ),
                    "function": function_data,
                }
            )
        return normalized

    @staticmethod
    def _assign_deterministic_ids(
        tool_calls: list[OpenAIFunctionToolCall],
        step_idx: int,
    ) -> list[OpenAIFunctionToolCall]:
        return [
            OpenAIFunctionToolCall(
                id=deterministic_tool_call_id(
                    step_idx=step_idx,
                    call_index=call_index,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
                type=tool_call.type,
                function=tool_call.function,
            )
            for call_index, tool_call in enumerate(tool_calls)
        ]

    def get_tool_bash_command(self, tool_call: OpenAIFunctionToolCall) -> str:
        function: OpenAIFunctionCallSchema = tool_call.function
        func_name: str = function.name
        func_params: dict = function.arguments

        if func_name == "submit":
            return "echo '<<<Finished>>>'"

        if func_name == "execute_bash":
            return func_params.get("command", "")

        if func_name == "lark-cli":
            command = func_params.get("command", "")
            return f"lark-cli {command}" if command else ""

        # Start building the command
        cmd_parts = [shlex.quote(func_name)]

        # If there's a 'command' parameter, put that next
        base_command = func_params.get("command")
        if base_command is not None:
            cmd_parts.append(shlex.quote(base_command))

        # Append all other parameters
        for param_key, param_value in func_params.items():
            if param_key == "command":
                continue

            # Use JSON for structured types so the script can json.loads them
            if isinstance(param_value, list | dict):
                param_str = json.dumps(param_value, ensure_ascii=False)
            else:
                param_str = str(param_value)
            cmd_parts.append(f"--{param_key}")
            cmd_parts.append(shlex.quote(param_str))

        return " ".join(cmd_parts)
