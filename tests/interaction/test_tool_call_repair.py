import asyncio
from unittest.mock import patch

import pytest

from uni_agent.interaction.interaction import AgentInteraction
from uni_agent.interaction.tool_parser import FunctionCallFormatError, HermesToolParser
from uni_agent.interaction.tool_schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from uni_agent.interaction.tools_manager import ToolsManager

TOOLS = [
    OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "execute_bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    ),
    OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "str_replace_editor",
            "description": "Edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    },
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "view_range": {"type": "array"},
                },
                "required": ["command", "path"],
            },
        },
    ),
    OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "submit",
            "description": "Finish",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ),
]


def _parse(body: str) -> OpenAIFunctionToolCall:
    _, calls = HermesToolParser().extract_tool_calls(f"<tool_call>{body}</tool_call>", TOOLS)
    assert len(calls) == 1
    return calls[0]


def test_valid_json_does_not_invoke_repair():
    body = '{"name":"execute_bash","arguments":{"command":"pwd"}}'
    with patch("uni_agent.interaction.tool_parser._repair_json_syntax", side_effect=AssertionError):
        assert _parse(body).function.arguments == {"command": "pwd"}


def test_repairs_unescaped_inner_quotes():
    body = '{"name":"execute_bash","arguments":{"command":"find /testbed -name "*.py" -type f"}}'
    call = _parse(body)
    assert call.function.arguments["command"] == 'find /testbed -name "*.py" -type f'


def test_repairs_inner_quote_followed_by_command_comma():
    body = """{"name":"execute_bash","arguments":{"command":"python -c 'print("left", value)'"}}"""
    call = _parse(body)
    assert call.function.arguments["command"] == "python -c 'print(\"left\", value)'"


def test_repairs_raw_newline_in_string():
    body = '{"name":"str_replace_editor","arguments":{"command":"create","path":"/tmp/a","file_text":"a\nb"}}'
    call = _parse(body)
    assert call.function.arguments["file_text"] == "a\nb"


def test_repair_preserves_existing_json_escapes():
    body = '{"name":"execute_bash","arguments":{"command":"echo \\"ok\\"\nnext"}}'
    call = _parse(body)
    assert call.function.arguments["command"] == 'echo "ok"\nnext'


def test_repairs_single_quote_after_arguments_object():
    body = '{"name":"execute_bash","arguments":{"command":"pwd"}"}'
    call = _parse(body)
    assert call.function.arguments == {"command": "pwd"}


def test_repairs_missing_final_object_close():
    body = '{"name":"execute_bash","arguments":{"command":"pwd"}'
    call = _parse(body)
    assert call.function.arguments == {"command": "pwd"}


def test_ambiguous_json_is_rejected():
    body = '{"name":"execute_bash","arguments":{"command":"echo "unterminated}}'
    with pytest.raises(FunctionCallFormatError, match="Invalid tool_call JSON"):
        _parse(body)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"name":"missing","arguments":{}}', "not defined"),
        ('{"name":"execute_bash","arguments":{}}', "missing required"),
        (
            '{"name":"execute_bash","arguments":{"command":"pwd","extra":1}}',
            "unknown parameter",
        ),
        ('{"name":"execute_bash","arguments":{"command":2}}', "expected string"),
        (
            '{"name":"str_replace_editor","arguments":{"command":"delete","path":"/tmp/a"}}',
            "is not in",
        ),
    ],
)
def test_schema_validation_rejects_invalid_calls(body, message):
    with pytest.raises(FunctionCallFormatError, match=message):
        _parse(body)


def test_structured_calls_use_the_same_schema_validation():
    manager = object.__new__(ToolsManager)
    manager.tools_schemas = [tool.model_dump() for tool in TOOLS]

    async def execute():
        await manager.parse_structured_action(
            "",
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_bash", "arguments": {"command": 2}},
                }
            ],
        )

    with pytest.raises(FunctionCallFormatError, match="expected string"):
        asyncio.run(execute())


class _FakeEnv:
    def __init__(self):
        self.calls = 0

    async def run_action(self, command, action_timeout=60):
        self.calls += 1
        return "OK"


class _FakeModel:
    supports_toolcall_speculation = False
    tool_schema_signature = "schema"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.queries = 0
        self.query_kwargs = []

    async def prepare_rollout_cache(self, messages):
        return {
            "request_id": "request",
            "prompt_ids": [1],
            "response_mask": [],
            "response_logprobs": [],
            "metrics": {},
            "extra_fields": {},
        }

    async def query(self, messages, rollout_cache, **kwargs):
        self.queries += 1
        self.query_kwargs.append(kwargs)
        output = self.outputs.pop(0)
        return output, [], rollout_cache, {"prompt_tokens": 1, "completion_tokens": 1}

    async def append_messages_to_rollout_cache(self, messages, rollout_cache):
        return rollout_cache


def _manager() -> ToolsManager:
    manager = object.__new__(ToolsManager)
    manager.tools_schemas = [tool.model_dump() for tool in TOOLS]
    manager._tool_parser = HermesToolParser()
    return manager


def test_format_retry_does_not_consume_agent_turn_and_feedback_is_actionable():
    async def execute():
        model = _FakeModel(
            [
                "not a tool call",
                '<tool_call>{"name":"submit","arguments":{}}</tool_call>',
            ]
        )
        env = _FakeEnv()
        interaction = AgentInteraction(
            "run",
            env,
            model,
            _manager(),
            [{"role": "user", "content": "go"}],
            max_turns=1,
            max_format_retries_per_turn=3,
            max_generation_tokens=8192,
        )
        return model, env, await interaction.run()

    model, env, result = asyncio.run(execute())
    assert model.queries == 2
    assert model.query_kwargs == [
        {"sampling_params": {"max_tokens": 8192}},
        {"sampling_params": {"max_tokens": 8192}},
    ]
    assert env.calls == 1
    assert [item.step_idx for item in result["trajectory"]] == [1, 1]
    assert [item.exit_reason for item in result["trajectory"]] == ["format_error", "finished"]
    assert all(item.generation_wait_time_s > 0 for item in result["trajectory"])
    assert all(item.elapsed_time_s > 0 for item in result["trajectory"])

    correction = result["messages"][2]
    assert correction["role"] == "user"
    assert "Tool call rejected: No function call found" in correction["content"]
    assert "Escape inner double quotes" in correction["content"]
    assert "Do not explain" in correction["content"]


def test_format_retry_has_an_independent_limit():
    async def execute():
        model = _FakeModel(["bad"] * 3)
        interaction = AgentInteraction(
            "run",
            _FakeEnv(),
            model,
            _manager(),
            [{"role": "user", "content": "go"}],
            max_turns=1,
            max_format_retries_per_turn=2,
        )
        return model, await interaction.run()

    model, result = asyncio.run(execute())
    assert model.queries == 3
    assert [item.step_idx for item in result["trajectory"]] == [1, 1, 1, 1]
    assert [item.exit_reason for item in result["trajectory"]] == [
        "format_error",
        "format_error",
        "format_error",
        "format_retry_exhausted",
    ]
