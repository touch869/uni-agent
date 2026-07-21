"""Focused tests for the SWE-Lego-only parser adapter."""

import unittest

from examples.agent_interaction.swe_lego_tool_parser import SWELegoToolParser
from uni_agent.interaction.tool_schemas import OpenAIFunctionToolSchema


TOOLS = [
    OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "execute_bash",
            "description": "run shell",
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
            "name": "submit",
            "description": "finish",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ),
    OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "str_replace_editor",
            "description": "edit files",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "view_range": {"type": "array"},
                },
                "required": ["command", "path"],
            },
        },
    ),
]


class SWELegoToolParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = SWELegoToolParser()

    def test_strict_hermes_is_unchanged(self):
        text = '<tool_call>{"name":"execute_bash","arguments":{"command":"pwd"}}</tool_call>'
        _, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(calls[0].function.arguments, {"command": "pwd"})
        self.assertEqual(self.parser.last_event["mode"], "strict_hermes")
        self.assertEqual(self.parser.events, [])

    def test_legacy_execute_bash_tag(self):
        text = 'search\n<execute_bash>\nfind /testbed -name "*.py"\n</execute_bash>'
        content, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(content, "search\n")
        self.assertEqual(calls[0].function.arguments["command"], 'find /testbed -name "*.py"')
        self.assertEqual(self.parser.last_event["mode"], "legacy_xml")

    def test_unescaped_command_quotes(self):
        text = (
            '<tool_call>{"name": "execute_bash", "arguments": '
            '{"command": "find /testbed -name "*.py" | head"}}</tool_call>'
        )
        _, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(calls[0].function.arguments["command"], 'find /testbed -name "*.py" | head')
        self.assertEqual(self.parser.last_event["mode"], "repaired_hermes")

    def test_span_polluted_command_suffix(self):
        text = (
            "<tool_call>{\"name\": \"execute_bash\", \"arguments\": "
            "{\"command\": \"find /testbed -name \"*.py\" | head</span>}}</tool_call>"
        )
        _, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(calls[0].function.arguments["command"], "find /testbed -name \"*.py\" | head")
        self.assertEqual(self.parser.last_event["mode"], "repaired_hermes")

    def test_function_call_parameters(self):
        text = (
            "<functionCall><parameter>name\">execute_bash</parameter>"
            "<parameter>command\">cd /testbed && ls -la</parameter></functionCall>"
        )
        _, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(calls[0].function.arguments["command"], "cd /testbed && ls -la")
        self.assertEqual(self.parser.last_event["mode"], "legacy_function_call")


    def test_editor_view_alias(self):
        text = (
            "<tool_call>{\"name\":\"view\",\"path\":\"/testbed/a.py\","
            "\"view_range\":[1,2]}</tool_call>"
        )
        _, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(calls[0].function.name, "str_replace_editor")
        self.assertEqual(calls[0].function.arguments["command"], "view")
        self.assertEqual(calls[0].function.arguments["path"], "/testbed/a.py")

    def test_editor_create_with_raw_newlines(self):
        text = (
            "<tool_call>{\"name\":\"str_replace_editor\",\"arguments\":{"
            "\"command\":\"create\",\"file_text\":\"line 1\nprint('x')\n\","
            " \"path\":\"/testbed/reproduce.py\"}}</tool_call>"
        )
        _, calls = self.parser.extract_tool_calls(text, TOOLS)
        self.assertEqual(calls[0].function.arguments["command"], "create")
        self.assertEqual(calls[0].function.arguments["file_text"], "line 1\nprint('x')\n")
        self.assertEqual(calls[0].function.arguments["path"], "/testbed/reproduce.py")


    def test_ambiguous_malformed_call_still_fails(self):
        text = '<tool_call>{"name":"execute_bash","arguments":{"command":"unterminated}}</tool_call>'
        with self.assertRaises(Exception):
            self.parser.extract_tool_calls(text, TOOLS)


if __name__ == "__main__":
    unittest.main()
