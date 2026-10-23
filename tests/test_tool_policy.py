import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy

TOOL = lambda name: {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}


def stream(events, tools=("terminal",), choice="auto"):
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.wfile = io.BytesIO()
    handler.send_response = lambda *args: None
    handler.send_header = lambda *args: None
    handler.end_headers = lambda *args: None
    body = {"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hello"}], "tools": [TOOL(name) for name in tools], "tool_choice": choice}
    envelope = proxy._build_gemini_request(body, "gemini-3-flash")
    with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
        handler._handle_stream(envelope, "gemini-3.8-flash")
    return [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines() if line.startswith("data: {")]


class ToolPolicyTests(unittest.TestCase):
    def test_none_does_not_turn_shorthand_into_tool_call(self):
        envelope = proxy._build_gemini_request({"messages": [{"role": "user", "content": "hello"}], "tools": [TOOL("terminal")], "tool_choice": "none"}, "gemini-3-flash")
        allowed = proxy._declared_tool_names(envelope)
        response = {"candidates": [{"content": {"parts": [{"text": "call:default_api:terminal{command:pwd}"}]}, "finishReason": "STOP"}]}
        result = proxy.gemini_response_to_openai(response, "gemini-3.8-flash", allowed)["choices"][0]
        self.assertEqual(result["finish_reason"], "stop")
        self.assertNotIn("tool_calls", result["message"])

    def test_forced_function_limits_text_parser(self):
        envelope = proxy._build_gemini_request({"messages": [{"role": "user", "content": "hello"}], "tools": [TOOL("terminal"), TOOL("lookup")], "tool_choice": {"type": "function", "function": {"name": "lookup"}}}, "gemini-3-flash")
        self.assertEqual(proxy._declared_tool_names(envelope), {"lookup"})

    def test_two_calls_same_name_use_distinct_stream_indices(self):
        events = [{"response": {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "terminal", "args": {"command": "pwd"}}},
            {"functionCall": {"name": "terminal", "args": {"command": "date"}}},
        ]}, "finishReason": "STOP"}]}}]
        chunks = stream(events)
        calls = [tc for c in chunks for ch in c["choices"] for tc in ch["delta"].get("tool_calls", [])]
        self.assertEqual([tc["index"] for tc in calls], [0, 1])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_text_and_structured_call_do_not_execute_twice(self):
        events = [{"response": {"candidates": [{"content": {"parts": [
            {"text": "call:default_api:terminal{command:pwd}"},
            {"functionCall": {"name": "terminal", "args": {"command": "date"}}},
        ]}, "finishReason": "STOP"}]}}]
        chunks = stream(events)
        calls = [tc for c in chunks for ch in c["choices"] for tc in ch["delta"].get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "date"})


if __name__ == "__main__":
    unittest.main()
