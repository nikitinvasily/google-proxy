import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class TextToolCallTests(unittest.TestCase):
    def test_nonstream_shorthand_terminal_becomes_structured_call(self):
        response = {"candidates": [{"content": {"parts": [{"text": "call:default_api:terminal{command:pwd}"}]}, "finishReason": "STOP"}]}
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "terminal")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"command": "pwd"})

    def test_no_declared_tool_keeps_text_and_no_execution(self):
        response = {"candidates": [{"content": {"parts": [{"text": "call:default_api:terminal{command:pwd}"}]}, "finishReason": "STOP"}]}
        choice = proxy.gemini_response_to_openai(response, "gemini-3-flash", set())["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])

    def test_json_call_and_prose_safety(self):
        response = {"candidates": [{"content": {"parts": [{"text": 'call:default_api:terminal{"command":"echo hi"}'}]}, "finishReason": "STOP"}]}
        choice = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"command": "echo hi"})
        prose = {"candidates": [{"content": {"parts": [{"text": "Example: call:default_api:terminal{command:pwd}"}]}, "finishReason": "STOP"}]}
        choice = proxy.gemini_response_to_openai(prose, "gemini-3-flash", {"terminal"})["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])

    def test_stream_split_shorthand_terminal_becomes_tool_call(self):
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda *args: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}
        events = iter([
            {"response": {"candidates": [{"content": {"parts": [{"text": "call:default_api:terminal{command:"}]}}]}},
            {"response": {"candidates": [{"content": {"parts": [{"text": "pwd}"}]}, "finishReason": "STOP"}]}},
        ])
        with patch.object(proxy, "stream_generate_content", return_value=events):
            handler._handle_stream(envelope, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines() if line.startswith("data: {")]
        self.assertFalse(any("call:default_api:" in str(c) for c in chunks))
        calls = [tc for c in chunks for choice in c.get("choices", []) for tc in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "terminal")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "pwd"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("data: [DONE]", handler.wfile.getvalue().decode())

    def test_stream_structured_function_call_keeps_tool_finish(self):
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda *args: None
        events = iter([
            {"response": {"candidates": [{"content": {"parts": [{"functionCall": {"name": "terminal", "args": {"command": "pwd"}}}]}}]}},
            {"response": {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}},
        ])
        with patch.object(proxy, "stream_generate_content", return_value=events):
            handler._handle_stream({"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_stream_normal_text_still_streams(self):
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda *args: None
        events = iter([{"response": {"candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}]}}])
        with patch.object(proxy, "stream_generate_content", return_value=events):
            handler._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        body = handler.wfile.getvalue().decode()
        self.assertIn('"content": "hello"', body)
        self.assertIn('"finish_reason": "stop"', body)


if __name__ == "__main__":
    unittest.main()
