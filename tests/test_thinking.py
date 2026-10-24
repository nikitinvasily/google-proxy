"""Readable Gemini thought summaries travel in reasoning_content, not content."""
import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class ThinkingTests(unittest.TestCase):
    def test_gemini_requests_thought_summaries_without_changing_other_models(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        with patch.object(proxy, "get_project_id", return_value="synthetic-project"):
            gemini = proxy._build_gemini_request(body, "gemini-3-flash")
            claude = proxy._build_gemini_request(body, "claude-sonnet-5-5-high")
        self.assertIs(gemini["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"], True)
        self.assertNotIn("thinkingConfig", claude["request"].get("generationConfig", {}))

    def test_nonstream_thought_is_reasoning_not_visible_content(self):
        parts = [{"thought": True, "text": "SYNTHETIC_THOUGHT"},
                 {"thoughtSignature": "SYNTHETIC_SIGNATURE"}, {"text": "visible"}]
        result = proxy.gemini_response_to_openai(
            {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]},
            "gemini-3-flash")
        message = result["choices"][0]["message"]
        self.assertEqual(message["content"], "visible")
        self.assertEqual(message["reasoning_content"], "SYNTHETIC_THOUGHT")
        self.assertNotIn("SYNTHETIC_SIGNATURE", json.dumps(result))

    def test_stream_thought_delta_precedes_visible_text_and_tool_call(self):
        events = [
            {"response": {"candidates": [{"content": {"parts": [
                {"thought": True, "text": "SYNTHETIC_THOUGHT"}]}}]}},
            {"response": {"candidates": [{"content": {"parts": [
                {"text": "visible"},
                {"functionCall": {"name": "lookup", "id": "fc1", "args": {"q": "test"}},
                 "thoughtSignature": "SYNTHETIC_SIGNATURE"}]}, "finishReason": "STOP"}]}}
        ]
        handler: Any = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "lookup"}]}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
            handler._handle_stream(envelope, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {") and '"choices"' in line]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
        self.assertEqual([delta["reasoning_content"] for delta in deltas
                          if "reasoning_content" in delta], ["SYNTHETIC_THOUGHT"])
        self.assertEqual([delta["content"] for delta in deltas if delta.get("content")],
                         ["visible"])
        calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"],
                         "SYNTHETIC_SIGNATURE")
        self.assertNotIn("SYNTHETIC_THOUGHT", [delta.get("content") for delta in deltas])
        self.assertIn("data: [DONE]", handler.wfile.getvalue().decode())

    def test_transform_failure_does_not_fall_back_to_private_thought_text(self):
        upstream = {"candidates": [{"content": {"parts": [
            {"thought": True, "text": "SYNTHETIC_PRIVATE_THOUGHT"},
            {"text": "visible"}]}, "finishReason": "STOP"}]}
        handler: Any = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        statuses = []
        handler.send_response = statuses.append
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        with patch.object(proxy, "call_generate_content", return_value=upstream), \
             patch.object(proxy, "gemini_response_to_openai", side_effect=ValueError("bad part")):
            handler._handle_nonstream({"request": {}}, "gemini-3-flash")
        self.assertEqual(statuses, [502])
        self.assertNotIn("SYNTHETIC_PRIVATE_THOUGHT", handler.wfile.getvalue().decode())


if __name__ == "__main__":
    unittest.main()
