import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class RequestConversionTests(unittest.TestCase):
    def build(self, messages):
        with patch.object(proxy, "get_project_id", return_value="test-project"):
            envelope = proxy._build_gemini_request(
                {"model": "gemini-3.8-flash", "messages": messages}, "gemini-3-flash"
            )
        return envelope["request"]

    def test_text_and_tool_history(self):
        request = self.build([
            {"role": "system", "content": "system text"},
            {"role": "developer", "content": "developer text"},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "call_abc", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"ok":true}'},
        ])
        self.assertEqual(request["systemInstruction"]["parts"], [{"text": "system text\n\ndeveloper text"}])
        self.assertEqual(request["contents"][0], {"role": "user", "parts": [{"text": "ping"}]})
        self.assertEqual(request["contents"][1]["parts"][0], {"text": "checking"})
        self.assertEqual(request["contents"][2]["parts"][0]["functionResponse"]["name"], "lookup")

    def test_text_blocks_are_preserved(self):
        request = self.build([{"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "input_text", "text": "second"},
        ]}])
        self.assertEqual(request["contents"][0]["parts"], [
            {"text": "first"}, {"text": "second"},
        ])


if __name__ == "__main__":
    unittest.main()
