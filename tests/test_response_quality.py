import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class ResponseQualityTests(unittest.TestCase):
    def test_model_thinking_level_is_sent(self):
        env = proxy._build_gemini_request({"messages": [{"role": "user", "content": "hello"}]}, "gemini-4-argon-low", "low")
        self.assertEqual(env["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low")

    def test_nonstream_thought_text_is_private(self):
        response = {"candidates": [{"content": {"parts": [{"thought": True, "text": "PRIVATE_THOUGHT"}, {"text": "public"}]}, "finishReason": "STOP"}]}
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash")
        self.assertEqual(result["choices"][0]["message"]["content"], "public")

    def test_stream_thought_text_is_private(self):
        text, calls, names = proxy._extract_delta_from_parts([{"thought": True, "text": "PRIVATE_THOUGHT"}, {"text": "public"}], {})
        self.assertEqual(text, "public")

    def test_logged_status_matches_rejected_request(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.path = "/v1/chat/completions"
        h.wfile = io.BytesIO()
        h._read_body = lambda: None
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        logged = []
        h._log_req = lambda method, path, status, duration: logged.append(status)
        h.do_POST()
        self.assertEqual(logged, [400])


if __name__ == "__main__":
    unittest.main()
