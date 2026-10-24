import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


def collect_stream(events):
    h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    h.wfile = io.BytesIO()
    h.send_response = lambda *args: None
    h.send_header = lambda *args: None
    h.end_headers = lambda *args: None
    with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
        h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
    return [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith("data: {")]


class StreamIntegrityTests(unittest.TestCase):
    def test_sse_error_first_event_returns_http_error(self):
        raw = b'event: error\ndata: {"error":{"code":429,"message":"SECRET_SENTINEL"}}\n\n'
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = lambda code: statuses.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", return_value=io.BytesIO(raw)):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        self.assertEqual(statuses, [429])
        self.assertNotIn("SECRET_SENTINEL", h.wfile.getvalue().decode())

    def test_sse_error_after_stop_does_not_report_success(self):
        raw = (b'data: {"response":{"candidates":[{"finishReason":"STOP"}]}}\n\n'
               b'event: error\ndata: {"error":{"code":429,"message":"SECRET_SENTINEL"}}\n\n')
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", return_value=io.BytesIO(raw)):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        body = h.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertNotIn("SECRET_SENTINEL", body)

    def test_upstream_error_before_first_event_uses_http_error(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = lambda code: statuses.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        def fail(_):
            raise proxy.UpstreamError("quota exceeded", status=429)
            yield
        with patch.object(proxy, "stream_generate_content", side_effect=fail):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        self.assertEqual(statuses, [429])
        self.assertNotIn("chat.completion.chunk", h.wfile.getvalue().decode())

    def test_upstream_error_after_first_event_does_not_report_stop(self):
        def events(_):
            yield {"response": {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}}
            raise proxy.UpstreamError("SECRET_SENTINEL", status=429)
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "stream_generate_content", side_effect=events):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        body = h.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertNotIn("SECRET_SENTINEL", body)

    def test_eof_without_finish_reason_does_not_report_stop(self):
        chunks = collect_stream([{"response": {"candidates": [{"content": {"parts": [{"text": "incomplete"}]}}]}}])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")

    def test_prompt_block_is_not_reported_as_success(self):
        chunks = collect_stream([{"response": {"promptFeedback": {"blockReason": "SAFETY"}}}])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "content_filter")
        self.assertTrue(any("blocked" in str(chunk).lower() for chunk in chunks))

    def test_nonstream_preserves_prompt_feedback(self):
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": {"promptFeedback": {"blockReason": "SAFETY"}}}])):
            response = proxy.call_generate_content({"request": {}})
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash")
        self.assertEqual(result["choices"][0]["finish_reason"], "content_filter")

    def test_nonstream_incomplete_generation_raises(self):
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}}])):
            with self.assertRaises(proxy.UpstreamError):
                proxy.call_generate_content({"request": {}})

    def test_multiline_sse_event_is_combined(self):
        raw = b'data: {"response":\ndata: {"candidates":[{"finishReason":"STOP"}]}}\n\n'
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", return_value=io.BytesIO(raw)):
            events = list(proxy.stream_generate_content({"request": {}}))
        self.assertEqual(events[0]["response"]["candidates"][0]["finishReason"], "STOP")

    def test_max_tokens_is_not_masked_by_tool_call(self):
        response = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "terminal", "args": {"command": "pwd"}}}]}, "finishReason": "MAX_TOKENS"}]}
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})
        self.assertEqual(result["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
