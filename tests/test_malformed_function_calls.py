"""Regression tests for malformed upstream tool calls and quota routes."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy

MALFORMED = {
    "candidates": [{
        "content": {"parts": [{"text": "internal thought", "thought": True}]},
        "finishReason": "MALFORMED_FUNCTION_CALL",
    }]
}
ENVELOPE = {"request": {"contents": [{"role": "user", "parts": [{"text": "test"}]}]}}


def handler():
    h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    h.wfile = io.BytesIO()
    h.statuses = []
    h.send_response = h.statuses.append
    h.send_header = lambda *args: None
    h.end_headers = lambda: None
    return h


class MalformedFunctionCallTests(unittest.TestCase):
    def test_stream_reports_malformed_instead_of_false_truncation(self):
        h = handler()
        events = iter([
            {"response": {"candidates": [{"content": {"parts": [{"text": "internal thought", "thought": True}]}}]}},
            {"response": MALFORMED},
        ])
        with patch.object(proxy, "stream_generate_content", return_value=events):
            h._handle_stream(ENVELOPE, "gemini-3-flash")
        payload = h.wfile.getvalue().decode()
        self.assertEqual(h.statuses, [200])
        self.assertIn("event: error\n", payload)
        errors = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: {") and '"error"' in line]
        self.assertEqual(errors[0]["error"]["code"], "upstream_malformed_function_call")
        self.assertNotIn("[Upstream returned empty response]", payload)
        self.assertNotIn('"finish_reason": "length"', payload)
        self.assertNotIn("data: [DONE]", payload)

    def test_stream_recognizes_malformed_stop_reason(self):
        h = handler()
        response = {"candidates": [{"content": {"parts": [{"text": "internal thought", "thought": True}]},
                                   "stopReason": "MALFORMED_FUNCTION_CALL"}]}
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": response}])):
            h._handle_stream(ENVELOPE, "gemini-3-flash")
        payload = h.wfile.getvalue().decode()
        self.assertIn("event: error\n", payload)
        self.assertNotIn("data: [DONE]", payload)

    def test_nonstream_reports_malformed_even_with_visible_text(self):
        h = handler()
        response = {"candidates": [{"content": {"parts": [{"text": "partial output"}]}, "finishReason": "MALFORMED_FUNCTION_CALL"}]}
        with patch.object(proxy, "call_generate_content", return_value=response):
            h._handle_nonstream(ENVELOPE, "gemini-3-flash")
        self.assertEqual(h.statuses, [502])
        result = json.loads(h.wfile.getvalue())
        self.assertEqual(result["error"]["code"], "upstream_malformed_function_call")
        self.assertNotIn("partial output", h.wfile.getvalue().decode())

    def test_nonstream_malformed_parts_cannot_trigger_success_fallback(self):
        h = handler()
        response = {"candidates": [{"content": {"parts": [
            {"text": "partial output"}, {"functionCall": "bad-shape"},
        ]}, "finishReason": "MALFORMED_FUNCTION_CALL"}]}
        with patch.object(proxy, "call_generate_content", return_value=response):
            h._handle_nonstream(ENVELOPE, "gemini-3-flash")
        self.assertEqual(h.statuses, [502])
        self.assertEqual(json.loads(h.wfile.getvalue())["error"]["code"],
                         "upstream_malformed_function_call")
        self.assertNotIn("partial output", h.wfile.getvalue().decode())

    def test_nonstream_does_not_overwrite_malformed_with_later_stop(self):
        h = handler()
        later_stop = {"candidates": [{"content": {"parts": [{"text": "must not be sent"}]},
                                      "finishReason": "STOP"}]}
        with patch.object(proxy, "stream_generate_content", return_value=iter([
            {"response": MALFORMED}, {"response": later_stop},
        ])):
            h._handle_nonstream(ENVELOPE, "gemini-3-flash")
        self.assertEqual(h.statuses, [502])
        self.assertEqual(json.loads(h.wfile.getvalue())["error"]["code"],
                         "upstream_malformed_function_call")
        self.assertNotIn("must not be sent", h.wfile.getvalue().decode())

    def test_stream_does_not_forward_call_from_malformed_candidate(self):
        h = handler()
        envelope = {"request": {"contents": ENVELOPE["request"]["contents"],
                                "tools": [{"functionDeclarations": [{"name": "lookup"}]}]}}
        malformed = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "lookup", "args": {"value": "private-marker"}}}
        ]}, "finishReason": "MALFORMED_FUNCTION_CALL"}]}
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": malformed}])):
            h._handle_stream(envelope, "gemini-3-flash")
        payload = h.wfile.getvalue().decode()
        self.assertIn("event: error\n", payload)
        self.assertNotIn('"tool_calls"', payload)
        self.assertNotIn("private-marker", payload)
        self.assertNotIn("data: [DONE]", payload)

    def test_stream_rejects_malformed_in_nonfirst_candidate(self):
        h = handler()
        response = {"candidates": [
            {"content": {"parts": [{"text": "do not send"}]}, "finishReason": "STOP"},
            {"content": {"parts": []}, "finishReason": "MALFORMED_FUNCTION_CALL"},
        ]}
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": response}])):
            h._handle_stream(ENVELOPE, "gemini-3-flash")
        payload = h.wfile.getvalue().decode()
        self.assertIn("event: error\n", payload)
        self.assertNotIn("do not send", payload)
        self.assertNotIn("data: [DONE]", payload)

    def test_quota_route_uses_quota_fetcher(self):
        h = handler()
        h.path = "/v1/quota"
        h._log_req = lambda *args: None
        expected = {"source": "retrieveUserQuota", "buckets": [],
                    "groups_source": "retrieveUserQuotaSummary",
                    "groups": [{"name": "Gemini Models", "buckets": [{
                        "id": "gemini-weekly", "window": "weekly",
                        "remaining_fraction": 1, "reset_time": None,
                    }]}]}
        with patch.object(proxy, "get_access_token", return_value="synthetic-token"), \
             patch.object(proxy, "fetch_user_quota", return_value={
                 "source": "retrieveUserQuota", "buckets": [],
             }) as fetch, \
             patch.object(proxy, "fetch_quota_groups", return_value=expected["groups"]) as groups:
            h.do_GET()
        self.assertEqual(h.statuses, [200])
        self.assertEqual(json.loads(h.wfile.getvalue()), expected)
        fetch.assert_called_once_with("synthetic-token")
        groups.assert_called_once_with("synthetic-token")

    def test_openai_sdk_raises_on_stream_error_event(self):
        import threading
        from http.server import ThreadingHTTPServer
        try:
            from openai import APIError, OpenAI
        except ImportError:
            self.skipTest("OpenAI SDK is not installed in this test environment")

        events = [
            {"response": {"candidates": [{"content": {"parts": [{"text": "internal thought", "thought": True}]}}]}},
            {"response": MALFORMED},
        ]
        server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.ProxyHandler)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.object(proxy, "stream_generate_content", side_effect=lambda envelope: iter(events)):
                client = OpenAI(api_key="test-only", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                max_retries=0, timeout=5)
                with self.assertRaises(APIError) as caught:
                    list(client.chat.completions.create(model="gemini-3-flash", stream=True,
                                                        messages=[{"role": "user", "content": "test"}]))
                self.assertIn("invalid function call", str(caught.exception))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
