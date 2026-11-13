import io
import json
import sys
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


def _make_http_429(body: bytes = b'{"error":{"code":429,"message":"quota exhausted"}}'):
    return HTTPError(
        proxy.STREAM_GENERATE_CONTENT_URL,
        429,
        "Too Many Requests",
        Message(),
        io.BytesIO(body),
    )


def _sse_success_bytes(text: str = "ok from credits"):
    payload = {
        "response": {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
            "modelVersion": "gemini-3-flash",
        }
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


class CreditsFallbackTests(unittest.TestCase):
    def test_http_429_retries_once_with_google_one_ai_credits_and_succeeds(self):
        envelope = {
            "project": "proj-1",
            "model": "gemini-3-flash",
            "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        }
        original_snapshot = json.loads(json.dumps(envelope))
        seen_envelopes = []

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8"))
            seen_envelopes.append(body)
            if len(seen_envelopes) == 1:
                raise _make_http_429()
            return io.BytesIO(_sse_success_bytes("recovered via credits"))

        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", True), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen):
            resp = proxy.call_generate_content(envelope)

        self.assertEqual(len(seen_envelopes), 2)
        self.assertNotIn("enabledCreditTypes", seen_envelopes[0])
        self.assertEqual(seen_envelopes[1].get("enabledCreditTypes"), ["GOOGLE_ONE_AI"])
        self.assertEqual(envelope, original_snapshot)
        self.assertNotIn("enabledCreditTypes", envelope)
        self.assertEqual(
            resp["candidates"][0]["content"]["parts"][0]["text"],
            "recovered via credits",
        )

    def test_sse_error_429_first_event_retries_and_handle_stream_succeeds(self):
        envelope = {
            "project": "proj-1",
            "model": "gemini-3-flash",
            "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": []},
        }
        seen_envelopes = []
        sse_429 = b'event: error\ndata: {"error":{"code":429,"message":"quota exhausted"}}\n\n'

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8"))
            seen_envelopes.append(body)
            if len(seen_envelopes) == 1:
                return io.BytesIO(sse_429)
            return io.BytesIO(_sse_success_bytes("stream recovered"))

        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = lambda code, message=None: statuses.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None

        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", True), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen):
            h._handle_stream(envelope, "gemini-3-flash")

        self.assertEqual(statuses, [200])
        self.assertEqual(len(seen_envelopes), 2)
        self.assertNotIn("enabledCreditTypes", seen_envelopes[0])
        self.assertEqual(seen_envelopes[1].get("enabledCreditTypes"), ["GOOGLE_ONE_AI"])
        self.assertNotIn("enabledCreditTypes", envelope)
        body = h.wfile.getvalue().decode("utf-8")
        self.assertIn("stream recovered", body)

    def test_second_429_raises_upstream_error_with_two_urlopen_calls(self):
        envelope = {"model": "gemini-3-flash", "request": {}}
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(json.loads(req.data.decode("utf-8")))
            raise _make_http_429()

        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", True), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(proxy.UpstreamError) as ctx:
                list(proxy.stream_generate_content(envelope))

        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("enabledCreditTypes", calls[0])
        self.assertEqual(calls[1].get("enabledCreditTypes"), ["GOOGLE_ONE_AI"])

    def test_disabled_flag_skips_retry_on_429(self):
        envelope = {"model": "gemini-3-flash", "request": {}}
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(json.loads(req.data.decode("utf-8")))
            raise _make_http_429()

        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", False), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(proxy.UpstreamError) as ctx:
                list(proxy.stream_generate_content(envelope))

        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(calls), 1)

    def test_existing_enabled_credit_types_skips_retry_on_429(self):
        envelope = {
            "model": "gemini-3-flash",
            "enabledCreditTypes": ["GOOGLE_ONE_AI"],
            "request": {},
        }
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(json.loads(req.data.decode("utf-8")))
            raise _make_http_429()

        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", True), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(proxy.UpstreamError) as ctx:
                list(proxy.stream_generate_content(envelope))

        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(calls), 1)

    def test_429_after_valid_sse_chunk_does_not_retry(self):
        envelope = {"model": "gemini-3-flash", "request": {}}
        mid_stream_429 = (
            b'data: {"response":{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}}\n\n'
            b'event: error\ndata: {"error":{"code":429,"message":"quota exhausted"}}\n\n'
        )
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(json.loads(req.data.decode("utf-8")))
            return io.BytesIO(mid_stream_429)

        yielded = []
        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", True), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen):
            gen = proxy.stream_generate_content(envelope)
            yielded.append(next(gen))
            with self.assertRaises(proxy.UpstreamError) as ctx:
                next(gen)

        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(yielded), 1)
        self.assertEqual(len(calls), 1)

    def test_cli_no_credits_fallback_disables_flag(self):
        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", True), \
             patch.object(sys, "argv", ["antigravity_proxy.py", "--no-credits-fallback"]), \
             patch.object(proxy, "_preflight_check", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                proxy.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertFalse(proxy.CREDITS_FALLBACK_ENABLED)


if __name__ == "__main__":
    unittest.main()
