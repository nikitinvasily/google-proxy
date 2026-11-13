import io
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


def _error_body(reason="QUOTA_EXHAUSTED", retry_delay=None, reset_ts=None, model="gemini-4-hydrogen-medium"):
    metadata = {"model": model}
    if reset_ts:
        metadata["quotaResetTimeStamp"] = reset_ts
    details = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": reason, "metadata": metadata}]
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay})
    return json.dumps({"error": {"code": 429, "message": "Individual quota reached",
                                 "status": "RESOURCE_EXHAUSTED", "details": details}}).encode("utf-8")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_http_429(body: bytes = None, headers=None):
    if body is None:
        body = _error_body()
    return HTTPError(
        proxy.STREAM_GENERATE_CONTENT_URL,
        429,
        "Too Many Requests",
        headers if headers is not None else Message(),
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
    def setUp(self):
        with proxy._credits_exhausted_lock:
            self._saved_cache = dict(proxy._credits_exhausted_until)
            proxy._credits_exhausted_until.clear()
        self._saved_force = os.environ.pop("ANTIGRAVITY_FORCE_CREDITS", None)

    def tearDown(self):
        with proxy._credits_exhausted_lock:
            proxy._credits_exhausted_until.clear()
            proxy._credits_exhausted_until.update(self._saved_cache)
        if self._saved_force is not None:
            os.environ["ANTIGRAVITY_FORCE_CREDITS"] = self._saved_force
        else:
            os.environ.pop("ANTIGRAVITY_FORCE_CREDITS", None)

    def _run(self, envelope, responder, fallback=True):
        """Run stream_generate_content hermetically; returns (events|exc, calls, sleeps)."""
        calls, sleeps = [], []

        def fake_urlopen(req, timeout=None):
            calls.append(json.loads(req.data.decode("utf-8")))
            return responder(len(calls))

        result = None
        with patch.object(proxy, "CREDITS_FALLBACK_ENABLED", fallback), \
             patch.object(proxy, "get_access_token", return_value="dummy-token"), \
             patch.object(proxy, "urlopen", side_effect=fake_urlopen), \
             patch.object(proxy.time, "sleep", side_effect=sleeps.append):
            try:
                result = list(proxy.stream_generate_content(envelope))
            except proxy.UpstreamError as exc:
                result = exc
        return result, calls, sleeps

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

    # ── quota-group cache / preventive flag ─────────────────────────────────

    def test_quota_group_by_model_prefix(self):
        self.assertEqual(proxy._quota_group("claude-sonnet-4"), "3p")
        self.assertEqual(proxy._quota_group("gpt-oss-120b"), "3p")
        self.assertEqual(proxy._quota_group("gemini-3-flash"), "gemini")
        self.assertEqual(proxy._quota_group(None), "gemini")

    def test_parse_429_info_extracts_reason_reset_and_retry_delay(self):
        reset = time.time() + 3600
        info = proxy._parse_429_info(_error_body("QUOTA_EXHAUSTED", "26980.722129870s", _iso(reset)))
        self.assertEqual(info["reason"], "QUOTA_EXHAUSTED")
        self.assertAlmostEqual(info["reset_epoch"], int(reset), delta=1)
        self.assertAlmostEqual(info["retry_delay_s"], 26980.72212987, places=3)

    def test_parse_429_info_retry_after_header_fallback(self):
        hdrs = Message()
        hdrs["Retry-After"] = "7"
        info = proxy._parse_429_info(b"not json", hdrs)
        self.assertIsNone(info["reason"])
        self.assertEqual(info["retry_delay_s"], 7.0)

    def test_quota_exhausted_cached_per_group(self):
        reset = time.time() + 6 * 3600
        body = _error_body("QUOTA_EXHAUSTED", "21600s", _iso(reset))

        def first(n):
            if n == 1:
                raise _make_http_429(body)
            return io.BytesIO(_sse_success_bytes("via credits"))

        res, calls, sleeps = self._run({"model": "gemini-3-flash", "request": {}}, first)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("enabledCreditTypes", calls[0])
        self.assertEqual(calls[1]["enabledCreditTypes"], ["GOOGLE_ONE_AI"])
        self.assertAlmostEqual(proxy._credits_exhausted_until["gemini"], int(reset), delta=1)
        self.assertNotIn("3p", proxy._credits_exhausted_until)
        self.assertEqual(sleeps, [])

        # second gemini request: preventive flag, single urlopen call
        res, calls, _ = self._run({"model": "gemini-3-pro", "request": {}},
                                  lambda n: io.BytesIO(_sse_success_bytes("second")))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["enabledCreditTypes"], ["GOOGLE_ONE_AI"])

        # claude request is a different group: no flag
        env = {"model": "claude-sonnet-4", "request": {}}
        res, calls, _ = self._run(env, lambda n: io.BytesIO(_sse_success_bytes("claude")))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("enabledCreditTypes", calls[0])
        self.assertNotIn("enabledCreditTypes", env)

    def test_cache_expires_after_reset_timestamp(self):
        proxy._credits_cache_store("gemini", time.time() + 100)
        real_time = time.time()
        with patch.object(proxy.time, "time", return_value=real_time + 101):
            res, calls, _ = self._run({"model": "gemini-3-flash", "request": {}},
                                      lambda n: io.BytesIO(_sse_success_bytes("after reset")))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("enabledCreditTypes", calls[0])
        self.assertNotIn("gemini", proxy._credits_exhausted_until)

    def test_short_rate_limit_without_credits_is_not_cached(self):
        body = _error_body("RATE_LIMIT_EXCEEDED", "2s", _iso(time.time() + 3600))

        def responder(n):
            if n == 1:
                raise _make_http_429(body)
            return io.BytesIO(_sse_success_bytes("ok"))

        res, calls, _ = self._run({"model": "gemini-3-flash", "request": {}}, responder)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["enabledCreditTypes"], ["GOOGLE_ONE_AI"])
        self.assertEqual(proxy._credits_exhausted_until, {})

    def test_reset_timestamp_beyond_horizon_or_past_is_not_cached(self):
        for ts in (_iso(time.time() + 30 * 86400), _iso(time.time() - 60)):
            body = _error_body("QUOTA_EXHAUSTED", "21600s", ts)

            def responder(n, body=body):
                if n == 1:
                    raise _make_http_429(body)
                return io.BytesIO(_sse_success_bytes("ok"))

            self._run({"model": "gemini-3-flash", "request": {}}, responder)
            self.assertEqual(proxy._credits_exhausted_until, {})

    def test_force_credits_env_sends_flag_on_first_attempt(self):
        for value in ("1", "true", "YES", "on"):
            os.environ["ANTIGRAVITY_FORCE_CREDITS"] = value
            res, calls, _ = self._run({"model": "claude-sonnet-4", "request": {}},
                                      lambda n: io.BytesIO(_sse_success_bytes("forced")))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["enabledCreditTypes"], ["GOOGLE_ONE_AI"])

    def test_force_credits_ignored_when_fallback_disabled(self):
        os.environ["ANTIGRAVITY_FORCE_CREDITS"] = "1"
        res, calls, _ = self._run({"model": "gemini-3-flash", "request": {}},
                                  lambda n: io.BytesIO(_sse_success_bytes("x")), fallback=False)
        self.assertNotIn("enabledCreditTypes", calls[0])

    # ── transient 429 backoff ──────────────────────────────────────────────

    def test_transient_429_on_credits_backs_off_and_succeeds(self):
        body = _error_body("RATE_LIMIT_EXCEEDED", "1.5s")

        def responder(n):
            if n <= 2:
                raise _make_http_429(body)
            return io.BytesIO(_sse_success_bytes("recovered"))

        env = {"model": "gemini-3-flash", "enabledCreditTypes": ["GOOGLE_ONE_AI"], "request": {}}
        res, calls, sleeps = self._run(env, responder)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1.5, 1.5])
        self.assertEqual(len(res), 1)

    def test_transient_429_without_retry_delay_uses_default_backoff(self):
        def responder(n):
            if n <= 2:
                raise _make_http_429(_error_body("RATE_LIMIT_EXCEEDED"))
            return io.BytesIO(_sse_success_bytes("ok"))

        env = {"model": "gemini-3-flash", "request": {}}
        res, calls, sleeps = self._run(env, responder, fallback=False)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1.0, 2.5])
        self.assertNotIn("enabledCreditTypes", calls[0])

    def test_transient_429_retries_limited_to_two(self):
        env = {"model": "gemini-3-flash", "request": {}}
        res, calls, sleeps = self._run(
            env, lambda n: (_ for _ in ()).throw(_make_http_429(_error_body("RATE_LIMIT_EXCEEDED", "0.1s"))),
            fallback=False)
        self.assertIsInstance(res, proxy.UpstreamError)
        self.assertEqual(res.status, 429)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.5, 0.5])  # clamped to the 0.5s minimum

    def test_sse_transient_429_backoff_retry(self):
        sse_429 = ('event: error\ndata: ' + json.dumps({"error": {"code": 429, "details": [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "RATE_LIMIT_EXCEEDED"},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}}) + '\n\n')
        sse_429_short = sse_429.replace("30s", "2s")

        env = {"model": "gemini-3-flash", "enabledCreditTypes": ["GOOGLE_ONE_AI"], "request": {}}
        res, calls, sleeps = self._run(env, lambda n: io.BytesIO(
            sse_429_short.encode() if n == 1 else _sse_success_bytes("sse ok")))
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [2.0])
        self.assertEqual(len(res), 1)

        # long retryDelay (> 15s) => raised immediately, no sleep
        res, calls, sleeps = self._run(env, lambda n: io.BytesIO(sse_429.encode()))
        self.assertIsInstance(res, proxy.UpstreamError)
        self.assertEqual((res.status, len(calls), sleeps), (429, 1, []))

    def test_quota_exhausted_on_credits_raises_immediately_without_sleep(self):
        env = {"model": "gemini-3-flash", "enabledCreditTypes": ["GOOGLE_ONE_AI"], "request": {}}
        for body in (_error_body("QUOTA_EXHAUSTED", "5s"),
                     _error_body("CREDITS_EXHAUSTED"),
                     _error_body("INSUFFICIENT_CREDITS", "1s"),
                     _error_body("RATE_LIMIT_EXCEEDED", "26980.7s")):
            res, calls, sleeps = self._run(
                env, lambda n, body=body: (_ for _ in ()).throw(_make_http_429(body)))
            self.assertIsInstance(res, proxy.UpstreamError)
            self.assertEqual(res.status, 429)
            self.assertEqual(len(calls), 1)
            self.assertEqual(sleeps, [])

    def test_second_429_with_quota_exhausted_after_fallback_raises_without_sleep(self):
        res, calls, sleeps = self._run({"model": "gemini-3-flash", "request": {}},
                                       lambda n: (_ for _ in ()).throw(_make_http_429()))
        self.assertEqual(res.status, 429)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [])

    # ── _capture_upstream_error with pre-read body ─────────────────────────

    def test_capture_upstream_error_uses_preread_body(self):
        body = _error_body("QUOTA_EXHAUSTED", "5s")
        err = _make_http_429(body)
        preread = err.read()
        self.assertEqual(err.read(), b"")  # body is consumed; capture must use bytes
        with tempfile.TemporaryDirectory() as tmp:
            directory = proxy.Path(tmp) / "errs"
            with patch.dict(os.environ, {"ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS": "1"}), \
                 patch.object(proxy, "UPSTREAM_ERROR_DIR", directory), \
                 patch.object(proxy, "LOG_FILE", os.path.join(tmp, "proxy.log")):
                proxy._capture_upstream_error(err, {"model": "gemini-3-flash", "request": {"secret": 1}}, preread)
            files = list(directory.glob("upstream-*.json"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text())
            self.assertEqual(record["body"], body.decode())
            self.assertEqual(record["status"], 429)
            self.assertNotIn("secret", files[0].read_text())

    def test_capture_enabled_end_to_end_keeps_body_and_still_parses(self):
        body = _error_body("QUOTA_EXHAUSTED", "21600s", _iso(time.time() + 3600))

        def responder(n):
            if n == 1:
                raise _make_http_429(body)
            return io.BytesIO(_sse_success_bytes("ok"))

        with tempfile.TemporaryDirectory() as tmp:
            directory = proxy.Path(tmp) / "errs"
            with patch.dict(os.environ, {"ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS": "1"}), \
                 patch.object(proxy, "UPSTREAM_ERROR_DIR", directory), \
                 patch.object(proxy, "LOG_FILE", os.path.join(tmp, "proxy.log")):
                res, calls, _ = self._run({"model": "gemini-3-flash", "request": {}}, responder)
            files = list(directory.glob("upstream-*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())["body"], body.decode())
        self.assertEqual(len(calls), 2)
        self.assertIn("gemini", proxy._credits_exhausted_until)  # parsing not broken by capture


if __name__ == "__main__":
    unittest.main()
