import io
import json
import os
import stat
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class PrivacyTests(unittest.TestCase):
    def test_opt_in_upstream_http_error_is_saved_only_in_private_bounded_file(self):
        envelope = {"model": "claude-sonnet-5-5-medium", "request": {
            "contents": [{"parts": [{"text": "PRIVATE_PROMPT_SENTINEL"}]}]}}
        response_body = b'{"error":{"code":400,"message":"Unsupported field"}}'
        error = HTTPError(proxy.STREAM_GENERATE_CONTENT_URL, 400, "Bad Request", Message(),
                          io.BytesIO(response_body))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "upstream-errors"
            logged = []
            with patch.dict(os.environ, {"ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS": "1"}), \
                 patch.object(proxy, "UPSTREAM_ERROR_DIR", directory, create=True), \
                 patch.object(proxy, "get_access_token", return_value="dummy"), \
                 patch.object(proxy, "urlopen", side_effect=error), \
                 patch.object(proxy, "_log", side_effect=logged.append):
                with self.assertRaises(proxy.UpstreamError) as captured:
                    list(proxy.stream_generate_content(envelope))
            files = list(directory.glob("*.json"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text())
            self.assertEqual(record["status"], 400)
            self.assertEqual(record["model"], "claude-sonnet-5-5-medium")
            self.assertEqual(record["body"], response_body.decode())
            self.assertFalse(record["truncated"])
            self.assertNotIn("PRIVATE_PROMPT_SENTINEL", files[0].read_text())
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
            self.assertNotIn("Unsupported field", " ".join(logged) + str(captured.exception))
            self.assertEqual(captured.exception.status, 400)

    def test_opt_in_upstream_error_body_is_size_limited(self):
        error = HTTPError(proxy.STREAM_GENERATE_CONTENT_URL, 400, "Bad Request", Message(),
                          io.BytesIO(b"x" * 20000))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "upstream-errors"
            with patch.dict(os.environ, {"ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS": "1"}), \
                 patch.object(proxy, "UPSTREAM_ERROR_DIR", directory, create=True), \
                 patch.object(proxy, "get_access_token", return_value="dummy"), \
                 patch.object(proxy, "urlopen", side_effect=error), \
                 patch.object(proxy, "_log"):
                with self.assertRaises(proxy.UpstreamError):
                    list(proxy.stream_generate_content({"model": "test", "request": {}}))
            record = json.loads(next(directory.glob("*.json")).read_text())
            self.assertTrue(record["truncated"])
            self.assertLessEqual(len(record["body"].encode()), 8192)

    def test_opt_in_record_bounds_invalid_utf8_and_untrusted_model_id(self):
        error = HTTPError(proxy.STREAM_GENERATE_CONTENT_URL, 400, "Bad Request", Message(),
                          io.BytesIO(b"\xff" * 20000))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "upstream-errors"
            with patch.dict(os.environ, {"ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS": "1"}), \
                 patch.object(proxy, "UPSTREAM_ERROR_DIR", directory), \
                 patch.object(proxy, "get_access_token", return_value="dummy"), \
                 patch.object(proxy, "urlopen", side_effect=error), patch.object(proxy, "_log"):
                with self.assertRaises(proxy.UpstreamError):
                    list(proxy.stream_generate_content({"model": "x" * 20000, "request": {}}))
            saved = next(directory.glob("*.json"))
            record = json.loads(saved.read_text())
            self.assertLessEqual(len(record["model"]), 128)
            self.assertLessEqual(len(record["body"].encode("utf-8")), 8192)
            self.assertTrue(record["truncated"])
            self.assertLessEqual(saved.stat().st_size, 20000)

    def test_upstream_echo_of_prompt_stays_in_private_record_only(self):
        secret = "PRIVATE_PROMPT_SENTINEL"
        error = HTTPError(proxy.STREAM_GENERATE_CONTENT_URL, 400, "Bad Request", Message(),
                          io.BytesIO(json.dumps({"error": secret}).encode()))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "upstream-errors"
            logged = []
            with patch.dict(os.environ, {"ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS": "1"}), \
                 patch.object(proxy, "UPSTREAM_ERROR_DIR", directory), \
                 patch.object(proxy, "get_access_token", return_value="dummy"), \
                 patch.object(proxy, "urlopen", side_effect=error), \
                 patch.object(proxy, "_log", side_effect=logged.append):
                with self.assertRaises(proxy.UpstreamError) as captured:
                    list(proxy.stream_generate_content({"model": "test", "request": {
                        "contents": [{"parts": [{"text": secret}]}]}}))
            self.assertIn(secret, next(directory.glob("*.json")).read_text())
            self.assertNotIn(secret, " ".join(logged) + str(captured.exception))

    def test_upstream_error_does_not_persist_prompt(self):
        envelope = {"request": {"contents": [{"parts": [{"text": "PRIVATE_PROMPT_SENTINEL"}]}]}}
        error = HTTPError(proxy.STREAM_GENERATE_CONTENT_URL, 400, "Bad Request", {}, io.BytesIO(b"bad request"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            old_cwd = os.getcwd()
            try:
                os.chdir(directory)
                with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", side_effect=error):
                    with self.assertRaises(proxy.UpstreamError):
                        list(proxy.stream_generate_content(envelope))
                self.assertFalse((Path(directory) / "last_failed_request.json").exists())
            finally:
                os.chdir(old_cwd)
    def test_upstream_error_body_is_not_logged(self):
        envelope = {"request": {"contents": [{"parts": [{"text": "PRIVATE_PROMPT_SENTINEL"}]}]}}
        error = HTTPError(proxy.STREAM_GENERATE_CONTENT_URL, 400, "Bad Request", {}, io.BytesIO(b"private prompt=SECRET_SENTINEL"))
        logged = []
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", side_effect=error), patch.object(proxy, "_log", side_effect=logged.append):
            with self.assertRaises(proxy.UpstreamError) as captured:
                list(proxy.stream_generate_content(envelope))
        self.assertNotIn("SECRET_SENTINEL", " ".join(logged) + str(captured.exception))


if __name__ == "__main__":
    unittest.main()
