import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class PrivacyTests(unittest.TestCase):
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
