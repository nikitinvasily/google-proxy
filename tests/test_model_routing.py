"""Model IDs sent to Google are exactly the IDs requested by the client."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class ModelRoutingTests(unittest.TestCase):
    def test_chat_does_not_silently_map_legacy_model_names(self):
        for model in ("gemini-3.8-flash", "gemini-3.8-flash-low",
                      "gemini-4-argon", "claude-3-7-sonnet", "gemini-3-flash"):
            with self.subTest(model=model):
                handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
                handler._read_body = lambda: {
                    "model": model, "messages": [{"role": "user", "content": "hello"}],
                }
                sent = []
                handler._handle_nonstream = lambda envelope, openai_model: sent.append((envelope, openai_model))
                with patch.object(proxy, "get_project_id", return_value="test-project"):
                    handler._handle_chat()
                self.assertEqual(len(sent), 1)
                self.assertEqual(sent[0][0]["model"], model)
                self.assertEqual(sent[0][1], model)

    def test_native_passthrough_does_not_silently_map_legacy_model_names(self):
        for model in ("gemini-3.8-flash", "gemini-3.8-flash-low",
                      "gemini-4-argon", "claude-3-7-sonnet", "gemini-3-flash"):
            with self.subTest(model=model):
                body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
                with patch.object(proxy, "get_project_id", return_value="test-project"):
                    envelope = proxy._build_passthrough_envelope(model, body)
                self.assertEqual(envelope["model"], model)
                self.assertNotIn("generationConfig", envelope["request"])
                self.assertNotIn("generationConfig", body)


if __name__ == "__main__":
    unittest.main()
