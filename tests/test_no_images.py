import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class NoImagesTests(unittest.TestCase):
    def test_openai_image_url_is_rejected_before_fetch(self):
        with self.assertRaisesRegex(ValueError, "[Ii]mage.*not supported"):
            proxy._content_to_parts([{"type": "image_url", "image_url": {"url": "file:///etc/hosts"}}])

    def test_openai_input_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Ii]mage.*not supported"):
            proxy._content_to_parts([{"type": "input_image", "image_url": "data:image/png;base64,AA=="}])

    def test_native_inline_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Ii]mage.*not supported"):
            proxy._build_passthrough_envelope("gemini-3-flash", {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "AA=="}}]}]})

    def test_native_file_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Ii]mage.*not supported"):
            proxy._build_passthrough_envelope("gemini-3-flash", {"contents": [{"parts": [{"fileData": {"fileUri": "gs://x"}}]}]})

    def test_text_schema_can_name_a_field_image(self):
        env = proxy._build_passthrough_envelope("gemini-3-flash", {
            "contents": [{"parts": [{"text": "describe the label"}]}],
            "generationConfig": {"responseSchema": {"type": "OBJECT", "properties": {"image": {"type": "STRING"}}}},
        })
        self.assertIn("image", env["request"]["generationConfig"]["responseSchema"]["properties"])

    def test_text_schema_can_name_a_field_image_url(self):
        env = proxy._build_passthrough_envelope("gemini-3-flash", {
            "contents": [{"parts": [{"text": "describe the label"}]}],
            "generationConfig": {"responseSchema": {"type": "OBJECT", "properties": {"image_url": {"type": "STRING"}}}},
        })
        self.assertIn("image_url", env["request"]["generationConfig"]["responseSchema"]["properties"])

    def test_text_only_parts_and_native_passthrough_still_work(self):
        self.assertEqual(proxy._content_to_parts([{"type": "text", "text": "hello"}]), [{"text": "hello"}])
        env = proxy._build_passthrough_envelope("gemini-3-flash", {"contents": [{"parts": [{"text": "hello"}]}]})
        self.assertEqual(env["request"]["contents"][0]["parts"], [{"text": "hello"}])


if __name__ == "__main__":
    unittest.main()
