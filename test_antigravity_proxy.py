import unittest
from antigravity_proxy import _extract_usage


class TestExtractUsage(unittest.TestCase):
    def test_extract_usage_basic(self):
        resp = {
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            }
        }
        usage = _extract_usage(resp)
        self.assertEqual(
            usage,
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )
        self.assertNotIn("prompt_tokens_details", usage)
        self.assertNotIn("completion_tokens_details", usage)

    def test_extract_usage_with_cached_and_thoughts(self):
        resp = {
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
                "cachedContentTokenCount": 64,
                "thoughtsTokenCount": 32,
            }
        }
        usage = _extract_usage(resp)
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 64})
        self.assertEqual(usage["completion_tokens_details"], {"reasoning_tokens": 32})

    def test_extract_usage_from_candidates(self):
        resp = {
            "candidates": [
                {
                    "usageMetadata": {
                        "promptTokenCount": 40,
                        "candidatesTokenCount": 10,
                        "totalTokenCount": 50,
                        "cachedContentTokenCount": 20,
                    }
                }
            ]
        }
        usage = _extract_usage(resp)
        self.assertEqual(usage["prompt_tokens"], 40)
        self.assertEqual(usage["completion_tokens"], 10)
        self.assertEqual(usage["total_tokens"], 50)
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 20})
        self.assertNotIn("completion_tokens_details", usage)


if __name__ == "__main__":
    unittest.main()
