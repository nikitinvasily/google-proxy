"""All Google API calls identify this proxy as the installed agy CLI."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class UserAgentTests(unittest.TestCase):
    def test_generation_uses_installed_agy_user_agent(self):
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'):
            for streaming in (False, True):
                with self.subTest(streaming=streaming):
                    request = proxy._make_upstream_request(
                        proxy.GENERATE_CONTENT_URL, {'model': 'gemini-3-flash'},
                        streaming=streaming,
                    )
                    self.assertEqual(request.get_header('User-agent'), 'antigravity/cli/1.4.7')
                    self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-token')


if __name__ == '__main__':
    unittest.main()
