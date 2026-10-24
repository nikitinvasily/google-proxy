"""The OpenAI model picker reflects the active Antigravity catalog."""
import io
import json
import sys
import unittest
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return self.payload


class ModelCatalogTests(unittest.TestCase):
    def handle(self, path):
        handler: Any = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.path = path
        handler.wfile = io.BytesIO()
        handler.statuses = []
        handler.logged = []
        handler.send_response = handler.statuses.append
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler._log_req = lambda *args: handler.logged.append(args)
        handler.do_GET()
        return handler, json.loads(handler.wfile.getvalue())

    def test_current_catalog_replaces_old_claude_and_shows_gpt_oss(self):
        available = {'models': {
            'claude-sonnet-5-5-high': {'displayName': 'Claude Sonnet 5.5'},
            'claude-opus-5-5-low': {'displayName': 'Claude Opus 5.5'},
            'gpt-oss-120b-medium': {'displayName': 'GPT-OSS 120B'},
            'gemini-3-flash': {},
            'gemini-4-argon-low': {},
            'gemini-4-argon-medium': {},
            'gemini-4-argon-high': {},
            'gemini-3.1-flash-image': {'supportsImages': True},
            'chat_20706': {'isInternal': True},
            'tab_flash_lite_preview': {'isInternal': True},
            'private-marker\n': {'displayName': 'private-marker'},
        }, 'privateData': 'private-marker'}
        with patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
             patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', return_value=FakeResponse(available)) as fetch:
            handler, body = self.handle('/v1/models')
        self.assertEqual(handler.statuses, [200])
        self.assertEqual([item['id'] for item in body['data']], [
            'claude-opus-5-5-low', 'claude-sonnet-5-5-high',
            'gemini-3-flash', 'gemini-4-argon-high', 'gemini-4-argon-low',
            'gemini-4-argon-medium', 'gpt-oss-120b-medium',
        ])
        self.assertEqual(fetch.call_count, 1)
        request = fetch.call_args.args[0]
        self.assertEqual(request.get_method(), 'POST')
        self.assertTrue(request.full_url.endswith('/v1internal:fetchAvailableModels'))
        self.assertEqual(json.loads(request.data), {'project': proxy.get_project_id()})
        self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-token')
        self.assertEqual(request.get_header('User-agent'), 'antigravity/cli/1.4.7')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_model_picker_reuses_recent_catalog_then_refreshes(self):
        first = {'models': {'claude-sonnet-5-5-high': {}}}
        second = {'models': {'gpt-oss-120b-medium': {}}}
        with patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
             patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[FakeResponse(first), FakeResponse(second)]) as fetch, \
             patch.object(proxy.time, 'monotonic', return_value=100.0) as clock:
            _, one = self.handle('/models')
            _, two = self.handle('/v1/models')
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(one['data'], two['data'])
            clock.return_value = 401.0
            _, three = self.handle('/models')
            self.assertEqual(fetch.call_count, 2)
        self.assertEqual([item['id'] for item in three['data']], ['gpt-oss-120b-medium'])

    def test_catalog_exposes_valid_per_model_context_and_output_limits(self):
        available = {'models': {
            'gemini-3-flash': {'maxTokens': 1048576, 'maxOutputTokens': 65536,
                               'privateData': 'private-marker'},
            'gemini-4-argon-high': {'maxTokens': 524288, 'maxOutputTokens': 32768},
            'claude-sonnet-5-5-high': {'maxTokens': True, 'maxOutputTokens': '65536'},
            'gpt-oss-120b-medium': {'maxTokens': -1, 'maxOutputTokens': 0},
        }}
        with patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
             patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', return_value=FakeResponse(available)) as fetch:
            first, body = self.handle('/v1/models')
            second, alias = self.handle('/models')
        self.assertEqual(first.statuses, [200])
        self.assertEqual(second.statuses, [200])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(alias, body)
        by_id = {item['id']: item for item in body['data']}
        self.assertEqual(by_id['gemini-3-flash']['context_length'], 1048576)
        self.assertEqual(by_id['gemini-3-flash']['max_output_tokens'], 65536)
        self.assertEqual(by_id['gemini-4-argon-high']['context_length'], 524288)
        for model_id in ('claude-sonnet-5-5-high', 'gpt-oss-120b-medium'):
            self.assertNotIn('context_length', by_id[model_id])
            self.assertNotIn('max_output_tokens', by_id[model_id])
        self.assertNotIn('private-marker', first.wfile.getvalue().decode())

    def test_missing_or_non_text_catalog_does_not_advertise_stale_models(self):
        for payload in ({'models': []}, {'models': {}},
                        {'models': {'gemini-3.1-flash-image': {}, 'chat_20706': {'isInternal': True}}}):
            with self.subTest(payload=payload), \
                 patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
                 patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
                 patch.object(proxy, 'urlopen', return_value=FakeResponse(payload)):
                handler, body = self.handle('/models')
                self.assertEqual(handler.statuses, [502])
                self.assertEqual(handler.logged[0][2], 502)
                self.assertNotIn('data', body)
                self.assertNotIn('claude-sonnet-4-6', handler.wfile.getvalue().decode())

    def test_new_agent_family_appears_without_code_update_but_media_is_excluded(self):
        available = {'models': {
            'o5-reasoning': {'displayName': 'Future OpenAI reasoning model'},
            'vision-xl': {'displayName': 'Image model'},
            'gemini-tab': {'displayName': 'Tab completion model'},
            'gemini-3-flash': {},
        }, 'agentModelSorts': [{'category': {'modelId': 'o5-reasoning'}},
                              {'modelId': 'vision-xl'}, {'modelId': 'gemini-tab'}],
           'imageGenerationModelIds': ['vision-xl'], 'tabModelIds': ['gemini-tab']}
        with patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
             patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', return_value=FakeResponse(available)):
            handler, body = self.handle('/models')
        self.assertEqual(handler.statuses, [200])
        self.assertEqual([item['id'] for item in body['data']],
                         ['gemini-3-flash', 'o5-reasoning'])

    def test_upstream_error_does_not_leak_body_or_cached_stale_catalog(self):
        private_body = b'private-upstream-token'
        upstream_error = HTTPError('https://upstream.invalid/', 429, 'private-reason', Message(),
                                   io.BytesIO(private_body))
        with patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
             patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse({'models': {'gemini-3-flash': {}}}),
                 upstream_error,
                 FakeResponse({'models': {'gpt-oss-120b-medium': {}}}),
             ]) as fetch, \
             patch.object(proxy.time, 'monotonic', return_value=100.0) as clock:
            _, first = self.handle('/models')
            self.assertEqual(first['data'][0]['id'], 'gemini-3-flash')
            clock.return_value = 401.0
            handler, body = self.handle('/v1/models')
            self.assertEqual(handler.statuses, [429])
            self.assertEqual(handler.logged[0][2], 429)
            self.assertNotIn('data', body)
            self.assertNotIn('private-upstream-token', handler.wfile.getvalue().decode())
            self.assertNotIn('private-reason', handler.wfile.getvalue().decode())
            _, recovered = self.handle('/v1/models')
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual(recovered['data'][0]['id'], 'gpt-oss-120b-medium')

    def test_missing_oauth_returns_401_without_upstream_request(self):
        with patch.dict(proxy.__dict__, {'_models_cache_ids': None, '_models_cache_until': 0.0}), \
             patch.object(proxy, 'get_access_token', side_effect=RuntimeError('private-token')), \
             patch.object(proxy, 'urlopen') as fetch:
            handler, body = self.handle('/models')
        self.assertEqual(handler.statuses, [401])
        self.assertEqual(handler.logged[0][2], 401)
        self.assertNotIn('private-token', handler.wfile.getvalue().decode())
        self.assertNotIn('data', body)
        fetch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
