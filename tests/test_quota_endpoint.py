"""Quota routes return only read-only, whitelisted Google quota buckets."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy

QUOTA_RESPONSE = {
    'buckets': [
        {'modelId': 'gemini-3-flash', 'tokenType': 'WTUS', 'remainingFraction': 0,
         'resetTime': '2026-10-24T00:00:00Z', 'privateData': 'private-marker'},
        {'modelId': 'gemini-2.5-flash', 'tokenType': 'WTUS', 'remainingFraction': 0.42},
    ],
    'accessToken': 'private-marker',
}

SUMMARY_RESPONSE = {
    'groups': [
        {'displayName': 'Gemini Models', 'description': 'private-marker', 'buckets': [
            {'bucketId': 'gemini-weekly', 'window': 'weekly', 'remainingFraction': 0.91,
             'resetTime': '2026-10-29T20:51:37Z'},
        ]},
        {'displayName': 'Gemini 4 Argon', 'buckets': [
            {'bucketId': 'gemini-argon-wtu-weekly', 'window': 'weekly',
             'remainingFraction': 0, 'resetTime': '2026-10-29T20:22:57Z',
             'privateData': 'private-marker'},
        ]},
    ],
    'accessToken': 'private-marker',
}


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return self.payload


class QuotaRouteTests(unittest.TestCase):
    def handle(self, path):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.path = path
        h.wfile = io.BytesIO()
        h.statuses = []
        h.logged = []
        h.send_response = h.statuses.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        h._log_req = lambda *args: h.logged.append(args)
        h.do_GET()
        return h, json.loads(h.wfile.getvalue())

    def test_route_includes_direct_subscription_groups_and_argon_without_cli(self):
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse(QUOTA_RESPONSE), FakeResponse(SUMMARY_RESPONSE),
             ]) as fetch:
            handler, body = self.handle('/v1/quota')
        self.assertEqual(handler.statuses, [200])
        self.assertEqual(body['groups'], [
            {'name': 'Gemini Models', 'buckets': [
                {'id': 'gemini-weekly', 'window': 'weekly', 'remaining_fraction': 0.91,
                 'reset_time': '2026-10-29T20:51:37Z'},
            ]},
            {'name': 'Gemini 4 Argon', 'buckets': [
                {'id': 'gemini-argon-wtu-weekly', 'window': 'weekly',
                 'remaining_fraction': 0, 'reset_time': '2026-10-29T20:22:57Z'},
            ]},
        ])
        self.assertEqual(body['groups_source'], 'retrieveUserQuotaSummary')
        self.assertEqual(fetch.call_count, 2)
        request = fetch.call_args.args[0]
        self.assertEqual(request.get_method(), 'POST')
        self.assertTrue(request.full_url.endswith('/v1internal:retrieveUserQuotaSummary'))
        self.assertEqual(json.loads(request.data), {'project': proxy.get_project_id()})
        self.assertEqual(request.get_header('Authorization'), 'Bearer synthetic-token')
        self.assertEqual(request.get_header('User-agent'), 'antigravity/cli/1.4.7')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_partial_summary_does_not_claim_complete_quotas(self):
        summary = {'groups': [{'displayName': 'Claude and GPT models', 'buckets': [
            {'bucketId': '3p-weekly', 'window': 'weekly', 'remainingFraction': 1},
        ]}]}
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse(QUOTA_RESPONSE), FakeResponse(summary),
             ]):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [502])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')

    def test_missing_fraction_does_not_become_fake_zero(self):
        summary = {'groups': [{'displayName': 'Gemini 4 Argon', 'buckets': [
            {'bucketId': 'gemini-argon-wtu-weekly', 'window': 'weekly',
             'resetTime': '2026-10-29T20:22:57Z'},
        ]}]}
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse(QUOTA_RESPONSE), FakeResponse(summary),
             ]):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [502])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')

    def test_summary_http_error_is_sanitized(self):
        upstream = HTTPError('https://example.invalid', 429, 'private-marker', {},
                             io.BytesIO(b'{"secret":"private-marker"}'))
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse(QUOTA_RESPONSE), upstream,
             ]):
            handler, body = self.handle('/v1/quota')
        self.assertEqual(handler.statuses, [429])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_invalid_summary_group_rejects_partial_result(self):
        summary = {'groups': [
            {'displayName': 'private-marker\nsecret', 'buckets': [
                {'bucketId': 'bad', 'window': 'weekly', 'remainingFraction': 1},
            ]},
            {'displayName': 'Gemini 4 Argon', 'buckets': [
                {'bucketId': 'private-marker\n', 'window': 'weekly', 'remainingFraction': 0},
                {'bucketId': 'gemini-argon-wtu-weekly', 'window': 'weekly',
                 'remainingFraction': 0, 'resetTime': 'private-marker'},
            ]},
        ], 'privateData': 'private-marker'}
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse(QUOTA_RESPONSE), FakeResponse(summary),
             ]):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [502])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_invalid_bucket_in_valid_group_rejects_partial_result(self):
        summary = {'groups': [{'displayName': 'Gemini 4 Argon', 'buckets': [
            {'bucketId': 'gemini-argon-wtu-weekly', 'window': 'weekly',
             'remainingFraction': 0},
            {'bucketId': 'private-marker\n', 'window': 'weekly',
             'remainingFraction': 0.5},
        ]}]}
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=[
                 FakeResponse(QUOTA_RESPONSE), FakeResponse(summary),
             ]):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [502])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_both_routes_return_fresh_quota_without_private_fields(self):
        def token_without_outer_lock():
            acquired = proxy._token_lock.acquire(blocking=False)
            self.assertTrue(acquired, 'quota handler must not acquire token lock before get_access_token')
            proxy._token_lock.release()
            return 'synthetic-token'

        with patch.object(proxy, 'get_access_token', side_effect=token_without_outer_lock), \
             patch.object(proxy, 'urlopen', side_effect=lambda request, **kwargs: FakeResponse(
                 SUMMARY_RESPONSE if request.full_url.endswith(':retrieveUserQuotaSummary')
                 else QUOTA_RESPONSE)) as fetch:
            for path in ('/quota', '/v1/quota'):
                with self.subTest(path=path):
                    handler, body = self.handle(path)
                    self.assertEqual(handler.statuses, [200])
                    self.assertEqual(handler.logged[0][2], 200)
                    self.assertEqual(body['source'], 'retrieveUserQuota')
                    self.assertEqual(body['buckets'], [
                        {'model': 'gemini-3-flash', 'token_type': 'WTUS', 'remaining_fraction': 0,
                         'reset_time': '2026-10-24T00:00:00Z'},
                        {'model': 'gemini-2.5-flash', 'token_type': 'WTUS', 'remaining_fraction': 0.42,
                         'reset_time': None},
                    ])
                    self.assertNotIn('private-marker', handler.wfile.getvalue().decode())
            self.assertEqual(fetch.call_count, 4)
            req = fetch.call_args_list[0].args[0]
            self.assertEqual(req.get_method(), 'POST')
            self.assertTrue(req.full_url.endswith('/v1internal:retrieveUserQuota'))
            self.assertEqual(json.loads(req.data), {'project': proxy.get_project_id()})
            self.assertEqual(req.get_header('Authorization'), 'Bearer synthetic-token')

    def test_rejects_untrusted_model_ids_and_reset_time(self):
        payload = {'buckets': [
            {'modelId': 'private-marker\n', 'remainingFraction': 0.5},
            {'modelId': 'gemini-3-flash', 'remainingFraction': 0.5, 'resetTime': 'private-marker'},
        ]}
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=lambda request, **kwargs: FakeResponse(
                 SUMMARY_RESPONSE if request.full_url.endswith(':retrieveUserQuotaSummary')
                 else payload)):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [200])
        self.assertEqual(len(body['buckets']), 1)
        self.assertEqual(body['buckets'][0]['reset_time'], None)
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_upstream_http_error_does_not_expose_error_body(self):
        error = HTTPError('https://example.invalid', 429, 'quota error', {},
                          io.BytesIO(b'{"secret":"private-marker"}'))
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=error):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [429])
        self.assertEqual(handler.logged[0][2], 429)
        self.assertEqual(body['error']['code'], 'quota_upstream_error')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_invalid_upstream_shape_is_not_empty_success(self):
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', return_value=FakeResponse({'models': {}})):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [502])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')

    def test_backend_runtime_error_is_not_reported_as_auth_failure(self):
        with patch.object(proxy, 'get_access_token', return_value='synthetic-token'), \
             patch.object(proxy, 'urlopen', side_effect=RuntimeError('private-marker')):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [502])
        self.assertEqual(body['error']['code'], 'quota_upstream_error')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())

    def test_auth_error_is_sanitized(self):
        with patch.object(proxy, 'get_access_token', side_effect=RuntimeError('private-marker')):
            handler, body = self.handle('/quota')
        self.assertEqual(handler.statuses, [401])
        self.assertEqual(body['error']['code'], 'auth_error')
        self.assertNotIn('private-marker', handler.wfile.getvalue().decode())


if __name__ == '__main__':
    unittest.main()
