import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


EMPTY_STOP = {'candidates': [{'content': {'parts': [{'text': '', 'thought': True}, {'text': '', 'thoughtSignature': 'sig'}]}, 'finishReason': 'STOP'}]}


class EmptyCompletionTests(unittest.TestCase):
    def test_literal_empty_after_tool_result_can_be_requested(self):
        response = {'candidates': [{'content': {'parts': [{'text': '(empty)'}]}, 'finishReason': 'STOP'}]}
        envelope = {'request': {'contents': [{'role': 'user', 'parts': [{'functionResponse': {'name': 'lookup', 'response': {}}}]}]}}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = statuses.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'call_generate_content', return_value=response):
            h._handle_nonstream(envelope, 'gemini-3.8-flash')
        self.assertEqual(statuses, [200])
        self.assertEqual(json.loads(h.wfile.getvalue().decode())['choices'][0]['message']['content'], '(empty)')

    def test_stream_split_literal_empty_after_tool_result_can_be_requested(self):
        envelope = {'request': {'contents': [{'role': 'user', 'parts': [{'functionResponse': {'name': 'lookup', 'response': {}}}]}]}}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        events = iter([{'response': {'candidates': [{'content': {'parts': [{'text': '(em'}]}}]}},
                       {'response': {'candidates': [{'content': {'parts': [{'text': 'pty)'}]}, 'finishReason': 'STOP'}]}}])
        with patch.object(proxy, 'stream_generate_content', return_value=events):
            h._handle_stream(envelope, 'gemini-3.8-flash')
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith('data: {')]
        text = ''.join((c['choices'][0]['delta'].get('content') or '') for c in chunks)
        self.assertEqual(text, '(empty)')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_literal_empty_without_tool_history_remains_valid_text(self):
        response = {'candidates': [{'content': {'parts': [{'text': '(empty)'}]}, 'finishReason': 'STOP'}]}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = statuses.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'call_generate_content', return_value=response):
            h._handle_nonstream({'request': {'contents': [{'role': 'user', 'parts': [{'text': 'Reply with (empty)'}]}]}}, 'gemini-3.8-flash')
        self.assertEqual(statuses, [200])
        self.assertEqual(json.loads(h.wfile.getvalue().decode())['choices'][0]['message']['content'], '(empty)')

    def test_nonstream_whitespace_only_is_not_success(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = statuses.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        response = {'candidates': [{'content': {'parts': [{'text': '  \n'}]}, 'finishReason': 'STOP'}]}
        with patch.object(proxy, 'call_generate_content', return_value=response):
            h._handle_nonstream({'request': {'contents': []}}, 'gemini-3.8-flash')
        self.assertEqual(statuses, [502])

    def test_stream_whitespace_only_is_not_success(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        events = iter([{'response': {'candidates': [{'content': {'parts': [{'text': '   '}]}}]}},
                       {'response': {'candidates': [{'content': {'parts': [{'text': '\n'}]}, 'finishReason': 'STOP'}]}}])
        with patch.object(proxy, 'stream_generate_content', return_value=events):
            h._handle_stream({'request': {'contents': [], 'tools': []}}, 'gemini-3.8-flash')
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith('data: {')]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'length')

    def test_nonstream_empty_stop_is_not_success(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        status = []
        h.send_response = status.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'call_generate_content', return_value=EMPTY_STOP):
            h._handle_nonstream({'request': {'contents': []}}, 'gemini-3.8-flash')
        self.assertEqual(status, [502])
        self.assertIn('upstream', h.wfile.getvalue().decode().lower())

    def test_stream_empty_stop_is_not_success(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        events = iter([{'response': {'candidates': [{'content': {'parts': [{'text': '', 'thought': True}]}}]}},
                       {'response': EMPTY_STOP}])
        with patch.object(proxy, 'stream_generate_content', return_value=events):
            h._handle_stream({'request': {'contents': [], 'tools': []}}, 'gemini-3.8-flash')
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith('data: {')]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'length')
        self.assertTrue(any('empty' in (c['choices'][0]['delta'].get('content') or '').lower() for c in chunks))

    def test_nonstream_tool_call_without_text_stays_valid(self):
        response = {'candidates': [{'content': {'parts': [{'functionCall': {'name': 'lookup', 'args': {'item': 'x'}}}]}, 'finishReason': 'STOP'}]}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        status = []
        h.send_response = status.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'call_generate_content', return_value=response):
            h._handle_nonstream({'request': {'tools': [{'functionDeclarations': [{'name': 'lookup'}]}]}}, 'gemini-3.8-flash')
        self.assertEqual(status, [200])
        choice = json.loads(h.wfile.getvalue().decode())['choices'][0]
        self.assertEqual(choice['finish_reason'], 'tool_calls')


if __name__ == '__main__':
    unittest.main()
