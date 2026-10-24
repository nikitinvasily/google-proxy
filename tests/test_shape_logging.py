import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import antigravity_proxy as proxy

SECRET = 'PRIVATE_PROMPT_SENTINEL'

class ShapeLoggingTests(unittest.TestCase):
    def test_bad_middle_candidate_does_not_hide_later_function_call(self):
        trace = proxy.ShapeTrace({'request': {'tools': [{'functionDeclarations': [{'name': 'lookup'}]}]}}, stream=True)
        trace.observe({'response': {'candidates': [
            {'content': {'parts': [{'text': 'Checking now'}]}},
            {'content': {'parts': 17}},
            {'content': {'parts': [{'functionCall': {'name': 'lookup', 'args': {}}}]}, 'finishReason': 'STOP'},
        ]}})
        with patch.object(proxy, '_log') as log:
            trace.emit('chatcmpl-test', 'stop', 0)
        data = json.loads(log.call_args.args[0].split(' ', 1)[1])
        self.assertEqual(data['upstream_candidates'], 3)
        self.assertEqual(data['upstream_function_parts'], 1)
        self.assertEqual(data['upstream_nonfirst_function_parts'], 1)

    def test_log_failure_never_blocks_nonstream_response(self):
        response = {'candidates': [{'content': {'parts': [{'text': 'OK'}]}, 'finishReason': 'STOP'}]}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'call_generate_content', return_value=response), patch.object(proxy, '_log', side_effect=OSError('disk full')):
            h._handle_nonstream({'request': {}}, 'synthetic')
        self.assertEqual(json.loads(h.wfile.getvalue())['choices'][0]['message']['content'], 'OK')

    def test_log_failure_never_interrupts_stream(self):
        event = {'response': {'candidates': [{'content': {'parts': [{'text': 'OK'}]}, 'finishReason': 'STOP'}]}}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'stream_generate_content', return_value=iter([event])), patch.object(proxy, '_log', side_effect=OSError('disk full')):
            h._handle_stream({'request': {}}, 'synthetic')
        wire = h.wfile.getvalue().decode()
        self.assertIn('data: [DONE]', wire)
        chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_bad_second_candidate_does_not_interrupt_valid_stream(self):
        event = {'response': {'candidates': [
            {'content': {'parts': [{'text': 'OK'}]}, 'finishReason': 'STOP'},
            {'content': 'not-a-dict', 'finishReason': 'STOP'},
        ]}}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'stream_generate_content', return_value=iter([event])), patch.object(proxy, '_log'):
            h._handle_stream({'request': {}}, 'synthetic')
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith('data: {')]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content','') for c in chunks), 'OK')

    def test_bad_second_candidate_parts_do_not_interrupt_valid_stream(self):
        event = {'response': {'candidates': [
            {'content': {'parts': [{'text': 'OK'}]}, 'finishReason': 'STOP'},
            {'content': {'parts': 17}, 'finishReason': 'STOP'},
        ]}}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'stream_generate_content', return_value=iter([event])), patch.object(proxy, '_log'):
            h._handle_stream({'request': {}}, 'synthetic')
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith('data: {')]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'stop')

    def test_textual_function_call_is_counted_in_stream_log(self):
        event = {'response': {'candidates': [{'content': {'parts': [
            {'text': 'call:terminal{"command":"pwd"}'}]}, 'finishReason': 'STOP'}]}}
        envelope = {'request': {'tools': [{'functionDeclarations': [{'name': 'terminal'}]}]}}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'stream_generate_content', return_value=iter([event])), patch.object(proxy, '_log') as log:
            h._handle_stream(envelope, 'synthetic')
        line = next(c.args[0] for c in log.call_args_list if c.args[0].startswith('shape '))
        data = json.loads(line.split(' ', 1)[1])
        self.assertEqual(data['output_tool_calls'], 1)
        self.assertEqual(data['finish_reason'], 'tool_calls')

    def test_trace_flags_function_call_outside_first_candidate(self):
        envelope = {'request': {'tools': [{'functionDeclarations': [{'name': 'lookup'}]}]}}
        trace = proxy.ShapeTrace(envelope, stream=True)
        trace.observe({'response': {'candidates': [
            {'index': 1, 'content': {'parts': [{'text': 'Checking now'}]}, 'finishReason': 'STOP'},
            {'index': 0, 'content': {'parts': [{'functionCall': {'name': 'lookup', 'args': {}}}]}, 'finishReason': 'STOP'},
        ]}})
        with patch.object(proxy, '_log') as log:
            trace.emit('chatcmpl-nonfirst', 'stop', 0)
        data = json.loads(log.call_args.args[0].split(' ', 1)[1])
        self.assertEqual(data['upstream_candidate_indices'], [1, 0])
        self.assertEqual(data['upstream_nonfirst_function_parts'], 1)
        self.assertEqual(data['upstream_allowed_function_parts'], 1)
        self.assertEqual(data['output_tool_calls'], 0)

    def test_unknown_or_malformed_finish_reason_never_breaks_logging(self):
        trace = proxy.ShapeTrace({'request': {}}, stream=True)
        trace.observe({'response': {'candidates': [
            {'finishReason': ['PRIVATE_FINISH_SENTINEL'], 'content': {'parts': []}},
            {'finishReason': 'PRIVATE_FINISH_SENTINEL', 'content': {'parts': []}},
        ]}})
        with patch.object(proxy, '_log') as log:
            trace.emit('chatcmpl-test', 'stop', 0)
        self.assertEqual(json.loads(log.call_args.args[0].split(' ', 1)[1])['finish_reasons'], ['UNKNOWN'])
        self.assertNotIn('PRIVATE_FINISH_SENTINEL', log.call_args.args[0])

    def test_stream_handler_correlates_raw_function_call_with_emitted_call(self):
        upstream = {'response': {'candidates': [{'index': 0, 'content': {'parts': [
            {'functionCall': {'name': 'lookup', 'args': {'secret': SECRET}}, 'thoughtSignature': SECRET}
        ]}, 'finishReason': 'STOP'}]}}
        envelope = {'request': {'tools': [{'functionDeclarations': [{'name': 'lookup'}]}],
                                'contents': [{'role': 'user', 'parts': [{'text': SECRET}]}]},
                    'requestId': 'agent-test-stream'}
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda *args: None
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        with patch.object(proxy, 'stream_generate_content', return_value=iter([upstream])), patch.object(proxy, '_log') as log:
            h._handle_stream(envelope, 'gemini-3.8-flash')
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith('data: {')]
        line = next(c.args[0] for c in log.call_args_list if c.args[0].startswith('shape '))
        self.assertNotIn(SECRET, line)
        data = json.loads(line.split(' ', 1)[1])
        self.assertEqual(data['completion_id'], chunks[0]['id'])
        self.assertEqual(data['upstream_function_parts'], 1)
        self.assertEqual(data['output_tool_calls'], 1)
        self.assertEqual(data['finish_reason'], 'tool_calls')

    def test_trace_keeps_only_structural_metadata_and_correlation_id(self):
        envelope = {
            'requestId': 'agent-test123',
            'model': 'gemini-3-flash',
            'request': {
                'contents': [{'role':'user','parts':[{'text':SECRET}]}],
                'systemInstruction': {'parts':[{'text': SECRET}]},
                'tools': [{'functionDeclarations':[{'name':'lookup','description':SECRET}]}],
                'toolConfig': {'functionCallingConfig': {'mode':'AUTO'}},
            },
        }
        upstream = {'response': {'candidates': [{'index':0, 'content':{'parts':[
            {'text': SECRET, 'thought':True, 'thoughtSignature': SECRET},
            {'text': SECRET},
            {'functionCall':{'name':'lookup','args':{'secret':SECRET}},'thoughtSignature':SECRET},
            {'functionCall':{'name':SECRET,'args':{'secret':SECRET}}},
        ]},'finishReason':'STOP'}]}}
        trace = proxy.ShapeTrace(envelope, stream=True)
        trace.observe(upstream)
        with patch.object(proxy,'_log') as log:
            trace.emit('chatcmpl-test123', 'tool_calls', 1)
        self.assertEqual(log.call_count,1)
        line = log.call_args.args[0]
        self.assertNotIn(SECRET,line)
        self.assertNotIn('lookup',line)
        self.assertIn('chatcmpl-test123',line)
        data = json.loads(line.split(' ',1)[1])
        self.assertEqual(data['offered_tools'],1)
        self.assertEqual(data['mode'],'AUTO')
        self.assertEqual(data['upstream_function_parts'],2)
        self.assertEqual(data['upstream_allowed_function_parts'],1)
        self.assertEqual(data['upstream_thought_parts'],1)
        self.assertEqual(data['upstream_visible_text_parts'],1)
        self.assertEqual(data['output_tool_calls'],1)
        self.assertEqual(data['finish_reasons'],['STOP'])

    def test_nonstream_handler_emits_shape_with_completion_id(self):
        response = {'candidates':[{'content':{'parts':[{'text':'PUBLIC'}]},'finishReason':'STOP'}]}
        h=proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile=io.BytesIO()
        h.send_response=lambda *args: None
        h.send_header=lambda *args: None
        h.end_headers=lambda: None
        def fake_call(envelope, trace=None):
            trace.observe({'response':response})
            return response
        with patch.object(proxy,'call_generate_content',side_effect=fake_call),patch.object(proxy,'_log') as log:
            h._handle_nonstream({'request': {'tools': [],'contents':[]},'requestId':'agent-test'}, 'gemini-3.8-flash')
        body=json.loads(h.wfile.getvalue())
        line=next(c.args[0] for c in log.call_args_list if c.args[0].startswith('shape '))
        data=json.loads(line.split(' ',1)[1])
        self.assertEqual(data['completion_id'],body['id'])
        self.assertEqual(data['finish_reason'],'stop')
        self.assertEqual(data['upstream_visible_text_parts'],1)

if __name__=='__main__': unittest.main()
