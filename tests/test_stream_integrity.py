import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


def collect_stream(events):
    h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    h.wfile = io.BytesIO()
    h.send_response = lambda *args: None
    h.send_header = lambda *args: None
    h.end_headers = lambda *args: None
    with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
        h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
    return [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines() if line.startswith("data: {")]


class StreamIntegrityTests(unittest.TestCase):
    def test_sse_error_first_event_returns_http_error(self):
        raw = b'event: error\ndata: {"error":{"code":429,"message":"SECRET_SENTINEL"}}\n\n'
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = lambda code: statuses.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", return_value=io.BytesIO(raw)):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        self.assertEqual(statuses, [429])
        self.assertNotIn("SECRET_SENTINEL", h.wfile.getvalue().decode())

    def test_sse_error_after_stop_does_not_report_success(self):
        raw = (b'data: {"response":{"candidates":[{"finishReason":"STOP"}]}}\n\n'
               b'event: error\ndata: {"error":{"code":429,"message":"SECRET_SENTINEL"}}\n\n')
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", return_value=io.BytesIO(raw)):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        body = h.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertNotIn("SECRET_SENTINEL", body)

    def test_upstream_error_before_first_event_uses_http_error(self):
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        statuses = []
        h.send_response = lambda code: statuses.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda *args: None
        def fail(_):
            raise proxy.UpstreamError("quota exceeded", status=429)
            yield
        with patch.object(proxy, "stream_generate_content", side_effect=fail):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        self.assertEqual(statuses, [429])
        self.assertNotIn("chat.completion.chunk", h.wfile.getvalue().decode())

    def test_upstream_error_after_first_event_does_not_report_stop(self):
        def events(_):
            yield {"response": {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}}
            raise proxy.UpstreamError("SECRET_SENTINEL", status=429)
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "stream_generate_content", side_effect=events):
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        body = h.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertNotIn("SECRET_SENTINEL", body)

    def test_eof_without_finish_reason_does_not_report_stop(self):
        chunks = collect_stream([{"response": {"candidates": [{"content": {"parts": [{"text": "incomplete"}]}}]}}])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")

    def test_prompt_block_is_not_reported_as_success(self):
        chunks = collect_stream([{"response": {"promptFeedback": {"blockReason": "SAFETY"}}}])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "content_filter")
        self.assertTrue(any("blocked" in str(chunk).lower() for chunk in chunks))

    def test_nonstream_preserves_prompt_feedback(self):
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": {"promptFeedback": {"blockReason": "SAFETY"}}}])):
            response = proxy.call_generate_content({"request": {}})
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash")
        self.assertEqual(result["choices"][0]["finish_reason"], "content_filter")

    def test_nonstream_incomplete_generation_raises(self):
        with patch.object(proxy, "stream_generate_content", return_value=iter([{"response": {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}}])):
            with self.assertRaises(proxy.UpstreamError):
                proxy.call_generate_content({"request": {}})

    def test_multiline_sse_event_is_combined(self):
        raw = b'data: {"response":\ndata: {"candidates":[{"finishReason":"STOP"}]}}\n\n'
        with patch.object(proxy, "get_access_token", return_value="dummy"), patch.object(proxy, "urlopen", return_value=io.BytesIO(raw)):
            events = list(proxy.stream_generate_content({"request": {}}))
        self.assertEqual(events[0]["response"]["candidates"][0]["finishReason"], "STOP")

    def test_max_tokens_is_not_masked_by_tool_call(self):
        response = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "terminal", "args": {"command": "pwd"}}}]}, "finishReason": "MAX_TOKENS"}]}
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})
        self.assertEqual(result["choices"][0]["finish_reason"], "length")

    def test_stream_retries_once_on_immediate_prohibited_content(self):
        call_1 = [{"response": {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}}]
        call_2 = [{"response": {"candidates": [{"content": {"parts": [{"text": "recovered"}]}, "finishReason": "STOP"}]}}]
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "stream_generate_content", side_effect=[iter(call_1), iter(call_2)]) as mock_stream:
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        self.assertEqual(mock_stream.call_count, 2)
        body = h.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        texts = [c["choices"][0]["delta"].get("content") for c in chunks if c.get("choices") and "content" in c["choices"][0].get("delta", {})]
        self.assertIn("recovered", "".join(t for t in texts if t))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_stream_surfaces_content_filter_when_prohibited_content_persists(self):
        call_1 = [{"response": {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}}]
        call_2 = [{"response": {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}}]
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda *args: None
        with patch.object(proxy, "stream_generate_content", side_effect=[iter(call_1), iter(call_2)]) as mock_stream:
            h._handle_stream({"request": {"tools": []}}, "gemini-3-flash")
        self.assertEqual(mock_stream.call_count, 2)
        body = h.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "content_filter")

    def test_nonstream_retries_once_on_prohibited_content(self):
        call_1 = [{"response": {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]}}]
        call_2 = [{"response": {"candidates": [{"content": {"parts": [{"text": "recovered"}]}, "finishReason": "STOP"}]}}]
        with patch.object(proxy, "stream_generate_content", side_effect=[iter(call_1), iter(call_2)]) as mock_stream:
            response = proxy.call_generate_content({"request": {}})
        self.assertEqual(mock_stream.call_count, 2)
        self.assertEqual(response["candidates"][0]["finishReason"], "STOP")
        self.assertEqual(response["candidates"][0]["content"]["parts"][0]["text"], "recovered")
        result = proxy.gemini_response_to_openai(response, "gemini-3-flash")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")

    def test_argon_stream_splits_seam_with_bold_header(self):
        events = [
            {"response": {"candidates": [{"content": {"parts": [
                {"text": "Размышляю над деталями запроса. Формирую финальный ответ.**Вечерний разбор задач**"}
            ]}, "finishReason": "STOP"}]}}
        ]
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda: None
        with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
            h._handle_stream({"request": {"tools": []}}, "gemini-4-argon-medium")
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {") and '"choices"' in line]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        reasonings = [d["reasoning_content"] for d in deltas if d.get("reasoning_content")]
        contents = [d["content"] for d in deltas if d.get("content")]
        self.assertEqual(reasonings, ["Размышляю над деталями запроса. Формирую финальный ответ."])
        self.assertEqual(contents, ["**Вечерний разбор задач**"])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_argon_stream_splits_seam_with_glued_capital_word(self):
        events = [
            {"response": {"candidates": [{"content": {"parts": [
                {"text": "Проверил параметры. Дам чёткий отчёт пользователю.Проверил через API все сервисы."}
            ]}, "finishReason": "STOP"}]}}
        ]
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda: None
        with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
            h._handle_stream({"request": {"tools": []}}, "gemini-4-argon-medium")
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {") and '"choices"' in line]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        reasonings = [d["reasoning_content"] for d in deltas if d.get("reasoning_content")]
        contents = [d["content"] for d in deltas if d.get("content")]
        self.assertEqual(reasonings, ["Проверил параметры. Дам чёткий отчёт пользователю."])
        self.assertEqual(contents, ["Проверил через API все сервисы."])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_argon_stream_splits_seam_across_chunks(self):
        events = [
            {"response": {"candidates": [{"content": {"parts": [
                {"text": "Всё проверено, можно отвечать пользователю."}
            ]}}]}},
            {"response": {"candidates": [{"content": {"parts": [
                {"text": "Крон **b395c5213912** работает штатно."}
            ]}, "finishReason": "STOP"}]}}
        ]
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda: None
        with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
            h._handle_stream({"request": {"tools": []}}, "gemini-4-argon-medium")
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {") and '"choices"' in line]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        reasonings = [d["reasoning_content"] for d in deltas if d.get("reasoning_content")]
        contents = [d["content"] for d in deltas if d.get("content")]
        self.assertEqual(reasonings, ["Всё проверено, можно отвечать пользователю."])
        self.assertEqual(contents, ["Крон **b395c5213912** работает штатно."])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_argon_stream_tool_calls_moves_unmarked_thought_to_reasoning(self):
        events = [
            {"response": {"candidates": [{"content": {"parts": [
                {"text": "Давай проверим статус контейнеров в системе."},
                {"functionCall": {"name": "terminal", "id": "fc1", "args": {"command": "docker ps"}}}
            ]}, "finishReason": "STOP"}]}}
        ]
        h = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        h.wfile = io.BytesIO()
        h.send_response = lambda code, message=None: None
        h.send_header = lambda keyword, value: None
        h.end_headers = lambda: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter(events)):
            h._handle_stream(envelope, "gemini-4-argon-medium")
        chunks = [json.loads(line[6:]) for line in h.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {") and '"choices"' in line]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        reasonings = [d["reasoning_content"] for d in deltas if d.get("reasoning_content")]
        contents = [d["content"] for d in deltas if d.get("content")]
        calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
        self.assertEqual(reasonings, ["Давай проверим статус контейнеров в системе."])
        self.assertEqual(contents, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "terminal")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("empty", h.wfile.getvalue().decode().lower())

    def test_argon_nonstream_splits_seam_and_tool_calls(self):
        # Non-stream with seam
        resp_seam = {
            "candidates": [{
                "content": {"parts": [{"text": "Сформулирую ясный, структурированный и исчерпывающий отчёт для Василия.Я провёл сквозной аудит."}]},
                "finishReason": "STOP"
            }]
        }
        res_seam = proxy.gemini_response_to_openai(resp_seam, "gemini-4-argon-medium")
        msg_seam = res_seam["choices"][0]["message"]
        self.assertEqual(msg_seam["reasoning_content"], "Сформулирую ясный, структурированный и исчерпывающий отчёт для Василия.")
        self.assertEqual(msg_seam["content"], "Я провёл сквозной аудит.")
        self.assertEqual(res_seam["choices"][0]["finish_reason"], "stop")

        # Non-stream with tool calls
        resp_tc = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "Let's check the database logs first."},
                    {"functionCall": {"name": "terminal", "args": {"command": "cat /var/log/syslog"}}}
                ]},
                "finishReason": "STOP"
            }]
        }
        res_tc = proxy.gemini_response_to_openai(resp_tc, "gemini-4-argon-medium", {"terminal"})
        msg_tc = res_tc["choices"][0]["message"]
        self.assertEqual(msg_tc["reasoning_content"], "Let's check the database logs first.")
        self.assertIsNone(msg_tc["content"])
        self.assertEqual(len(msg_tc["tool_calls"]), 1)
        self.assertEqual(msg_tc["tool_calls"][0]["function"]["name"], "terminal")
        self.assertEqual(res_tc["choices"][0]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
