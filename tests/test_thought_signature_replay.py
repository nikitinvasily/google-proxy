import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class ThoughtSignatureReplayTests(unittest.TestCase):
    def test_no_id_parallel_calls_in_separate_events_remain_distinct(self):
        first = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "args": {"command": "true"}},
            "thoughtSignature": "SYNTHETIC_FIRST_SIGNATURE",
        }]}}]}}
        second = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "args": {"command": "pwd"}},
        }]}, "finishReason": "STOP"}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([first, second])):
            response = proxy.call_generate_content({"request": {}})
        calls = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})[
            "choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual([json.loads(c["function"]["arguments"])["command"] for c in calls],
                         ["true", "pwd"])

    def test_parallel_no_id_calls_in_one_event_remain_distinct(self):
        event = {"response": {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "terminal", "args": {"command": "true"}},
             "thoughtSignature": "SYNTHETIC_PARALLEL_SIGNATURE"},
            {"functionCall": {"name": "terminal", "args": {"command": "pwd"}}}]},
            "finishReason": "STOP"}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([event])):
            response = proxy.call_generate_content({"request": {}})
        calls = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})[
            "choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual([json.loads(c["function"]["arguments"])["command"] for c in calls],
                         ["true", "pwd"])

    def test_stream_no_id_snapshot_updates_existing_call(self):
        first = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "args": {"command": "tr"}},
        }]}}]}}
        second = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "args": {"command": "true"}},
            "thoughtSignature": "SYNTHETIC_NO_ID_SIGNATURE",
        }]}, "finishReason": "STOP"}]}}
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([first, second])):
            handler._handle_stream(envelope, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {")]
        calls = [call for chunk in chunks for choice in chunk.get("choices", [])
                 for call in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "true"})
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"],
                         "SYNTHETIC_NO_ID_SIGNATURE")

    def test_nonstream_no_id_snapshot_updates_existing_call(self):
        first = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "args": {"command": "tr"}},
        }]}}]}}
        second = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "args": {"command": "true"}},
            "thoughtSignature": "SYNTHETIC_NO_ID_SIGNATURE",
        }]}, "finishReason": "STOP"}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([first, second])):
            response = proxy.call_generate_content({"request": {}})
        choice = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})["choices"][0]
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "true"})
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"],
                         "SYNTHETIC_NO_ID_SIGNATURE")

    def test_parallel_unsigned_second_call_is_replayed_without_sentinel(self):
        calls = [
            {"id": "call_first", "type": "function",
             "function": {"name": "ping", "arguments": "{}"},
             "extra_content": {"google": {"thought_signature": "SYNTHETIC_FIRST_SIGNATURE"}}},
            {"id": "call_second", "type": "function",
             "function": {"name": "ping", "arguments": "{}"}},
        ]
        messages = [{"role": "user", "content": "run parallel checks"},
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "call_first", "content": "{}"},
                    {"role": "tool", "tool_call_id": "call_second", "content": "{}"}]
        with patch.object(proxy, "get_project_id", return_value="synthetic-project"):
            upstream = proxy._build_gemini_request({"messages": messages}, "gemini-3-flash")
        parts = upstream["request"]["contents"][1]["parts"]
        self.assertEqual(parts[0]["thoughtSignature"], "SYNTHETIC_FIRST_SIGNATURE")
        self.assertNotIn("thoughtSignature", parts[1])

    def test_nonstream_repeated_function_snapshot_yields_one_final_call(self):
        first = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc-duplicate", "args": {"command": "tr"}},
        }]}}]}}
        second = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc-duplicate", "args": {"command": "true"}},
            "thoughtSignature": "SYNTHETIC_FINAL_SIGNATURE",
        }]}, "finishReason": "STOP"}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([first, second])):
            response = proxy.call_generate_content({"request": {}})
        choice = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})["choices"][0]
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "true"})
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"],
                         "SYNTHETIC_FINAL_SIGNATURE")

    def test_stream_later_signature_only_snapshot_retains_previous_arguments(self):
        first = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc-args", "args": {"command": "true"}},
        }]}}]}}
        second = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc-args"},
            "thoughtSignature": "SYNTHETIC_LATE_SIGNATURE",
        }]}, "finishReason": "STOP"}]}}
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([first, second])):
            handler._handle_stream(envelope, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {")]
        calls = [call for chunk in chunks for choice in chunk.get("choices", [])
                 for call in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "true"})

    def test_stream_repeated_call_keeps_signature_from_later_snapshot(self):
        first = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc-late", "args": {"command": "true"}},
        }]}}]}}
        second = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc-late", "args": {"command": "true"}},
            "thoughtSignature": "SYNTHETIC_LATE_SIGNATURE",
        }]}, "finishReason": "STOP"}]}}
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([first, second])):
            handler._handle_stream(envelope, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {")]
        calls = [call for chunk in chunks for choice in chunk.get("choices", [])
                 for call in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["extra_content"]["google"]["thought_signature"],
                         "SYNTHETIC_LATE_SIGNATURE")

    def test_nonstream_signature_uses_client_replay_sidecar(self):
        signature = "SYNTHETIC_NON_SECRET_SIGNATURE"
        response = {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc456", "args": {"command": "true"}},
            "thoughtSignature": signature,
        }]}, "finishReason": "STOP"}]}
        choice = proxy.gemini_response_to_openai(response, "gemini-3-flash", {"terminal"})["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["id"], "call_fc456|" + signature)  # legacy clients still decode the ID
        self.assertEqual(call["extra_content"], {"google": {"thought_signature": signature}})
        canonical_id = call["id"].split("|", 1)[0]  # Hermes normalizes the ID before replay

        messages = [
            {"role": "user", "content": "run a harmless check"},
            {"role": "assistant", "content": None, "tool_calls": [{**call, "id": canonical_id}]},
            {"role": "tool", "tool_call_id": canonical_id, "content": '{"exit_code": 0}'},
        ]
        with patch.object(proxy, "get_project_id", return_value="synthetic-project"):
            upstream = proxy._build_gemini_request({"messages": messages}, "gemini-3-flash")
        self.assertEqual(upstream["request"]["contents"][1]["parts"][-1]["thoughtSignature"], signature)

    def test_stream_signature_survives_canonical_tool_id_and_replay(self):
        signature = "SYNTHETIC_NON_SECRET_SIGNATURE"
        event = {"response": {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "terminal", "id": "fc123", "args": {"command": "true"}},
            "thoughtSignature": signature,
        }]}, "finishReason": "STOP"}]}}
        handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        envelope = {"request": {"tools": [{"functionDeclarations": [{"name": "terminal"}]}]}}
        with patch.object(proxy, "stream_generate_content", return_value=iter([event])):
            handler._handle_stream(envelope, "gemini-3-flash")
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                  if line.startswith("data: {")]
        tool_calls = [call for chunk in chunks for choice in chunk.get("choices", [])
                      for call in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(tool_calls), 1)
        call = tool_calls[0]
        self.assertEqual(call["id"], "call_fc123|" + signature)  # keep the legacy wire contract
        self.assertEqual(call["extra_content"], {"google": {"thought_signature": signature}})
        canonical_id = call["id"].split("|", 1)[0]  # Hermes stores only this prefix

        messages = [
            {"role": "user", "content": "run a harmless check"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": canonical_id, "type": call["type"],
                "function": call["function"], "extra_content": call["extra_content"],
            }]},
            {"role": "tool", "tool_call_id": canonical_id, "content": '{"exit_code": 0}'},
        ]
        with patch.object(proxy, "get_project_id", return_value="synthetic-project"):
            upstream = proxy._build_gemini_request({"messages": messages}, "gemini-3-flash")
        contents = upstream["request"]["contents"]
        self.assertEqual(contents[1]["parts"][-1]["thoughtSignature"], signature)
        self.assertEqual(contents[1]["parts"][-1]["functionCall"]["id"], "fc123")
        self.assertEqual(contents[2]["parts"][0]["functionResponse"]["id"], "fc123")


if __name__ == "__main__":
    unittest.main()
