"""Verify the live Claude task bridge's supported text protocol locally."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

from starlette.testclient import TestClient

from experiments.compact_cache.openai_messages_bridge import create_bridge
from experiments.compact_cache.openai_gemini_bridge import create_bridge as create_gemini_bridge
from relay.gemini import GeminiInput
from experiments.compact_cache.live_local_files import _stage


def test_claude_messages_bridge_streams_text_from_responses() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="response-1", output_text="ACK", usage=SimpleNamespace(
                input_tokens=12, output_tokens=2))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    app = create_bridge(client, "gpt-6-luna")
    with TestClient(app) as test_client:
        response = test_client.post("/v1/messages", json={
            "model": "gpt-6-luna", "stream": True,
            "messages": [{"role": "user", "content": "Reply ACK"}],
        })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in response.text
    assert '"text": "ACK"' in response.text
    assert "event: message_stop" in response.text
    assert calls[0]["model"] == "gpt-6-luna"
    assert calls[0]["input"] == [{"type": "message", "role": "user",
                                   "content": "Reply ACK"}]


def test_gemini_bridge_streams_text_from_responses() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="response-2", output_text="ACK", usage=SimpleNamespace(
                input_tokens=12, output_tokens=2))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    seen = []
    app = create_gemini_bridge(client, "gpt-6-luna", seen)
    with TestClient(app) as test_client:
        response = test_client.post("/v1beta/models/gemini-3.8-flash:streamGenerateContent",
                                    json={"contents": [{"role": "user",
                                                        "parts": [{"text": "Reply ACK"}]}]})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"text": "ACK"' in response.text
    assert len(seen) == 1
    assert calls[0]["model"] == "gpt-6-luna"
    assert calls[0]["input"] == [{"type": "message", "role": "user",
                                   "content": "Reply ACK"}]


def test_gemini_bridge_returns_function_call() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output=[SimpleNamespace(
            type="function_call", name="read_file", call_id="call_1",
            arguments='{"file_path":"config/production.json"}')],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    app = create_gemini_bridge(client, "gpt-6-luna", [])
    with TestClient(app) as test_client:
        response = test_client.post("/v1beta/models/gemini:generateContent", json={
            "contents": [{"role": "user", "parts": [{"text": "Read config"}]}],
            "tools": [{"functionDeclarations": [{"name": "read_file",
                       "parameters": {"type": "OBJECT", "properties": {
                           "file_path": {"type": "STRING"}}}}]}],
        })
    assert response.status_code == 200, response.text
    assert response.json()["candidates"][0]["content"]["parts"] == [{
        "functionCall": {"name": "read_file", "args": {
            "file_path": "config/production.json"}, "id": "call_1"}}]
    assert calls[0]["tools"][0]["parameters"]["type"] == "object"


def test_claude_bridge_streams_tool_use() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="response-3", output=[SimpleNamespace(
            type="function_call", name="Bash", call_id="call_2",
            arguments='{"command":"cat config/production.json"}')],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    app = create_bridge(client, "gpt-6-luna")
    with TestClient(app) as test_client:
        response = test_client.post("/v1/messages", json={
            "model": "gpt-6-luna", "stream": True,
            "messages": [{"role": "user", "content": "Read config"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object",
                       "properties": {"command": {"type": "string"}}}}],
        })
    assert response.status_code == 200, response.text
    assert '"type": "tool_use"' in response.text
    assert '"stop_reason": "tool_use"' in response.text
    assert calls[0]["tools"][0]["name"] == "Bash"


def test_gemini_input_deduplicates_identical_tool_result() -> None:
    call = {"name": "read_file", "id": "call_1", "args": {"file_path": "a"}}
    result = {"name": "read_file", "id": "call_1", "response": {"content": "x"}}
    body = {"contents": [
        {"role": "model", "parts": [{"functionCall": call}]},
        {"role": "user", "parts": [{"functionResponse": result},
                                  {"functionResponse": result}]},
    ]}
    normalized = GeminiInput(body, "gpt-6-luna").request["input"]
    assert [item["type"] for item in normalized] == [
        "function_call", "function_call_output"]


def test_claude_bridge_omits_empty_read_pages() -> None:
    def create(**kwargs):
        return SimpleNamespace(id="response-4", output=[SimpleNamespace(
            type="function_call", name="Read", call_id="call_3",
            arguments='{"file_path":"/tmp/config/production.json","pages":""}')],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with TestClient(create_bridge(client, "gpt-6-luna")) as test_client:
        response = test_client.post("/v1/messages", json={
            "model": "gpt-6-luna", "messages": [
                {"role": "user", "content": "Read config"}],
        })
    assert response.status_code == 200
    assert response.json()["content"][0]["input"] == {
        "file_path": "/tmp/config/production.json"}


def test_gemini_cli_answer_is_taken_from_response_field() -> None:
    result = subprocess.CompletedProcess(
        [], 0, '{"response":"750 × 3 = 2,250 毫秒","stats":{"tokens":2250}}', "")
    assert _stage(result, "gemini-cli")["answer_text"] == "750 × 3 = 2,250 毫秒"
