"""Real Claude Code CLI through experimental Compact/cache Messages ingress.

The management and task models are local scripted services. This is not a
Gemini 3.8 Flash live-model pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from openai import OpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from relay.strategies.compact import CODEX_SUMMARY_PREFIX
from tests.harness_e2e_support import clean_env
from tests.test_codex_e2e import _serve

from experiments.compact_cache.checks import (
    RecordingCache, StableCompactUpstream, assert_contract,
)
from experiments.compact_cache.anthropic_adapter import create_anthropic_adapter, to_relay_request
from experiments.compact_cache.case import FIRST, SECOND


def _anthropic_sse(model: str, message: str, number: int) -> str:
    response = {"id": f"msg_local_{number}", "type": "message", "role": "assistant",
                "content": [], "model": model, "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 0}}
    events = [
        ("message_start", {"type": "message_start", "message": response}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": message}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def _anthropic_tool_sse(model: str) -> str:
    response = {"id": "msg_local_tool", "type": "message", "role": "assistant",
                "content": [], "model": model, "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 0}}
    events = [
        ("message_start", {"type": "message_start", "message": response}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "tool_use", "id": "toolu_local_1",
                                                   "name": "Bash", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "input_json_delta", "partial_json":
                                           json.dumps({"command": "printf RELAY_CLAUDE_TOOL_OK"})}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def run_claude_contract():
    binary = shutil.which("claude")
    if not binary:
        pytest.skip("Claude Code CLI is not installed")

    manager = StableCompactUpstream()
    task_requests = []

    async def task(request: Request):
        body = await request.json()
        task_requests.append(body)
        number = len(task_requests)
        text = f"RELAY_CLAUDE_TURN_{number}_OK"
        if body.get("stream"):
            return Response(_anthropic_sse(body["model"], text, number),
                            media_type="text/event-stream")
        return JSONResponse({"id": f"msg_local_{number}", "type": "message",
                             "role": "assistant", "content": [{"type": "text", "text": text}],
                             "model": body["model"], "stop_reason": "end_turn",
                             "usage": {"input_tokens": 20, "output_tokens": 5}})

    task_app = Starlette(routes=[Route("/v1/messages", task, methods=["POST"])])
    cache = RecordingCache()
    with _serve(manager.app) as manager_url, _serve(task_app) as task_url:
        with OpenAI(base_url=f"{manager_url}/v1", api_key="local-manager") as client:
            adapter = create_anthropic_adapter(
                task_base_url=task_url, management_responses=client.responses,
                manager_model="gemini-3.8-flash", cache=cache, compact_threshold=100,
            )
            with _serve(adapter) as relay_url, tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = clean_env(root)
                env.update({
                    "ANTHROPIC_BASE_URL": relay_url,
                    "ANTHROPIC_API_KEY": "local-test-key",
                    "ANTHROPIC_CUSTOM_MODEL_OPTION": "gemini-3.8-flash",
                    "CLAUDE_CONFIG_DIR": str(root / "claude-config"),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_MAX_RETRIES": "0",
                    "DISABLE_COMPACT": "1",
                    "DISABLE_TELEMETRY": "1",
                })
                base = [binary, "--print", "--output-format", "json", "--model",
                        "gemini-3.8-flash", "--tools", "", "--setting-sources", ""]
                first = subprocess.run(
                    [*base, FIRST],
                    cwd=root, env=env, text=True, capture_output=True,
                    stdin=subprocess.DEVNULL, timeout=45, check=False,
                )
                assert first.returncode == 0, first.stdout + first.stderr
                assert "RELAY_CLAUDE_TURN_1_OK" in first.stdout
                second = subprocess.run(
                    [*base, "--continue", SECOND],
                    cwd=root, env=env, text=True, capture_output=True,
                    stdin=subprocess.DEVNULL, timeout=45, check=False,
                )
                assert second.returncode == 0, second.stdout + second.stderr
                assert "RELAY_CLAUDE_TURN_2_OK" in second.stdout

    assert len(task_requests) == 2
    assert len(manager.summary_requests) == 1
    assert all(body["model"] == "gemini-3.8-flash" for body in task_requests)
    assert manager.summary_requests[0]["model"] == "gemini-3.8-flash"
    assert cache.stats().hits >= 1
    forwarded = str(task_requests[1]["messages"])
    assert CODEX_SUMMARY_PREFIX in forwarded
    assert "Turn 2" in forwarded
    return (
        cache, manager.summary_requests,
        [lookup.trajectory for lookup in cache.lookups],
        [to_relay_request(body, manager_model="gemini-3.8-flash")["input"]
         for body in task_requests],
        "Turn 2",
    )
