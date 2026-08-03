from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from relay import (
    RLM,
    AgentFold,
    AutoCompact,
    Checkpoint,
    Compact,
    ContextFolding,
    PrefixCheckpointCache,
    RollingMemory,
    SlidingWindow,
)
from relay.proxy import ProxyConfig, create_app
from relay.strategies.auto_compact import AUTO_CONTEXT_SUMMARY
from relay.strategies.compact import CODEX_COMPACTION_PROMPT, CODEX_SUMMARY_PREFIX
from relay.strategies.context_folding import CONTEXT_FOLDING_RETURN_PREFIX
from relay.strategies.rlm import RLM_HANDOFF_PREFIX
from relay.strategies.rolling_memory import (
    ROLLING_MEMORY_PREFIX,
    ROLLING_MEMORY_PROMPT,
)


def _text(item: dict[str, Any]) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _is_summary_request(body: dict[str, Any]) -> bool:
    input_items = body.get("input") or []
    return bool(
        isinstance(input_items, list)
        and input_items
        and isinstance(input_items[-1], dict)
        and _text(input_items[-1])
        in {CODEX_COMPACTION_PROMPT, ROLLING_MEMORY_PROMPT}
    )


def _response(body: dict[str, Any], text: str, response_id: str) -> dict[str, Any]:
    item = {
        "id": f"msg_{response_id}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "model": body["model"],
        "output": [item],
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "temperature": None,
        "top_p": None,
        "truncation": body.get("truncation", "disabled"),
        "usage": {
            "input_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 35,
        },
    }


def _reasoning_item(response_id: str) -> dict[str, Any]:
    return {
        "id": f"rs_{response_id}",
        "type": "reasoning",
        "summary": [],
        "content": None,
        "encrypted_content": "cmVsYXktY29kZXgtZTJlLXJlYXNvbmluZw==",
        "status": "completed",
    }


def _tool_response(body: dict[str, Any], response_id: str) -> dict[str, Any]:
    response = _response(body, "", response_id)
    response["output"] = [
        _reasoning_item(response_id),
        {
            "id": f"fc_{response_id}",
            "type": "function_call",
            "call_id": f"call_{response_id}",
            "name": "exec_command",
            "arguments": json.dumps(
                {
                    "cmd": "pwd",
                    "yield_time_ms": 1000,
                    "max_output_tokens": 1000,
                },
                separators=(",", ":"),
            ),
            "status": "completed",
        },
    ]
    return response


def _sse(body: dict[str, Any], text: str, response_id: str) -> bytes:
    completed = _response(body, text, response_id)
    item = completed["output"][0]
    empty_item = {**item, "content": []}
    part = item["content"][0]
    in_progress = {**completed, "status": "in_progress", "output": [], "usage": None}
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": in_progress,
        },
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": in_progress,
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": empty_item,
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "part": {**part, "text": ""},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 4,
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 5,
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "text": text,
            "logprobs": [],
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 6,
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "part": part,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.completed",
            "sequence_number": 8,
            "response": completed,
        },
    ]
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in events
    )


def _tool_sse(body: dict[str, Any], response_id: str) -> bytes:
    completed = _tool_response(body, response_id)
    reasoning, call = completed["output"]
    in_progress = {**completed, "status": "in_progress", "output": [], "usage": None}
    added_call = {**call, "arguments": "", "status": "in_progress"}
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": in_progress,
        },
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": in_progress,
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": reasoning,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 0,
            "item": reasoning,
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 1,
            "item": added_call,
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 5,
            "output_index": 1,
            "item_id": call["id"],
            "delta": call["arguments"],
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 6,
            "output_index": 1,
            "item_id": call["id"],
            "name": call["name"],
            "arguments": call["arguments"],
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 1,
            "item": call,
        },
        {
            "type": "response.completed",
            "sequence_number": 8,
            "response": completed,
        },
    ]
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in events
    )


class _FakeResponsesUpstream:
    def __init__(self, *, tool_first: bool = False, mini_mode: bool = False) -> None:
        self.tool_first = tool_first
        self.mini_mode = mini_mode
        self.lock = threading.Lock()
        self.main_requests: list[dict[str, Any]] = []
        self.summary_requests: list[dict[str, Any]] = []
        self.count_requests: list[dict[str, Any]] = []
        self.rlm_requests: list[dict[str, Any]] = []
        self.manager_requests: list[dict[str, Any]] = []
        self.app = Starlette(
            routes=[Route("/{path:path}", self.dispatch, methods=["POST"])]
        )

    async def dispatch(self, request: Request) -> Response:
        body = await request.json()
        if request.url.path == "/v1/chat/completions":
            with self.lock:
                self.rlm_requests.append(body)
                number = len(self.rlm_requests)
            if number % 2:
                content = (
                    "```repl\n"
                    "print(len(context['input']), sorted(context.keys()))\n"
                    "```"
                )
            else:
                content = (
                    "```repl\n"
                    "answer['content'] = "
                    "'Return the requested result without using tools.'\n"
                    "answer['ready'] = True\n"
                    "```"
                )
            return JSONResponse(
                {
                    "id": f"chat_rlm_{number}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body.get("model", "relay-rlm-test-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                }
            )
        if request.url.path == "/v1/responses/input_tokens":
            with self.lock:
                self.count_requests.append(body)
            # Normal items cross the deliberately small test thresholds, while
            # previously generated summaries are cheap enough to prove eviction.
            count = 10 if _is_summary_request(body) else sum(
                5
                if isinstance(item, dict)
                and _text(item).startswith(
                    (CODEX_SUMMARY_PREFIX, ROLLING_MEMORY_PREFIX)
                )
                else 30
                for item in body.get("input") or []
            )
            return JSONResponse(
                {"object": "response.input_tokens", "input_tokens": count}
            )
        if request.url.path != "/v1/responses":
            return JSONResponse({"error": "not found"}, status_code=404)

        schema_name = body.get("text", {}).get("format", {}).get("name")
        if isinstance(schema_name, str) and schema_name.startswith("relay_"):
            with self.lock:
                self.manager_requests.append(body)
                same_kind = [
                    value
                    for value in self.manager_requests
                    if value.get("text", {}).get("format", {}).get("name")
                    == schema_name
                ]
                number = len(same_kind)
            if schema_name == "relay_context_folding_decision":
                value = (
                    {"action": "open", "objective": "resume subtask", "summary": ""}
                    if number == 1
                    else {
                        "action": "return",
                        "objective": "",
                        "summary": "hidden branch completed",
                    }
                )
            elif schema_name == "relay_agent_fold_directive":
                value = {
                    "compress_range": [1, number],
                    "compress_text": f"agent fold state {number}",
                }
            elif schema_name == "relay_auto_compact_decision":
                value = (
                    {
                        "action": "compact",
                        "summary": "auto compact working state",
                        "reason": "phase boundary",
                    }
                    if number == 1
                    else {"action": "keep", "summary": "", "reason": "continue"}
                )
            else:
                return JSONResponse({"error": "unknown manager schema"}, status_code=400)
            return JSONResponse(
                _response(body, json.dumps(value), f"resp_manager_{len(self.manager_requests)}")
            )

        if _is_summary_request(body):
            with self.lock:
                self.summary_requests.append(body)
                number = len(self.summary_requests)
            return JSONResponse(
                _response(
                    body,
                    f"relay checkpoint summary {number}",
                    f"resp_sum_{number}",
                )
            )

        with self.lock:
            self.main_requests.append(body)
            number = len(self.main_requests)
        if body.get("stream") is not True:
            if self.mini_mode:
                command = (
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo relay-mini-ok"
                    if number == 3
                    else "pwd"
                )
                response = _tool_response(body, f"resp_mini_{number}")
                response["output"][1]["name"] = "bash"
                response["output"][1]["arguments"] = json.dumps(
                    {"command": command}, separators=(",", ":")
                )
                return JSONResponse(response)
            return JSONResponse({"error": "Codex did not request SSE"}, status_code=400)
        if self.tool_first and number == 1:
            return Response(
                _tool_sse(body, "resp_tool_1"),
                media_type="text/event-stream",
            )
        marker = f"RELAY_CODEX_TURN_{number}_OK"
        return Response(
            _sse(body, marker, f"resp_turn_{number}"),
            media_type="text/event-stream",
        )


@contextmanager
def _serve(app: Starlette) -> Iterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="on", access_log=False)
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("test HTTP server failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


@contextmanager
def _serve_relay_command(upstream_url: str) -> Iterator[str]:
    command = Path(sys.executable).with_name("relay")
    if not command.is_file():
        raise unittest.SkipTest("the Relay console script is not installed")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    env = dict(os.environ)
    env.update(
        {
            "RELAY_UPSTREAM_BASE_URL": f"{upstream_url}/v1",
            "RELAY_UPSTREAM_API_KEY": "upstream-test-key",
            "RELAY_STRATEGY": "compact",
            "RELAY_COMPACT_THRESHOLD": "20",
            "RELAY_CHECKPOINT_MODE": "cache",
            "RELAY_CACHE_SECRET": "codex-command-e2e-test",
            "RELAY_HOST": "127.0.0.1",
            "RELAY_PORT": str(port),
        }
    )
    process = subprocess.Popen(
        [str(command)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.1)
        except OSError:
            time.sleep(0.02)
        else:
            connection.close()
            break
    else:
        stdout, stderr = process.communicate(timeout=5)
        raise RuntimeError(f"Relay command failed to start: {stdout}\n{stderr}")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def _run_codex(
    codex: str,
    codex_home: Path,
    cwd: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "CODEX_HOME": str(codex_home),
            "RELAY_TEST_API_KEY": "tenant-a",
            "NO_COLOR": "1",
        }
    )
    return subprocess.run(
        [codex, *arguments],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _write_codex_config(codex_home: Path, relay_url: str) -> None:
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                'model = "relay-test-model"',
                'model_provider = "relay"',
                'model_auto_compact_token_limit = 1000000000',
                "",
                "[features]",
                "plugins = false",
                "",
                "[model_providers.relay]",
                'name = "Relay"',
                f'base_url = "{relay_url}/v1"',
                'env_key = "RELAY_TEST_API_KEY"',
                'wire_api = "responses"',
                "supports_websockets = false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _resume_case(
    codex: str, strategy: Any, cache_secret: bytes
) -> tuple[
    _FakeResponsesUpstream,
    PrefixCheckpointCache,
    subprocess.CompletedProcess[str],
    subprocess.CompletedProcess[str],
]:
    upstream = _FakeResponsesUpstream()
    cache = PrefixCheckpointCache(secret=cache_secret)
    with _serve(upstream.app) as upstream_url:
        relay = create_app(
            strategy,
            ProxyConfig(
                upstream_base_url=f"{upstream_url}/v1",
                upstream_api_key="upstream-test-key",
                checkpoint_mode="cache",
            ),
            checkpoint_cache=cache,
        )
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            codex_home = root_path / "codex-home"
            workspace = root_path / "workspace"
            codex_home.mkdir()
            workspace.mkdir()
            _write_codex_config(codex_home, relay_url)
            first = _run_codex(
                codex,
                codex_home,
                workspace,
                [
                    "exec",
                    "--json",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--cd",
                    str(workspace),
                    "Reply with the model result and do not use tools.",
                ],
            )
            second = _run_codex(
                codex,
                codex_home,
                workspace,
                [
                    "exec",
                    "resume",
                    "--last",
                    "--json",
                    "--skip-git-repo-check",
                    "Continue and return the new model result.",
                ],
            )
    return upstream, cache, first, second


def _adaptive_resume_case(
    codex: str, strategy: Any, cache_secret: bytes
) -> tuple[
    _FakeResponsesUpstream,
    PrefixCheckpointCache,
    list[subprocess.CompletedProcess[str]],
]:
    upstream = _FakeResponsesUpstream()
    cache = PrefixCheckpointCache(secret=cache_secret)
    with _serve(upstream.app) as upstream_url:
        relay = create_app(
            strategy,
            ProxyConfig(
                upstream_base_url=f"{upstream_url}/v1",
                upstream_api_key="upstream-test-key",
                checkpoint_mode="cache",
            ),
            checkpoint_cache=cache,
        )
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            codex_home = root_path / "codex-home"
            workspace = root_path / "workspace"
            codex_home.mkdir()
            workspace.mkdir()
            _write_codex_config(codex_home, relay_url)
            results = [
                _run_codex(
                    codex,
                    codex_home,
                    workspace,
                    [
                        "exec",
                        "--json",
                        "--skip-git-repo-check",
                        "--sandbox",
                        "read-only",
                        "--cd",
                        str(workspace),
                        "Reply with the model result and do not use tools.",
                    ],
                )
            ]
            for prompt in ("Continue once.", "Continue twice."):
                results.append(
                    _run_codex(
                        codex,
                        codex_home,
                        workspace,
                        [
                            "exec",
                            "resume",
                            "--last",
                            "--json",
                            "--skip-git-repo-check",
                            prompt,
                        ],
                    )
                )
    return upstream, cache, results


@unittest.skipUnless(shutil.which("codex"), "Codex CLI is not installed")
class CodexEndToEndTests(unittest.TestCase):
    def _assert_adaptive_strategy(
        self,
        strategy: Any,
        secret: bytes,
        schema_name: str,
        expected_third_input: str,
        min_cache_hits: int = 2,
    ) -> None:
        codex = shutil.which("codex")
        assert codex is not None
        upstream, cache, results = _adaptive_resume_case(codex, strategy, secret)
        for index, result in enumerate(results, start=1):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"RELAY_CODEX_TURN_{index}_OK", result.stdout)
        self.assertEqual(len(upstream.main_requests), 3)
        self.assertEqual(len(upstream.manager_requests), 2)
        self.assertEqual(
            {
                body["text"]["format"]["name"]
                for body in upstream.manager_requests
            },
            {schema_name},
        )
        self.assertGreaterEqual(cache.stats().hits, min_cache_hits)
        self.assertIn(expected_third_input, str(upstream.main_requests[2]["input"]))
        for body in upstream.main_requests:
            self.assertTrue(body["stream"])
            self.assertFalse(
                any(item.get("type") == "compaction" for item in body["input"])
            )
            self.assertFalse(
                any(item.get("name") in {"branch", "return", "compact"} for item in body["input"])
            )

    def _assert_resume_compatibility(
        self,
        strategy: Any,
        secret: bytes,
        *,
        first_is_compacted: bool,
        context_prefix: str = CODEX_SUMMARY_PREFIX,
    ) -> None:
        codex = shutil.which("codex")
        assert codex is not None
        upstream, cache, first, second = _resume_case(codex, strategy, secret)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("RELAY_CODEX_TURN_1_OK", first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("RELAY_CODEX_TURN_2_OK", second.stdout)
        self.assertEqual(len(upstream.main_requests), 2)
        self.assertGreaterEqual(len(upstream.summary_requests), 2)
        self.assertGreaterEqual(cache.stats().hits, 1)
        for body in upstream.main_requests:
            self.assertTrue(body["stream"])
            self.assertNotIn("previous_response_id", body)
            self.assertNotIn("conversation", body)
            self.assertFalse(
                any(
                    isinstance(item, dict) and item.get("type") == "compaction"
                    for item in body["input"]
                )
            )
        has_summary = [
            any(
                isinstance(item, dict)
                and _text(item).startswith(context_prefix)
                for item in body["input"]
            )
            for body in upstream.main_requests
        ]
        self.assertEqual(has_summary, [first_is_compacted, True])

    def test_compact_streams_and_resumes_from_cached_checkpoint(self) -> None:
        self._assert_resume_compatibility(
            Compact(compact_threshold=20),
            b"codex-compact-e2e-test",
            first_is_compacted=True,
        )

    def test_checkpoint_streams_and_resumes_from_cached_checkpoint(self) -> None:
        self._assert_resume_compatibility(
            Checkpoint(checkpoint_threshold=25, context_threshold=110),
            b"codex-checkpoint-e2e-test",
            first_is_compacted=False,
        )

    def test_rolling_memory_streams_and_resumes_from_cached_state(self) -> None:
        self._assert_resume_compatibility(
            RollingMemory(update_input_tokens=110),
            b"codex-rolling-memory-e2e-test",
            first_is_compacted=True,
            context_prefix=ROLLING_MEMORY_PREFIX,
        )

    def test_context_folding_runs_through_real_codex_resumes(self) -> None:
        self._assert_adaptive_strategy(
            ContextFolding(),
            b"codex-context-folding-e2e-test",
            "relay_context_folding_decision",
            CONTEXT_FOLDING_RETURN_PREFIX,
        )

    def test_agent_fold_runs_through_real_codex_resumes(self) -> None:
        self._assert_adaptive_strategy(
            AgentFold(),
            b"codex-agent-fold-e2e-test",
            "relay_agent_fold_directive",
            "Multi-Scale State Summaries",
        )

    def test_auto_compact_runs_through_real_codex_resumes(self) -> None:
        self._assert_adaptive_strategy(
            AutoCompact(fallback_threshold=1_000_000),
            b"codex-auto-compact-e2e-test",
            "relay_auto_compact_decision",
            AUTO_CONTEXT_SUMMARY,
            min_cache_hits=1,
        )

    @unittest.skipUnless(find_spec("rlm"), "official RLM package is not installed")
    def test_rlm_streams_and_resumes_with_a_fresh_official_query(self) -> None:
        codex = shutil.which("codex")
        assert codex is not None
        upstream, cache, first, second = _resume_case(
            codex,
            RLM(manager_model="relay-rlm-test-model", max_iterations=3),
            b"codex-rlm-e2e-test",
        )

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("RELAY_CODEX_TURN_1_OK", first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("RELAY_CODEX_TURN_2_OK", second.stdout)
        self.assertEqual(len(upstream.rlm_requests), 4)
        self.assertEqual(
            [len(body["messages"]) for body in upstream.rlm_requests],
            [3, 6, 3, 6],
        )
        self.assertEqual(len(upstream.main_requests), 2)
        self.assertEqual(upstream.summary_requests, [])
        self.assertEqual(cache.stats().entries, 0)
        self.assertEqual(cache.stats().hits, 0)
        for body in upstream.main_requests:
            self.assertTrue(body["stream"])
            self.assertTrue(
                _text(body["input"][-1]).startswith(RLM_HANDOFF_PREFIX)
            )
            self.assertFalse(
                any(item.get("type") == "compaction" for item in body["input"])
            )

    def test_readme_console_command_and_codex_provider_config(self) -> None:
        codex = shutil.which("codex")
        assert codex is not None
        upstream = _FakeResponsesUpstream()
        with _serve(upstream.app) as upstream_url:
            with _serve_relay_command(upstream_url) as relay_url:
                with tempfile.TemporaryDirectory() as root:
                    root_path = Path(root)
                    codex_home = root_path / "codex-home"
                    workspace = root_path / "workspace"
                    codex_home.mkdir()
                    workspace.mkdir()
                    _write_codex_config(codex_home, relay_url)
                    result = _run_codex(
                        codex,
                        codex_home,
                        workspace,
                        [
                            "exec",
                            "--json",
                            "--skip-git-repo-check",
                            "--sandbox",
                            "read-only",
                            "--cd",
                            str(workspace),
                            "Return the model result without using tools.",
                        ],
                    )

        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertIn("RELAY_CODEX_TURN_1_OK", result.stdout)
        self.assertEqual(len(upstream.main_requests), 1)
        self.assertTrue(upstream.main_requests[0]["stream"])

    def test_sliding_window_trims_a_real_codex_request_transparently(self) -> None:
        codex = shutil.which("codex")
        assert codex is not None
        upstream = _FakeResponsesUpstream()
        with _serve(upstream.app) as upstream_url:
            relay = create_app(
                SlidingWindow(max_input_tokens=70),
                ProxyConfig(
                    upstream_base_url=f"{upstream_url}/v1",
                    upstream_api_key="upstream-test-key",
                    checkpoint_mode="cache",
                ),
                checkpoint_cache=PrefixCheckpointCache(secret=b"codex-window-e2e-test"),
            )
            with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as root:
                root_path = Path(root)
                codex_home = root_path / "codex-home"
                workspace = root_path / "workspace"
                codex_home.mkdir()
                workspace.mkdir()
                _write_codex_config(codex_home, relay_url)
                result = _run_codex(
                    codex,
                    codex_home,
                    workspace,
                    [
                        "exec",
                        "--json",
                        "--skip-git-repo-check",
                        "--sandbox",
                        "read-only",
                        "--cd",
                        str(workspace),
                        "Return the model result without using tools.",
                    ],
                )

        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertIn("RELAY_CODEX_TURN_1_OK", result.stdout)
        self.assertEqual(upstream.summary_requests, [])
        full_input = upstream.count_requests[0]["input"]
        forwarded = upstream.main_requests[0]["input"]
        self.assertLess(len(forwarded), len(full_input))
        self.assertIn("Return the model result", _text(forwarded[-1]))

    def test_codex_tool_loop_preserves_encrypted_reasoning_and_call_transaction(
        self,
    ) -> None:
        codex = shutil.which("codex")
        assert codex is not None
        upstream = _FakeResponsesUpstream(tool_first=True)

        with _serve(upstream.app) as upstream_url:
            relay = create_app(
                SlidingWindow(max_input_tokens=1000),
                ProxyConfig(
                    upstream_base_url=f"{upstream_url}/v1",
                    upstream_api_key="upstream-test-key",
                    checkpoint_mode="cache",
                ),
                checkpoint_cache=PrefixCheckpointCache(secret=b"codex-tool-e2e-test"),
            )
            with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as root:
                root_path = Path(root)
                codex_home = root_path / "codex-home"
                workspace = root_path / "workspace"
                codex_home.mkdir()
                workspace.mkdir()
                _write_codex_config(codex_home, relay_url)
                result = _run_codex(
                    codex,
                    codex_home,
                    workspace,
                    [
                        "exec",
                        "--json",
                        "--skip-git-repo-check",
                        "--sandbox",
                        "read-only",
                        "--cd",
                        str(workspace),
                        "Run the requested model tool, then return its final result.",
                    ],
                )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RELAY_CODEX_TURN_2_OK", result.stdout)
        self.assertEqual(len(upstream.main_requests), 2)
        self.assertEqual(upstream.summary_requests, [])
        replayed = upstream.main_requests[1]["input"]
        reasoning = next(item for item in replayed if item.get("type") == "reasoning")
        call = next(item for item in replayed if item.get("type") == "function_call")
        output = next(
            item for item in replayed if item.get("type") == "function_call_output"
        )
        self.assertEqual(
            reasoning["encrypted_content"],
            "cmVsYXktY29kZXgtZTJlLXJlYXNvbmluZw==",
        )
        self.assertEqual(call["call_id"], output["call_id"])
        self.assertEqual(call["name"], "exec_command")


if __name__ == "__main__":
    unittest.main()
