"""Real CLI harnesses, isolated configuration, and a local scripted model service.

Only the model replies are fake: the CLI, Relay strategies, HTTP, SSE, session
serialization and (in tool tests) the shell execution are real.
"""
from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

from starlette.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from relay import (
    AgentFold, AutoCompact, Checkpoint, Compact, ContextFolding,
    PrefixCheckpointCache, ProLong, RLM, RollingMemory, SelectiveDiscard,
    SlidingWindow, MultiGranCompact,
)
from relay.proxy import ProxyConfig, create_app
from relay.strategies.compact import _protected_prefix
from tests.test_codex_e2e import (
    _FakeResponsesUpstream, _is_summary_request, _response, _text, _tool_sse,
)


STRATEGIES = (
    "compact", "checkpoint", "rolling_memory", "context_folding",
    "agent_fold", "auto_compact", "prolong", "sliding_window",
    "selective_discard", "rlm", "multi_gran_compact",
)


def strategy_for(name):
    return {
        "compact": lambda: Compact(compact_threshold=100),
        "checkpoint": lambda: Checkpoint(checkpoint_threshold=65, context_threshold=180),
        "rolling_memory": lambda: RollingMemory(update_input_tokens=180),
        "context_folding": ContextFolding,
        "agent_fold": AgentFold,
        "auto_compact": lambda: AutoCompact(fallback_threshold=1_000_000),
        "prolong": lambda: ProLong(context_threshold=180),
        "sliding_window": lambda: SlidingWindow(max_input_tokens=180),
        "selective_discard": lambda: SelectiveDiscard(keep_recent_interactions=0),
        "rlm": lambda: RLM(manager_model="relay-rlm-test-model", max_iterations=3),
        "multi_gran_compact": lambda: MultiGranCompact(compact_threshold=30,
                                                       compact_model="relay-note-model"),
    }[name]()


class HarnessUpstream(_FakeResponsesUpstream):
    def __init__(self, *, tool_name=None):
        super().__init__()
        self.tool_name = tool_name
        self.paths = []

    async def dispatch(self, request):
        self.paths.append(request.url.path)
        body = await request.json()
        if request.url.path == "/v1/chat/completions" and body.get("model") == "relay-note-model":
            self.summary_requests.append(body)
            return JSONResponse({"id": "chat_note", "object": "chat.completion", "created": 1,
                "model": body["model"], "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "You preserved BLUE-ORCHID-731 in the memory note."},
                    "finish_reason": "stop"}]})
        schema = body.get("text", {}).get("format", {}).get("name")
        if schema == "relay_context_folding_decision":
            self.manager_requests.append(body)
            number = sum(i.get("text", {}).get("format", {}).get("name") == schema
                         for i in self.manager_requests)
            # Unlike Codex's three-turn fixture, these tests continue past return.
            value = ({"action": "open", "objective": "resume subtask", "summary": ""}
                     if number % 2 else {"action": "return", "objective": "", "summary": "hidden branch completed"})
            return JSONResponse(_response(body, json.dumps(value), f"resp_fold_{number}"))
        if schema == "relay_selective_discard_decision":
            self.manager_requests.append(body)
            return JSONResponse(_response(body, json.dumps({
                "drop_steps": [1], "tool_output_edits": [], "reason": "scripted obsolete step",
            }), f"resp_discard_{len(self.manager_requests)}"))
        is_task = (request.url.path == "/v1/responses" and not schema
                   and not _is_summary_request(body))
        if self.tool_name and is_task and not self.main_requests:
            self.main_requests.append(body)
            # Same Responses SSE envelope, but each harness has its own tool schema.
            payload = _tool_sse(body, "resp_harness_tool")
            events = []
            arguments = json.dumps({"command": "printf RELAY_TOOL_OK",
                                    "description": "local E2E marker"})
            for block in payload.decode().strip().split("\n\n"):
                event = json.loads(block.split("data: ", 1)[1])
                if event.get("item", {}).get("type") == "reasoning":
                    continue
                if event.get("type", "").startswith("response.function_call_arguments"):
                    event["name"] = self.tool_name
                    if "delta" in event:
                        event["delta"] = arguments
                    if "arguments" in event:
                        event["arguments"] = arguments
                if "response" in event:
                    event["response"]["output"] = [
                        item for item in event["response"].get("output", [])
                        if item.get("type") != "reasoning"
                    ]
                for item in [event.get("item", {}), *event.get("response", {}).get("output", [])]:
                    if item.get("type") == "function_call":
                        item["name"] = self.tool_name
                        if item.get("arguments"):
                            item["arguments"] = arguments
                if "output_index" in event:
                    event["output_index"] = 0
                event["sequence_number"] = len(events)
                events.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
            return Response("".join(events), media_type="text/event-stream")
        return await super().dispatch(request)


def relay_for(strategy, upstream_url):
    cache = PrefixCheckpointCache(secret=b"isolated-harness-e2e")
    app = create_app(strategy, ProxyConfig(
        upstream_base_url=f"{upstream_url}/v1", upstream_api_key="fake-upstream-key",
        checkpoint_mode="cache",
    ), checkpoint_cache=cache)
    raw = []
    class CaptureInput(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path == "/v1/responses":
                raw.append(deepcopy(await request.json()))
            return await call_next(request)

    app.add_middleware(CaptureInput)
    return app, cache, raw


def clean_env(root):
    # Allowlist, not inherited credentials/plugins/config from the user's shell.
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    env.update({"XDG_CONFIG_HOME": str(root / "config"),
                "XDG_DATA_HOME": str(root / "data"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "XDG_STATE_HOME": str(root / "state"),
                "NO_PROXY": "127.0.0.1,localhost", "RELAY_TEST_API_KEY": "tenant-test"})
    return env


def configure(harness, root, relay_url, model_id="relay-test-model"):
    env = clean_env(root)
    if harness == "opencode":
        config = {
            "model": f"openai/{model_id}", "small_model": f"openai/{model_id}",
            "enabled_providers": ["openai"], "plugin": [],
            "agent": {"title": {"disable": True}, "summary": {"disable": True}},
            "compaction": {"auto": False, "prune": False},
            "provider": {"openai": {"options": {"baseURL": f"{relay_url}/v1", "apiKey": "tenant-test"},
                "models": {model_id: {"name": "Relay test", "limit": {"context": 1000000, "output": 4096}}}}},
        }
        config_path = root / "opencode.json"
        config_path.write_text(json.dumps(config))
        env.update({"OPENCODE_CONFIG": str(config_path), "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                    "OPENCODE_DISABLE_AUTOUPDATE": "true", "OPENCODE_DISABLE_MODELS_FETCH": "true"})
    else:
        agent_dir = root / "pi-agent"
        agent_dir.mkdir(exist_ok=True)
        (agent_dir / "settings.json").write_text(json.dumps({
            "compaction": {"enabled": False}, "retry": {"enabled": False},
            "transport": "sse", "packages": [], "enableInstallTelemetry": False,
        }))
        (agent_dir / "models.json").write_text(json.dumps({"providers": {"relay": {
            "baseUrl": f"{relay_url}/v1", "api": "openai-responses", "apiKey": "RELAY_TEST_API_KEY",
            "models": [{"id": model_id, "name": "Relay test", "reasoning": False,
                        "input": ["text"], "contextWindow": 1000000, "maxTokens": 4096,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
        }}}))
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    return env


def run_turn(harness, binary, root, env, turn, *, tools=False):
    prompt = f"Turn {turn}: preserve BLUE-ORCHID-731. Return the model result."
    if harness == "opencode":
        args = [binary, "run", "--pure", "--format", "json", "--model", "openai/relay-test-model"]
        if turn > 1:
            args += ["--continue"]
    else:
        args = [binary, "--mode", "json", "--print", "--provider", "relay", "--model", "relay-test-model",
                "--session", str(root / "session.jsonl"), "--no-extensions", "--no-skills", "--no-prompt-templates"]
        args += ["--tools", "bash"] if tools else ["--no-tools"]
    return subprocess.run([*args, prompt], cwd=root, env=env, capture_output=True, text=True, timeout=45)


def assert_strategy_result(test, name, upstream, raw):
    test.assertEqual(len(upstream.main_requests), 4)
    test.assertEqual(len(raw), 4, "Every task model request must enter Relay preparation")
    test.assertNotIn("/v1/responses/compact", upstream.paths)
    for body in upstream.main_requests:
        test.assertTrue(body["stream"])
        test.assertIsNone(body.get("previous_response_id"))
        test.assertIsNone(body.get("conversation"))
        test.assertFalse(any(item.get("type") == "compaction" for item in body["input"]))
        if name != "rlm":
            test.assertIn("Turn 4" if body is upstream.main_requests[-1] else "Turn", "\n".join(_text(i) for i in body["input"]))
    changed = [(a["input"], b["input"]) for a, b in zip(raw, upstream.main_requests) if a["input"] != b["input"]]
    test.assertTrue(changed, f"{name} must actually transform a real harness input")
    for before, after in zip(raw, upstream.main_requests):
        test.assertEqual(_protected_prefix(before["input"]), _protected_prefix(after["input"]))
        test.assertFalse(after.get("text", {}).get("format", {}).get("name", "").startswith("relay_"))
        test.assertFalse(any(tool.get("name") in {"log_read", "log_python"}
                             for tool in after.get("tools", [])))
    final_text = "\n".join(_text(i) for body in upstream.main_requests for i in body["input"])
    markers = {
        "compact": "relay checkpoint summary", "checkpoint": "relay checkpoint summary",
        "rolling_memory": "relay checkpoint summary", "context_folding": "hidden branch completed",
        "agent_fold": "agent fold state", "auto_compact": "auto compact working state",
        "prolong": "PRO-LONG working context", "rlm": "Result from the Recursive Language Model",
        "multi_gran_compact": "You preserved BLUE-ORCHID-731 in the memory note.",
    }
    if name in markers:
        test.assertIn(markers[name], final_text)
    if name in {"compact", "checkpoint", "rolling_memory", "multi_gran_compact"}:
        test.assertTrue(upstream.summary_requests)
    elif name == "rlm":
        test.assertEqual(len(upstream.rlm_requests), 8)
    elif name == "sliding_window":
        test.assertTrue(any(len(after) < len(before) for before, after in changed))
        test.assertFalse(upstream.manager_requests)
    else:
        test.assertTrue(upstream.manager_requests)
    # All transformed histories retain protocol-atomic call/result pairs.
    for body in upstream.main_requests:
        calls = {i["call_id"] for i in body["input"] if i.get("type") == "function_call"}
        results = {i["call_id"] for i in body["input"] if i.get("type") == "function_call_output"}
        test.assertEqual(calls, results)
