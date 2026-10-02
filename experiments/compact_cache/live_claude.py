"""Bounded Claude Code → Relay Messages adapter → OpenAI Responses case."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import OpenAI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool

from relay.middleware import trajectory_digest
from relay.strategies.compact import CODEX_SUMMARY_PREFIX
from tests.harness_e2e_support import clean_env
from tests.test_codex_e2e import _serve

from experiments.compact_cache.anthropic_adapter import create_anthropic_adapter, to_relay_request
from experiments.compact_cache.case import CONTEXT, FIRST, FOURTH, MORE_CONTEXT, THIRD
from experiments.compact_cache.checks import RecordingCache
from experiments.compact_cache.budget import TaskBudget
from experiments.compact_cache.live_responses import _Management
from experiments.compact_cache.providers import route_for
from experiments.compact_cache.openai_messages_bridge import create_bridge
from experiments.compact_cache.threshold import calibrate_trigger_only
from experiments.compact_cache.diagnostic_compact import DiagnosticCompact


def run(model: str) -> dict[str, Any]:
    if os.getenv("RELAY_LIVE_TESTS") != "1":
        raise ValueError("set RELAY_LIVE_TESTS=1 before a billable run")
    route = route_for(model)
    binary = shutil.which("claude")
    if not binary:
        raise ValueError("Claude Code CLI is not installed")
    cache = RecordingCache()
    raw: list[dict[str, Any]] = []
    task_requests: list[dict[str, Any]] = []
    calibration: dict[str, int] = {}
    with OpenAI(base_url=route.base_url, api_key=route.api_key,
                timeout=90, max_retries=0) as client:
        manager = _Management(client)
        budget = TaskBudget(client)
        bridge = create_bridge(client, model, task_requests, budget)
        with _serve(bridge) as bridge_url:
            adapter = create_anthropic_adapter(
                task_base_url=bridge_url, management_responses=manager,
                manager_model=model, task_model=model, task_auth_token="local-bridge",
                cache=cache, compact_threshold=1_000_000_000,
            )
            adapter.state.engine.strategy = DiagnosticCompact(
                compact_threshold=1_000_000_000)

            class Capture(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    if request.url.path == "/v1/messages":
                        if len(raw) >= 4:
                            from starlette.responses import JSONResponse
                            return JSONResponse({"error": "live request budget exceeded"},
                                                status_code=429)
                        body = await request.json()
                        if len(raw) == 1:
                            calibration.update(await run_in_threadpool(
                                calibrate_trigger_only, manager,
                                to_relay_request(raw[0], manager_model=model),
                                to_relay_request(body, manager_model=model)))
                            adapter.state.engine.strategy.compact_threshold = (
                                calibration["compact_threshold"])
                        raw.append(deepcopy(body))
                    return await call_next(request)

            adapter.add_middleware(Capture)
            with _serve(adapter) as relay_url, tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = clean_env(root)
                env.update({
                    "ANTHROPIC_BASE_URL": relay_url,
                    "ANTHROPIC_API_KEY": "local-tenant",
                    "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
                    "CLAUDE_CONFIG_DIR": str(root / "claude-config"),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CODE_MAX_RETRIES": "0", "DISABLE_COMPACT": "1",
                    "DISABLE_TELEMETRY": "1",
                })
                base = [binary, "--print", "--output-format", "json",
                        "--model", model, "--tools", "", "--setting-sources", ""]
                results = []
                before_last = None
                turn_traces: list[dict[str, int]] = []
                final_turn_sent = False
                for index in range(4):
                    if index == 0:
                        prompt = FIRST
                    elif index == 1:
                        prompt = CONTEXT
                    elif index == 2:
                        prompt = THIRD if manager.summary_inputs else MORE_CONTEXT
                    elif manager.summary_inputs:
                        prompt = FOURTH
                    else:
                        break
                    if prompt in (THIRD, FOURTH):
                        final_turn_sent = True
                        before_last = len(manager.summary_inputs)
                    prior = (len(raw), budget.requests, len(manager.summary_inputs),
                             cache.stats().hits)
                    args = [*base, *(["--continue"] if index else []), prompt]
                    result = subprocess.run(args, cwd=root, env=env,
                                            stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True,
                                            timeout=90, check=False)
                    results.append({"exit_code": result.returncode,
                                    "stdout": result.stdout[-2000:],
                                    "stderr": result.stderr[-1000:]})
                    turn_traces.append({"turn": index + 1,
                                        "relay_requests": len(raw) - prior[0],
                                        "task_model_requests": budget.requests - prior[1],
                                        "summary_calls": len(manager.summary_inputs) - prior[2],
                                        "cache_hits": cache.stats().hits - prior[3]})
                    if result.returncode:
                        break

    hit = next((lookup for lookup in cache.lookups[2:]
                if lookup.matched_items is not None), None)
    exact = bool(hit and any(
        saved.covered_items == hit.matched_items
        and saved.prefix_digest == trajectory_digest(hit.trajectory[:hit.matched_items])
        and saved.artifact_digest == trajectory_digest([hit.artifact])
        for saved in cache.stored))
    visible_private = any("relay checkpoint" in result["stdout"]
                          or '"type":"compaction"' in result["stdout"]
                          for result in results)
    protocol = (final_turn_sent and len(results) in {3, 4}
                and len(results) == len(raw) == len(task_requests)
                and all(result["exit_code"] == 0 for result in results)
                and not visible_private)
    transformed = any(CODEX_SUMMARY_PREFIX in str(body.get("messages"))
                      for body in task_requests)
    compacted = bool(manager.summary_inputs) and transformed
    reused = (exact and before_last is not None and
              all(manager.summary_from_checkpoint[before_last:]))
    answer = bool(results and "BLUE-ORCHID-731" in results[-1]["stdout"])
    passed = protocol and compacted and reused
    return {"harness": "claude-code", "model": model,
            "strategy": "compact", "checkpoint_mode": "cache",
            "task_requests": len(task_requests),
            "relay_requests": len(raw), "task_model_requests": budget.requests,
            "turn_traces": turn_traces,
            "threshold_calibration": calibration,
            "budgeted_task_requests": budget.requests,
            "budgeted_task_input_tokens": budget.input_tokens,
            "summary_calls": len(manager.summary_inputs),
            "summary_calls_during_final_turn": (len(manager.summary_inputs) - before_last
                                                if before_last is not None else None),
            "new_summaries_use_checkpoint": (bool(before_last is not None and
                all(manager.summary_from_checkpoint[before_last:]))),
            "cache_hits": cache.stats().hits,
            "exact_checkpoint_reused": exact,
            "harness_response_private_state": visible_private,
            "answer_marker_retained": answer,
            "live_protocol_pass": protocol, "compaction_pass": compacted,
            "cache_pass": reused, "answer_pass": answer,
            "stages": results, "status": "pass" if passed else "fail"}
