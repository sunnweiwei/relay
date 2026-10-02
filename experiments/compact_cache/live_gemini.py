"""Gemini CLI native wire through Relay and a GPT Responses task bridge."""
from __future__ import annotations

import json
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

from relay import Compact
from relay.gemini import GeminiInput
from relay.middleware import trajectory_digest
from relay.proxy import ProxyConfig, create_app
from relay.strategies.compact import CODEX_SUMMARY_PREFIX
from tests.harness_e2e_support import clean_env
from tests.test_codex_e2e import _serve

from experiments.compact_cache.case import CONTEXT, FIRST, FOURTH, MORE_CONTEXT, THIRD
from experiments.compact_cache.checks import RecordingCache
from experiments.compact_cache.budget import TaskBudget
from experiments.compact_cache.live_responses import _Management
from experiments.compact_cache.openai_gemini_bridge import create_bridge
from experiments.compact_cache.providers import route_for
from experiments.compact_cache.threshold import calibrate_trigger_only
from experiments.compact_cache.diagnostic_compact import DiagnosticCompact


def run(model: str) -> dict[str, Any]:
    if os.getenv("RELAY_LIVE_TESTS") != "1":
        raise ValueError("set RELAY_LIVE_TESTS=1 before a billable run")
    route = route_for(model)
    local = Path(__file__).resolve().parent / ".tools/node_modules/.bin/gemini"
    binary = str(local) if local.is_file() else shutil.which("gemini")
    if not binary:
        raise ValueError("Gemini CLI is not installed")
    cache = RecordingCache()
    strategy = DiagnosticCompact(compact_threshold=1_000_000_000)
    raw: list[dict[str, Any]] = []
    forwarded: list[dict[str, Any]] = []
    calibration: dict[str, int] = {}
    with OpenAI(base_url=route.base_url, api_key=route.api_key,
                timeout=90, max_retries=0) as client:
        manager = _Management(client)
        budget = TaskBudget(client, max_total_input=36_000)
        bridge = create_bridge(client, model, forwarded, budget)
        with _serve(bridge) as bridge_url:
            relay = create_app(
                strategy,
                ProxyConfig(upstream_base_url=bridge_url,
                            checkpoint_mode="cache", gemini_enabled=True,
                            management_model=model),
                checkpoint_cache=cache, management_responses=manager,
            )

            class Capture(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    if request.url.path.endswith((":generateContent",
                                                  ":streamGenerateContent")):
                        if len(raw) >= 4:
                            from starlette.responses import JSONResponse
                            return JSONResponse({"error": "live request budget exceeded"},
                                                status_code=429)
                        body = await request.json()
                        if len(raw) == 1:
                            calibration.update(await run_in_threadpool(
                                calibrate_trigger_only, manager,
                                GeminiInput(raw[0], model).request,
                                GeminiInput(body, model).request))
                            strategy.compact_threshold = calibration["compact_threshold"]
                        raw.append(deepcopy(body))
                    return await call_next(request)

            relay.add_middleware(Capture)
            with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workspace = root / "workspace"
                workspace.mkdir()
                config = root / "gemini-home" / ".gemini"
                config.mkdir(parents=True)
                shutil.copyfile(Path(__file__).resolve().parents[2] / "tests" /
                                "fixtures" / "gemini_proxy_settings.json",
                                config / "settings.json")
                env = clean_env(root)
                env.update({"GEMINI_CLI_HOME": str(config.parent),
                            "GEMINI_API_KEY": "local-tenant",
                            "GEMINI_CLI_TRUST_WORKSPACE": "true",
                            "GOOGLE_GEMINI_BASE_URL": relay_url,
                            "GOOGLE_GENAI_API_VERSION": "v1beta"})
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
                    args = [binary, "-p", prompt, "-m", "gemini-3.8-flash",
                            "--output-format", "json"]
                    if index:
                        args += ["--resume", "latest"]
                    result = subprocess.run(args, cwd=workspace, env=env,
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
                    if index >= 2 and cache.stats().hits > prior[3]:
                        try:
                            answered = json.loads(result.stdout).get("response")
                        except (TypeError, ValueError):
                            answered = None
                        if answered == "BLUE-ORCHID-731":
                            break

    hit = cache.lookups[-1] if cache.lookups else None
    exact = bool(hit and hit.matched_items is not None and any(
        saved.covered_items == hit.matched_items
        and saved.prefix_digest == trajectory_digest(hit.trajectory[:hit.matched_items])
        and saved.artifact_digest == trajectory_digest([hit.artifact])
        for saved in cache.stored))
    transformed = any(CODEX_SUMMARY_PREFIX in str(body.get("contents"))
                      for body in forwarded)
    no_repeat = (before_last is not None and
                 all(manager.summary_from_checkpoint[before_last:]))
    protocol = (final_turn_sent and len(results) in {3, 4}
                and len(results) == len(raw) == len(forwarded)
                and all(item["exit_code"] == 0 for item in results))
    answer = bool(results and "BLUE-ORCHID-731" in results[-1]["stdout"])
    return {"harness": "gemini-cli", "model": model, "strategy": "compact",
            "checkpoint_mode": "cache", "task_requests": len(forwarded),
            "relay_requests": len(raw), "task_model_requests": budget.requests,
            "turn_traces": turn_traces,
            "threshold_calibration": calibration,
            "budgeted_task_requests": budget.requests,
            "budgeted_task_input_tokens": budget.input_tokens,
            "summary_calls": len(manager.summary_inputs),
            "summary_calls_during_final_turn": (len(manager.summary_inputs) - before_last
                                                if before_last is not None else None),
            "new_summaries_use_checkpoint": no_repeat,
            "cache_hits": cache.stats().hits,
            "live_protocol_pass": protocol,
            "compaction_pass": bool(manager.summary_inputs) and transformed,
            "cache_pass": exact and no_repeat,
            "answer_pass": answer,
            "stages": results,
            "status": "pass" if protocol and transformed and exact and no_repeat else "fail"}
