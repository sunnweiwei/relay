"""Bounded real Responses Harness → Relay Compact/Cache → model API."""
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
from relay.middleware import trajectory_digest
from relay.middleware import _CACHE_SCOPE_KEYS
from relay.proxy import ProxyConfig, create_app
from relay.strategies.compact import CODEX_SUMMARY_PREFIX
from tests.test_codex_e2e import _serve
from tests.harness_e2e_support import configure

from experiments.compact_cache.checks import RecordingCache
from experiments.compact_cache.case import CONTEXT, FIRST, FOURTH, MORE_CONTEXT, THIRD
from experiments.compact_cache.providers import route_for
from experiments.compact_cache.budget import (
    BoundedResponsesTransport, MAX_INPUT_PER_REQUEST, MAX_SUMMARY_OUTPUT,
    TaskBudget,
)
from experiments.compact_cache.threshold import calibrate, calibrate_trigger_only
from experiments.compact_cache.diagnostic_compact import DiagnosticCompact

class _Management:
    def __init__(self, client: OpenAI, *, max_calls: int = 4,
                 max_input_per_request: int = MAX_INPUT_PER_REQUEST,
                 max_total_input: int = 24_000) -> None:
        self.responses = client.responses
        self.input_tokens = self
        self.count_calls = 0
        self.summary_inputs: list[str] = []
        self.summary_from_checkpoint: list[bool] = []
        self.summary_input_tokens = 0
        self.max_calls = max_calls
        self.max_input_per_request = max_input_per_request
        self.max_total_input = max_total_input

    def count(self, **kwargs):
        self.count_calls += 1
        return self.responses.input_tokens.count(**kwargs)

    def create(self, **kwargs):
        if len(self.summary_inputs) >= self.max_calls:
            raise ValueError("summary request budget exceeded")
        count_fields = {key: kwargs[key] for key in
                        ("model", "input", "instructions", "reasoning") if key in kwargs}
        tokens = self.responses.input_tokens.count(**count_fields).input_tokens
        if tokens > self.max_input_per_request:
            raise ValueError("summary input token budget exceeded")
        if self.summary_input_tokens + tokens > self.max_total_input:
            raise ValueError("cumulative summary input token budget exceeded")
        self.summary_input_tokens += tokens
        kwargs["max_output_tokens"] = MAX_SUMMARY_OUTPUT
        kwargs["reasoning"] = {"effort": "none"}
        result = self.responses.create(**kwargs)
        self.summary_inputs.append(trajectory_digest(kwargs["input"]))
        self.summary_from_checkpoint.append(CODEX_SUMMARY_PREFIX in str(kwargs["input"]))
        return result


class _Compact(Compact):
    def __init__(self) -> None:
        super().__init__(compact_threshold=1_000_000_000)
        self.forwarded: list[list[dict[str, Any]]] = []

    def prepare(self, responses, request, trajectory, checkpoint=None):
        prepared = super().prepare(responses, request, trajectory, checkpoint)
        self.forwarded.append(deepcopy(prepared.input))
        return prepared


class _DiagnosticCompact(DiagnosticCompact):
    def __init__(self) -> None:
        super().__init__(compact_threshold=1_000_000_000)
        self.forwarded: list[list[dict[str, Any]]] = []

    def prepare(self, responses, request, trajectory, checkpoint=None):
        prepared = super().prepare(responses, request, trajectory, checkpoint)
        self.forwarded.append(deepcopy(prepared.input))
        return prepared


def _run_codex(binary: str, home: Path, workspace: Path, prompt: str,
               *, resume: bool) -> subprocess.CompletedProcess[str]:
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TMPDIR") if key in os.environ}
    env.update({"CODEX_HOME": str(home), "RELAY_TEST_API_KEY": "local-tenant",
                "HOME": str(home.parent), "NO_COLOR": "1",
                "NO_PROXY": "127.0.0.1,localhost"})
    args = ([binary, "-a", "never", "exec", "resume", "--last", "--json",
             "--skip-git-repo-check", prompt] if resume else
            [binary, "-a", "never", "exec", "--json", "--skip-git-repo-check",
             "--sandbox", "read-only", "--cd", str(workspace), prompt])
    return subprocess.run(args, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=90, check=False)


def _run_opencode_or_pi(harness: str, binary: str, root: Path,
                        relay_url: str, model: str, prompt: str,
                        *, resume: bool, allow_tools: bool = False) -> subprocess.CompletedProcess[str]:
    env = configure(harness, root, relay_url, model_id=model)
    if harness == "opencode":
        args = [binary, "run", "--pure", "--format", "json",
                "--model", f"openai/{model}"]
        if resume:
            args.append("--continue")
    else:
        args = [binary, "--mode", "json", "--print", "--provider", "relay",
                "--model", model, "--session", str(root / "session.jsonl"),
                "--no-extensions", "--no-skills", "--no-prompt-templates",
                *([] if allow_tools else ["--no-tools"])]
    return subprocess.run([*args, prompt], cwd=root, env=env,
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=90, check=False)


def _run_mini_swe(root: Path, relay_url: str, model: str) -> dict[str, Any]:
    previous = {name: os.environ.get(name) for name in (
        "MSWEA_GLOBAL_CONFIG_DIR", "MSWEA_SILENT_STARTUP",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT")}
    os.environ.update({
        "MSWEA_GLOBAL_CONFIG_DIR": str(root / "mini-config"),
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
    })
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_response_model import LitellmResponseModel

        task_model = LitellmResponseModel(
            model_name=f"openai/{model}",
            model_kwargs={"api_base": f"{relay_url}/v1", "api_key": "tenant-test"},
            cost_tracking="ignore_errors",
        )
        agent = DefaultAgent(
            task_model, LocalEnvironment(cwd=str(root)),
            system_template=(
                "You are a coding agent. Every response must call the bash tool. "
                "Run the requested two ACK commands in separate tool calls. "
                "To submit, call bash with a command whose output starts with "
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on its own first line, "
                "followed by BLUE-ORCHID-731 on the next line. Do not access files."
            ),
            instance_template="{{task}}", step_limit=6, cost_limit=0,
        )
        return agent.run("Remember BLUE-ORCHID-731. Run bash `printf ACK` twice "
                         "in separate model steps, then submit BLUE-ORCHID-731.")
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(harness: str, model: str) -> dict[str, Any]:
    if os.getenv("RELAY_LIVE_TESTS") != "1":
        raise ValueError("set RELAY_LIVE_TESTS=1 before a billable run")
    if harness not in {"codex", "opencode", "pi", "mini-swe"}:
        raise ValueError(f"{harness}: not a Responses CLI route")
    route = route_for(model)
    local_pi = Path(__file__).resolve().parent / ".tools/node_modules/.bin/pi"
    binary = (str(local_pi) if harness == "pi" and local_pi.is_file()
              else shutil.which(harness)) if harness != "mini-swe" else None
    if harness != "mini-swe" and not binary:
        raise ValueError(f"{harness} CLI is not installed")

    cache = RecordingCache()
    diagnostic = harness in {"pi", "mini-swe"}
    strategy = _DiagnosticCompact() if diagnostic else _Compact()
    raw: list[dict[str, Any]] = []
    visible: list[bytes] = []
    before_last: list[int] = []
    calibration: dict[str, int] = {}
    turn_traces: list[dict[str, int]] = []
    with OpenAI(base_url=route.base_url, api_key=route.api_key,
                timeout=90, max_retries=0) as client:
        management = _Management(client)
        budget = TaskBudget(client, max_requests=6 if harness == "mini-swe" else 4)
        relay = create_app(strategy, ProxyConfig(
            upstream_base_url=route.base_url, upstream_api_key=route.api_key,
            checkpoint_mode="cache"), checkpoint_cache=cache,
            management_responses=management,
            upstream_transport=BoundedResponsesTransport(budget))

        class Capture(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path != "/v1/responses":
                    return await call_next(request)
                if len(raw) >= (6 if harness == "mini-swe" else 4):
                    from starlette.responses import JSONResponse
                    return JSONResponse({"error": "live request budget exceeded"}, status_code=429)
                if len(raw) == 1:
                    calibration.update(await run_in_threadpool(
                        calibrate_trigger_only if diagnostic else calibrate,
                        management, raw[0], await request.json()))
                    strategy.compact_threshold = calibration["compact_threshold"]
                if harness == "mini-swe" and len(raw) == 2:
                    before_last.append(len(management.summary_inputs))
                prior = (budget.requests, len(management.summary_inputs), cache.stats().hits)
                raw.append(deepcopy(await request.json()))
                response = await call_next(request)
                if harness == "mini-swe":
                    turn_traces.append({"turn": len(raw), "relay_requests": 1,
                                        "task_model_requests": budget.requests - prior[0],
                                        "summary_calls": len(management.summary_inputs) - prior[1],
                                        "cache_hits": cache.stats().hits - prior[2]})
                chunks: list[bytes] = []
                iterator = response.body_iterator

                async def observe():
                    async for chunk in iterator:
                        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
                        yield chunk
                    visible.append(b"".join(chunks))

                response.body_iterator = observe()
                return response

        relay.add_middleware(Capture)
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, workspace = root / "codex-home", root / "workspace"
            workspace.mkdir()
            if harness == "codex":
                home.mkdir()
                (home / "config.toml").write_text("\n".join([
                    f'model = "{model}"', 'model_provider = "relay"',
                    'model_auto_compact_token_limit = 1000000000', '',
                    '[features]', 'plugins = false', '', '[model_providers.relay]',
                    'name = "Relay"', f'base_url = "{relay_url}/v1"',
                    'env_key = "RELAY_TEST_API_KEY"', 'wire_api = "responses"',
                    'supports_websockets = false', '',
                ]))
            results = []
            if harness == "mini-swe":
                outcome = _run_mini_swe(root, relay_url, model)
                results.append({"exit_code": 0 if outcome.get("exit_status") == "Submitted" else 1,
                                "stdout": str(outcome.get("submission", ""))[-2000:],
                                "stderr": str(outcome.get("exit_status", ""))})
            else:
                final_turn_sent = False
                for index in range(4):
                    if index == 0:
                        prompt = FIRST
                    elif index == 1:
                        prompt = CONTEXT
                    elif index == 2:
                        prompt = THIRD if management.summary_inputs else MORE_CONTEXT
                    elif management.summary_inputs:
                        prompt = FOURTH
                    else:
                        break
                    if prompt in (THIRD, FOURTH):
                        final_turn_sent = True
                        before_last.append(len(management.summary_inputs))
                    prior = (len(raw), budget.requests, len(management.summary_inputs),
                             cache.stats().hits)
                    result = (_run_codex(binary, home, workspace, prompt,
                                         resume=index > 0)
                              if harness == "codex" else
                              _run_opencode_or_pi(harness, binary, root, relay_url,
                                                  model, prompt, resume=index > 0))
                    results.append({"exit_code": result.returncode,
                                    "api_error": (harness == "pi" and
                                                  ('"stopReason":"error"' in result.stdout or
                                                   '"errorMessage"' in result.stdout)),
                                    "stdout": result.stdout[-2000:],
                                    "stderr": result.stderr[-1000:]})
                    turn_traces.append({"turn": index + 1,
                                        "relay_requests": len(raw) - prior[0],
                                        "task_model_requests": budget.requests - prior[1],
                                        "summary_calls": len(management.summary_inputs) - prior[2],
                                        "cache_hits": cache.stats().hits - prior[3]})
                    if result.returncode or results[-1].get("api_error"):
                        break

    compacted = any(CODEX_SUMMARY_PREFIX in str(items) for items in strategy.forwarded)
    hit = next((lookup for lookup in cache.lookups[2:]
                if lookup.matched_items is not None), None)
    exact = bool(hit and any(
        saved.covered_items == hit.matched_items
        and saved.prefix_digest == trajectory_digest(hit.trajectory[:hit.matched_items])
        and saved.artifact_digest == trajectory_digest([hit.artifact])
        for saved in cache.stored))
    private_visible = any(b'"type":"compaction"' in body or
                          b'"type": "compaction"' in body for body in visible)
    stream_error = any(b'"response.failed"' in body or b'"type":"error"' in body
                       for body in visible)
    expected_results = 1 if harness == "mini-swe" else (4 if len(results) == 4 else 3)
    expected_requests = (len(raw) >= 3 if harness == "mini-swe" else
                         len(raw) == len(results) and len(results) >= 3 and final_turn_sent)
    expected_requests = expected_requests and budget.requests == len(raw)
    no_repeat = bool(before_last and
                     all(management.summary_from_checkpoint[before_last[0]:]))
    answer = bool(results and expected_requests and
                  "BLUE-ORCHID-731" in results[-1]["stdout"])
    passed = (len(results) == expected_results
              and all(r["exit_code"] == 0 and not r.get("api_error") for r in results)
              and expected_requests and bool(management.summary_inputs)
              and compacted and exact and not private_visible and not stream_error
              and no_repeat and answer)
    return {"harness": harness, "model": model, "strategy": "compact",
            "checkpoint_mode": "cache", "task_requests": len(raw),
            "relay_requests": len(raw), "task_model_requests": budget.requests,
            "turn_traces": turn_traces,
            "budgeted_task_requests": budget.requests,
            "budgeted_task_input_tokens": budget.input_tokens,
            "threshold_calibration": calibration,
            "cache_lookup_depths": [item.matched_items for item in cache.lookups],
            "cache_stored_depths": [item.covered_items for item in cache.stored],
            "raw_input_lengths": [len(item.get("input", [])) for item in raw],
            "saved_prefix_matches_third_raw": bool(len(raw) >= 3 and any(
                saved.prefix_digest == trajectory_digest(raw[2]["input"][:saved.covered_items])
                for saved in cache.stored)),
            "scope_fields_changed": ([key for key in _CACHE_SCOPE_KEYS
                                      if raw[1].get(key) != raw[2].get(key)]
                                     if len(raw) >= 3 else []),
            "count_calls": management.count_calls,
            "summary_calls": len(management.summary_inputs),
            "budgeted_summary_input_tokens": management.summary_input_tokens,
            "summary_calls_during_final_turn": (len(management.summary_inputs) - before_last[0]
                                                if before_last else None),
            "new_summaries_use_checkpoint": no_repeat,
            "cache_hits": cache.stats().hits, "exact_checkpoint_reused": exact,
            "harness_response_private_state": private_visible,
            "stream_error": stream_error, "answer_marker_retained": answer,
            "live_protocol_pass": (len(results) == expected_results and
                                   all(r["exit_code"] == 0 and not r.get("api_error")
                                       for r in results) and
                                   expected_requests and not private_visible and
                                   not stream_error),
            "compaction_pass": bool(management.summary_inputs) and compacted,
            "cache_pass": exact and no_repeat,
            "answer_pass": answer,
            "stages": results, "status": "pass" if passed else "fail"}
