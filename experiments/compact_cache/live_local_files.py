"""Bounded two-user-turn local-file Compact/cache case using a real model API."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import OpenAI
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from relay import Compact
from relay.gemini import GeminiInput
from relay.middleware import trajectory_digest
from relay.middleware import _CACHE_SCOPE_KEYS
from relay.proxy import ProxyConfig, create_app
from relay.strategies.compact import CODEX_SUMMARY_PREFIX
from tests.harness_e2e_support import clean_env
from tests.test_codex_e2e import _serve

from experiments.compact_cache.anthropic_adapter import create_anthropic_adapter, to_relay_request
from experiments.compact_cache.budget import BoundedResponsesTransport, TaskBudget
from experiments.compact_cache.checks import RecordingCache
from experiments.compact_cache.diagnostic_compact import DiagnosticCompact
from experiments.compact_cache.live_responses import _Management, _run_codex, _run_opencode_or_pi
from experiments.compact_cache.local_file_case import FIRST, SECOND, install
from experiments.compact_cache.openai_gemini_bridge import create_bridge as gemini_bridge
from experiments.compact_cache.openai_messages_bridge import create_bridge as messages_bridge
from experiments.compact_cache.providers import route_for
from experiments.compact_cache.threshold import calibrate, calibrate_trigger_only

MAX_RELAY_REQUESTS = 8
MAX_TASK_REQUESTS = 8
MAX_TASK_INPUT = 24_000
MAX_TOTAL_TASK_INPUT = 100_000
MAX_SUMMARY_INPUT = 24_000
MAX_TOTAL_SUMMARY_INPUT = 90_000


class _ObservedCompact(Compact):
    def __init__(self) -> None:
        super().__init__(compact_threshold=1_000_000_000)
        self.forwarded: list[list[dict[str, Any]]] = []

    def prepare(self, responses, request, trajectory, checkpoint=None):
        prepared = super().prepare(responses, request, trajectory, checkpoint)
        self.forwarded.append(deepcopy(prepared.input))
        return prepared


class _ObservedDiagnostic(DiagnosticCompact):
    def __init__(self) -> None:
        super().__init__(compact_threshold=1_000_000_000,
                         max_summary_input=MAX_SUMMARY_INPUT)
        self.forwarded: list[list[dict[str, Any]]] = []

    def prepare(self, responses, request, trajectory, checkpoint=None):
        prepared = super().prepare(responses, request, trajectory, checkpoint)
        self.forwarded.append(deepcopy(prepared.input))
        return prepared


def _stage(result: subprocess.CompletedProcess[str], harness: str) -> dict[str, Any]:
    answer = ""
    if harness in {"gemini-cli", "claude-code"}:
        try:
            parsed = json.loads(result.stdout)
            answer = parsed.get("response" if harness == "gemini-cli" else "result") or ""
        except (TypeError, ValueError):
            pass
    else:
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if harness == "codex" and event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    answer = item.get("text", "")
            elif harness == "opencode" and event.get("type") == "text":
                answer = event.get("part", {}).get("text", "")
            elif harness == "pi" and event.get("type") == "turn_end":
                content = event.get("message", {}).get("content", [])
                answer = "\n".join(part.get("text", "") for part in content
                                   if part.get("type") == "text")
    return {"exit_code": result.returncode, "answer_text": answer[:1000],
            "stdout": result.stdout[-3000:], "stderr": result.stderr[-1200:]}


def _binary(harness: str) -> str:
    local = Path(__file__).resolve().parent / ".tools/node_modules/.bin" / (
        "gemini" if harness == "gemini-cli" else harness)
    binary = str(local) if local.is_file() else shutil.which(
        "gemini" if harness == "gemini-cli" else
        "claude" if harness == "claude-code" else harness)
    if not binary:
        raise ValueError(f"{harness} CLI is not installed")
    return binary


def _tool_pair(items: list[dict[str, Any]]) -> bool:
    calls = {item.get("call_id") for item in items
             if item.get("type") == "function_call"}
    return any(item.get("type") == "function_call_output"
               and item.get("call_id") in calls for item in items)


def _save_reasoning_items(path: Path, requests: list[dict[str, Any]]) -> None:
    captured = {
        "requests": [
            {"request": number, "reasoning_items": [
                {"input_index": index, "item": item}
                for index, item in enumerate(body.get("input", []), start=1)
                if item.get("type") == "reasoning"
            ]}
            for number, body in enumerate(requests, start=1)
        ]
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                         0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(captured, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _gemini_tool_shape(body: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for content in body.get("contents", []):
        for part in content.get("parts", []):
            for key in ("functionCall", "functionResponse"):
                if key in part:
                    result.append({"role": content.get("role"), "kind": key,
                                   "name": part[key].get("name"),
                                   "id": part[key].get("id")})
    return result


def _finish(harness: str, model: str, *, cache: RecordingCache,
            manager: _Management, budget: TaskBudget, strategy: Any,
            raw: list[dict[str, Any]], normalized: list[list[dict[str, Any]]],
            forwarded: list[list[dict[str, Any]]], stages: list[dict[str, Any]],
            second_start: int, second_summary_start: int,
            calibration: dict[str, int], protocol: str) -> dict[str, Any]:
    later = cache.lookups[second_start:]
    exact = bool(second_start > 0 and len(stages) == 2 and any(
        lookup.matched_items is not None and any(
        saved.covered_items == lookup.matched_items
        and saved.artifact_digest == trajectory_digest([lookup.artifact])
        for saved in cache.stored) for lookup in later))
    tool_pair = any(_tool_pair(items) for items in normalized[:second_start])
    summary_forwarded = any(CODEX_SUMMARY_PREFIX in str(items) for items in forwarded)
    no_old_repeat = all(manager.summary_from_checkpoint[second_summary_start:])
    first_answer = bool(stages and "750" in stages[0].get("answer_text", "") and
                        "production.json" in stages[0].get("answer_text", ""))
    second_text = (stages[1].get("answer_text", "") if len(stages) == 2 else "")
    second_answer = "2250" in re.sub(r"(?<=\d),(?=\d{3}\b)", "", second_text)
    protocol_pass = (len(stages) == 2 and len(raw) == len(normalized)
                     and len(raw) == len(forwarded) == budget.requests
                     and all(stage["exit_code"] == 0 for stage in stages))
    private_state = any('"type":"compaction"' in stage["stdout"]
                        or '"type": "compaction"' in stage["stdout"]
                        for stage in stages)
    trace = [{"request": index + 1,
              "user_turn": 1 if index < second_start else 2,
              "protocol": protocol,
              "reasoning_effort": (raw[index].get("reasoning", {}).get("effort")
                                   if index < len(raw) and isinstance(
                                       raw[index].get("reasoning"), dict) else None),
              "raw_items": len(items),
              "types": [item.get("type", "message") for item in items],
              "tool_call_ids": [item.get("call_id") for item in items
                                if item.get("type") == "function_call"],
              "tool_names": [item.get("name") for item in items
                             if item.get("type") == "function_call"],
              "tool_result_ids": [item.get("call_id") for item in items
                                  if item.get("type") == "function_call_output"],
              "tool_result_error_flags": [any(word in str(item.get("output", "")).lower()
                                               for word in ("error", "denied", "failed"))
                                          for item in items if item.get("type") == "function_call_output"],
              "tool_result_error_samples": [str(item.get("output", ""))[:240]
                  for item in items if item.get("type") == "function_call_output"
                  and any(word in str(item.get("output", "")).lower()
                          for word in ("error", "denied", "failed"))][-2:],
              "forwarded_items": len(forwarded[index]) if index < len(forwarded) else None,
              "summary_in_forwarded": (CODEX_SUMMARY_PREFIX in str(forwarded[index])
                                       if index < len(forwarded) else False),
              "cache_match_depth": (cache.lookups[index].matched_items
                                    if index < len(cache.lookups) else None),
              "matching_stored_prefix_depths": [saved.covered_items
                  for saved in cache.stored if len(items) >= saved.covered_items
                  and trajectory_digest(items[:saved.covered_items]) == saved.prefix_digest],
              "item_digests": [trajectory_digest([item])[:16] for item in items]}
             for index, items in enumerate(normalized)]
    passed = (protocol_pass and tool_pair and bool(manager.summary_inputs)
              and summary_forwarded and exact and no_old_repeat
              and not private_state and first_answer and second_answer)
    return {"status": "pass" if passed else "fail", "harness": harness,
            "model": model, "case": "local-files", "mode": "live",
            "strategy": "compact", "checkpoint_mode": "cache",
            "threshold_policy": ("experiment_separate_summary_input_budget"
                                 if isinstance(strategy, DiagnosticCompact)
                                 else "committed_compact"),
            "threshold_calibration": calibration,
            "task_model_requests": budget.requests,
            "task_input_tokens": budget.input_tokens,
            "summary_calls": len(manager.summary_inputs),
            "summary_input_tokens": manager.summary_input_tokens,
            "cache_hits": cache.stats().hits,
            "first_turn_tool_pair": tool_pair,
            "compaction_pass": bool(manager.summary_inputs) and summary_forwarded,
            "cache_pass": exact and no_old_repeat,
            "exact_second_turn_checkpoint": exact,
            "old_prefix_not_resummarized": no_old_repeat,
            "live_protocol_pass": protocol_pass and not private_state,
            "harness_response_private_state": private_state,
            "first_answer_pass": first_answer, "second_answer_pass": second_answer,
            "answer_pass": first_answer and second_answer,
            "relay_requests": len(raw), "second_turn_starts_at_request": second_start + 1,
            "cache_stored_depths": [item.covered_items for item in cache.stored],
            "scope_fields_changed_at_second_turn": ([key for key in _CACHE_SCOPE_KEYS
                if second_start and raw[second_start].get(key) != raw[second_start-1].get(key)]
                if second_start < len(raw) else []),
            "wire_trace": trace, "stages": stages}


def _run_responses(harness: str, model: str, client: OpenAI, route: Any,
                   reasoning_capture_path: Path | None = None) -> dict[str, Any]:
    cache = RecordingCache()
    diagnostic = harness in {"pi", "mini-swe"}
    strategy = _ObservedDiagnostic() if diagnostic else _ObservedCompact()
    manager = _Management(client, max_calls=6,
                          max_input_per_request=MAX_SUMMARY_INPUT,
                          max_total_input=MAX_TOTAL_SUMMARY_INPUT)
    budget = TaskBudget(client, max_requests=MAX_TASK_REQUESTS,
                        max_total_input=MAX_TOTAL_TASK_INPUT,
                        max_input_per_request=MAX_TASK_INPUT, max_output=512)
    raw: list[dict[str, Any]] = []
    calibration: dict[str, int] = {}
    relay = create_app(strategy, ProxyConfig(
        upstream_base_url=route.base_url, upstream_api_key=route.api_key,
        checkpoint_mode="cache"), checkpoint_cache=cache,
        management_responses=manager,
        upstream_transport=BoundedResponsesTransport(budget))

    class Capture(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path != "/v1/responses":
                return await call_next(request)
            if len(raw) >= MAX_RELAY_REQUESTS:
                return JSONResponse({"error": "relay request budget exceeded"}, status_code=429)
            body = await request.json()
            if not calibration and raw and _tool_pair(body.get("input", [])):
                fn = calibrate_trigger_only if diagnostic else calibrate
                calibration.update(await run_in_threadpool(fn, manager, raw[0], body))
                strategy.compact_threshold = calibration["compact_threshold"]
            raw.append(deepcopy(body))
            if reasoning_capture_path is not None:
                _save_reasoning_items(reasoning_capture_path, raw)
            return await call_next(request)

    relay.add_middleware(Capture)
    with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = root / "workspace" if harness == "codex" else root
        workspace.mkdir(exist_ok=True)
        install(workspace)
        if harness == "codex":
            home = root / "codex-home"
            home.mkdir()
            (home / "config.toml").write_text("\n".join([
                f'model = "{model}"', 'model_provider = "relay"',
                *([] if reasoning_capture_path is not None else
                  ['model_reasoning_effort = "none"']),
                'model_auto_compact_token_limit = 1000000000', '',
                '[features]', 'plugins = false', '', '[model_providers.relay]',
                'name = "Relay"', f'base_url = "{relay_url}/v1"',
                'env_key = "RELAY_TEST_API_KEY"', 'wire_api = "responses"',
                'supports_websockets = false', '',
            ]))
        binary = _binary(harness) if harness != "mini-swe" else None
        stages = []
        second_start = 0
        second_summary_start = 0
        if harness == "mini-swe":
            stages, second_start, second_summary_start = _run_mini(
                root, relay_url, model, raw, manager)
        else:
            for index, prompt in enumerate((FIRST, SECOND)):
                if index:
                    second_start = len(raw)
                    second_summary_start = len(manager.summary_inputs)
                result = (_run_codex(binary, home, workspace, prompt, resume=bool(index))
                          if harness == "codex" else
                          _run_opencode_or_pi(harness, binary, root, relay_url,
                                              model, prompt, resume=bool(index),
                                              allow_tools=True))
                stages.append(_stage(result, harness))
                if result.returncode or '"stopReason":"error"' in result.stdout:
                    break
    normalized = [item.get("input", []) for item in raw]
    return _finish(harness, model, cache=cache, manager=manager, budget=budget,
                   strategy=strategy, raw=raw, normalized=normalized,
                   forwarded=strategy.forwarded, stages=stages,
                   second_start=second_start,
                   second_summary_start=second_summary_start,
                   calibration=calibration, protocol="responses")


def _run_mini(root: Path, relay_url: str, model: str,
              raw: list[dict[str, Any]], manager: _Management
              ) -> tuple[list[dict[str, Any]], int, int]:
    previous = {name: os.environ.get(name) for name in (
        "MSWEA_GLOBAL_CONFIG_DIR", "MSWEA_SILENT_STARTUP",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT")}
    os.environ.update({"MSWEA_GLOBAL_CONFIG_DIR": str(root / "mini-config"),
                       "MSWEA_SILENT_STARTUP": "1",
                       "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1"})
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_response_model import LitellmResponseModel
        from minisweagent.exceptions import Submitted, FormatError

        class TwoTurnEnvironment(LocalEnvironment):
            first_submission: str | None = None

            def _check_finished(self, output):
                try:
                    super()._check_finished(output)
                except Submitted as exc:
                    if self.first_submission is None:
                        self.first_submission = exc.messages[0]["extra"]["submission"]
                        return
                    raise

        task_model = LitellmResponseModel(
            model_name=f"openai/{model}", model_kwargs={
                "api_base": f"{relay_url}/v1", "api_key": "tenant-test"},
            cost_tracking="ignore_errors")
        environment = TwoTurnEnvironment(cwd=str(root))
        agent = DefaultAgent(task_model, environment,
            system_template=("You are a coding agent. Every model response must call the bash "
                             "tool. Read local files as needed. To finish an answer, call "
                             "bash with a command that prints "
                             "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on its first line, "
                             "followed by the answer. Do not modify files."),
            instance_template="{{task}}", step_limit=8, cost_limit=0)
        agent.add_messages(
            task_model.format_message(role="system", content=agent.config.system_template),
            task_model.format_message(role="user", content=FIRST))
        first_error = None
        while environment.first_submission is None and agent.n_calls < 5:
            try:
                agent.step()
            except FormatError as exc:
                first_error = str(exc)
                agent.add_messages(*exc.messages)
        first = {"exit_code": 0 if environment.first_submission else 1,
                 "answer_text": environment.first_submission or "",
                 "stdout": environment.first_submission or "",
                 "stderr": first_error or ""}
        second_start = len(raw)
        second_summary_start = len(manager.summary_inputs)
        if not environment.first_submission:
            return [first], second_start, second_summary_start
        agent.add_messages(task_model.format_message(role="user", content=SECOND))
        second = {"exit_code": 1, "stdout": "", "stderr": "step limit"}
        while agent.n_calls < 8:
            try:
                agent.step()
            except Submitted as exc:
                second = {"exit_code": 0,
                          "answer_text": exc.messages[0]["extra"]["submission"],
                          "stdout": exc.messages[0]["extra"]["submission"],
                          "stderr": "Submitted"}
                break
            except FormatError as exc:
                agent.add_messages(*exc.messages)
                second["stderr"] = str(exc)
        return [first, second], second_start, second_summary_start
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_native(harness: str, model: str, client: OpenAI) -> dict[str, Any]:
    cache = RecordingCache()
    strategy = _ObservedDiagnostic()
    manager = _Management(client, max_calls=6,
                          max_input_per_request=MAX_SUMMARY_INPUT,
                          max_total_input=MAX_TOTAL_SUMMARY_INPUT)
    budget = TaskBudget(client, max_requests=MAX_TASK_REQUESTS,
                        max_total_input=MAX_TOTAL_TASK_INPUT,
                        max_input_per_request=MAX_TASK_INPUT, max_output=512)
    raw: list[dict[str, Any]] = []
    normalized: list[list[dict[str, Any]]] = []
    calibration: dict[str, int] = {}
    ingress_errors: list[dict[str, Any]] = []
    task_requests: list[dict[str, Any]] = []
    bridge = (gemini_bridge(client, model, task_requests, budget)
              if harness == "gemini-cli" else
              messages_bridge(client, model, task_requests, budget))
    with _serve(bridge) as bridge_url:
        if harness == "gemini-cli":
            relay = create_app(strategy, ProxyConfig(
                upstream_base_url=bridge_url, checkpoint_mode="cache",
                gemini_enabled=True, management_model=model),
                checkpoint_cache=cache, management_responses=manager)

            def normalize(body: dict[str, Any]) -> dict[str, Any]:
                return GeminiInput(body, model).request

            def route_matches(path: str) -> bool:
                return path.endswith((":generateContent", ":streamGenerateContent"))
        else:
            relay = create_anthropic_adapter(
                task_base_url=bridge_url, management_responses=manager,
                manager_model=model, task_model=model,
                task_auth_token="local-bridge", cache=cache,
                compact_threshold=1_000_000_000)
            relay.state.engine.strategy = strategy

            def normalize(body: dict[str, Any]) -> dict[str, Any]:
                return to_relay_request(body, manager_model=model)

            def route_matches(path: str) -> bool:
                return path == "/v1/messages"

        class Capture(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if not route_matches(request.url.path):
                    return await call_next(request)
                if len(raw) >= int(os.getenv("RELAY_LOCAL_RELAY_LIMIT", str(MAX_RELAY_REQUESTS))):
                    return JSONResponse({"error": "relay request budget exceeded"},
                                        status_code=429)
                body = await request.json()
                try:
                    current = normalize(body)
                except (ValueError, KeyError) as exc:
                    ingress_errors.append({"error": str(exc),
                        "tool_shape": (_gemini_tool_shape(body)
                                       if harness == "gemini-cli" else [])})
                    return JSONResponse({"error": {"message": str(exc)}},
                                        status_code=400)
                if not calibration and raw and _tool_pair(current["input"]):
                    calibration.update(await run_in_threadpool(
                        calibrate_trigger_only, manager, normalize(raw[0]), current))
                    strategy.compact_threshold = calibration["compact_threshold"]
                raw.append(deepcopy(body))
                normalized.append(deepcopy(current["input"]))
                return await call_next(request)

        relay.add_middleware(Capture)
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace" if harness == "gemini-cli" else root
            workspace.mkdir(exist_ok=True)
            install(workspace)
            env = clean_env(root)
            binary = _binary(harness)
            if harness == "gemini-cli":
                config = root / "gemini-home" / ".gemini"
                config.mkdir(parents=True)
                shutil.copyfile(Path(__file__).resolve().parents[2] / "tests" /
                                "fixtures" / "gemini_proxy_settings.json",
                                config / "settings.json")
                settings = json.loads((config / "settings.json").read_text())
                settings["tools"] = {"exclude": ["update_topic"]}
                (config / "settings.json").write_text(json.dumps(settings))
                env.update({"GEMINI_CLI_HOME": str(config.parent),
                            "GEMINI_API_KEY": "local-tenant",
                            "GEMINI_CLI_TRUST_WORKSPACE": "true",
                            "GOOGLE_GEMINI_BASE_URL": relay_url,
                            "GOOGLE_GENAI_API_VERSION": "v1beta"})
            else:
                env.update({"ANTHROPIC_BASE_URL": relay_url,
                            "ANTHROPIC_API_KEY": "local-tenant",
                            "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
                            "CLAUDE_CONFIG_DIR": str(root / "claude-config"),
                            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                            "CLAUDE_CODE_MAX_RETRIES": "0", "DISABLE_COMPACT": "1",
                            "DISABLE_TELEMETRY": "1"})
            stages = []
            second_start = 0
            second_summary_start = 0
            session_id = None
            for index, prompt in enumerate((FIRST, SECOND)):
                if index:
                    second_start = len(raw)
                    second_summary_start = len(manager.summary_inputs)
                if harness == "gemini-cli":
                    args = [binary, "-p", prompt, "-m", "gemini-3.8-flash",
                            "--output-format", "json"]
                    if index:
                        args += ["--resume", "latest"]
                else:
                    args = [binary, "--print", "--output-format", "json",
                            "--model", model, "--tools", "Read,Glob,Grep",
                            "--allowedTools", "Read,Glob,Grep",
                            "--permission-mode", "dontAsk",
                            "--setting-sources", ""]
                    if index and session_id:
                        args += ["--resume", session_id]
                    elif index:
                        args += ["--continue"]
                    args.append(prompt)
                result = subprocess.run(args, cwd=workspace, env=env,
                                        stdin=subprocess.DEVNULL, capture_output=True,
                                        text=True, timeout=150, check=False)
                stages.append(_stage(result, harness))
                if harness == "claude-code" and not index:
                    try:
                        session_id = json.loads(result.stdout).get("session_id")
                    except (TypeError, ValueError):
                        pass
                if result.returncode:
                    break
    forwarded = ([GeminiInput(body, model).request["input"]
                  for body in task_requests] if harness == "gemini-cli" else
                 [to_relay_request(body, manager_model=model)["input"]
                  for body in task_requests])
    record = _finish(harness, model, cache=cache, manager=manager, budget=budget,
                   strategy=strategy, raw=raw, normalized=normalized,
                   forwarded=forwarded, stages=stages,
                   second_start=second_start,
                   second_summary_start=second_summary_start,
                   calibration=calibration,
                   protocol="gemini" if harness == "gemini-cli" else "anthropic")
    record["ingress_errors"] = ingress_errors[:2]
    return record


def run(harness: str, model: str,
        reasoning_capture_path: Path | None = None) -> dict[str, Any]:
    if os.getenv("RELAY_LIVE_TESTS") != "1":
        raise ValueError("set RELAY_LIVE_TESTS=1 before a billable run")
    route = route_for(model)
    with OpenAI(base_url=route.base_url, api_key=route.api_key,
                timeout=90, max_retries=0) as client:
        if harness in {"codex", "opencode", "pi", "mini-swe"}:
            return _run_responses(harness, model, client, route,
                                  reasoning_capture_path)
        if harness in {"gemini-cli", "claude-code"}:
            return _run_native(harness, model, client)
    raise ValueError(f"{harness}: no verified live route")
