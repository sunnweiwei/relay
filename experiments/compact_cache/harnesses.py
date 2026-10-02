"""Harness-specific launch, resume and wire-normalization code."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from importlib.util import find_spec

import pytest
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from relay import Compact
from relay.gemini import GeminiInput
from relay.proxy import ProxyConfig, create_app
from tests.harness_e2e_support import clean_env, configure, run_turn
from tests.test_codex_e2e import _run_codex, _serve, _write_codex_config
from experiments.compact_cache.case import FIRST, SECOND
from experiments.compact_cache.checks import (
    HARNESSES, THRESHOLD, RecordingCache, StableCompactUpstream,
    assert_contract, assert_responses_wire_contract, make_responses_relay,
)

def run_codex_contract():
    binary = shutil.which("codex")
    if not binary:
        pytest.skip("Codex CLI not installed")
    upstream = StableCompactUpstream()
    cache = RecordingCache()
    with _serve(upstream.app) as upstream_url:
        relay, raw = make_responses_relay(upstream_url, cache)
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codex_home, workspace = root / "codex-home", root / "workspace"
            codex_home.mkdir()
            workspace.mkdir()
            _write_codex_config(codex_home, relay_url)
            first = _run_codex(binary, codex_home, workspace, [
                "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
                "--cd", str(workspace), FIRST,
            ])
            second = _run_codex(binary, codex_home, workspace, [
                "exec", "resume", "--last", "--json", "--skip-git-repo-check",
                SECOND,
            ])
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert "RELAY_CODEX_TURN_2_OK" in second.stdout
    assert_responses_wire_contract(raw, upstream.main_requests)
    return cache, upstream.summary_requests, [body["input"] for body in raw], [
        body["input"] for body in upstream.main_requests
    ], "Turn 2"


def run_opencode_or_pi_contract(harness: str):
    local_pi = Path(__file__).resolve().parent / ".tools/node_modules/.bin/pi"
    binary = (os.environ.get(f"RELAY_{harness.upper()}_BIN")
              or (str(local_pi) if harness == "pi" and local_pi.is_file() else None)
              or shutil.which(harness))
    if not binary:
        pytest.skip(f"{harness} CLI not installed")
    upstream = StableCompactUpstream()
    cache = RecordingCache()
    with _serve(upstream.app) as upstream_url:
        relay, raw = make_responses_relay(upstream_url, cache)
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = configure(harness, root, relay_url)
            for turn in range(1, 3):
                prompt = FIRST if turn == 1 else SECOND
                if harness == "opencode":
                    args = [binary, "run", "--pure", "--format", "json",
                            "--model", "openai/relay-test-model"]
                    if turn > 1:
                        args.append("--continue")
                else:
                    args = [binary, "--mode", "json", "--print", "--provider", "relay",
                            "--model", "relay-test-model", "--session",
                            str(root / "session.jsonl"), "--no-extensions", "--no-skills",
                            "--no-prompt-templates", "--no-tools"]
                result = subprocess.run([*args, prompt], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=45)
                assert result.returncode == 0, result.stdout + result.stderr
                assert f"RELAY_CODEX_TURN_{turn}_OK" in result.stdout
    assert_responses_wire_contract(raw, upstream.main_requests)
    return cache, upstream.summary_requests, [body["input"] for body in raw], [
        body["input"] for body in upstream.main_requests
    ], "Turn 2"


def run_gemini_contract():
    local_gemini = Path(__file__).resolve().parent / ".tools/node_modules/.bin/gemini"
    binary = (os.environ.get("RELAY_GEMINI_BIN")
              or (str(local_gemini) if local_gemini.is_file() else None)
              or shutil.which("gemini"))
    if not binary:
        pytest.skip("Gemini CLI not installed")
    captured: list[dict[str, Any]] = []

    async def native(request):
        body = await request.json()
        captured.append(body)
        turn = len(captured)
        payload = {"candidates": [{
            "content": {"role": "model", "parts": [{"text": f"RELAY_GEMINI_TURN_{turn}_OK"}]},
            "finishReason": "STOP", "index": 0,
        }], "usageMetadata": {
            "promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25,
        }}
        return Response("data: " + json.dumps(payload) + "\n\n", media_type="text/event-stream")

    native_app = Starlette(routes=[Route("/{path:path}", native, methods=["POST"])])
    manager = StableCompactUpstream()
    cache = RecordingCache()
    with _serve(native_app) as native_url, _serve(manager.app) as management_url:
        relay = create_app(
            Compact(compact_threshold=THRESHOLD),
            ProxyConfig(
                upstream_base_url=native_url, checkpoint_mode="cache",
                gemini_enabled=True, management_base_url=f"{management_url}/v1",
                management_api_key="fake-manager", management_model="relay-test-model",
            ),
            checkpoint_cache=cache,
        )
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            config = root / "gemini-home" / ".gemini"
            config.mkdir(parents=True)
            shutil.copyfile(Path(__file__).resolve().parents[2] / "tests" / "fixtures" /
                            "gemini_proxy_settings.json", config / "settings.json")
            env = clean_env(root)
            env.update({
                "GEMINI_CLI_HOME": str(config.parent), "GEMINI_API_KEY": "fake-local-key",
                "GEMINI_CLI_TRUST_WORKSPACE": "true", "GOOGLE_GEMINI_BASE_URL": relay_url,
                "GOOGLE_GENAI_API_VERSION": "v1beta",
            })
            for turn in range(1, 3):
                args = [binary, "-p", FIRST if turn == 1 else SECOND,
                        "-m", "gemini-3.8-flash", "--output-format", "json"]
                if turn > 1:
                    args += ["--resume", "latest"]
                result = subprocess.run(args, cwd=workspace, env=env, capture_output=True,
                                        text=True, timeout=60)
                assert result.returncode == 0, result.stdout + result.stderr
                assert f"RELAY_GEMINI_TURN_{turn}_OK" in result.stdout
    raw_inputs = [lookup.trajectory for lookup in cache.lookups]
    forwarded_inputs = [GeminiInput(body, "relay-test-model").request["input"] for body in captured]
    return cache, manager.summary_requests, raw_inputs, forwarded_inputs, "Turn 2"


def run_mini_swe_contract():
    if not find_spec("minisweagent"):
        pytest.skip("mini-swe-agent package not installed")
    with tempfile.TemporaryDirectory() as directory:
        os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(Path(directory) / "mini-config")
        os.environ["MSWEA_SILENT_STARTUP"] = "1"
        os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_response_model import LitellmResponseModel

        upstream = StableCompactUpstream(mini_mode=True)
        cache = RecordingCache()
        with _serve(upstream.app) as upstream_url:
            relay, raw = make_responses_relay(upstream_url, cache)
            with _serve(relay) as relay_url:
                model = LitellmResponseModel(
                    model_name="openai/relay-test-model",
                    model_kwargs={"api_base": f"{relay_url}/v1", "api_key": "tenant-test"},
                    cost_tracking="ignore_errors",
                )
                agent = DefaultAgent(
                    model, LocalEnvironment(cwd=directory),
                    system_template="You are a coding agent. Always call bash.",
                    instance_template="Solve: {{task}}", step_limit=5, cost_limit=0,
                )
                result = agent.run("Preserve BLUE-ORCHID-731 and complete the task")
    assert result["exit_status"] == "Submitted", result
    assert "relay-mini-ok" in result["submission"]
    assert_responses_wire_contract(raw, upstream.main_requests, expected_stream=False)
    return cache, upstream.summary_requests, [body["input"] for body in raw], [
        body["input"] for body in upstream.main_requests
    ], "BLUE-ORCHID-731"


def run_responses_tool_contract(harness: str):
    binary = shutil.which(harness)
    if not binary:
        pytest.skip(f"{harness} CLI not installed")
    upstream = (
        StableCompactUpstream(tool_first=True)
        if harness == "codex" else StableCompactUpstream(tool_name="bash")
    )
    cache = RecordingCache()
    with _serve(upstream.app) as upstream_url:
        relay, raw = make_responses_relay(upstream_url, cache)
        with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if harness == "codex":
                codex_home, workspace = root / "codex-home", root / "workspace"
                codex_home.mkdir()
                workspace.mkdir()
                _write_codex_config(codex_home, relay_url)
                result = _run_codex(binary, codex_home, workspace, [
                    "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
                    "--cd", str(workspace),
                    "Run the requested model tool, then return its final result.",
                ])
            else:
                env = configure(harness, root, relay_url)
                result = run_turn(harness, binary, root, env, 1, tools=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RELAY_CODEX_TURN_2_OK" in result.stdout
    assert len(upstream.main_requests) == 2
    assert_responses_wire_contract(raw, upstream.main_requests)
    assert_contract(
        cache, upstream.summary_requests,
        [body["input"] for body in raw],
        [body["input"] for body in upstream.main_requests],
        latest_marker="Run the requested model tool" if harness == "codex" else "Turn 1",
    )
    replayed = upstream.main_requests[1]["input"]
    calls = [item for item in replayed if item.get("type") == "function_call"]
    outputs = [item for item in replayed if item.get("type") == "function_call_output"]
    assert len(calls) == len(outputs) == 1
    assert calls[0]["call_id"] == outputs[0]["call_id"]
    assert calls[0]["name"] == ("exec_command" if harness == "codex" else "bash")


RUNNERS = {
    "gemini-cli": run_gemini_contract,
    "codex": run_codex_contract,
    "mini-swe": run_mini_swe_contract,
    "opencode": lambda: run_opencode_or_pi_contract("opencode"),
    "pi": lambda: run_opencode_or_pi_contract("pi"),
}

from experiments.compact_cache.claude_harness import run_claude_contract

RUNNERS["claude-code"] = run_claude_contract
