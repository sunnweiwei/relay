"""Account-enforced billing must not disable the experiment's other limits."""

import json

import httpx
import pytest

from evals.pilot.run import QuotaGate, check_upstream_request, observe_proxy_error
from relay.core.budget import BudgetExceeded, InputBudget


class ForbiddenAccountRead:
    def call(self, *args, **kwargs):
        raise AssertionError("Account usage must not be queried")


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAY_PILOT_QUOTA_MODE", "account_enforced")
    monkeypatch.setenv("RELAY_PILOT_ACCOUNT_NO_OVERAGE_CONFIRMED", "1")
    monkeypatch.setenv("RELAY_PILOT_EXPECTED_EFFORT", "medium")
    monkeypatch.setenv("RELAY_PILOT_ALL_CALL_BUDGET", "1")
    return QuotaGate(ForbiddenAccountRead(), tmp_path / "quota.jsonl", 100, 0, 23560)


def test_no_account_calls_even_on_start_and_completion(gate):
    assert gate.check(force=True, starting=True)
    assert gate.check()
    assert gate.check(force=True)
    assert gate.used is None
    assert not gate.path.exists()
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses",
                           json={"model": "gpt-6-luna", "reasoning": {"effort": "medium"}})
    check_upstream_request(request, gate, "gpt-6-luna")
    gate.reason = "model_request_limit"
    assert not gate.check()


def test_no_polling_requires_user_confirmed_account_control(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAY_PILOT_QUOTA_MODE", "account_enforced")
    monkeypatch.delenv("RELAY_PILOT_ACCOUNT_NO_OVERAGE_CONFIRMED", raising=False)
    with pytest.raises(ValueError, match="confirmed no-overage"):
        QuotaGate(ForbiddenAccountRead(), tmp_path / "quota.jsonl", 100, 0)


@pytest.mark.parametrize("model,effort,host,match", [
    ("gpt-6.1-sol", "medium", "chatgpt.com", "Unexpected model"),
    ("gpt-6-luna", "low", "chatgpt.com", "model_reasoning_effort_mismatch"),
    ("gpt-6-luna", "medium", "api.openai.com", "plan endpoint"),
])
def test_model_effort_endpoint_guards_remain(gate, model, effort, host, match):
    request = httpx.Request("POST", f"https://{host}/responses",
                           json={"model": model, "reasoning": {"effort": effort}})
    with pytest.raises(RuntimeError, match=match):
        check_upstream_request(request, gate, "gpt-6-luna")


def test_input_recovery_and_actual_usage_guards_remain(gate):
    gate.input_budget = InputBudget(100, dispatch_slack=10)
    request = httpx.Request("POST", "https://chatgpt.com/responses", json={
        "model": "gpt-6-luna", "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": "large " * 500}],
    })
    with pytest.raises(BudgetExceeded):
        check_upstream_request(request, gate, "gpt-6-luna")
    assert gate.reason is None
    gate.observe_auxiliary_tokens(111)
    assert gate.reason == "provider_input_budget_mismatch"
    assert not gate.check()


def test_provider_account_rejection_stops_without_polling(gate):
    observe_proxy_error(gate, json.dumps({"error": {"code": "usage_limit_reached"}}).encode())
    assert gate.reason == "account_limit_reached"
    assert not gate.check()


def test_relocated_runtime_keeps_original_evaluator_hidden(tmp_path, monkeypatch):
    from evals.pilot.run import config
    home, work = tmp_path / "codex", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    previous = tmp_path / "old-evaluator"
    monkeypatch.setenv("RELAY_PILOT_EXTRA_DENY_PATHS", json.dumps([str(previous)]))
    config(home, work, tmp_path / "private", [])
    assert f'{json.dumps(str(previous))} = "deny"' in (home / "config.toml").read_text()


def test_runtime_inside_hidden_root_needs_no_second_denied_mount(tmp_path, monkeypatch):
    import evals.pilot.run as runner
    hidden = tmp_path / "private"
    nested_runtime = hidden / "runtime"
    monkeypatch.setattr(runner, "REPO", nested_runtime)
    home, work = hidden / "codex", hidden / "work"
    home.mkdir(parents=True)
    work.mkdir()
    runner.config(home, work, hidden, [])
    text = (home / "config.toml").read_text()
    assert f'{json.dumps(str(hidden))} = "deny"' in text
    assert f'{json.dumps(str(nested_runtime))} = "deny"' not in text


def test_nested_denial_survives_a_readable_exception(tmp_path, monkeypatch):
    import evals.pilot.run as runner
    hidden = tmp_path / "private"
    readable = hidden / "public"
    secret = readable / "secret"
    home, work = hidden / "codex", hidden / "work"
    home.mkdir(parents=True)
    work.mkdir()
    runner.config(home, work, hidden, [readable], denied=[secret])
    text = (home / "config.toml").read_text()
    assert f'{json.dumps(str(secret))} = "deny"' in text


def test_runner_input_counter_option_defaults_and_accepts_media(tmp_path, monkeypatch):
    import sys
    import evals.pilot.run as runner
    from relay.core.budget import MEDIA_INPUT_BUDGET_COUNTER

    captured = []
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args) or 0)
    common = ["--root", str(tmp_path), "--out", str(tmp_path / "out"),
              "--prompt", str(tmp_path / "prompt"), "--strategy", "compaction"]
    monkeypatch.setattr(sys, "argv", ["run.py", *common])
    assert runner.main() == 0
    assert captured[-1].input_budget_counter is None

    monkeypatch.setattr(sys, "argv", ["run.py", *common,
                                      "--input-budget-counter", MEDIA_INPUT_BUDGET_COUNTER])
    assert runner.main() == 0
    assert captured[-1].input_budget_counter == MEDIA_INPUT_BUDGET_COUNTER


def test_runner_rejects_explicit_input_counter_without_all_call_budget(monkeypatch):
    import argparse
    import evals.pilot.run as runner
    from relay.core.budget import MEDIA_INPUT_BUDGET_COUNTER

    monkeypatch.delenv("RELAY_PILOT_ALL_CALL_BUDGET", raising=False)
    with pytest.raises(ValueError, match="require RELAY_PILOT_ALL_CALL_BUDGET=1"):
        runner._run(argparse.Namespace(input_budget_counter=MEDIA_INPUT_BUDGET_COUNTER))
