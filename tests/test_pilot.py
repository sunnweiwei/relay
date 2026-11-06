"""Stopping conditions for the optional benchmark pilot runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.pilot.run import QuotaGate
from evals.pilot.collect import collect, normalized


class Account:
    def __init__(self, *, used: float = 58, ordinary: bool = True, balance: str = "10") -> None:
        self.used, self.ordinary, self.balance = used, ordinary, balance

    def call(self, method: str) -> dict[str, Any]:
        assert method == "account/rateLimits/read"
        return {"ordinaryUsageAllowed": self.ordinary, "rateLimits": {
            "primary": {"usedPercent": self.used}, "secondary": None,
            "credits": {"balance": self.balance}}}


def test_admission_stop_is_scoped_to_one_experiment_tree(tmp_path: Path) -> None:
    from evals.pilot.run import closed_admission

    pilot = tmp_path / "pilot"
    pilot.mkdir()
    marker = pilot / ".stop-new-workers"
    marker.write_text("Preserve quota for random tasks")
    assert closed_admission(pilot / "model/method", tmp_path) == marker
    assert closed_admission(tmp_path / "sample/model/method", tmp_path) is None


def test_upstream_guard_accepts_codex_compression_and_rejects_fallback(tmp_path: Path) -> None:
    import json
    import httpx
    import pytest
    import zstandard
    from evals.pilot.run import check_upstream_request

    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 10, 0)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses",
                            content=zstandard.ZstdCompressor().compress(json.dumps({"model": "gpt-6.1-sol"}).encode()),
                            headers={"content-encoding": "zstd"})
    check_upstream_request(request, gate, "gpt-6.1-sol")
    with pytest.raises(RuntimeError, match="Unexpected model"):
        check_upstream_request(request, gate, "gpt-6-luna")
    with pytest.raises(RuntimeError, match="plan endpoint"):
        check_upstream_request(httpx.Request("POST", "https://api.openai.com/v1/responses"), gate, "gpt-6.1-sol")


def test_upstream_guard_stops_strategy_calls_without_plan_quota(tmp_path: Path) -> None:
    import httpx
    import pytest
    from evals.pilot.run import check_upstream_request

    gate = QuotaGate(Account(ordinary=False), tmp_path / "quota.jsonl", 10, 0)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses", json={"model": "gpt-6-luna"})
    with pytest.raises(RuntimeError, match="ordinary_plan_usage_not_allowed"):
        check_upstream_request(request, gate, "gpt-6-luna")


def test_refuse_credit_only_usage(tmp_path: Path) -> None:
    gate = QuotaGate(Account(ordinary=False), tmp_path / "quota.jsonl", 100, 0)
    assert not gate.check(force=True)
    assert gate.reason == "ordinary_plan_usage_not_allowed"


def test_provider_capacity_failure_is_not_confused_with_retry_warning(tmp_path: Path) -> None:
    import json
    from evals.pilot.run import harness_failure

    events = [{"type": "error", "message": "Falling back from WebSockets to HTTPS transport"},
              {"type": "turn.failed", "error": {"message": "Selected model is at capacity."}}]
    (tmp_path / "codex.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    assert harness_failure(tmp_path) == "Selected model is at capacity."


def test_preserve_quota_reserve_before_starting(tmp_path: Path) -> None:
    gate = QuotaGate(Account(used=80), tmp_path / "quota.jsonl", 100, 0)
    assert not gate.check(force=True, starting=True)
    assert gate.reason == "plan_quota_reserve_reached"


def test_actual_context_overflow_prevents_further_inference(tmp_path: Path) -> None:
    import httpx
    import pytest
    from evals.pilot.run import check_upstream_request

    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, 0, input_limit=23560)
    assert gate.check(force=True)
    gate.observe_input_tokens(None)
    gate.observe_input_tokens(23560)
    assert gate.reason is None
    gate.observe_input_tokens(23561)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses",
                            json={"model": "gpt-6-luna"})
    with pytest.raises(RuntimeError, match="context_budget_exceeded"):
        check_upstream_request(request, gate, "gpt-6-luna")


def test_context_overflow_does_not_hide_a_prior_quota_stop(tmp_path: Path) -> None:
    gate = QuotaGate(Account(used=90), tmp_path / "quota.jsonl", 100, 0, input_limit=32000)
    assert not gate.check(force=True)
    gate.observe_input_tokens(33000)
    assert gate.reason == "plan_quota_reserve_reached"


def test_stop_on_decreased_credit_balance(tmp_path: Path) -> None:
    account = Account()
    gate = QuotaGate(account, tmp_path / "quota.jsonl", 100, 0)
    assert gate.check(force=True)
    account.balance = "9.99"
    assert not gate.check(force=True)
    assert gate.reason == "credit_balance_decreased"


def test_quota_history_keeps_window_identity_without_private_account_data(tmp_path: Path) -> None:
    import json

    class WindowAccount(Account):
        def call(self, method: str) -> dict[str, Any]:
            result = super().call(method)
            result["rateLimits"]["primary"].update(
                windowDurationMins=10080, resetsAt=1800000000, private_field="not-for-logs")
            return result

    gate = QuotaGate(WindowAccount(balance="123.456"), tmp_path / "quota.jsonl", 100, 0)
    assert gate.check(force=True)
    content = (tmp_path / "quota.jsonl").read_text()
    window = json.loads(content)["windows"]["primary"]
    assert window["windowDurationMins"] == 10080 and window["resetsAt"] == 1800000000
    assert "123.456" not in content and "private_field" not in content


def test_stop_without_account_metadata(tmp_path: Path) -> None:
    class Unavailable:
        def call(self, method: str) -> dict[str, Any]:
            raise TimeoutError("offline")

    gate = QuotaGate(Unavailable(), tmp_path / "quota.jsonl", 100, 0)
    assert not gate.check(force=True)
    assert gate.reason == "quota_check_failed:TimeoutError"


def test_usage_subsets_are_not_double_counted() -> None:
    usage = normalized({"input_tokens": 100, "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 80},
                        "output_tokens_details": {"reasoning_tokens": 12}})
    assert usage["total_tokens"] == 120
    assert usage["cached_input_tokens"] == 80
    assert usage["reasoning_output_tokens"] == 12


def test_relay_summary_usage_is_added_to_worker_usage(tmp_path: Path) -> None:
    import json

    (tmp_path / "relay").mkdir()
    sessions = tmp_path / "codex/sessions"
    sessions.mkdir(parents=True)
    (tmp_path / "run.json").write_text(json.dumps({"window": 32000, "status": "completed"}))
    (sessions / "rollout-test.jsonl").write_text(json.dumps({"payload": {
        "type": "token_count", "info": {"total_token_usage": {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}}}) + "\n")
    (tmp_path / "relay/trace.jsonl").write_text(json.dumps({
        "summary_request": {"model": "gpt-6-luna"}, "summary_status": 200,
        "answer": {"usage": {"input_tokens": 40, "output_tokens": 5,
                               "input_tokens_details": {"cached_tokens": 30}}}}) + "\n")
    result = collect(tmp_path)
    assert result["combined_usage"]["total_tokens"] == 165
    assert result["combined_usage"]["input_tokens"] == 140
    assert result["summary_usage"]["cached_input_tokens"] == 30
    assert result["summary_usage_missing_attempts"] == 0
