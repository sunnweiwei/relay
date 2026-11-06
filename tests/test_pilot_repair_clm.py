"""CLM repair is bounded, preserves quota stops, and distinguishes method limits."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.pilot import arc_suite, repair_clm


def budget_worker(**changes: Any) -> dict[str, Any]:
    return {"model": "gpt-6-luna", "strategy": "clm", "status": "stopped",
            "stop_reason": "context_budget_exceeded", "exit_code": -15,
            "requests_over_context_budget": 1, "strategy_errors": 0,
            "sandbox_execution_errors": 0, "protocol_valid": False, **changes}


@pytest.mark.parametrize("changes,expected", [
    ({}, True), ({"stop_reason": "model_request_limit"}, True),
    ({"strategy_errors": 1}, False), ({"sandbox_execution_errors": 1}, False),
    ({"status": "infrastructure_failed"}, False),
    ({"stop_reason": "plan_quota_reserve_reached"}, False),
    ({"error": "provider_capacity"}, False), ({"stop_reason": None}, False),
])
def test_only_measured_context_violation_gets_method_exception(changes: dict[str, Any], expected: bool) -> None:
    assert repair_clm.context_budget_only(budget_worker(**changes)) is expected


def setup_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, list[str]]:
    pilot = tmp_path / "matrix"
    old = pilot / "browse/gpt-6-luna/clm"
    old.mkdir(parents=True)
    (old / "run.json").write_text(json.dumps(budget_worker()))
    args = argparse.Namespace(benchmark="browse", scratch=tmp_path, pilot_root=pilot,
        python=Path("python"), out=pilot / "repairs/test", wait_session=["pilot"], max_hours=24)
    calls: list[str] = []
    monkeypatch.setattr(repair_clm, "live_queue_sessions", lambda names: [])

    class RPC:
        def __init__(self, env: Any) -> None:
            pass

        def close(self) -> None:
            pass

    class Gate:
        reason = None

        def __init__(self, *args: Any) -> None:
            pass

        def check(self, **kwargs: Any) -> bool:
            assert kwargs == {"force": True, "starting": True}
            return True

    monkeypatch.setattr(repair_clm, "CodexRPC", RPC)
    monkeypatch.setattr(repair_clm, "QuotaGate", Gate)
    monkeypatch.setattr(repair_clm, "repair_plan", lambda args, model, old, fix_hash: {
        "model": model, "fix_hash": fix_hash, "output": str(old / "attempt-002"), "command": ["native-suite"]})

    def execute(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
        calls.append(entry["model"])
        repair_clm.write_json(Path(entry["output"]) / "run.json", budget_worker())
        return budget_worker()

    monkeypatch.setattr(repair_clm, "execute_repair", execute)
    return args, calls


def test_residual_budget_limit_is_audited_and_never_repaired_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, calls = setup_queue(tmp_path, monkeypatch)
    assert repair_clm.run_queue(args) == 0
    state = repair_clm.read_json(args.out / "repair-state.json")
    entry = state["attempts"]["gpt-6-luna"]
    assert entry["status"] == "budget_limitation_after_repair"
    audit = repair_clm.read_json(Path(entry["output"]) / "evaluation-audit.json")
    assert audit["classification"] == "context_budget_exceeded"
    assert audit["benchmark_score"] is None
    assert repair_clm.run_queue(args) == 0
    args.out = args.pilot_root / "repairs/different-output"
    assert repair_clm.run_queue(args) == 0
    assert calls == ["gpt-6-luna"]


def test_saved_quota_stop_prevents_even_the_launch_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, calls = setup_queue(tmp_path, monkeypatch)
    repair_clm.write_json(args.pilot_root / "arc/other/worker/run.json", {
        "stop_reason": "credit_balance_decreased"})
    monkeypatch.setattr(repair_clm, "CodexRPC", lambda env: pytest.fail("Must not open account after a saved stop"))
    assert repair_clm.run_queue(args) == 2
    assert calls == []
    assert repair_clm.read_json(args.out / "repair-state.json")["status"] == "quota_stopped_no_repair"


def test_current_start_quota_refusal_does_not_consume_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, calls = setup_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(repair_clm.QuotaGate, "check", lambda *a, **k: False)
    monkeypatch.setattr(repair_clm.QuotaGate, "reason", "plan_quota_reserve_reached")
    assert repair_clm.run_queue(args) == 2
    assert calls == []
    assert not list((args.pilot_root / "repairs/clm-attempts").rglob("*.json"))


def test_wait_deadline_survives_driver_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, calls = setup_queue(tmp_path, monkeypatch)
    repair_clm.write_json(args.out / "repair-state.json", {"benchmark": "browse",
        "fix": repair_clm.fix_fingerprint(), "attempts": {}, "started_unix": 0})
    monkeypatch.setattr(repair_clm, "live_queue_sessions", lambda names: names)
    assert repair_clm.run_queue(args) == 2
    assert calls == []
    assert repair_clm.read_json(args.out / "repair-state.json")["status"] == "wait_timeout"


@pytest.mark.parametrize("defect,expected", [
    (None, "context_budget_exceeded"), ("strategy", "infrastructure_failed"),
    ("observation", "infrastructure_failed"),
])
def test_arc_exception_does_not_hide_other_defects(tmp_path: Path, defect: str | None, expected: str) -> None:
    worker = budget_worker(strategy_errors=int(defect == "strategy"))
    repair_clm.write_json(tmp_path / "worker/run.json", worker)
    if defect == "observation":
        service = tmp_path / "service"
        service.mkdir()
        (service / "arc_trace.jsonl").write_text(json.dumps({"action_count": 3,
            "observation": {"state": "NOT_FINISHED", "frame": [[[1]]], "frame_ascii": None}}) + "\n")
    summary = arc_suite.apply_worker_audit(tmp_path, {"worker_status": "stopped", "scored": True})
    assert summary["worker_status"] == expected
    assert summary["scored"] is False


@pytest.mark.parametrize("strategy_errors,expected_calls", [(0, ["clm", "compaction"]), (1, ["clm"])])
def test_arc_queue_continues_only_for_pure_context_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy_errors: int, expected_calls: list[str],
) -> None:
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc/frozen_prompt.txt").write_text("Frozen ARC prompt")
    output = tmp_path / "matrix/arc"
    calls = []

    def run(arm: str, *args: Any, model: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(arm)
        if arm == "compaction":
            return {"worker_status": "completed", "scored": True}
        slot = output / model / arm
        repair_clm.write_json(slot / "worker/run.json", budget_worker(strategy_errors=strategy_errors))
        return arc_suite.apply_worker_audit(slot, {"worker_status": "stopped", "scored": True})

    monkeypatch.setattr(arc_suite, "run_arm", run)
    monkeypatch.setattr(sys, "argv", ["arc_suite", "--scratch", str(tmp_path),
        "--python", sys.executable, "--out-root", str(output), "--models", "gpt-6-luna",
        "--arms", "clm", "compaction"])
    assert arc_suite.main() == (0 if not strategy_errors else 2)
    assert calls == expected_calls
