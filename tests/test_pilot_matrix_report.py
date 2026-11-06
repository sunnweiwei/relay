"""Reporting must keep missing results and failed launch attempts unscored."""

import json
from pathlib import Path

from evals.pilot.matrix_report import collect_matrix


def test_empty_matrix_contains_all_planned_cells_without_scores(tmp_path: Path) -> None:
    report = collect_matrix(tmp_path)
    assert report["planned_cells"] == 66
    assert report["status_counts"] == {"not_started": 66}
    assert all(row["score"] is None for row in report["rows"])


def test_verifier_timeout_and_unstarted_game_do_not_become_zero(tmp_path: Path) -> None:
    science = tmp_path / "science/gpt-6-luna/clm/attempt-001"
    science.mkdir(parents=True)
    (science / "result.json").write_text(json.dumps({"status": "timeout", "verifier": {
        "status": "timeout", "timeout": True, "reward": None, "phase": "candidate"}}))
    arc = tmp_path / "arc/gpt-6.1-sol/rlm"
    (arc / "worker").mkdir(parents=True)
    (arc / "worker/run.json").write_text(json.dumps({"status": "preflight_failed", "window": 32000}))
    (arc / "summary.json").write_text(json.dumps({"scored": False}))
    (arc / "official_scorecard.json").write_text(json.dumps({"score": 0}))
    report = collect_matrix(tmp_path)
    active = [row for row in report["rows"] if row["status"] != "not_started"]
    assert {row["status"] for row in active} == {"timeout", "preflight_failed"}
    assert all(row["score"] is None for row in active)


def test_infrastructure_audit_excludes_an_earlier_diagnostic_score(tmp_path: Path) -> None:
    slot = tmp_path / "browse/gpt-6.1-sol/compaction"
    slot.mkdir(parents=True)
    (slot / "browse-grade.json").write_text(json.dumps({
        "strict_answer_score": 0, "searches": 0, "fetches": 0}))
    (slot / "evaluation-audit.json").write_text(json.dumps({
        "valid": False, "classification": "infrastructure_failed", "reason": "missing_shell_runtime"}))
    report = collect_matrix(tmp_path)
    row = next(row for row in report["rows"] if row["path"] == str(slot))
    assert row["status"] == "infrastructure_failed"
    assert row["score"] is None


def test_science_launcher_log_is_not_treated_as_an_attempt(tmp_path: Path) -> None:
    slot = tmp_path / "science/gpt-6.1-sol/compaction"
    attempt = slot / "attempt-002"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({"status": "worker_running"}))
    (slot / "attempt-002-launcher.log").write_text("worker started\n")
    report = collect_matrix(tmp_path)
    row = next(row for row in report["rows"] if row["benchmark"] == "Terminal-Bench-Science"
               and row["model"] == "gpt-6.1-sol" and row["strategy"] == "compaction")
    assert row["status"] == "running"
    assert row["attempts"] == 1
    assert row["path"] == str(attempt)


def test_unforced_failure_stays_unscored_without_manual_audit(tmp_path: Path) -> None:
    slot = tmp_path / "browse/gpt-6.1-sol/compaction"
    slot.mkdir(parents=True)
    (slot / "run.json").write_text(json.dumps({"status": "stopped", "exit_code": 1,
                                               "stop_reason": None, "window": 23560}))
    (slot / "browse-grade.json").write_text(json.dumps({"strict_answer_score": 0}))
    report = collect_matrix(tmp_path)
    row = next(row for row in report["rows"] if row["path"] == str(slot))
    assert row["status"] == "infrastructure_failed"
    assert row["score"] is None
    assert row["protocol_valid"] is False


def test_science_retry_takes_precedence_over_invalid_old_grade(tmp_path: Path) -> None:
    slot = tmp_path / "science/gpt-6.1-sol/compaction"
    old = slot / "attempt-002"
    old.mkdir(parents=True)
    (old / "result.json").write_text(json.dumps({"status": "completed_failed", "verifier": {"reward": 0}}))
    (old / "evaluation-audit.json").write_text(json.dumps({"valid": False, "reason": "provider_capacity"}))
    new = slot / "attempt-003"
    new.mkdir()
    (new / "result.json").write_text(json.dumps({"status": "worker_running"}))
    row = next(row for row in collect_matrix(tmp_path)["rows"] if row["benchmark"] == "Terminal-Bench-Science"
               and row["model"] == "gpt-6.1-sol" and row["strategy"] == "compaction")
    assert row["status"] == "running"
    assert row["score"] is None
    assert row["path"] == str(new)


def test_invalid_science_attempt_removes_stale_verifier_score_and_detail(tmp_path: Path) -> None:
    for reason in ("protocol", "harness", "audit"):
        root = tmp_path / reason
        attempt = root / "science/gpt-6.1-sol/compaction/attempt-001"
        (attempt / "worker").mkdir(parents=True)
        worker = {"status": "completed", "window": 32000, "exit_code": 0}
        if reason == "protocol":
            worker["protocol_valid"] = False
        elif reason == "harness":
            worker["exit_code"] = 1
        else:
            (attempt / "evaluation-audit.json").write_text(json.dumps({
                "valid": False, "reason": "provider_capacity"}))
        (attempt / "worker/run.json").write_text(json.dumps(worker))
        (attempt / "result.json").write_text(json.dumps({
            "status": "completed_failed", "verifier": {
                "reward": 0, "junit": {"tests": 163, "failures": 163}}}))
        row = next(row for row in collect_matrix(root)["rows"] if row["path"] == str(attempt))
        assert row["status"] in {"protocol_invalid", "infrastructure_failed"}
        assert row["score"] is None
        assert row["score_kind"] is None
        assert row["detail"] is None


def test_quota_truncated_science_and_arc_scores_remain_unknown(tmp_path: Path) -> None:
    from evals.pilot.matrix_report import collect_slot

    worker = {"status": "stopped", "exit_code": -15, "stop_reason": "plan_quota_reserve_reached",
              "protocol_valid": True, "window": 32000}
    science = tmp_path / "science/model/compaction"
    attempt = science / "attempt-001"
    (attempt / "worker").mkdir(parents=True)
    (attempt / "worker/run.json").write_text(json.dumps(worker))
    (attempt / "result.json").write_text(json.dumps({
        "status": "completed_failed", "verifier": {"reward": 0, "junit": {"tests": 163, "failures": 163}}}))
    arc = tmp_path / "arc/model/compaction"
    (arc / "worker").mkdir(parents=True)
    (arc / "worker/run.json").write_text(json.dumps(worker))
    (arc / "summary.json").write_text(json.dumps({"scored": True}))
    (arc / "official_scorecard.json").write_text(json.dumps({"score": 3.5}))
    for benchmark, slot in (("science", science), ("arc", arc)):
        row = collect_slot(benchmark, "model", "compaction", slot)
        assert row["status"] == "quota_stopped"
        assert row["score"] is None
        assert row["detail"] is None


def test_sampled_science_check_details_do_not_replace_official_reward(tmp_path: Path) -> None:
    from evals.pilot.matrix_report import collect_slot

    slot = tmp_path / "science/model/folding"
    attempt = slot / "attempt-001"
    (attempt / "verifier/logs").mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "status": "completed_failed", "verifier": {"reward": 0}}))
    (attempt / "verifier/logs/ctrf.json").write_text(json.dumps({"results": {
        "summary": {"tests": 8, "passed": 7, "failed": 1},
        "tests": [{"name": "test_state.py::test_speed_ratio", "status": "failed"}]}}))
    row = collect_slot("science", "model", "folding", slot)
    assert row["score"] == 0
    assert row["detail"] == "7/8 checks; failed: test_speed_ratio"
    assert row["failed_checks"] == ["test_speed_ratio"]
