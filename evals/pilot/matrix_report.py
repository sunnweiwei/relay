"""Collect the 66-cell pilot matrix without treating absent scores as zero."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from evals.pilot.methods import EVALUATED_METHODS

from evals.pilot.collect import collect


MODELS = ("gpt-6-luna", "gpt-6.1-sol")
BENCHMARKS = {"science": "Terminal-Bench-Science", "browse": "BrowseComp-Plus", "arc": "ARC-AGI-3"}
STATUS_LABELS = {
    "not_started": "not started", "running": "running", "grading": "grading",
    "scored": "scored", "timeout": "verifier timeout", "preflight_failed": "preflight failed",
    "ungraded": "not yet scored", "infrastructure_failed": "environment or execution failure",
    "protocol_invalid": "protocol check failed",
    "context_budget_exceeded": "context budget exceeded",
    "deferred": "deferred, not run",
    "quota_stopped": "stopped by the quota guard, unscored",
}
# tmux sessions whose queues --watch waits for; name them with --session.
FOLLOWON_SESSIONS: tuple[str, ...] = ()
QUEUE_SESSIONS: tuple[str, ...] = ()
TERMINAL_STATUSES = {"scored", "timeout", "preflight_failed", "infrastructure_failed", "protocol_invalid", "context_budget_exceeded", "quota_stopped"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def usage(path: Path) -> dict[str, Any]:
    if not (path / "run.json").is_file():
        return {}
    try:
        row = collect(path)
        return {key: row.get(key) for key in (
            "worker_usage", "summary_usage", "combined_usage", "summary_attempts",
            "summary_usage_missing_attempts", "context_event_records",
            "requests_over_context_budget", "harness_requests", "model_slugs_observed",
            "inference_started", "upstream_attempts", "unaccounted_upstream_attempts",
            "context_budget_scope", "auxiliary_peak_input_tokens", "auxiliary_requests_above_main_window",
            "all_saved_credit_checks_unchanged", "http_error_statuses")}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {"collection_error": type(exc).__name__}


def base_row(benchmark: str, model: str, method: str, slot: Path) -> dict[str, Any]:
    return {"benchmark": BENCHMARKS[benchmark], "model": model, "strategy": method,
            "status": "not_started", "score": None, "score_kind": None,
            "detail": None, "stop_reason": None, "path": str(slot), "usage": {}}


def worker_status(row: dict[str, Any], worker: dict[str, Any]) -> None:
    status = worker.get("status")
    row["worker_status"] = status
    row["protocol_valid"] = worker.get("protocol_valid")
    row["strategy_errors"] = worker.get("strategy_errors", [])
    row["failure_kind"] = worker.get("failure_kind")
    row["stop_reason"] = worker.get("stop_reason") or worker.get("error")
    row["unforced_harness_failure"] = worker.get("exit_code") not in (None, 0) and not worker.get("stop_reason")
    if status == "preflight_failed":
        row["status"] = "preflight_failed"
    elif status == "running":
        row["status"] = "running"
    elif status:
        row["status"] = "ungraded"
    if status == "infrastructure_failed" or (worker.get("exit_code") not in (None, 0)
                                               and not worker.get("stop_reason")):
        row.update(status="infrastructure_failed", score=None,
                   stop_reason=worker.get("provider_error") or worker.get("error") or "worker_exit_without_enforced_stop")


def science_row(row: dict[str, Any], slot: Path) -> None:
    attempts = sorted(path for path in slot.glob("attempt-*") if path.is_dir())
    if not attempts:
        return
    def valid_grade(path: Path) -> bool:
        worker = read_json(path / "worker/run.json")
        return (read_json(path / "result.json").get("status") in {"completed", "completed_failed", "timeout"}
                and read_json(path / "evaluation-audit.json").get("valid") is not False
                and worker.get("status") != "infrastructure_failed"
                and not (worker.get("exit_code") not in (None, 0) and not worker.get("stop_reason")))
    graded = [path for path in attempts if valid_grade(path)]
    selected = graded[-1] if graded else attempts[-1]
    outer = read_json(selected / "result.json")
    worker = read_json(selected / "worker/run.json")
    worker_status(row, worker)
    row.update(path=str(selected), usage=usage(selected / "worker"), attempts=len(attempts))
    attempt_usages = [usage(path / "worker") for path in attempts]
    row["attempt_usage"] = attempt_usages
    for key in ("combined_usage", "summary_usage"):
        totals = [entry[key]["total_tokens"] for entry in attempt_usages if entry.get(key)]
        row[f"all_attempts_{key}_tokens"] = sum(totals) if totals else None
    if outer.get("status") in {"worker_running", "grading"}:
        row["status"] = "running" if outer["status"] == "worker_running" else "grading"
    verifier = read_json(selected / "verifier/verifier-result.json") or outer.get("verifier", {})
    if verifier.get("timeout") or verifier.get("status") == "timeout":
        row.update(status="timeout", detail=f"verifier phase: {verifier.get('phase', 'unknown')}")
    elif verifier.get("reward") is not None:
        row.update(status="scored", score=verifier["reward"], score_kind="official_local_verifier")
        junit = verifier.get("junit") or {}
        if junit.get("tests") is not None:
            passed = junit["tests"] - sum(junit.get(key, 0) for key in ("failures", "errors", "skipped"))
            row["detail"] = f"{passed}/{junit['tests']} checks"
        else:
            ctrf = read_json(selected / "verifier/logs/ctrf.json").get("results", {})
            checks = ctrf.get("summary", {})
            if checks.get("tests") is not None and checks.get("passed") is not None:
                row["detail"] = f"{checks['passed']}/{checks['tests']} checks"
                failed = [test["name"].rsplit("::", 1)[-1] for test in ctrf.get("tests", [])
                          if test.get("status") == "failed" and isinstance(test.get("name"), str)]
                row["failed_checks"] = failed
                if failed:
                    row["detail"] += "; failed: " + ", ".join(failed)
    elif outer.get("status") == "worker_preflight_failed":
        row["status"] = "preflight_failed"
    elif outer.get("status") in {"worker_launcher_failed", "verifier_launcher_failed", "missing_submission"}:
        row.update(status="infrastructure_failed", detail=outer["status"])


def browse_row(row: dict[str, Any], slot: Path) -> None:
    attempts = sorted(path for path in slot.glob("attempt-*") if path.is_dir())
    if attempts:
        slot = attempts[-1]
        row["path"] = str(slot)
    worker_status(row, read_json(slot / "run.json"))
    row["usage"] = usage(slot)
    score = read_json(slot / "browse-grade.json")
    if score and score.get("strict_answer_score") is not None and row["status"] != "infrastructure_failed":
        row.update(status="scored", score=score.get("strict_answer_score"),
                   score_kind="strict_answer_diagnostic", official_score=None,
                   detail=f"search={score.get('searches')}; fetch={score.get('fetches')}",
                   retrieval=score)


def arc_row(row: dict[str, Any], slot: Path) -> None:
    worker_status(row, read_json(slot / "worker/run.json"))
    row["usage"] = usage(slot / "worker")
    summary = read_json(slot / "summary.json")
    if not summary and (slot / "arm.json").is_file() and row["status"] == "not_started":
        row["status"] = "running"
    if summary.get("scored"):
        score = read_json(slot / "official_scorecard.json")
        if score.get("score") is not None:
            row.update(status="scored", score=score["score"], score_kind="official_local_scorecard",
                       detail=f"{score.get('total_levels_completed')}/{score.get('total_levels')} levels; "
                              f"{score.get('total_actions')} actions")


def collect_slot(benchmark: str, model: str, method: str, slot: Path) -> dict[str, Any]:
    """Read one native benchmark slot, excluding invalid attempts from scores."""
    collectors = {"science": science_row, "browse": browse_row, "arc": arc_row}
    row = base_row(benchmark, model, method, slot)
    collectors[benchmark](row, slot)
    if row.get("protocol_valid") is False:
        row["status"] = ("context_budget_exceeded" if row.get("failure_kind") == "context_budget_exceeded"
                         else "protocol_invalid")
    if row.get("unforced_harness_failure"):
        from evals.pilot.run import harness_failure

        worker_dir = Path(row["path"]) / "worker" if benchmark != "browse" else Path(row["path"])
        row.update(status="infrastructure_failed", score=None, protocol_valid=False,
                   stop_reason=harness_failure(worker_dir) or row.get("stop_reason") or "harness exited unexpectedly")
    audit = read_json(Path(row["path"]) / "evaluation-audit.json")
    if audit.get("valid") is False:
        row.update(status=audit.get("classification", "infrastructure_failed"),
                   score=None, evaluation_audit=audit, stop_reason=audit.get("reason"))
    if row.get("stop_reason") == "experiment_admission_closed":
        row.update(status="deferred", score=None, score_kind=None, detail=None)
    if any(term in str(row.get("stop_reason") or "").lower()
           for term in ("quota", "credit", "ordinary_plan", "account_limit")):
        row["status"] = "quota_stopped"
    if row["status"] in {"protocol_invalid", "infrastructure_failed", "preflight_failed", "context_budget_exceeded", "quota_stopped"}:
        row.update(score=None, score_kind=None, detail=None)
    return row


def collect_matrix(root: Path, prior: Path | None = None) -> dict[str, Any]:
    rows = []
    for benchmark in BENCHMARKS:
        for method in EVALUATED_METHODS:
            for model in MODELS:
                slot = root / benchmark / model / method
                rows.append(collect_slot(benchmark, model, method, slot))
    diagnostics = []
    archive_root = root / "diagnostics"
    archived_workers = list(archive_root.glob("*/*/*/run.json")) + list(archive_root.glob("*/*/*/worker/run.json"))
    for path in sorted(archived_workers):
        worker = read_json(path)
        run_dir = path.parent
        audit_root = run_dir.parent if run_dir.name == "worker" else run_dir
        audit = read_json(audit_root / "evaluation-audit.json")
        benchmark_key = path.relative_to(archive_root).parts[0].split("-")[0]
        diagnostics.append({"benchmark": BENCHMARKS.get(benchmark_key, benchmark_key),
                            "model": worker.get("model"), "strategy": worker.get("strategy"),
                            "path": str(audit_root), "reason": audit.get("reason", "archived_attempt"),
                            "usage": usage(run_dir)})
    for row in rows:
        if row["benchmark"] == BENCHMARKS["browse"]:
            slot = root / "browse" / row["model"] / row["strategy"]
            attempts = [slot] + sorted(path for path in slot.glob("attempt-*") if path.is_dir())
            for attempt in attempts:
                audit = read_json(attempt / "evaluation-audit.json")
                if audit.get("valid") is False and str(attempt) != row["path"]:
                    diagnostics.append({"benchmark": BENCHMARKS["browse"],
                                        "model": row["model"], "strategy": row["strategy"],
                                        "path": str(attempt), "reason": audit.get("reason"), "usage": usage(attempt)})
        if row["benchmark"] != BENCHMARKS["science"]:
            continue
        slot = root / "science" / row["model"] / row["strategy"]
        for attempt in sorted(slot.glob("attempt-*")):
            if not attempt.is_dir():
                continue
            audit = read_json(attempt / "evaluation-audit.json")
            if audit.get("valid") is False and str(attempt) != row["path"]:
                diagnostics.append({"benchmark": BENCHMARKS["science"],
                                    "model": row["model"], "strategy": row["strategy"],
                                    "path": str(attempt), "reason": audit.get("reason"),
                                    "usage": usage(attempt / "worker")})
    previous = read_json(prior).get("runs", []) if prior else []
    old = [{key: row.get(key) for key in ("benchmark", "model", "strategy", "score", "detail", "grade_status",
                                          "combined_usage", "summary_usage", "directory")}
           for row in previous]
    return {"generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "root": str(root), "planned_cells": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
            "admission_closed": read_json(root / ".stop-new-workers"),
            "rows": rows, "diagnostics": diagnostics, "prior_luna_runs": old,
            "limitations": ["Each benchmark uses one frozen task; no benchmark-wide ranking is supported.",
                            "BrowseComp uses a fixed local corpus and an exact-answer diagnostic, not the official LLM judge.",
                            "Token totals include known Relay summary usage; unavailable or in-flight usage is unknown.",
                            "RLM execution isolation passed; bounded model probes timed out and do not establish benchmark performance."]}


def esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else "—"


def number(value: Any) -> str:
    return f"{value:,}" if isinstance(value, (int, float)) else "—"



def live_queue_sessions(names: list[str]) -> list[str]:
    try:
        result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        # Unknown state must not be mistaken for finished experiments.
        return names
    active = set(result.stdout.splitlines())
    return [name for name in names if name in active]


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def save_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output.with_suffix(".json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", type=Path, help="Earlier pilot results JSON to list separately")
    parser.add_argument("--watch", action="store_true", help="Refresh while benchmark tmux queues are active")
    parser.add_argument("--interval", type=float, default=60, help="Refresh interval in seconds (at least 5)")
    parser.add_argument("--max-hours", type=float, default=24, help="Watch limit, at most 24 hours")
    parser.add_argument("--session", action="append", help="Benchmark tmux session to monitor; repeatable")
    args = parser.parse_args()
    if args.interval < 5 or not 0 < args.max_hours <= 24:
        parser.error("--interval must be at least 5 seconds and --max-hours must be in (0, 24]")
    deadline = time.monotonic() + args.max_hours * 3600
    sessions = args.session or list(QUEUE_SESSIONS)
    while True:
        report = collect_matrix(args.root.resolve(), args.prior)
        stop_reason = None
        if args.watch:
            active = live_queue_sessions(sessions)
            if all(row["status"] in TERMINAL_STATUSES for row in report["rows"]) and not set(active).intersection(FOLLOWON_SESSIONS):
                stop_reason = "all_cells_terminal"
            elif not active:
                stop_reason = "all_queue_sessions_gone"
            elif time.monotonic() >= deadline:
                stop_reason = "watch_time_limit"
            report["watch"] = {"active_sessions": active, "stop_reason": stop_reason}
        save_report(report, args.out)
        print(json.dumps({"out": str(args.out), "planned_cells": report["planned_cells"],
                          "status_counts": report["status_counts"], "watch_stop_reason": stop_reason}), flush=True)
        if not args.watch or stop_reason:
            break
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
