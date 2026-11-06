"""Durably run the paired Science matrix, preserving every attempt.

Run under tmux or Slurm. The driver exits on a quota stop; a later invocation
skips graded or audited interrupted attempts, and allows one capacity retry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from evals.pilot.methods import EVALUATED_METHODS
from evals.pilot.repair_clm import context_budget_only
from evals.pilot.failure_policy import (
    UPSTREAM_STREAM_INTERRUPTED, reviewed_terminal_failure, upstream_stream_interrupted,
)

try:
    from .science_suite import classify_worker, quota_stop_reason, write_json
except ImportError:
    from science_suite import classify_worker, quota_stop_reason, write_json


METHODS = EVALUATED_METHODS
MODELS = ("gpt-6-luna", "gpt-6.1-sol")
GRADED = {"completed", "completed_failed", "timeout"}
INFRASTRUCTURE_FAILURES = {"worker_infrastructure_failed", "worker_protocol_invalid", "worker_preflight_failed",
                           "worker_launcher_failed", "verifier_launcher_failed", "verifier_failed", "launcher_failed"}
REPO = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def quota_stopped(result: dict[str, Any]) -> bool:
    worker = result.get("worker_status", {})
    return (quota_stop_reason(worker) is not None or quota_stop_reason(result) is not None
            or result.get("status") == "worker_quota_stopped")


def saved_quota_attempt(slots: list[Path]) -> tuple[Path, str] | None:
    """Check the whole queue before resuming any earlier unfinished cell."""
    for slot in slots:
        for path in sorted(slot.glob("attempt-[0-9][0-9][0-9]")):
            result = read_json(path / "result.json")
            worker = read_json(path / "worker/run.json") or result.get("worker_status", {})
            result["worker_status"] = worker
            if quota_stopped(result):
                reason = quota_stop_reason(worker) or quota_stop_reason(result) or "quota_stopped"
                attempt_result(path)
                return path, reason
    return None


def attempt_result(path: Path) -> dict[str, Any]:
    result = read_json(path / "result.json")
    metadata = read_json(path / "worker/run.json") or result.get("worker_status", {})
    metadata = classify_worker(path / "worker", metadata)
    result["worker_status"] = metadata
    if quota_stopped(result):
        reason = quota_stop_reason(metadata) or quota_stop_reason(result) or "quota_stopped"
        result.update(status="worker_quota_stopped", stop_reason=reason, benchmark_score=None)
        write_json(path / "evaluation-audit.json", {
            "valid": False, "classification": "quota_stopped",
            "reason": reason, "benchmark_score": None,
        })
    elif reviewed_terminal_failure(path):
        result.update(status="worker_infrastructure_failed", benchmark_score=None)
    elif (path / "continuation-audit.json").exists():
        result.update(status="worker_infrastructure_failed", benchmark_score=None,
                      reviewed_failure_validation_failed=True)
    elif context_budget_only(metadata):
        result["status"] = "worker_context_budget_exceeded"
        write_json(path / "evaluation-audit.json", {
            "valid": False, "classification": "context_budget_exceeded",
            "reason": "context_budget_exceeded", "benchmark_score": None,
        })
    elif metadata.get("status") == "infrastructure_failed":
        result.update(status="worker_infrastructure_failed", benchmark_score=None)
        write_json(path / "evaluation-audit.json", {
            "valid": False, "classification": "infrastructure_failed",
            "reason": metadata.get("failure_kind", "infrastructure_failed"),
            "detail": metadata.get("error"), "benchmark_score": None,
        })
    elif metadata.get("protocol_valid") is False:
        result["status"] = "worker_protocol_invalid"
        write_json(path / "evaluation-audit.json", {
            "valid": False, "classification": "protocol_invalid",
            "reason": "worker_protocol_invalid", "benchmark_score": None,
        })
    elif (path / "evaluation-audit.json").is_file():
        audit = read_json(path / "evaluation-audit.json")
        if audit.get("valid") is False:
            result.update(status="worker_protocol_invalid", benchmark_score=None)
    return result


def at_capacity(result: dict[str, Any]) -> bool:
    return result.get("worker_status", {}).get("failure_kind") == "provider_capacity"


def stream_interrupted(path: Path, result: dict[str, Any]) -> bool:
    """Revalidate raw evidence before continuing; an annotation never suffices."""
    return (not quota_stopped(result)
            and result.get("worker_status", {}).get("failure_kind") == UPSTREAM_STREAM_INTERRUPTED
            and upstream_stream_interrupted(read_json(path / "worker/run.json"), path / "worker/relay/runtime.log"))


def run_matrix(args: argparse.Namespace) -> int:
    output = args.out.resolve()
    output.relative_to(args.scratch.resolve())
    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"benchmark": "Terminal-Bench-Science", "task": "traffic-flux-inversion",
                             "models": args.models, "methods": args.methods,
                             "started_unix": time.time(), "status": "running", "entries": []}

    def save() -> None:
        target = output / "matrix.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(target)

    save()
    slots = [output / model / method for method in args.methods for model in args.models]
    if stopped := saved_quota_attempt(slots):
        path, reason = stopped
        state.update(status="quota_stopped", stopped_attempt=str(path), stop_reason=reason,
                     finished_unix=time.time())
        save()
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    for method in args.methods:
        for model in args.models:
            slot = output / model / method
            attempts = sorted(slot.glob("attempt-[0-9][0-9][0-9]"))
            old_results = [(path, attempt_result(path)) for path in attempts]
            if any(quota_stopped(result) for _, result in old_results):
                state.update(status="quota_stopped", finished_unix=time.time())
                save()
                return 2
            if any(result.get("reviewed_failure_validation_failed") for _, result in old_results):
                state.update(status="infrastructure_failed", finished_unix=time.time())
                save()
                return 1
            limitations = [path for path, result in old_results
                           if result.get("status") == "worker_context_budget_exceeded"]
            if limitations:
                state["entries"].append({"model": model, "strategy": method,
                                         "path": str(limitations[-1]), "status": "context_budget_exceeded"})
                save()
                continue
            graded = [path for path, result in old_results if result.get("status") in GRADED]
            if graded:
                state["entries"].append({"model": model, "strategy": method,
                                         "path": str(graded[-1]), "status": "already_graded"})
                save()
                continue
            interrupted = [path for path, result in old_results if stream_interrupted(path, result)]
            reviewed = [(path, reason) for path, _ in old_results if (reason := reviewed_terminal_failure(path))]
            reviewed_paths = {path for path, _ in reviewed}
            failures = [(path, result) for path, result in old_results
                        if result.get("status") in INFRASTRUCTURE_FAILURES and not at_capacity(result)
                        and not stream_interrupted(path, result) and path not in reviewed_paths]
            if reviewed and not failures:
                path, reason = reviewed[-1]
                state["entries"].append({"model": model, "strategy": method, "path": str(path),
                                         "status": "worker_infrastructure_failed", "reason": reason,
                                         "benchmark_score": None})
                save()
                continue
            if interrupted and not failures:
                state["entries"].append({"model": model, "strategy": method, "path": str(interrupted[-1]),
                                         "status": "worker_infrastructure_failed",
                                         "reason": UPSTREAM_STREAM_INTERRUPTED, "benchmark_score": None})
                save()
                continue
            if failures:
                path, result = failures[-1]
                state["entries"].append({"model": model, "strategy": method, "path": str(path),
                                         "status": result["status"]})
                state.update(status="infrastructure_failed", finished_unix=time.time())
                save()
                return 1
            capacity_failures = sum(at_capacity(result) for _, result in old_results)
            if capacity_failures >= 2:
                state["entries"].append({"model": model, "strategy": method,
                                         "path": str(attempts[-1]), "status": "provider_capacity"})
                save()
                continue
            for retry in range(2 - capacity_failures):
                attempt = slot / f"attempt-{len(attempts) + retry + 1:03d}"
                command = [str(args.python.absolute()), str(REPO / "evals/pilot/science_suite.py"),
                           "--python", str(args.python.absolute()), "--scratch", str(args.scratch),
                           "--out", str(attempt), "--model", model, "--strategy", method]
                entry = {"model": model, "strategy": method, "path": str(attempt),
                         "status": "running", "command": command}
                state["entries"].append(entry)
                save()
                slot.mkdir(parents=True, exist_ok=True)
                print(json.dumps({"event": "started", "model": model, "strategy": method,
                                  "attempt": str(attempt)}), flush=True)
                with (slot / f"{attempt.name}-launcher.log").open("w") as log:
                    process = subprocess.run(command, cwd=REPO, env=env,
                                             stdout=log, stderr=subprocess.STDOUT)
                result = attempt_result(attempt)
                entry.update(status=result.get("status", "launcher_failed"), exit_code=process.returncode)
                save()
                print(json.dumps({"event": "finished", "model": model, "strategy": method,
                                  "status": entry["status"], "exit_code": process.returncode}), flush=True)
                if quota_stopped(result):
                    state.update(status="quota_stopped", finished_unix=time.time())
                    save()
                    return 2
                if result.get("reviewed_failure_validation_failed"):
                    state.update(status="infrastructure_failed", finished_unix=time.time())
                    save()
                    return 1
                if at_capacity(result):
                    entry["status"] = "provider_capacity"
                    save()
                    continue
                if entry["status"] == "worker_context_budget_exceeded":
                    entry["status"] = "context_budget_exceeded"
                    save()
                    break
                if stream_interrupted(attempt, result):
                    entry.update(reason=UPSTREAM_STREAM_INTERRUPTED, benchmark_score=None)
                    save()
                    break
                if reason := reviewed_terminal_failure(attempt):
                    entry.update(reason=reason, benchmark_score=None)
                    save()
                    break
                if entry["status"] in INFRASTRUCTURE_FAILURES:
                    state.update(status="infrastructure_failed", finished_unix=time.time())
                    save()
                    return 1
                break
    state.update(status="complete" if all(e["status"] in GRADED | {"already_graded"}
                                          for e in state["entries"]) else "complete_with_failures",
                 finished_unix=time.time())
    save()
    return 0 if state["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    return run_matrix(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
