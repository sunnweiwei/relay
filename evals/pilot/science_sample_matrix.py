"""Task-first paired Science sample queue with durable attempts and quota stops."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from evals.pilot.science_container import write_json
from evals.pilot.science_matrix import (
    GRADED, INFRASTRUCTURE_FAILURES, METHODS, MODELS, UPSTREAM_STREAM_INTERRUPTED,
    at_capacity, attempt_result, quota_stopped, reviewed_terminal_failure, saved_quota_attempt, stream_interrupted,
)
from evals.pilot.science_sample_suite import REPO, require_readiness
from evals.pilot.repair_clm import context_budget_only
from evals.pilot.matrix_report import live_queue_sessions


def run_matrix(args: argparse.Namespace) -> int:
    output = args.out.resolve()
    output.relative_to(args.scratch.resolve())
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    task_ids = manifest["selected_ids"]
    state: dict[str, Any] = {"benchmark": "Terminal-Bench-Science", "manifest": str(args.manifest.resolve()),
                             "tasks": task_ids, "models": args.models, "methods": args.methods,
                             "status": "running", "started_unix": time.time(), "entries": []}

    def save() -> None:
        write_json(output / "matrix.json", state)

    save()
    slots = [output / task_id / model / method for task_id in task_ids
             for method in args.methods for model in args.models]
    if stopped := saved_quota_attempt(slots):
        path, reason = stopped
        state.update(status="quota_stopped", stopped_attempt=str(path), stop_reason=reason,
                     finished_unix=time.time())
        save()
        return 2
    wait_deadline = time.monotonic() + getattr(args, "wait_ready_seconds", 0)
    dependencies = getattr(args, "after_session", []) or []
    while dependencies and live_queue_sessions(dependencies):
        state.update(status="waiting_for_existing_queues", waiting_sessions=live_queue_sessions(dependencies))
        save()
        if time.monotonic() >= wait_deadline:
            state.update(status="wait_deadline_reached", finished_unix=time.time())
            save()
            return 1
        time.sleep(min(30, wait_deadline - time.monotonic()))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    for task_id in task_ids:
        task = args.tasks_root / task_id
        images = args.images_root / task_id
        wait_deadline = time.monotonic() + getattr(args, "wait_ready_seconds", 0)
        while True:
            try:
                require_readiness(task, images)
                break
            except (OSError, ValueError, RuntimeError) as error:
                state.update(status="task_not_ready", task_id=task_id, error=str(error))
                save()
                if time.monotonic() >= wait_deadline:
                    state.update(finished_unix=time.time())
                    save()
                    return 1
                time.sleep(min(30, wait_deadline - time.monotonic()))
        state.update(status="running", task_id=task_id)
        state.pop("error", None)
        save()
        for method in args.methods:
            for model in args.models:
                slot = output / task_id / model / method
                attempts = sorted(path for path in slot.glob("attempt-[0-9][0-9][0-9]") if path.is_dir())
                previous = [(path, attempt_result(path)) for path in attempts]
                if any(quota_stopped(result) for _, result in previous):
                    state.update(status="quota_stopped", task_id=task_id, finished_unix=time.time())
                    save()
                    return 2
                if any(result.get("reviewed_failure_validation_failed") for _, result in previous):
                    state.update(status="infrastructure_failed", finished_unix=time.time())
                    save()
                    return 1
                unfinished = [(path, result) for path, result in previous
                              if result.get("status") in {"preparing", "worker_running", "grading", "running"}
                              or result.get("worker_status", {}).get("status") == "running"]
                if unfinished:
                    path, result = unfinished[-1]
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "path": str(path), "status": "unfinished_attempt",
                                             "result_status": result.get("status"),
                                             "worker_status": result.get("worker_status", {}).get("status")})
                    state.update(status="unfinished_attempt_no_retry", task_id=task_id,
                                 finished_unix=time.time())
                    save()
                    return 2
                valid = [path for path, result in previous if result.get("status") in GRADED
                         and not (path / "evaluation-audit.json").exists()]
                if valid:
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "path": str(valid[-1]), "status": "already_graded"})
                    save()
                    continue
                limitations = [path for path, result in previous
                               if context_budget_only(result.get("worker_status", {}))]
                if limitations:
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "status": "context_budget_exceeded", "path": str(limitations[-1])})
                    save()
                    continue
                interrupted = [path for path, result in previous if stream_interrupted(path, result)]
                reviewed = [(path, reason) for path, _ in previous if (reason := reviewed_terminal_failure(path))]
                reviewed_paths = {path for path, _ in reviewed}
                failures = [(path, result) for path, result in previous
                            if result.get("status") in INFRASTRUCTURE_FAILURES and not at_capacity(result)
                            and not stream_interrupted(path, result) and path not in reviewed_paths]
                if reviewed and not failures:
                    path, reason = reviewed[-1]
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "status": "worker_infrastructure_failed", "path": str(path),
                                             "reason": reason, "benchmark_score": None})
                    save()
                    continue
                if interrupted and not failures:
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "status": "worker_infrastructure_failed", "path": str(interrupted[-1]),
                                             "reason": UPSTREAM_STREAM_INTERRUPTED, "benchmark_score": None})
                    save()
                    continue
                if failures:
                    path, result = failures[-1]
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "path": str(path), "status": result["status"]})
                    state.update(status="infrastructure_failed", finished_unix=time.time())
                    save()
                    return 1
                capacity_count = sum(at_capacity(result) for _, result in previous)
                if capacity_count >= 2:
                    state["entries"].append({"task_id": task_id, "model": model, "strategy": method,
                                             "status": "provider_capacity", "path": str(attempts[-1])})
                    save()
                    continue
                for retry in range(2 - capacity_count):
                    attempt = slot / f"attempt-{len(attempts) + retry + 1:03d}"
                    slot.mkdir(parents=True, exist_ok=True)
                    command = [str(args.python.absolute()), "-m", "evals.pilot.science_sample_suite",
                               "--task", str(task), "--images", str(images), "--scratch", str(args.scratch),
                               "--out", str(attempt), "--python", str(args.python.absolute()),
                               "--model", model, "--strategy", method]
                    entry = {"task_id": task_id, "model": model, "strategy": method,
                             "path": str(attempt), "status": "running", "command": command}
                    state["entries"].append(entry)
                    save()
                    print(json.dumps({"event": "started", **entry}), flush=True)
                    with (slot / f"{attempt.name}-launcher.log").open("w") as log:
                        process = subprocess.run(command, cwd=REPO, env=environment,
                                                 stdout=log, stderr=subprocess.STDOUT)
                    result = attempt_result(attempt)
                    entry.update(status=result.get("status", "launcher_failed"), exit_code=process.returncode)
                    save()
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
                    if context_budget_only(result.get("worker_status", {})):
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
                    if entry["status"] not in GRADED:
                        state.update(status="infrastructure_failed", finished_unix=time.time())
                        save()
                        return 1
                    break
    state.update(status="complete" if all(entry["status"] in GRADED | {"already_graded"}
                                          for entry in state["entries"]) else "complete_with_failures",
                 finished_unix=time.time())
    save()
    return 0 if state["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["gpt-6.1-sol", "gpt-6-luna"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--after-session", action="append", default=[],
                        help="Wait for an existing tmux queue to exit; repeatable")
    parser.add_argument("--wait-ready-seconds", type=float, default=86400,
                        help="Maximum wait for prerequisite queues or each task's readiness")
    return run_matrix(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
