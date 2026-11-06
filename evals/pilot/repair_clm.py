"""Run at most one fixed-pilot CLM repair per model and source-fix hash."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from evals.pilot.collect import records
from evals.pilot.cost_report import worker_directories
from evals.pilot.matrix_report import live_queue_sessions
from evals.pilot.run import CodexRPC, QuotaGate

REPO = Path(__file__).resolve().parents[2]
MODELS = ("gpt-6.1-sol", "gpt-6-luna")
FIX_FILES = ("relay/strategies/clm.py", "relay/transport/proxy.py")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fix_fingerprint() -> dict[str, Any]:
    files = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FIX_FILES}
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}


def quota_stopped(worker: dict[str, Any]) -> bool:
    reason = " ".join(str(worker.get(key) or "") for key in ("stop_reason", "error"))
    return any(term in reason.lower() for term in ("quota", "credit", "ordinary_plan", "account_limit"))


def context_budget_only(worker: dict[str, Any]) -> bool:
    """A measured context violation is distinct from an unrelated worker defect."""
    return (worker.get("status") in {"completed", "stopped"}
            and worker.get("requests_over_context_budget", 0) > 0
            and not worker.get("strategy_errors") and not worker.get("sandbox_execution_errors")
            and not quota_stopped(worker)
            and worker.get("error") in (None, "", "context_budget_exceeded")
            and worker.get("stop_reason") in (None, "", "model_request_limit", "wall_time_limit", "context_budget_exceeded")
            and (worker.get("exit_code") in (None, 0) or worker.get("stop_reason") in {
                "model_request_limit", "wall_time_limit", "context_budget_exceeded"}))


def saved_quota_stop(pilot_root: Path) -> str | None:
    for directory in worker_directories(pilot_root):
        path = directory / "run.json"
        try:
            worker = read_json(path)
        except json.JSONDecodeError:
            worker = {}  # Another benchmark can be finishing its metadata write.
        if quota_stopped(worker):
            return str(path)
        path = directory / "quota.jsonl"
        for row in records(path):
            if quota_stopped(row) or row.get("credit_balance_unchanged") is False:
                return str(path)
    return None


def attempt_dirs(slot: Path, benchmark: str) -> list[Path]:
    attempts = sorted(path for path in slot.glob("attempt-[0-9][0-9][0-9]") if path.is_dir())
    if benchmark == "arc":
        return [slot] if (slot / "worker/run.json").is_file() else []
    return ([slot] if benchmark == "browse" and (slot / "run.json").is_file() else []) + attempts


def worker_path(attempt: Path, benchmark: str) -> Path:
    return attempt if benchmark == "browse" else attempt / "worker"


def repair_plan(args: argparse.Namespace, model: str, old: Path, fix_hash: str) -> dict[str, Any]:
    """Preserve the task, prompt, limits, and native grading/service implementation."""
    slot = args.pilot_root / args.benchmark / model / "clm"
    worker = read_json(worker_path(old, args.benchmark) / "run.json")
    expected = (23560, 100) if args.benchmark == "browse" else (32000, 64 if args.benchmark == "science" else 100)
    if (worker.get("window"), worker.get("max_requests")) != expected or worker.get("seconds_limit") != 1800:
        raise ValueError("Original CLM limits differ from the frozen native pilot")
    if worker.get("model") != model or worker.get("strategy") != "clm":
        raise ValueError("Original worker identity differs from this repair cell")
    if worker.get("reasoning_effort") != "medium" or worker.get("harness") != "codex":
        raise ValueError("Original worker settings differ from the frozen native pilot")
    old_prompt = worker_path(old, args.benchmark) / "work/task.txt"
    prompt = {"browse": args.scratch / "browse/prompt.md",
              "science": args.scratch / "science/runs/fixed-python-compaction-01/work/task.txt",
              "arc": args.scratch / "arc/frozen_prompt.txt"}[args.benchmark]
    if old_prompt.read_bytes() != prompt.read_bytes():
        raise ValueError("Frozen prompt changed since the original CLM attempt")
    entry: dict[str, Any] = {"model": model, "strategy": "clm", "fix_hash": fix_hash,
        "original_attempt": str(old), "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()}
    if args.benchmark == "arc":
        arm = read_json(old / "arm.json")
        if (arm.get("game_id"), arm.get("seed"), arm.get("action_limit")) != ("ls20-9607627b", 0, 300):
            raise ValueError("ARC repair is restricted to the frozen pilot game")
        archive = args.pilot_root / "diagnostics/arc-clm-repair" / fix_hash / model / "clm"
        if archive.exists():
            raise FileExistsError(archive)
        entry.update(output=str(slot), archive=str(archive), port=arm["service_port"],
                     invocation="evals.pilot.arc_suite.run_arm", game_id=arm["game_id"], seed=arm["seed"])
        return entry
    numbers = [int(path.name.split("-")[-1]) for path in attempt_dirs(slot, args.benchmark) if path != slot]
    number = max(numbers, default=1 if args.benchmark == "browse" else 0) + 1
    output = slot / f"attempt-{number:03d}"
    if args.benchmark == "science":
        command = [str(args.python), str(REPO / "evals/pilot/science_suite.py"),
                   "--scratch", str(args.scratch), "--python", str(args.python),
                   "--model", model, "--strategy", "clm", "--out", str(output)]
    else:
        label = "clm" if old == slot else f"clm-{old.name}"
        original = args.pilot_root / "browse/driver-logs" / model / f"{label}.command.json"
        command = json.loads(original.read_text())
        for flag, expected_value in (("--model", model), ("--strategy", "clm"),
                                     ("--data-root", str(args.scratch / "browse")),
                                     ("--hidden-root", str(args.scratch))):
            if command[command.index(flag) + 1] != expected_value:
                raise ValueError("Saved Browse command differs from the original pilot cell")
        if "--query-id" in command and command[command.index("--query-id") + 1] != "183":
            raise ValueError("Browse repair is restricted to the frozen pilot query")
        if "--prompt-file" in command and Path(command[command.index("--prompt-file") + 1]).read_bytes() != prompt.read_bytes():
            raise ValueError("Saved Browse command uses a different prompt")
        service = args.pilot_root / "browse/services" / model / "clm" / output.name
        for flag, value in (("--out", output), ("--service-root", service)):
            command[command.index(flag) + 1] = str(value)
        entry["service_root"] = str(service)
    entry.update(output=str(output), command=command)
    return entry


def execute_repair(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
    output = Path(entry["output"])
    if args.benchmark == "arc":
        from evals.pilot.arc_suite import run_arm

        archive = Path(entry["archive"])
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), archive)
        run_arm("clm", args.python, REPO / "evals/pilot/run.py", REPO / "evals/pilot/arc_server.py",
                args.scratch, args.scratch / "arc", args.scratch / "arc/frozen_prompt.txt",
                "clm-repair", model=entry["model"], out_root=args.pilot_root / "arc",
                port=entry["port"], game_id=entry["game_id"], seed=entry["seed"])
    else:
        with (args.out / f"{entry['model']}-launcher.log").open("a") as log:
            result = subprocess.run(entry["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        entry["launcher_returncode"] = result.returncode
        if args.benchmark == "browse" and (output / "run.json").exists():
            from evals.pilot.browse_grade import grade

            write_json(output / "browse-grade.json", grade(output, Path(entry["service_root"]) / "retrieval.jsonl",
                       args.scratch / "browse/grader-private/selected-qid-183-gold-qrels.json", expected_query_id="183"))
    return read_json(worker_path(output, args.benchmark) / "run.json")


def run_queue(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = fix_fingerprint()
    state_path = args.out / "repair-state.json"
    state = read_json(state_path) or {"benchmark": args.benchmark, "fix": fingerprint, "attempts": {},
                                     "wait_sessions": args.wait_session, "started_unix": time.time()}
    if state["fix"] != fingerprint or state["benchmark"] != args.benchmark:
        raise ValueError("Repair state belongs to different fixes or benchmark")

    def record(status: str, **values: Any) -> None:
        state.update(status=status, updated_unix=time.time(), **values)
        write_json(state_path, state)

    record("waiting_for_pilot")
    deadline = state["started_unix"] + args.max_hours * 3600
    while live_queue_sessions(args.wait_session):
        if time.time() >= deadline:
            record("wait_timeout")
            return 2
        time.sleep(15)
    rpc = None
    try:
        for model in MODELS:
            marker = args.pilot_root / "repairs/clm-attempts" / fingerprint["sha256"] / args.benchmark / f"{model}.json"
            if marker.exists() and model not in state["attempts"]:
                state["attempts"][model] = read_json(marker)
            if model in state["attempts"]:
                if state["attempts"][model].get("status") == "running":
                    record("interrupted_repair_not_retried")
                    return 2
                continue
            if stop := saved_quota_stop(args.pilot_root):
                record("quota_stopped_no_repair", evidence=stop)
                return 2
            slot = args.pilot_root / args.benchmark / model / "clm"
            attempts = attempt_dirs(slot, args.benchmark)
            if not attempts:
                continue
            old = attempts[-1]
            worker = read_json(worker_path(old, args.benchmark) / "run.json")
            if worker.get("status") == "running":
                record("unfinished_worker_no_repair", evidence=str(old))
                return 2
            if not context_budget_only(worker):
                continue
            entry = repair_plan(args, model, old, fingerprint["sha256"])
            if fix_fingerprint() != fingerprint:
                record("source_changed_no_repair")
                return 2
            if rpc is None:
                rpc = CodexRPC(dict(os.environ))
                gate = QuotaGate(rpc, args.out / "launch-quota.jsonl", 1, time.monotonic())
            if not gate.check(force=True, starting=True):
                record("quota_stopped_no_repair", stop_reason=gate.reason)
                return 2
            # Persist consumption before any launch or archive operation. A crash
            # never grants a second attempt under this same source-fix hash.
            entry["status"] = "running"
            marker.parent.mkdir(parents=True, exist_ok=True)
            with marker.open("x") as stream:
                stream.write(json.dumps(entry, indent=2) + "\n")
            state["attempts"][model] = entry
            write_json(args.out / f"{model}-invocation.json", entry)
            record("repair_running")
            repaired = execute_repair(args, entry)
            audit_path = Path(entry["output"]) / "evaluation-audit.json"
            audit = read_json(audit_path)
            pure = context_budget_only(repaired) and audit.get("reason") in {
                None, "protocol_invalid", "worker_protocol_invalid", "context_budget_exceeded"}
            if pure:
                entry["status"] = "budget_limitation_after_repair"
                write_json(audit_path, {"valid": False, "classification": "context_budget_exceeded",
                    "reason": "context_budget_exceeded", "budget_limitation_after_repair": True,
                    "repair_fix_hash": fingerprint["sha256"], "benchmark_score": None})
            elif repaired.get("protocol_valid") and repaired.get("status") in {"completed", "stopped"}:
                entry["status"] = "repair_completed"
            else:
                entry["status"] = "repair_failed_no_further_retry"
            write_json(marker, entry)
            record("repair_finished")
            if quota_stopped(repaired):
                record("quota_stopped_no_repair")
                return 2
        failed = any(entry["status"] == "repair_failed_no_further_retry" for entry in state["attempts"].values())
        record("finished_with_failures" if failed else "finished")
        return 1 if failed else 0
    except Exception as exc:
        record("repair_failed", error=f"{type(exc).__name__}: {exc}")
        return 2
    finally:
        if rpc is not None:
            rpc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("science", "browse", "arc"), required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-session", action="append", required=True)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 24:
        parser.error("--max-hours must be in (0, 24]")
    args.scratch, args.pilot_root, args.out = args.scratch.resolve(), args.pilot_root.resolve(), args.out.resolve()
    args.pilot_root.relative_to(args.scratch)
    args.out.relative_to(args.pilot_root)
    args.python = args.python.absolute()
    lock_path = args.pilot_root / args.benchmark / ".clm-repair.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())
