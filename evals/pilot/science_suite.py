"""Run and grade one frozen Science task with an isolated Relay worker.

The worker receives only the public task. Grading starts after it has exited,
and uses an immutable copy of its submitted solver. A failed preflight is not
reported as a benchmark failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from evals.pilot.repair_clm import context_budget_only
from evals.pilot.failure_policy import UPSTREAM_STREAM_INTERRUPTED, upstream_stream_interrupted


REPO = Path(__file__).resolve().parents[2]
# The evaluation's working tree (data, runtimes and run outputs); see evals/pilot/README.md.
DEFAULT_SCRATCH = Path(os.environ.get("RELAY_PILOT_ROOT", "relay-pilot"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def quota_stop_reason(metadata: dict[str, Any]) -> str | None:
    """Account or quota-metadata failures invalidate a partially finished task."""
    for key in ("stop_reason", "error"):
        reason = str(metadata.get(key) or "")
        if any(term in reason.lower() for term in ("quota", "credit", "ordinary_plan", "account_limit")):
            return reason
    return None


def classify_worker(worker: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Recognize provider aborts from runners started before this classifier."""
    metadata = dict(metadata)
    if quota_stop_reason(metadata):
        return metadata
    failed = None
    events = worker / "codex.jsonl"
    if events.is_file():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.failed":
                failed = event.get("error", {}).get("message") or str(event)
    if metadata.get("status") != "running" and not metadata.get("stop_reason") and (
        metadata.get("exit_code") not in (None, 0) or failed
    ):
        metadata.update(status="infrastructure_failed", error=failed or "worker exited unsuccessfully",
                        protocol_valid=False)
    if metadata.get("status") == "infrastructure_failed":
        message = " ".join(str(metadata.get(key) or "") for key in ("error", "stop_reason"))
        if failed:
            message += " " + failed
        metadata["failure_kind"] = "provider_capacity" if "capacity" in message.lower() else "infrastructure_failed"
    if upstream_stream_interrupted(metadata, worker / "relay/runtime.log"):
        metadata.update(status="infrastructure_failed", protocol_valid=False,
                        failure_kind=UPSTREAM_STREAM_INTERRUPTED)
    return metadata


def run_one(args: argparse.Namespace) -> int:
    scratch = args.scratch.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    # All old private references and all new workers must remain under the
    # hidden scratch root. A second output tree needs an explicit deny rule.
    output.relative_to(scratch)
    prompt_source = (scratch / "science/runs/fixed-python-compaction-01/work/task.txt")
    public = scratch / "science/public/environment"
    if not prompt_source.is_file() or not (public / ".deps/numpy/__init__.py").is_file():
        raise FileNotFoundError("Prepare the frozen Science task and runtime first")
    output.mkdir(parents=True, mode=0o700)
    prompt = output / "prompt.txt"
    shutil.copyfile(prompt_source, prompt)
    worker = output / "worker"
    python = args.python.absolute()  # Keep the virtualenv interpreter symlink.
    command = [
        str(python), str(REPO / "evals/pilot/run.py"),
        "--root", str(scratch), "--out", str(worker),
        "--prompt", str(prompt), "--public", str(public),
        "--model", args.model, "--strategy", args.strategy,
        "--window", "32000", "--max-requests", "64", "--seconds", "1800",
    ]
    result: dict[str, Any] = {
        "benchmark": "Terminal-Bench-Science", "task": "traffic-flux-inversion",
        "model": args.model, "strategy": args.strategy, "status": "worker_running",
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "worker": str(worker), "window": 32000, "max_requests": 64,
        "seconds": 1800, "worker_command": command,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "started_unix": time.time(),
    }
    write_json(output / "result.json", result)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    with (output / "worker-launcher.log").open("w") as log:
        run = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    result["worker_exit_code"] = run.returncode
    if not (worker / "run.json").exists():
        result.update(status="worker_launcher_failed", finished_unix=time.time())
        write_json(output / "result.json", result)
        return 1
    metadata = classify_worker(worker, json.loads((worker / "run.json").read_text()))
    result["worker_status"] = metadata
    if reason := quota_stop_reason(metadata):
        result.update(status="worker_quota_stopped", stop_reason=reason,
                      benchmark_score=None, finished_unix=time.time())
        write_json(output / "evaluation-audit.json", {
            "valid": False, "classification": "quota_stopped",
            "reason": reason, "benchmark_score": None,
        })
        write_json(output / "result.json", result)
        return 2
    artifact = worker / "work/traffic_solver.py"
    if context_budget_only(metadata):
        result.update(status="worker_context_budget_exceeded", finished_unix=time.time())
        write_json(output / "evaluation-audit.json", {
            "valid": False, "classification": "context_budget_exceeded",
            "reason": "context_budget_exceeded", "benchmark_score": None,
        })
        write_json(output / "result.json", result)
        return 0
    if metadata.get("status") in {"preflight_failed", "infrastructure_failed"} or not metadata.get("harness_requests"):
        status = ("worker_infrastructure_failed" if metadata.get("status") == "infrastructure_failed"
                  else "worker_preflight_failed")
        result.update(status=status, finished_unix=time.time())
        if status == "worker_infrastructure_failed":
            write_json(output / "evaluation-audit.json", {
                "valid": False, "classification": "infrastructure_failed",
                "reason": metadata.get("failure_kind", "infrastructure_failed"),
                "detail": metadata.get("error"), "benchmark_score": None,
            })
        write_json(output / "result.json", result)
        return 1
    if metadata.get("protocol_valid") is False:
        result.update(status="worker_protocol_invalid", finished_unix=time.time())
        write_json(output / "evaluation-audit.json", {
            "valid": False, "classification": "protocol_invalid",
            "reason": "worker_protocol_invalid", "benchmark_score": None,
        })
        write_json(output / "result.json", result)
        return 1
    if not artifact.is_file():
        result.update(status="missing_submission", finished_unix=time.time())
        write_json(output / "result.json", result)
        return 1
    frozen = output / "frozen/traffic_solver.py"
    frozen.parent.mkdir()
    shutil.copyfile(artifact, frozen)
    frozen.chmod(0o400)
    result["artifact_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    result["artifact"] = str(frozen)
    verifier = output / "verifier"
    verify_command = [str(python), str(REPO / "evals/pilot/science.py"),
                      "verify", "--artifact", str(frozen), "--run-dir", str(verifier)]
    result.update(status="grading", verifier_command=verify_command)
    write_json(output / "result.json", result)
    with (output / "verifier-launcher.log").open("w") as log:
        grade = subprocess.run(verify_command, cwd=REPO, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    result["verifier_exit_code"] = grade.returncode
    verifier_result = verifier / "verifier-result.json"
    if verifier_result.exists():
        result["verifier"] = json.loads(verifier_result.read_text())
        result["status"] = result["verifier"].get("status", "verifier_failed")
    else:
        result["status"] = "verifier_launcher_failed"
    result["finished_unix"] = time.time()
    write_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    # Nonzero benchmark reward is a valid terminal result, not a suite failure.
    return 0 if result["status"] in {"completed", "completed_failed", "timeout"} else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("gpt-6-luna", "gpt-6.1-sol"))
    parser.add_argument("--strategy", required=True)
    return run_one(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
