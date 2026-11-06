#!/usr/bin/env python3
"""Run a frozen ARC task across models and registered Relay strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from relay.core.budget import MEDIA_INPUT_BUDGET_COUNTER, InputBudget
from relay.strategies import STRATEGIES

from evals.pilot.failure_policy import (
    REVIEWED_TERMINAL_REASONS, UPSTREAM_STREAM_INTERRUPTED,
    reviewed_terminal_failure, upstream_stream_interrupted,
)


GAME_ID = "ls20-9607627b"
SEED = 0
PORT = 18761
MAX_ACTIONS = 300
MAX_REQUESTS = 100
WINDOW = 32_000
SECONDS = 1_800
ARMS = ("compaction", "clm")
MODELS = ("gpt-6-luna", "gpt-6.1-sol")


def audit_worker_failure(arm_dir: Path) -> dict[str, Any] | None:
    """Inspect raw worker termination, including results written by old runners."""
    from evals.pilot.repair_clm import quota_stopped

    result_path = arm_dir / "worker/run.json"
    if not result_path.exists():
        return None
    worker = json.loads(result_path.read_text())
    stop_reason = str(worker.get("stop_reason") or "")
    if quota_stopped(worker):
        quota_reason = next(str(worker[key]) for key in ("stop_reason", "error")
                            if quota_stopped({key: worker.get(key)}))
        audit = {"valid": False, "classification": "quota_stopped", "capacity_failure": False,
                 "reason": quota_reason, "benchmark_score": None}
        write_json(arm_dir / "evaluation-audit.json", audit)
        return audit
    enforced = ("model_request_limit", "wall_time_limit", "plan_quota", "ordinary_plan",
                "credit_balance", "account_limit", "quota_check", "context_budget_exceeded")
    if worker.get("status") == "running" or any(reason in stop_reason for reason in enforced):
        return None
    failures = []
    events = arm_dir / "worker/codex.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.failed":
                failures.append(event.get("error", {}))
    if worker.get("exit_code") in (None, 0) and not failures:
        return None
    message = str(worker.get("error") or (failures[-1] if failures else ""))
    capacity = "capacity" in message.lower()
    audit = {"valid": False, "classification": "infrastructure_failed",
             "reason": "provider_capacity" if capacity else "worker_process_failed",
             "capacity_failure": capacity, "exit_code": worker.get("exit_code")}
    write_json(arm_dir / "evaluation-audit.json", audit)
    return audit


def apply_worker_audit(arm_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    from evals.pilot.arc_server import observation_missing
    from evals.pilot.repair_clm import context_budget_only

    audit_path = arm_dir / "evaluation-audit.json"
    prior_audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    audit = audit_worker_failure(arm_dir)
    worker_path = arm_dir / "worker/run.json"
    worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
    trace = arm_dir / "service/arc_trace.jsonl"
    observations_checked = False
    if audit is None and trace.exists():
        missing = []
        parsed, malformed = 0, False
        for line in trace.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed = True
                continue
            parsed += 1
            observation = event.get("observation")
            if event.get("kind") == "step_failed" or observation_missing(observation):
                missing.append(event.get("action_count"))
        if missing:
            audit = {"valid": False, "classification": "infrastructure_failed",
                     "reason": "arc_observation_missing", "capacity_failure": False,
                     "missing_observation_actions": missing}
            write_json(arm_dir / "evaluation-audit.json", audit)
        observations_checked = parsed > 0 and not malformed and not missing
    if (audit is None and isinstance(prior_audit, dict) and prior_audit.get("valid") is False
            and prior_audit.get("reason") not in {"protocol_invalid", UPSTREAM_STREAM_INTERRUPTED}):
        # A specific prior defect must survive even if its evidence was removed.
        audit = prior_audit
    if audit is None and (worker.get("protocol_valid") is False
                          or summary.get("protocol_valid") is False):
        reason = "context_budget_exceeded" if context_budget_only(worker) else "protocol_invalid"
        if observations_checked and upstream_stream_interrupted(worker, arm_dir / "worker/relay/runtime.log"):
            reason = UPSTREAM_STREAM_INTERRUPTED
            if isinstance(prior_audit, dict) and prior_audit.get("reason") == "protocol_invalid":
                preserved = arm_dir / "evaluation-audit.before-upstream-stream-interrupted.json"
                if not preserved.exists():
                    write_json(preserved, prior_audit)
        audit = {"valid": False,
                 "classification": "context_budget_exceeded" if reason == "context_budget_exceeded" else "infrastructure_failed",
                 "reason": reason, "capacity_failure": False}
        write_json(arm_dir / "evaluation-audit.json", audit)
    if audit is not None:
        summary = {**summary, "worker_status": audit["classification"], "scored": False,
                   "error": audit["reason"], "capacity_failure": audit["capacity_failure"]}
        if audit["reason"] not in REVIEWED_TERMINAL_REASONS:
            write_json(arm_dir / "summary.json", summary)
    return summary


def read_json(url: str, token: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return parsed


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def clean_observation_trace(arm_dir: Path) -> bool:
    """Reviewed worker failures do not excuse an independent ARC service defect."""
    from evals.pilot.arc_server import observation_missing

    try:
        events = [json.loads(line) for line in (arm_dir / "service/arc_trace.jsonl").read_text().splitlines()]
        return bool(events) and all(isinstance(event, dict) and event.get("kind") != "step_failed"
                                    and not observation_missing(event.get("observation")) for event in events)
    except (OSError, ValueError):
        return False


def start_service(
    python: Path, server_script: Path, arc_cwd: Path, arm_dir: Path,
    port: int = PORT,
    game_id: str = GAME_ID,
    seed: int = SEED,
    max_actions: int = MAX_ACTIONS,
) -> tuple[subprocess.Popen[bytes], str]:
    service_dir = arm_dir / "service"
    service_dir.mkdir(parents=True, mode=0o700)
    token = secrets.token_urlsafe(36)
    token_path = service_dir / "score-token"
    token_path.write_text(token + "\n")
    token_path.chmod(0o600)
    log = (service_dir / "server.log").open("wb")
    command = [
        str(python), str(server_script), "--game-id", game_id, "--seed", str(seed),
        "--host", "127.0.0.1", "--port", str(port),
        "--trace", str(service_dir / "arc_trace.jsonl"),
        "--score-token-file", str(token_path), "--max-actions", str(max_actions),
    ]
    proc = subprocess.Popen(command, cwd=arc_cwd, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    (service_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError(f"ARC server exited with status {proc.returncode}")
        try:
            state = read_json(f"http://127.0.0.1:{port}/state")
            if state.get("game_id") == game_id and state.get("action_count") == 1:
                # A stale service on this port must not supply another run's game.
                read_json(f"http://127.0.0.1:{port}/score", token)
                return proc, token
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.1)
    stop_service(proc)
    raise TimeoutError("ARC server did not become ready")


def stop_service(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run_arm(
    arm: str,
    python: Path,
    runner: Path,
    server_script: Path,
    scratch: Path,
    arc_cwd: Path,
    prompt: Path,
    tag: str,
    *,
    model: str = MODELS[0],
    out_root: Path | None = None,
    port: int = PORT,
    game_id: str = GAME_ID,
    seed: int = SEED,
    max_requests: int = MAX_REQUESTS,
    seconds: int = SECONDS,
    max_actions: int = MAX_ACTIONS,
    input_budget_counter: str | None = None,
    no_capacity_retry: bool = False,
) -> dict[str, Any]:
    arm_dir = (out_root / model / arm if out_root is not None
               else scratch / "runs" / f"arc-{arm}-{tag}")
    arm_dir.mkdir(parents=True, mode=0o700)
    # The endpoint is the only permitted prompt change for a parallel service.
    source_prompt = prompt.read_text()
    effective_prompt = source_prompt.replace(
        f"http://127.0.0.1:{PORT}", f"http://127.0.0.1:{port}")
    run_prompt = arm_dir / "prompt.txt"
    run_prompt.write_text(effective_prompt)
    service, token = start_service(python, server_script, arc_cwd, arm_dir, port, game_id, seed,
                                   max_actions)
    run_process: subprocess.CompletedProcess[str] | None = None
    try:
        initial = read_json(f"http://127.0.0.1:{port}/state")
        run_command = [
            str(python), str(runner), "--root", str(scratch), "--out", str(arm_dir / "worker"),
            "--prompt", str(run_prompt), "--strategy", arm, "--model", model,
            "--window", str(WINDOW), "--max-requests", str(max_requests),
            "--seconds", str(seconds),
        ]
        if input_budget_counter is not None:
            run_command.extend(["--input-budget-counter", input_budget_counter])
        if out_root is not None:
            run_command.extend(["--deny-path", str(out_root)])
        write_json(arm_dir / "arm.json", {
            "arm": arm, "model": model, "game_id": game_id, "seed": seed,
            "action_limit": max_actions, "request_limit": max_requests,
            "window": WINDOW, "seconds": seconds, "service_port": port,
            "input_budget_counter": input_budget_counter or InputBudget.name,
            "no_capacity_retry": no_capacity_retry,
            "initial_action_count": initial["action_count"],
            "prompt_sha256": hashlib.sha256(run_prompt.read_bytes()).hexdigest(),
            "source_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
            "service_command_file": str(arm_dir / "service" / "command.json"),
        })
        (arm_dir / "worker_command.json").write_text(json.dumps(run_command, indent=2) + "\n")
        # Set the frozen local endpoint only in the prompt's environment context.
        env = os.environ.copy()
        env["ARC_SERVICE_URL"] = f"http://127.0.0.1:{port}"
        run_process = subprocess.run(run_command, env=env, text=True, capture_output=True)
        (arm_dir / "runner.stdout").write_text(run_process.stdout)
        (arm_dir / "runner.stderr").write_text(run_process.stderr)
        try:
            state = read_json(f"http://127.0.0.1:{port}/state")
            score = read_json(f"http://127.0.0.1:{port}/score", token)
            write_json(arm_dir / "final_state.json", state)
            write_json(arm_dir / "official_scorecard.json", score)
            (arm_dir / "official_scorecard.json").chmod(0o600)
        except Exception as exc:
            write_json(arm_dir / "collection_error.json", {
                "error": f"{type(exc).__name__}: {exc}",
            })
        (arm_dir / "service" / "score-token").chmod(0o600)
        result_path = arm_dir / "worker" / "run.json"
        worker = json.loads(result_path.read_text()) if result_path.exists() else {}
        summary = {
            "arm": arm, "model": model, "game_id": game_id, "seed": seed,
            "runner_returncode": run_process.returncode,
            "run_json": str(result_path),
            "worker_status": worker.get("status", "missing_result"),
            "stop_reason": worker.get("stop_reason"),
            "error": worker.get("error"),
            "protocol_valid": worker.get("protocol_valid"),
            "action_limit": max_actions, "request_limit": max_requests, "seconds_limit": seconds,
            "input_budget_counter": input_budget_counter or InputBudget.name,
            # A no-inference setup failure is not a benchmark score of zero.
            "scored": worker.get("status") in {"completed", "stopped"}
                      and worker.get("protocol_valid", True)
                      and (arm_dir / "official_scorecard.json").is_file(),
            "service_trace": str(arm_dir / "service" / "arc_trace.jsonl"),
            "final_state": str(arm_dir / "final_state.json"),
            "scorecard": str(arm_dir / "official_scorecard.json"),
        }
        write_json(arm_dir / "summary.json", summary)
        summary = apply_worker_audit(arm_dir, summary)
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        stop_service(service)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--tag", default="20261023")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=list(STRATEGIES), default=list(ARMS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=[MODELS[0]])
    parser.add_argument("--out-root", type=Path, help="Fresh matrix output root; scratch data stay read-only")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--game-id", default=GAME_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    parser.add_argument("--seconds", type=int, default=SECONDS)
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--input-budget-counter", choices=(InputBudget.name, MEDIA_INPUT_BUDGET_COUNTER),
                        help="Input counter passed to the worker; omitted keeps the text-only default")
    parser.add_argument("--no-capacity-retry", action="store_true",
                        help="Do not archive and rerun an arm after provider-capacity failure")
    parser.add_argument("--arc-cwd", type=Path, help="Directory containing the pinned offline games")
    parser.add_argument("--prompt", type=Path, help="Frozen observation/action protocol prompt")
    parser.add_argument("--resume", action="store_true", help="Skip arms with a completed summary")
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    out_root = args.out_root.resolve() if args.out_root else None
    if len(args.models) > 1 and out_root is None:
        parser.error("--out-root is required for a multi-model matrix")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if args.max_requests < 1 or args.seconds < 1 or not 1 <= args.max_actions <= MAX_ACTIONS:
        parser.error(f"Budgets must satisfy requests>=1, seconds>=1, actions in [1, {MAX_ACTIONS}]")
    arc_cwd = args.arc_cwd.resolve() if args.arc_cwd else scratch / "arc"
    # Preserve the venv's python symlink; resolving it can escape the venv.
    python = args.python.absolute()
    here = Path(__file__).resolve().parent
    runner = here / "run.py"
    server_script = here / "arc_server.py"
    prompt = args.prompt.resolve() if args.prompt else arc_cwd / "frozen_prompt.txt"
    if not prompt.is_file():
        raise FileNotFoundError(f"Create and freeze the shared prompt at {prompt}")
    digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
    freeze_root = out_root if out_root else arc_cwd
    freeze_root.mkdir(parents=True, exist_ok=True)
    freeze_path = freeze_root / "prompt-freeze.json"
    freeze = {
        "sha256": digest, "game_id": args.game_id, "seed": args.seed,
        "arms": list(args.arms), "models": args.models, "service_port": args.port,
        "max_requests": args.max_requests, "seconds": args.seconds,
        "max_actions": args.max_actions,
        "frozen_before_first_run": True,
    }
    if args.input_budget_counter is not None:
        freeze["input_budget_counter"] = args.input_budget_counter
    if args.no_capacity_retry:
        freeze["no_capacity_retry"] = True
    if freeze_path.exists() and args.resume:
        if json.loads(freeze_path.read_text()) != freeze:
            parser.error("Resume configuration differs from the frozen matrix")
    else:
        write_json(freeze_path, freeze)
    for arm in args.arms:
        for model in args.models:
            arm_dir = (out_root / model / arm if out_root else
                       scratch / "runs" / f"arc-{arm}-{args.tag}")
            summary_path = arm_dir / "summary.json"
            if args.resume and summary_path.is_file():
                summary = apply_worker_audit(arm_dir, json.loads(summary_path.read_text()))
                print(json.dumps({"event": "resumed", **summary}), flush=True)
            else:
                summary = run_arm(arm, python, runner, server_script, scratch,
                                  arc_cwd, prompt, args.tag, model=model,
                                  out_root=out_root, port=args.port,
                                  game_id=args.game_id, seed=args.seed,
                                  max_requests=args.max_requests, seconds=args.seconds,
                                  max_actions=args.max_actions,
                                  input_budget_counter=args.input_budget_counter,
                                  no_capacity_retry=args.no_capacity_retry)
            failure = " ".join(str(summary.get(key) or "") for key in ("stop_reason", "error"))
            if summary.get("worker_status") == "quota_stopped" or any(reason in failure for reason in (
                "plan_quota", "ordinary_plan", "credit_balance", "account_limit", "quota_check",
            )):
                print(json.dumps({"event": "matrix_stopped", "reason": failure}), flush=True)
                return 2
            if (arm_dir / "continuation-audit.json").exists():
                reviewed_reason = reviewed_terminal_failure(arm_dir)
                if (reviewed_reason and summary.get("error") == reviewed_reason
                        and clean_observation_trace(arm_dir)):
                    print(json.dumps({"event": "reviewed_terminal_failure_no_retry",
                                      "reason": reviewed_reason, "model": model, "arm": arm}), flush=True)
                    continue
                print(json.dumps({"event": "matrix_stopped", "reason": "continuation_review_rejected"}), flush=True)
                return 2
            if (summary.get("capacity_failure") and out_root is not None
                    and not args.no_capacity_retry):
                retry_path = out_root / "capacity-retries.json"
                retries = json.loads(retry_path.read_text()) if retry_path.exists() else {}
                key = f"{model}/{arm}"
                if retries.get(key, 0) < 1:
                    retries[key] = 1
                    write_json(retry_path, retries)
                    archive_root = (out_root.parent / "diagnostics/arc-capacity-attempt1"
                                    if args.game_id == GAME_ID else
                                    out_root.parent / "diagnostics/arc-capacity-attempt1" / args.game_id)
                    archived = archive_root / model / arm
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    if archived.exists():
                        raise FileExistsError(f"Refusing to overwrite archived attempt: {archived}")
                    shutil.move(str(arm_dir), archived)
                    summary = run_arm(arm, python, runner, server_script, scratch,
                                      arc_cwd, prompt, args.tag, model=model,
                                      out_root=out_root, port=args.port,
                                      game_id=args.game_id, seed=args.seed,
                                      max_requests=args.max_requests, seconds=args.seconds,
                                      max_actions=args.max_actions,
                                      input_budget_counter=args.input_budget_counter,
                                      no_capacity_retry=args.no_capacity_retry)
                if summary.get("capacity_failure"):
                    print(json.dumps({"event": "capacity_retry_exhausted", "model": model, "arm": arm}), flush=True)
                    continue
            if summary.get("capacity_failure") and args.no_capacity_retry:
                print(json.dumps({"event": "capacity_retry_disabled", "model": model, "arm": arm}), flush=True)
                continue
            failure = " ".join(str(summary.get(key) or "") for key in ("stop_reason", "error"))
            if summary.get("worker_status") == "quota_stopped" or any(reason in failure for reason in (
                "plan_quota", "ordinary_plan", "credit_balance", "account_limit", "quota_check",
            )):
                print(json.dumps({"event": "matrix_stopped", "reason": failure}), flush=True)
                return 2
            if summary.get("worker_status") == "context_budget_exceeded":
                print(json.dumps({"event": "method_context_budget_exceeded", "model": model, "arm": arm}), flush=True)
                continue
            if summary.get("error") == UPSTREAM_STREAM_INTERRUPTED:
                worker_path = arm_dir / "worker/run.json"
                worker = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
                if upstream_stream_interrupted(worker, arm_dir / "worker/relay/runtime.log"):
                    print(json.dumps({"event": "upstream_stream_interrupted_no_retry",
                                      "model": model, "arm": arm}), flush=True)
                    continue
            if summary.get("worker_status") in {"infrastructure_failed", "preflight_failed", "missing_result"}:
                print(json.dumps({"event": "matrix_stopped", "reason": failure or summary["worker_status"]}), flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
