"""Run one sampled Science task in official task and verifier images.

Inference is refused unless the task has an independently validated readiness
record. Image construction alone does not satisfy that requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

from evals.pilot.science_container import (
    HERE, cpu_prefix, freeze_artifacts, portable_python_prefix, start_service, stop_process, write_json,
)
from evals.pilot.science_suite import classify_worker, quota_stop_reason
from evals.pilot.repair_clm import context_budget_only


REPO = HERE.parents[1]
CONTROLLED_RUNTIME_FILES = ("science_container.py", "science_service.py", "science_client.py",
                            "science_sample_suite.py", "science_isolation.py")


def verify(task: Path, images: Path, frozen: Path, output: Path, python: Path) -> dict[str, Any]:
    """Run the original test.sh with declared artifacts and the validated shim."""
    from evals.pilot.science_isolation import create_bundle

    config = tomllib.loads((task / "task.toml").read_text())
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    artifacts = output / "artifacts"
    shutil.copytree(frozen, artifacts)
    task_id = config["task"]["name"]
    if "/" not in task_id:
        task_id = "terminal-bench-science/" + task_id
    bundle = create_bundle(task_id, output / "isolation")
    bundle["environment"] = verifier_environment(task_id, bundle["environment"])
    write_json(Path(bundle["bundle_dir"]) / "bundle.json", bundle)
    python_root = portable_python_prefix(python)
    workdir = tests_image_workdir(images)
    verifier_console_log = verifier_console_log_path(task_id, output, logs)
    command = ["apptainer", "exec", *fakeroot_args(task_id), "--writable-tmpfs", "--cleanenv",
               "--containall", "--no-home", "--no-mount", "hostfs,bind-paths",
               "--net", "--network=none", "--pwd", workdir,
               "--bind", str(logs) + ":/logs/verifier",
               "--bind", str(bundle["bundle_dir"]) + ":/opt/relay-isolation:ro",
               "--bind", "/usr/bin/bwrap:/opt/relay-tools/bwrap:ro",
               "--bind", "/dev/full:/dev/full",
               "--bind", str(python_root) + ":/opt/relay-runtime:ro"]
    for record in json.loads((artifacts / "artifacts.json").read_text()):
        if record["present"]:
            source = artifacts / Path(record["path"]).relative_to("/")
            command += ["--bind", str(source) + ":" + record["path"]]
    for key, value in bundle["environment"].items():
        command += ["--env", key + "=" + str(value)]
    command += official_test_command(config["task"]["name"], images / "tests.sif")
    cpus = int(config["verifier"].get("environment", {}).get("cpus", config["environment"]["cpus"]))
    command = cpu_prefix(cpus) + command
    seconds = float(config["verifier"]["timeout_sec"])
    result: dict[str, Any] = {"status": "running", "phase": "official_test_sh", "reward": None,
                             "task_id": config["task"]["name"], "timeout": False,
                             "started_unix": time.time(), "timeout_seconds": seconds, "command": command}
    write_json(output / "verifier-result.json", result)
    with verifier_console_log.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            stop_process(process)
            result.update(status="timeout", timeout=True, finished_unix=time.time())
            write_json(output / "verifier-result.json", result)
            return result
    reward = None
    reward_file = logs / "reward.txt"
    if reward_file.exists():
        try:
            reward = float(reward_file.read_text().strip())
        except ValueError:
            pass
    reward_json = logs / "reward.json"
    if reward_json.exists():
        result["reward_metrics"] = json.loads(reward_json.read_text())
    result.update(exit_code=exit_code, reward=reward, finished_unix=time.time())
    verifier_log = verifier_console_log.read_text(errors="replace")
    infrastructure_lines = [line for line in verifier_log.splitlines() if (
        line.startswith(("bwrap:", "FATAL:", "Science isolation initialization failed:"))
        or "Missing public candidate input:" in line
        or "Missing candidate output/scratch path:" in line)]
    if infrastructure_lines:
        result.update(status="verifier_failed", reward=None, phase="isolation",
                      infrastructure_errors=infrastructure_lines)
    elif reward is None:
        result.update(status="verifier_failed", phase="official_test_sh")
    else:
        result.update(status="completed" if reward > 0 else "completed_failed", phase="complete")
    write_json(output / "verifier-result.json", result)
    return result


def tests_image_workdir(images: Path) -> str:
    """Use the tests image's declared Docker WORKDIR for its official test.sh."""
    build_path = images / "build.json"
    if not build_path.is_file():
        return "/"
    try:
        build = json.loads(build_path.read_text())
        workdir = build["images"]["tests"].get("workdir")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return "/"
    if (not isinstance(workdir, str) or not workdir.startswith("/")
            or os.path.normpath(workdir) != workdir or "\x00" in workdir):
        raise ValueError("tests image workdir must be an absolute normalized path")
    return workdir


def verifier_console_log_path(task_id: str, output: Path, logs: Path) -> Path:
    """Keep Leaky-Bloch console capture outside the test script's cleaned log mount."""
    if task_id.endswith("/leaky-bloch-meep"):
        return output / "verifier-console.log"
    return logs / "verifier.log"


def verifier_environment(task_id: str, environment: dict[str, str]) -> dict[str, str]:
    """Retain the pinned tests image's extra interpreter path for Leaky-Bloch."""
    result = dict(environment)
    if task_id.endswith("/leaky-bloch-meep"):
        result["PATH"] = "/opt/conda/envs/pymeep-runtime/bin:/opt/conda/bin:" + result["PATH"]
    return result


def fakeroot_args(task_id: str) -> list[str]:
    """Keep root mapping while skipping an incompatible injected fakeroot binary."""
    if task_id == "terminal-bench-science/leaky-bloch-meep":
        return ["--fakeroot", "--ignore-fakeroot-command"]
    return ["--fakeroot"]


def official_test_command(task_id: str, test_image: Path) -> list[str]:
    """Return the official test entrypoint, adding a PID boundary where needed."""
    if task_id == "terminal-bench-science/traffic-flux-inversion":
        # This pinned test.sh runs `su nobody; kill -9 -1` after the candidate.
        # Under root-mapped fakeroot, su's apparent nobody UID maps to the
        # trusted shell's kernel UID. Make the test script PID 1 so that its
        # cleanup cannot signal the verifier process itself.
        return [str(test_image), "/usr/bin/unshare", "--pid", "--fork", "--mount-proc",
                "/bin/bash", "/tests/test.sh"]
    return [str(test_image), "/bin/bash", "/tests/test.sh"]


def require_readiness(task: Path, images: Path) -> dict[str, Any]:
    config = tomllib.loads((task / "task.toml").read_text())
    ready_path = images / "readiness.json"
    if not ready_path.exists():
        raise RuntimeError("Task images have no validated readiness record; refusing model inference")
    ready = json.loads(ready_path.read_text())
    if (ready.get("task_id") != config["task"]["name"] or ready.get("ready") is not True
            or not ready.get("positive_control_passed") or not ready.get("isolation_control_passed")
            or not ready.get("public_environment_passed")):
        raise RuntimeError("Task readiness record does not validate environment, positive, and isolation controls")
    for kind in ("environment", "tests"):
        image = images / (kind + ".sif")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if ready.get("image_sha256", {}).get(kind) != digest:
            raise RuntimeError("Validated image differs from the image about to run")
    for name in CONTROLLED_RUNTIME_FILES:
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if ready.get("runtime_sha256", {}).get(name) != digest:
            raise RuntimeError(f"Validated runtime differs from current {name}")
    return ready


def run_one(args: argparse.Namespace) -> int:
    task, images, output = args.task.resolve(), args.images.resolve(), args.out.resolve()
    scratch = args.scratch.resolve()
    worker_seconds = int(getattr(args, "seconds", 1800))
    max_requests = int(getattr(args, "max_requests", 64))
    if worker_seconds < 1 or max_requests < 1:
        raise ValueError("Science worker seconds and request cap must be positive")
    output.relative_to(scratch)
    if output.exists():
        raise FileExistsError(output)
    ready = require_readiness(task, images)
    config = tomllib.loads((task / "task.toml").read_text())
    build = json.loads((images / "build.json").read_text())
    output.mkdir(parents=True, mode=0o700)
    result: dict[str, Any] = {"benchmark": "Terminal-Bench-Science", "task": config["task"]["name"],
                             "task_id": config["task"]["name"], "model": args.model,
                             "strategy": args.strategy, "status": "preparing", "window": 32000,
                             "max_requests": max_requests, "seconds": worker_seconds, "started_unix": time.time(),
                             "readiness": ready}
    write_json(output / "result.json", result)
    try:
        service, endpoint = start_service(config["task"]["name"], images / "environment.sif",
                                          build["images"]["environment"]["workdir"], output / "service", args.python,
                                          cpus=int(config["environment"]["cpus"]))
    except Exception as error:
        result.update(status="worker_preflight_failed", error=str(error), finished_unix=time.time())
        write_json(output / "result.json", result)
        return 1
    try:
        public = output / "client"
        public.mkdir()
        shutil.copyfile(HERE / "science_client.py", public / "science_client.py")
        write_json(public / "science-endpoint.json", endpoint)
        prompt = output / "prompt.txt"
        prompt.write_text(
            "Solve the following Terminal-Bench-Science task in its isolated scientific environment.\n\n"
            "All paths in the official instruction refer to that environment. Execute Bash commands there with "
            "`python3 science_client.py --command 'COMMAND'`. Use `--cwd /absolute/path` to set a command's directory "
            "and `--timeout SECONDS` for at most 300 seconds per call. Files and background processes persist across "
            "calls. Write the requested artifacts at their official paths. The local Codex workspace contains only "
            "this client and is suitable for your notes. It is not the scientific environment. Do not inspect files "
            f"outside the local workspace or use other services. Your evaluation runner allows {worker_seconds} seconds and "
            f"{max_requests} model requests with a 32000-token main-agent context budget; longer time limits in the original "
            "task instruction do not extend this budget.\n\n" +
            (task / "instruction.md").read_text())
        worker = output / "worker"
        command = [str(args.python.absolute()), str(HERE / "run.py"), "--root", str(scratch),
                   "--out", str(worker), "--prompt", str(prompt), "--public", str(public),
                   "--model", args.model, "--strategy", args.strategy, "--window", "32000",
                   "--max-requests", str(max_requests), "--seconds", str(worker_seconds)]
        result.update(status="worker_running", worker=str(worker), worker_command=command,
                      prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest())
        write_json(output / "result.json", result)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO)
        with (output / "worker-launcher.log").open("w") as log:
            completed = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
        result["worker_exit_code"] = completed.returncode
    finally:
        stop_process(service)
    worker_file = output / "worker/run.json"
    if not worker_file.exists():
        result.update(status="worker_launcher_failed", finished_unix=time.time())
        write_json(output / "result.json", result)
        return 1
    metadata = classify_worker(output / "worker", json.loads(worker_file.read_text()))
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
    frozen = output / "frozen"
    result["artifacts"] = freeze_artifacts(output / "service/rootfs", config["artifacts"], frozen)
    # No worker process exists when verifier-only data becomes available.
    if context_budget_only(metadata):
        result.update(status="worker_context_budget_exceeded", finished_unix=time.time())
        write_json(output / "evaluation-audit.json", {"valid": False,
                   "classification": "context_budget_exceeded", "reason": "context_budget_exceeded"})
        write_json(output / "result.json", result)
        return 0
    if metadata.get("status") in {"preflight_failed", "infrastructure_failed"} or metadata.get("protocol_valid") is False:
        result.update(status="worker_infrastructure_failed", finished_unix=time.time())
        write_json(output / "evaluation-audit.json", {"valid": False,
                   "classification": "infrastructure_failed", "reason": metadata.get("failure_kind") or "worker_protocol_invalid"})
        write_json(output / "result.json", result)
        return 1
    result.update(status="grading")
    write_json(output / "result.json", result)
    try:
        verifier = verify(task, images, frozen, output / "verifier", args.python)
    except Exception as error:
        result.update(status="verifier_launcher_failed", error=str(error), finished_unix=time.time())
        write_json(output / "result.json", result)
        return 1
    result.update(status=verifier["status"], verifier=verifier, finished_unix=time.time())
    write_json(output / "result.json", result)
    return 0 if result["status"] in {"completed", "completed_failed", "timeout"} else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", choices=("gpt-6-luna", "gpt-6.1-sol"), required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--seconds", type=int, default=1800,
                        help="Per-cell main worker wall-time budget; verifier timeout remains official")
    parser.add_argument("--max-requests", type=int, default=64,
                        help="Per-cell main-agent request cap; shared input budget also covers auxiliary calls")
    return run_one(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
