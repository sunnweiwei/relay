"""Run a fixed list of BrowseComp-Plus cells (question, method) under the evaluation's condition.

Each cell runs `browse_suite.py` exactly as the full-population queue did (the same prompt,
window, caps, Browse profile and overflow fallback), in a fresh run directory with its own
retrieval service, and is graded by `browse_grade.py` after it ends. The queue state is written
after every change. A cell that was started but not recorded as finished is never retried
automatically. The quota guard of `run.py` polls plan usage; once a run stops on it, no further
cell is admitted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from evals.pilot.browse_grade import grade, quota_stop_reason

MODEL = "gpt-6-luna"
REPO = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    temporary.replace(path)


def worker_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO),
               RELAY_PILOT_ALL_CALL_BUDGET="1", RELAY_PILOT_EXPECTED_EFFORT="medium",
               RELAY_PILOT_INPUT_SLACK="2356", RELAY_PILOT_OVERFLOW_FALLBACK="truncate",
               RELAY_PILOT_QUOTA_MODE="poll",
               RELAY_PILOT_QUOTA_START_PERCENT=str(args.quota_start_percent),
               RELAY_PILOT_QUOTA_STOP_PERCENT=str(args.quota_stop_percent),
               RELAY_PILOT_EXTRA_DENY_PATHS=json.dumps([str(p) for p in args.deny_path]))
    return env


def run(args: argparse.Namespace) -> int:
    manifest = {str(t["query_id"]): t for t in json.loads(args.manifest.read_text())["tasks"]}
    wanted = [(str(task), method) for task, method in json.loads(args.cells.read_text())]
    if len(set(wanted)) != len(wanted) or any(task not in manifest for task, _ in wanted):
        raise ValueError("cells must be unique questions of the manifest")
    state_path = args.out / "queue-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if [(c["task_id"], c["method"]) for c in state["cells"]] != wanted:
            raise ValueError("the cell list changed since the queue was created")
    else:
        state = {"benchmark": "BrowseComp-Plus", "model": MODEL, "runtime": str(REPO),
                 "prompt": str(args.prompt), "manifest": str(args.manifest), "status": "ready",
                 "cells": [{"task_id": t, "method": m, "status": "not_started"} for t, m in wanted]}
    for cell in state["cells"]:
        if cell["status"] == "running":
            cell["status"] = "interrupted"  # started before a restart: kept, never retried
    lock, halted = threading.Lock(), threading.Event()

    def save() -> None:
        write_json(state_path, state)

    def launch(cell: dict[str, Any], slot: int) -> None:
        task_id, method = cell["task_id"], cell["method"]
        task = manifest[task_id]
        run_dir = args.out / MODEL / task_id / method
        service_dir = args.out / "services" / MODEL / task_id / method
        log_path = args.out / "launcher-logs" / task_id / f"{method}.log"
        command = [str(args.python), str(REPO / "evals/pilot/browse_suite.py"),
                   "--data-root", str(args.root / "browse"), "--hidden-root", str(args.root),
                   "--python", str(args.python), "--tmp-dir", str(args.tmp_dir),
                   "--model", MODEL, "--strategy", method, "--port", str(args.base_port + slot),
                   "--query-id", task_id, "--question-file", task["question_file"],
                   "--prompt-file", str(args.prompt), "--out", str(run_dir),
                   "--service-root", str(service_dir)]
        with lock:
            cell.update(status="running", slot=slot, started_unix=time.time())
            save()
        try:
            if run_dir.exists() or service_dir.exists():
                raise ValueError("refusing to overwrite an earlier attempt")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as log:
                proc = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                      env=worker_env(args), check=False)
            raw = json.loads((run_dir / "run.json").read_text())
            scored = grade(run_dir, service_dir / "retrieval.jsonl", Path(task["reference_file"]),
                           expected_query_id=task_id)
            write_json(run_dir / "browse-grade.json", scored)
            reason = quota_stop_reason(raw)
            with lock:
                cell.update(status="finished", finished_unix=time.time(), exit_code=proc.returncode,
                            raw_run=str(run_dir / "run.json"), score_file=str(run_dir / "browse-grade.json"),
                            worker_status=raw.get("status"), stop_reason=raw.get("stop_reason"),
                            failure_kind=raw.get("failure_kind"), quota_stop_reason=reason,
                            raw_score=scored.get("strict_answer_score"),
                            quota_used_percent=raw.get("quota_used_percent"))
                if reason:
                    state["halt_reason"] = reason
                    halted.set()
                save()
        except Exception as exc:
            with lock:
                cell.update(status="controller_error", error=f"{type(exc).__name__}: {exc}")
                state["halt_reason"] = "controller_error"
                halted.set()
                save()

    todo = [cell for cell in state["cells"] if cell["status"] == "not_started"]
    state.update(status="running", started_unix=state.get("started_unix", time.time()))
    state.pop("halt_reason", None)
    save()
    free = list(range(args.jobs))
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        active: dict[concurrent.futures.Future[None], int] = {}
        queue = iter(todo)
        while True:
            while free and not halted.is_set() and (cell := next(queue, None)) is not None:
                slot = free.pop(0)
                active[pool.submit(launch, cell, slot)] = slot
            if not active:
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                free.append(active.pop(future))
    finished = all(cell["status"] == "finished" for cell in state["cells"])
    state.update(status="complete" if finished else "halted", ended_unix=time.time())
    save()
    return 0 if finished else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "manifest", "prompt", "cells", "python", "tmp-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--base-port", type=int, default=23800)
    parser.add_argument("--quota-start-percent", type=float, default=48)
    parser.add_argument("--quota-stop-percent", type=float, default=50)
    parser.add_argument("--deny-path", type=Path, action="append", default=[],
                        help="Further paths hidden from the agent (for example source checkouts)")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
