"""Collect every ended same-condition ARC or Science cell of the full-population runs into one table.

Rows use the same fields as browse_context_collect, plus a benchmark-specific
partial score (ARC: fraction of game levels completed; Science: fraction of
verifier tests passed). Earlier attempts under other budgets are left out:
ARC keeps only its full queue (200 main calls, 3600 s); Science keeps only
attempts with 128 main calls, 7200 s and a 32,000-token window.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evals.pilot.outcomes import failure_cause, fallback_counts, outcome
from typing import Any

# The evaluation's working tree (data, runtimes and run outputs); see evals/pilot/README.md.
ROOT = Path(os.environ.get("RELAY_PILOT_ROOT", "relay-pilot"))
ARC = ROOT / "full-arc-20261030/queue-state.json"
SCIENCE = ROOT / "full-science-20261030"
SCIENCE_LIVE = SCIENCE / "inference-luna-medium-7200s-128main-media-resume-20261101-v2"
SCIENCE_CONDITION = {"max_requests": 128, "seconds": 7200, "window": 32000}
# The queue of 2026-11-06 for the two added methods (truncation, acm_forced).
SCIENCE_RERUN = ROOT / "reruns-20261106/science/queue-state.json"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def unmanaged_tokens(worker: Path) -> int | None:
    conversations = read_json(worker / "relay/conversations.json")
    if not isinstance(conversations, dict) or not conversations:
        return None
    return max(int(entry.get("whole") or 0) for entry in conversations.values()) or None


def run_fields(worker: Path) -> dict[str, Any]:
    run = read_json(worker / "run.json")
    run = run if isinstance(run, dict) else {}
    aux = run.get("auxiliary_input_tokens_per_request") or []
    main = run.get("provider_input_tokens_per_request") or []
    return fallback_counts(run) | {
        "run_status": run.get("status"), "stop_reason": run.get("stop_reason"),
        "unmanaged_tokens": unmanaged_tokens(worker),
        "harness_requests": run.get("harness_requests"),
        "peak_main_input": run.get("peak_provider_input_tokens"),
        "main_input_tokens": sum(main) if main else None,
        "auxiliary_input_tokens": sum(aux), "auxiliary_requests": len(aux),
        "context_changes": run.get("context_changes"),
        "elapsed_seconds": run.get("elapsed_seconds"),
    }


def arc_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = json.loads(ARC.read_text())
    rows = []
    for cell in state["queue"]:
        attempt = cell["attempts"][-1]
        run_dir = Path(attempt["run_dir"])
        card = read_json(run_dir / "official_scorecard.json") or {}
        env = (card.get("environments") or [{}])[0]
        levels, total = env.get("levels_completed"), env.get("level_count")
        scored = cell.get("method_score") is not None
        worker = run_dir / "worker"
        run = read_json(worker / "run.json")
        cause = None if scored else failure_cause(worker, run if isinstance(run, dict) else {})
        result = outcome(scored, cause)
        zero = 0 if result == "task_failure" else None  # (see outcomes.py)
        rows.append({
            "benchmark": "arc", "task_id": cell["game_id"], "method": cell["method"],
            "outcome": result, "failure_cause": cause,
            # Official ARC score is 0-100; store it as a fraction like the other benchmarks.
            "method_score": cell["method_score"] / 100 if scored else zero,
            "success": (1 if env.get("completed") else 0) if scored else zero,
            "partial": (levels / total if total else None) if scored else zero,
            "levels_completed": levels if scored else None, "level_count": total,
            "game_actions": env.get("actions"),
            "failure_kind": attempt.get("stop_reason") if not scored else None,
            "attempts": len(cell["attempts"]), "run_dir": str(run_dir), "source": "arc-full-queue",
        } | run_fields(run_dir / "worker"))
    meta = {"status": state["status"], "counts": state["counts"], "caps": state["caps"],
            "excluded_prior_cells": len(state["prior_attempted_cells"]), "window": 32000}
    return rows, meta


def ctrf_fraction(attempt: Path) -> tuple[int | None, int | None]:
    ctrf = read_json(attempt / "verifier/logs/ctrf.json")
    summary = ((ctrf or {}).get("results") or {}).get("summary") or {}
    return summary.get("passed"), summary.get("tests")


def science_row(task_id: str, method: str, attempt: Path, source: str) -> dict[str, Any] | None:
    result = read_json(attempt / "result.json")
    if not isinstance(result, dict) or any(result.get(k) != v for k, v in SCIENCE_CONDITION.items()):
        return None
    verifier = result.get("verifier") or {}
    reward = verifier.get("reward")
    scored = reward is not None and not verifier.get("timeout")
    passed, tests = ctrf_fraction(attempt)
    run = read_json(attempt / "worker/run.json")
    cause = None if scored else failure_cause(
        attempt / "worker", run if isinstance(run, dict) else {},
        verifier_timeout=bool(verifier.get("timeout")) or result.get("status") == "timeout")
    status = outcome(scored, cause)
    zero = 0 if status == "task_failure" else None  # (see outcomes.py)
    return {
        "benchmark": "science", "task_id": task_id.split("/")[-1], "method": method,
        "outcome": status, "failure_cause": cause,
        "method_score": float(reward) if scored else zero,
        "success": (1 if float(reward) >= 1 else 0) if scored else zero,
        "partial": (passed / tests if tests else None) if scored else zero,
        "tests_passed": passed, "tests_total": tests,
        "failure_kind": None if scored else result.get("status"),
        "result_status": result.get("status"), "source": source, "run_dir": str(attempt),
    } | run_fields(attempt / "worker")


def science_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, skipped = [], {}
    prior = json.loads((SCIENCE_LIVE / "prior-attempt-index.json").read_text())
    for key, attempts in prior.items():
        task_id, method = key.split("::")
        last = attempts[-1]
        row = science_row(task_id, method, Path(last["path"]), last["source"])
        if row is None:
            skipped[last["source"]] = skipped.get(last["source"], 0) + 1
        else:
            rows.append(row)
    state = json.loads((SCIENCE_LIVE / "queue-state.json").read_text())
    for cell in state["cells"]:
        # Unscored cells (no verifier reward) are kept so method failures stay visible.
        if cell["status"] in ("finished", "unscored") and cell["attempt_paths"]:
            row = science_row(cell["task_id"], cell["method"], Path(cell["attempt_paths"][-1]), "media_resume_v2")
            if row is not None:
                rows.append(row)
    if SCIENCE_RERUN.is_file():
        for cell in json.loads(SCIENCE_RERUN.read_text())["cells"]:
            if cell["status"] in ("finished", "unscored") and cell["attempt_paths"]:
                row = science_row(cell["task_id"], cell["method"], Path(cell["attempt_paths"][-1]), "reruns_20261106")
                if row is not None:
                    rows.append(row)
    counts: dict[str, int] = {}
    for cell in state["cells"]:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    meta = {"status": state["status"], "counts": counts, "condition": state["condition"],
            "excluded_other_condition": skipped, "window": 32000,
            "active": [(c["task_id"].split("/")[-1], c["method"]) for c in state["active_cells"]]}
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["arc", "science"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, meta = arc_rows() if args.benchmark == "arc" else science_rows()
    keys = [(r["task_id"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("A task/method cell appears twice")
    args.out.write_text(json.dumps(meta | {"rows": rows}, indent=1) + "\n")
    print(f"{args.benchmark}: {len(rows)} rows; queue {meta['status']} {meta['counts']}")


if __name__ == "__main__":
    main()
