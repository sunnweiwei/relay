"""Collect every ended Luna Browse cell of the full-population run into one table.

The full run is the 429-exempt queue plus the prior cells its plan carried over.
Prior cells that the plan selected for a pure-429 retry are replaced by the
retry. The rerun queue of 2026-11-06 (``browse_queue.py``) adds two methods and
reruns two kinds of earlier cells: those excluded as infrastructure failures and
those in which the Browse profile was not applied. A rerun row replaces the
earlier row of its cell and names it in ``replaces``. Each row also records the unmanaged context length: the largest token
count of Codex's own history that Relay saw (``relay/conversations.json``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evals.pilot.browse_grade import contains_gold, final_answer
from evals.pilot.outcomes import browse_profile_applied, failure_cause, fallback_counts, outcome
from typing import Any

# The evaluation's working tree (data, runtimes and run outputs); see evals/pilot/README.md.
ROOT = Path(os.environ.get("RELAY_PILOT_ROOT", "relay-pilot"))
RUN = ROOT / "full-browse-429-exempt-20261030"
RERUN = ROOT / "reruns-20261106/browse/queue-state.json"
GOLD = ROOT / "full-browse-20261030/private"  # one {query_id, answer, qrels} file per question


def batch_name(state_path: str) -> str:
    path = Path(state_path)
    return path.parent.parent.name if path.parent.name == "results" else path.parent.name


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def unmanaged_tokens(run_dir: Path) -> int | None:
    conversations = read_json(run_dir / "relay/conversations.json")
    if not isinstance(conversations, dict) or not conversations:
        return None
    return max(int(entry.get("whole") or 0) for entry in conversations.values()) or None


def row(cell: dict[str, Any], batch: str, method_score: Any, evaluation_class: str) -> dict[str, Any]:
    run_path = Path(cell["raw_run"]) if cell.get("raw_run") else None
    run = read_json(run_path) if run_path else None
    run = run if isinstance(run, dict) else {}
    grade = read_json(Path(cell["score_file"])) if cell.get("score_file") else None
    grade = grade if isinstance(grade, dict) else {}
    usage = run.get("codex_final_total_usage") or {}
    aux = run.get("auxiliary_input_tokens_per_request") or []
    main = run.get("provider_input_tokens_per_request") or []
    worker = run_path.parent if run_path else None
    scored = cell.get("raw_score") is not None
    cause = None if scored or worker is None else failure_cause(worker, run)
    result = outcome(scored, cause)
    gold = read_json(GOLD / f"{cell['task_id']}.json") if scored else None
    answer = (final_answer((worker / "answer.txt").read_text(errors="replace"))
              if scored and worker and (worker / "answer.txt").is_file() else None)
    return {
        "task_id": str(cell["task_id"]), "method": cell["method"], "batch": batch,
        "outcome": result, "failure_cause": cause,
        # One policy for every benchmark: the strict score, 0 for a task failure, and no
        # score for an infrastructure failure (see outcomes.py).
        "score": cell["raw_score"] if scored else (0 if result == "task_failure" else None),
        "answer_contains_gold": contains_gold(answer, gold["answer"]) if isinstance(gold, dict) else None,
        "harness_profile_applied": browse_profile_applied(worker) if worker else None,
        **fallback_counts(run),
        "queue_evaluation_class": evaluation_class, "queue_method_score": method_score,
        "raw_score": cell.get("raw_score"), "failure_kind": cell.get("failure_kind"),
        "run_status": run.get("status"), "stop_reason": run.get("stop_reason"),
        "unmanaged_tokens": unmanaged_tokens(run_path.parent) if run_path else None,
        "harness_requests": run.get("harness_requests"),
        "peak_main_input": run.get("peak_provider_input_tokens"),
        "main_input_tokens": sum(main) if main else None,
        "auxiliary_input_tokens": sum(aux),
        "auxiliary_requests": len(aux),
        "codex_input_tokens": usage.get("input_tokens"),
        "codex_cached_tokens": usage.get("cached_input_tokens"),
        "codex_output_tokens": usage.get("output_tokens"),
        "context_changes": run.get("context_changes"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "searches": grade.get("searches"), "fetches": grade.get("fetches"),
        "explicit_final_answer": grade.get("explicit_final_answer"),
        "run_dir": str(run_path.parent) if run_path else None,
    }


def collect() -> dict[str, Any]:
    plan = json.loads((RUN / "scale-plan.json").read_text())
    retried = {(c["task_id"], c["method"]) for c in plan["retry_cells"]}
    carried = {tuple(c) for c in plan["completed_cells"]} - retried
    rows = []
    for state_path in plan["source_states"]:
        state = json.loads(Path(state_path).read_text())
        for task in state["tasks"]:
            for cell in task["cells"]:
                if cell["status"] == "finished" and (str(cell["task_id"]), cell["method"]) in carried:
                    # Earlier queues had no rate-limit class: an attempted failure scores 0.
                    rows.append(row(cell, batch_name(state_path), cell.get("end_to_end_score"), "task_result"))
    state = json.loads((RUN / "results/batch-state.json").read_text())
    counts: dict[str, int] = {}
    for task in state["tasks"]:
        for cell in task["cells"]:
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
            if cell["status"] == "finished":
                rows.append(row(cell, RUN.name, cell.get("method_score"), cell.get("evaluation_class") or "task_result"))
    keys = [(r["task_id"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("A task/method cell appears twice")
    reruns = {"finished": 0, "replaced": 0, "pending": 0}
    if RERUN.is_file():
        index = {key: n for n, key in enumerate(keys)}
        for cell in json.loads(RERUN.read_text())["cells"]:
            if cell["status"] != "finished":
                reruns["pending"] += 1
                continue
            new = row(cell, "reruns-20261106", cell.get("raw_score"), "task_result")
            key = (new["task_id"], new["method"])
            reruns["finished"] += 1
            if key in index:
                old = rows[index[key]]
                new["replaces"] = f"{old['batch']}:{old['outcome']}"
                rows[index[key]] = new
                reruns["replaced"] += 1
            else:
                index[key] = len(rows)
                rows.append(new)
    return {"run": str(RUN), "target_cells": plan["target_cells"], "window": 23560,
            "queue_status": state.get("status"), "queue_counts": counts,
            "carried_cells": len(carried), "reruns": reruns, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = collect()
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['rows'])} ended cells; queue {data['queue_status']} {data['queue_counts']}")


if __name__ == "__main__":
    main()
