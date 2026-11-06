"""Report shared sampled tasks without averaging missing or invalid scores as zero."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from evals.pilot.methods import EVALUATED_METHODS

from evals.pilot.matrix_report import (
    BENCHMARKS, MODELS, atomic_write, collect_slot, read_json,
)


def aggregate(rows: list[dict[str, Any]], selected: list[str]) -> dict[str, Any]:
    """Compute per-method scores and model differences on exactly matched tasks."""
    groups: list[dict[str, Any]] = []
    common = set(selected)
    for method in EVALUATED_METHODS:
        model_groups: dict[str, Any] = {}
        task_scores: dict[str, dict[str, float]] = {}
        for model in MODELS:
            subset = [row for row in rows if row["model"] == model and row["strategy"] == method]
            scores = {row["task_id"]: row["score"] for row in subset
                      if row["status"] == "scored" and row["score"] is not None}
            task_scores[model] = scores
            common.intersection_update(scores)
            known = [(row.get("usage", {}).get("combined_usage") or {}).get("total_tokens")
                     for row in subset]
            incomplete = sum(
                bool(row["status"] != "not_started" and row.get("usage", {}).get("inference_started") is not False
                and (not row.get("usage", {}).get("worker_usage")
                     or row.get("usage", {}).get("summary_usage_missing_attempts", 0)
                     or row.get("usage", {}).get("unaccounted_upstream_attempts", 0)
                     or row["status"] == "running"))
                for row in subset
            )
            model_groups[model] = {
                "n": len(scores), "planned": len(selected),
                "mean_score": mean(scores.values()) if scores else None,
                "task_scores": scores, "status_counts": dict(Counter(row["status"] for row in subset)),
                "known_current_attempt_tokens": sum(n for n in known if n is not None),
                "attempts_with_known_tokens": sum(n is not None for n in known),
                "incomplete_usage_attempts": incomplete,
            }
        paired = sorted(set(task_scores[MODELS[0]]) & set(task_scores[MODELS[1]]))
        differences = [task_scores[MODELS[1]][task] - task_scores[MODELS[0]][task] for task in paired]
        groups.append({"strategy": method, "models": model_groups, "paired_tasks": paired,
                       "paired_n": len(paired),
                       "paired_sol_minus_luna": mean(differences) if differences else None})
    return {"groups": groups, "all_methods_common_tasks": sorted(common)}


def collect_sample(root: Path) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    notes = read_json(root / "report-notes.json").get("benchmarks", {})
    for benchmark in BENCHMARKS:
        manifest = read_json(root / "manifests" / f"{benchmark}.json")
        selected = [str(value) for value in manifest.get("selected_ids", [])]
        rows: list[dict[str, Any]] = []
        for task_id in selected:
            for method in EVALUATED_METHODS:
                for model in MODELS:
                    slot = root / benchmark / task_id / model / method
                    row = collect_slot(benchmark, model, method, slot)
                    row["task_id"] = task_id
                    rows.append(row)
        benchmarks[benchmark] = {"manifest": manifest, "rows": rows,
                                 "protocol_notes": notes.get(benchmark, []),
                                 **aggregate(rows, selected)}
        all_rows.extend(rows)
    cost_path = root / "cost-current.json"
    cost = read_json(cost_path)
    cost_snapshot = {
        "all_retained_attempts": cost.get("groups", {}).get("all", {}),
        "observed_at": (datetime.fromtimestamp(cost_path.stat().st_mtime,
                                              ZoneInfo("America/New_York")).isoformat()
                        if cost_path.is_file() else None),
        "roots": cost.get("roots", []),
    }
    return {"generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "root": str(root), "planned_runs": len(all_rows),
            "status_counts": dict(Counter(row["status"] for row in all_rows)),
            "benchmarks": benchmarks,
            "cost_snapshot": cost_snapshot,
            "quota_snapshot": read_json(root / "quota-current.json"),
            "admission_closed": (root / ".stop-new-workers").is_file(),
            "admission": read_json(root / ".stop-new-workers"),
            "limitations": [
                "Five tasks are a preliminary sample, not a stable benchmark ranking.",
                "Model differences use only matched tasks with valid scores for both models.",
                "The deliberately selected integration pilot is excluded from this random sample.",
                "Browse scores use exact-answer diagnostics, not the official LLM judge.",
                "Known current-attempt tokens omit earlier retries and unfinished provider usage.",
                "ARC uses whole public games under the declared reduced interaction budget.",
            ]}


def score_text(value: float | None, count: int, benchmark: str) -> str:
    if value is None:
        return "—"
    score = f"{value:.2f}" if benchmark == "arc" else f"{100 * value:.1f}%"
    return f"{score} (n={count})"



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--max-hours", type=float, default=168)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 168:
        parser.error("--max-hours must be in (0,168]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_hours * 3600
    while True:
        report = collect_sample(args.root.resolve())
        atomic_write(args.out.with_suffix(".json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"generated_at": report["generated_at"], "status_counts": report["status_counts"]}), flush=True)
        if not args.watch or time.monotonic() >= deadline:
            break
        time.sleep(min(60, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
