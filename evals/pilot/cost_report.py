"""Count known usage from every retained attempt, including failed integrations."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from evals.pilot.collect import collect


def worker_directories(root: Path) -> Iterator[Path]:
    """Avoid walking large workspaces, installed plugins, datasets and runtimes."""
    ignored = {"codex", "work", "relay", "service", "services", "verifier", "frozen",
               "sdk_recordings", "science-tasks", "science-images", "arc-runtime", "preflight"}
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name not in ignored and not name.startswith(".")]
        if "run.json" in files:
            yield Path(directory)
            # Browse retry directories are nested inside the first attempt.
            names[:] = [name for name in names if name.startswith("attempt-")]


def collect_cost(roots: list[Path]) -> dict[str, Any]:
    rows = []
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "attempts": 0, "known_tokens": 0, "known_cached_input_tokens": 0,
        "known_output_tokens": 0, "known_elapsed_seconds": 0.0,
        "incomplete_usage_attempts": 0,
    })
    for path in sorted({path for root in roots for path in worker_directories(root)}):
        result = collect(path)
        worker = result.get("worker_usage") or {}
        summary = result.get("summary_usage") or {}
        known_tokens = worker.get("total_tokens", 0) + summary.get("total_tokens", 0)
        incomplete = (result.get("inference_started") is not False
                      and (not result.get("worker_usage") or bool(result.get("summary_usage_missing_attempts"))
                           or bool(result.get("unaccounted_upstream_attempts"))
                           or result.get("status") == "running"))
        row = {"path": str(path), "model": result.get("model"), "strategy": result.get("strategy"),
               "status": result.get("status"), "protocol_valid": result.get("protocol_valid"),
               "known_tokens": known_tokens,
               "known_worker_tokens": worker.get("total_tokens", 0),
               "known_strategy_tokens": summary.get("total_tokens", 0),
               "known_cached_input_tokens": worker.get("cached_input_tokens", 0) + summary.get("cached_input_tokens", 0),
               "known_output_tokens": worker.get("output_tokens", 0) + summary.get("output_tokens", 0),
               "known_elapsed_seconds": result.get("elapsed_seconds", 0),
               "usage_incomplete": incomplete,
               "unaccounted_upstream_attempts": result.get("unaccounted_upstream_attempts", 0),
               "saved_credit_checks_unchanged": result.get("all_saved_credit_checks_unchanged")}
        rows.append(row)
        for key in ("all", f"{row['model']}/{row['strategy']}"):
            group = groups[key]
            group["attempts"] += 1
            for field in ("known_tokens", "known_cached_input_tokens", "known_output_tokens", "known_elapsed_seconds"):
                group[field] += row[field]
            group["incomplete_usage_attempts"] += incomplete
    return {"roots": [str(root) for root in roots], "groups": dict(groups), "attempts": rows,
            "note": "Known lower bound includes archived/retried attempts and completed strategy-owned calls. Cached input is a subset. Unfinished or missing provider usage is unknown. Elapsed time is summed worker wall time, not parallel elapsed time. This is not API billing."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect_cost([root.resolve() for root in args.roots])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["groups"].get("all", {})))


if __name__ == "__main__":
    main()
