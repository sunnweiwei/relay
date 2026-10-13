"""Score over time and over cost for single-agent ProgramBench runs (marl's run_team.py), one curve per run.

    python evals/programbench_curves.py RUN [RUN ...] --out DIR [--label RUN=NAME ...]

A run is a marl run directory, graded (`grader/grade_run.py RUN --log-grid`): grading/scores.csv gives the
hidden-test score of `main` at each grid time. Cost is API list price (GPT-6.1 Sol: $2 per million fresh input
tokens, $0.10 cached, $10 output) of every model request up to that time: the agents' own (their event logs,
`thread/tokenUsage/updated`), and, for a run through Relay, Relay's summary requests (`summary_usage` in its event
log). Times are the run's experiment clock (clock.json: [[wall start, elapsed start], ...] per segment). Writes
score_time.png, score_cost.png and curves.md to DIR.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

PRICE = {"input": 2.0, "cached": 0.10, "output": 10.0}  # $ per million tokens


def cost(fresh: int, cached: int, output: int) -> float:
    return (fresh * PRICE["input"] + cached * PRICE["cached"] + output * PRICE["output"]) / 1e6


def elapsed(clock: list[list[float]], t: float) -> float:
    """Experiment time of wall time `t`: the latest segment started at or before it."""

    starts = [w for w, _ in clock]
    w, e = clock[max(0, bisect.bisect_right(starts, t) - 1)]
    return e + t - w


def spending(run: Path) -> list[tuple[float, float]]:
    """(experiment seconds, $) of every model request, in time order."""

    clock = json.loads((run / "clock.json").read_text())
    out = []
    for log in (run / "logs").glob("*.events.jsonl"):
        for line in log.read_text().splitlines():
            event = json.loads(line)
            if event.get("kind") == "thread/tokenUsage/updated":
                last = event["data"]["tokenUsage"]["last"]
                cached = last.get("cachedInputTokens", 0)
                out.append((elapsed(clock, event["t"]), cost(last["inputTokens"] - cached, cached, last["outputTokens"])))
    if (events := run / "relay" / "events.jsonl").exists():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            for usage in event.get("summary_usage") or []:
                cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                out.append((elapsed(clock, event["time"]),
                            cost(usage.get("input_tokens", 0) - cached, cached, usage.get("output_tokens", 0))))
    return sorted(out)


def curve(run: Path) -> list[dict]:
    spent = spending(run)
    times, total, rows = [t for t, _ in spent], 0.0, []
    cumulative = []
    for _, c in spent:
        total += c
        cumulative.append(total)
    for row in csv.DictReader((run / "grading" / "scores.csv").open()):
        if row["score"] == "":
            continue
        t = float(row["grid_t"])
        n = bisect.bisect_right(times, t)
        rows.append({"minutes": t / 60, "score": 100 * float(row["score"]), "cost": cumulative[n - 1] if n else 0.0})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", action="append", default=[], help="RUN=NAME for a run's curve")
    options = parser.parse_args()
    labels = dict(x.split("=", 1) for x in options.label)
    curves = {labels.get(run.name, run.name): curve(run) for run in options.runs}
    options.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for x, name, title in (("minutes", "score_time.png", "experiment time (min)"), ("cost", "score_cost.png", "cost ($, API list price)")):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for label, rows in curves.items():
            axis.plot([r[x] for r in rows], [r["score"] for r in rows], marker="o", markersize=3, label=label)
        axis.set_xlabel(title)
        axis.set_ylabel("hidden tests passed (%)")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(options.out / name, dpi=150)
    grid = sorted({round(r["minutes"]) for rows in curves.values() for r in rows})
    lines = ["| minutes | " + " | ".join(curves) + " |", "|---|" + "---|" * len(curves)]
    for m in grid:
        cells = []
        for rows in curves.values():
            r = next((r for r in rows if round(r["minutes"]) == m), None)
            cells.append(f"{r['score']:.1f}% (${r['cost']:.2f})" if r else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (options.out / "curves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
