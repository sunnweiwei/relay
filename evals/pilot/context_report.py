"""Luna method scores by context length on Browse, ARC and Science, as one HTML page plus JSON.

Inputs are the tables written by browse_context_collect (Browse) and
context_collect (ARC, Science). Context length is the unmanaged length: the
token count of Codex's own full history at the end of a run, before any
method rewrites it. A task's context demand is the median unmanaged length
over the five core methods on that task. Two methods were added on 2026-11-06 where
they have runs: a truncation-only baseline and ACM with a forced compression. RLM is
shown for reference only: its main agent often does not receive the task in these runs.

Every benchmark uses the outcome policy of outcomes.py: a task failure scores 0 and an
infrastructure failure has no score. Method comparisons use the tasks on which every
working method has a score under that policy.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKING = ["compaction", "folding", "clm", "acm", "selfcompact"]  # the core methods; they define context demand
# Run on 2026-11-06 on part of Browse and Science (an account limit stopped the queues). A
# same-day rerun of released ACM ("acm_control", 24 questions) is too small to report.
ADDED = ["acm_forced", "truncation"]
ALL_METHODS = WORKING + ADDED + ["rlm"]
LABEL = {"compaction": "Compaction", "folding": "Folding", "clm": "CLM", "acm": "ACM",
         "selfcompact": "Self-Compact", "rlm": "RLM", "acm_forced": "ACM (forced)",
         "truncation": "Truncation only", "acm_control": "ACM (rerun 11-06)"}
# Reference categorical palette, slots 1-6 in fixed order (validated: scripts/validate_palette.js). The
# added methods reuse a related hue: ACM (forced) a darker ACM amber, the truncation baseline neutral grey.
COLOR = {"compaction": "#2a78d6", "folding": "#eb6834", "clm": "#1baf7a", "acm": "#eda100",
         "selfcompact": "#e87ba4", "rlm": "#008300", "acm_forced": "#9a6400", "truncation": "#7d7c76",
         "acm_control": "#eda100"}
DASH = {"rlm": "5 4", "acm_forced": "2 3", "acm_control": "8 3"}

BENCH = {
    "browse": {"name": "BrowseComp-Plus", "metric": "score", "binary": True, "window": 23560,
               "edges": [2, 3, 4, 6], "metric_name": "Accuracy",
               "metric_note": "Strict exact match of the normalized final answer against the gold answer, "
                              "not the official LLM judge, so it is a lower bound on judged accuracy.",
               "budget": "23,560-token input window (+2,356 dispatch tolerance), 100 main requests, 30 minutes"},
    "arc": {"name": "ARC-AGI-3", "metric": "partial", "binary": False, "window": 32000, "edges": None,
            "metric_name": "Levels completed",
            "metric_note": "Levels completed divided by the game's level count. No run completed a whole game.",
            "budget": "32,000-token input window (+3,200 dispatch tolerance), 200 main requests, 3,600 s, "
                      "300 game actions"},
    "science": {"name": "Terminal-Bench-Science", "metric": "partial", "binary": False, "window": 32000, "edges": None,
                "metric_name": "Verifier tests passed",
                "metric_note": "Share of the task's verifier tests that pass, a diagnostic partial score; the "
                               "official reward is 1 only when all pass. The share has a floor: RLM runs that "
                               "barely worked on the task still pass about 16% on average.",
                "budget": "32,000-token input window (+3,200 dispatch tolerance), 128 main requests, 7,200 s"},
}


# ---------- statistics ----------

def wilson(k: float, n: int) -> tuple[float, float]:
    if not n:
        return (math.nan, math.nan)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def boot_mean(values: list[float], draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return (means[int(0.025 * draws)], means[int(0.975 * draws) - 1])


def summary(rows: list[dict[str, Any]], key: str, binary: bool) -> dict[str, Any]:
    values = [r[key] for r in rows if r.get(key) is not None]
    n = len(values)
    mean = sum(values) / n if n else math.nan
    lo, hi = wilson(sum(values), n) if binary else boot_mean(values)
    return {"k": sum(values), "n": n, "mean": mean, "lo": lo, "hi": hi}


def paired_diff(a: list[float], b: list[float], seed: int = 0) -> dict[str, float]:
    """Mean of a - b with a 95% bootstrap interval over tasks; a and b are aligned by task."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return {"diff": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    lo, hi = boot_mean(diffs, seed=seed)
    return {"diff": sum(diffs) / len(diffs), "lo": lo, "hi": hi, "n": len(diffs)}


def share(rows: list[dict[str, Any]], stop: str) -> float:
    return sum(r["stop_reason"] == stop for r in rows) / len(rows) if rows else math.nan


def input_tokens(r: dict[str, Any]) -> int:
    return (r["main_input_tokens"] or 0) + (r["auxiliary_input_tokens"] or 0)


def analyse(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    key, binary, window = spec["metric"], spec["binary"], spec["window"]
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["method"]] = r
    present = {r["method"] for r in rows}
    methods = [m for m in ALL_METHODS if m in present]
    compared = [m for m in WORKING if m in present]  # the core methods, run on every task

    overall = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        scored = [r for r in rs if r["outcome"] == "scored"]
        counted = [r for r in rs if r["outcome"] != "infrastructure"]
        overall[m] = summary(rs, key, binary) | {
            "ended": len(rs), "scored": len(scored),
            "task_failures": sum(r["outcome"] == "task_failure" for r in rs),
            "infrastructure": sum(r["outcome"] == "infrastructure" for r in rs),
            "valid_only_mean": summary(scored, key, binary)["mean"],
            "success": sum(r.get("success") or 0 for r in counted),
            "nonzero": sum(1 for r in counted if (r.get(key) or 0) > 0),
            "official_mean": (sum(r.get("method_score") or 0 for r in counted) / len(counted)
                              if counted else math.nan),
            "truncated_runs": sum((r.get("main_truncations") or 0) > 0 for r in rs) / len(rs) if rs else math.nan,
            "main_truncations": sum(r.get("main_truncations") or 0 for r in rs),
            **({"profile_not_applied": sum(r.get("harness_profile_applied") is False for r in rs),
                "contains_gold": summary([r for r in scored], "answer_contains_gold", True)["mean"]}
               if "harness_profile_applied" in (rs[0] if rs else {}) else {}),
            "request_cap": share(rs, "model_request_limit"), "time_cap": share(rs, "wall_time_limit"),
            "median_requests": statistics.median(r["harness_requests"] or 0 for r in rs),
            "median_minutes": statistics.median(r["elapsed_seconds"] or 0 for r in rs) / 60,
            "median_unmanaged": statistics.median([r["unmanaged_tokens"] for r in rs if r["unmanaged_tokens"]] or [math.nan]),
            "mean_input_tokens": sum(input_tokens(r) for r in rs) / len(rs),
        }

    # Comparison population: tasks where every core method has a score.
    tasks = sorted(t for t, cells in by_task.items()
                   if all(m in cells and cells[m].get(key) is not None for m in compared))

    def demand(t: str, methods: list[str]) -> float:
        return statistics.median(by_task[t][m]["unmanaged_tokens"] for m in methods if by_task[t][m]["unmanaged_tokens"])

    if spec["edges"]:
        edges = [e * window for e in spec["edges"]]
    else:  # Few tasks: three bins of about equal size (tertiles of demand).
        ds = sorted(demand(t, WORKING) for t in tasks)
        edges = [ds[len(ds) // 3 - 1], ds[2 * len(ds) // 3 - 1]]
    kt = [f"{round(e / 1000)}K" for e in edges]
    labels = [f"≤{kt[0]}"] + [f"{a}–{b}" for a, b in zip(kt, kt[1:])] + [f">{kt[-1]}"]
    n_bins = len(labels)
    bin_of = lambda tokens: sum(tokens > e for e in edges)
    task_bin = {t: bin_of(demand(t, WORKING)) for t in tasks}
    in_bin = lambda b: [t for t in tasks if task_bin[t] == b]

    def cells(m: str, ts: list[str]) -> list[dict[str, Any]]:
        return [by_task[t][m] for t in ts if m in by_task[t]]

    by_bin = {m: [summary(cells(m, in_bin(b)), key, binary) for b in range(n_bins)] for m in methods}
    cap_by_bin = {m: [share(cells(m, in_bin(b)), "model_request_limit") for b in range(n_bins)] for m in methods}
    tokens_by_bin = {m: [statistics.mean([input_tokens(r) for r in cells(m, in_bin(b))] or [math.nan])
                         for b in range(n_bins)] for m in methods}

    def contrast(a: str, b: str) -> dict[Any, dict[str, float]]:
        """a minus b on the comparison tasks where both have a score (the added methods ran on part)."""
        both = lambda t: all(m in by_task[t] and by_task[t][m].get(key) is not None for m in (a, b))
        ts_all = [t for t in tasks if both(t)]
        out: dict[Any, dict[str, float]] = {"all": paired_diff([by_task[t][a][key] for t in ts_all],
                                                               [by_task[t][b][key] for t in ts_all])}
        for k in range(n_bins):
            ts = [t for t in in_bin(k) if both(t)]
            out[k] = paired_diff([by_task[t][a][key] for t in ts], [by_task[t][b][key] for t in ts], seed=k + 1)
        return out

    # Paired differences: the best compared method minus each other compared method, on the same tasks.
    ref = max(compared, key=lambda m: overall[m]["mean"])
    paired = {m: contrast(ref, m) for m in compared if m != ref}
    # Fixed contrasts, chosen before the runs: each method against the truncation baseline, and the
    # forced compression against released ACM.
    pairs = ([(m, "truncation") for m in [*compared, "acm_forced"] if m in present]
             if "truncation" in present else [])
    pairs += [("acm_forced", "acm")] if {"acm_forced", "acm"} <= present else []
    baseline = {f"{a_}|{b_}": contrast(a_, b_) for a_, b_ in pairs}

    # Robustness: bin each method's cell by the other working methods' median only.
    loo = {}
    for m in WORKING:
        others = [o for o in WORKING if o != m]
        groups = defaultdict(list)
        for t in tasks:
            groups[bin_of(demand(t, others))].append(by_task[t][m])
        loo[m] = [summary(groups[b], key, binary) for b in range(n_bins)]

    return {
        "window": window, "metric": key, "overall": overall, "tasks": len(tasks), "ref": ref,
        "methods": methods, "compared": compared, "baseline": baseline,
        "bins": {"labels": labels, "edges": edges, "counts": [len(in_bin(b)) for b in range(n_bins)]},
        "by_bin": by_bin, "cap_by_bin": cap_by_bin, "tokens_by_bin": tokens_by_bin, "paired": paired, "loo": loo,
        "solved_by_none": sum(1 for t in tasks if all(not by_task[t][m][key] for m in compared)),
    }


# ---------- HTML ----------

def pct(x: float, digits: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def esc(text: Any) -> str:
    return html.escape(str(text))


def nice_max(values: list[float]) -> float:
    top = max([v for v in values if not math.isnan(v)] or [0.01])
    for step in (0.01, 0.02, 0.05, 0.1, 0.2):
        if top * 1.15 <= step * 5:
            return step * 5
    return 1.0


def line_chart(series: dict[str, list[Any]], labels: list[str], *, ytitle: str, note: list[str], ci: bool) -> str:
    """Ordinal x, one line per method, 2px lines, 8px markers, recessive grid, hover via data-tip."""
    value = lambda v: v["mean"] if isinstance(v, dict) else v
    ymax = nice_max([value(v) for vs in series.values() for v in vs])
    width, height, left, right, top, bottom = 720, 320, 60, 20, 14, 60
    pw, ph = width - left - right, height - top - bottom
    step = pw / len(labels)
    x = lambda i: left + step * (i + 0.5)
    y = lambda v: top + ph * (1 - v / ymax)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(ytitle)}">']
    for i in range(6):
        v = ymax * i / 5
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"axis" if i == 0 else "grid"}"/>')
        out.append(f'<text x="{left - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{esc(pct(v, 0 if ymax >= 0.25 else 1))}</text>')
    for i, label in enumerate(labels):
        out.append(f'<text x="{x(i):.1f}" y="{top + ph + 20}" class="tick" text-anchor="middle">{esc(label)}</text>')
        out.append(f'<text x="{x(i):.1f}" y="{top + ph + 38}" class="tick muted" text-anchor="middle">{esc(note[i])}</text>')
    out.append(f'<text x="14" y="{top + ph / 2:.1f}" class="tick" transform="rotate(-90 14 {top + ph / 2:.1f})" text-anchor="middle">{esc(ytitle)}</text>')
    for m, values in series.items():
        pts = [(i, value(v)) for i, v in enumerate(values)]
        pts = [(i, v) for i, v in pts if not math.isnan(v)]
        path = " ".join(f'{"M" if j == 0 else "L"}{x(i):.1f},{y(v):.1f}' for j, (i, v) in enumerate(pts))
        dash = f' stroke-dasharray="{DASH[m]}"' if m in DASH else ""
        out.append(f'<path d="{path}" fill="none" stroke="{COLOR[m]}" stroke-width="2" stroke-linejoin="round"{dash}/>')
        for i, v in pts:
            cell = values[i]
            tip = f"{LABEL[m]} · {labels[i]}: {pct(v)}"
            if isinstance(cell, dict):
                tip += f" (n={cell['n']}" + (f"; 95% CI {pct(cell['lo'])}–{pct(cell['hi'])})" if ci else ")")
            out.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="{COLOR[m]}" stroke="#fcfcfb" stroke-width="2" data-tip="{esc(tip)}"/>')
            out.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="11" fill="transparent" data-tip="{esc(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def dot_chart(overall: dict[str, Any], metric_name: str) -> str:
    """Overall score per method with 95% intervals."""
    order = sorted(overall, key=lambda m: -overall[m]["mean"])
    xmax = nice_max([overall[m]["hi"] for m in order])
    width, row_h, left, right, top = 720, 30, 110, 150, 8
    height = top + row_h * len(order) + 30
    pw = width - left - right
    x = lambda v: left + pw * v / xmax
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(metric_name)}">']
    for i in range(6):
        v = xmax * i / 5
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{top + row_h * len(order)}" class="{"axis" if i == 0 else "grid"}"/>')
        out.append(f'<text x="{x(v):.1f}" y="{top + row_h * len(order) + 18}" class="tick" text-anchor="middle">{esc(pct(v, 0 if xmax >= 0.25 else 1))}</text>')
    for j, m in enumerate(order):
        o, cy = overall[m], top + row_h * (j + 0.5)
        tip = f"{LABEL[m]}: {pct(o['mean'])} (n={o['n']}; 95% CI {pct(o['lo'])}–{pct(o['hi'])})"
        out.append(f'<text x="{left - 12}" y="{cy + 4:.1f}" class="lab" text-anchor="end">{esc(LABEL[m])}</text>')
        out.append(f'<line x1="{x(o["lo"]):.1f}" x2="{x(o["hi"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{COLOR[m]}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(o["mean"]):.1f}" cy="{cy:.1f}" r="5" fill="{COLOR[m]}" stroke="#fcfcfb" stroke-width="2" data-tip="{esc(tip)}"/>')
        out.append(f'<rect x="{left}" y="{cy - row_h / 2:.1f}" width="{pw}" height="{row_h}" fill="transparent" data-tip="{esc(tip)}"/>')
        out.append(f'<text x="{x(o["hi"]) + 10:.1f}" y="{cy + 4:.1f}" class="lab2">{esc(pct(o["mean"]))}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(methods: list[str]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{COLOR[m]}"></i>{esc(LABEL[m])}</span>' for m in methods) + "</div>"


def bin_table(source: dict[str, list[dict[str, Any]]], labels: list[str], methods: list[str]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in labels)
    body = ""
    for m in methods:
        tds = "".join(f"<td>{pct(c['mean'])}<br><span class=muted>n={c['n']} · {pct(c['lo'], 0)}–{pct(c['hi'], 0)}</span></td>"
                      for c in source[m])
        body += f"<tr><td>{esc(LABEL[m])}</td>{tds}</tr>"
    return f'<div class="scroll"><table class=bins><thead><tr><th>Method</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section(bench: str, a: dict[str, Any], meta: dict[str, Any], number: int) -> str:
    spec, o = BENCH[bench], a["overall"]
    labels, counts = a["bins"]["labels"], a["bins"]["counts"]
    order = sorted(a["methods"], key=lambda m: -o[m]["mean"])
    extra_head = {"browse": "<th>Answer contains gold (lenient)</th><th>Profile not applied</th>",
                  "arc": "<th>Official score (0–100)</th><th>Runs with score &gt; 0</th>",
                  "science": "<th>Official reward = 1</th>"}[bench]

    def extra(m: str) -> str:
        if bench == "arc":
            return f"<td>{o[m]['official_mean'] * 100:.2f}</td><td>{o[m]['nonzero']}</td>"
        if bench == "science":
            return f"<td>{o[m]['success']}/{o[m]['n']}</td>"
        return f"<td>{pct(o[m]['contains_gold'])}</td><td>{o[m]['profile_not_applied']}</td>"

    overall_rows = "".join(
        f"<tr><td>{esc(LABEL[m])}</td><td>{o[m]['ended']}</td><td>{o[m]['task_failures']}</td><td>{o[m]['infrastructure']}</td>"
        f"<td><b>{pct(o[m]['mean'])}</b></td><td>{pct(o[m]['lo'])}–{pct(o[m]['hi'])}</td><td>{pct(o[m]['valid_only_mean'])}</td>"
        f"{extra(m)}<td>{pct(o[m]['truncated_runs'], 0)}</td><td>{pct(o[m]['request_cap'], 0)}</td><td>{pct(o[m]['time_cap'], 0)}</td>"
        f"<td>{o[m]['median_requests']:.0f}</td><td>{o[m]['median_minutes']:.1f}</td>"
        f"<td>{o[m]['median_unmanaged'] / 1000:.0f}K</td><td>{o[m]['mean_input_tokens'] / 1000:.0f}K</td></tr>"
        for m in order)
    paired_rows = "".join(
        f"<tr><td>{esc(LABEL[a['ref']])} − {esc(LABEL[m])}</td>" + "".join(
            f"<td>{100 * p['diff']:+.1f}%<br><span class=muted>{100 * p['lo']:+.1f} to {100 * p['hi']:+.1f}</span></td>"
            for p in [a["paired"][m]["all"]] + [a["paired"][m][b] for b in range(len(labels))]) + "</tr>"
        for m in a["paired"])
    contrast_rows = "".join(
        f"<tr><td>{esc(LABEL[k.split('|')[0]])} − {esc(LABEL[k.split('|')[1]])}<br><span class=muted>{c['all']['n']} tasks</span></td>" + "".join(
            f"<td>{100 * p['diff']:+.1f}%<br><span class=muted>{100 * p['lo']:+.1f} to {100 * p['hi']:+.1f}</span></td>"
            for p in [c["all"]] + [c[b] for b in range(len(labels))]) + "</tr>"
        for k, c in a["baseline"].items())
    contrast_table = (f"<p>Fixed contrasts, each on the comparison tasks where both methods have a score: each method minus "
                      f"the truncation-only baseline, and ACM (forced) minus released ACM (95% bootstrap interval). The added "
                      f"methods ran a week after the others and on part of the tasks; differences can include drift between "
                      f"the two dates.</p>"
                      f"<div class=\"scroll\"><table><thead><tr><th>Difference</th><th>All tasks</th>"
                      f"{''.join(f'<th>{esc(l)}</th>' for l in labels)}</tr></thead><tbody>{contrast_rows}</tbody></table></div>"
                      if contrast_rows else "")
    cap_rows = "".join(
        f"<tr><td>{esc(LABEL[m])}</td>" + "".join(f"<td>{pct(v, 0)}</td>" for v in a["cap_by_bin"][m])
        + "".join(f"<td>{v / 1000:.0f}K</td>" for v in a["tokens_by_bin"][m]) + "</tr>" for m in a["methods"])
    note = [f"{c} tasks" for c in counts]
    bin_rule = (f"multiples of the window W = {a['window']:,} ({', '.join(str(e) for e in spec['edges'])} W)"
                if spec["edges"] else "tertiles of demand, so the groups have about equal numbers of tasks")
    best = lambda src, b: LABEL[max(WORKING, key=lambda m: src[m][b]["mean"])]
    added = [m for m in ADDED if m in a["methods"]]
    added_note = (f" The methods added on 2026-11-06 ({', '.join(LABEL[m] for m in added)}) ran on part of these tasks; "
                  f"their figures and contrasts use the tasks they have." if added else "")
    return f"""
<h2 id="{bench}">{number}. {esc(spec['name'])}</h2>
<p class="muted">Budget: {esc(spec['budget'])}. Metric: {esc(spec['metric_name'])}. {esc(spec['metric_note'])}</p>
<h3>{number}.1 Overall</h3>
<figure>{dot_chart(o, spec['metric_name'])}
<figcaption>Figure {number}a. {esc(spec['metric_name'])} per method over its ended runs (task failures count as 0; infrastructure failures are excluded), with 95% intervals ({'Wilson' if spec['binary'] else 'bootstrap over runs'}). RLM is for reference only.</figcaption></figure>
<div class="scroll"><table><thead><tr><th>Method</th><th>Ended</th><th>Task failures (0)</th><th>Infrastructure (excluded)</th><th>{esc(spec['metric_name'])}</th><th>95% CI</th><th>Scored runs only</th>{extra_head}<th>Runs with truncation fallback</th><th>Hit request cap</th><th>Hit time cap</th><th>Median requests</th><th>Median minutes</th><th>Median context length</th><th>Mean input tokens / run</th></tr></thead><tbody>{overall_rows}</tbody></table></div>

<h3>{number}.2 By task context length</h3>
<p>Comparisons use the {a['tasks']} tasks on which every core method has a score; on {a['solved_by_none']} of them every core method scores 0.{added_note} Groups: {bin_rule}.</p>
{legend(a['methods'])}
<figure>{line_chart(a['by_bin'], labels, ytitle=spec['metric_name'], note=note, ci=True)}
<figcaption>Figure {number}b. {esc(spec['metric_name'])} by the task's context demand. Hover for sample sizes and 95% intervals. RLM (dashed) is for reference only.</figcaption></figure>
{bin_table(a['by_bin'], labels, a['methods'])}
<p>Paired differences on the same tasks: the best overall method, {esc(LABEL[a['ref']])}, minus each other compared method (95% bootstrap interval, 2,000 resamples of tasks). The reference method is chosen after seeing the data.</p>
<div class="scroll"><table><thead><tr><th>Difference</th><th>All tasks</th>{''.join(f'<th>{esc(l)}</th>' for l in labels)}</tr></thead><tbody>{paired_rows}</tbody></table></div>
{contrast_table}
<p><b>Leave-one-out check.</b> The demand above includes the method's own run. Grouping each core method's runs by the other four core methods' median instead, the best core method per group is {', '.join(best(a['loo'], b) for b in range(len(labels)))} (above: {', '.join(best(a['by_bin'], b) for b in range(len(labels)))}).</p>

<h3>{number}.3 Caps and cost</h3>
<div class="scroll"><table><thead><tr><th rowspan=2>Method</th><th colspan={len(labels)}>Share of runs that hit the request cap</th><th colspan={len(labels)}>Mean input tokens / run</th></tr>
<tr>{''.join(f'<th>{esc(l)}</th>' for l in labels) * 2}</tr></thead><tbody>{cap_rows}</tbody></table></div>
"""


def render(results: dict[str, Any], metas: dict[str, Any], generated: str) -> str:
    sections = "".join(section(b, results[b], metas[b], i + 2) for i, b in enumerate(["browse", "arc", "science"]))
    status = []
    for bench, label in (("browse", "BrowseComp-Plus"), ("arc", "ARC-AGI-3"), ("science", "Terminal-Bench-Science")):
        rows = results[bench]["overall"]
        ended = sum(rows[m]["ended"] for m in rows)
        infra = sum(rows[m]["infrastructure"] for m in rows)
        failures = sum(rows[m]["task_failures"] for m in rows)
        status.append(f"{label}: {ended:,} ended runs of {len(rows)} methods; {failures} task failures scored 0; "
                      f"{infra} infrastructure failures excluded.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Context Length by Method</title>
<style>
:root {{ --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --rule:rgba(11,11,11,0.10); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:24px; font-weight:600; margin:0 0 4px; }}
h2 {{ font-size:19px; font-weight:600; margin:44px 0 8px; padding-top:16px; border-top:1px solid var(--rule); }}
h3 {{ font-size:16px; font-weight:600; margin:26px 0 6px; }}
p, li {{ max-width:80ch; }}
.sub, .muted {{ color:var(--muted); }} .sub {{ margin:0 0 20px; font-size:13px; }}
figure {{ margin:14px 0 8px; background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:12px; overflow-x:auto; }}
figcaption {{ color:var(--ink2); font-size:13px; margin-top:6px; }}
svg {{ width:100%; min-width:600px; height:auto; display:block; }}
svg .grid {{ stroke:var(--grid); stroke-width:1; }} svg .axis {{ stroke:var(--axis); stroke-width:1; }}
svg .tick {{ fill:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
svg .lab {{ fill:var(--ink); font-size:13px; }} svg .lab2 {{ fill:var(--ink2); font-size:12px; }}
svg .muted {{ fill:var(--muted); }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; color:var(--ink2); margin:4px 0 6px; }}
.legend i {{ display:inline-block; width:14px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:13px; margin:10px 0; font-variant-numeric:tabular-nums; }}
th, td {{ padding:6px 8px; border-bottom:1px solid var(--grid); text-align:right; vertical-align:top; }}
table.bins td {{ white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ font-weight:600; color:var(--ink2); border-bottom:1px solid var(--axis); }}
.callout {{ border-left:3px solid var(--axis); background:var(--surface); padding:10px 14px; margin:12px 0; }}
.tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--surface); font-size:12px; padding:6px 8px; border-radius:4px; max-width:340px; display:none; z-index:10; }}
</style></head><body><main>
<h1>Context-management methods by task context length</h1>
<p class="sub">gpt-6-luna, medium reasoning effort, full-population runs on three benchmarks. Generated {esc(generated)}. Only runs under one budget condition per benchmark are included; see evals/pilot/README.md for the protocol and its limitations.</p>

<h2>1. Runs</h2>
<ul>{''.join(f'<li>{esc(line)}</li>' for line in status)}</ul>
<div class="callout"><b>Context length.</b> A run's unmanaged context length is the token count of Codex's own full history at the end of the run, before any method rewrites it: how long the context would be without context management. A task's context demand is the median unmanaged length of the five core methods (Compaction, Folding, CLM, ACM, Self-Compact) on that task. Longer tasks also tend to be harder, so a falling curve mixes longer context with harder tasks; comparisons between methods on the same tasks are less affected.</div>
<div class="callout"><b>Truncation fallback.</b> All runs used Relay's common overflow fallback: when a method's own forced recovery still left a request over the window, Relay dropped the oldest tool call groups. Released ACM has no forced recovery and the model rarely asked it to compress, so ACM needed the fallback in most runs, the other methods almost never (column "Runs with truncation fallback"). Two methods added on 2026-11-06 separate the two parts on Browse and Science: <i>Truncation only</i> leaves the context unchanged, so the fallback alone manages it, and <i>ACM (forced)</i> compresses, as manage_context does, the oldest history since its last summary when a request is over the window. They ran on 2026-11-06, a week after the other methods, until the account's weekly usage limit stopped them: on 404 and 403 of the 825 Browse questions (the first in the fixed random order) and on 23 Science tasks. A same-day rerun of released ACM, meant to measure drift between the two dates, stopped after 24 questions and is not reported.</div>
<div class="callout"><b>RLM is not a measurement of the RLM method.</b> Relay replaces the conversation with the RLM's handoff, and in many runs the main agent never sees the task (its final answer asks for the question). RLM is drawn for reference and is left out of the paired comparisons.</div>
{sections}
</main>
<div class="tip" id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.addEventListener('mousemove', e => {{
  const t = e.target.closest('[data-tip]');
  if (!t) {{ tip.style.display = 'none'; return; }}
  tip.textContent = t.getAttribute('data-tip');
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 350) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
}});
</script>
</body></html>
"""


# The public per-run table: identifiers, outcomes and measurements only (no paths, answers or text).
CELL_COLUMNS = ["benchmark", "task_id", "method", "outcome", "failure_cause", "score", "partial", "success",
                "stop_reason", "harness_requests", "elapsed_seconds", "unmanaged_tokens", "peak_main_input",
                "main_input_tokens", "auxiliary_input_tokens", "auxiliary_requests", "context_changes",
                "main_truncations", "auxiliary_truncations", "recovery_exhausted",
                "harness_profile_applied", "answer_contains_gold",
                "levels_completed", "level_count", "game_actions", "tests_passed", "tests_total", "source"]


def write_cells(path: Path, tables: dict[str, list[dict[str, Any]]]) -> int:
    rows = 0
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, CELL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for bench, cells in tables.items():
            for row in sorted(cells, key=lambda r: (r["task_id"], r["method"])):
                score = row["score"] if bench == "browse" else row.get("method_score")
                source = row.get("batch") or row.get("source")
                writer.writerow({**row, "benchmark": bench, "score": score, "source": source})
                rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browse", type=Path, required=True)
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument("--science", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--cells", type=Path, help="Also write the per-run table as CSV")
    args = parser.parse_args()
    metas, results, tables = {}, {}, {}
    for bench in ("browse", "arc", "science"):
        data = json.loads(getattr(args, bench).read_text())
        metas[bench] = {k: v for k, v in data.items() if k not in ("rows", "run")}
        results[bench] = analyse(data["rows"], BENCH[bench])
        tables[bench] = data["rows"]
    generated = time.strftime("%Y-%m-%d")
    args.json.write_text(json.dumps({"generated": generated, "meta": metas, "results": results}, indent=1, default=str) + "\n")
    args.html.write_text(render(results, metas, generated))
    if args.cells:
        print("cells:", write_cells(args.cells, tables))
    for bench, a in results.items():
        print(bench, a["tasks"], a["bins"]["labels"], a["bins"]["counts"],
              {m: round(a["overall"][m]["mean"], 3) for m in a["methods"]})


if __name__ == "__main__":
    main()
