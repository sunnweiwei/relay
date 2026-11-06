r"""ContextBench through Relay: the CLM paper's context-management methods on its ContextBench, as the paper
ran them and through Relay, at the paper's 32K budget.

Setup (from the Relay checkout, whose Python runs this driver and Relay; Docker required)::

    pip install -e '.[dev]'
    git clone https://github.com/RulinShao/ContextBench CB && git -C CB checkout fb533c1
    git clone https://github.com/facebookresearch/context-language-models CLM && git -C CLM checkout 4831982
    python -m venv clm-venv && clm-venv/bin/pip install 'harbor==0.16.1' 'rlms==0.1.3' -e CB -e CLM
    python -m venv harbor-venv && harbor-venv/bin/pip install 'harbor==0.24.0' -e CB   # Codex arms only
    # Codex arms and oracle_summary: codex-package-x86_64-unknown-linux-musl of Codex 0.162.1, unpacked

This adapter targets ContextBench at fb533c1 (the paper's 256 tasks, its skills and configs) and the CLM
authors' code at 4831982. Harbor 0.16.1, the CLM package's pin, runs every arm but the Codex ones. Those agents
reach the model at --base-url, an OpenAI-compatible endpoint, with OPENAI_API_KEY from the file RELAY_TEST_KEYS
names, as the other evals take keys. ContextBench's own agent sends temperature 0.7 and max_tokens. Reasoning
models on OpenAI's API refuse both, so the endpoint must accept or drop them. Harbor 0.24.0 runs the Codex arms,
whose Codex signs in with the host's ChatGPT login (copied with its refresh token voided). Every arm runs the
same model. The agents themselves, and the paper modes' prompts and tools, are in context_bench_agent.py, run by
Harbor from its own environment.

A short smoke run, launched from the Relay checkout::

    RELAY_TEST_KEYS=keys.env python evals/context_bench.py --tasks-dir CB/tasks --out runs/cb-smoke \
        --tasks kv-store-R130-seed42 --arms oracle_base chat_acm --oracle-harbor clm-venv/bin/harbor
    python evals/context_bench.py --report runs/cb-smoke --csv runs/cb.csv

Every arm runs the same tasks. Defaults are the oracles and the Codex arms (rlm aside) on TASKS:

    oracle_base            The paper's append-only loop (ContextBenchAgent), its base skill.
    oracle_summary         Codex's own compaction at 75% of the budget (ContextBench's CodexStream).
    oracle_clm             The CLM authors' ClmStream, their settings and skill.
    paper_<mode>           The paper's modes on ContextBench's loop (PaperAgent): base, acm, self_compact,
                           context_folding, rlm, with the methods' own prompts and tools.
    chat_<arm>             Relay's strategy for <arm> (below) in front of that same loop (RelayChatAgent).
    base                   Codex through Relay, nothing managed (Relay's window out of reach).
    summary                Relay's Codex-style compaction at 75% of the budget (Codex keeps 20K of user text).
    summary_paper          The same, keeping no user messages, as the paper's summary harness did.
    self_compact, acm,     Relay's selfcompact, acm, folding, rlm_persistent, autocompact and mem1.
    context_folding, rlm,
    autocompact, mem1
    clm                    Relay's clm, nudges and guard off, as in the paper's ContextBench runs.

ContextBench streams more than the context holds, one operation per user turn. It grades what survived in the
context the model was sent: kv_store the GETs answered, log_triage the questions answered, needle_retention
the needle lines kept verbatim, sudoku_sketchpad the board versions shown. The paper's figure ran GPT-5.4 at a
32,768-token limit (2,048 reserved; CB_BUDGET sets another, the skills' numbers scaled to it). Each arm gets
the method's skill (`clm_exp/skills/<arm>/<family>.md`, short versions where the paper used them), placed
where the paper placed it. Where Relay's interface differs, the skill keeps its text and gains a note ahead
of it: ACM's and folding's tools are shell signals, and CLM's file is the one Relay names. RLM runs without a
skill. The paper's RLM skill describes a REPL whose model writes the code, while Relay's RLM works out each
step itself.

The Codex arms run ContextBench's CodexStream (one `codex exec resume` per operation), with Codex's own
compaction off, through a Relay per trial on the Docker bridge. What the model saw is what Relay forwarded
(its trace), and the budget wall (30,720 o200k tokens, ContextBench's count) applies to that. Codex's tool
definitions, about 7,100 tokens (OVERHEAD), reach the model but not ContextBench's count. Relay counts them,
so its window is the budget plus them, which puts every limit Relay holds at the budget's. Self-Compact's
probe gate is 37% of Relay's window, so it starts asking at about 7.7K of ContextBench's count where the
paper's asked at 12.1K. Codex's own history only grows. Codex keeps every operation in its home, where models
found and grepped them, so the model's commands run as the host's user, without access to it. For the same
reason every agent keeps ContextBench's snapshots out of the task container until it is done.

The paper's own harness for its baselines is not released, so the paper_* arms rebuild those modes on
ContextBench's loop, and the chat_* arms put Relay in front of the same loop. That way the paper's way and
Relay's way differ only in where the method runs. The chat harness's one tool definition is likewise outside
ContextBench's count (CHAT_OVERHEAD). Operations arrive as user turns, as ContextBench delivers them, so how a
method treats the user's later messages matters:
- Relay's ACM and Self-Compact compress them, as their released code does.
- compaction keeps the newest 20K tokens of them, as Codex does.
- CLM's guidance asks the model to leave them as they are.

TASKS are the paper's rungs nearest 3x and 6x pressure, seeds 42 and 43. `--tasks figure2` is its Figure 2
ladder (FIGURE2). An --out directory can be run again to finish it. run.json records the model and budget,
and a run with others is refused. A trial that reached the agent's or the verifier's end is skipped. Any
other trial (Docker, the subscription's rate limit) is moved aside to <task>.failed-N and run again, in the
next pass (--passes) or the next invocation. Per arm, OUT/<arm>/<task>/ holds the trial's skill, Harbor's job
and log, and its Relay (settings, trace, events, log).

The report prints each rung's mean reward per arm with how its trials ended, each arm's mean, and the trials
run.json names that have no result. --csv writes one row per trial.

Local checks (no Harbor, Docker or model calls; the agent checks skip without ContextBench installed)::

    python -m pytest tests/test_context_bench.py -q

These cover the task ladder, the arms' Relay settings and skills, the trial's Harbor command, Relay's cleanup
when Harbor fails, resuming and reporting. A real smoke run is still needed to check Harbor, Docker and the
model end to end.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from terminal_bench import HOST, NEVER, login, relay  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
# ContextBench's budget as it counts it; CB_BUDGET sets another (read here, so the trials' processes see it too).
WINDOW, RESERVE = int(os.environ.get("CB_BUDGET", "32768")), 2048
# Codex's tool definitions, which ContextBench does not count and Relay does (measured: 7,078-7,217).
OVERHEAD = 7100
RELAY_WINDOW = WINDOW + OVERHEAD
FAMILIES = ("kv_store", "log_triage", "needle_retention", "sudoku_sketchpad")
TASKS = [f"{family}-{rung}-seed{seed}" for family, rungs in (
    ("kv-store", ("R2000", "R4000")), ("log-triage", ("L3200", "L6400")),
    ("needle-retention", ("K4-C24", "K8-C48")), ("sudoku-sketchpad", ("t30", "t60")))
    for rung in rungs for seed in (42, 43)]
# The paper's Figure 2 ladder (paper.yaml, main_figure), longest first, so a run cut short has those done.
_LONG = ("kv-store-R16000 needle-retention-K2-C224 needle-retention-K2-C192 log-triage-L12800 sudoku-sketchpad-t150 "
         "kv-store-R8000 needle-retention-K4-C96 sudoku-sketchpad-t100 log-triage-L6400 log-triage-L3200 "
         "needle-retention-K8-C60 kv-store-R4000 sudoku-sketchpad-t60 needle-retention-K8-C48 kv-store-R2000 "
         "sudoku-sketchpad-t30 needle-retention-K4-C24").split()
_SHORT = ("log-triage-L1600 kv-store-R1000 needle-retention-K4-C8 sudoku-sketchpad-t15 log-triage-L640 kv-store-R520 "
          "sudoku-sketchpad-t8 needle-retention-K4-C4 log-triage-L320 kv-store-R260 sudoku-sketchpad-t4 log-triage-L160 "
          "kv-store-R130").split()
FIGURE2 = [f"{r}-seed{s}" for r in _LONG for s in (42, 43)] + [f"{r}-seed{s}" for r in _SHORT for s in (42, 43, 44, 45)]

# arm -> (Relay's strategy, its settings, the paper's skill directory or None)
ARMS: dict[str, tuple[str, dict[str, Any], str | None]] = {
    "base": ("compaction", {"context_window": NEVER}, "base"),
    "summary": ("compaction", {"compact_ratio": round((0.75 * WINDOW + OVERHEAD) / RELAY_WINDOW, 4)}, "summary"),
    "self_compact": ("selfcompact", {}, "self_compact"),
    "acm": ("acm", {}, "acm"),
    "rlm": ("rlm_persistent", {"rlm_environment": "docker"}, None),
    "context_folding": ("folding", {}, "context_folding"),
    "clm": ("clm", {"clm_nudges": "off", "clm_guard": "off", "clm_reserve": RESERVE}, "clm"),
    "summary_paper": ("compaction", {"compact_ratio": round((0.75 * WINDOW + OVERHEAD) / RELAY_WINDOW, 4),
                                     "retain_user_tokens": 0}, "summary"),
    # No paper skill describes AutoCompact or MEM1: AutoCompact gets summary's (its fallback), MEM1 none.
    "autocompact": ("autocompact", {"compact_ratio": round((0.75 * WINDOW + OVERHEAD) / RELAY_WINDOW, 4)}, "summary"),
    "mem1": ("mem1", {}, None),
}

# The paper's own implementations, as the authors ship them. arm -> (agent, skill directory)
ORACLES: dict[str, tuple[str, str | None]] = {
    "oracle_base": ("context_bench_agent:OracleBase", "base"),
    "oracle_summary": ("context_bench_agent:OracleSummary", "summary"),
    "oracle_clm": ("clm_contextbench:ClmStream", None),  # (it inserts the paper's CLM skill itself)
}

# arm -> (PaperAgent's mode, skill directory); arm -> Relay's arm, for RelayChatAgent
PAPER: dict[str, tuple[str, str]] = {
    "paper_base": ("base", "base"), "paper_acm": ("acm", "acm"),
    "paper_self_compact": ("self_compact", "self_compact"), "paper_context_folding": ("folding", "context_folding"),
    "paper_rlm": ("rlm", "rlm"),
}
CHAT: dict[str, str] = {"chat_base": "base", "chat_acm": "acm", "chat_self_compact": "self_compact",
                        "chat_context_folding": "context_folding", "chat_clm": "clm", "chat_rlm": "rlm",
                        "chat_summary": "summary", "chat_summary_paper": "summary_paper",
                        "chat_autocompact": "autocompact", "chat_mem1": "mem1"}
# Relay's RLM hands each step to a plain bash agent, so that agent gets base's skill (the paper's is for the REPL).
CHAT_SKILL = {"chat_rlm": "base"}
# What the chat harness sends that ContextBench does not count: its one tool's definition. Relay counts it.
CHAT_OVERHEAD = 150
CHAT_NOTES = {
    "acm": """**In this harness `manage_context` and `query_memory` are shell signals, not separate tools.**
Call `manage_context` by running `echo "[manage_context]"` as a command of its own, and
`query_memory(N, "query")` by running `echo "[query_memory] N :: query"`. The gauge after a command
reads `[CURRENT CONTEXT TOKEN: N]`. Everything below about the two tools holds as written.
""",
}

NOTES = {
    "acm": """**In this harness `manage_context` and `query_memory` are shell signals, not separate tools.**
Call `manage_context` by running `echo "[manage_context]"` as a command of its own, and
`query_memory(N, "query")` by running `echo "[query_memory] N :: query"`. The gauge after a command
reads `[CURRENT CONTEXT TOKEN: N]`, and it also counts about {overhead:,} tokens of tool definitions that
the budget below does not: subtract {overhead:,} before you compare it with any number below (22,000 below
is a gauge of {t22:,}). Everything below about the two tools holds as written.
""".format(overhead=OVERHEAD, t22=22000 + OVERHEAD),
    "context_folding": """**In this harness `branch` and `return` are shell signals, not separate tools.**
`branch(description, prompt)` is the command `echo "[branch] <description> :: <prompt>"`, and
`return(summary)` is `echo "[return] <summary>"`. Everything below about branches holds as written.
""",
    "clm": """**In this harness your context file is not `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`**: it is the file
named under "Managing your context". Use that path wherever the text below says
`/tmp/.live_ctx/LIVE_CTX_MAIN.txt`. Its turns are `[[CTX_TURN ...]]` blocks; text you cut out of them
leaves your context, as below.
""",
}


# The rows of the paper's figure that ran base's and summary's short skill (`<family>.short.md`), or none
# (clm_exp/skills/README.md); every other row ran `<family>.md`. "*": every rung (sudoku, seeds 42-45).
_LOW = ("R130", "R260", "R520", "R1000", "R2000", "R4000", "L160", "L320", "L640", "L1600", "L3200")
_NEEDLE = ("K4-C4", "K4-C6", "K4-C8", "K4-C24", "K8-C48", "K8-C60", "K8-C62")
SHORT = {"base": {"kv_store": _LOW, "log_triage": _LOW, "sudoku_sketchpad": "*"},
         "summary": {"kv_store": _LOW, "log_triage": _LOW, "needle_retention": _NEEDLE, "sudoku_sketchpad": "*"}}
NO_SKILL = {"base": {"needle_retention": _NEEDLE}}


def level(task: str) -> str:
    return re.sub(r"^(kv-store|log-triage|needle-retention|sudoku-sketchpad)-|-seed\d+$", "", task)


# The skills are written for the 32,768 budget; under another, their budget numbers (not the task's sizes) scale.
_SCALED = (r"32,768", r"30,720", r"22,000", r"25,000", r"24,000", r"12\.1K")
_COUNTS = {"six": 6, "thirty": 30}
_WORDS = {12: "twelve", 24: "twenty-four", 60: "sixty", 120: "a hundred and twenty"}


def scaled(text: str) -> str:
    if WINDOW == 32768:
        return text
    factor = WINDOW / 32768

    def number(match: re.Match) -> str:
        value = match.group(0)
        if value.endswith("K"):
            return f"{float(value[:-1]) * factor:.1f}K"
        new = int(value.replace(",", "")) * factor
        return f"{WINDOW - RESERVE:,}" if value == "30,720" else f"{WINDOW:,}" if value == "32,768" else f"{int(new):,}"

    text = re.sub("|".join(_SCALED), number, text)
    def count(match: re.Match) -> str:
        n = round(_COUNTS[match.group(2)] * factor)
        return f"{match.group(1)} {_WORDS.get(n, str(n))}"

    return re.sub(r"\b(hold about|order of) (six|thirty)\b", count, text)


def skill(arm: str, family: str, rung: str, skills: Path) -> str:
    name = (ORACLES[arm][1] if arm in ORACLES else PAPER[arm][1] if arm in PAPER
            else CHAT_SKILL.get(arm) or ARMS[CHAT[arm]][2] if arm in CHAT else ARMS[arm][2])
    if name is None or rung in NO_SKILL.get(name, {}).get(family, ()):
        return ""
    short = SHORT.get(name, {}).get(family, ())
    text = (skills / name / f"{family}{'.short' if short == '*' or rung in short else ''}.md").read_text(encoding="utf-8")
    note = NOTES.get(name, "") if arm in ARMS else (CHAT_NOTES.get(name) or NOTES.get(name, "")) if arm in CHAT else ""
    return scaled(note + ("\n" if note else "") + text)


def with_skill(instruction: str, text: str, family: str) -> str:
    """Where the paper's task builder put the skill (clm_contextbench's README)."""

    if not text:
        return instruction
    if family == "sudoku_sketchpad":
        anchor = "## Turn protocol (IMPORTANT)"
        return instruction.replace(anchor, text.rstrip("\n") + "\n\n\n" + anchor, 1)
    anchor = "The run ends automatically"
    at = instruction.rfind(anchor)
    return instruction[:at] + "\n" + text + "\n" + instruction[at:] if at >= 0 else instruction + "\n" + text




def finished(out: Path) -> bool:
    """A trial of it came to an end of the agent's or the verifier's. One the environment or the
    upstream failed (a build, Docker, the subscription's rate limit: 429s past Codex's retries) is
    moved aside to OUT/<arm>/<task>.failed-N, to run again."""

    for result in (out / "jobs").glob("*/*__*/result.json"):
        data = json.loads(result.read_text())
        error = (data.get("exception_info") or {}).get("exception_type") or ""
        if error in ("AgentTimeoutError", "VerifierTimeoutError") or (data.get("verifier_result") and not error):
            return True  # (a verifier result after an upstream error, e.g. a 500 for the 429, is not an end)
    if out.exists():
        out.rename(out.with_name(f"{out.name}.failed-{sum(1 for _ in out.parent.glob(f'{out.name}.failed-*'))}"))
    return False


def run_trial(arm: str, task: str, port: int, options: argparse.Namespace, auth: Path | None) -> None:
    out = options.out / arm / task
    if finished(out):
        return
    home = out / "relay"
    home.mkdir(parents=True, exist_ok=True)
    family = (options.tasks_dir / task / "task.toml").read_text().split('contextbench_family = "')[1].split('"')[0]
    text = skill(arm, family, level(task), options.skills)
    (out / "skill.md").write_text(text)
    codex_mounts = [{"type": "bind", "source": str(options.codex / source), "target": f"/usr/local/{target}", "read_only": True}
                    for source, target in {**{f"bin/{e.name}": f"bin/{e.name}" for e in (options.codex / "bin").iterdir()},
                                           "codex-resources": "codex-resources", "codex-path": "codex-path",
                                           "codex-path/rg": "bin/rg"}.items()] if options.codex else []
    if arm in PAPER or arm in CHAT:
        kwargs: dict[str, Any] = {"skill_text": str(out / "skill.md"), "family": family,
                                  "context_budget_tokens": WINDOW, "context_budget_reserve_tokens": RESERVE}
        mounts, process = [], None
        if arm in PAPER:
            agent = "context_bench_agent:PaperAgent"
            kwargs |= {"api_base": options.base_url, "mode": PAPER[arm][0]}
        else:
            agent = "context_bench_agent:RelayChatAgent"
            strategy, extra, _ = ARMS[CHAT[arm]]
            home = out / "relay"
            home.mkdir(parents=True, exist_ok=True)
            window = WINDOW + CHAT_OVERHEAD
            extra = {k: v for k, v in extra.items() if k != "compact_ratio"} | (
                {"compact_ratio": round((0.75 * WINDOW + CHAT_OVERHEAD) / window, 4)} if "compact_ratio" in extra else {})
            process = relay(strategy, port, home, window, trace=str(home / "trace.jsonl"),
                            openai_base_url=options.base_url, **extra)
            kwargs |= {"api_base": f"http://{HOST}:{port}/v1", "relay_trace": str(home / "trace.jsonl"),
                       "gauge": "off" if strategy in ("acm", "clm") else "on", "settle": "on" if strategy == "clm" else "off"}
            if strategy == "clm":
                mounts.append({"type": "bind", "source": str(home / "live_ctx"), "target": str(home / "live_ctx")})
        command = [options.oracle_harbor, "run", "-p", str(options.tasks_dir), "-i", task, "-a", agent,
                   "-m", f"openai/{options.model}", *(x for k, v in kwargs.items() for x in ("--ak", f"{k}={v}")),
                   *(["--mounts-json", json.dumps(mounts)] if mounts else []),
                   "-o", str(out / "jobs"), "--job-name", task, "-n", "1", "--delete", "-y"]
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent), "OPENAI_API_KEY": options.api_key}
        try:
            with open(out / "harbor.log", "a") as log:
                subprocess.run(command, stdout=log, stderr=log, env=env)
        finally:
            if process is not None:
                process.terminate()
        print(f"{arm} {task}: done", flush=True)
        return
    if arm in ORACLES:
        kwargs = {"oracle_base": {"api_base": options.base_url, "skill_text": str(out / "skill.md"), "family": family},
                  "oracle_summary": {"codex_auth": str(auth), "skill_text": str(out / "skill.md"), "family": family},
                  "oracle_clm": {"api_base": options.base_url}}[arm]
        if arm != "oracle_summary":
            kwargs |= {"context_budget_tokens": WINDOW, "context_budget_reserve_tokens": RESERVE}
        command = [options.oracle_harbor, "run", "-p", str(options.tasks_dir), "-i", task, "-a", ORACLES[arm][0],
                   "-m", f"openai/{options.model}", *(x for k, v in kwargs.items() for x in ("--ak", f"{k}={v}")),
                   *(["--mounts-json", json.dumps(codex_mounts)] if arm == "oracle_summary" else []),
                   "-o", str(out / "jobs"), "--job-name", task, "-n", "1", "--delete", "-y"]
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent), "OPENAI_API_KEY": options.api_key}
        with open(out / "harbor.log", "a") as log:
            subprocess.run(command, stdout=log, stderr=log, env=env)
        print(f"{arm} {task}: done", flush=True)
        return
    strategy, extra, _ = ARMS[arm]
    process = relay(strategy, port, home, RELAY_WINDOW, trace=str(home / "trace.jsonl"), **extra)
    mounts = list(codex_mounts)
    if strategy == "clm":
        mounts.append({"type": "bind", "source": str(home / "live_ctx"), "target": str(home / "live_ctx")})
    kwargs = {"relay_url": f"http://{HOST}:{port}/backend-api/codex", "relay_trace": str(home / "trace.jsonl"),
              "codex_auth": str(auth), "skill_text": str(out / "skill.md"), "family": family,
              "context_budget_tokens": WINDOW, "context_budget_reserve_tokens": RESERVE}
    command = [options.harbor, "run", "-p", str(options.tasks_dir), "-i", task,
               "-a", "context_bench_agent:RelayCodexStream", "-m", f"openai/{options.model}",
               *(x for k, v in kwargs.items() for x in ("--ak", f"{k}={v}")),
               "--mounts", json.dumps(mounts), "-o", str(out / "jobs"), "--job-name", task, "-n", "1", "--delete", "-y"]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
    try:
        with open(out / "harbor.log", "a") as log:
            subprocess.run(command, stdout=log, stderr=log, env=env)
    finally:
        process.terminate()
    print(f"{arm} {task}: done", flush=True)


# The settings a resumed --out must keep (its arms and tasks may grow).
SETTINGS = ("model", "budget")
CODEX_ARMS = (*ARMS, "oracle_summary")  # (the arms that need --codex and the host's Codex login)


def record(out: Path, options: argparse.Namespace, tasks: list[str]) -> str | None:
    """Write run.json, or add this invocation's arms and tasks to it; a reason to refuse, if any."""

    path = out / "run.json"
    mine = {"model": options.model, "budget": WINDOW}
    data = json.loads(path.read_text()) if path.exists() else {**mine, "arms": [], "tasks": []}
    if changed := [key for key in SETTINGS if data.get(key) != mine[key]]:
        return f"--out holds a run with other {', '.join(changed)} ({path}); use a new directory"
    data["arms"] = list(dict.fromkeys([*data["arms"], *options.arms]))
    data["tasks"] = list(dict.fromkeys([*data["tasks"], *tasks]))
    path.write_text(json.dumps(data, indent=1))
    return None


def report(out: Path, csv_path: Path | None = None) -> None:
    rows: dict[str, dict[str, list[dict]]] = {}
    trials: list[dict] = []
    for result in sorted(out.glob("*/*/jobs/*/*__*/result.json")):
        arm, task = result.parts[-6], result.parts[-5]
        if ".failed-" in task:
            continue
        data = json.loads(result.read_text())
        meta = (data.get("agent_result") or {}).get("metadata") or {}
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        level = re.sub(r"-seed\d+$", "", task)
        trials.append({"arm": arm, "task": task, "level": level, "reward": reward, "end": meta.get("terminated_by"),
                       "peak": meta.get("peak_context_tokens"),
                       "error": (data.get("exception_info") or {}).get("exception_type")})
        rows.setdefault(level, {}).setdefault(arm, []).append(trials[-1])
    arms = [a for a in (*ORACLES, *PAPER, *ARMS, *CHAT) if any(a in r for r in rows.values())]
    print("| level | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    totals: dict[str, list[float]] = {a: [] for a in arms}
    for level in sorted(rows):
        cells = []
        for arm in arms:
            mine = rows[level].get(arm, [])
            scores = [t["reward"] for t in mine if t["reward"] is not None]
            totals[arm] += scores
            ends = ",".join(sorted({str(t["end"] or t["error"]) for t in mine}))
            cells.append(f"{sum(scores) / len(scores):.2f} ({ends})" if scores else (ends or "—"))
        print(f"| {level} | " + " | ".join(cells) + " |")
    print("| **mean** | " + " | ".join(f"**{sum(v) / len(v):.2f}** (n={len(v)})" if v else "—"
                                       for v in totals.values()) + " |")
    metadata = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    graded = {(t["arm"], t["task"]) for t in trials if t["reward"] is not None}
    missing = [f"{arm}/{task}" for arm in metadata.get("arms", []) for task in metadata.get("tasks", [])
               if (arm, task) not in graded]
    if missing:
        print(f"{len(missing)} trials without a result: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    if csv_path and trials:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
        print(f"{len(trials)} rows in {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, help="ContextBench's tasks/ (the paper's 256 tasks)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, help="report on a run's OUT instead")
    parser.add_argument("--csv", type=Path, help="write the report's rows, one per trial")
    parser.add_argument("--arms", nargs="+", choices=(*ORACLES, *PAPER, *ARMS, *CHAT), metavar="ARM",
                        default=[*ORACLES, *(a for a in ARMS if a != "rlm")], help="see the arms above")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="task names, or `figure2` for FIGURE2")
    parser.add_argument("--passes", type=int, default=1, help="go over the trials this many times (failed ones rerun)")
    parser.add_argument("--skills", type=Path, help="ContextBench's clm_exp/skills; beside --tasks-dir by default")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--parallel", type=int, default=8,
                        help="trials at once (the subscription answers 429s past about 12 Codex trials)")
    parser.add_argument("--port", type=int, default=9100, help="the first trial's Relay port; one port per trial")
    parser.add_argument("--harbor", default="harbor",
                        help="Harbor 0.24's executable, for the Codex arms (not its environment on PATH: Relay "
                             "would run on its Python, without zstandard)")
    parser.add_argument("--oracle-harbor", default="harbor",
                        help="Harbor 0.16.1's executable, with ContextBench and the CLM package installed beside it")
    parser.add_argument("--base-url", default="https://api.openai.com/v1",
                        help="the OpenAI-compatible endpoint every arm but the Codex ones reaches the model at")
    parser.add_argument("--codex", type=Path, help="Codex 0.162.1's static Linux package, unpacked")
    options = parser.parse_args()

    if options.report:
        report(options.report.expanduser().resolve(), options.csv)
        return 0
    if not (options.tasks_dir and options.out):
        parser.error("--tasks-dir and --out are required to run")
    codex = [a for a in options.arms if a in CODEX_ARMS]
    if codex and not options.codex:
        parser.error(f"--codex is required for {', '.join(codex)}")
    options.api_key = None
    if set(options.arms) - set(ARMS):
        if not os.environ.get("RELAY_TEST_KEYS"):
            parser.error("RELAY_TEST_KEYS must name a keys file holding OPENAI_API_KEY, for every arm but the Codex ones")
        keys = dict(line.split("=", 1) for line in Path(os.environ["RELAY_TEST_KEYS"]).read_text().split() if "=" in line)
        if not keys.get("OPENAI_API_KEY"):
            parser.error(f"{os.environ['RELAY_TEST_KEYS']} holds no OPENAI_API_KEY")
        options.api_key = keys["OPENAI_API_KEY"]
    import zstandard  # noqa: F401  (Relay passes Codex's zstd bodies through unread without it)

    options.tasks_dir = options.tasks_dir.expanduser().resolve()
    options.out = options.out.expanduser().resolve()
    options.skills = (options.skills or options.tasks_dir.parent / "clm_exp" / "skills").expanduser().resolve()
    options.codex = options.codex and options.codex.expanduser().resolve()
    for name in ("harbor", "oracle_harbor"):  # (a path, as given relative to here; a bare name, from PATH)
        if os.sep in getattr(options, name):
            setattr(options, name, str(Path(getattr(options, name)).expanduser().resolve()))
    tasks = [t for name in options.tasks for t in (FIGURE2 if name == "figure2" else [name])]
    if not options.tasks_dir.is_dir():
        parser.error("--tasks-dir must name ContextBench's tasks/ directory")
    if unknown := [t for t in tasks if not (options.tasks_dir / t / "task.toml").exists()]:
        parser.error(f"no such tasks in --tasks-dir: {', '.join(unknown[:5])}")
    if not options.skills.is_dir():
        parser.error("--skills must name ContextBench's clm_exp/skills (beside --tasks-dir by default)")
    if options.codex and not (options.codex / "bin").is_dir():
        parser.error("--codex must name the unpacked Codex package (it holds bin/ and codex-path/)")
    options.out.mkdir(parents=True, exist_ok=True)
    if refused := record(options.out, options, tasks):
        parser.error(refused)
    # The RLM Docker REPLs' working directories (Relay's and the paper harness's), under OUT.
    os.environ["RLM_DOCKER_WORKSPACE_DIR"] = str(options.out / ".rlm_workspace")
    auth = login(options.out) if codex else None
    trials = [(arm, task) for task in tasks for arm in options.arms]
    ports = itertools.count(options.port)  # (a port of its own per trial, so a Relay still stopping holds none)
    lock = threading.Lock()

    def one(trial: tuple[str, str]) -> None:
        with lock:
            port = next(ports)
        run_trial(*trial, port, options, auth)

    for n in range(options.passes):
        print(f"=== pass {n + 1} of {options.passes}: {len(trials)} trials", flush=True)
        with ThreadPoolExecutor(options.parallel) as pool:
            list(pool.map(one, trials))
    return 0


if __name__ == "__main__":
    sys.exit(main())
