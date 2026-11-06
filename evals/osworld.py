r"""OSWorld through Relay: a GUI agent on CarryOn's long OSWorld 2.1 tasks under each context-management method,
as Relay runs it and as the method's own harness does.

Setup (from the Relay checkout; Linux with Docker and KVM, uv, and a Hugging Face login with access to
xlangai/osworld_v2_assets_gated)::

    git clone https://github.com/xlang-ai/OSWorld-V2 OSWORLD
    cd OSWORLD && git switch --detach osworld-v2.1 && uv sync --frozen
    uv run scripts/tools/download_osworld_v2_tasks.py --benchmark-release osworld-v2.1
    uv run scripts/tools/download_osworld_v2_assets.py --benchmark-release osworld-v2.1 \
        --target-dir cache/osworld_v2_assets
    # and the Docker VM image, docker_vm_data/ (OSWORLD/docs/OSWORLD_SETUP_GUIDELINE.md)

This adapter targets OSWorld-V2 at osworld-v2.1 (3d778a3), its Docker provider and its Qwen agent
(`scripts/python/run_multienv_qwen.py`). The agent, the grader and the
user simulator all use one model at --base-url, an OpenAI-compatible endpoint, with OPENAI_API_KEY from the
file RELAY_TEST_KEYS names, as the other evals take keys. The endpoint must take images in tool results, since
the agent's screenshots arrive there. OpenAI's Chat Completions API takes none; the Responses API does, as a
function call's output. Keep the Hugging Face cache and login that the downloads used (HF_HOME, HF_TOKEN) in
the environment: the trials read the public assets through them.

A short smoke run, launched from the Relay checkout::

    RELAY_TEST_KEYS=keys.env python evals/osworld.py --osworld OSWORLD --out runs/osworld-smoke --tasks 004 \
        --arms none relay_compaction --steps 10
    python evals/osworld.py --report runs/osworld-smoke --csv runs/osworld.csv

Every arm runs the same tasks. Defaults are these ten arms, 150 steps and a 64K budget:

    none                 The whole history, append-only, until the budget wall.
    native_fold          The scaffold's own screenshot folding (newest --image-max kept); no wall.
    relay_compaction     Relay's Codex-style compaction.
    relay_selfcompact    Relay's Self-Compact.
    relay_acm            Relay's ACM (manage_context through an echo-only shell).
    relay_folding        Relay's context folding (branch/return through an echo-only shell).
    relay_clm            Relay's CLM, nudges and guard off, as in the paper's ContextBench runs.
    native_self_compact  Self-Compact in its own harness (osworld_agent.py MethodClient).
    native_acm           ACM in its own harness: manage_context and query_memory tools.
    native_folding       FoldAgent's branch and return tools.

Any relay_<strategy> works the same way. The agent is OSWorld's Qwen scaffold over native function calling
(osworld_agent.py), with absolute 1920x1080 coordinates: gpt-5.6-sol on the ChatGPT backend acts only through
declared tools and emits pixels. The task is the only user message; screenshots arrive as tool results. Every
trial gets its own Relay on the Docker bridge. Where the arm manages its own context, Relay's window is out of
reach and it only passes requests through, so every arm's traffic is logged the same way. Every arm but
native_fold ends at the budget wall: once the model's own count of a request passes --window less 2,048, the
run ends as a failure. The scaffold has no shell, so for relay_clm the agent offers the model a bash tool in a
sandbox container (SANDBOX_IMAGE, no network) holding only the directory Relay writes the context file to. CLM
has no native arm: its authors released no harness for a GUI agent (Terminal-Bench and EdgeBench run CLM through
Relay alone too).

TASKS defaults to the 24 of CarryOn's 40 (arXiv 2610.18784) that need no website. The other 16 (WEB) need
Task-Web/OSWorld-web deployed at osworld-v2.1; `--tasks carryon40` runs all 40. The grader is not the
leaderboard's, so scores compare across arms, not with published numbers.

Trials run task-major, `--parallel` VMs at a time. Each VM takes 4 cores and 4 GB, so keep this small on a
shared machine. An --out directory can be run again to finish it. run.json records the settings, and a run with
other settings is refused. A trial with a score is skipped. A trial without one (Docker, the subscription's
limit) is moved aside to <task>.failed-N and run again. Per arm and task, OUT/<arm>/<task>/ holds agent.log,
OSWorld's results/ (trajectory, result.txt), method_log.json (the method's calls, the wall) and relay/
(settings, events, log).

The report prints each trial's score with its steps (`w`: it hit the wall), each arm's mean, and the trials
run.json names that have no score. --csv writes one row per trial.

Local checks (no Docker, VMs or model calls)::

    python -m pytest tests/test_osworld.py -q

These cover the arms' settings, the trial's command and environment, sandbox cleanup when the agent fails,
resuming and reporting. A real smoke run is still needed to check the VMs and the model end to end.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from terminal_bench import HOST, NEVER, relay  # noqa: E402

EVALS = Path(__file__).resolve().parent
MODEL = "gpt-5.6-sol"
RESERVE = 2048
CARRYON40 = ("001 002 003 004 005 006 008 009 012 013 014 018 030 031 038 039 040 046 047 048 049 057 059 061 063 "
             "064 066 070 071 073 076 079 082 083 087 089 093 094 095 102").split()
WEB = set("001 002 005 008 013 018 031 038 039 046 057 070 073 079 082 102".split())
TASKS = [t for t in CARRYON40 if t not in WEB]
ARMS = ("none", "native_fold", "relay_compaction", "relay_selfcompact", "relay_acm", "relay_folding", "relay_clm",
        "native_self_compact", "native_acm", "native_folding")
# The whole history reaches the method: the scaffold's own folding, history cap and context rescue out of reach.
FULL = ["--image_max", "100000", "--history_n", "100000", "--no-context_rescue"]
SANDBOX_IMAGE = "python:3.12-slim"


def paper_modes(*names: str) -> tuple[Any, ...]:
    """The paper modes' prompts and tools, as context_bench_agent.py defines them for ContextBench. That module
    imports ContextBench and Harbor, which OSWorld's environment lacks; its paper-modes section needs only the
    standard library, so osworld_agent.py gets them by running that section on its own."""

    path = EVALS / "context_bench_agent.py"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(n for n, line in enumerate(lines) if line.startswith("# ===") and line.rstrip().endswith("the paper modes"))
    end = next(n for n in range(start + 1, len(lines)) if lines[n].startswith("# ==="))
    namespace = {"__file__": str(path), "__name__": "context_bench_agent", "ast": ast, "re": re, "Path": Path, "Any": Any}
    exec(compile("\n" * start + "".join(lines[start:end]), str(path), "exec"), namespace)  # (its own line numbers)
    return tuple(namespace[name] for name in names)


def arm_config(arm: str, window: int, image_max: int, out: Path) -> tuple[str, int, list[str], dict[str, str], dict]:
    """(Relay's strategy, Relay's window, the agent's options, the agent's environment, Relay's further settings).
    Where the arm manages its own context, Relay's window is out of reach: it passes the requests through."""

    if arm == "native_fold":
        return "compaction", NEVER, ["--image_max", str(image_max), "--fold_size", "5"], {}, {}
    if arm.startswith("relay_"):
        strategy = arm.removeprefix("relay_")
        env = {"OSW_MODE": "relay", "OSW_STRATEGY": strategy}
        extra: dict = {}
        if strategy == "clm":
            env["OSW_CTX_DIR"] = str(out / "relay" / "live_ctx")  # (where Relay writes the file: relay() names it so)
            extra = {"clm_nudges": "off", "clm_guard": "off", "clm_reserve": RESERVE}
        return strategy, window, FULL, env, extra
    mode = "none" if arm == "none" else arm.removeprefix("native_")
    if mode not in ("none", "self_compact", "acm", "folding"):
        raise ValueError(f"unknown arm {arm}")
    return "compaction", NEVER, FULL, {"OSW_MODE": mode}, {}


def score(out: Path) -> float | None:
    found = sorted(out.glob("results/**/result.txt"))
    return float(found[0].read_text().strip()) if found else None


def finished(out: Path) -> bool:
    """The trial has a score; one without is moved aside to <task>.failed-N, to run again."""

    if score(out) is not None:
        return True
    if out.exists():
        n = sum(1 for _ in out.parent.glob(f"{out.name}.failed-*"))
        shutil.move(str(out), str(out.with_name(f"{out.name}.failed-{n}")))
    return False


def run_trial(arm: str, task: str, port: int, options: argparse.Namespace) -> float | None:
    out = options.out / arm / task
    home = out / "relay"
    home.mkdir(parents=True, exist_ok=True)
    strategy, window, agent, method_env, extra = arm_config(arm, options.window, options.image_max, out)
    base = f"http://{HOST}:{port}/v1"
    pythonpath = [str(EVALS), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, **method_env, "OSW_BUDGET": str(options.window), "OSW_LOG_DIR": str(out),
           "PYTHONPATH": os.pathsep.join(filter(None, pythonpath)),
           "OSWORLD_FILE_BASE_URL": str(options.assets), "OPENAI_API_KEY": options.api_key, "OPENAI_BASE_URL": base,
           "OSWORLD_EVAL_MODEL_PROVIDER": "openai_compatible", "OSWORLD_EVAL_MODEL_NAME": options.model,
           "OSWORLD_EVAL_MODEL_BASE_URL": options.base_url, "OSWORLD_EVAL_MODEL_API_KEY": options.api_key,
           "OSWORLD_USER_SIM_PROVIDER": "openai_compatible", "OSWORLD_USER_SIM_MODEL": options.model,
           "OSWORLD_USER_SIM_BASE_URL": options.base_url, "OSWORLD_USER_SIM_API_KEY": options.api_key}
    process = relay(strategy, port, home, window, openai_base_url=options.base_url, **extra)
    sandbox = None
    try:
        if "OSW_CTX_DIR" in method_env:
            ctx = Path(method_env["OSW_CTX_DIR"])
            ctx.mkdir(parents=True, exist_ok=True)
            sandbox = env["OSW_SANDBOX"] = f"osw-sbx-{port}"
            subprocess.run(["docker", "run", "-d", "--rm", "--name", sandbox, "--network", "none",
                            "--user", f"{os.getuid()}:{os.getgid()}", "--cpus", "1", "--memory", "512m",
                            "-v", f"{ctx}:{ctx}", SANDBOX_IMAGE, "sleep", "86400"], check=True, capture_output=True)
        command = [str(options.osworld / ".venv" / "bin" / "python"), str(EVALS / "osworld_agent.py"),
                   "--headless", "--provider_name", "docker", "--no-enable_proxy", "--model", options.model,
                   "--base_url", base, "--no-stream", "--no-enable_thinking", "--coord", "absolute",
                   "--model_screen_width", "1920", "--model_screen_height", "1080", *agent,
                   "--specific_task_id", task, "--test_all_meta_path", str(options.meta),
                   "--max_steps", str(options.steps), "--num_envs", "1", "--result_dir", str(out / "results")]
        with open(out / "agent.log", "w") as log:
            subprocess.run(command, cwd=options.osworld, env=env, stdout=log, stderr=log)
    finally:
        process.terminate()
        if sandbox:
            subprocess.run(["docker", "rm", "-f", sandbox], capture_output=True)
    return score(out)


def steps(out: Path) -> int:
    found = sorted(out.glob("results/**/traj.jsonl"))
    return sum(1 for _ in found[0].open()) if found else 0


def walled(out: Path) -> bool:
    log = out / "method_log.json"
    return log.exists() and any(e.get("kind") == "budget_wall" for e in json.loads(log.read_text()))


# The settings a resumed --out must keep (its arms and tasks may grow).
SETTINGS = ("model", "window", "steps", "image_max")


def record(out: Path, options: argparse.Namespace, tasks: list[str]) -> str | None:
    """Write run.json, or add this invocation's arms and tasks to it; a reason to refuse, if any."""

    path = out / "run.json"
    mine = {key: getattr(options, key) for key in SETTINGS}
    data = json.loads(path.read_text()) if path.exists() else {**mine, "arms": [], "tasks": []}
    if changed := [key for key in SETTINGS if data.get(key) != mine[key]]:
        return f"--out holds a run with other {', '.join(changed)} ({path}); use a new directory"
    data["arms"] = list(dict.fromkeys([*data["arms"], *options.arms]))
    data["tasks"] = list(dict.fromkeys([*data["tasks"], *tasks]))
    path.write_text(json.dumps(data, indent=1))
    return None


def report(out: Path, csv_path: Path | None = None) -> None:
    metadata = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    arms = [a for a in dict.fromkeys([*ARMS, *metadata.get("arms", []), *sorted(p.name for p in out.iterdir())])
            if (out / a).is_dir() and (a in metadata.get("arms", []) or any((out / a).glob("*/results")))]
    tasks = sorted(set(metadata.get("tasks", [])) | {
        p.name for a in arms for p in (out / a).iterdir() if p.is_dir() and ".failed-" not in p.name})
    rows: list[dict] = []
    missing: list[str] = []
    print("| arm | " + " | ".join(tasks) + " | mean |\n|---|" + "---|" * (len(tasks) + 1))
    for arm in arms:
        cells = []
        for task in tasks:
            trial = out / arm / task
            value = score(trial)
            if value is None:
                cells.append("—")
                if arm in metadata.get("arms", []) and task in metadata.get("tasks", []):
                    missing.append(f"{arm}/{task}")
                continue
            rows.append({"arm": arm, "task": task, "score": value, "steps": steps(trial), "walled": walled(trial)})
            cells.append(f"{value:.2f} ({rows[-1]['steps']}{' w' if rows[-1]['walled'] else ''})")
        mine = [r["score"] for r in rows if r["arm"] == arm]
        mean = f"**{sum(mine) / len(mine):.2f}** (n={len(mine)})" if mine else "—"
        print(f"| {arm} | " + " | ".join(cells) + f" | {mean} |")
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        if mine:
            print(f"{arm}: mean {sum(r['score'] for r in mine) / len(mine):.2f} over {len(mine)} tasks, "
                  f"{sum(r['walled'] for r in mine)} at the wall")
    if missing:
        print(f"no score yet for {', '.join(missing)}")
    if csv_path and rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} rows in {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osworld", type=Path, help="the OSWorld-V2 checkout at osworld-v2.1, its .venv synced")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, help="report on a run's OUT instead")
    parser.add_argument("--csv", type=Path, help="write the report's rows, one per trial")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), help="ARMS, or any relay_<strategy>")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="task ids, or `carryon40` for all 40")
    parser.add_argument("--window", type=int, default=65536, help="the context budget: Relay's window and the wall")
    parser.add_argument("--steps", type=int, default=150, help="the agent's step limit")
    parser.add_argument("--image-max", type=int, default=10, help="native_fold: the screenshots it keeps")
    parser.add_argument("--parallel", type=int, default=3, help="VMs at once (4 cores, 4 GB each)")
    parser.add_argument("--port", type=int, default=9600, help="the first slot's Relay port; one port per slot")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default="https://api.openai.com/v1",
                        help="the OpenAI-compatible endpoint the agent (through its Relay), the grader and the user "
                             "simulator reach the model at")
    parser.add_argument("--assets", type=Path, help="the downloaded assets; OSWORLD/cache/osworld_v2_assets by default")
    options = parser.parse_args()

    if options.report:
        report(options.report.expanduser().resolve(), options.csv)
        return 0
    if not (options.osworld and options.out):
        parser.error("--osworld and --out are required to run")
    if not os.environ.get("RELAY_TEST_KEYS"):
        parser.error("RELAY_TEST_KEYS must name a keys file holding OPENAI_API_KEY")
    keys = dict(line.split("=", 1) for line in Path(os.environ["RELAY_TEST_KEYS"]).read_text().split() if "=" in line)
    if not keys.get("OPENAI_API_KEY"):
        parser.error(f"{os.environ['RELAY_TEST_KEYS']} holds no OPENAI_API_KEY")
    options.api_key = keys["OPENAI_API_KEY"]
    options.osworld = options.osworld.expanduser().resolve()
    options.out = options.out.expanduser().resolve()
    options.assets = (options.assets or options.osworld / "cache" / "osworld_v2_assets").expanduser().resolve()
    if not (options.osworld / ".venv" / "bin" / "python").exists():
        parser.error("--osworld must name an OSWorld-V2 checkout with its .venv synced (uv sync --frozen)")
    if not options.assets.is_dir():
        parser.error("--assets must name the downloaded assets (download_osworld_v2_assets.py)")
    for arm in options.arms:
        try:
            arm_config(arm, options.window, options.image_max, Path())
        except ValueError as error:
            parser.error(str(error))
    tasks = [t for name in options.tasks for t in (CARRYON40 if name == "carryon40" else [name])]
    options.out.mkdir(parents=True, exist_ok=True)
    if refused := record(options.out, options, tasks):
        parser.error(refused)
    options.meta = options.out / "tasks.json"
    options.meta.write_text(json.dumps({"tasks": CARRYON40}))

    trials: queue.Queue[tuple[str, str]] = queue.Queue()
    for task in tasks:
        for arm in options.arms:
            if not finished(options.out / arm / task):
                trials.put((arm, task))
    print(f"{trials.qsize()} trials, {options.parallel} at a time", flush=True)

    def worker(slot: int) -> None:
        while True:
            try:
                arm, task = trials.get_nowait()
            except queue.Empty:
                return
            started = time.time()
            value = run_trial(arm, task, options.port + slot, options)
            print(f"{time.strftime('%H:%M')} {arm} {task} ({(time.time() - started) / 60:.0f} min): score={value}",
                  flush=True)

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(options.parallel)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
