"""Terminal-Bench through Relay: Codex, signed in with ChatGPT, under each strategy.

    python evals/terminal_bench.py --dataset DIR --out DIR [--arms native compaction clm folding]
                                   [--tasks name ...] [--window 128000] [-n 2]
    python evals/terminal_bench.py --report OUT [--window 128000]

DIR is a downloaded Terminal-Bench (`harbor download terminal-bench/terminal-bench@4.0.0`). Every
arm is one Harbor job of the same tasks, run side by side: `native` is Codex compacting itself,
the others go through a Relay of their own on this machine with Codex's compaction off. Every arm
sees the same context window (`--window`: Codex's for `native`, Relay's for the others).

The host's Codex login is copied with its refresh token voided (the trials cannot rotate the token
the host holds); its access token must outlast the run. Relay listens on the Docker bridge only,
where the task containers reach it. CLM's context files live in a host directory mounted at the
same path in every task container. Per arm, OUT/<arm> holds Harbor's job and Relay's settings,
events and log.

The report reads every trial's result and Codex's own session log: the reward, the time it took,
its model requests, the largest context the model saw (and how many requests saw more than the
window), the times the context shrank by a third or more (a compaction, a CLM edit, a fold), and
the compactions Codex recorded itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL, CODEX = "openai/gpt-6.1-sol", "0.162.1"
HOST, PORT = "172.17.0.1", 8900  # the Docker bridge; one port per arm from here
NEVER = 10**9  # as `relay install` sets Codex's own compaction out of reach
# The ten longest TB 4.0 tasks GPT-6.1 Sol / Codex (max effort) solves some but not all of the
# time on the leaderboard, without a GPU or a closed network.
TASKS = ["cumulative-layout-shift", "live-database-cutover", "ctr-optimization", "kv-live-surgery",
         "vba-userform-port", "retro-console-soc", "ks-solver-cpp", "biped-contact-dynamics",
         "vf2-speedup-networkx", "distributed-dedup"]
ARMS = ("native", "compaction", "clm", "folding")


def login(out: Path) -> Path:
    """The host's Codex login, its refresh token voided."""

    auth = json.loads(Path("~/.codex/auth.json").expanduser().read_text())
    auth["tokens"]["refresh_token"] = "voided-by-relay-evals"
    path = out / "auth.json"
    path.write_text(json.dumps(auth))
    path.chmod(0o600)
    return path


def relay(arm: str, port: int, home: Path, window: int, **extra: object) -> subprocess.Popen:
    settings = {"host": HOST, "port": port, "strategy": arm, "context_window": window,
                "event_log": str(home / "events.jsonl"), "cache_path": str(home / "store.json"), **extra}
    if arm == "clm":
        (home / "live_ctx").mkdir(mode=0o777, exist_ok=True)
        (home / "live_ctx").chmod(0o1777)  # (the task's user writes the model's edits there)
        settings["clm_dir"] = str(home / "live_ctx")
    (home / "settings.json").write_text(json.dumps(settings, indent=2))
    log = open(home / "relay.log", "a")
    process = subprocess.Popen([sys.executable, "-m", "relay.cli", "serve"], cwd=REPO, stdout=log, stderr=log,
                               env={**os.environ, "RELAY_HOME": str(home)})
    for _ in range(100):
        try:
            subprocess.run(["curl", "-sf", f"http://{HOST}:{port}/relay/v1/status"], check=True, capture_output=True)
            return process
        except subprocess.CalledProcessError:
            time.sleep(0.2)
    raise RuntimeError(f"Relay for {arm} did not start: see {home / 'relay.log'}")


def harbor(arm: str, out: Path, dataset: Path, tasks: list[str], auth: Path, window: int, port: int | None,
           concurrent: int) -> subprocess.Popen:
    config = out / "codex.toml"
    native = {"model_context_window": window} if port is None else {
        "model_context_window": NEVER, "model_auto_compact_token_limit": NEVER}
    config.write_text("".join(f"{key} = {value}\n" for key, value in native.items()))
    command = ["harbor", "run", "-p", str(dataset), *(x for t in tasks for x in ("-i", t)),
               "-a", "codex", "-m", MODEL, "--ak", f"version={CODEX}", "--ak", "reasoning_effort=high",
               "--ak", f"config={config}", "-o", str(out / "jobs"), "--job-name", arm, "-n", str(concurrent), "-y"]
    if arm == "clm":
        live = out / "relay" / "live_ctx"
        command += ["--mounts", json.dumps([{"type": "bind", "source": str(live), "target": str(live)}])]
    env = {**os.environ, "CODEX_AUTH_JSON_PATH": str(auth)}
    if port is not None:
        env["OPENAI_BASE_URL"] = f"http://{HOST}:{port}/backend-api/codex"
    log = open(out / "harbor.log", "a")
    return subprocess.Popen(command, stdout=log, stderr=log, env=env)


def trial(path: Path, window: int) -> dict:
    result = json.loads((path / "result.json").read_text())
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    sizes, compacted = [], 0
    for rollout in sorted(path.glob("agent/sessions/**/rollout-*.jsonl")):
        for line in rollout.read_text().splitlines():
            event = json.loads(line)
            payload = event.get("payload") or {}
            compacted += event.get("type") == "compacted"
            if payload.get("type") == "token_count" and payload.get("info"):
                sizes.append((payload["info"].get("last_token_usage") or {}).get("input_tokens", 0))
    start, end = (result.get(k) for k in ("started_at", "finished_at"))
    minutes = (_time(end) - _time(start)) / 60 if start and end else None
    error = (result.get("exception_info") or {}).get("exception_type")
    return {"task": path.name.split("__")[0], "reward": reward, "minutes": minutes, "requests": len(sizes),
            "max": max(sizes, default=0), "over": sum(s > window for s in sizes),
            "shrank": sum(b < 2 / 3 * a for a, b in zip(sizes, sizes[1:])), "codex_compacted": compacted,
            "error": error}


def _time(stamp: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def cell(r: dict | None) -> str:
    if r is None:
        return "—"
    notes = [f"{r['minutes']:.0f}m", f"{r['requests']} req", f"max {r['max'] // 1000}k"]
    notes += [f"{r['over']} over"] if r["over"] else []
    notes += [f"shrank {r['shrank']}"]
    notes += [f"codex compacted {r['codex_compacted']}"] if r["codex_compacted"] else []
    notes += [r["error"]] if r["error"] else []
    return f"{r['reward']} ({', '.join(notes)})"


def report(out: Path, window: int) -> None:
    rows = {arm: [trial(p, window) for p in sorted((out / arm / "jobs" / arm).glob("*__*")) if (p / "result.json").exists()]
            for arm in ARMS if (out / arm).exists()}
    tasks = sorted({r["task"] for arm in rows.values() for r in arm})
    print("| task | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
    for task in tasks:
        cells = []
        for arm in rows.values():
            r = next((r for r in arm if r["task"] == task), None)
            cells.append(cell(r))
        print(f"| {task} | " + " | ".join(cells) + " |")
    print("| **score** | " + " | ".join(f"{sum(r['reward'] or 0 for r in arm)}/{len(arm)}" for arm in rows.values()) + " |")
    for arm in rows:
        events = out / arm / "relay" / "events.jsonl"
        if events.exists():
            print(f"{arm}: Relay changed the context {sum(1 for _ in events.open())} times")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, help="report on a run's OUT instead")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--window", type=int, default=128_000)
    parser.add_argument("-n", "--concurrent", type=int, default=2, help="trials at once, per arm")
    options = parser.parse_args()
    if options.report:
        report(options.report, options.window)
        return 0
    if not (options.dataset and options.out):
        parser.error("--dataset and --out are required to run")
    options.out.mkdir(parents=True, exist_ok=True)
    auth = login(options.out)
    relays, jobs = [], []
    try:
        for n, arm in enumerate(options.arms):
            out = options.out / arm
            (out / "relay").mkdir(parents=True, exist_ok=True)
            port = None if arm == "native" else PORT + n
            if port is not None:
                relays.append(relay(arm, port, out / "relay", options.window))
            jobs.append(harbor(arm, out, options.dataset, options.tasks, auth, options.window, port, options.concurrent))
        return max(job.wait() for job in jobs)
    finally:
        for process in relays:
            process.terminate()


if __name__ == "__main__":
    sys.exit(main())
