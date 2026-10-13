"""CLM in pi, the authors' way and Relay's: one Terminal-Bench task, the same budget, side by side.

    RELAY_TEST_KEYS=keys.env python evals/clm_parity.py --dataset DIR --out DIR [--task heat-pump-warranty]
                                                        [--budget 40000] [--arms pi-clm relay relay-noguard]
    python evals/clm_parity.py --report OUT [--budget 40000]

`pi-clm` is pi with the authors' extension (@lolipopshock/pi-clm, unpacked at --pi-clm) and its
defaults; `relay` is pi through Relay's `clm` strategy (its overflow guard on), `relay-noguard` with the
guard off (the budget left to the model). Every arm is a Harbor job of pi with the same model and
thinking level, on the OpenAI API (the key from RELAY_TEST_KEYS). The report reads pi's own log: per
request the context the model saw (the provider's count), how many requests went past the budget, and
the reward.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from terminal_bench import HOST, PORT, relay  # noqa: E402

MODEL, THINKING = "openai/gpt-6.1-sol", "high"
AGENT_DIR = "/tmp/harbor-pi-agent"  # Harbor's pi config directory when a base URL is set
ARMS = ("pi-clm", "relay", "relay-noguard")


def run(arm: str, n: int, out: Path, options: argparse.Namespace, key: str, relays: list) -> subprocess.Popen:
    out.mkdir(parents=True, exist_ok=True)
    mounts, env = [], {**os.environ, "OPENAI_API_KEY": key}
    command = ["harbor", "run", "-p", str(options.dataset), "-i", options.task, "-a", "pi", "-m", MODEL,
               "--ak", f"thinking={THINKING}", "--ak", "model_api=openai-responses",
               "-o", str(out / "jobs"), "--job-name", arm, "-n", "1", "-y"]
    if arm == "pi-clm":
        env["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
        mounts.append({"type": "bind", "source": str(options.pi_clm), "target": f"{AGENT_DIR}/extensions/pi-clm",
                       "read_only": True})
        command += ["--ae", f"PI_CLM_BUDGET={options.budget}"]
    else:
        home = out / "relay"
        home.mkdir(exist_ok=True)
        port = PORT + 10 + n
        (home / "pi-settings.json").write_text(json.dumps({"compaction": {"enabled": False}}))  # as `relay install` sets
        mounts.append({"type": "bind", "source": str(home / "pi-settings.json"), "target": f"{AGENT_DIR}/settings.json",
                       "read_only": True})
        relays.append(relay("clm", port, home, options.budget, clm_budget=options.budget,
                            clm_guard="on" if arm == "relay" else "off"))
        live = home / "live_ctx"
        mounts.append({"type": "bind", "source": str(live), "target": str(live)})
        env["OPENAI_BASE_URL"] = f"http://{HOST}:{port}/v1"
    command += ["--mounts", json.dumps(mounts)]
    return subprocess.Popen(command, stdout=open(out / "harbor.log", "a"), stderr=subprocess.STDOUT, env=env)


def trial(path: Path, budget: int) -> dict:
    result = json.loads((path / "result.json").read_text())
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    sizes = []
    log = path / "agent" / "pi.txt"
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        message = event.get("message") or {}
        if event.get("type") == "message_end" and message.get("role") == "assistant":
            usage = message.get("usage") or {}
            sizes.append(usage.get("totalTokens", 0) - usage.get("output", 0))  # (however it splits the cache)
    return {"reward": reward, "requests": len(sizes), "max": max(sizes, default=0),
            "over": sum(s > budget for s in sizes), "mean": sum(sizes) // max(1, len(sizes)),
            "error": (result.get("exception_info") or {}).get("exception_type")}


def report(out: Path, budget: int) -> None:
    print("| arm | reward | requests | mean context | max context | requests over budget | error |\n|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for path in sorted((out / arm / "jobs" / arm).glob("*__*")):
            if (path / "result.json").exists():
                r = trial(path, budget)
                print(f"| {arm} | {r['reward']} | {r['requests']} | {r['mean']:,} | {r['max']:,} | {r['over']} | {r['error'] or ''} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--task", default="heat-pump-warranty")
    parser.add_argument("--budget", type=int, default=40_000)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--pi-clm", type=Path, help="the unpacked @lolipopshock/pi-clm package")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    if options.report:
        report(options.report, options.budget)
        return 0
    keys = dict(line.split("=", 1) for line in Path(os.environ["RELAY_TEST_KEYS"]).read_text().split() if "=" in line)
    relays: list[subprocess.Popen] = []
    try:
        jobs = [run(arm, n, options.out / arm, options, keys["OPENAI_API_KEY"], relays) for n, arm in enumerate(options.arms)]
        return max(job.wait() for job in jobs)
    finally:
        for process in relays:
            process.terminate()


if __name__ == "__main__":
    sys.exit(main())
