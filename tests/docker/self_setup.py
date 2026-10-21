"""A harness sets Relay up for itself, told in a sentence, then uses it session after session.

    RELAY_TEST_KEYS=keys.env python tests/docker/self_setup.py codex:gpt [claude_code:gpt ...] [--ref BRANCH]

In the image, on a user's machine with one harness and nothing of Relay's (something else listens on
Relay's default port):

1. session 1: the harness is asked to install Relay with its installer, from GitHub (`--ref`, as
   `relay help` gives it) or from the checkout at /relay, and to set it up for itself;
2. Relay's thresholds are lowered, as a user tuning it would (`relay set`);
3. session 2: a task under compaction, the strategy setup chose;
4. session 3: the harness is asked to switch Relay to CLM;
5. session 4: the task under CLM;
6. `relay strategy folding`, then session 5: the task under Folding.

Checks: Relay runs, on a port of its own, and the harness goes through it; each task's answer is
right and its requests had the strategy (compaction: one of them compacted; CLM and Folding:
their guidance in what reached the model). Sessions after the install start from a new login
shell, as after the installer put `relay` on PATH.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check import CODES, RULES, SPECS, container  # noqa: E402

RELAY = "~/.local/bin/relay"
INSTALL = ("Install Relay for yourself, {harness}: run `{install}`, then set it up for {harness} with `relay setup`; "
           "`relay help` explains it. Reply with what you did.")
CHECKOUT = "sh /relay/install.sh /relay"
GITHUB = ("curl -fsSL https://raw.githubusercontent.com/sunnweiwei/relay/{ref}/install.sh | "
          "sh -s -- git+https://github.com/sunnweiwei/relay.git@{ref}")
SWITCH = "Switch Relay to the clm strategy (`relay help` says how). Reply with what you did."
TASK = "[{tag}] Read the files file_1.txt to file_4.txt in the current directory, " + RULES + (
    " After reading them, reply with their 4 codes in order, comma-separated, and nothing else.")
MARKS = {"clm": "## Managing your context", "folding": "## Branches (context folding)"}


def run(name: str, ref: str | None = None) -> dict:
    harness, spec = name.split(":")[0], SPECS[name]
    root = Path(tempfile.mkdtemp(prefix=f"relay-self-setup-{harness}-"))
    (root / "home").mkdir()
    (root / "project").mkdir()
    for n, code in enumerate(CODES[:4], start=1):
        text = "".join(f"file {n} line {i}: the quick brown fox jumps over the lazy dog\n" for i in range(1, 121))
        (root / "project" / f"file_{n}.txt").write_text(f"{text}CODE: {code}\n")

    def session(n: int, prompt: str) -> str:  # (after the install, from a new login shell: its PATH)
        return (f"{'. ~/.profile; ' if n > 1 else ''}export PROMPT={shlex.quote(prompt)} && cd /project && "
                f"timeout 900 {spec.command} > ~/session{n}.out 2> ~/session{n}.err < /dev/null")

    turns = "\n".join([
        "(python3 -m http.server 8787 --bind 127.0.0.1 > /dev/null 2>&1 &) && sleep 1",  # (not Relay)
        session(1, INSTALL.format(harness=harness, install=GITHUB.format(ref=ref) if ref else CHECKOUT)),
        f"{RELAY} status > ~/status.out 2>&1; {RELAY} set compact_growth=3000 compact_min_gain=0 >> ~/status.out 2>&1",
        session(2, TASK.format(tag="compaction")),
        session(3, SWITCH),
        f"{RELAY} strategy >> ~/status.out 2>&1",
        session(4, TASK.format(tag="clm")),
        f"{RELAY} strategy folding >> ~/status.out 2>&1",
        session(5, TASK.format(tag="folding")),
        f"{RELAY} status >> ~/status.out 2>&1",
    ])
    container(name, root, {}, turns, timeout=5400, install=False, serve=False, importable=False)
    home = root / "home"

    def read(file: str) -> str:
        return (home / file).read_text(errors="replace") if (home / file).exists() else ""

    trace = [json.loads(line) for line in read("trace.jsonl").splitlines() if line.strip()]
    installed = json.loads(read(".config/relay/installed.json") or "{}")
    settings = json.loads(read(".config/relay/settings.json") or "{}")
    problems = []
    if harness not in installed:
        problems.append(f"{harness} was not set up: {read('session1.out')[-300:]!r}")
    if str(settings.get("port", "8787")) == "8787":
        problems.append("Relay did not leave the port something else held")
    if "strategy: clm" not in read("status.out"):
        problems.append(f"the harness did not switch Relay to clm: {read('session3.out')[-300:]!r}")
    for n, strategy in ((2, "compaction"), (4, "clm"), (5, "folding")):
        tag = f"[{strategy}]"
        requests = [r for r in trace if "items" in r and tag in json.dumps(r.get("body"))]
        answer = read(f"session{n}.out").lower()
        if not requests:
            problems.append(f"session {n} ({strategy}): no request went through Relay")
            continue
        if strategy == "compaction":
            acted = any(r.get("compacted") for r in requests)
        else:
            acted = any(MARKS[strategy] in json.dumps(r.get("forwarded")) for r in requests)
        if not acted:
            problems.append(f"session {n}: {strategy} did not act on its requests")
        if not all(code in answer for code in CODES[:4]):
            problems.append(f"session {n} ({strategy}): answer is wrong: {answer[-150:]!r}")
    return {"name": name, "passed": not problems, "problems": problems, "home": str(home),
            "port": settings.get("port"), "said": read("session1.out").strip()[-400:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", choices=list(SPECS))
    parser.add_argument("--ref", help="install from this branch or tag on GitHub (default: the checkout at /relay)")
    options = parser.parse_args()
    results = [run(name, options.ref) for name in options.names]
    for r in results:
        print(f"== {r['name']} sets Relay up and uses it: {'PASS' if r['passed'] else 'FAIL'}  ({r['home']})")
        print(f"  port: {r['port']}; it said: {r['said']!r}")
        for problem in r["problems"]:
            print(f"  problem: {problem}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
