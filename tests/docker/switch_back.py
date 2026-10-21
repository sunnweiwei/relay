"""Going back to a harness on its own: can a conversation that grew through Relay go on without it?

    RELAY_TEST_KEYS=keys.env python tests/docker/switch_back.py codex:gpt [claude_code:gpt ...]
    python tests/docker/switch_back.py --report HOME [...]

Relay never shortens the harness's own history, so once the harness goes direct it sends all of
it. A user who wants to keep a conversation compacts it with the harness's own command first,
while Relay is still set up: the harness's compaction request gets Relay's compacted context (it
fits), and the harness keeps the summary as its own history. This plays that, one session per
harness: two turns through Relay compacting every few thousand tokens (the harness's history
grows far past what Relay sends), the harness's own compaction command (`check.NATIVE_COMPACT`),
then Relay set to only record, which forwards what the harness sends as going direct would, and a
last turn. The report compares the history the harness sent before its compaction with what it
sends after it (`kept`: the share left; about 1 means its own history did not shrink), and whether
the last turn still knew a code read in the first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check import CODES, NATIVE_COMPACT, NOTES, PROMPT, RULES, SPECS, compact_turn, container  # noqa: E402

GROWTH = 3_000  # Relay compacts every 3k new tokens: the harness's own history grows far past it
SECOND = f"Background notes, not needed for the task:\n{NOTES}\n\nNow read file_5.txt to file_8.txt, {RULES} Then reply done."
LAST = "Without reading any file again: what was the CODE at the end of file_2.txt? Reply with the code only."
RECORD = "cd /relay && python3 -m relay.cli set compact_growth=1000000000 >> ~/relay.log 2>&1"  # (going direct)


def run(name: str) -> Path:
    spec, harness = SPECS[name], name.split(":")[0]
    root = Path(tempfile.mkdtemp(prefix=f"relay-switch-{harness}-", dir=os.getenv("TMPDIR")))
    home, project = root / "home", root / "project"
    home.mkdir()
    project.mkdir()
    for n, code in enumerate(CODES, start=1):
        lines = [f"file {n} line {i}: the quick brown fox jumps over the lazy dog" for i in range(1, 121)]
        (project / f"file_{n}.txt").write_text("\n".join([*lines, f"CODE: {code}"]) + "\n")

    def turn(n: int, prompt: str, first: bool = False) -> str:  # (the trace's length as it starts: its requests)
        return (f"(wc -l < ~/trace.jsonl 2>/dev/null || echo 0) > ~/start{n} && export PROMPT={json.dumps(prompt)} && "
                f"cd /project && timeout 900 {spec.command if first else spec.resume} > ~/turn{n}.out 2> ~/turn{n}.err < /dev/null")

    compact = (f"(wc -l < ~/trace.jsonl) > ~/start3 && {compact_turn(harness, spec, '~/turn3.out')}"
               if harness in NATIVE_COMPACT else "")
    turns = [turn(1, PROMPT, first=True), turn(2, SECOND), compact,
             f"{RECORD} && sleep 2", turn(4, LAST)]
    env = {"compact_growth": str(GROWTH), "compact_min_gain": "0"}
    container(name, root, env, "\n".join(turns), timeout=3600)
    return home


def report(home: Path) -> dict:
    trace = [json.loads(line) for line in (home / "trace.jsonl").read_text().splitlines()]
    start = {n: int((home / f"start{n}").read_text()) for n in (1, 2, 3, 4) if (home / f"start{n}").exists()}
    size = lambda record: len(json.dumps(record["body"])) // 4 if record.get("body") and "summary_request" not in record else 0
    grown = max(map(size, trace[: start.get(3, start.get(4, len(trace)))]), default=0)  # its history before compacting
    direct = next((size(r) for r in trace[start[4]:] if size(r)), 0) if 4 in start else 0  # what it sends going direct
    last = (home / "turn4.out").read_text(errors="replace") if (home / "turn4.out").exists() else ""
    return {"history before (~tokens)": grown, "sent direct (~tokens)": direct,
            "kept": round(direct / grown, 2) if grown else None, "own compaction": 3 in start,
            "remembers file_2's code": CODES[1] in last.lower()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--report", nargs="+", type=Path)
    options = parser.parse_args()
    homes = options.report or [run(name) for name in options.names]
    for home in homes:
        print(home, json.dumps(report(home)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
