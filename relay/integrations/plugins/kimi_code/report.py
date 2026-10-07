"""Kimi Code's hook for Relay (relay/integrations/kimi_code.py): told the session by Kimi Code (its
hook input, on stdin), it reports to Relay, at the URL it is given, where the session's event log
is and how many lines it has. The standard library only, and quiet (Kimi Code may show what a
hook prints); it counts only the lines appended since its last report.

python3 report.py RELAY_URL < hook-input.json
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


def report(hook: dict[str, Any], home: Path, state: Path) -> dict[str, Any] | None:
    """What Relay is told: the session's event log and its length."""

    session = hook.get("session_id")
    if not isinstance(session, str) or not session:
        return None
    log = next(iter(sorted((home / "sessions").glob(f"*/{session}/agents/main/wire.jsonl"))), None)
    return {"harness": "kimi_code", "session": session, "cwd": hook.get("cwd"), "transcript": str(log) if log else None,
            "lines": lines(log, state / f"{session}.json") if log else 0}


def lines(log: Path, mark: Path) -> int:
    """The log's lines: those counted last time (`mark`), and those appended since."""

    try:
        seen, count = json.loads(mark.read_text())
    except (OSError, ValueError):
        seen, count = 0, 0
    size = log.stat().st_size
    if size < seen:  # a new log under the same name
        seen, count = 0, 0
    with log.open("rb") as stream:
        stream.seek(seen)
        count += stream.read(size - seen).count(b"\n")
    try:
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(json.dumps([size, count]))
    except OSError:
        pass
    return count


def main() -> None:
    try:
        home = Path(os.getenv("KIMI_CODE_HOME", "~/.kimi-code")).expanduser()
        state = Path(os.getenv("RELAY_HOME", "~/.config/relay")).expanduser() / "kimi_code"
        found = report(json.loads(sys.stdin.read() or "{}"), home, state)
        if found:
            request = urllib.request.Request(f"{sys.argv[1].rstrip('/')}/relay/v1/local", json.dumps(found).encode(),
                                             {"content-type": "application/json"})
            urllib.request.urlopen(request, timeout=5).close()
    except Exception:
        pass  # never the hook's failure: Kimi Code goes on, its summary without the log's pointer


if __name__ == "__main__":
    main()
