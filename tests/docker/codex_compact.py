"""Codex compacting its latest session, as `/compact` does in its TUI (`codex exec` has no commands:
it sends `/compact` to the model as a prompt). Through `codex app-server`: `thread/resume`, then
`thread/compact/start`, waiting for the compaction to finish. Prints how it ended."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/relay")
from tests.test_setup_e2e import AppServer  # noqa: E402


class Compacting(AppServer):
    def _read(self) -> None:
        self.ended: list[dict] = []
        for line in self.process.stdout:
            message = json.loads(line)
            if message.get("method") in ("thread/compacted", "error"):
                self.ended.append(message)
            if "id" in message and "method" not in message:
                with self.changed:
                    self.replies[message["id"]] = message
                    self.changed.notify_all()
            elif message.get("method") in ("turn/completed", "thread/compacted"):
                with self.changed:
                    self.finished += 1
                    self.changed.notify_all()


def main() -> int:
    sessions = Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser() / "sessions"
    latest = max(sessions.rglob("rollout-*.jsonl"), key=lambda path: path.stat().st_mtime)
    thread = re.search(r"([0-9a-f]{8}-[0-9a-f-]{27})\.jsonl$", latest.name).group(1)
    app = Compacting(dict(os.environ), Path.cwd())
    try:
        app.call("thread/resume", {"threadId": thread})
        done = app.finished
        app.call("thread/compact/start", {"threadId": thread})
        with app.changed:
            app.changed.wait_for(lambda: app.finished > done, 600)
        print(json.dumps({"thread": thread, "ended": [m.get("method") for m in app.ended],
                          "errors": [m["params"]["error"]["message"][:200] for m in app.ended if m.get("method") == "error"]}))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
