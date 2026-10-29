"""`relay uninstall` of a harness with a conversation that grew past the window through Relay, with
the real Codex (its app server, as the app keeps a thread open) and a scripted upstream.

The conversation grows through Relay; `relay uninstall codex` names it and leaves Codex pointed at
Relay, which goes on serving it; Codex compacts it itself (`thread/compact/start`, its TUI's
`/compact`) through Relay, which hands it back; Codex's configuration is then restored, and the
conversation goes on direct, against an upstream that rejects what does not fit its window.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeUpstream, serve
from tests.test_setup_e2e import AppServer

MODEL = "relay-leaving-test"  # (no catalog knows it: the window is Relay's `context_window` setting)
WINDOW = 30_000


class Watching(AppServer):
    """The app server, and the errors it reported."""

    def _read(self) -> None:
        self.errors: list[str] = []
        for line in self.process.stdout:
            message = json.loads(line)
            if message.get("method") == "error" and not message["params"].get("willRetry"):
                self.errors.append(message["params"]["error"]["message"][:200])
            if message.get("method", "").endswith("requestApproval"):
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": {"decision": "accept"}})
            elif "id" in message and "method" not in message:
                with self.changed:
                    self.replies[message["id"]] = message
                    self.changed.notify_all()
            elif message.get("method") == "turn/completed":
                with self.changed:
                    self.finished += 1
                    self.changed.notify_all()

    def done(self, call: str, params: dict) -> None:
        finished = self.finished
        self.call(call, params)
        with self.changed:
            assert self.changed.wait_for(lambda: self.finished > finished, 180), call


@unittest.skipUnless(shutil.which("codex"), "codex is not installed")
class LeavingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeUpstream(tool_calls=4, command="seq 1 3000")
        self.upstream = self.enterContext(serve(self.fake.app))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.home, self.work = self.root / "home", self.root / "work"
        (self.home / ".codex").mkdir(parents=True)
        self.work.mkdir()
        subprocess.run(["git", "init", "-q", str(self.work)], check=True)
        self.config = self.home / ".codex/config.toml"
        self.config.write_text(f'model = "{MODEL}"\nopenai_base_url = "{self.upstream}/v1"\n'
                               f'[projects."{self.work}"]\ntrust_level = "trusted"\n')
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CODEX", "OPENAI", "RELAY"))},
                    "HOME": str(self.home), "CODEX_HOME": str(self.home / ".codex"), "NO_COLOR": "1"}
        subprocess.run(["codex", "login", "--with-api-key"], input="sk-test", env=self.env, check=True,
                       capture_output=True, text=True)
        self.addCleanup(lambda: subprocess.run([sys.executable, "-c", "from relay import service; service.stop_process()"],
                                               env=self.env, cwd=Path(__file__).parents[1]))

    def relay(self, *args: str) -> str:
        result = subprocess.run([sys.executable, "-m", "relay.cli", *args], env=self.env, cwd=Path(__file__).parents[1],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def app(self) -> Watching:
        app = Watching(self.env, self.work)
        self.addCleanup(app.close)
        return app

    def turn(self, app: Watching, thread: str, prompt: str) -> None:
        app.done("turn/start", {"threadId": thread, "input": [{"type": "text", "text": prompt, "text_elements": []}]})
        self.assertEqual(app.errors, [], prompt)

    def test_a_long_conversation_is_handed_back_before_the_harness_goes_direct(self) -> None:
        self.relay("setup", "codex", "--set", f"context_window={WINDOW}")
        app = self.app()
        thread = app.call("thread/start", {"cwd": str(self.work), "model": MODEL, "approvalPolicy": "never",
                                           "sandbox": "workspace-write"})["thread"]["id"]
        for n in range(3):
            self.turn(app, thread, f"read the logs, part {n}")  # (four long tool outputs a turn)

        out = self.relay("uninstall", "codex")
        self.assertIn("codex is leaving Relay", out)
        self.assertIn('"read the logs, part 0"', out)  # the conversation that would fail direct
        self.assertIn("/up/codex-0", self.config.read_text())  # still through Relay
        self.assertIn("leaving Relay (uninstall not finished): codex", self.relay("status"))
        self.turn(app, thread, "while leaving")  # Relay keeps it working

        app.done("thread/compact/start", {"threadId": thread})  # Codex's own compaction, through Relay
        self.assertEqual(app.errors, [])
        status = self.relay("status")
        self.assertIn("harnesses set up: none", status)
        self.assertIn("left Relay: codex, restored by itself just now", status)
        self.assertNotIn("leaving Relay", status)
        self.assertIn(f'openai_base_url = "{self.upstream}/v1"', self.config.read_text())  # restored by itself

        app.close()
        self.fake.max_prompt_tokens = WINDOW  # direct: what does not fit is rejected
        before = len(self.fake.bodies("/v1/responses"))
        direct = self.app()
        direct.call("thread/resume", {"threadId": thread})
        self.turn(direct, thread, "direct after leaving")
        sent = [body for body in self.fake.bodies("/v1/responses")[before:] if body]  # (not its WebSocket tries)
        self.assertTrue(sent and all(len(json.dumps(body)) // 4 < WINDOW for body in sent))
        self.assertTrue(any(item.get("type") == "compaction" for item in sent[0]["input"]))  # its own summary


if __name__ == "__main__":
    unittest.main()
