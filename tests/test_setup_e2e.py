"""`relay setup` in the middle of a user's Codex work, with the real binary and a scripted upstream.

Codex is set up as a user has it (the built-in provider, an API key). One conversation is
running through `codex exec` and its resumes, another is open in the app server (as the app or
Remote Control keeps it). Then, as the user (or Codex itself) would: `relay setup codex
--strategy clm`, later `relay strategy folding` and `relay uninstall codex`, each as the command
line runs it. What reached the upstream shows which strategy each request had.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from tests.fakes import FakeUpstream, serve

MODEL = "gpt-5.1-codex"
CLM, FOLDING = "## Managing your context", "## Branches (context folding)"


class AppServer:
    """`codex app-server` over stdio, as an app holds it: threads stay loaded between turns."""

    def __init__(self, env: dict[str, str], cwd: Path) -> None:
        self.process = subprocess.Popen(["codex", "app-server"], cwd=cwd, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.cwd, self.replies, self.finished, self.rid = cwd, {}, 0, 0
        self.changed = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()
        self.call("initialize", {"clientInfo": {"name": "relay-test", "version": "0"}})
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def _send(self, message: dict) -> None:
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _read(self) -> None:
        for line in self.process.stdout:
            message = json.loads(line)
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

    def call(self, method: str, params: dict) -> dict:
        self.rid += 1
        rid = self.rid
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        with self.changed:
            assert self.changed.wait_for(lambda: rid in self.replies, 60), method
        return self.replies[rid]["result"]

    def thread(self) -> str:
        return self.call("thread/start", {"cwd": str(self.cwd), "model": MODEL, "approvalPolicy": "never",
                                          "sandbox": "workspace-write"})["thread"]["id"]

    def turn(self, thread: str, prompt: str) -> None:
        done = self.finished
        self.call("turn/start", {"threadId": thread, "input": [{"type": "text", "text": prompt, "text_elements": []}]})
        with self.changed:
            assert self.changed.wait_for(lambda: self.finished > done, 120), prompt

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


@unittest.skipUnless(shutil.which("codex"), "codex is not installed")
class SetupMidSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeUpstream(tool_calls=0)  # (an answer at once: the strategy shows in each request)
        self.upstream = self.enterContext(serve(self.fake.app))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.home, self.work = self.root / "home", self.root / "work"
        (self.home / ".codex").mkdir(parents=True)
        self.work.mkdir()
        subprocess.run(["git", "init", "-q", str(self.work)], check=True)
        (self.home / ".codex/config.toml").write_text(
            f'model = "{MODEL}"\nopenai_base_url = "{self.upstream}/v1"\n'
            f'[projects."{self.work}"]\ntrust_level = "trusted"\n')
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CODEX", "OPENAI", "RELAY"))},
                    "HOME": str(self.home), "CODEX_HOME": str(self.home / ".codex"), "NO_COLOR": "1"}
        subprocess.run(["codex", "login", "--with-api-key"], input="sk-test", env=self.env, check=True,
                       capture_output=True, text=True)
        self.addCleanup(self.stop_relay)

    def stop_relay(self) -> None:  # (the test's own: there is no stopping Relay from its command line)
        subprocess.run([sys.executable, "-c", "from relay import service; service.stop_process()"], env=self.env,
                       cwd=Path(__file__).parents[1], check=True)

    def relay(self, *args: str) -> str:
        result = subprocess.run([sys.executable, "-m", "relay.cli", *args], env=self.env, cwd=Path(__file__).parents[1],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def codex_exec(self, prompt: str, thread: str | None = None) -> str:
        """A turn of `codex exec` (a new conversation, or `resume` of `thread`); the conversation."""

        command = ["codex", "exec", *(["resume", thread] if thread else []), "--json", "--skip-git-repo-check",
                   "--dangerously-bypass-approvals-and-sandbox", "-m", MODEL, prompt]
        result = subprocess.run(command, cwd=self.work, env=self.env, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout[-1500:] + result.stderr[-1500:])
        return thread or next(json.loads(line)["thread_id"] for line in result.stdout.splitlines()
                              if '"thread.started"' in line)

    def strategy(self, prompt: str) -> str:
        """The strategy the last request of a turn had, by what reached the upstream."""

        bodies = [json.dumps(b) for b in self.fake.bodies("/v1/responses") if prompt in json.dumps(b)]
        self.assertTrue(bodies, prompt)
        return "clm" if CLM in bodies[-1] else "folding" if FOLDING in bodies[-1] else "none"

    def test_setup_strategy_and_uninstall_while_conversations_go_on(self) -> None:
        app = AppServer(self.env, self.work)
        self.addCleanup(app.close)
        session = self.codex_exec("exec turn 1")
        opened = app.thread()
        app.turn(opened, "open thread turn 1")
        self.assertEqual((self.strategy("exec turn 1"), self.strategy("open thread turn 1")), ("none", "none"))

        out = self.relay("setup", "codex", "--strategy", "clm")
        self.assertIn("conversations started from now on use clm", out)
        self.assertIn("strategy clm", self.relay("status"))
        self.codex_exec("exec turn 2", session)  # a new process: it reads the new configuration
        app.turn(opened, "open thread turn 2")  # loaded before the setup: still the provider's
        new = app.thread()
        app.turn(new, "new thread turn 1")
        self.assertEqual([self.strategy(p) for p in ("exec turn 2", "open thread turn 2", "new thread turn 1")],
                         ["clm", "none", "clm"])

        self.assertIn("for every conversation from its next request", self.relay("strategy", "folding"))
        self.codex_exec("exec turn 3", session)
        app.turn(new, "new thread turn 2")  # the same thread, the running Relay's new strategy
        self.assertEqual([self.strategy(p) for p in ("exec turn 3", "new thread turn 2")], ["folding", "folding"])

        self.relay("uninstall", "codex")
        self.assertIn(f'openai_base_url = "{self.upstream}/v1"', (self.home / ".codex/config.toml").read_text())
        after = app.thread()
        app.turn(after, "after uninstall")  # new conversations go to the provider again
        app.turn(new, "new thread turn 3")  # one through Relay goes on through it
        self.assertEqual([self.strategy(p) for p in ("after uninstall", "new thread turn 3")], ["none", "folding"])
        result = subprocess.run([sys.executable, "-m", "relay.cli", "status"], env=self.env,
                                cwd=Path(__file__).parents[1], capture_output=True, text=True)
        self.assertIn("running at", result.stdout)  # (it runs on for the conversation still open on it)


if __name__ == "__main__":
    unittest.main()
