"""Real harness binaries driven through Relay against the fake upstream."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay import Compaction, Engine, PrefixStore, ProxyConfig, create_app
from relay.integrations import ClaudeCodeHook
from relay.prompts import SUMMARY_PREFIX
from relay.strategies import ContextFolding
from tests.fakes import COMPACTION_MARKER, FINAL_TEXT, FakeUpstream, local_upstreams, serve


def _env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC", "CODEX", "OPENAI"))}
    return {**env, "HOME": str(home), "CODEX_HOME": str(home), "NO_COLOR": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"}


def _commands(harness: str, work: Path) -> list[list[str]]:
    if harness == "codex":
        common = ["--json", "--skip-git-repo-check"]
        return [["codex", "exec", *common, "--sandbox", "workspace-write", "--cd", str(work), "List the files."],
                ["codex", "exec", "resume", "--last", *common, "Do it again."]]
    common = ["--output-format", "json", "--permission-mode", "bypassPermissions"]
    return [["claude", "-p", "List the files.", *common], ["claude", "-p", "Do it again.", "--continue", *common]]


def _text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


class HarnessEndToEndTests(unittest.TestCase):
    """Two turns with enough tool output to force several compactions."""

    cases = {"codex": ("/v1/responses", "input", 11_500), "claude": ("/v1/messages", "messages", 16_500)}

    def run_harness(self, harness: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        path, key, threshold = self.cases[harness]
        fake = FakeUpstream(tool_calls=6, command="seq 1 300")
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            config = ProxyConfig(upstreams=local_upstreams(upstream))
            engine = Engine(Compaction(threshold=threshold), PrefixStore())
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            env = _env(home)
            with serve(create_app(engine, config)) as relay:
                if harness == "codex":
                    (home / "config.toml").write_text(
                        'model = "gpt-5.1-codex"\nmodel_provider = "relay"\n[model_providers.relay]\n'
                        f'name = "Relay"\nbase_url = "{relay}/v1"\nenv_key = "RELAY_TEST_KEY"\n'
                        'wire_api = "responses"\nsupports_websockets = false\n')
                    env["RELAY_TEST_KEY"] = "tenant"
                else:
                    env.update(ANTHROPIC_BASE_URL=relay, ANTHROPIC_API_KEY="tenant")
                for command in _commands(harness, work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)

        bodies = fake.bodies(path)
        # (the prompt, which Claude Code's system messages may follow)
        summaries = [b for b in bodies if COMPACTION_MARKER in _text([i for i in b[key] if i.get("role") != "system"][-1])]
        main = [b for b in bodies if b not in summaries]
        first = bodies.index(summaries[0])
        self.assertGreaterEqual(len(summaries), 2)
        self.assertLess(len(summaries), len(main) / 2)  # rewrites are reused, not recomputed
        for body in bodies[first + 1 :]:
            if body not in summaries:
                self.assertTrue(any(_text(i).startswith(SUMMARY_PREFIX) for i in body[key]))

    def test_codex(self) -> None:
        self.run_harness("codex")

    def test_claude_code(self) -> None:
        self.run_harness("claude")


class ClaudeCodeHookTests(unittest.TestCase):
    """Claude Code on its hook: its requests go to its own endpoint (here the fake), and Relay's
    strategy decides through the plugin, before every request."""

    def run_claude(self, strategy: Any, fake: FakeUpstream) -> list[dict[str, Any]]:
        if shutil.which("claude") is None:
            self.skipTest("claude is not installed")
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            (home / ".claude").mkdir(parents=True), work.mkdir()
            with serve(create_app(Engine(strategy, PrefixStore()), ProxyConfig(upstreams={}))) as relay:
                config: dict[str, Any] = {}
                for setting in ClaudeCodeHook().installation(relay).settings:
                    config.setdefault(setting.key[0], {})[setting.key[1]] = setting.value
                (home / ".claude" / "settings.json").write_text(json.dumps(config))
                env = {**_env(home), "ANTHROPIC_BASE_URL": upstream, "ANTHROPIC_API_KEY": "tenant"}
                self.compactions = []  # Claude Code's own record of its compactions, after each turn
                for command in _commands("claude", work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)
                    self.compactions.append(sum(line.count('"compact_boundary"') for path in
                                                (home / ".claude" / "projects").rglob("*.jsonl")
                                                for line in path.read_text().splitlines()))
        return [b for b in fake.bodies("/v1/messages") if b.get("tools")]

    def test_compaction_through_the_hook(self) -> None:
        bodies = self.run_claude(Compaction(threshold=16_500), FakeUpstream(tool_calls=6, command="seq 1 300"))
        compacted = [b for b in bodies if any(SUMMARY_PREFIX in _text(m) for m in b["messages"])]
        self.assertTrue(compacted)  # (the summary, Claude Code's own model's, in Claude Code's transcript)
        first = bodies.index(compacted[0])
        for body in bodies[first:]:  # (turn 2 too: resumed, Claude Code would bring the history back)
            self.assertIn(body, compacted)
        self.assertTrue(all(_size(b) < 16_500 * 4 + 40_000 for b in bodies))  # (its own context besides)

    def test_the_strategys_instructions_end_the_system_prompt(self) -> None:
        bodies = self.run_claude(ContextFolding(), FakeUpstream(tool_calls=2))
        self.assertTrue(bodies)
        for body in bodies:
            self.assertIn("## Branches (context folding)", body["system"][-1]["text"][-3_000:])

    def test_a_result_the_strategy_rewrites_and_a_branch_that_folds(self) -> None:
        def bash(command: str) -> dict[str, Any]:
            return {"name": "Bash", "input": {"command": command, "description": "Run"}}

        def script(steps: int, turns: list) -> Any:
            sent = json.dumps(fake.requests[-1]["body"]["messages"])
            if "Branch has finished its task" in sent:
                return FINAL_TEXT
            if "MODE: BRANCH" in sent:
                return bash('echo "[return] the numbers run from 1 to 50"') if "\\n50" in sent else bash("seq 1 50")
            return bash('echo "[branch] count numbers :: print 1 to 50"')

        fake = FakeUpstream(tool_calls=99, script=script)
        bodies = self.run_claude(ContextFolding(), fake)
        sent = [json.dumps(b["messages"]) for b in bodies]
        self.assertTrue(any("MODE: BRANCH" in s for s in sent))  # (the branch's echo, rewritten)
        last = sent[-1]
        self.assertIn("Branch has finished its task, the returned message is", last)
        self.assertNotIn("\\n49\\n50", last)  # (the branch's steps, folded away)
        self.assertEqual(self.compactions[0], 1)  # (the branch's echo was written as it came in; the fold compacted)


def _size(body: dict[str, Any]) -> int:
    return len(json.dumps(body))


class RelayRunTests(unittest.TestCase):
    """`relay run <harness>` points the harness at a private proxy."""

    def run_cli(self, harness: str, args: list[str], path: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        fake = FakeUpstream()
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            relay_home = Path(root, "relay")  # Relay's settings: its upstreams are the fake
            relay_home.mkdir()
            (relay_home / "settings.json").write_text(json.dumps({"openai_base_url": f"{upstream}/v1",
                                                                  "anthropic_base_url": upstream}))
            env = {**_env(home), "RELAY_HOME": str(relay_home), "OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
            result = subprocess.run([sys.executable, "-m", "relay.cli", "run", harness, "--", *args],
                                    cwd=work, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn(FINAL_TEXT, result.stdout)
        self.assertTrue(fake.bodies(path))

    def test_codex(self) -> None:
        self.run_cli("codex", ["exec", "--json", "--skip-git-repo-check", "-m", "gpt-5.1-codex", "Hi."],
                     "/v1/responses")

    def test_claude_code(self) -> None:
        self.run_cli("claude", ["-p", "Hi.", "--output-format", "json"], "/v1/messages")


if __name__ == "__main__":
    unittest.main()
