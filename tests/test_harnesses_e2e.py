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
from relay.prompts import SUMMARY_PREFIX
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
        summaries = [b for b in bodies if COMPACTION_MARKER in _text(b[key][-1])]
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


class RelayRunTests(unittest.TestCase):
    """`relay run <harness>` points the harness at a private proxy."""

    def run_cli(self, harness: str, args: list[str], path: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        fake = FakeUpstream()
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            env = {**_env(home), "RELAY_OPENAI_BASE_URL": f"{upstream}/v1",
                   "RELAY_ANTHROPIC_BASE_URL": upstream, "OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
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
