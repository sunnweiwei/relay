"""`relay transfer` with the real harnesses: a session one of them recorded, through a fake
upstream, moved to the other, which resumes it and sends it on as its own conversation."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.fakes import FakeUpstream, serve

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")
PATCH = "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: notes.txt\n@@\n-CODE: amber\n+CODE: birch\n*** End Patch\nEOF"


class TransferEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        for harness in ("claude", "codex"):
            if shutil.which(harness) is None:
                self.skipTest(f"{harness} is not installed")
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.home = Path(root.name, "home")
        self.work = Path(root.name, "w" * 120, "a project folder whose path is longer than Claude Code's names")  # (> 200)
        (self.home / ".codex").mkdir(parents=True)
        self.work.mkdir(parents=True)
        (self.work / "notes.txt").write_text("CODE: amber\nline two\n")
        (self.work / "pic.png").write_bytes(PNG)
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC", "CODEX", "OPENAI", "RELAY"))},
                    "HOME": str(self.home), "CODEX_HOME": str(self.home / ".codex"), "CLAUDE_CONFIG_DIR": str(self.home / ".claude"),
                    "RELAY_HOME": str(self.home / "relay"), "NO_COLOR": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "DISABLE_AUTOUPDATER": "1", "FAKE_KEY": "k", "ANTHROPIC_API_KEY": "k"}

    def run_harness(self, harness: str, fake: FakeUpstream, *args: str) -> subprocess.CompletedProcess:
        with serve(fake.app) as url:
            (self.home / ".codex/config.toml").write_text(
                'model = "gpt-5.1-codex"\nmodel_provider = "fake"\n[model_providers.fake]\nname = "Fake"\n'
                f'base_url = "{url}/v1"\nenv_key = "FAKE_KEY"\nwire_api = "responses"\nsupports_websockets = false\n')
            if harness == "codex":
                command = ["codex", "exec", "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", *args]
            else:
                command = ["claude", "--output-format", "json", "--permission-mode", "bypassPermissions", *args]
            result = subprocess.run(command, cwd=self.work, env={**self.env, "ANTHROPIC_BASE_URL": url}, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result

    def transfer(self, source: str, session: str, target: str) -> str:
        result = subprocess.run([sys.executable, "-m", "relay.cli", "transfer", source, session, "--to", target],
                                cwd=Path(__file__).resolve().parents[1], env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Go on with it: cd ", result.stdout)
        return result.stdout.rsplit("--resume " if target == "claude" else "codex resume ", 1)[1].strip()

    def test_codex_to_claude_code(self) -> None:
        script = [{"name": "exec_command", "input": {"cmd": "cat notes.txt"}}, {"name": "exec_command", "input": {"cmd": PATCH}},
                  {"name": "exec_command", "input": {"cmd": "echo out; exit 3"}}, {"name": "view_image", "input": {"path": "pic.png"}}]
        self.run_harness("codex", FakeUpstream(tool_calls=len(script), script=script), "--cd", str(self.work), "Work on the files.")
        session = json.loads(next((self.home / ".codex/sessions").rglob("rollout-*.jsonl")).read_text().splitlines()[0])["payload"]["id"]
        resumed = self.transfer("codex", session, "claude")
        fake = FakeUpstream(tool_calls=0)
        self.run_harness("claude", fake, "-p", "What did you change?", "--resume", resumed)
        messages = fake.bodies("/v1/messages")[0]["messages"]
        blocks = [b for m in messages if isinstance(m["content"], list) for b in m["content"]]
        uses = [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"]
        self.assertEqual(uses[:3], [("Bash", {"command": "cat notes.txt"}),
                                    ("Edit", {"file_path": str(self.work / "notes.txt"), "old_string": "CODE: amber",
                                              "new_string": "CODE: birch", "replace_all": False}),
                                    ("Bash", {"command": "echo out; exit 3"})])
        self.assertEqual(uses[3], ("Read", {"file_path": str(self.work / "pic.png")}))
        results = [b for b in blocks if b["type"] == "tool_result"]
        self.assertEqual(results[0]["content"], "CODE: amber\nline two\n")
        self.assertTrue(results[1]["content"].startswith(f"The file {self.work / 'notes.txt'} has been updated successfully."))
        self.assertEqual((results[2]["content"], results[2].get("is_error")), ("Exit code 3\nout\n", True))
        self.assertEqual(results[3]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"] if isinstance(messages[0]["content"], str) else messages[0]["content"][0]["text"],
                         "Work on the files.")

    def test_claude_code_to_codex(self) -> None:
        notes = str(self.work / "notes.txt")
        script = [{"name": "Read", "input": {"file_path": notes}},
                  {"name": "Edit", "input": {"file_path": notes, "old_string": "amber", "new_string": "birch"}},
                  {"name": "Bash", "input": {"command": "echo out; exit 3", "description": "Fail"}},
                  {"name": "Read", "input": {"file_path": str(self.work / "pic.png")}}]
        self.run_harness("claude", FakeUpstream(tool_calls=len(script), script=script), "-p", "Work on the files.")
        session = next((self.home / ".claude/projects").glob("*/*.jsonl")).stem
        resumed = self.transfer("claude", session, "codex")
        fake = FakeUpstream(tool_calls=0)
        self.run_harness("codex", fake, "resume", resumed, "What did you change?")
        items: list[dict[str, Any]] = fake.bodies("/v1/responses")[0]["input"]
        made = [(i["name"], json.loads(i["arguments"])) for i in items if i["type"] == "function_call"]
        self.assertEqual(made, [("exec_command", {"cmd": "cat notes.txt"}),
                                ("exec_command", {"cmd": "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: notes.txt\n@@\n"
                                                         "-CODE: amber\n+CODE: birch\n line two\n*** End Patch\nEOF"}),
                                ("exec_command", {"cmd": "echo out; exit 3"}), ("view_image", {"path": "pic.png"})])
        outputs = [i["output"] for i in items if i["type"] == "function_call_output"]
        self.assertTrue(outputs[0].endswith("Output:\nCODE: amber\nline two\n"))
        self.assertTrue(outputs[1].endswith("Success. Updated the following files:\nM notes.txt\n"))
        self.assertIn("Process exited with code 3\n", outputs[2])
        self.assertEqual(outputs[3][0]["type"], "input_image")
        users = [i["content"][0]["text"] for i in items if i.get("role") == "user" and not i["content"][0]["text"].startswith("<")]
        self.assertEqual(users, ["Work on the files.", "What did you change?"])


if __name__ == "__main__":
    unittest.main()
