from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from relay.strategies import Compaction, ProLong

CODEC, HARNESS = OpenAIResponses(), Harness()
SUMMARY = (200, {"status": "completed", "output": [{"type": "message", "role": "assistant",
                                                     "content": [{"type": "output_text", "text": "codes c1, c2"}]}]})


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def run(i: int, command: str, output: str) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": output}]


class ProLongTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log = Path(tempfile.mkdtemp()) / ".prolong" / "log.jsonl"
        self.engine = Engine(ProLong(Compaction(threshold=10**9), str(self.log)))
        self.history = [msg("developer", "rules"), msg("user", "read the files"), *run(1, "cat file_1.txt", "color teal")]

    def send(self, *items: dict[str, Any], engine: Engine | None = None) -> Any:
        return (engine or self.engine).prepare(CODEC, HARNESS, {"model": "m", "input": list(items), "tools": [{}]},
                                               tenant="t", post=lambda r: SUMMARY)

    def entries(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_events_go_to_the_log_as_prolongs_codex_hooks_write_them_and_its_texts_to_the_model(self) -> None:
        history = [*self.history[:2], msg("assistant", "reading file 1"), *self.history[2:]]
        sent = self.send(*history).body["input"]
        text = sent[1]["content"][0]["text"]
        self.assertTrue(text.startswith(f"<!-- prolong:start -->\n## PRO-LONG memory\n\nPRO-LONG maintains a local, "
                                        f"append-only coding-session log at `{self.log}`."))
        self.assertIn(f"---\nname: prolong\n", text)  # the skill, whole
        self.assertIn(f"jq -c 'select(.type == \"tool_result\")' {self.log} | tail -n 20", text)
        self.assertEqual(sent[2:], history[1:])  # the log never enters the prompt
        self.send(*history, msg("assistant", "file 1 is teal"), msg("user", "and file 2?"))  # (the turn before ended)
        entries = self.entries()
        self.assertEqual([e["type"] for e in entries],
                         ["session_start", "user_prompt", "tool_result", "assistant_message", "user_prompt"])
        self.assertTrue(all(list(e) == ["timestamp", "client", "sessionId", "type", "content"] and e["client"] == "relay"
                            and e["timestamp"].endswith("Z") for e in entries))
        session = entries[0]["sessionId"]
        self.assertEqual([e["content"] for e in entries], [
            {"session_id": session, "hook_event_name": "SessionStart"},
            {"session_id": session, "hook_event_name": "UserPromptSubmit", "prompt": "read the files"},
            {"session_id": session, "hook_event_name": "PostToolUse", "tool_name": "sh", "tool_input": {"cmd": "cat file_1.txt"},
             "tool_response": "color teal"},
            {"session_id": session, "hook_event_name": "Stop", "last_assistant_message": "file 1 is teal"},  # (the turn's last)
            {"session_id": session, "hook_event_name": "UserPromptSubmit", "prompt": "and file 2?"}])
        self.assertIn('"type":"tool_result"', self.log.read_text())  # (written as PRO-LONG's runtime writes JSON)
        self.assertTrue((self.log.parent / ".gitignore").exists())

    def test_a_log_that_cannot_be_written_leaves_the_compaction_on_and_its_events_come_later_once(self) -> None:
        self.engine = Engine(ProLong(Compaction(threshold=10), str(self.log)))  # (compacts on every request)
        self.log.parent.mkdir(parents=True)
        self.log.write_text("")
        self.log.chmod(0o400)
        try:
            with self.assertLogs("relay", "WARNING"):
                self.assertTrue(self.send(*self.history).compacted)  # (the inner compaction went on)
        finally:
            self.log.chmod(0o600)
        self.assertEqual(self.entries(), [])
        self.send(*self.history, *run(2, "cat file_2.txt", "color red"))
        self.assertEqual([e["type"] for e in self.entries()], ["session_start", "user_prompt", "tool_result", "tool_result"])

    def test_events_are_logged_once_though_the_request_failed_after_logging_them(self) -> None:
        class Failing(Compaction):
            def plan(self, request: Any, summarizer: Any) -> Any:
                raise RuntimeError("upstream down")

        engine = Engine(ProLong(Failing(threshold=10), str(self.log)), retry_after=0)
        with self.assertLogs("relay", "WARNING"):  # (unbounded: the last context forwarded, the state unsaved)
            self.send(*self.history, engine=engine)
            self.send(*self.history, *run(2, "cat file_2.txt", "color red"), engine=engine)
        self.assertEqual([e["type"] for e in self.entries()], ["session_start", "user_prompt", "tool_result", "tool_result"])

    def test_an_events_images_are_saved_beside_the_log_and_named_in_it(self) -> None:
        screen = {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(b"png bytes").decode()}
        shot = run(2, "screenshot", "")
        shot[1]["output"] = [{"type": "input_text", "text": "the page"}, screen]
        self.send(*self.history, *shot)
        entry = self.entries()[-1]
        (saved,) = entry["content"]["images"]
        self.assertEqual((entry["content"]["tool_response"], Path(saved).read_bytes()), ("the page", b"png bytes"))
        self.assertEqual(Path(saved).parent, self.log.parent / "images")

    def test_reading_the_log_is_not_recorded_whatever_runs_beside_it(self) -> None:
        reading = run(2, f"rg -n teal {self.log}", '{"type": "tool_result", "content": {"text": "color teal"}}')
        other = run(3, "cat file_2.txt", "color amber")
        self.send(*self.history, reading[0], other[0], reading[1], other[1], msg("assistant", "teal"), msg("user", "thanks"))
        entries = self.entries()
        self.assertEqual([e["type"] for e in entries],
                         ["session_start", "user_prompt", "tool_result", "tool_result", "assistant_message", "user_prompt"])
        self.assertEqual([e["content"].get("tool_input") for e in entries[2:4]],
                         [{"cmd": "cat file_1.txt"}, {"cmd": "cat file_2.txt"}])
        self.assertEqual(entries[3]["content"]["tool_response"], "color amber")  # each result with its own call

    def test_the_inner_strategy_decides_what_the_model_sees_and_the_log_keeps_it_all(self) -> None:
        engine = Engine(ProLong(Compaction(threshold=10, min_gain=0), str(self.log)))
        sent = self.send(*self.history, *run(2, "cat file_2.txt", "color amber"), engine=engine)
        self.assertTrue(sent.compacted)
        self.assertNotIn("color teal", json.dumps(sent.body["input"]))  # summarized away
        self.assertIn("color teal", self.log.read_text())  # and still in the log


if __name__ == "__main__":
    unittest.main()
