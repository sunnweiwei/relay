from __future__ import annotations

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

    def test_every_event_goes_to_the_log_once_and_the_skill_to_the_model(self) -> None:
        sent = self.send(*self.history).body["input"]
        self.assertTrue(sent[1]["content"][0]["text"].startswith("## PRO-LONG memory"))
        self.assertIn(str(self.log), sent[1]["content"][0]["text"])
        self.assertEqual(sent[2:], self.history[1:])  # the log never enters the prompt
        self.send(*self.history, msg("assistant", "file 1 is teal"))
        self.assertEqual([e["type"] for e in self.entries()], ["user_prompt", "tool_call", "tool_result", "assistant_message"])
        self.assertEqual(self.entries()[2]["content"]["text"], "color teal")
        self.assertTrue((self.log.parent / ".gitignore").exists())

    def test_reading_the_log_is_not_recorded(self) -> None:
        reading = run(2, f"rg -n teal {self.log}", '{"type": "tool_result", "content": {"text": "color teal"}}')
        self.send(*self.history, *reading, msg("assistant", "teal"))
        self.assertEqual([e["type"] for e in self.entries()], ["user_prompt", "tool_call", "tool_result", "assistant_message"])

    def test_the_inner_strategy_decides_what_the_model_sees_and_the_log_keeps_it_all(self) -> None:
        engine = Engine(ProLong(Compaction(threshold=10, min_gain=0), str(self.log)))
        sent = self.send(*self.history, *run(2, "cat file_2.txt", "color amber"), engine=engine)
        self.assertTrue(sent.compacted)
        self.assertNotIn("color teal", json.dumps(sent.body["input"]))  # summarized away
        self.assertIn("color teal", self.log.read_text())  # and still in the log


if __name__ == "__main__":
    unittest.main()
