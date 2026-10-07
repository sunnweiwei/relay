"""CLM across a Relay restart (P3): the model's edited context comes back from its checkpoint,
for this conversation only."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from relay.strategies import ContextLanguageModel
from relay.strategies.clm_checkpoint import FOLDER

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]
RESTORED = "your edited context was restored"


def unreachable(request: dict[str, Any]) -> tuple[int, Any]:
    raise AssertionError("CLM makes no requests of its own")


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def call(i: int, size: int = 2_000) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": f"cat f{i}"})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": f"output {i} " + "x" * size}]


class ClmRestartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.history = [msg("user", "read the files"), *call(1), *call(2)]

    def engine(self) -> Engine:  # a fresh engine is a Relay that restarted: an empty store, a new secret
        return Engine(ContextLanguageModel(budget=200_000, directory=str(self.directory)))

    def send(self, engine: Engine, *items: dict[str, Any]):
        return engine.prepare(CODEC, HARNESS, {"model": "m", "input": list(items), "tools": TOOLS},
                              tenant="t", post=unreachable)

    def mirror(self) -> Path:
        return max(self.directory.glob("*.md"), key=lambda p: p.stat().st_mtime)

    def edit(self, change) -> None:
        self.mirror().write_text(change(self.mirror().read_text()))

    def text(self, exchange) -> str:
        return json.dumps(exchange.body)

    def test_the_edited_context_survives_a_restart(self) -> None:
        relay = self.engine()
        self.send(relay, *self.history)
        self.edit(lambda s: re.sub(r"output 1 x+", "output 1: nothing useful", s))
        self.history += call(3, 10)
        self.assertNotIn("output 1 xxxx", self.text(self.send(relay, *self.history)))
        restarted = self.engine()
        sent = self.send(restarted, *self.history, *call(4, 10))
        self.assertNotIn("output 1 xxxx", self.text(sent))
        self.assertIn("output 1: nothing useful", self.text(sent))
        self.assertIn("output 2 xxxx", self.text(sent))  # what the model kept is still there
        self.assertIn("output 4 x", self.text(sent))  # and what came since
        self.assertIn(RESTORED, self.text(sent))
        self.assertTrue(sent.compacted)  # the engine stores the restored context again
        later = self.send(restarted, *self.history, *call(4, 10), *call(5, 10))
        self.assertNotIn("output 1 xxxx", self.text(later))
        self.assertNotIn(RESTORED, self.text(later))  # said once

    def test_an_edit_made_just_before_the_restart_is_applied_after_it(self) -> None:
        relay = self.engine()
        self.send(relay, *self.history)
        self.edit(lambda s: re.sub(r"output 2 x+", "output 2: SHORT", s))
        sent = self.send(self.engine(), *self.history, *call(3, 10))  # restarted before the next request
        self.assertIn("output 2: SHORT", self.text(sent))
        self.assertNotIn("output 2 xxxx", self.text(sent))
        self.assertIn("edit applied", self.text(sent))

    def test_further_edits_after_the_restart_are_read(self) -> None:
        relay = self.engine()
        self.send(relay, *self.history)
        self.edit(lambda s: re.sub(r"output 1 x+", "one", s))
        self.history += call(3, 10)
        self.send(relay, *self.history)
        restarted = self.engine()
        self.history += call(4, 10)
        self.send(restarted, *self.history)
        self.edit(lambda s: re.sub(r"output 2 x+", "two", s))
        sent = self.send(restarted, *self.history, *call(5, 10))
        self.assertIn("edit applied", self.text(sent))
        self.assertNotIn("output 2 xxxx", self.text(sent))
        self.assertNotIn("output 1 xxxx", self.text(sent))

    def test_another_branch_does_not_get_this_branchs_edits(self) -> None:
        relay = self.engine()
        a = [*self.history, *call(3)]
        self.send(relay, *a)
        self.edit(lambda s: re.sub(r"output 3 x+", "three", s))
        self.send(relay, *a, *call(4, 10))
        b = [*self.history, *call(9)]  # left the history before call 3, after a restart
        sent = self.send(self.engine(), *b)
        self.assertNotIn("three", self.text(sent))
        self.assertIn("output 9 xxxx", self.text(sent))

    def test_a_rewritten_history_finds_no_checkpoint(self) -> None:
        relay = self.engine()
        self.send(relay, *self.history)
        self.edit(lambda s: re.sub(r"output 2 x+", "two", s))
        self.send(relay, *self.history, *call(3, 10))
        rewritten = [dict(item) for item in self.history]
        rewritten[2]["output"] = "[masked]"  # the harness changed its own history, then Relay restarted
        sent = self.send(self.engine(), *rewritten, *call(3, 10))
        self.assertNotIn(RESTORED, self.text(sent))
        self.assertIn("output 2 xxxx", self.text(sent))  # starts from the history it was sent

    def test_checkpoints_stay_out_of_the_repository_and_are_private(self) -> None:
        self.send(self.engine(), *self.history)
        folder = self.directory / FOLDER
        (checkpoint,) = folder.glob("*.json")
        self.assertEqual(checkpoint.stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.directory / ".gitignore").read_text(), "*\n")


if __name__ == "__main__":
    unittest.main()
