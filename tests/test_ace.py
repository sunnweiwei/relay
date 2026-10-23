from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from relay.strategies import ACE, Compaction
from relay.strategies.ace import INITIAL, _apply, _operations

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]
NEVER = Compaction(threshold=10**9)


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def call(i: int, output: str = "ok") -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": "{\"cmd\": \"pytest\"}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": output}]


def answer(text: str) -> tuple[int, dict[str, Any]]:
    return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant",
                                                   "content": [{"type": "output_text", "text": text}]}]}


REFLECTION = json.dumps({"reasoning": "r", "error_identification": "e", "root_cause_analysis": "c",
                         "correct_approach": "a", "key_insight": "Run the tests before replying."})
CURATION = json.dumps({"reasoning": "add it", "operations": [
    {"type": "ADD", "section": "verification_checklist", "content": "Run the tests before replying."},
    {"type": "ADD", "section": "made_up_section", "content": "dropped"}]})


class Upstream:
    def __init__(self, *answers: tuple[int, Any]) -> None:
        self.answers, self.requests = list(answers), []

    def __call__(self, request: dict[str, Any]) -> tuple[int, Any]:
        self.requests.append(request)
        return self.answers.pop(0) if self.answers else answer("unused")


class AceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.playbook = Path(tempfile.mkdtemp()) / "playbook.md"
        self.engine = Engine(ACE(str(self.playbook), NEVER))

    def send(self, *items: dict[str, Any], upstream: Upstream | None = None):
        return self.engine.prepare(CODEC, HARNESS, {"model": "m", "input": list(items), "tools": TOOLS}, tenant="t",
                                   post=upstream or Upstream())

    def test_the_playbook_is_in_every_request_in_ace_s_words(self) -> None:
        sent = self.send(msg("user", "fix the bug"), *call(1)).body["input"]
        placed = sent[0]["content"][0]["text"]
        self.assertIn("### PLAYBOOK BEGIN\n## STRATEGIES AND HARD RULES", placed)
        self.assertTrue(placed.startswith("You are also provided with a curated playbook"))
        self.assertFalse(self.playbook.exists())  # (nothing learned yet)

    def test_a_task_the_agent_ended_is_reflected_on_and_curated_once(self) -> None:
        task = [msg("user", "fix the bug"), *call(1, "1 failed"), *call(2, "3 passed"), msg("assistant", "Fixed.")]
        mid = Upstream()
        self.send(*task[:-1], upstream=mid)
        self.assertEqual(mid.requests, [])  # mid-task: the agent has not ended it
        upstream = Upstream(answer(REFLECTION), answer(CURATION))
        sent = self.send(*task, msg("user", "now the docs"), upstream=upstream).body["input"]
        reflector, curator = (r["input"][0]["content"][0]["text"] for r in upstream.requests)
        self.assertTrue(reflector.startswith("You are an expert coding agent and educator."))
        self.assertIn("=== FULL CONVERSATION HISTORY ===\n[0] USER: fix the bug", reflector)
        self.assertIn("now the docs", sent[-1]["content"][0]["text"] if isinstance(sent[-1]["content"], list)
                      else sent[-1]["content"])
        self.assertNotIn("now the docs", reflector)  # (the next task is not part of it)
        self.assertIn("Run the tests before replying.", curator)  # the reflection, as its guidebook
        self.assertIn("## VERIFICATION CHECKLIST\n\n[vc-00001] Run the tests before replying.\n## TROUBLESHOOTING",
                      self.playbook.read_text())  # (laid out as ACE's own playbooks are)
        self.assertNotIn("dropped", self.playbook.read_text())  # (not one of ACE's sections)
        self.assertIn("[vc-00001] Run the tests", sent[0]["content"][0]["text"])  # learned: in this request already
        again = Upstream()
        self.send(*task, msg("user", "now the docs"), *call(3), upstream=again)
        self.assertEqual(again.requests, [])  # reflected on once

    def test_a_failed_reflection_leaves_the_playbook_and_the_request(self) -> None:
        task = [msg("user", "fix the bug"), *call(1), msg("assistant", "Fixed."), msg("user", "next")]
        with self.assertLogs("relay", "WARNING"):
            sent = self.send(*task, upstream=Upstream(answer(REFLECTION), answer("not json")))
        self.assertFalse(self.playbook.exists())
        self.assertIn("### PLAYBOOK BEGIN", sent.body["input"][0]["content"][0]["text"])

    def test_the_window_is_compaction_s_and_the_playbook_stays(self) -> None:
        self.engine = Engine(ACE(str(self.playbook), Compaction(threshold=300, min_gain=0)))
        sent = self.send(msg("user", "fix the bug"), *call(1, "x" * 2_000), *call(2, "x" * 2_000),
                         upstream=Upstream(answer("SUMMARY")))
        self.assertTrue(sent.compacted)
        self.assertIn("### PLAYBOOK BEGIN", sent.body["input"][0]["content"][0]["text"])


class PlaybookTests(unittest.TestCase):
    """ACE's own rules (playbook.py, adaptation_react.py)."""

    def test_additions_take_fresh_ids_at_the_end_of_their_section(self) -> None:
        playbook = INITIAL.replace("## VERIFICATION CHECKLIST", "## VERIFICATION CHECKLIST\n[vc-00007] Read the diff.")
        added = _apply(playbook, [{"type": "ADD", "section": "Verification Checklist", "content": "Run the tests."},
                                  {"type": "ADD", "section": "others", "content": "Ask when unsure."}], 8)
        self.assertIn("## VERIFICATION CHECKLIST\n[vc-00007] Read the diff.\n\n[vc-00008] Run the tests.\n## TROUBLESHOOTING", added)
        self.assertIn("## OTHERS\n[misc-00009] Ask when unsure.", added)

    def test_only_add_operations_and_ace_s_sections(self) -> None:
        with self.assertRaises(ValueError):  # (one that is not an ADD: none of them)
            _operations(json.dumps({"reasoning": "r", "operations": [{"type": "UPDATE", "section": "others",
                                                                      "content": "x"}]}))
        found = _operations("Here:\n```json\n" + CURATION + "\n```")
        self.assertEqual([op["section"] for op in found], ["verification_checklist"])


if __name__ == "__main__":
    unittest.main()
