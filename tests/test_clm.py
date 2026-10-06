from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import HARNESSES, Harness
from relay.protocols import OpenAIResponses
from relay.strategies import ContextLanguageModel

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def call(i: int, command: str = "cat file") -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": f"output {i} " + "x" * 400}]


def body(*items: dict[str, Any], tools: list | None = TOOLS) -> dict[str, Any]:
    return {"model": "m", "input": list(items), **({"tools": tools} if tools else {})}


def unreachable(request: dict[str, Any]) -> tuple[int, Any]:
    raise AssertionError("CLM makes no requests of its own")


class ClmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.engine = Engine(ContextLanguageModel(budget=4_000, directory=str(self.directory)))
        self.history = [msg("developer", "rules"), msg("user", "read the files"), *call(1), *call(2)]

    def send(self, *items: dict[str, Any], tools: list | None = TOOLS):
        return self.engine.prepare(CODEC, HARNESS, body(*items, tools=tools), tenant="t", post=unreachable)

    def mirror(self) -> Path:
        (path,) = self.directory.glob("*.md")
        return path

    def edit(self, change) -> None:
        self.mirror().write_text(change(self.mirror().read_text()))

    def test_the_conversation_is_mirrored_and_the_model_told_how_to_edit_it(self) -> None:
        sent = self.send(*self.history).body["input"]
        text = self.mirror().read_text()
        self.assertTrue(text.startswith("[[LIVE_CONTEXT version=1 revision=0 document="))
        # The task is protected: not in the file.
        self.assertEqual(re.findall(r"role=(\w+)", text), ["tool_call", "tool", "tool_call", "tool"])
        guidance = sent[1]["content"][0]["text"]  # after the harness's own developer message
        self.assertTrue(guidance.startswith("## Managing your context") and str(self.mirror()) in guidance)
        self.assertEqual([sent[0], *sent[2:-1]], self.history)  # the harness's own items, untouched
        self.assertRegex(sent[-1]["content"][0]["text"], r"\[context: ~\d+/1952 tokens\]$")

    def test_an_edit_becomes_the_next_requests_context(self) -> None:
        self.send(*self.history)
        # The model drops the first result (its call goes with it, as a note) and adds a note.
        self.edit(lambda s: re.sub(r"(\[\[CTX_TURN [^\]]* index=2 [^\]]*\]\]\n)output 1 x+", r"\1", s)
                  + "\n\n" + re.search(r"\[\[CTX_TURN document=\w+", s).group(0)
                  + " index=9 role=notes id=new-plan]]\nfile 1 holds amber")
        later = [*self.history, *call(3, "python3 edit.py")]
        sent = self.send(*later)
        self.assertTrue(sent.compacted)
        items, text = sent.body["input"], json.dumps(sent.body)
        self.assertNotIn("output 1 x", text)
        self.assertIn(later[1], items)  # untouched turns are the original items
        self.assertEqual(items[-3:-1], later[-2:])  # what came after the file was written follows it
        notes = [i["content"][0]["text"] for i in items if i.get("role") == "user" and isinstance(i["content"], list)]
        self.assertIn("[context role=tool_call]\nsh {\"cmd\": \"cat file\"}", notes)
        self.assertIn("[context role=notes]\nfile 1 holds amber", notes)
        self.assertIn("[context file: edit applied", items[-1]["content"][0]["text"])
        self.assertIn("revision=1", self.mirror().read_text())

    def test_the_rewrite_holds_for_every_later_request(self) -> None:
        self.send(*self.history)
        self.edit(lambda s: re.sub(r"\[\[CTX_TURN [^\]]* index=[12] [^\]]*\]\]\n[^\[]*", "", s))
        later = [*self.history, *call(3)]
        self.send(*later)
        sent = self.send(*later, *call(4))
        self.assertFalse(sent.compacted)
        self.assertNotIn("output 1 x", json.dumps(sent.body))
        self.assertEqual(sent.body["input"][-3:-1], call(4))

    def test_a_broken_file_is_refused_and_says_why(self) -> None:
        self.send(*self.history)
        self.edit(lambda s: s.replace("revision=0", "revision=7"))
        sent = self.send(*self.history, *call(3))
        self.assertFalse(sent.compacted)
        self.assertIn("keep its first line exactly", sent.body["input"][-1]["content"][0]["text"])
        self.edit(lambda s: s.replace("id=1-", "id=x-"))
        self.assertIn("is unknown", self.send(*self.history, *call(3), *call(4)).body["input"][-1]["content"][0]["text"])

    def test_a_header_is_read_by_its_attributes_or_refused(self) -> None:
        # Headers models wrote: `index=new`, attributes reordered, no document at all.
        self.send(*self.history)
        self.edit(lambda s: s + "\n\n[[CTX_TURN role=notes id=new-a index=new]]\nMARKER A")
        sent = self.send(*self.history, *call(3))
        self.assertTrue(sent.compacted)
        self.assertIn(msg("user", [{"type": "input_text", "text": "[context role=notes]\nMARKER A"}]), sent.body["input"])
        # A file written back without its first line still names its version in every header.
        self.edit(lambda s: s.split("\n", 1)[1].replace("MARKER A", "MARKER A2"))
        self.assertTrue(self.send(*self.history, *call(3), *call(5)).compacted)
        # A header from another version of the file is refused, not taken for text.
        self.edit(lambda s: s + "\n\n[[CTX_TURN document=0000000000000000 index=9 role=notes id=new-b]]\nMARKER B")
        receipt = self.send(*self.history, *call(3), *call(4)).body["input"][-1]["content"][0]["text"]
        self.assertIn("this header cannot be read", receipt)

    def test_wiping_every_block_keeps_the_task(self) -> None:
        self.send(*self.history)
        # The edit a model made in a real run: every block's text emptied.
        self.edit(lambda s: re.sub(r"(\[\[CTX_TURN [^\]]*\]\]\n).*?(?=\n\[\[CTX_TURN |\Z)", r"\1", s, flags=re.S))
        sent = self.send(*self.history, *call(3)).body["input"]
        self.assertEqual([i.get("role") or i["type"] for i in sent],
                         ["developer", "developer", "user", "function_call", "function_call_output", "user"])
        self.assertEqual(sent[2], self.history[1])

    def test_a_shortened_tool_result_still_answers_its_call(self) -> None:
        self.send(*self.history)
        self.edit(lambda s: re.sub(r"output 1 x+", "output 1: nothing found", s))
        sent = self.send(*self.history, *call(3)).body["input"]
        self.assertIn(self.history[2], sent)  # the call, as it was
        self.assertIn({**self.history[3], "output": "output 1: nothing found"}, sent)
        self.assertNotIn("[context role=", json.dumps(sent[:-1]))  # nothing lowered to a note

    def test_the_harnesss_context_stays_where_it_was_sent(self) -> None:
        codex = HARNESSES["codex"]
        update = msg("user", "<environment_context>\n  <cwd>/elsewhere</cwd>\n</environment_context>")
        history = [*self.history[:4], update, msg("user", "and the next file"), *call(2)]
        self.engine.prepare(CODEC, codex, body(*history), tenant="t", post=unreachable)
        self.edit(lambda s: re.sub(r"output 1 x+", "output 1: nothing found", s))
        sent = self.engine.prepare(CODEC, codex, body(*history, *call(3)), tenant="t", post=unreachable).body["input"]
        self.assertEqual(sent.index(update), sent.index(history[5]) - 1)

    def test_the_harnesss_context_stays_ahead_of_the_edited_conversation(self) -> None:
        # Codex re-renders its context after its own compaction; an edit is not one (a real run
        # had Codex's instructions and AGENTS.md moved below the model's notes).
        codex = HARNESSES["codex"]
        agents = msg("user", "# AGENTS.md instructions for /p\n\n<INSTRUCTIONS>\ncodename ALPHA\n</INSTRUCTIONS>")
        history = [msg("developer", "rules"), msg("developer", "<skills_instructions>s</skills_instructions>"),
                   agents, *self.history[1:]]
        self.engine.prepare(CODEC, codex, body(*history), tenant="t", post=unreachable)
        self.edit(lambda s: re.sub(r"(\[\[CTX_TURN [^\]]* index=2 [^\]]*\]\]\n)output 1 x+", r"\1", s))
        sent = self.engine.prepare(CODEC, codex, body(*history, *call(3)), tenant="t", post=unreachable)
        self.assertTrue(sent.compacted)
        items = sent.body["input"]
        self.assertEqual([items[0], items[1], *items[3:5]], history[:4])  # the strategy's section is items[2]

    def test_plain_text_replaces_everything_after_the_task(self) -> None:
        self.send(*self.history)
        self.mirror().write_text("codes so far: amber, birch")
        sent = self.send(*self.history, *call(3)).body["input"]
        self.assertEqual(sent[2], self.history[1])  # the task, after the harness's and the strategy's instructions
        self.assertIn({"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "[context role=notes]\ncodes so far: amber, birch"}]}, sent)
        self.assertNotIn("output 2 x", json.dumps(sent))

    def test_reasoning_stays_only_with_the_item_it_preceded(self) -> None:
        reasoning = {"type": "reasoning", "summary": [], "encrypted_content": "e" * 40}
        history = [*self.history[:2], reasoning, *call(1)]
        self.send(*history)
        self.edit(lambda s: re.sub(r"\[\[CTX_TURN [^\]]* index=[23] [^\]]*\]\]\n[^\[]*", "", s))  # the call and result
        sent = self.send(*history, *call(2)).body["input"]
        self.assertNotIn(reasoning, sent)

    def test_side_calls_are_left_alone(self) -> None:
        request = body(msg("user", "title this"), tools=None)
        self.assertIs(self.engine.prepare(CODEC, HARNESS, request, tenant="t", post=unreachable).body, request)

    def test_a_steering_policy_joins_the_guidance_and_nudges_can_be_off(self) -> None:
        policy = "Mask each tool result once you have used it."
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory),
                                                  steering=policy, nudges=False))
        history = [msg("user", "read the files")]
        for i in range(1, 45):
            history += call(i)
            sent = self.send(*history).body["input"]
            self.assertNotIn("CONTEXT BUDGET NUDGE", sent[-1]["content"][0]["text"])
        self.assertTrue(sent[0]["content"][0]["text"].endswith(policy))
        self.assertRegex(sent[-1]["content"][0]["text"], r"\[context: ~\d+/3800 tokens — OVER")

    def test_nudges_escalate_once_per_tier_then_every_request_near_the_limit(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory)))
        history, found = [msg("user", "read the files")], []
        for i in range(1, 36):
            history += call(i)
            note = self.send(*history).body["input"][-1]["content"][0]["text"]
            found += [m.group(1) or "urgent" for m in re.finditer(r"at ~(\d+%) of your|\(URGENT\)", note)]
        self.assertEqual(found[:3], ["25%", "50%", "75%"])
        self.assertTrue(len(found) > 4 and set(found[3:]) == {"urgent"})


if __name__ == "__main__":
    unittest.main()
