from __future__ import annotations

import base64
import json
import os
import re
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from relay.core.engine import Engine
from relay.harnesses import HARNESSES, Harness
from relay.protocols import OpenAIResponses
from relay.strategies import ContextLanguageModel
from relay.strategies import clm

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

    def test_an_edit_on_the_first_turn_holds_though_the_conversation_is_named_later(self) -> None:
        first = [msg("developer", "rules"), msg("user", "read the files"), msg("user", "batch " + "y" * 400)]
        self.send(*first)
        self.edit(lambda s: s.replace("batch " + "y" * 400, "[batch offloaded]"))
        sent = self.send(*first, *call(1, "python3 edit.py"))
        self.assertTrue(sent.compacted)
        self.assertNotIn("y" * 400, json.dumps(sent.body))
        self.assertIn("[context file: edit applied", sent.body["input"][-1]["content"][0]["text"])

    def test_a_broken_file_is_refused_and_says_why(self) -> None:
        self.send(*self.history)
        self.edit(lambda s: s.replace("revision=0", "revision=7"))
        sent = self.send(*self.history, *call(3))
        self.assertFalse(sent.compacted)
        self.assertIn("keep its first line exactly", sent.body["input"][-1]["content"][0]["text"])
        self.edit(lambda s: s.replace("id=1-", "id=x-"))
        self.assertIn("is unknown", self.send(*self.history, *call(3), *call(4)).body["input"][-1]["content"][0]["text"])

    def test_bytes_that_are_not_utf8_are_read_as_the_paper_reads_them(self) -> None:
        self.send(*self.history)
        raw = self.mirror().read_bytes()
        self.mirror().write_bytes(raw.replace(b"output 1 " + b"x" * 400, b"short \xff note"))
        sent = json.dumps(self.send(*self.history, *call(3)).body["input"])
        self.assertIn("short \\ufffd note", sent)  # (errors="replace", context_env/env.py)

    @unittest.skipIf(os.geteuid() == 0, "root reads any file")
    def test_an_unreadable_file_is_said_and_the_context_moves_to_a_new_one(self) -> None:
        self.send(*self.history)
        old = self.mirror()
        old.chmod(0)
        try:
            sent = self.send(*self.history, *call(3)).body["input"]
        finally:
            old.chmod(0o600)
        self.assertIn(f"could not read {old}", sent[-1]["content"][0]["text"])
        (fresh,) = [path for path in self.directory.glob("*.md") if path != old]
        self.assertIn(str(fresh), sent[1]["content"][0]["text"])  # (the guidance names the new file)

    def test_a_file_that_cannot_be_written_moves_to_a_new_one(self) -> None:
        self.send(*self.history)
        old, write = self.mirror(), clm._write

        def refusing(path: Path, text: str) -> None:
            if path == old:
                raise PermissionError(13, "Permission denied")
            write(path, text)

        with mock.patch.object(clm, "_write", refusing):
            note = self.send(*self.history, *call(3)).body["input"][-1]["content"][0]["text"]
        self.assertIn(f"could not write {old}", note)
        self.assertEqual(len(list(self.directory.glob("*.md"))), 2)

    def test_a_directory_every_user_may_write_to_gets_files_every_user_may_edit(self) -> None:
        self.directory.chmod(0o1777)
        self.send(*self.history)
        self.assertEqual(self.mirror().stat().st_mode & 0o777, 0o666)
        self.directory.chmod(0o700)
        self.send(*self.history, *call(3))
        self.assertEqual(self.mirror().stat().st_mode & 0o777, 0o600)

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
        receipt = self.send(*self.history, *call(3), *call(5), *call(4)).body["input"][-1]["content"][0]["text"]
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

    def test_a_turns_images_are_lines_the_model_keeps_or_deletes(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=20_000, directory=str(self.directory)))
        head = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 1920, 1080) + b"\x08\x02\x00\x00\x00"
        screen = {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(head).decode()}
        shot = [{"type": "function_call", "call_id": "c5", "name": "sh", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c5",
                 "output": [{"type": "input_text", "text": "the dialog opened"}, screen]}]
        self.send(*self.history, *shot)
        text = self.mirror().read_text()
        self.assertIn("the dialog opened\n[image 1 of this turn, ~2,765 tokens]", text)
        self.assertIn("keep a line to keep its attachment", text)
        self.edit(lambda s: s.replace("the dialog opened", "the save dialog opened"))
        sent = json.dumps(self.send(*self.history, *shot, *call(6)).body["input"])
        self.assertIn("the save dialog opened", sent)
        self.assertIn(screen["image_url"], sent)
        self.edit(lambda s: s.replace("\n[image 1 of this turn, ~2,765 tokens]", ""))
        sent = json.dumps(self.send(*self.history, *shot, *call(6), *call(7)).body["input"])
        self.assertIn("the save dialog opened", sent)
        self.assertNotIn(screen["image_url"], sent)

    def test_a_file_is_a_line_as_an_image_is(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=20_000, directory=str(self.directory)))
        report = {"type": "input_file", "file_data": "data:application/pdf;base64," + "A" * 4000, "filename": "report.pdf"}
        shot = [{"type": "function_call", "call_id": "c5", "name": "sh", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c5", "output": [{"type": "input_text", "text": "fetched"}, report]}]
        self.send(*self.history, *shot)
        self.assertIn("fetched\n[file 1 of this turn, ~188 tokens]", self.mirror().read_text())
        self.edit(lambda s: s.replace("fetched\n[file 1 of this turn, ~188 tokens]", "fetched"))
        sent = json.dumps(self.send(*self.history, *shot, *call(6)).body["input"])
        self.assertNotIn("report.pdf", sent)
        self.assertIn("fetched\\n[1 file removed]", sent)

    def test_the_tasks_attachments_are_the_models_to_drop_and_its_text_is_not(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=20_000, directory=str(self.directory)))
        head = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 750, 100) + b"\x08\x02\x00\x00\x00"
        screen = {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(head).decode()}
        task = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is this"}, screen]}
        history = [msg("developer", "rules"), task, *call(1), *call(2)]
        self.send(*history)
        text = self.mirror().read_text()
        self.assertRegex(text, r"index=0 role=user id=task\]\]\n\[image 1 of this turn, ~100 tokens\]")
        self.assertNotIn("what is this", text)
        self.edit(lambda s: s.replace("[image 1 of this turn, ~100 tokens]\n", "not what it asked\n"))
        sent = self.send(*history, *call(3))
        self.assertNotIn(screen["image_url"], json.dumps(sent.body["input"]))
        self.assertIn("what is this", json.dumps(sent.body["input"]))
        self.assertNotIn("not what it asked", json.dumps(sent.body["input"]))
        self.assertNotIn("id=task", self.mirror().read_text())

    def test_a_task_without_attachments_has_no_block(self) -> None:
        self.send(*self.history)
        self.assertNotIn("id=task", self.mirror().read_text())

    def test_a_result_of_images_alone_gives_way_to_the_overflow_guard(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=20_000, directory=str(self.directory)))
        head = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 1920, 1080) + b"\x08\x02\x00\x00\x00"
        screen = {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(head).decode()}
        shots = [item for n in range(5, 15) for item in (
            {"type": "function_call", "call_id": f"c{n}", "name": "sh", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{n}", "output": [screen]})]
        sent = json.dumps(self.send(*self.history, *shots).body["input"])
        self.assertIn("withheld over budget. Its 1 image went with it.", sent)

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
                                                  steering=policy, nudges=False, guard=False))
        history = [msg("user", "read the files")]
        for i in range(1, 45):
            history += call(i)
            sent = self.send(*history).body["input"]
            self.assertNotIn("CONTEXT BUDGET NUDGE", sent[-1]["content"][0]["text"])
        self.assertTrue(sent[0]["content"][0]["text"].endswith(policy))
        self.assertRegex(sent[-1]["content"][0]["text"], r"\[context: ~\d+/3800 tokens — OVER")

    def test_nudges_escalate_once_per_tier_then_every_request_near_the_limit(self) -> None:  # (the search runs' urgent nudge)
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory), urgent=0.9))
        history, found = [msg("user", "read the files")], []
        for i in range(1, 36):
            history += call(i)
            note = self.send(*history).body["input"][-1]["content"][0]["text"]
            found += [m.group(1) or "urgent" for m in re.finditer(r"at ~(\d+%) of your|\(URGENT\)", note)]
        self.assertEqual(found[:3], ["25%", "50%", "75%"])
        self.assertTrue(len(found) > 4 and set(found[3:]) == {"urgent"})

    def test_past_25_percent_the_nudges_carry_the_papers_compaction_hint(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory), urgent=0.9))
        history, notes = [msg("user", "read the files")], {}
        for i in range(1, 36):
            history += call(i)
            sent = self.send(*history).body["input"]
            note = sent[-1]["content"][0]["text"]
            tier = re.search(r"at ~(\d+%) of your|\(URGENT\)", note)
            if tier:
                notes.setdefault(tier.group(1) or "urgent", note)
        self.assertIn("3800 tokens", sent[0]["content"][0]["text"])  # (the enforced budget, not the window)
        self.assertNotIn("by locating stale regions", notes["25%"])
        for tier in ("50%", "75%", "urgent"):
            self.assertIn(f"Compact {self.mirror()} by locating stale regions with code", notes[tier])

    def test_by_default_no_urgent_nudge_as_in_the_papers_coding_runs(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory)))
        history, notes = [msg("user", "read the files")], []
        for i in range(1, 36):
            history += call(i)
            notes.append(self.send(*history).body["input"][-1]["content"][0]["text"])
        self.assertFalse(any("URGENT" in note for note in notes))

    def test_past_the_limit_the_oldest_new_tool_results_are_withheld_to_files(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory)))
        history = [msg("user", "read the files")]
        for i in range(1, 45):
            history += call(i)
            sent = self.send(*history)
        items, readout = sent.body["input"], sent.body["input"][-1]["content"][0]["text"]
        self.assertLessEqual(int(re.search(r"\[context: ~(\d+)/3800 tokens\]$", readout).group(1)), 3800)
        outputs = [i["output"] for i in items if i.get("type") == "function_call_output"]
        self.assertTrue(outputs[0].startswith("[context manager: overflow guard] tool result"))
        self.assertEqual(outputs[-1], call(44)[1]["output"])  # the newest stays
        file = Path(re.search(r"Full text: (\S+)", outputs[0]).group(1))
        self.assertEqual(file.read_text(), call(1)[1]["output"])
        # Withheld for good: the next request carries the same notes, before what is new.
        later = self.send(*history, *call(45)).body["input"]
        self.assertEqual([i for i in later if i.get("type") == "function_call_output"][0]["output"], outputs[0])

    def test_a_request_the_upstream_rejected_as_too_long_is_compacted_as_pi_clm_leaves_it_to_pi(self) -> None:
        history = [*self.history, *call(3), *call(4)]
        self.send(*history)  # (no request of its own: under the budget, nothing to recover from)
        summaries = []

        def upstream(request: dict[str, Any]) -> tuple[int, Any]:
            summaries.append(request)
            return 200, {"status": "completed", "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "SUMMARY"}]}]}

        forced = self.engine.prepare(CODEC, HARNESS, body(*history), tenant="t", post=upstream, force=True)
        self.assertTrue(forced.compacted)
        self.assertEqual(len(summaries), 1)
        texts = [json.dumps(item) for item in forced.body["input"]]
        self.assertTrue(any("SUMMARY" in text for text in texts))
        self.assertFalse(any("output 1 " in text for text in texts))  # the old tool results are summarized
        mirror = self.mirror().read_text()
        self.assertIn("revision=1", mirror.splitlines()[0])
        self.assertIn("SUMMARY", mirror)  # the model's file starts over from the compacted context
        later = self.send(*history, *call(5)).body["input"]  # and it holds for later requests
        self.assertTrue(any("SUMMARY" in json.dumps(item) for item in later))

    def test_overflow_guard_does_not_grow_a_short_tool_result(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory)))
        tiny = call(1)
        tiny[-1]["output"] = "done"
        sent = self.send(msg("user", "task"), msg("assistant", "x" * 16_000), *tiny)
        output = next(item["output"] for item in sent.body["input"]
                      if item.get("type") == "function_call_output")
        self.assertEqual(output, "done")
        self.assertFalse((self.directory / "withheld").exists())
        # Assistant text remains the model's responsibility; the guard must not
        # claim that withholding a tiny result could make this request fit.
        self.assertGreater(sent.estimate, 4_000)

    def test_what_an_edit_kept_is_never_withheld(self) -> None:
        self.engine = Engine(ContextLanguageModel(budget=4_000, reserve=200, directory=str(self.directory)))
        history = [msg("user", "read the files"), *call(1), *call(2)]
        self.send(*history)
        self.edit(lambda s: re.sub(r"output 2 x+", "output 2: SHORT", s))
        for i in range(3, 40):
            history += call(i)
            sent = self.send(*history)
        outputs = [i["output"] for i in sent.body["input"] if i.get("type") == "function_call_output"]
        self.assertEqual(outputs[:2], [call(1)[1]["output"], "output 2: SHORT"])
        self.assertTrue(outputs[2].startswith("[context manager: overflow guard]"))


if __name__ == "__main__":
    unittest.main()
