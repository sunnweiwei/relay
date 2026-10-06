"""The CLM paper's baselines as strategies: Self-Compact, AutoCompact, ACM and MEM1."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Item, Kind, Request
from relay.harnesses import Harness
from relay.protocols import Gemini, OpenAIChat, OpenAIResponses
from relay.strategies import ACM, MEM1, AutoCompact, Compaction, SelfCompact
from relay.strategies.acm import NOTHING, QUERY_MEMORY_PROMPT, SUMMARY_INSTRUCTION
from relay.strategies.autocompact import PROMPT
from relay.prompts import SUMMARY_PREFIX
from relay.strategies.selfcompact import RUBRIC, SUMMARIZER

FIRE = "C1: Y -- done\nC2: Y -- 1. a \"x\"\nC3: Y -- new fact\nN1: N -- found x"


class Scripted:
    def __init__(self, *answers: str) -> None:
        self.answers, self.calls = list(answers), []

    def summarize(self, cut: int, prompt: str) -> str:
        self.calls.append((cut, prompt))
        return self.answers.pop(0)


def request(*items: Item, tokens: int = 1_000, window: int | None = 10_000, conversation: str = "c") -> Request:
    return Request(items, items, frozenset(range(len(items) + 1)), tokens, window, conversation=conversation)


def step(i: int, command: str = "", output: str = "", say: str = "") -> list[Item]:
    """One model response: optional text, then a shell call and its result."""

    return [*([Item(Kind.ASSISTANT, say, 100 + i)] if say else []),
            Item(Kind.TOOL_CALL, json.dumps({"cmd": command or f"cat file_{i}.txt"}), 200 + i),
            Item(Kind.TOOL_RESULT, output or f"file {i} text\nCODE: c{i}", 300 + i)]


TASK = Item(Kind.USER, "read the files", 1)


class SelfCompactTests(unittest.TestCase):
    strategy = SelfCompact()  # probes every 2 responses past 37% of the window (10k)

    def test_probes_every_second_response_past_the_gate(self) -> None:
        two, three = (TASK, *step(1), *step(2)), (TASK, *step(1), *step(2), *step(3))
        self.assertIsNone(self.strategy.plan(request(*two, tokens=3_600), Scripted()))  # below the gate
        self.assertIsNone(self.strategy.plan(request(*three, tokens=5_000), Scripted()))  # an odd response
        probe = Scripted("C1: Y\nC2: N -- dispersed\nC3: Y\nN1: N")
        self.assertIsNone(self.strategy.plan(request(*two, tokens=3_800), probe))  # the rubric says CONTINUE
        self.assertEqual(probe.calls, [(5, RUBRIC)])

    def test_a_firing_rubric_replaces_the_trajectory_with_the_summary(self) -> None:
        later = Item(Kind.USER, "and then?", 9)
        items = (TASK, *step(1), later, *step(2))
        summarizer = Scripted("**C1:** Y -- x\n**C2:** Y -- y\n**C3:** Y -- z\n**N1:** N -- w", "SUMMARY")
        context = self.strategy.plan(request(*items, tokens=4_000), summarizer)
        self.assertEqual([prompt for _, prompt in summarizer.calls], [RUBRIC, SUMMARIZER])
        self.assertEqual(context.items, (TASK, later, Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nSUMMARY")))
        # The count starts again after the summary: one response since, no probe.
        self.assertIsNone(self.strategy.plan(request(*context.items, *step(3), tokens=4_000), Scripted()))

    def test_stuck_or_unproven_does_not_fire(self) -> None:
        for answer in ("C1: Y\nC2: Y\nC3: Y\nN1: Y -- duplicates", "C1: Y\nC2: Y\nN1: N"):
            self.assertIsNone(self.strategy.plan(request(TASK, *step(1), *step(2), tokens=4_000), Scripted(answer)))

    def test_the_backstop_summarizes_without_asking(self) -> None:
        summarizer = Scripted("SUMMARY")
        context = self.strategy.plan(request(TASK, *step(1), tokens=9_600), summarizer)
        self.assertEqual(summarizer.calls, [(3, SUMMARIZER)])
        self.assertEqual(context.items, (TASK, Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nSUMMARY")))


class AutoCompactTests(unittest.TestCase):
    strategy = AutoCompact()

    def test_compact_keeps_the_task_the_summary_and_the_last_turn(self) -> None:
        compact = step(4, 'echo "[compact]"', "[compact]", say="Localization is done.")
        items = (TASK, *step(1), *step(2), *step(3), *compact)
        summarizer = Scripted("Objective: read the files. Codes c1, c2, c3.")
        context = self.strategy.plan(request(*items), summarizer)
        self.assertEqual(summarizer.calls, [(len(items), PROMPT)])
        guidance, task, *recent = context.items
        self.assertIs(guidance.kind, Kind.SYSTEM)
        self.assertEqual(task, TASK)
        # The latest command and result, then the compact call, answered by the summary.
        self.assertEqual(recent[:-1], [*step(3), *compact[:2]])
        self.assertEqual(recent[-1], Item(Kind.TOOL_RESULT, "# Auto Context Summary\n\nObjective: read the files. "
                                          "Codes c1, c2, c3.", compact[2].ref))
        # Answered, the call is not taken again.
        self.assertEqual(self.strategy.plan(request(*context.items[1:], *step(5)), Scripted()).items[1:],
                         (*context.items[1:], *step(5)))

    def test_without_a_call_the_fallback_threshold_holds(self) -> None:
        items = (TASK, *step(1))
        self.assertEqual(self.strategy.plan(request(*items), Scripted()).items[1:], items)
        context = AutoCompact(Compaction(threshold=10, min_gain=0)).plan(
            request(*items), Scripted("S"))
        self.assertIs(context.items[-1].kind, Kind.SUMMARY)


class ACMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.strategy = ACM(self.directory)

    def test_manage_context_compresses_since_the_last_call_and_saves_the_originals(self) -> None:
        manage = step(3, 'echo "[manage_context]"', "[manage_context]")
        items = (TASK, *step(1), *step(2), *manage)
        summarizer = Scripted("<think>plan</think><memory>## Knowledge state\nfile_1: c1, file_2: c2 (both read in full)</memory>")
        context = self.strategy.plan(request(*items), summarizer)
        _, prompt = summarizer.calls[0]
        self.assertEqual(summarizer.calls[0][0], 0)
        self.assertTrue(prompt.startswith("Original question: read the files"))
        self.assertIn("[tool] file 2 text\nCODE: c2", prompt)
        self.assertEqual(prompt.split("Conversation to compress:")[1][-200:], SUMMARY_INSTRUCTION[-200:])
        guidance, *kept = context.items
        self.assertIn("manage_context", guidance.text)
        self.assertEqual(kept[:2], [TASK, manage[0]])
        self.assertEqual(kept[2].text, "[summary_id: 1] ## Knowledge state\nfile_1: c1, file_2: c2 (both read in full)")
        self.assertEqual(kept[2].ref, manage[1].ref)  # still the call's own result
        saved = json.loads((Path(self.directory) / "c" / "summary_1.json").read_text())
        self.assertEqual([m["role"] for m in saved], ["tool_call", "tool", "tool_call", "tool"])
        self.assertTrue(context.notes[0].startswith("[CURRENT CONTEXT TOKEN: "))

        # The next call compresses only what came after the summary; a summary is never compressed again.
        again = step(5, 'echo "[manage_context]"', "[manage_context]")
        summarizer = Scripted("<memory>" + "file_4: c4 read in full, nothing else found so far." + "</memory>")
        later = self.strategy.plan(request(*kept, *step(4), *again), summarizer).items[1:]
        self.assertEqual(list(later[:3]), kept)
        self.assertTrue(later[-1].text.startswith("[summary_id: 2] file_4: c4"))
        self.assertNotIn("CODE: c1", summarizer.calls[0][1])

        # Nothing between two calls: ACM's error.
        twice = step(6, 'echo "[manage_context]"', "[manage_context]")
        self.assertEqual(self.strategy.plan(request(*later, *twice), Scripted()).items[-1].text, NOTHING)

    def test_query_memory_recalls_from_the_saved_originals(self) -> None:
        manage = step(3, 'echo "[manage_context]"', "[manage_context]")
        kept = self.strategy.plan(request(TASK, *step(1), *step(2), *manage), Scripted("<memory>short</memory>")).items[1:]
        query = step(4, 'echo "[query_memory] 1 :: the CODE of file_2"', "[query_memory] 1 :: the CODE of file_2")
        summarizer = Scripted("- **Relevant findings:** file_2 CODE: c2")
        recalled = self.strategy.plan(request(*kept, *query), summarizer).items[-1]
        self.assertEqual(recalled.text, "[query_memory: summary_id=1]\n\n- **Relevant findings:** file_2 CODE: c2")
        cut, prompt = summarizer.calls[0]
        self.assertEqual(cut, 0)
        self.assertTrue(prompt.startswith("Saved messages under summary_id=1:"))
        self.assertIn("CODE: c2", prompt)
        self.assertIn("extract content relevant to: the CODE of file_2", prompt)
        self.assertEqual(prompt[-120:], QUERY_MEMORY_PROMPT.format(summary_id=1, history="", query="")[-120:])
        missing = step(5, 'echo "[query_memory] 7 :: x"', "[query_memory] 7 :: x")
        self.assertEqual(self.strategy.plan(request(*kept, *missing), Scripted()).items[-1].text, "Error: summary_id 7 not found.")

    def test_through_the_engine_the_rewrite_holds_and_the_marker_ends_the_request(self) -> None:
        codec, engine = OpenAIResponses(), Engine(self.strategy)

        def call(i: int, command: str, output: str) -> list[dict[str, Any]]:
            return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
                    {"type": "function_call_output", "call_id": f"c{i}", "output": output}]

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "<memory>file_1 holds CODE c1; it was read in full, nothing else yet.</memory>"}]}]}

        history = [{"type": "message", "role": "user", "content": "read the files"}, *call(1, "cat file_1.txt", "CODE: c1"),
                   *call(2, 'echo "[manage_context]"', "[manage_context]")]
        sent = engine.prepare(codec, Harness(), {"model": "m", "input": history, "tools": [{}]}, tenant="t", post=upstream)
        items = sent.body["input"]
        self.assertTrue(sent.compacted)
        self.assertIn("## Context memory", json.dumps(items[0]))  # the guidance, as the system prompt
        self.assertEqual([i.get("call_id") for i in items[2:4]], ["c2", "c2"])
        self.assertTrue(items[3]["output"].startswith("[summary_id: 1] file_1 holds CODE c1"))
        self.assertIn("[CURRENT CONTEXT TOKEN: ", json.dumps(items[-1]))
        later = engine.prepare(codec, Harness(), {"model": "m", "input": [*history, *call(3, "ls", "file_1.txt")],
                                                   "tools": [{}]}, tenant="t", post=upstream).body["input"]
        self.assertEqual(later[:4], items[:4])


class GeminiTests(unittest.TestCase):
    """On Gemini the rewritten result is the call's own `functionResponse`, and an answered call stays answered."""

    def send(self, engine: Engine, *contents: dict[str, Any], summary: str) -> Any:
        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            return 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": summary}]}, "finishReason": "STOP"}]}

        return engine.prepare(Gemini(), Harness(), {"contents": list(contents), "tools": [{}]}, tenant="g", post=upstream)

    @staticmethod
    def shell(command: str, output: str) -> list[dict[str, Any]]:
        return [{"role": "model", "parts": [{"functionCall": {"name": "run_shell_command", "args": {"command": command}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "run_shell_command", "response": {"output": output}}}]}]

    def test_autocompact_and_acm_answer_once(self) -> None:
        task = {"role": "user", "parts": [{"text": "read the files"}]}
        reads = [*self.shell("cat file_1.txt", "CODE: c1"), *self.shell("cat file_2.txt", "CODE: c2")]
        for strategy, signal, mark in ((AutoCompact(), "[compact]", "# Auto Context Summary"),
                                       (ACM(tempfile.mkdtemp()), "[manage_context]", "[summary_id: 1]")):
            engine, history = Engine(strategy), [task, *reads, *self.shell(f'echo "{signal}"', signal)]
            summary = "<memory>file_1 holds CODE c1 and file_2 holds CODE c2, both read in full.</memory>"
            first = self.send(engine, *history, summary=summary)
            self.assertTrue(first.compacted)
            self.assertIn(mark, json.dumps(first.body["contents"][-1]))
            later = self.send(engine, *history, *self.shell("ls", "file_1.txt"), summary="never asked")
            self.assertFalse(later.compacted)
            self.assertNotIn("never asked", json.dumps(later.body))


class MEM1Tests(unittest.TestCase):
    def test_only_the_newest_internal_state_and_what_follows_stay(self) -> None:
        later = Item(Kind.USER, "and file_3?", 9)
        items = (TASK, *step(1, say="<IS>reading</IS>"), *step(2, say="<IS>file_1: c1</IS>"), *step(3), later, *step(4))
        guidance, *kept = MEM1().plan(request(*items), Scripted()).items
        self.assertIn("<IS>", guidance.text)
        self.assertEqual(kept, [TASK, *step(2, say="<IS>file_1: c1</IS>"), *step(3), later, *step(4)])

    def test_an_internal_state_written_beside_the_calls_counts(self) -> None:
        # Chat Completions: one assistant message holds the text and the call.
        codec, engine = OpenAIChat(), Engine(MEM1())

        def turn(i: int, text: str = "") -> list[dict[str, Any]]:
            call = {"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": f'{{"file": "file_{i}.txt"}}'}}
            return [{"role": "assistant", "content": text, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": f"file {i} text"}]

        task = {"role": "user", "content": "read the files"}
        messages = [task, *turn(1, "<IS>reading</IS>"), *turn(2, "<IS>file_1 read</IS>")]
        sent = engine.prepare(codec, Harness(), {"model": "m", "messages": messages, "tools": [{}]}, tenant="t",
                              post=lambda r: (500, {})).body["messages"]
        self.assertEqual(sent[1:], [task, *turn(2, "<IS>file_1 read</IS>")])  # after the guidance

    def test_no_internal_state_yet_keeps_everything(self) -> None:
        items = (TASK, *step(1), *step(2))
        self.assertEqual(MEM1().plan(request(*items), Scripted()).items[1:], items)


if __name__ == "__main__":
    unittest.main()
