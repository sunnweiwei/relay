"""The CLM paper's baselines as strategies: Self-Compact, AutoCompact, ACM and MEM1."""

from __future__ import annotations

import json
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Item, Kind, Media, Request
from relay.core.tokens import item_tokens
from relay.harnesses import Harness
from relay.protocols import Gemini, OpenAIChat, OpenAIResponses
from relay.strategies import ACM, MEM1, AutoCompact, Compaction, SelfCompact
from relay.strategies.acm import QUERY_MEMORY_PROMPT, RESERVE, STOP_PROMPT, SUMMARY_INSTRUCTION
from relay.strategies.autocompact import PROMPT
from relay.strategies.mem1 import GUIDANCE
from relay.strategies.conversation import IGNORED
from relay.strategies.selfcompact import CONTINUE, QUESTION, RESEARCH, RUBRIC, SUMMARIZER, _summary

FIRE = "C1: Y -- done\nC2: Y -- 1. a \"x\"\nC3: Y -- new fact\nN1: N -- found x"


class Scripted:
    def __init__(self, *answers: str) -> None:
        self.answers, self.calls = list(answers), []

    def summarize(self, cut: int, prompt: str) -> str:
        self.calls.append((cut, prompt))
        return self.answers.pop(0)

    def complete(self, items) -> str:
        self.calls.append(("complete", items))
        return self.answers.pop(0)


def request(*items: Item, tokens: int = 1_000, window: int | None = 10_000, conversation: str = "c",
            history: tuple[Item, ...] | None = None) -> Request:
    return Request(items if history is None else history, items, frozenset(range(len(items) + 1)), tokens, window,
                   conversation=conversation)


def step(i: int, command: str = "", output: str = "", say: str = "") -> list[Item]:
    """One model response: optional text, then a shell call and its result."""

    return [*([Item(Kind.ASSISTANT, say, 100 + i)] if say else []),
            Item(Kind.TOOL_CALL, json.dumps({"cmd": command or f"cat file_{i}.txt"}), 200 + i),
            Item(Kind.TOOL_RESULT, output or f"file {i} text\nCODE: c{i}", 300 + i)]


TASK = Item(Kind.USER, "read the files", 1)


class SelfCompactTests(unittest.TestCase):
    strategy = SelfCompact()  # probes after a tool round from the third call, 2 apart, past 37% of the window (10k)

    def test_probes_after_a_tool_round_from_the_third_call_two_apart(self) -> None:
        two, three = (TASK, *step(1), *step(2)), (TASK, *step(1), *step(2), *step(3))
        self.assertIsNone(self.strategy.plan(request(*two, tokens=5_000), Scripted()))  # too few calls
        self.assertIsNone(self.strategy.plan(request(*three, tokens=3_600), Scripted()))  # too few tokens
        self.assertIsNone(self.strategy.plan(request(*three, Item(Kind.USER, "next", 9), tokens=5_000), Scripted()))  # (no tool round)
        probe = Scripted("C1: Y\nC2: N -- dispersed\nC3: Y\nN1: N")
        kept = self.strategy.plan(request(*three, tokens=3_800), probe)  # the rubric says CONTINUE
        self.assertEqual((probe.calls, kept.items, kept.state), ([(7, RUBRIC)], three, {"probe": 3}))
        four, five = (*three, *step(4)), (*three, *step(4), *step(5))
        self.assertIsNone(self.strategy.plan(replace(request(*four, tokens=4_000), state=kept.state), Scripted()))  # 1 since
        self.assertIsNotNone(self.strategy.plan(replace(request(*five, tokens=4_000), state=kept.state), Scripted(FIRE, "S")))

    def test_the_release_parses_the_rubric(self) -> None:  # (agent.py parse_rubric, first_principles_verdict)
        three = (TASK, *step(1), *step(2), *step(3))
        for answer, fires in (("C1: Yes\nC2: yes\nC3: Y\nN1: no", True),  # (Y(es)/N(o), any case)
                              ("C1: N\nC2: Y\nC3: Y\nN1: N\nC1: Y -- now closed", True),  # (a field's last line)
                              ("**C1:** Y\n**C2:** Y\n**C3:** Y\n**N1:** N", False),  # (a line must start with it)
                              ("C1: Y\nC2: Y\nC3: Y", False)):  # (a missing answer: CONTINUE)
            context = self.strategy.plan(request(*three, tokens=4_000), Scripted(answer, "S"))
            self.assertEqual(any(_summary(item) for item in context.items), fires, answer)

    def test_a_firing_rubric_replaces_the_trajectory_and_every_user_message_stays(self) -> None:
        later = Item(Kind.USER, "and then?", 9)
        items = (TASK, *step(1), later, *step(2), *step(3), *step(4))
        summarizer = Scripted(FIRE, "<think>x</think>\n<summary>\n- Essential Information: c1, c2\n</summary>")
        context = self.strategy.plan(request(*items, tokens=4_000), summarizer)
        self.assertEqual([prompt for _, prompt in summarizer.calls],
                         [RUBRIC, SUMMARIZER + QUESTION.format(question="read the files\n\nand then?")])
        self.assertEqual(context.items, (TASK, later, Item(Kind.ASSISTANT, RESEARCH + "- Essential Information: c1, c2"),
                                         Item(Kind.USER, CONTINUE)))
        self.assertEqual(context.state, {"probe": 4, "base": 4})
        # The calls go on counting: one since the probe, no probe.
        self.assertIsNone(self.strategy.plan(replace(request(*context.items, *step(5), tokens=4_000), state=context.state),
                                             Scripted()))
        # The next summary keeps the user's messages again, its own "continue" not among them.
        new = Item(Kind.USER, "now the other files", 10)
        again = self.strategy.plan(replace(request(*context.items, new, *step(5), *step(6), tokens=4_000), state=context.state),
                                   Scripted(FIRE, "S2"))
        self.assertEqual(again.items, (TASK, later, new, Item(Kind.ASSISTANT, RESEARCH + "S2"), Item(Kind.USER, CONTINUE)))

    def test_later_user_messages_kept_as_codex_keeps_them(self) -> None:
        big = [Item(Kind.USER, f"turn {n} " + "x" * 400, 20 + n) for n in range(4)]
        items = (TASK, *step(1), big[0], *step(2), big[1], *step(3), big[2], *step(4), big[3])
        strategy = SelfCompact(user_tokens=250)  # (room for two of the later ones and part of a third)
        context = strategy.plan(request(*items, *step(5), *step(6), tokens=4_000), Scripted(FIRE, "S"))
        first, cut, *newest = context.items[:4]
        self.assertEqual((first, newest), (TASK, [big[2], big[3]]))  # (the first whole, then the newest)
        self.assertTrue(cut.text.startswith("turn 1") and len(cut.text) < len(big[1].text))  # (the oldest kept, cut)
        self.assertNotIn(big[0].text, [item.text for item in context.items])
        only = SelfCompact(user_tokens=0).plan(request(*items, *step(5), *step(6), tokens=4_000), Scripted(FIRE, "S"))
        self.assertEqual(only.items[:3], (TASK, Item(Kind.ASSISTANT, RESEARCH + "S"), Item(Kind.USER, CONTINUE)))  # (the release's)

    def test_stuck_or_unproven_does_not_fire(self) -> None:
        items = (TASK, *(i for n in range(1, 5) for i in step(n)))
        for answer in ("C1: Y\nC2: Y\nC3: Y\nN1: Y -- duplicates", "C1: Y\nC2: Y\nN1: N"):
            self.assertEqual(self.strategy.plan(request(*items, tokens=4_000), Scripted(answer)).items, items)

    def test_the_backstop_summarizes_without_asking(self) -> None:
        summarizer = Scripted("SUMMARY")
        context = self.strategy.plan(request(TASK, *step(1), tokens=9_600), summarizer)
        self.assertEqual(summarizer.calls, [(3, SUMMARIZER + QUESTION.format(question="read the files"))])
        self.assertEqual(context.items, (TASK, Item(Kind.ASSISTANT, RESEARCH + "SUMMARY"), Item(Kind.USER, CONTINUE)))

    def test_the_release_gates_by_default(self) -> None:
        self.assertEqual((SelfCompact().gate * 108_000, SelfCompact().min_round, SelfCompact().period), (40_000, 3, 2))


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

    def saved(self, number: int = 1) -> list[dict[str, Any]]:
        return json.loads((Path(self.directory) / "c" / f"summary_{number}.json").read_text())

    def test_manage_context_compresses_through_the_call_and_only_the_summary_stays(self) -> None:
        manage = [Item(Kind.REASONING, "file_1 and file_2 are read; time to compress", 403),
                  *step(3, 'echo "[manage_context]"', "[manage_context]", say="Compressing the reads.")]
        items = (TASK, *step(1), *step(2), *manage)
        summarizer = Scripted("<think>plan</think><memory>## Knowledge state\nfile_1: c1, file_2: c2 (both read in full)</memory>")
        context = self.strategy.plan(request(*items), summarizer)
        (how, (asked,)), = summarizer.calls  # one user message alone: no system prompt, no tools
        self.assertEqual((how, asked.kind), ("complete", Kind.USER))
        prompt = asked.text
        self.assertTrue(prompt.startswith("Original question: read the files\n\nConversation to compress:\n\n[tool] file 1 text"))
        self.assertIn("[tool] file 2 text\nCODE: c2\n\n[assistant] Compressing the reads.\n\nCompress the conversation", prompt)
        for hidden in ("cat file_1.txt", "manage_context]", "time to compress", "[tool_call]", "[reasoning]"):
            self.assertNotIn(hidden, prompt)  # `[role] content`: no calls, no reasoning
        self.assertEqual(prompt.split("Conversation to compress:")[1][-200:], SUMMARY_INSTRUCTION[-200:])
        guidance, task, summary = context.items  # the calling response and its result went with the range
        self.assertIn('`echo "[manage_context]"`', guidance.text)
        self.assertEqual(task, TASK)
        text = "[summary_id: 1] ## Knowledge state\nfile_1: c1, file_2: c2 (both read in full)"
        tokens = 1_000 - sum(map(item_tokens, items[1:])) + item_tokens(Item(Kind.USER, text))
        self.assertEqual(summary, Item(Kind.USER, f"{text}\n\n[CURRENT CONTEXT TOKEN: {tokens + RESERVE}]"))
        self.assertEqual(context.notes, ())
        self.assertEqual([m["role"] for m in self.saved()], ["tool_call", "tool", "tool_call", "tool", "reasoning", "assistant",
                                                             "tool_call"])

        # The next call compresses only what came after the summary; a summary is never compressed again.
        again = step(5, 'echo "[manage_context]"', "[manage_context]")
        summarizer = Scripted("<memory>" + "file_4: c4 read in full, nothing else found so far." + "</memory>")
        later = self.strategy.plan(request(*context.items[1:], *step(4), *again), summarizer).items[1:]
        self.assertEqual(later[:2], (TASK, Item(Kind.USER, text)))  # (its marker gone: the last summary has it)
        self.assertTrue(later[2].text.startswith("[summary_id: 2] file_4: c4"))
        self.assertTrue(later[2].text.endswith("]") and "[CURRENT CONTEXT TOKEN: " in later[2].text)
        self.assertNotIn("CODE: c1", summarizer.calls[0][1][0].text)
        self.assertNotIn("[CURRENT CONTEXT TOKEN", summarizer.calls[0][1][0].text)

    def test_a_summary_without_a_clean_memory_block_is_asked_for_once_more(self) -> None:
        manage = step(2, 'echo "[manage_context]"', "[manage_context]")
        retried = Scripted("Here it is: file_1 holds c1",
                           "<memory>file_1 holds CODE c1; it was read in full, nothing else.</memory>")
        summary = self.strategy.plan(request(TASK, *step(1), *manage), retried).items[2]
        self.assertEqual(len(retried.calls), 2)
        self.assertTrue(summary.text.startswith("[summary_id: 1] file_1 holds CODE c1; it was read in full"))
        twice = Scripted("<memory>short</memory>", "<memory>still short</memory>")  # (under 50 characters)
        summary = self.strategy.plan(request(TASK, *step(1), *manage, conversation="d"), twice).items[2]
        self.assertTrue(summary.text.startswith("[summary_id: 1] <memory>still short</memory>"))  # the last answer, whole

    def test_the_users_later_messages_are_compressed_too(self) -> None:
        later = Item(Kind.USER, "file_2 holds CODE c2", 9)
        manage = step(3, 'echo "[manage_context]"', "[manage_context]")
        summarizer = Scripted("<memory>The user said file_2 holds CODE c2; file_1 was read.</memory>")
        context = self.strategy.plan(request(TASK, *step(1), later, *step(2), *manage), summarizer)
        _, task, summary = context.items
        self.assertEqual(task, TASK)
        self.assertTrue(summary.text.startswith("[summary_id: 1] The user said"))
        prompt = summarizer.calls[0][1][0].text
        self.assertTrue(prompt.startswith("Original question: read the files\n"))
        self.assertIn("[user] file_2 holds CODE c2", prompt)
        self.assertEqual([m["role"] for m in self.saved()], ["tool_call", "tool", "user", "tool_call", "tool", "tool_call"])

    def test_a_ranges_images_go_to_the_summarizer_and_the_saved_originals(self) -> None:
        screen = Media("image", "image/png", "aGVsbG8=")
        shot = Item(Kind.TOOL_RESULT, "clicked", 302, media=(screen,))
        manage = step(3, 'echo "[manage_context]"', "[manage_context]")
        summarizer = Scripted("<memory>The dialog shows the save button at the bottom right of the window.</memory>")
        kept = self.strategy.plan(request(TASK, *step(1), *step(2)[:2], shot, *manage), summarizer).items[1:]
        how, (sent,) = summarizer.calls[0]
        self.assertEqual((how, sent.kind, sent.media), ("complete", Kind.USER, (screen,)))
        self.assertIn("[tool] clicked [image 1]", sent.text)
        self.assertEqual([Media(**m) for m in next(m for m in self.saved() if "media" in m)["media"]], [screen])

        query = step(4, 'echo "[query_memory] 1 :: where is the save button"', "[query_memory] 1 :: where is the save button")
        summarizer = Scripted("bottom right")
        self.strategy.plan(request(*kept, *query), summarizer)
        how, (sent,) = summarizer.calls[0]
        self.assertEqual((how, sent.media), ("complete", (screen,)))
        self.assertIn("[tool] clicked [image 1]", sent.text)

    def test_query_memory_recalls_from_the_saved_originals(self) -> None:
        manage = step(3, 'echo "[manage_context]"', "[manage_context]")
        short = Scripted("<memory>file_1 and file_2 were read; their codes are in the files.</memory>")
        kept = self.strategy.plan(request(TASK, *step(1), *step(2), *manage), short).items[1:]
        query = step(4, 'echo "[query_memory] 1 :: the CODE of file_2"', "[query_memory] 1 :: the CODE of file_2")
        summarizer = Scripted("<think>file_2</think>\n- **Relevant findings:** file_2 CODE: c2")
        recalled = self.strategy.plan(request(*kept, *query), summarizer).items[-1]
        self.assertTrue(recalled.text.startswith("[query_memory: summary_id=1]\n\n- **Relevant findings:** file_2 CODE: c2\n\n"
                                                 "[CURRENT CONTEXT TOKEN: "))
        (how, (asked,)), = summarizer.calls
        self.assertEqual(how, "complete")
        prompt = asked.text
        self.assertTrue(prompt.startswith("Saved messages under summary_id=1:\n\n[tool] file 1 text"))
        self.assertIn("CODE: c2", prompt)
        self.assertNotIn("cat file_2.txt", prompt)
        self.assertIn("extract content relevant to: the CODE of file_2", prompt)
        self.assertEqual(prompt[-120:], QUERY_MEMORY_PROMPT.format(summary_id=1, history="", query="")[-120:])
        missing = step(5, 'echo "[query_memory] 7 :: x"', "[query_memory] 7 :: x")
        self.assertTrue(self.strategy.plan(request(*kept, *missing), Scripted()).items[-1].text.startswith(
            "Error: summary_id 7 not found.\n\n[CURRENT CONTEXT TOKEN: "))

    def test_the_marker_ends_the_last_tool_result_and_moves_on(self) -> None:
        items = (TASK, *step(1), *step(2))
        context = self.strategy.plan(request(*items, tokens=5_000), Scripted())
        self.assertEqual(context.items[1:-1], items[:-1])
        marked = f"{items[-1].text}\n\n[CURRENT CONTEXT TOKEN: {5_000 + RESERVE}]"
        self.assertEqual(context.items[-1], Item(Kind.TOOL_RESULT, marked, items[-1].ref))
        self.assertIn("Your context window is 10000 tokens.", context.items[0].text)
        # A new user message: the marker stays on the last tool result, the previous one taken off.
        echoed = step(3, output="text\n\n[CURRENT CONTEXT TOKEN: 7]")  # (a harness's own text ending so stays)
        later = Item(Kind.USER, "and file_3?", 9)
        harness = (*items, *echoed, later)
        again = self.strategy.plan(request(*context.items[1:], *echoed, later, tokens=6_000, history=harness), Scripted())
        self.assertEqual(again.items[1:-2], harness[:-2])
        tokens = 6_000 - item_tokens(context.items[-1]) + item_tokens(items[-1])  # (the old marker no longer counts)
        self.assertEqual(again.items[-2].text,
                         f"text\n\n[CURRENT CONTEXT TOKEN: 7]\n\n[CURRENT CONTEXT TOKEN: {tokens + RESERVE}]")
        self.assertEqual(again.items[-1], later)
        self.assertNotIn("Your context window", self.strategy.plan(request(*items, window=None), Scripted()).items[0].text)

    def test_from_95_percent_of_the_window_the_request_ends_with_the_stop_prompt(self) -> None:
        items = (TASK, *step(1))
        stopped = self.strategy.plan(request(*items, tokens=9_500), Scripted())
        self.assertEqual((stopped.items[1:], stopped.notes), (items, (STOP_PROMPT,)))  # (no marker with it)
        self.assertEqual(self.strategy.plan(request(*items, tokens=9_499), Scripted()).notes, ())
        # A new user message (the next task) is answered without it: the model may compress first.
        later = (*items, Item(Kind.USER, "and file_3?", 9))
        self.assertEqual(self.strategy.plan(request(*later, tokens=9_600), Scripted()).notes, ())
        self.assertEqual(self.strategy.plan(request(*later, *step(2), tokens=9_600), Scripted()).notes, (STOP_PROMPT,))

    def test_through_the_engine_the_summary_holds_and_the_marker_moves(self) -> None:
        codec, engine = OpenAIResponses(), Engine(self.strategy)
        asked: list[dict[str, Any]] = []

        def call(i: int, command: str, output: str) -> list[dict[str, Any]]:
            return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
                    {"type": "function_call_output", "call_id": f"c{i}", "output": output}]

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            asked.append(body)
            return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "<memory>file_1 holds CODE c1; it was read in full, nothing else yet.</memory>"}]}]}

        history = [{"type": "message", "role": "user", "content": "read the files"}, *call(1, "cat file_1.txt", "CODE: c1"),
                   *call(2, 'echo "[manage_context]"', "[manage_context]")]
        body = {"model": "m", "instructions": "codex", "input": history, "tools": [{}]}
        sent = engine.prepare(codec, Harness(), body, tenant="t", post=upstream)
        (summary_request,) = asked  # ACM's: the prompt alone
        self.assertEqual((summary_request["instructions"], len(summary_request["input"]), "tools" in summary_request),
                         ("", 1, False))
        items = sent.body["input"]
        self.assertTrue(sent.compacted)
        self.assertIn("## Context memory", sent.body["instructions"])  # the guidance, in the system prompt
        self.assertEqual([i.get("role") for i in items], ["user", "user"])  # the question, then the summary
        self.assertRegex(json.dumps(items[1]), r"\[summary_id: 1\] file_1 holds CODE c1.*\[CURRENT CONTEXT TOKEN: \d+\]")
        later = engine.prepare(codec, Harness(), {**body, "input": [*history, *call(3, "ls", "file_1.txt")]},
                               tenant="t", post=upstream).body["input"]
        self.assertEqual(later[0], items[0])
        self.assertNotIn("CURRENT CONTEXT TOKEN", json.dumps(later[1]))
        self.assertEqual([i.get("call_id") for i in later[2:]], ["c3", "c3"])
        self.assertRegex(later[3]["output"], r"^file_1.txt\n\n\[CURRENT CONTEXT TOKEN: \d+\]$")
        self.assertEqual(len(asked), 1)


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
            self.assertEqual(later.compacted, strategy.name == "acm")  # (ACM moves its marker to the new result)
            self.assertNotIn("never asked", json.dumps(later.body))


class MEM1Tests(unittest.TestCase):
    def test_each_step_sees_the_user_the_newest_response_and_its_results(self) -> None:
        later = Item(Kind.USER, "and file_3?", 9)
        items = (TASK, *step(1, say="<IS>reading</IS>"), *step(2, say="<IS>file_1: c1</IS>"), later, *step(3))
        task, *kept = MEM1().plan(request(*items), Scripted()).items
        self.assertEqual(task, Item(Kind.USER, GUIDANCE + "read the files", 1))  # (MEM1's prompt before the question)
        self.assertEqual(kept, [later, *step(3)])  # (a response without a state too: every earlier step is cleared)

    def test_a_response_beside_its_calls_and_one_with_results_to_come(self) -> None:
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
        self.assertEqual(sent, [{"role": "user", "content": GUIDANCE + "read the files"}, *turn(2, "<IS>file_1 read</IS>")])

    def test_the_first_step_sees_the_prompt_and_the_task(self) -> None:
        self.assertEqual(MEM1().plan(request(TASK), Scripted()).items, (Item(Kind.USER, GUIDANCE + "read the files", 1),))

    def test_the_prompt_goes_in_once(self) -> None:  # (the context as last sent comes back with it)
        sent = MEM1().plan(request(TASK, *step(1)), Scripted()).items
        self.assertEqual(MEM1().plan(request(*sent, *step(2)), Scripted()).items[0].text, GUIDANCE + "read the files")


if __name__ == "__main__":
    unittest.main()


class ParallelSignalTests(unittest.TestCase):
    """A signal echoed beside other calls (Claude Code answers them in one message) is ignored, the
    results as they were, and the model told so on that request."""

    def send(self, strategy: Any, command: str) -> list[dict[str, Any]]:
        from relay.harnesses import ClaudeCode
        from relay.protocols import AnthropicMessages

        def use(i: str, command: str) -> dict[str, Any]:
            return {"type": "tool_use", "id": i, "name": "Bash", "input": {"command": command}}

        messages = [{"role": "user", "content": "work"},
                    {"role": "assistant", "content": [use("x", "cat a.py"), use("s", command)]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "A TEXT"},
                                                 {"type": "tool_result", "tool_use_id": "s", "content": command[6:-1]}]}]
        self.messages = messages
        return Engine(strategy).prepare(AnthropicMessages(), ClaudeCode(), {"model": "m", "messages": messages, "tools": [{}]},
                                        tenant="t", post=lambda r: (200, {"content": [{"type": "text", "text": "SUMMARY"}],
                                                                          "stop_reason": "end_turn"})).body["messages"]

    def test_autocompact(self) -> None:
        from relay.strategies import AutoCompact
        sent = self.send(AutoCompact(), 'echo "[compact]"')
        self.assertEqual(sent[-1]["content"][:2], self.messages[-1]["content"])
        self.assertEqual(sent[-1]["content"][2]["text"], IGNORED.format(name="compact"))

    def test_acm(self) -> None:
        from relay.strategies import ACM
        with tempfile.TemporaryDirectory() as directory:
            sent = self.send(ACM(directory=directory), 'echo "[manage_context]"')
        self.assertEqual(sent[-1]["content"][:2], self.messages[-1]["content"])
        self.assertTrue(sent[-1]["content"][2]["text"].startswith(IGNORED.format(name="manage_context")))  # (its token count after)
