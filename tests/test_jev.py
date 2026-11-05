"""fast-jev-compaction as the `jev` strategy: Jev's state and questions as the library writes them,
its decisions applied to each protocol's requests, and the fallback to compaction."""

from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.prompts import SUMMARY_PREFIX
from relay.protocols import AnthropicMessages, OpenAIResponses
from relay.strategies import Compaction, Jev
from relay.strategies.jev import STATE_CONTEXT, _tokens

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]
SUMMARY = (200, {"status": "completed", "output": [{"type": "message", "role": "assistant",
                                                     "content": [{"type": "output_text", "text": "S"}]}]})


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def run(i: int, command: str, output: str) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": output}]


class FakeJev:
    """Jev answering every question with the probability given for it (0.9 by default)."""

    def __init__(self, **answers: float) -> None:
        self.answers, self.asked = answers, []

    def __call__(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        self.asked.append((state, questions))
        return {"model": "jev-1.13.0", "answers": {name: {"noul": self.answers.get(name, 0.9)} for name in questions}}


def jev(ask: Any, threshold: int = 1000, **options: Any) -> Jev:
    return Jev(trigger=Compaction(threshold=threshold), fallback=Compaction(threshold=threshold), ask=ask, **options)


def steps(n: int, size: int = 2000) -> list[dict[str, Any]]:
    return [item for i in range(1, n + 1) for item in run(i, f"cat f{i}.py", f"{i}" * size)]


class JevTests(unittest.TestCase):
    def send(self, strategy: Jev, items: list[dict[str, Any]], codec: Any = CODEC) -> list[dict[str, Any]]:
        body = {"model": "m", "tools": TOOLS, **({"input": items} if codec is CODEC else {"messages": items})}
        exchange = Engine(strategy).prepare(codec, HARNESS, body, tenant="t", post=lambda r: SUMMARY)
        return exchange.body["input" if codec is CODEC else "messages"]

    def test_the_state_is_the_whole_history_with_results_told_by_size_and_two_questions_a_call(self) -> None:
        ask = FakeJev()
        self.send(jev(ask), [msg("developer", "rules"), msg("user", "fix the bug"), *steps(6)])
        (state, questions), = ask.asked
        self.assertEqual((state["context"], state["goal"]), (STATE_CONTEXT, "fix the bug"))
        self.assertEqual(state["history"][:2], [
            {"i": 0, "role": "user", "text": "fix the bug"},
            {"i": 1, "role": "assistant", "text": "",
             "tool_calls": [{"id": "t1", "tool": "sh", "input": '{"cmd":"cat f1.py"}', "result": "2000 chars (omitted)"}]}])
        self.assertEqual(len(state["history"]), 7)  # (the messages of results alone say nothing more)
        # The first message and the newest six (the last three calls with their results) are pinned.
        self.assertEqual(sorted(questions), ["call_t1", "call_t2", "call_t3", "result_t1", "result_t2", "result_t3"])
        self.assertEqual(questions["call_t1"], {"type": "noul", "instructions": (
            "Tool call t1 (sh) should stay in the history: knowing this call was made, with its input, still matters "
            "for what the assistant does next")})
        self.assertEqual(questions["result_t2"]["instructions"], (
            "The full output of tool call t2 (sh, 2000 chars) should stay in the history verbatim: the assistant still "
            "needs its contents and re-running the tool would not do"))

    def test_a_call_and_its_result_are_kept_truncated_or_dropped_and_the_rest_stays(self) -> None:
        ask = FakeJev(result_t2=0.1, call_t3=0.1, result_t3=0.1)
        history = [msg("developer", "rules"), msg("user", "fix the bug"), *steps(6)]
        sent = self.send(jev(ask), history)
        self.assertEqual(sent[:4], history[:4])  # the instructions, the task, the first call kept whole
        self.assertEqual(sent[4]["call_id"], "c2")
        self.assertEqual(sent[5]["output"], "2" * 300 + "\n[fast-jev-compaction truncated 1700 chars of this tool result; "
                                            "re-run the tool if needed]")
        self.assertEqual([i["call_id"] for i in sent[6:]], ["c4", "c4", "c5", "c5", "c6", "c6"])  # c3 gone, with its result
        self.assertEqual(sent[6:], history[8:])

    def test_freeing_less_than_a_quarter_is_compaction_instead(self) -> None:
        sent = self.send(jev(FakeJev()), [msg("user", "fix the bug"), *steps(6)])
        self.assertTrue(any(SUMMARY_PREFIX in CODEC.classify(item).text for item in sent))

    def test_the_share_freed_is_of_the_conversation_not_of_the_harnesss_instructions(self) -> None:
        # (Codex's instructions and tools come as items: half of the conversation freed is a seventh of all)
        ask = FakeJev(**{f"{q}_t{i}": 0.1 for q in ("call", "result") for i in (1, 2, 3)})
        sent = self.send(jev(ask), [msg("developer", "r" * 30_000), msg("user", "fix the bug"), *steps(6)])
        self.assertFalse(any(SUMMARY_PREFIX in CODEC.classify(item).text for item in sent))
        self.assertEqual([i.get("call_id") for i in sent[2:]], ["c4", "c4", "c5", "c5", "c6", "c6"])

    def test_whatever_keeps_jev_from_answering_is_compaction_instead(self) -> None:
        def broken(state: Any, questions: Any) -> dict[str, Any]:
            raise RuntimeError("Jev request failed (503)")
        for strategy in (jev(broken), jev(None), jev(lambda s, q: {"answers": {}})):  # (failing, no key, no answer)
            sent = self.send(strategy, [msg("user", "fix the bug"), *steps(6)])
            self.assertTrue(any(SUMMARY_PREFIX in CODEC.classify(item).text for item in sent))

    def test_below_its_threshold_nothing_is_asked(self) -> None:
        ask = FakeJev(result_t1=0.0, call_t1=0.0)
        history = [msg("user", "fix the bug"), *steps(6, size=10)]
        self.assertEqual(self.send(jev(ask, threshold=100_000), history), history)
        self.assertEqual(ask.asked, [])

    def test_a_responses_calls_go_together_and_with_them_the_message_they_are_written_in(self) -> None:
        def turn(i: int, text: str) -> list[dict[str, Any]]:
            uses = [{"type": "tool_use", "id": f"{i}{x}", "name": "Read", "input": {"file_path": f"{i}{x}.py"}} for x in "ab"]
            return [{"role": "assistant", "content": [{"type": "text", "text": text}, *uses]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"{i}{x}", "content": x * 2000}
                                                 for x in "ba"]}]  # (in the order they finished)
        history = [{"role": "user", "content": "fix the bug"}, *(item for i in range(1, 8) for item in turn(i, f"Files {i}."))]
        low = {f"{q}_t{n}": 0.1 for q in ("call", "result") for n in range(3, 9)}
        ask = FakeJev(result_t1=0.1, **low)  # t2 kept: its response stays; the next three go
        sent = self.send(jev(ask), history, AnthropicMessages())
        state, questions = ask.asked[0]
        self.assertEqual(state["history"][1], {"i": 1, "role": "assistant", "text": "Files 1.", "tool_calls": [
            {"id": "t1", "tool": "Read", "input": '{"file_path":"1a.py"}', "result": "4001 chars (omitted)"},
            {"id": "t2", "tool": "Read", "input": '{"file_path":"1b.py"}', "result": "4001 chars (omitted)"}]})
        self.assertEqual(len(questions), 16)  # (the last three responses, pinned, are not asked about)
        self.assertEqual(sent, [*history[:3], *history[9:]])  # each of the three gone with its text

    def test_a_history_too_large_is_shrunk_in_the_librarys_stages(self) -> None:
        ask = FakeJev()
        long = [msg("user", "fix the bug " + "x" * 3000), *steps(4), msg("assistant", "y" * 3000), msg("user", "next"), *steps(3)]
        self.send(jev(ask, state_tokens=900), long)
        state = ask.asked[0][0]
        self.assertLessEqual(_tokens(json.dumps(state, separators=(",", ":"))), 900)
        texts = [entry["text"] for entry in state["history"]]
        self.assertTrue(any("chars omitted …]" in text for text in texts))
        self.assertEqual(state["history"][0]["text"][:16], "fix the bug xxxx")  # (the first message, pinned, abridged last)

    def test_the_questions_go_in_as_many_requests_as_fit_each_with_the_whole_state(self) -> None:
        ask = FakeJev()
        self.send(jev(ask, threshold=100, request_tokens=820), [msg("user", "fix the bug"), *steps(6, size=50)])
        self.assertEqual(len(ask.asked), 3)  # (a state of ~643 tokens and ~121 a call: one call's questions a request)
        self.assertEqual(len({json.dumps(state) for state, _ in ask.asked}), 1)

    def test_the_librarys_token_estimate(self) -> None:
        # (fast-jev-compaction's estimateTokens, run on the same strings)
        for text, tokens in (("", 0), ("hello world", 2), ("abcdefghijklmn", 3), ("12345", 3), ('{"a":1}', 6),
                             ("naïve café", 5), ("t12 Read file_path=src/a.ts → ok 480ch", 16)):
            self.assertEqual(_tokens(text), tokens, text)


if __name__ == "__main__":
    unittest.main()
