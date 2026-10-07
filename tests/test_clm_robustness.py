"""CLM under the conditions PR #6 checked for compaction: other tokenizers, opaque content, a
Relay restart, an upstream overflow, branches of one session, later user messages, and history
the harness rewrites. Each test states the behavior CLM should have; the ones marked
`expectedFailure` reproduce a known gap (see the verification notes), so fixing one shows up
as an unexpected success."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.protocols import AnthropicMessages, OpenAIResponses
from relay.strategies import ContextLanguageModel

RESPONSES, ANTHROPIC, HARNESS = OpenAIResponses(), AnthropicMessages(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]
ANTHROPIC_TOOLS = [{"name": "sh", "input_schema": {"type": "object"}}]
RECEIPT = re.compile(r"\[context file:[^\]]*\]")


def unreachable(request: dict[str, Any]) -> tuple[int, Any]:
    raise AssertionError("CLM makes no requests of its own")


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def call(i: int, size: int = 400) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": f"cat f{i}"})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": f"output {i} " + "x" * size}]


def body(*items: dict[str, Any], model: str = "gpt-x") -> dict[str, Any]:
    return {"model": model, "input": list(items), "tools": TOOLS}


def anthropic_call(i: int, size: int = 400) -> list[dict[str, Any]]:
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "sh", "input": {"cmd": f"cat f{i}"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": f"output {i} " + "y" * size}]}]


def anthropic_body(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"model": "claude-sonnet-x", "system": "rules", "messages": list(messages), "tools": ANTHROPIC_TOOLS}


def receipt(exchange) -> str | None:
    found = RECEIPT.search(json.dumps(exchange.body, ensure_ascii=False))
    return found.group(0) if found else None


class ClmRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())

    def engine(self, **options: Any) -> Engine:
        window = options.pop("window", None)
        options.setdefault("budget", 200_000)
        return Engine(ContextLanguageModel(directory=str(self.directory), **options), window=window)

    def send(self, engine: Engine, *items: dict[str, Any], **options: Any):
        return engine.prepare(RESPONSES, HARNESS, body(*items), tenant="t", post=unreachable, **options)

    def mirrors(self) -> list[Path]:
        return sorted(self.directory.glob("*.md"))

    def edit(self, change, path: Path | None = None) -> None:
        path = path or self.mirrors()[0]
        path.write_text(change(path.read_text()))

    # Found replaying Claude Code under CLM: its request ends with a system message, and the
    # notes must not follow it (a system message may not precede a user turn).
    def test_notes_never_follow_a_trailing_system_message(self) -> None:
        engine = self.engine()
        environment = {"role": "system", "content": [{"type": "text", "text": "# Environment"}]}
        history = [{"role": "user", "content": "read the files"}, *anthropic_call(1), environment]
        sent = engine.prepare(ANTHROPIC, HARNESS, anthropic_body(*history), tenant="t", post=unreachable).body["messages"]
        self.assertEqual(sent[-1], environment)
        self.assertIn("[context: ~", json.dumps(sent[-2]))
        self.assertEqual(sent[-2]["role"], "user")
        self.assertEqual(sent[-2]["content"][0], history[2]["content"][0])  # joined to the tool results, still first

    # P1: the receipt's arithmetic follows the model's tokenizer, as the engine's does.
    @unittest.expectedFailure
    def test_p1_receipt_counts_tokens_like_the_engine_on_claude(self) -> None:
        engine = self.engine()
        history = [{"role": "user", "content": "read the files"}, *anthropic_call(1, 40_000), *anthropic_call(2, 4_000)]
        first = engine.prepare(ANTHROPIC, HARNESS, anthropic_body(*history), tenant="t", post=unreachable)
        self.edit(lambda s: re.sub(r"output 1 y+", "output 1: nothing useful", s))
        later = [*history, *anthropic_call(3, 10)]
        second = engine.prepare(ANTHROPIC, HARNESS, anthropic_body(*later), tenant="t", post=unreachable)
        before, after = map(int, re.search(r"~(\d+)->(\d+)", receipt(second)).groups())
        engine_freed = first.estimate - second.estimate
        self.assertGreater(before - after, 0.85 * engine_freed, "the receipt undercounts what the edit freed")

    # P2: removing encrypted reasoning shrinks the context, and is told so.
    def test_p2_removing_encrypted_reasoning_is_not_reported_as_growth(self) -> None:
        engine = self.engine()
        reasoning = {"type": "reasoning", "summary": [], "encrypted_content": "e" * 30_000}
        history = [msg("user", "task"), reasoning, *call(1, 200), *call(2, 200)]
        self.send(engine, *history)
        self.edit(lambda s: re.sub(r"\[\[CTX_TURN [^\]]* role=reasoning [^\]]*\]\]\n[^\[]*", "", s))
        sent = self.send(engine, *history, *call(3, 10))
        self.assertNotIn("e" * 100, json.dumps(sent.body))  # the edit took effect
        self.assertNotIn("GREW", receipt(sent))

    # P2, at the limit: the fit gate must not refuse an edit that shrinks the context.
    def test_p2_the_fit_gate_accepts_a_shrinking_edit_over_the_limit(self) -> None:
        engine = self.engine(budget=3_000, reserve=200)
        reasoning = {"type": "reasoning", "summary": [], "encrypted_content": "e" * 30_000}
        history = [msg("user", "task"), reasoning, *call(1, 200), *call(2, 200)]
        self.send(engine, *history)
        self.edit(lambda s: re.sub(r"\[\[CTX_TURN [^\]]* role=reasoning [^\]]*\]\]\n[^\[]*", "", s))
        sent = self.send(engine, *history, *call(3, 10))
        self.assertNotIn("REJECTED", receipt(sent))

    # P3: after a restart the model either keeps its edits or is told they are gone.
    def test_p3_a_restart_keeps_the_edits_or_says_so(self) -> None:
        engine = self.engine()
        history = [msg("user", "task"), *call(1, 4_000), *call(2, 4_000)]
        self.send(engine, *history)
        self.edit(lambda s: re.sub(r"output [12] x+", "done", s))
        history += call(3, 10)
        self.assertNotIn("x" * 1_000, json.dumps(self.send(engine, *history).body))
        restarted = self.engine()
        sent = self.send(restarted, *history, *call(4, 10))
        kept = "x" * 1_000 not in json.dumps(sent.body)
        told = receipt(sent) is not None and "NOT applied" in receipt(sent)
        self.assertTrue(kept or told, "after a restart the edits are silently gone")

    # P4: a request the upstream rejected as too long is retried smaller.
    @unittest.expectedFailure
    def test_p4_a_forced_retry_fits_the_window(self) -> None:
        engine = self.engine(budget=None, window=8_000)
        history = [msg("user", "task"), *[item for i in range(1, 30) for item in call(i, 2_000)]]
        forced = self.send(engine, *history, force=True)
        self.assertLessEqual(forced.estimate, 8_000)

    # P5: one conversation, one mirror file.
    def test_p5_the_mirror_path_is_stable_from_the_first_request(self) -> None:
        engine, history, paths = self.engine(), [msg("user", "task")], []
        for i in range(1, 4):
            sent = json.dumps(self.send(engine, *history).body)
            paths.append(re.search(r"mirrored to `([^`]+)`", sent).group(1))
            history += call(i, 100)
        self.assertEqual(len(set(paths)), 1, paths)

    # P6: two branches of one session do not overwrite each other's file.
    def test_p6_branches_keep_their_own_edits(self) -> None:
        engine = self.engine()
        base = [msg("user", "task"), *call(1, 1_000), *call(2, 1_000)]
        a, b = [*base, *call(3, 1_000)], [*base, *call(9, 1_000)]
        self.send(engine, *a)
        path_a = self.mirrors()[0]
        self.edit(lambda s: re.sub(r"output 3 x+", "output 3: SHORT", s), path_a)
        self.send(engine, *b)  # the other branch, before a's next request
        sent = self.send(engine, *a, *call(4, 10))
        self.assertIn("output 3: SHORT", json.dumps(sent.body))

    # P7: the user's later instructions survive the model's edits.
    @unittest.expectedFailure
    def test_p7_a_later_user_message_survives_a_plain_text_edit(self) -> None:
        engine = self.engine()
        history = [msg("user", "task"), *call(1, 500), msg("user", "also never touch prod.db"), *call(2, 500)]
        self.send(engine, *history)
        self.mirrors()[0].write_text("codes so far: none")
        sent = self.send(engine, *history, *call(3, 10))
        self.assertIn("never touch prod.db", json.dumps(sent.body))

    # P8: an edit Relay can no longer place is refused out loud, never dropped silently.
    @unittest.expectedFailure
    def test_p8_an_edit_lost_to_a_history_rewrite_is_reported(self) -> None:
        engine = self.engine()
        history = [msg("user", "task"), *call(1, 3_000), *call(2, 3_000)]
        self.send(engine, *history)
        self.edit(lambda s: re.sub(r"output 2 x+", "output 2: SHORT", s))
        rewritten = [dict(item) for item in history]
        rewritten[2]["output"] = "[output masked by the harness]"
        sent = self.send(engine, *rewritten, *call(3, 10))
        applied = "output 2: SHORT" in json.dumps(sent.body)
        self.assertTrue(applied or receipt(sent) is not None, "the model's edit vanished without a receipt")


if __name__ == "__main__":
    unittest.main()
