"""CLM on real harness traffic: every recorded session (tests/replay) replayed through the
engine with the CLM strategy, and a model's edit made to its mirror file partway through.

Two edits, each in its own replay: a tool result shortened in place, and one tool call removed
with its result. After the edit, every later request of that conversation must carry it (the
marker in, the removed text out), say it was applied, stay well-formed for its protocol, carry
the strategy's guidance and readout, and keep finding its stored context."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.core.tokens import approx_tokens
from relay.harnesses import detect
from relay.protocols import codec_for
from relay.strategies import ContextLanguageModel
from tests.replay.build import load
from tests.test_replay import FIXTURES, legal_system_messages, orphans

PATH = re.compile(r"mirrored to `([^`]+)`")
BLOCK = re.compile(r"(\[\[CTX_TURN [^\]\n]* role=(\w+) [^\]\n]*\]\]\n)(.*?)(?=\n\n\[\[CTX_TURN |\n?\Z)", re.S)
MARK = "[CLM-REPLAY-EDIT]"


def shorten(text: str) -> tuple[str, str] | None:
    """The longest tool result, shortened in place; returns the new file and a snippet now gone."""

    blocks = [m for m in BLOCK.finditer(text) if m.group(2) == "tool" and len(m.group(3)) > 200]
    if not blocks:
        return None
    longest = max(blocks, key=lambda m: len(m.group(3)))
    gone = longest.group(3)[60:140]
    return text[: longest.start(3)] + MARK + text[longest.end(3):], gone


def remove_pair(text: str) -> tuple[str, str] | None:
    """A tool call and the result right after it, both removed (the model's own turns)."""

    blocks = list(BLOCK.finditer(text))
    for call, result in zip(blocks, blocks[1:]):
        if call.group(2) == "tool_call" and result.group(2) == "tool" and len(result.group(3)) > 200:
            gone = result.group(3)[60:140]
            edited = text[: call.start(3)] + text[call.end(3): result.start(3)] + text[result.end(3):]
            # A note in place of the pair says the edit happened.
            return edited + f"\n\n{call.group(1).split(' id=')[0]} id=new-mark]]\n{MARK}", gone
    return None


class ClmReplayTests(unittest.TestCase):
    def test_a_shortened_tool_result_holds(self) -> None:
        self.run_all(shorten)

    def test_a_removed_tool_call_holds(self) -> None:
        self.run_all(remove_pair)

    def run_all(self, edit) -> None:
        self.assertGreaterEqual(len(FIXTURES), 15)
        for fixture in FIXTURES:
            name, requests = load(fixture)
            with self.subTest(name):
                checked = self.replay(requests, edit)
                self.assertGreater(checked, 0, "no request after the edit was checked")

    def replay(self, requests: list[tuple[str, dict[str, Any]]], edit) -> int:
        directory = Path(tempfile.mkdtemp())
        engine = Engine(ContextLanguageModel(budget=10_000_000, directory=str(directory)))
        edited: Path | None = None
        gone = ""
        applied_seen, checked = False, 0
        for n, (path, body) in enumerate(requests):
            codec, harness = codec_for(path), detect({}, path=path)
            exchange = engine.prepare(codec, harness, body, tenant="t", post=lambda r: (500, "no summaries"))
            engine.record(exchange, approx_tokens(json.dumps(exchange.body)))
            text = json.dumps(exchange.body, ensure_ascii=False)
            sent = codec.items(exchange.body)
            self.assertEqual(orphans(codec.name, sent), [], f"request {n}: a tool result without its call")
            if codec.name == "anthropic_messages":
                self.assertTrue(legal_system_messages(sent), f"request {n}: a system message placed illegally")
            if harness.compacting(codec, codec.items(body)) or not codec.offers_tools(body):
                continue
            self.assertIn("## Managing your context", text, f"request {n}: no guidance")
            self.assertIn("[context: ~", text, f"request {n}: no readout")
            self.assertIsNone(exchange.diverged, f"request {n}: left its stored context at {exchange.diverged}")
            mirror = Path(json.loads(f'"{PATH.search(text).group(1)}"'))
            if edited is None:
                result = edit(mirror.read_text()) if mirror.exists() else None
                if result:
                    edited, (new, gone) = mirror, result
                    mirror.write_text(new)
                continue
            if mirror != edited:
                continue  # another conversation (a sub-agent, a title request)
            if "[context file:" in text:
                self.assertIn("edit applied", text, f"request {n}: the edit was refused: "
                              + re.search(r"\[context file:[^\]]*\]", text).group(0))
                applied_seen = True
            self.assertTrue(applied_seen, f"request {n}: no receipt for the edit")
            self.assertIn(MARK, text, f"request {n}: the edit is missing")
            self.assertNotIn(json.dumps(gone, ensure_ascii=False)[1:-1], text, f"request {n}: removed text came back")
            checked += 1
        return checked


if __name__ == "__main__":
    unittest.main()
