"""The compaction strategy on its own: it sees the conversation and never the harness's context."""

from __future__ import annotations

import unittest

from relay.core.ir import Item, Kind, View
from relay.prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from relay.strategies import Compaction


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def summarize(self, cut: int, prompt: str) -> str:
        self.calls.append((cut, prompt))
        return "SUMMARY"


def view(*items: Item, tokens: int = 1_000, force: bool = False) -> View:
    return View(tuple(items), frozenset(range(len(items) + 1)), tokens, None, force)


def item(kind: Kind, text: str = "", ref: int | None = None, media: bool = False) -> Item:
    return Item(kind, text, ref, media)


TASK = item(Kind.USER, "fix the bug", 2)
CALL = item(Kind.TOOL_CALL, "run()", 3)
RESULT = item(Kind.TOOL_RESULT, "x" * 4_000, 4)
SUMMARY = Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nSUMMARY")


class CompactionTests(unittest.TestCase):
    strategy = Compaction(threshold=500, min_gain=0)

    def test_below_threshold_does_nothing(self) -> None:
        summarizer = FakeSummarizer()
        self.assertIsNone(self.strategy.plan(view(TASK, CALL, RESULT, tokens=499), summarizer))
        self.assertEqual(summarizer.calls, [])

    def test_mid_turn_summarizes_everything_and_ends_with_the_summary(self) -> None:
        summarizer = FakeSummarizer()
        rewrite = self.strategy.plan(view(TASK, CALL, RESULT), summarizer)
        self.assertEqual(summarizer.calls, [(3, SUMMARIZATION_PROMPT)])
        self.assertEqual(rewrite.cut, 3)
        self.assertEqual(rewrite.head, (TASK, SUMMARY))

    def test_turn_start_keeps_the_new_turn_after_the_summary(self) -> None:
        follow_up = item(Kind.USER, "now test it", 6)
        rewrite = self.strategy.plan(view(TASK, CALL, RESULT, item(Kind.ASSISTANT, "done", 5), follow_up),
                                     FakeSummarizer())
        self.assertEqual(rewrite.cut, 4)
        self.assertEqual(rewrite.head, (TASK, SUMMARY))

    def test_keeps_newest_user_messages_within_budget_and_truncates_the_boundary(self) -> None:
        strategy = Compaction(threshold=500, retain_user_tokens=30, min_gain=0)
        old = item(Kind.USER, "o" * 400, 2)  # 100 tokens: truncated to the remaining 10
        recent = item(Kind.USER, "r" * 80, 5)  # 20 tokens: kept verbatim
        oldest = item(Kind.USER, "first", 0)  # dropped: the budget is spent
        rewrite = strategy.plan(view(oldest, CALL, RESULT, old, CALL, recent, CALL, RESULT), FakeSummarizer())
        truncated, kept, _ = rewrite.head
        self.assertIs(kept, recent)
        self.assertIsNone(truncated.ref)
        self.assertIn("tokens truncated", truncated.text)
        self.assertLess(len(truncated.text), 80)

    def test_previous_summaries_and_media_are_not_kept_verbatim(self) -> None:
        summary = item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nold", None)
        screenshot = item(Kind.USER, "see image", 5, media=True)
        rewrite = self.strategy.plan(view(TASK, summary, CALL, RESULT, screenshot, CALL, RESULT), FakeSummarizer())
        self.assertNotIn(summary, rewrite.head)
        self.assertEqual(rewrite.head[1], Item(Kind.USER, "see image"))

    def test_skips_compaction_that_frees_too_little(self) -> None:
        strategy = Compaction(threshold=500, min_gain=0.5)
        small = item(Kind.TOOL_RESULT, "x" * 100, 4)
        self.assertIsNone(strategy.plan(view(TASK, CALL, small), FakeSummarizer()))
        self.assertIsNotNone(strategy.plan(view(TASK, CALL, small, force=True), FakeSummarizer()))

    def test_nothing_to_compact_without_agent_items(self) -> None:
        self.assertIsNone(self.strategy.plan(view(TASK, force=True), FakeSummarizer()))

    def test_threshold_defaults_to_a_share_of_the_context_window(self) -> None:
        self.assertEqual(Compaction().limit(200_000), 180_000)  # Codex: 90%
        self.assertEqual(Compaction(ratio=0.5).limit(200_000), 100_000)
        self.assertEqual(Compaction(threshold=7).limit(200_000), 7)
        self.assertEqual(Compaction(ratio=0.5).limit(None), 64_000)


class GrowthTests(unittest.TestCase):
    def test_growth_counts_tokens_added_since_the_window_began(self) -> None:
        strategy = Compaction(growth=500, min_gain=0)
        items = (TASK, CALL, RESULT)
        quiet = View(items, frozenset(range(4)), 10_400, None, base=10_000)
        busy = View(items, frozenset(range(4)), 10_600, None, base=10_000)
        unknown = View(items, frozenset(range(4)), 99_999, None, base=None)
        self.assertIsNone(strategy.plan(quiet, FakeSummarizer()))
        self.assertIsNotNone(strategy.plan(busy, FakeSummarizer()))
        self.assertIsNone(strategy.plan(unknown, FakeSummarizer()))

    def test_growth_mode_still_compacts_near_the_full_window(self) -> None:
        items = (TASK, CALL, RESULT)
        near_full = View(items, frozenset(range(4)), 95_000, 100_000, base=94_000)
        self.assertIsNotNone(Compaction(growth=50_000, min_gain=0).plan(near_full, FakeSummarizer()))


if __name__ == "__main__":
    unittest.main()
