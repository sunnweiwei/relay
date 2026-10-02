from __future__ import annotations

import unittest

from relay.core.ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind, View
from relay.prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from relay.strategies import Compaction


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def summarize(self, cut: int, prompt: str) -> str:
        self.calls.append((cut, prompt))
        return "SUMMARY"


def view(*items: Item, tokens: int = 1_000, force: bool = False, initial: tuple[Item, ...] | None = None) -> View:
    if initial is None:  # context before the first model action, as on a conversation's first compaction
        first_action = next((i for i, it in enumerate(items) if it.kind in AGENT_KINDS), len(items))
        initial = tuple(it for it in items[:first_action] if it.kind in CONTEXT_KINDS)
    return View(tuple(items), frozenset(range(len(items) + 1)), tokens, None, force, initial=initial)


def item(kind: Kind, text: str = "", ref: int | None = None, media: bool = False) -> Item:
    return Item(kind, text, ref, media)


SYSTEM = item(Kind.SYSTEM, "rules", 0)
CONTEXT = item(Kind.CONTEXT, "<environment_context>", 1)
TASK = item(Kind.USER, "fix the bug", 2)
CALL = item(Kind.TOOL_CALL, "run()", 3)
RESULT = item(Kind.TOOL_RESULT, "x" * 4_000, 4)


class CompactionTests(unittest.TestCase):
    strategy = Compaction(threshold=500, min_gain=0)

    def test_below_threshold_does_nothing(self) -> None:
        summarizer = FakeSummarizer()
        self.assertIsNone(self.strategy.plan(view(SYSTEM, TASK, CALL, RESULT, tokens=499), summarizer))
        self.assertEqual(summarizer.calls, [])

    def test_mid_turn_summarizes_everything_and_ends_with_the_summary(self) -> None:
        summarizer = FakeSummarizer()
        rewrite = self.strategy.plan(view(SYSTEM, CONTEXT, TASK, CALL, RESULT), summarizer)
        self.assertEqual(summarizer.calls, [(5, SUMMARIZATION_PROMPT)])
        self.assertEqual(rewrite.cut, 5)
        self.assertEqual(rewrite.head[:3], (SYSTEM, CONTEXT, TASK))
        self.assertEqual(rewrite.head[3], Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nSUMMARY"))

    def test_turn_start_keeps_the_new_turn_after_the_summary(self) -> None:
        new_context = item(Kind.CONTEXT, "<environment_context>", 5)
        follow_up = item(Kind.USER, "now test it", 6)
        rewrite = self.strategy.plan(
            view(SYSTEM, TASK, CALL, RESULT, item(Kind.ASSISTANT, "done", 4), new_context, follow_up),
            FakeSummarizer(),
        )
        self.assertEqual(rewrite.cut, 5)
        self.assertEqual([i.kind for i in rewrite.head], [Kind.SYSTEM, Kind.USER, Kind.SUMMARY])

    def test_codex_layout_with_several_user_turns(self) -> None:
        first, second = item(Kind.USER, "first task", 2), item(Kind.USER, "second task", 6)
        done = item(Kind.ASSISTANT, "done", 5)
        history = (SYSTEM, CONTEXT, first, CALL, RESULT, done, second, item(Kind.TOOL_CALL, "", 7),
                   item(Kind.TOOL_RESULT, "y" * 4_000, 8))
        # mid-turn: initial context just above the last real user message, summary last
        mid = self.strategy.plan(view(*history), FakeSummarizer())
        self.assertEqual(mid.head[:5], (SYSTEM, first, CONTEXT, second, mid.head[4]))
        self.assertIs(mid.head[4].kind, Kind.SUMMARY)
        # turn start: context re-injected after the summary, then the new turn
        start = self.strategy.plan(view(*history[:7]), FakeSummarizer())
        self.assertEqual(start.cut, 6)
        self.assertEqual([i.kind for i in start.head], [Kind.SYSTEM, Kind.USER, Kind.SUMMARY, Kind.CONTEXT])

    def test_a_second_compaction_keeps_the_initial_context(self) -> None:
        first = self.strategy.plan(view(SYSTEM, CONTEXT, TASK, CALL, RESULT), FakeSummarizer())
        later = (*first.head, item(Kind.TOOL_CALL, "", 5), item(Kind.TOOL_RESULT, "z" * 4_000, 6))
        second = self.strategy.plan(view(*later, initial=(SYSTEM, CONTEXT)), FakeSummarizer())
        self.assertEqual([i.kind for i in second.head], [Kind.SYSTEM, Kind.CONTEXT, Kind.USER, Kind.SUMMARY])
        self.assertEqual(second.head[:3], (SYSTEM, CONTEXT, TASK))

    def test_initial_context_returns_even_if_an_earlier_rewrite_dropped_it(self) -> None:
        later = (TASK, item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nold"), CALL, RESULT)  # no SYSTEM, CONTEXT left
        rewrite = self.strategy.plan(view(*later, initial=(SYSTEM, CONTEXT)), FakeSummarizer())
        self.assertEqual(rewrite.head[:3], (SYSTEM, CONTEXT, TASK))

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
        self.assertIsNone(self.strategy.plan(view(SYSTEM, TASK, force=True), FakeSummarizer()))

    def test_threshold_defaults_to_a_share_of_the_context_window(self) -> None:
        self.assertEqual(Compaction().limit(200_000), 180_000)  # Codex: 90%
        self.assertEqual(Compaction(ratio=0.5).limit(200_000), 100_000)
        self.assertEqual(Compaction(threshold=7).limit(200_000), 7)
        self.assertEqual(Compaction(ratio=0.5).limit(None), 64_000)


if __name__ == "__main__":
    unittest.main()


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
