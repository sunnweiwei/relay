"""The compaction strategy on its own: it sees the conversation and never the harness's context."""

from __future__ import annotations

import unittest
from dataclasses import replace

from relay.core.ir import Item, Kind, Media, Native, Request
from relay.harnesses import keep
from relay.harnesses.goose import Summary as GooseSummary
from relay.harnesses.workbuddy import Summary as WorkBuddySummary
from relay.prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from relay.strategies import Compaction


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def summarize(self, cut: int, prompt: str) -> str:
        self.calls.append((cut, prompt))
        return "SUMMARY"


def request(*items: Item, tokens: int = 1_000, force: bool = False, window: int | None = None,
            base: int | None = None) -> Request:
    return Request(tuple(items), tuple(items), frozenset(range(len(items) + 1)), tokens, window, force, base)


def item(kind: Kind, text: str = "", ref: int | None = None, media: tuple[Media, ...] = ()) -> Item:
    return Item(kind, text, ref, media)


TASK = item(Kind.USER, "fix the bug", 2)
CALL = item(Kind.TOOL_CALL, "run()", 3)
RESULT = item(Kind.TOOL_RESULT, "x" * 4_000, 4)
SUMMARY = Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nSUMMARY")


class CompactionTests(unittest.TestCase):
    strategy = Compaction(threshold=500, min_gain=0)

    def test_below_threshold_does_nothing(self) -> None:
        summarizer = FakeSummarizer()
        self.assertIsNone(self.strategy.plan(request(TASK, CALL, RESULT, tokens=499), summarizer))
        self.assertEqual(summarizer.calls, [])

    def test_mid_turn_summarizes_everything_and_ends_with_the_summary(self) -> None:
        summarizer = FakeSummarizer()
        context = self.strategy.plan(request(TASK, CALL, RESULT), summarizer)
        self.assertEqual(summarizer.calls, [(3, SUMMARIZATION_PROMPT)])
        self.assertEqual(context.items, (TASK, SUMMARY))

    def test_turn_start_keeps_the_new_turn_after_the_summary(self) -> None:
        follow_up = item(Kind.USER, "now test it", 6)
        context = self.strategy.plan(request(TASK, CALL, RESULT, item(Kind.ASSISTANT, "done", 5), follow_up),
                                     FakeSummarizer())
        self.assertEqual(context.items, (TASK, SUMMARY, follow_up))

    def test_keeps_newest_user_messages_within_budget_and_truncates_the_boundary(self) -> None:
        strategy = Compaction(threshold=500, retain_user_tokens=30, min_gain=0)
        old = item(Kind.USER, "o" * 400, 2)  # 100 tokens: truncated to the remaining 10
        recent = item(Kind.USER, "r" * 80, 5)  # 20 tokens: kept verbatim
        oldest = item(Kind.USER, "first", 0)  # dropped: the budget is spent
        context = strategy.plan(request(oldest, CALL, RESULT, old, CALL, recent, CALL, RESULT), FakeSummarizer())
        truncated, kept, _ = context.items
        self.assertIs(kept, recent)
        self.assertIsNone(truncated.ref)
        self.assertIn("tokens truncated", truncated.text)
        self.assertLess(len(truncated.text), 80)

    def test_previous_summaries_and_media_are_not_kept_verbatim(self) -> None:
        summary = item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\nold", None)
        screenshot = item(Kind.USER, "see image", 5, media=(Media("image", "image/png", "AAAA"),))
        context = self.strategy.plan(request(TASK, summary, CALL, RESULT, screenshot, CALL, RESULT), FakeSummarizer())
        self.assertNotIn(summary, context.items)
        self.assertEqual(context.items[1], Item(Kind.USER, "see image"))

    def test_skips_compaction_that_frees_too_little(self) -> None:
        strategy = Compaction(threshold=500, min_gain=0.5)
        small = item(Kind.TOOL_RESULT, "x" * 100, 4)
        self.assertIsNone(strategy.plan(request(TASK, CALL, small), FakeSummarizer()))
        self.assertIsNotNone(strategy.plan(request(TASK, CALL, small, force=True), FakeSummarizer()))

    def test_nothing_to_compact_without_agent_items(self) -> None:
        self.assertIsNone(self.strategy.plan(request(TASK, force=True), FakeSummarizer()))

    def test_threshold_defaults_to_a_share_of_the_context_window(self) -> None:
        self.assertEqual(Compaction().limit(200_000), 180_000)  # Codex: 90%
        self.assertEqual(Compaction(ratio=0.5).limit(200_000), 100_000)
        self.assertEqual(Compaction(threshold=7).limit(200_000), 7)
        self.assertEqual(Compaction(ratio=0.5).limit(None), 64_000)


class GrowthTests(unittest.TestCase):
    def test_growth_counts_tokens_added_since_the_window_began(self) -> None:
        strategy = Compaction(growth=500, min_gain=0)
        items = (TASK, CALL, RESULT)
        quiet = request(*items, tokens=10_400, base=10_000)
        busy = request(*items, tokens=10_600, base=10_000)
        unknown = request(*items, tokens=99_999)
        self.assertIsNone(strategy.plan(quiet, FakeSummarizer()))
        self.assertIsNotNone(strategy.plan(busy, FakeSummarizer()))
        self.assertIsNone(strategy.plan(unknown, FakeSummarizer()))

    def test_growth_mode_still_compacts_near_the_full_window(self) -> None:
        near_full = request(TASK, CALL, RESULT, tokens=95_000, window=100_000, base=94_000)
        self.assertIsNotNone(Compaction(growth=50_000, min_gain=0).plan(near_full, FakeSummarizer()))


if __name__ == "__main__":
    unittest.main()


def native(request_: Request, how: Native) -> Request:
    return replace(request_, native=how)


class NativeKeepTests(unittest.TestCase):
    """What a harness keeps verbatim when it compacts by itself, and when it compacts at all."""

    strategy = Compaction(threshold=500, min_gain=0)
    rounds = [item(Kind.USER, "task", 0), *(i for n in range(4) for i in
                                             (item(Kind.TOOL_CALL, f"read({n})", 1 + 2 * n),
                                              item(Kind.TOOL_RESULT, "x" * 2_000, 2 + 2 * n)))]

    def compact(self, how: Native, *items: Item) -> tuple[Item, ...]:
        summarizer = FakeSummarizer()
        context = self.strategy.plan(native(request(*items), how), summarizer)
        self.assertEqual(summarizer.calls[0][1], how.prompt)
        return context.items

    def test_round_keeps_the_model_s_last_call_and_its_result(self) -> None:
        kept = self.compact(Native("P", "<", ">", keep.round()), *self.rounds)
        self.assertEqual([i.kind for i in kept], [Kind.SUMMARY, Kind.TOOL_CALL, Kind.TOOL_RESULT])
        self.assertEqual(kept[0].text, "<SUMMARY>")

    def test_tail_keeps_what_fits_its_budget_never_opening_on_a_result(self) -> None:
        kept = self.compact(Native("P", keep=keep.tail(tokens=600)), *self.rounds)  # one round's worth, and a little
        self.assertEqual([i.kind for i in kept], [Kind.SUMMARY, Kind.TOOL_CALL, Kind.TOOL_RESULT])
        least = self.compact(Native("P", keep=keep.tail(tokens=10, least=4)), *self.rounds)  # at least four items
        self.assertEqual(len(least), 5)
        most = self.compact(Native("P", keep=keep.tail(tokens=10_000, most=1)), *self.rounds)  # one round at most
        self.assertEqual(len(most), 3)
        head = self.compact(Native("P", keep=keep.tail(tokens=10, head=1)), *self.rounds)  # the task stays first
        self.assertEqual([i.kind for i in head][:2], [Kind.USER, Kind.SUMMARY])

    def test_turns_keeps_the_turn_in_progress(self) -> None:
        turn = [*self.rounds, item(Kind.ASSISTANT, "done", 9), item(Kind.USER, "next", 10),
                item(Kind.TOOL_CALL, "read(9)", 11), item(Kind.TOOL_RESULT, "y", 12)]
        kept = self.compact(Native("P", keep=keep.turns(0.001)), *turn)
        self.assertEqual([i.text for i in kept[1:]], ["next", "read(9)", "y"])

    def test_users_keeps_the_user_messages_ahead_of_a_summary_of_everything(self) -> None:
        kept = self.compact(Native("P", keep=keep.users(20_000)), *self.rounds)
        self.assertEqual([i.kind for i in kept], [Kind.USER, Kind.SUMMARY])

    def test_split_keeps_from_the_first_user_message_past_its_share(self) -> None:
        turns = [*self.rounds, item(Kind.ASSISTANT, "done", 9), item(Kind.USER, "next", 10),
                 item(Kind.TOOL_CALL, "read(9)", 11), item(Kind.TOOL_RESULT, "y" * 100, 12)]
        kept = self.compact(Native("P", keep=keep.split(0.3)), *turns)
        self.assertEqual([i.text for i in kept[1:]], ["next", "read(9)", "y" * 100])

    def test_last_keeps_a_pending_turn_or_mid_turn_the_last_round(self) -> None:
        self.assertEqual([i.kind for i in self.compact(Native("P", keep=keep.last()), *self.rounds)],
                         [Kind.SUMMARY, Kind.TOOL_CALL, Kind.TOOL_RESULT])
        waiting = [*self.rounds, item(Kind.ASSISTANT, "done", 9), item(Kind.USER, "next", 10)]
        self.assertEqual([i.text for i in self.compact(Native("P", keep=keep.last()), *waiting)][1:], ["next"])

    def test_a_harness_compacting_at_turn_starts_waits_for_one(self) -> None:
        how = Native("P", keep=keep.pending(), at_turns=True)
        self.assertIsNone(self.strategy.plan(native(request(*self.rounds), how), FakeSummarizer()))
        self.assertIsNotNone(self.strategy.plan(native(request(*self.rounds, force=True), how), FakeSummarizer()))
        waiting = [*self.rounds, item(Kind.ASSISTANT, "done", 9), item(Kind.USER, "next", 10)]
        self.assertIsNotNone(self.strategy.plan(native(request(*waiting), how), FakeSummarizer()))


class SummaryMessageTests(unittest.TestCase):
    def test_goose_renders_its_json_and_keeps_text_it_cannot_read(self) -> None:
        answer = ('<analysis>notes</analysis>\n```json\n{"user_intent": ["read files"], "files": [{"path": "a", '
                  '"summary": "CODE amber"}], "current_work": "reading b"}\n```')
        self.assertEqual(GooseSummary("").message(answer),
                         "# Conversation Summary\n\n## User Intent\n- read files\n\n## Files + Code\n### a\nCODE amber\n\n"
                         "## Current Work\nreading b")
        self.assertEqual(GooseSummary("").message("plain"), "plain")

    def test_workbuddy_keeps_the_summary_its_model_marks(self) -> None:
        message = WorkBuddySummary("", "<", ">").message
        self.assertEqual(message("<analysis>a</analysis><summary>\nS\n</summary>"), "<S>")
        self.assertEqual(message("S"), "<<conversation_history_summary>\nS\n</conversation_history_summary>>")
