from __future__ import annotations

import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from relay import ContextEngine, MultiGranCompact, PrefixCheckpointCache
from relay.strategies import strategy_from_env
from relay.strategies import multi_gran_compact as mgc
from relay.strategies.base import GeneratedCheckpoint
from relay.strategies.multi_gran_compact import (
    GENERAL_MEMORY_HEADER as MEMORY_HEADER,
    _item_text,
    _leading_prefix,
    _render_span,
)


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


class _FakeCompletions:
    """Records chat.completions.create calls and returns a deterministic note."""

    def __init__(self, owner: "_FakeChatClient") -> None:
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(deepcopy(kwargs))
        # Number the note by how many prior notes the compactor was shown, so the
        # test can confirm the persistent [system, note_1, ..., note_k] conversation
        # is rebuilt from the artifact each call.
        prior = sum(1 for m in kwargs["messages"] if m["role"] == "assistant")
        content = f"NOTE#{prior + 1}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeChat:
    def __init__(self, owner: "_FakeChatClient") -> None:
        self.completions = _FakeCompletions(owner)


class _FakeResponsesEndpoint:
    def __init__(self, owner: "_FakeChatClient") -> None:
        self.owner = owner

    def create(self, **kwargs):
        self.owner.responses_calls.append(deepcopy(kwargs))
        return SimpleNamespace(output_text="NOTE-GPT5")


class _FakeChatClient:
    """Stands in for an openai.OpenAI client (chat.completions + responses)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses_calls: list[dict] = []
        self.chat = _FakeChat(self)
        self.responses = _FakeResponsesEndpoint(self)


class MultiGranCompactTest(unittest.TestCase):
    def setUp(self) -> None:
        # Count 1 token per character so the fold cadence is fully deterministic and
        # the Qwen tokenizer is never loaded.
        patcher = patch.object(mgc, "_count_tokens", lambda text, name: max(1, len(text)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _strategy(self, **kwargs) -> tuple[MultiGranCompact, _FakeChatClient]:
        strategy = MultiGranCompact(**kwargs)
        client = _FakeChatClient()
        # Inject the fake compactor client so _client() never builds a real one.
        strategy._client_cache = client
        return strategy, client

    def _trajectory(self, n_interactions: int, *, task: str = "Q: find the entity"):
        traj = [message("system", "sys"), message("user", task)]
        for i in range(n_interactions):
            traj.append(message("assistant", "a" * 14 + f"{i:02d}"))  # 16 chars
            traj.append(message("user", "o" * 14 + f"{i:02d}"))  # 16 chars
        return traj

    # ------------------------------------------------------------------ folding --
    def test_fold_cadence_produces_accumulating_notes(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, max_compaction=10)
        traj = self._trajectory(3)  # each interaction ~32 chars >= 30 -> one note each

        prepared = strategy.prepare(None, {"model": "task"}, traj)

        notes = prepared.checkpoint.artifact["notes"]
        self.assertEqual([n["text"] for n in notes], ["NOTE#1", "NOTE#2", "NOTE#3"])
        # Notes fold whole interactions at tool-safe boundaries after the protected prefix.
        self.assertEqual([n["end"] for n in notes], [4, 6, 8])
        self.assertEqual(len(client.calls), 3)
        # Active = [system, task, note_1, note_2, note_3] (tail fully folded here).
        self.assertEqual(prepared.input[0], message("system", "sys"))
        self.assertEqual(prepared.input[1], message("user", "Q: find the entity"))
        self.assertTrue(prepared.input[2]["content"].startswith(MEMORY_HEADER))
        self.assertTrue(prepared.compacted)

    def test_protected_prefix_keeps_task_verbatim(self) -> None:
        strategy, _ = self._strategy(compact_threshold=30)
        traj = self._trajectory(3)

        prepared = strategy.prepare(None, {"model": "task"}, traj)

        # The system turn and the initial user task are never folded.
        self.assertEqual(prepared.input[1], message("user", "Q: find the entity"))
        # The first note begins strictly after the protected prefix (index 2).
        self.assertEqual(prepared.checkpoint.artifact["notes"][0]["end"], 4)

    def test_no_fold_below_threshold(self) -> None:
        strategy, client = self._strategy(compact_threshold=10_000)
        traj = self._trajectory(2)

        prepared = strategy.prepare(None, {"model": "task"}, traj)

        self.assertIsNone(prepared.checkpoint)
        self.assertFalse(prepared.compacted)
        self.assertEqual(prepared.input, traj)
        self.assertEqual(client.calls, [])

    # ------------------------------------------------------------- materialize --
    def test_materialize_round_trips_prepare(self) -> None:
        strategy, _ = self._strategy(compact_threshold=30, max_compaction=10)
        traj = self._trajectory(3)

        prepared = strategy.prepare(None, {"model": "task"}, traj)
        rebuilt = strategy.materialize(traj, prepared.checkpoint)

        self.assertEqual(rebuilt, prepared.input)

    def test_materialize_without_checkpoint_is_identity(self) -> None:
        strategy, _ = self._strategy()
        traj = self._trajectory(2)
        self.assertEqual(strategy.materialize(traj, None), traj)

    # ------------------------------------------------------------------- cap ----
    def test_max_compaction_caps_folds_and_appends_footer(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, max_compaction=2)
        traj = self._trajectory(5)  # would fold 5x, but the cap is 2

        prepared = strategy.prepare(None, {"model": "task"}, traj)

        notes = prepared.checkpoint.artifact["notes"]
        self.assertEqual(len(notes), 2)
        self.assertEqual(len(client.calls), 2)
        # The final (capped) note carries the commit-now footer; earlier notes do not.
        footer = strategy._footer()
        note_items = [it for it in prepared.input if it["content"].startswith(MEMORY_HEADER)]
        self.assertTrue(note_items[-1]["content"].endswith(footer))
        self.assertNotIn(footer, note_items[0]["content"])
        # Beyond the cap the tail stays raw and verbatim.
        self.assertEqual(prepared.input[-1], traj[-1])

    def test_capped_checkpoint_stops_folding_on_replay(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, max_compaction=2)
        traj = self._trajectory(5)
        prepared = strategy.prepare(None, {"model": "task"}, traj)
        self.assertEqual(len(client.calls), 2)

        # Feed the capped checkpoint back with more raw interactions: no new folds.
        extended = traj + [message("assistant", "x" * 40), message("user", "y" * 40)]
        again = strategy.prepare(None, {"model": "task"}, extended, prepared.checkpoint)

        self.assertIsNone(again.checkpoint)  # nothing changed
        self.assertEqual(len(client.calls), 2)  # compactor not called again
        # Still compacted (from the recovered notes) and the footer still present.
        self.assertTrue(again.compacted)
        self.assertTrue(any(strategy._footer() in it.get("content", "") for it in again.input))

    # -------------------------------------------------------------- incremental --
    def test_incremental_reuse_only_folds_new_spans(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, max_compaction=10)
        traj = self._trajectory(2)
        first = strategy.prepare(None, {"model": "task"}, traj)
        self.assertEqual(len(client.calls), 2)

        # Two more raw interactions arrive; the recovered checkpoint covers the old ones.
        traj2 = traj + [message("assistant", "b" * 16), message("user", "c" * 16)]
        second = strategy.prepare(None, {"model": "task"}, traj2, first.checkpoint)

        self.assertEqual(len(second.checkpoint.artifact["notes"]), 3)
        self.assertEqual(len(client.calls), 3)  # only ONE new fold
        # The most recent compactor call saw the two prior notes as context.
        prior_assistants = [m for m in client.calls[-1]["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(prior_assistants), 2)

    # ------------------------------------------------------------------ compact --
    def test_compact_folds_the_remaining_tail(self) -> None:
        strategy, _ = self._strategy(compact_threshold=10_000, max_compaction=10)
        active = self._trajectory(2)  # below threshold: prepare would not fold

        result = strategy.compact(None, {"model": "task"}, active)

        # compact() force-folds the sub-threshold tail into a final note.
        self.assertEqual(result[0], message("system", "sys"))
        self.assertEqual(result[1], message("user", "Q: find the entity"))
        self.assertTrue(result[-1]["content"].startswith(MEMORY_HEADER))
        self.assertEqual(len(result), 3)

    # ------------------------------------------------------------- gpt-5 branch --
    def test_gpt5_model_uses_responses_api(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, reasoning_effort="high")
        traj = self._trajectory(1)

        prepared = strategy.prepare(None, {"model": "gpt-5.6-luna"}, traj)

        self.assertEqual(client.calls, [])  # chat.completions not used
        self.assertEqual(len(client.responses_calls), 1)
        self.assertEqual(client.responses_calls[0]["reasoning"], {"effort": "high"})
        self.assertEqual(prepared.checkpoint.artifact["notes"][0]["text"], "NOTE-GPT5")

    # --------------------------------------------------------------- integrity --
    def test_tampered_note_boundary_is_rejected(self) -> None:
        strategy, _ = self._strategy(compact_threshold=30, max_compaction=3)
        traj = self._trajectory(3)
        bad = GeneratedCheckpoint(
            covered_items=len(traj),
            artifact={
                "version": 1,
                "kind": "multi_gran_compact",
                "compact_threshold": 30,
                "max_compaction": 3,
                "notes": [{"end": 999, "text": "x"}],  # end exceeds covered_items
            },
        )
        with self.assertRaises(ValueError):
            strategy.materialize(traj, bad)

    def test_threshold_change_mid_trajectory_is_rejected(self) -> None:
        strategy, _ = self._strategy(compact_threshold=30, max_compaction=3)
        traj = self._trajectory(3)
        stale = GeneratedCheckpoint(
            covered_items=len(traj),
            artifact={
                "version": 1,
                "kind": "multi_gran_compact",
                "compact_threshold": 999,  # differs from the strategy config
                "max_compaction": 3,
                "notes": [{"end": 4, "text": "n"}],
            },
        )
        with self.assertRaises(ValueError):
            strategy.materialize(traj, stale)

    # ------------------------------------------------------------ engine + cache --
    def test_cache_mode_engine_reuses_folds(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, max_compaction=10)
        cache = PrefixCheckpointCache(secret=b"test-secret")
        engine = ContextEngine(
            strategy, checkpoint_mode="cache", checkpoint_cache=cache
        )
        traj = self._trajectory(2)

        first = engine.prepare(object(), {"model": "task", "input": traj}, "tenant-a")
        self.assertEqual(len(client.calls), 2)
        # A folded note replaces the raw span in what would be forwarded upstream.
        self.assertTrue(any(it["content"].startswith(MEMORY_HEADER) for it in first.input))

        # Same tenant, extended raw trajectory: the cached prefix is reused.
        traj2 = traj + [message("assistant", "d" * 16), message("user", "e" * 16)]
        engine.prepare(object(), {"model": "task", "input": traj2}, "tenant-a")

        self.assertGreaterEqual(cache.stats().hits, 1)
        self.assertEqual(len(client.calls), 3)  # only the new span was folded

    # ---------------------------------------------------------------- profiles --
    def test_browsecomp_profile_swaps_the_whole_prompt_bundle(self) -> None:
        strategy, client = self._strategy(
            compact_threshold=30, max_compaction=1, task_profile="browsecomp"
        )
        traj = self._trajectory(2)
        prepared = strategy.prepare(None, {"model": "task"}, traj)

        # The compactor saw the BrowseComp system prompt (docid framing), not the general one.
        system = client.calls[0]["messages"][0]["content"]
        self.assertIn("BrowseComp", system)
        self.assertIn("docid", system)
        self.assertIn("The research question:", system)
        # Notes use the BrowseComp memory header and finish footer.
        note = next(it for it in prepared.input if "trustworthy" in it.get("content", ""))
        self.assertIn("open_page", note["content"])
        self.assertIn("`finish` function", note["content"])

    def test_general_profile_has_no_browsecomp_vocabulary(self) -> None:
        strategy, client = self._strategy(compact_threshold=30, max_compaction=1)
        strategy.prepare(None, {"model": "task"}, self._trajectory(2))
        system = client.calls[0]["messages"][0]["content"]
        self.assertNotIn("BrowseComp", system)
        self.assertNotIn("docid", system)
        self.assertIn("The task the agent is working on:", system)

    def test_finish_footer_comes_from_the_profile(self) -> None:
        general, _ = self._strategy(compact_threshold=30, max_compaction=1)
        browse, _ = self._strategy(
            compact_threshold=30, max_compaction=1, task_profile="browsecomp"
        )
        self.assertIn("finishing action", general._footer())
        self.assertIn("`finish` function", browse._footer())
        self.assertNotEqual(general._footer(), browse._footer())

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiGranCompact(task_profile="nonexistent")

    # ---------------------------------------------------------------- from_env --
    def test_strategy_from_env(self) -> None:
        env = {
            "RELAY_STRATEGY": "multi_gran_compact",
            "RELAY_MULTI_GRAN_THRESHOLD": "12345",
            "RELAY_MULTI_GRAN_MAX_COMPACTION": "7",
            "RELAY_MULTI_GRAN_MODEL": "Qwen/Qwen3.5-9B",
            "RELAY_MULTI_GRAN_BASE_URL": "http://localhost:8018/v1",
            "RELAY_MULTI_GRAN_TASK_PROFILE": "browsecomp",
        }
        with patch.dict(os.environ, env, clear=False):
            strategy = strategy_from_env()
        self.assertIsInstance(strategy, MultiGranCompact)
        self.assertEqual(strategy.name, "multi_gran_compact")
        self.assertEqual(strategy.compact_threshold, 12345)
        self.assertEqual(strategy.max_compaction, 7)
        self.assertEqual(strategy.compact_model, "Qwen/Qwen3.5-9B")
        self.assertEqual(strategy.compact_base_url, "http://localhost:8018/v1")
        self.assertEqual(strategy.task_profile, "browsecomp")


class RenderingHelpersTest(unittest.TestCase):
    def test_render_span_handles_tool_and_reasoning_items(self) -> None:
        span = [
            {"type": "function_call", "name": "search", "arguments": '{"q":"x"}'},
            {"type": "function_call_output", "call_id": "c", "output": "doc [20]"},
            {"type": "reasoning", "encrypted_content": "opaque"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ]
        rendered = _render_span(span)
        self.assertIn("[Action]\nsearch(", rendered)
        self.assertIn("[Observation]\ndoc [20]", rendered)
        self.assertIn("[Action]\nhello", rendered)
        self.assertNotIn("opaque", rendered)  # encrypted reasoning renders nothing

    def test_item_text_extracts_visible_content(self) -> None:
        self.assertEqual(_item_text(message("user", "plain")), "plain")
        self.assertEqual(
            _item_text(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "abc"}],
                }
            ),
            "abc",
        )
        self.assertEqual(_item_text({"type": "reasoning"}), "")

    def test_leading_prefix_covers_system_and_task(self) -> None:
        traj = [
            message("system", "s"),
            message("user", "task"),
            {"type": "function_call", "name": "search", "arguments": "{}"},
        ]
        self.assertEqual(_leading_prefix(traj), 2)
        # An assistant turn immediately ends the protected prefix.
        self.assertEqual(
            _leading_prefix([message("system", "s"), message("assistant", "a")]), 1
        )


if __name__ == "__main__":
    unittest.main()
