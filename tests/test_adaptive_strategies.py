from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from relay import AgentFold, AutoCompact, ContextFolding
from relay.strategies import strategy_from_env
from relay.strategies.agent_fold import OFFICIAL_AGENT_FOLD_COMMIT
from relay.strategies.auto_compact import AUTO_CONTEXT_SUMMARY
from relay.strategies.context_folding import (
    CONTEXT_FOLDING_RETURN_PREFIX,
    OFFICIAL_CONTEXT_FOLDING_COMMIT,
)


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


class FakeInputTokens:
    def __init__(self, owner: FakeManagerResponses) -> None:
        self.owner = owner

    def count(self, **request):
        self.owner.count_calls.append(request)
        return SimpleNamespace(input_tokens=self.owner.token_count)


class FakeManagerResponses:
    def __init__(self, *decisions: dict, token_count: int = 1) -> None:
        self.decisions = list(decisions)
        self.text_outputs: list[str] = []
        self.token_count = token_count
        self.create_calls: list[dict] = []
        self.count_calls: list[dict] = []
        self.input_tokens = FakeInputTokens(self)

    def create(self, **request):
        self.create_calls.append(request)
        if request.get("text", {}).get("format", {}).get("type") == "json_schema":
            return SimpleNamespace(
                output=[], output_text=json.dumps(self.decisions.pop(0))
            )
        return SimpleNamespace(output=[], output_text=self.text_outputs.pop(0))


class ContextFoldingTests(unittest.TestCase):
    def test_missing_checkpoint_replays_hidden_branch_state(self) -> None:
        api = FakeManagerResponses(
            {"action": "open", "objective": "investigate", "summary": ""},
            {
                "action": "return",
                "objective": "",
                "summary": "replayed branch report",
            },
        )
        raw = [
            message("user", "task"),
            message("assistant", "branch step one"),
            message("assistant", "branch step two"),
        ]

        prepared = ContextFolding().prepare(api, {"model": "task"}, raw)

        self.assertEqual(len(api.create_calls), 2)
        self.assertEqual(len(prepared.checkpoints), 2)
        self.assertIn("replayed branch report", prepared.input[-1]["content"])

    def test_hidden_branch_opens_then_returns_without_control_items(self) -> None:
        api = FakeManagerResponses(
            {"action": "open", "objective": "inspect parser", "summary": ""},
            {
                "action": "return",
                "objective": "",
                "summary": "Parser fixed; focused tests pass.",
            },
        )
        strategy = ContextFolding(manager_model="manager")
        task = [message("user", "Fix the parser")]

        initial = strategy.prepare(api, {"model": "task"}, task)
        assert initial.checkpoint is not None
        first_raw = [*task, message("assistant", "Located the parser")]
        opened = strategy.prepare(
            api, {"model": "task"}, first_raw, initial.checkpoint
        )
        assert opened.checkpoint is not None
        self.assertEqual(opened.input, first_raw)
        self.assertEqual(opened.checkpoint.artifact["branch"]["anchor"], 1)

        second_raw = [*first_raw, message("assistant", "Patched and tested it")]
        returned = strategy.prepare(
            api, {"model": "task"}, second_raw, opened.checkpoint
        )
        assert returned.checkpoint is not None
        self.assertEqual(returned.input[0], task[0])
        self.assertEqual(len(returned.input), 2)
        self.assertIn(CONTEXT_FOLDING_RETURN_PREFIX, returned.input[1]["content"])
        self.assertIn("focused tests pass", returned.input[1]["content"])
        self.assertIsNone(returned.checkpoint.artifact["branch"])
        self.assertFalse(
            any(item.get("name") in {"branch", "return"} for item in returned.input)
        )
        self.assertEqual(
            api.create_calls[0]["text"]["format"]["name"],
            "relay_context_folding_decision",
        )
        self.assertEqual(
            strategy.cache_scope()["official_commit"],
            OFFICIAL_CONTEXT_FOLDING_COMMIT,
        )

    def test_cache_state_rebuilds_the_same_active_input(self) -> None:
        api = FakeManagerResponses(
            {"action": "open", "objective": "investigate", "summary": ""}
        )
        strategy = ContextFolding()
        raw = [message("user", "task")]
        initial = strategy.prepare(api, {"model": "task"}, raw)
        assert initial.checkpoint is not None
        raw.append(message("assistant", "evidence"))
        opened = strategy.prepare(api, {"model": "task"}, raw, initial.checkpoint)
        assert opened.checkpoint is not None
        self.assertEqual(strategy.materialize(raw, opened.checkpoint), raw)


class AgentFoldTests(unittest.TestCase):
    def test_missing_checkpoint_rebuilds_each_multi_scale_summary(self) -> None:
        api = FakeManagerResponses(
            {"compress_range": [1, 1], "compress_text": "step one"},
            {"compress_range": [1, 2], "compress_text": "steps one and two"},
        )
        raw = [
            message("user", "task"),
            message("assistant", "raw one"),
            message("assistant", "raw two"),
        ]

        prepared = AgentFold().prepare(api, {"model": "task"}, raw)

        self.assertEqual(len(api.create_calls), 2)
        self.assertEqual(len(prepared.checkpoints), 2)
        self.assertIn("[Compressed Step 1]", prepared.input[1]["content"])
        self.assertEqual(prepared.input[-1]["content"], "raw two")
        assert prepared.checkpoint is not None
        self.assertEqual(
            prepared.checkpoint.artifact["summaries"][0]["content"],
            "steps one and two",
        )

    def test_granular_then_deep_fold_keeps_latest_interaction_raw(self) -> None:
        api = FakeManagerResponses(
            {"compress_range": [1, 1], "compress_text": "Found parser.py."},
            {
                "compress_range": [1, 2],
                "compress_text": "Localized and fixed parser.py.",
            },
        )
        strategy = AgentFold(manager_model="manager")
        task = [message("user", "Fix parser")]
        initial = strategy.prepare(api, {"model": "task"}, task)
        assert initial.checkpoint is not None

        first_raw = [*task, message("assistant", "Inspected parser.py")]
        first = strategy.prepare(
            api, {"model": "task"}, first_raw, initial.checkpoint
        )
        assert first.checkpoint is not None
        # AgentFold folds and acts concurrently: C_t still contains raw I_{t-1}.
        self.assertEqual(first.input, first_raw)
        self.assertEqual(
            first.checkpoint.artifact["summaries"],
            [{"start": 1, "end": 1, "content": "Found parser.py."}],
        )

        second_raw = [*first_raw, message("assistant", "Applied the fix")]
        second = strategy.prepare(
            api, {"model": "task"}, second_raw, first.checkpoint
        )
        assert second.checkpoint is not None
        self.assertIn("[Compressed Step 1]", second.input[1]["content"])
        self.assertEqual(second.input[-1]["content"], "Applied the fix")
        self.assertNotIn("Inspected parser.py", str(second.input))
        self.assertEqual(
            second.checkpoint.artifact["summaries"],
            [
                {
                    "start": 1,
                    "end": 2,
                    "content": "Localized and fixed parser.py.",
                }
            ],
        )
        materialized = strategy.materialize(second_raw, second.checkpoint)
        self.assertIn("[Compressed Step 1 to 2]", materialized[1]["content"])
        self.assertEqual(
            strategy.cache_scope()["official_commit"], OFFICIAL_AGENT_FOLD_COMMIT
        )

    def test_rejects_a_range_that_does_not_end_at_latest_step(self) -> None:
        api = FakeManagerResponses(
            {"compress_range": [1, 2], "compress_text": "invalid"}
        )
        strategy = AgentFold()
        task = [message("user", "task")]
        initial = strategy.prepare(api, {"model": "task"}, task)
        assert initial.checkpoint is not None
        with self.assertRaisesRegex(ValueError, "compression range"):
            strategy.prepare(
                api,
                {"model": "task"},
                [*task, message("assistant", "step one")],
                initial.checkpoint,
            )


class AutoCompactTests(unittest.TestCase):
    def test_manager_keeps_then_compacts_at_a_phase_boundary(self) -> None:
        api = FakeManagerResponses(
            {"action": "keep", "summary": "", "reason": "still localizing"},
            {
                "action": "compact",
                "summary": "Localization complete; edit parser.py next.",
                "reason": "phase complete",
            },
        )
        strategy = AutoCompact(
            manager_model="manager",
            fallback_threshold=10_000,
            keep_recent_interactions=1,
        )
        first = [
            message("user", "Fix parser"),
            message("assistant", "Searched several files"),
        ]
        kept = strategy.prepare(api, {"model": "task"}, first)
        self.assertEqual(kept.input, first)
        self.assertIsNone(kept.checkpoint)

        second = [*first, message("assistant", "Found parser.py")]
        compacted = strategy.prepare(api, {"model": "task"}, second)
        assert compacted.checkpoint is not None
        self.assertEqual(compacted.input[0]["content"], "Fix parser")
        self.assertTrue(compacted.input[1]["content"].startswith(AUTO_CONTEXT_SUMMARY))
        self.assertEqual(compacted.input[-1]["content"], "Found parser.py")
        self.assertNotIn("Searched several files", str(compacted.input))
        self.assertEqual(
            api.create_calls[-1]["text"]["format"]["name"],
            "relay_auto_compact_decision",
        )

    def test_fallback_threshold_forces_summary_without_decision_call(self) -> None:
        api = FakeManagerResponses(token_count=100)
        api.text_outputs.append("Forced working state.")
        strategy = AutoCompact(fallback_threshold=10)
        raw = [message("user", "task"), message("assistant", "work")]

        prepared = strategy.prepare(api, {"model": "task"}, raw)

        self.assertTrue(prepared.compacted)
        self.assertIn("Forced working state", prepared.input[1]["content"])
        self.assertNotIn("text", api.create_calls[0])

    def test_environment_selects_all_three_strategies(self) -> None:
        expected = {
            "context_folding": ContextFolding,
            "agent_fold": AgentFold,
            "auto_compact": AutoCompact,
        }
        for name, kind in expected.items():
            with self.subTest(name=name), patch.dict(
                "os.environ", {"RELAY_STRATEGY": name}, clear=False
            ):
                self.assertIsInstance(strategy_from_env(), kind)


if __name__ == "__main__":
    unittest.main()
