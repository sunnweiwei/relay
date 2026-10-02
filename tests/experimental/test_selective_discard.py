from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from relay.experimental import ContextEngine, SelectiveDiscard
from relay.experimental.strategies import strategy_from_env


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


def call(call_id: str, name: str = "exec_command") -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
    }


def output(call_id: str, text: object) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": text,
    }


class FakeResponses:
    def __init__(self, *decisions: dict) -> None:
        self.decisions = list(decisions)
        self.create_calls: list[dict] = []

    def create(self, **request):
        self.create_calls.append(deepcopy(request))
        return SimpleNamespace(output_text=json.dumps(self.decisions.pop(0)))


def decision(
    *,
    drop_steps: list[int] | None = None,
    tool_output_edits: list[dict] | None = None,
) -> dict:
    return {
        "drop_steps": drop_steps or [],
        "tool_output_edits": tool_output_edits or [],
        "reason": "test decision",
    }


class SelectiveDiscardTests(unittest.TestCase):
    def trajectory(self) -> list[dict]:
        return [
            message("user", "install and verify the project"),
            call("call_1"),
            output("call_1", "old download\nold success\n"),
            call("call_2"),
            output(
                "call_2",
                "Collecting package\nnoise one\nnoise two\nSuccessfully installed\n",
            ),
            call("call_3"),
            output("call_3", "latest verification\nall tests passed\n"),
            message("user", "continue"),
        ]

    def test_manager_drops_a_whole_step_and_selects_output_lines(self) -> None:
        api = FakeResponses(
            decision(
                drop_steps=[1],
                tool_output_edits=[
                    {
                        "call_id": "call_2",
                        "keep_ranges": [
                            {"start_line": 1, "end_line": 1},
                            {"start_line": 4, "end_line": 4},
                        ],
                    }
                ],
            )
        )
        original = self.trajectory()
        strategy = SelectiveDiscard(keep_recent_interactions=1)

        prepared = strategy.prepare(api, {"model": "task"}, original)

        self.assertTrue(prepared.compacted)
        self.assertFalse(
            any(item.get("call_id") == "call_1" for item in prepared.input)
        )
        edited = next(
            item
            for item in prepared.input
            if item.get("call_id") == "call_2"
            and item.get("type") == "function_call_output"
        )
        self.assertIn("Collecting package", edited["output"])
        self.assertIn("Successfully installed", edited["output"])
        self.assertNotIn("noise one", edited["output"])
        self.assertIn("Relay discarded lines 2-3", edited["output"])
        self.assertTrue(any(item.get("call_id") == "call_3" for item in prepared.input))
        self.assertEqual(prepared.input[-1], message("user", "continue"))
        self.assertEqual(original, self.trajectory())

        manager_call = api.create_calls[0]
        self.assertEqual(manager_call["model"], "task")
        self.assertFalse(manager_call["store"])
        self.assertEqual(
            manager_call["text"]["format"]["name"],
            "relay_selective_discard_decision",
        )
        catalog = manager_call["input"][0]["content"]
        self.assertIn('"step_id":1', catalog)
        self.assertIn('"discardable":false', catalog)
        self.assertIn("4 | Successfully installed", catalog)

    def test_empty_ranges_discard_only_the_output_body(self) -> None:
        api = FakeResponses(
            decision(
                tool_output_edits=[{"call_id": "call_1", "keep_ranges": []}]
            )
        )
        prepared = SelectiveDiscard(keep_recent_interactions=2).prepare(
            api, {"model": "task"}, self.trajectory()
        )

        call_item = next(
            item
            for item in prepared.input
            if item.get("call_id") == "call_1" and item.get("type") == "function_call"
        )
        output_item = next(
            item
            for item in prepared.input
            if item.get("call_id") == "call_1"
            and item.get("type") == "function_call_output"
        )
        self.assertEqual(call_item["call_id"], output_item["call_id"])
        self.assertIn("Relay discarded this tool output", output_item["output"])
        self.assertNotIn("old download", output_item["output"])

    def test_no_candidate_steps_means_no_manager_call(self) -> None:
        api = FakeResponses()
        original = self.trajectory()[:5]

        prepared = SelectiveDiscard(keep_recent_interactions=2).prepare(
            api, {"model": "task"}, original
        )

        self.assertFalse(prepared.compacted)
        self.assertEqual(prepared.input, original)
        self.assertEqual(api.create_calls, [])

    def test_rejects_a_protected_recent_step(self) -> None:
        api = FakeResponses(decision(drop_steps=[3]))
        with self.assertRaisesRegex(ValueError, "protected or unknown step"):
            SelectiveDiscard(keep_recent_interactions=1).prepare(
                api, {"model": "task"}, self.trajectory()
            )

    def test_rejects_an_invalid_output_range(self) -> None:
        api = FakeResponses(
            decision(
                tool_output_edits=[
                    {
                        "call_id": "call_1",
                        "keep_ranges": [{"start_line": 1, "end_line": 99}],
                    }
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "invalid line range"):
            SelectiveDiscard(keep_recent_interactions=2).prepare(
                api, {"model": "task"}, self.trajectory()
            )

    def test_engine_does_not_publish_manager_decisions_or_checkpoints(self) -> None:
        api = FakeResponses(decision(drop_steps=[1]))
        engine = ContextEngine(
            SelectiveDiscard(keep_recent_interactions=2),
            checkpoint_mode="inline",
        )
        prepared = engine.prepare(
            api, {"model": "task", "input": self.trajectory()}
        )
        response_output = [message("assistant", "done")]

        visible = engine.finalize(api, {}, prepared, response_output)

        self.assertIsNone(engine.checkpoint_item(prepared))
        self.assertEqual(visible, response_output)

    def test_compact_uses_the_same_model_driven_transformation(self) -> None:
        original = self.trajectory()
        strategy = SelectiveDiscard(keep_recent_interactions=2)
        prepared = strategy.prepare(
            FakeResponses(decision(drop_steps=[1])),
            {"model": "task"},
            original,
        )
        compacted = strategy.compact(
            FakeResponses(decision(drop_steps=[1])),
            {"model": "task"},
            original,
        )

        self.assertEqual(compacted, prepared.input)
        self.assertEqual(strategy.materialize(original), original)
        self.assertIsNone(prepared.checkpoint)

    def test_environment_selects_and_configures_strategy(self) -> None:
        env = {
            "RELAY_STRATEGY": "selective_discard",
            "RELAY_DISCARD_MODEL": "manager",
            "RELAY_DISCARD_KEEP_RECENT": "3",
            "RELAY_DISCARD_MIN_CANDIDATE_STEPS": "2",
            "RELAY_DISCARD_MAX_OUTPUT_TOKENS": "900",
        }
        with patch.dict(os.environ, env, clear=False):
            strategy = strategy_from_env()

        self.assertIsInstance(strategy, SelectiveDiscard)
        self.assertEqual(strategy.manager_model, "manager")
        self.assertEqual(strategy.keep_recent_interactions, 3)
        self.assertEqual(strategy.min_candidate_interactions, 2)
        self.assertEqual(strategy.max_manager_output_tokens, 900)


if __name__ == "__main__":
    unittest.main()
