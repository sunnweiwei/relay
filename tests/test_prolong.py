from __future__ import annotations

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from relay import ContextEngine, PrefixCheckpointCache, ProLong
from relay.strategies import strategy_from_env
from relay.strategies.prolong import PROLONG_CONTEXT_PREFIX


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


class FakeInputTokens:
    def __init__(self, owner: FakeProLongResponses) -> None:
        self.owner = owner

    def count(self, **request):
        self.owner.count_calls.append(deepcopy(request))
        return SimpleNamespace(input_tokens=40 * len(request.get("input") or []))


class FakeProLongResponses:
    def __init__(self, *, compact_manager: bool = False) -> None:
        self.input_tokens = FakeInputTokens(self)
        self.manager_calls: list[dict] = []
        self.count_calls: list[dict] = []
        self.compact_manager = compact_manager

    def create(self, **request):
        self.manager_calls.append(deepcopy(request))
        number = len(self.manager_calls)
        if request["tool_choice"] == "required":
            output = []
            if self.compact_manager and number == 1:
                output.append(
                    {
                        "id": "cmp_manager_1",
                        "type": "compaction",
                        "encrypted_content": "opaque-manager-compaction",
                    }
                )
            output.extend(
                [
                    {
                        "id": f"rs_manager_{number}",
                        "type": "reasoning",
                        "encrypted_content": "opaque-manager-reasoning",
                        "summary": [],
                    },
                    {
                        "id": f"fc_manager_{number}",
                        "type": "function_call",
                        "call_id": f"call_manager_{number}",
                        "name": "log_read",
                        "arguments": json.dumps({"start_line": 1, "end_line": 30}),
                        "status": "completed",
                    },
                ]
            )
            return SimpleNamespace(output_text="", output=output)
        value = {"context": "Task is active; earlier inspection found alpha.py."}
        return SimpleNamespace(
            output_text=json.dumps(value),
            output=[
                {
                    "id": f"msg_manager_{number}",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(value),
                        }
                    ],
                }
            ],
        )


class ProLongTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = [
            message("user", "Fix alpha.py"),
            message("assistant", "Located alpha.py"),
            message("user", "Inspect the parser"),
            message("assistant", "Parser uses parse_value"),
            message("user", "Apply the fix"),
            message("assistant", "Patched parse_value"),
            message("user", "Now run verification"),
        ]
        self.tools = [
            {
                "type": "function",
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            }
        ]
        self.request = {
            "model": "task-model",
            "instructions": "You are the main coding agent.",
            "tools": self.tools,
            "input": self.raw,
        }
        self.strategy = ProLong(
            manager_model="small-manager",
            context_threshold=250,
            manager_compact_threshold=220,
            max_manager_steps=4,
        )

    def test_private_manager_programmatically_reads_lossless_log(self) -> None:
        api = FakeProLongResponses()

        prepared = self.strategy.prepare(api, self.request, self.raw)

        self.assertEqual(len(api.manager_calls), 2)
        first, second = api.manager_calls
        self.assertEqual(first["model"], "small-manager")
        self.assertEqual(first["tool_choice"], "required")
        self.assertEqual(second["tool_choice"], "auto")
        self.assertEqual(
            {tool["name"] for tool in first["tools"]},
            {"log_read", "log_grep", "log_python"},
        )
        self.assertEqual(
            first["context_management"],
            [{"type": "compaction", "compact_threshold": 220}],
        )
        inherited = json.loads(first["input"][1]["content"])
        self.assertEqual(
            inherited["main_agent"]["instructions"],
            self.request["instructions"],
        )
        self.assertEqual(inherited["main_agent"]["tools"][0]["name"], "bash")
        self.assertEqual(inherited["active_context"], self.raw)

        second_types = [item.get("type") for item in second["input"]]
        self.assertIn("reasoning", second_types)
        self.assertIn("function_call", second_types)
        self.assertIn("function_call_output", second_types)
        self.assertIn(PROLONG_CONTEXT_PREFIX, prepared.input[-1]["content"])
        self.assertEqual(prepared.input[:-1], self.raw)
        self.assertFalse(
            any(
                item.get("type") in {"function_call", "function_call_output"}
                and str(item.get("call_id", "")).startswith("call_manager")
                for item in prepared.input
            )
        )
        assert prepared.checkpoint is not None
        state = self.strategy._state(prepared.checkpoint, self.raw)
        self.assertEqual(state.log_items, self.raw)
        self.assertTrue(
            any(item.get("type") == "reasoning" for item in state.manager_input)
        )

    def test_manager_is_available_from_the_first_short_turn(self) -> None:
        api = FakeProLongResponses()
        short = [message("user", "Inspect a small but structured observation")]

        prepared = self.strategy.prepare(
            api,
            {**self.request, "input": short},
            short,
        )

        self.assertEqual(len(api.manager_calls), 2)
        self.assertIn(PROLONG_CONTEXT_PREFIX, prepared.input[-1]["content"])

    def test_cache_resumes_both_model_prefixes_and_appends_new_log_items(self) -> None:
        api = FakeProLongResponses()
        cache = PrefixCheckpointCache(secret=b"prolong-test")
        engine = ContextEngine(
            self.strategy,
            checkpoint_mode="cache",
            checkpoint_cache=cache,
        )
        first = engine.prepare(api, self.request, cache_namespace="tenant-a")

        extended = [
            *self.raw,
            message("assistant", "Focused tests pass"),
            message("user", "Continue"),
        ]
        second_request = {**self.request, "input": extended}
        second = engine.prepare(api, second_request, cache_namespace="tenant-a")

        self.assertEqual(len(api.manager_calls), 4)
        self.assertEqual(second.input[: len(first.input)], first.input)
        self.assertEqual(second.input[len(first.input) : -1], extended[len(self.raw) :])
        self.assertEqual(
            api.manager_calls[2]["input"][: len(api.manager_calls[1]["input"])],
            api.manager_calls[1]["input"],
        )
        assert second.checkpoint is not None
        state = self.strategy._state(second.checkpoint, extended)
        self.assertEqual(state.log_items, extended)
        self.assertGreaterEqual(cache.stats().hits, 1)

    def test_main_tools_stay_private_and_native_compaction_is_enabled(self) -> None:
        api = FakeProLongResponses()
        engine = ContextEngine(self.strategy, checkpoint_mode="cache")

        prepared = engine.prepare(api, self.request, cache_namespace="tenant-a")
        forwarded = engine.upstream_request(self.request, prepared)

        self.assertEqual(forwarded["instructions"], self.request["instructions"])
        self.assertEqual(forwarded["tools"], self.tools)
        self.assertFalse(
            any(
                tool.get("name") in {"log_read", "log_grep", "log_python"}
                for tool in forwarded["tools"]
            )
        )
        self.assertEqual(
            forwarded["context_management"],
            [{"type": "compaction", "compact_threshold": 250}],
        )

    def test_native_compaction_alias_preserves_log_when_client_prunes(self) -> None:
        api = FakeProLongResponses()
        cache = PrefixCheckpointCache(secret=b"prolong-native-compaction-test")
        engine = ContextEngine(
            self.strategy,
            checkpoint_mode="cache",
            checkpoint_cache=cache,
        )
        prepared = engine.prepare(api, self.request, cache_namespace="tenant-a")
        compacted = {
            "id": "cmp_official_1",
            "type": "compaction",
            "encrypted_content": "opaque-main-compaction",
        }
        assistant = message("assistant", "Compacted continuation")
        engine.finalize(
            api,
            self.request,
            prepared,
            [compacted, assistant],
        )

        pruned = [compacted, assistant, message("user", "Continue after compact")]
        resumed = engine.prepare(
            api,
            {**self.request, "input": pruned},
            cache_namespace="tenant-a",
        )

        assert resumed.checkpoint is not None
        state = self.strategy._state(resumed.checkpoint, pruned)
        self.assertEqual(
            state.log_items,
            [*self.raw, compacted, assistant, pruned[-1]],
        )
        self.assertEqual(resumed.input[0], compacted)
        self.assertGreaterEqual(cache.stats().hits, 1)

    def test_inline_checkpoint_can_follow_a_native_compaction_item(self) -> None:
        api = FakeProLongResponses()
        engine = ContextEngine(self.strategy, checkpoint_mode="inline")
        prepared = engine.prepare(api, self.request)
        compacted = {
            "id": "cmp_official_inline",
            "type": "compaction",
            "encrypted_content": "opaque-main-compaction",
        }
        assistant = message("assistant", "Compacted continuation")

        visible = engine.finalize(
            api,
            self.request,
            prepared,
            [compacted, assistant],
        )

        self.assertEqual(visible[0], compacted)
        self.assertEqual(visible[1].type, "compaction")
        resumed_raw = [*self.raw, *visible, message("user", "Continue")]
        resumed = engine.prepare(
            api,
            {**self.request, "input": resumed_raw},
        )
        self.assertEqual(resumed.input[0], compacted)

    def test_manager_native_compaction_item_replaces_its_old_prefix(self) -> None:
        api = FakeProLongResponses(compact_manager=True)

        prepared = self.strategy.prepare(api, self.request, self.raw)

        assert prepared.checkpoint is not None
        state = self.strategy._state(prepared.checkpoint, self.raw)
        self.assertEqual(state.manager_input[0]["type"], "compaction")
        self.assertEqual(
            state.manager_input[0]["encrypted_content"],
            "opaque-manager-compaction",
        )
        self.assertFalse(
            any(item.get("role") == "developer" for item in state.manager_input)
        )

    def test_missing_checkpoint_rebuilds_equivalent_context(self) -> None:
        first = self.strategy.prepare(FakeProLongResponses(), self.request, self.raw)
        second = self.strategy.prepare(FakeProLongResponses(), self.request, self.raw)
        self.assertEqual(first.input, second.input)

    def test_python_tool_reads_the_structured_log(self) -> None:
        result = self.strategy._log_python(
            "alpha\nbeta\nalpha", "print(open('logs.txt').read().count('alpha'))"
        )
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"].strip(), "2")

    def test_environment_selects_prolong(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "RELAY_STRATEGY": "prolong",
                "RELAY_PROLONG_MODEL": "small-manager",
            },
            clear=False,
        ):
            selected = strategy_from_env()
        self.assertIsInstance(selected, ProLong)
        self.assertEqual(selected.manager_model, "small-manager")


if __name__ == "__main__":
    unittest.main()
