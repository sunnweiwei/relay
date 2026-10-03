"""Codex's initial context rebuilt from the context updates in its history (relay/harnesses/codex.py)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Codex
from relay.harnesses.codex import rebuild
from relay.prompts import SUMMARY_PREFIX
from relay.protocols import AnthropicMessages, OpenAIResponses
from relay.strategies import Compaction

TRACE = json.loads((Path(__file__).parent / "traces" / "codex-0.160.0-world-state.json").read_text())


def msg(role: str, *texts: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": t} for t in texts]}


def without_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in item.items() if k != "id"} for item in items]


def rendered(request: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context = rebuild(request)
    assert context is not None
    return without_ids([request[part] if isinstance(part, int) else part for part in context.context])


TOOLS = {"type": "additional_tools", "role": "developer", "tools": []}
BASE = msg("developer", "You are Codex.")
BUNDLE = msg("developer", "<permissions instructions>\nsandboxed\n</permissions instructions>",
             "<collaboration_mode>default</collaboration_mode>")
AGENTS = "# AGENTS.md instructions for /a\n\n<INSTRUCTIONS>\nbe brief\n</INSTRUCTIONS>"
ENV = "<environment_context>\n  <cwd>/a</cwd>\n</environment_context>"
TASK, CALL = msg("user", "fix it"), {"type": "function_call", "call_id": "c1", "name": "sh", "arguments": "{}"}
OUTPUT = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
START = [TOOLS, BASE, BUNDLE, msg("user", AGENTS, ENV), TASK, CALL, OUTPUT]


class RecordedCompactionTests(unittest.TestCase):
    """Native Codex compactions recorded after changing directory, AGENTS.md and model, and after
    approving command prefixes."""

    def test_rebuilt_context_matches_codex_byte_for_byte(self) -> None:
        for name, layout in (("mid_turn", slice(3, -2)), ("turn_start", slice(4, -1)),
                             ("approved_prefixes", slice(3, -2))):
            case = TRACE["cases"][name]
            with self.subTest(name):
                self.assertEqual(rendered(case["request"]), without_ids(case["expected"][layout]))

    def test_compacted_request_matches_codex_except_for_the_summary(self) -> None:
        codec = OpenAIResponses()
        summary = {"status": "completed", "output": [msg("assistant", "S")]}
        for name in TRACE["cases"]:
            case = TRACE["cases"][name]
            engine = Engine(Compaction(threshold=1_000, min_gain=0))
            exchange = engine.prepare(codec, Codex(), {"model": "gpt-6-sol", "input": case["request"]},
                                      tenant="t", post=lambda body: (200, summary))
            expected = [  # long texts in the trace are shortened, so match the summary by its opening
                codec.user_message(f"{SUMMARY_PREFIX}\nS") if item.get("content", [{}])[0].get("text", "")
                .startswith(SUMMARY_PREFIX[:40]) else item
                for item in case["expected"]
            ]
            with self.subTest(name):
                self.assertTrue(exchange.compacted)
                self.assertEqual(without_ids(exchange.body["input"]), without_ids(expected))


class RebuildRuleTests(unittest.TestCase):
    def test_without_updates_the_first_rendering_is_kept_by_reference(self) -> None:
        context = rebuild(START)
        self.assertEqual((context.pinned, context.context, context.current), ((0, 1), (2, 3), False))

    def test_latest_section_update_replaces_it_in_place(self) -> None:
        moved = "<environment_context>\n  <cwd>/b</cwd>\n</environment_context>"
        plan = "<collaboration_mode>plan</collaboration_mode>"
        history = [*START, msg("developer", plan), msg("user", moved), CALL, OUTPUT]
        self.assertEqual(rendered(history), [BUNDLE | {"content": BUNDLE["content"][:1] + [
            {"type": "input_text", "text": plan}]}, msg("user", AGENTS, moved)])

    def test_agents_md_replacement_and_removal_notices(self) -> None:
        replaced = AGENTS.replace("be brief", "These AGENTS.md instructions replace all previously provided "
                                  "AGENTS.md instructions.\n\nbe thorough")
        removed = "# AGENTS.md instructions\n\n<INSTRUCTIONS>\nThe previously provided AGENTS.md instructions " \
                  "no longer apply.\n</INSTRUCTIONS>"
        self.assertEqual(rendered([*START, msg("user", replaced)])[1],
                         msg("user", AGENTS.replace("be brief", "be thorough"), ENV))
        self.assertEqual(rendered([*START, msg("user", removed)])[1], msg("user", ENV))

    def test_new_section_is_inserted_in_render_order(self) -> None:
        skills = "<skills_instructions>\n## Skills\n</skills_instructions>"
        bundle = rendered([*START, msg("developer", skills)])[0]
        texts = [part["text"] for part in bundle["content"]]
        self.assertEqual(texts, [skills, *(part["text"] for part in BUNDLE["content"])])

    def test_model_switch_only_at_the_start_of_the_switching_turn(self) -> None:
        switch = "<model_switch>\nnew model\n</model_switch>"
        new_turn = [*START, msg("assistant", "done"), msg("developer", switch), msg("user", "again")]
        context = rebuild(new_turn)
        self.assertTrue(context.current)
        self.assertEqual(rendered(new_turn)[0]["content"][0]["text"], switch)
        self.assertEqual(rendered([*new_turn, CALL, OUTPUT]), rendered(START))

    def test_approved_prefixes_join_the_permissions_list_in_codex_order(self) -> None:
        listed = "## Approved command prefixes\nThe following prefix rules have already been approved: "
        permissions = f'<permissions instructions>\nAsk first.\n\n{listed}- ["pwd"]\n- ["git", "log"]\n' \
                      "The writable roots are `/a`.\n</permissions instructions>"
        bundle = msg("developer", permissions)
        saved = msg("developer", 'Approved command prefix saved:\n- ["ls"]\n- ["python3", "-m", "pytest"]')
        history = [TOOLS, BASE, bundle, msg("user", AGENTS, ENV), TASK, CALL, OUTPUT, saved, CALL, OUTPUT]
        merged = permissions.replace('- ["pwd"]\n- ["git", "log"]',
                                     '- ["ls"]\n- ["pwd"]\n- ["git", "log"]\n- ["python3", "-m", "pytest"]')
        self.assertEqual(rendered(history)[0], msg("developer", merged))

    def test_saved_prefixes_leave_a_permissions_text_without_a_list_alone(self) -> None:
        saved = msg("developer", 'Approved command prefix saved:\n- ["ls"]')
        self.assertEqual(rendered([*START, saved]), rendered(START))
        truncated = msg("developer", '<permissions instructions>\n## Approved command prefixes\nThe following prefix '
                        'rules have already been approved: - ["pwd"]...\n[Some commands were truncated]\n'
                        "</permissions instructions>")
        history = [TOOLS, BASE, truncated, msg("user", ENV), TASK, CALL, OUTPUT, saved]
        self.assertEqual(rendered(history), [without_ids([truncated])[0], msg("user", ENV)])

    def test_app_server_clients_are_detected_by_codex_turn_metadata(self) -> None:
        self.assertTrue(Codex().matches({"originator": "my-ide", "x-codex-turn-metadata": "{}"}))
        self.assertFalse(Codex().matches({"originator": "my-ide"}))

    def test_other_protocols_keep_the_default(self) -> None:
        self.assertIsNone(Codex().initial_context(AnthropicMessages(), START))
        self.assertIsNone(rebuild([TOOLS, BASE, TASK]))  # no model action yet


if __name__ == "__main__":
    unittest.main()
