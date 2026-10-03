"""Harness state: instructions re-rendered in place, and state restated in injected context."""

from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Kind
from relay.harnesses import ClaudeCode, DeepSeekHarness, GeminiCli, Harness, KimiCode, Pi, WorkBuddy
from relay.protocols import AnthropicMessages, Gemini, OpenAIResponses
from relay.strategies import Compaction

CODEC = OpenAIResponses()
SUMMARY = (200, {"status": "completed", "output": [{"type": "message", "role": "assistant",
                                                     "content": [{"type": "output_text", "text": "S"}]}]})


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def step(i: int, size: int = 800) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "x" * size}]


class ReRenderedInstructionsTests(unittest.TestCase):
    def test_a_rewritten_system_message_keeps_the_compaction_and_is_forwarded_new(self) -> None:
        engine, summaries = Engine(Compaction(threshold=300, min_gain=0)), []
        post = lambda request: (summaries.append(request), SUMMARY)[1]
        history = [msg("user", "task"), *step(1), *step(2)]
        first = engine.prepare(CODEC, Harness(), {"model": "m", "input": [msg("system", "codename ALPHA"), *history]},
                               tenant="t", post=post)
        self.assertTrue(first.compacted)
        engine.record(first, 100)  # the compacted request is small
        # OpenCode re-renders AGENTS.md into its system message on the next turn
        later = engine.prepare(CODEC, Harness(), {"model": "m", "input": [msg("system", "codename BETA"), *history, *step(3, 10)]},
                               tenant="t", post=post)
        self.assertFalse(later.compacted)
        self.assertEqual(len(summaries), 1)
        self.assertIn("codename BETA", json.dumps(later.body))
        self.assertNotIn("codename ALPHA", json.dumps(later.body))

    def test_injected_blocks_in_the_first_user_message_do_not_change_its_identity(self) -> None:
        def contents(gemini_md: str) -> list[dict[str, Any]]:
            context = f"<session_context>\\nToday's date is Saturday.\\n{gemini_md}\\n</session_context>"
            return [{"role": "user", "parts": [{"text": context}, {"text": "fix the bug"}]},
                    {"role": "model", "parts": [{"text": "ok"}]}]
        codec, harness = Gemini(), GeminiCli()
        self.assertEqual(harness.identity(codec, contents("ALPHA")), harness.identity(codec, contents("BETA")))
        other = [{"role": "user", "parts": [{"text": "<session_context>x</session_context>"}, {"text": "other task"}]}]
        self.assertNotEqual(harness.identity(codec, contents("ALPHA"))[0], harness.identity(codec, other)[0])
        self.assertEqual(harness.refine(codec.classify(contents("ALPHA")[0])).kind, Kind.USER)

    def test_a_slash_command_message_with_the_next_request_is_a_user_message(self) -> None:
        reminder = "<system-reminder>\ncodename BETA\n</system-reminder>"
        command = ("<local-command-caveat>Caveat</local-command-caveat><command-name>/compact</command-name>"
                   "<command-message>compact</command-message><command-args></command-args>"
                   "<local-command-stdout>Compacted</local-command-stdout>")
        mixed = {"role": "user", "content": [{"type": "text", "text": t} for t in (reminder, command, "now read file_5")]}
        only = {"role": "user", "content": [{"type": "text", "text": t} for t in (reminder, command)]}
        codec = AnthropicMessages()
        self.assertEqual(ClaudeCode().refine(codec.classify(mixed)).kind, Kind.USER)
        self.assertEqual(ClaudeCode().refine(codec.classify(only)).kind, Kind.CONTEXT)

    def test_reminders_with_attributes_are_injected_context(self) -> None:
        memory = msg("user", '<system-reminder data-role="memory"><memory>notes</memory></system-reminder>')
        self.assertEqual(WorkBuddy().refine(CODEC.classify(memory)).kind, Kind.CONTEXT)
        rules = "<always_applied_workspace_rules>codename ALPHA</always_applied_workspace_rules>"
        first = msg("user", f"{rules}<user_query>fix it</user_query>")
        rewritten = msg("user", f"{rules.replace('ALPHA', 'BETA')}<user_query>fix it</user_query>")
        self.assertEqual(WorkBuddy().identity(CODEC, [first]), WorkBuddy().identity(CODEC, [rewritten]))


class StateKeyTests(unittest.TestCase):
    def kimi(self, *extra: dict[str, Any]) -> list[dict[str, Any]]:
        date = msg("user", "<system-reminder>\\nToday's date is 2026-10-02.\\n</system-reminder>")
        mode = msg("user", "<system-reminder>\\nAuto permission mode is active.\\n</system-reminder>")
        return [msg("user", "task"), date, mode, *step(1), msg("assistant", "done"), *extra]

    def test_the_latest_restated_state_is_re_injected(self) -> None:
        plan = msg("user", "<system-reminder>\\nPlan mode is active.\\n</system-reminder>")
        items = self.kimi(msg("user", "next"), plan, *step(2))
        context = KimiCode().initial_context(CODEC, items)
        self.assertEqual([i.ref for i in context.items], [1, 7])  # the date, and the newest mode

    def test_a_pending_turn_keeps_its_own_state(self) -> None:
        plan = msg("user", "<system-reminder>\\nPlan mode is active.\\n</system-reminder>")
        context = KimiCode().initial_context(CODEC, self.kimi(msg("user", "next"), plan))
        self.assertEqual([i.ref for i in context.items], [1, 2])  # the new turn carries the plan mode itself

    def test_profiles_without_state_keep_the_default(self) -> None:
        self.assertIsNone(Harness().initial_context(CODEC, self.kimi()))

    def test_deepseek_runtime_snapshots_and_instruction_updates(self) -> None:
        items = [msg("developer", "rules"), msg("user", "task"),
                 msg("user", "<system-reminder>\\nworkspace instructions from AGENTS.md: ALPHA\\n</system-reminder>"),
                 msg("user", "Current runtime context. This snapshot supersedes earlier ones. policy A"), *step(1),
                 msg("user", "Current runtime context. This snapshot supersedes earlier ones. policy B"),
                 msg("user", "<system-reminder>\\nUpdated instructions from: AGENTS.md\\nBETA\\n</system-reminder>"), *step(2)]
        context = DeepSeekHarness().initial_context(CODEC, items)
        self.assertEqual([i.ref for i in context.items], [0, 2, 6, 7])  # original instructions, newest snapshot, update

    def test_deepseek_sub_agent_reports_are_context(self) -> None:
        report = msg("user", "Agent f1b5 sent a message: The CODE is ember.")
        self.assertEqual(DeepSeekHarness().refine(CODEC.classify(report)).kind, Kind.CONTEXT)

    def test_pi_keeps_the_system_prompt_it_appended_on_resume(self) -> None:
        items = [msg("system", "pi; codename ALPHA"), msg("user", "task"), *step(1), msg("assistant", "done"),
                 msg("system", "pi; codename BETA"), msg("user", "next"), *step(2)]
        context = Pi().initial_context(CODEC, items)
        self.assertEqual([(i.ref, i.kind) for i in context.items], [(5, Kind.SYSTEM)])

    def test_claude_code_keeps_the_latest_mcp_instructions(self) -> None:
        def tool(i: int) -> list[dict[str, Any]]:
            return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x"}]}]
        items = [{"role": "user", "content": "task"}, {"role": "system", "content": "# Environment\\ncwd /p"}, *tool(1),
                 {"role": "system", "content": "# MCP Server Instructions\\n## docs"}, *tool(2),
                 {"role": "system", "content": "[SYSTEM NOTIFICATION] a background task finished"}, *tool(3)]
        context = ClaudeCode().initial_context(AnthropicMessages(), items)
        self.assertEqual([i.ref for i in context.items], [1, 4])  # environment and MCP; the notification goes


if __name__ == "__main__":
    unittest.main()
