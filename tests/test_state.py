"""Harness state: instructions re-rendered in place, and state restated in injected context."""

from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Item, Kind
from relay.harnesses import ClaudeCode, DeepSeekHarness, GeminiCli, Harness, KimiCode, OpenClaw, Pi, WorkBuddy
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


class StateTests(unittest.TestCase):
    """`Harness.state`: what the harness has told the model, as it stands now."""

    def kimi(self, *extra: dict[str, Any]) -> list[dict[str, Any]]:
        date = msg("user", "<system-reminder>\nToday's date is 2026-10-02.\n</system-reminder>")
        mode = msg("user", "<system-reminder>\nAuto permission mode is active.\n</system-reminder>")
        return [msg("user", "task"), date, mode, *step(1), msg("assistant", "done"), *extra]

    def test_a_pi_section_update_keeps_the_prompt_it_updates(self) -> None:
        prompt = msg("developer", "You are an expert coding assistant operating inside pi.")
        update = msg("developer", 'Updated system prompt section "project_context":\n\n<project_context>BETA</project_context>')
        resumed = msg("developer", "You are an expert coding assistant operating inside pi. (resumed)")
        items = [prompt, msg("user", "task"), *step(1), update]
        self.assertEqual([i.ref for i in Pi().state(CODEC, items).items], [0, 4])
        self.assertEqual([i.ref for i in Pi().state(CODEC, [*items, resumed]).items], [5, 4])  # a whole new prompt replaces it

    def test_openclaws_internal_context_is_its_own(self) -> None:
        item = CODEC.classify(msg("user", "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nsessions: none"))
        self.assertIs(OpenClaw().refine(item).kind, Kind.CONTEXT)

    def test_the_latest_restated_state(self) -> None:
        plan = msg("user", "<system-reminder>\nPlan mode is active.\n</system-reminder>")
        state = KimiCode().state(CODEC, self.kimi(msg("user", "next"), plan, *step(2)))
        self.assertEqual([i.ref for i in state.items], [1, 7])  # the date, and the newest mode

    def test_without_named_state_the_context_before_the_first_action_is_kept(self) -> None:
        items = [msg("developer", "rules"), msg("user", "<system-reminder>cwd /p</system-reminder>"),
                 msg("user", "task"), *step(1), msg("user", "<system-reminder>later</system-reminder>")]
        state = Harness().state(CODEC, items)
        self.assertEqual([i.ref for i in state.items], [0, 1])

    def test_deepseek_runtime_snapshots_and_instruction_updates(self) -> None:
        items = [msg("developer", "rules"), msg("user", "task"),
                 msg("user", "<system-reminder>\nworkspace instructions from AGENTS.md: ALPHA\n</system-reminder>"),
                 msg("user", "Current runtime context. This snapshot supersedes earlier ones. policy A"), *step(1),
                 msg("user", "Current runtime context. This snapshot supersedes earlier ones. policy B"),
                 msg("user", "<system-reminder>\nUpdated instructions from: AGENTS.md\nBETA\n</system-reminder>"), *step(2)]
        state = DeepSeekHarness().state(CODEC, items)
        self.assertEqual([i.ref for i in state.items], [0, 2, 6, 7])  # original instructions, newest snapshot, update

    def test_deepseek_sub_agent_reports_are_context(self) -> None:
        for text in ("Agent f1b5 sent a message: The CODE is ember.",
                     "Background subagent f1b5 finished and will do no further work unless you send it more."):
            self.assertEqual(DeepSeekHarness().refine(CODEC.classify(msg("user", text))).kind, Kind.CONTEXT)

    def test_pi_keeps_the_system_prompt_it_appended_on_resume(self) -> None:
        items = [msg("system", "pi; codename ALPHA"), msg("user", "task"), *step(1), msg("assistant", "done"),
                 msg("system", "pi; codename BETA"), msg("user", "next"), *step(2)]
        state = Pi().state(CODEC, items)
        self.assertEqual([(i.ref, i.kind) for i in state.items], [(5, Kind.SYSTEM)])

    def test_claude_code_keeps_the_latest_mcp_instructions(self) -> None:
        def tool(i: int) -> list[dict[str, Any]]:
            return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x"}]}]
        items = [{"role": "user", "content": "task"}, {"role": "system", "content": "# Environment\ncwd /p"}, *tool(1),
                 {"role": "system", "content": "# MCP Server Instructions\n## docs"}, *tool(2),
                 {"role": "system", "content": "[SYSTEM NOTIFICATION] a background task finished"}, *tool(3)]
        state = ClaudeCode().state(AnthropicMessages(), items)
        self.assertEqual([i.ref for i in state.items], [1, 4])  # environment and MCP; the notification goes


class PlacementTests(unittest.TestCase):
    """`Harness.place`: where the state goes in a rewritten head (Codex's rule by default)."""

    system, context = Item(Kind.SYSTEM, "rules", 0), Item(Kind.CONTEXT, "<environment_context>", 1)
    first, last = Item(Kind.USER, "first task", 2), Item(Kind.USER, "second task", 6)
    summary = Item(Kind.SUMMARY, "S")

    def test_mid_turn_above_the_last_user_message_with_the_summary_last(self) -> None:
        head = Harness().place((self.first, self.last, self.summary), (self.system, self.context), mid_turn=True)
        self.assertEqual(head, (self.system, self.first, self.context, self.last, self.summary))

    def test_mid_turn_without_user_messages_just_above_the_summary(self) -> None:
        head = Harness().place((self.summary,), (self.system, self.context), mid_turn=True)
        self.assertEqual(head, (self.system, self.context, self.summary))

    def test_at_a_turn_start_after_the_summary(self) -> None:
        head = Harness().place((self.first, self.summary), (self.system, self.context), mid_turn=False)
        self.assertEqual(head, (self.system, self.first, self.summary, self.context))


class EngineStateTests(unittest.TestCase):
    """The engine puts the harness's state into any rewrite: no duplicates, nothing stale."""

    def test_a_pending_turn_keeps_its_own_newer_state_and_nothing_older_is_re_injected(self) -> None:
        engine = Engine(Compaction(threshold=300, min_gain=0))
        date = msg("user", "<system-reminder>\nToday's date is 2026-10-02.\n</system-reminder>")
        mode = msg("user", "<system-reminder>\nAuto permission mode is active.\n</system-reminder>")
        plan = msg("user", "<system-reminder>\nPlan mode is active.\n</system-reminder>")
        items = [msg("user", "task"), date, mode, *step(1), *step(2), msg("assistant", "done"), msg("user", "next"), plan]
        exchange = engine.prepare(CODEC, KimiCode(), {"model": "m", "input": items}, tenant="t", post=lambda r: SUMMARY)
        sent = json.dumps(exchange.body)
        self.assertTrue(exchange.compacted)
        self.assertEqual((sent.count("Today's date"), sent.count("Auto permission mode"), sent.count("Plan mode")), (1, 0, 1))
        self.assertTrue(sent.index("Today's date") < sent.index('"next"') < sent.index("Plan mode"))


class MaskedToolOutputTests(unittest.TestCase):
    def test_gemini_cli_masking_an_old_tool_output_keeps_the_prefix(self) -> None:
        codec = Gemini()
        call = {"role": "model", "parts": [{"functionCall": {"name": "read_file", "id": "r1", "args": {"path": "a"}}}]}
        result = {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "id": "r1",
                                                                  "response": {"output": "x" * 5000}}}]}
        masked = {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "id": "r1",
                                                                  "response": {"output": "<tool_output_masked>..."}}}]}
        task = {"role": "user", "parts": [{"text": "task"}]}
        self.assertEqual(GeminiCli().identity(codec, [task, call, result]), GeminiCli().identity(codec, [task, call, masked]))
        other = {**result, "parts": [{"functionResponse": {**result["parts"][0]["functionResponse"], "id": "r2"}}]}
        self.assertNotEqual(GeminiCli().identity(codec, [task, call, result]), GeminiCli().identity(codec, [task, call, other]))


class CacheTests(unittest.TestCase):
    """The prefix store: a stored compaction applies to every request that extends its history,
    and a request that rewrote that history is reported with where it diverged."""

    def test_hits_and_divergence(self) -> None:
        engine = Engine(Compaction(threshold=300, min_gain=0))
        history = [msg("user", "task"), *step(1), *step(2)]
        first = engine.prepare(CODEC, Harness(), {"model": "m", "input": history}, tenant="t", post=lambda r: SUMMARY)
        engine.record(first, 100)
        later = engine.prepare(CODEC, Harness(), {"model": "m", "input": [*history, *step(3, 10)]},
                               tenant="t", post=lambda r: SUMMARY)
        self.assertEqual((later.compacted, later.depth >= first.state["covered"], later.diverged), (False, True, None))
        rewritten = [*history[:3], {**history[3], "call_id": "renamed"}, *history[4:], *step(3, 10)]
        with self.assertLogs("relay", "WARNING"):
            missed = engine.prepare(CODEC, Harness(), {"model": "m", "input": rewritten}, tenant="t",
                                    post=lambda r: SUMMARY)
        self.assertEqual(missed.diverged, 3)

    def test_a_rewind_finds_the_earlier_compaction(self) -> None:
        engine = Engine(Compaction(threshold=300, min_gain=0))
        post = lambda r: SUMMARY  # noqa: E731
        early = [msg("user", "task"), *step(1), *step(2)]
        first = engine.prepare(CODEC, Harness(), {"model": "m", "input": early}, tenant="t", post=post)
        engine.record(first, 100)
        late = [*early, msg("user", "more"), *step(3), *step(4)]
        engine.record(engine.prepare(CODEC, Harness(), {"model": "m", "input": late}, tenant="t", post=post), 100)
        rewound = engine.prepare(CODEC, Harness(), {"model": "m", "input": [*early, msg("user", "instead")]},
                                 tenant="t", post=post)  # back to before the second compaction, then elsewhere
        self.assertEqual((rewound.compacted, rewound.state["covered"], rewound.diverged), (False, first.state["covered"], None))


if __name__ == "__main__":
    unittest.main()
