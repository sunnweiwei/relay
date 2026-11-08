"""Headroom as the `headroom` strategy, as its proxy compresses Codex's requests: the units it is
asked about (an output, each part of one, a long message but the newest user message), each
written back in place, and what was forwarded never compressed again."""

from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Item, Kind, Request
from relay.harnesses import Harness
from relay.harnesses.claude_code import ClaudeCode
from relay.protocols import AnthropicMessages, Gemini, OpenAIResponses
from relay.prompts import SUMMARY_PREFIX
from relay.strategies import Compaction, Headroom
from relay.strategies.headroom import _calls, _commands, _digest

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]
SUMMARY = (200, {"status": "completed", "output": [{"type": "message", "role": "assistant",
                                                     "content": [{"type": "output_text", "text": "S"}]}]})
NOTES = "notes " * 1400  # (8,400 bytes: a message Headroom's proxy compresses)
HEADER = "Script completed\nWall time 0.1 seconds\nOutput:\n"  # (Codex's code mode: the first part of an output)
SCRIPT = ('const r = await tools.exec_command({cmd:"./serve_log 1", workdir:"/project", yield_time_ms:10000});\n'
          "text(r.output);\ntext(`\\n[exit_code=${r.exit_code}]`);\n")


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def run(i: int, command: str, output: str) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": output}]


def script(i: int, *printed: str) -> list[dict[str, Any]]:
    return [{"type": "custom_tool_call", "call_id": f"c{i}", "name": "exec", "input": SCRIPT},
            {"type": "custom_tool_call_output", "call_id": f"c{i}",
             "output": [{"type": "input_text", "text": text} for text in (HEADER, *printed)]}]


class FakeHeadroom:
    """Headroom's /v1/compress, compressing each message past the frozen ones to its size."""

    def __init__(self, failures: int = 0) -> None:
        self.bodies, self.failures = [], failures

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        if self.failures:
            self.failures -= 1
            raise ValueError("Headroom answered 503: compression_error")
        frozen = body["config"]["frozen_message_count"]
        return {"messages": [m if n < frozen or not isinstance(m["content"], str) else {**m, "content": f"[{len(m['content'])} chars]"}
                             for n, m in enumerate(body["messages"])],
                "tokens_before": 300, "tokens_saved": 200, "transforms_applied": ["router:log:0.01"]}


class HeadroomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.headroom = FakeHeadroom()
        self.engine = Engine(Headroom(post=self.headroom))

    def send(self, items: list[dict[str, Any]], codec: Any = CODEC, harness: Any = HARNESS) -> list[dict[str, Any]]:
        field = {"openai_responses": "input", "gemini": "contents"}.get(codec.name, "messages")
        body = {"model": "gpt-6-luna", "tools": TOOLS, field: items}
        return self.engine.prepare(codec, harness, body, tenant="t", post=lambda r: SUMMARY).body[field]

    def test_an_output_goes_compressed_headroom_seeing_it_after_its_call_and_counting_with_the_requests_model(self) -> None:
        sent = self.send([msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 600)])
        (body,) = self.headroom.bodies  # (the newest user message is not asked about)
        self.assertEqual(body["messages"], [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1_0", "type": "function", "function": {"name": "sh", "arguments": '{"cmd": "cat app.log"}'}}]},
            {"role": "tool", "tool_call_id": "call_1_0", "content": "x" * 600}])
        self.assertEqual((body["model"], body["config"]), ("gpt-6-luna", {"frozen_message_count": 0}))
        self.assertEqual(sent[-1], {"type": "function_call_output", "call_id": "c1", "output": "[600 chars]"})

    def test_new_outputs_go_after_those_forwarded_frozen_and_only_from_512_bytes(self) -> None:
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 600)]
        self.send(first)
        sent = self.send([*first, *run(2, "ls", "z" * 511), *run(3, "cat test.log", "y" * 800)])
        body = self.headroom.bodies[-1]
        self.assertEqual([m["content"] for m in body["messages"]], [None, "[600 chars]", None, "y" * 800])
        self.assertEqual(body["config"]["frozen_message_count"], 2)  # what the provider has, which a new one may point at
        self.assertEqual([item.get("output") for item in sent[1:]], [None, "[600 chars]", None, "z" * 511, None, "[800 chars]"])

    def test_the_newest_user_message_stays_and_goes_compressed_once_a_newer_one_comes(self) -> None:
        first = [msg("user", NOTES), *run(1, "pytest", "x" * 600)]
        self.assertEqual(self.send(first)[0], msg("user", NOTES))
        self.assertEqual(len(self.headroom.bodies), 1)  # (the output alone)
        turn = [*first, msg("assistant", "Done. " + NOTES), msg("user", "short " * 10), msg("user", "next")]
        sent = self.send(turn)
        self.assertCountEqual([body["messages"] for body in self.headroom.bodies[1:]],
                              [[{"role": "user", "content": NOTES}], [{"role": "assistant", "content": "Done. " + NOTES}]])
        self.assertEqual(sent[0]["content"], [{"type": "input_text", "text": "[8400 chars]"}])
        self.assertEqual(sent[3]["content"], [{"type": "output_text", "text": "[8406 chars]"}])
        self.assertEqual(sent[4:], turn[4:])  # (under 8192 bytes; the newest)
        later = self.send([*turn, *run(2, "ls", "y" * 600)])
        self.assertEqual(later[:6], sent)
        self.assertEqual(self.headroom.bodies[-1]["messages"][-1]["content"], "y" * 600)

    def test_of_a_user_message_what_the_user_typed_is_compressed_and_the_harnesss_blocks_stay(self) -> None:
        reminder = "<system-reminder>\ncodename BETA\n</system-reminder>"
        user = {"role": "user", "content": [{"type": "text", "text": reminder}, {"type": "text", "text": NOTES}]}
        call = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}}]}
        results = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 600}]}
        turn = [user, call, results, {"role": "assistant", "content": "Done."}, {"role": "user", "content": "next"}]
        sent = self.send(turn, AnthropicMessages(), ClaudeCode())
        self.assertIn([{"role": "user", "content": NOTES.strip()}], [body["messages"] for body in self.headroom.bodies])
        (text,) = [block["text"] for block in sent[0]["content"]]
        self.assertEqual(text.strip(), f"{reminder}\n[8399 chars]")

    def test_in_codex_code_mode_each_part_goes_alone_after_the_scripts_command_and_back_in_its_place(self) -> None:
        log = "2026-11-04 10:00:00 INFO request served\n" * 20
        first = [msg("user", "read the logs"), *script(1, log, "\n[exit_code=0]")]
        sent = self.send(first)
        (body,) = self.headroom.bodies  # (the header and the exit code are under 512 bytes)
        self.assertEqual(body["messages"], [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1_0", "type": "function", "function": {"name": "exec", "arguments": '{"cmd": "./serve_log 1"}'}}]},
            {"role": "tool", "tool_call_id": "call_1_0", "content": log}])
        self.assertEqual([part["text"] for part in sent[-1]["output"]], [HEADER, f"[{len(log)} chars]", "\n[exit_code=0]"])
        again = self.send([*first, *script(2, log, "\n[exit_code=0]")])
        self.assertEqual(self.headroom.bodies[-1]["messages"][1:], [{"role": "tool", "tool_call_id": "call_3_0", "content": log}])
        self.assertEqual(again[:3], sent)  # (a part never points at an earlier one: none is shown to it)
        self.assertEqual(again[-1]["output"], sent[-1]["output"])

    def test_each_message_goes_as_headroom_compresses_it_the_first_time_and_a_retry_asks_nothing(self) -> None:
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 600)]
        self.send(first)
        self.send(first)  # (the harness retries)
        self.assertEqual(len(self.headroom.bodies), 1)

    def test_where_headroom_does_not_answer_the_results_go_as_they_are_and_stay_so(self) -> None:
        self.headroom.failures = 1
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 600)]
        self.assertEqual(self.send(first), first)
        sent = self.send([*first, *run(2, "pytest", "y" * 800)])
        self.assertEqual(self.headroom.bodies[-1]["config"]["frozen_message_count"], 2)
        self.assertEqual([item.get("output") for item in sent[1:]], [None, "x" * 600, None, "[800 chars]"])

    def test_on_anthropic_a_messages_results_go_as_one_and_of_several_tools_as_they_are(self) -> None:
        def turn(*tools: str) -> list[dict[str, Any]]:
            uses = [{"type": "tool_use", "id": f"t{k}", "name": tool, "input": {"path": f"f{k}"}} for k, tool in enumerate(tools)]
            results = [{"type": "tool_result", "tool_use_id": f"t{k}", "content": f"{k}" * 500} for k in range(len(tools))]
            return [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": uses},
                    {"role": "user", "content": results}]

        sent = self.send(turn("Bash", "Bash"), AnthropicMessages())
        self.assertEqual([m.get("tool_call_id") for m in self.headroom.bodies[-1]["messages"] if m["role"] == "tool"],
                         ["call_1_0"])
        self.assertEqual([part["content"] for part in sent[-1]["content"]], [[{"type": "text", "text": "[1001 chars]"}], ""])
        sent = self.send(turn("Read", "Bash"), AnthropicMessages())  # (Headroom keeps a Read's output: Relay can't split it)
        self.assertEqual(sent[-1], turn("Read", "Bash")[-1])

    def test_on_gemini_the_output_goes_compressed_where_it_was(self) -> None:
        call = {"role": "model", "parts": [{"functionCall": {"id": "r1", "name": "run_shell_command", "args": {"command": "pytest"}}}]}
        result = {"role": "user", "parts": [{"functionResponse": {"id": "r1", "name": "run_shell_command",
                                                                  "response": {"output": "x" * 600}}}]}
        sent = self.send([{"role": "user", "parts": [{"text": "fix the bug"}]}, call, result], Gemini())
        assistant, tool = self.headroom.bodies[-1]["messages"]
        self.assertEqual(assistant["tool_calls"][0]["function"], {"name": "run_shell_command", "arguments": '{"command": "pytest"}'})
        self.assertEqual(tool["content"], "x" * 600)
        self.assertEqual(sent[-1]["parts"][0]["functionResponse"], {"id": "r1", "name": "run_shell_command",
                                                                    "response": {"output": "[600 chars]"}})

    def test_past_the_window_compaction_keeps_it_within_and_headroom_goes_on_from_there(self) -> None:
        self.engine = Engine(Headroom(fallback=Compaction(threshold=300, min_gain=0), post=self.headroom))
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 100)]
        self.send(first)
        over = [*first, *run(2, "cat test.log", "y" * 3000)]
        asked = len(self.headroom.bodies)
        sent = self.send(over)
        self.assertEqual(len(self.headroom.bodies), asked)  # (compacted, not compressed)
        self.assertTrue(any(SUMMARY_PREFIX in CODEC.classify(item).text for item in sent))
        later = self.send([*over, *run(3, "ls", "z" * 600)])
        body = self.headroom.bodies[-1]
        self.assertEqual(body["config"]["frozen_message_count"], 0)  # (no output left before it)
        self.assertEqual(later[:len(sent)], sent)
        self.assertEqual(later[-1]["output"], "[600 chars]")

    def test_a_request_the_upstream_finds_too_long_is_compacted_and_the_user_message_held_back_goes_compressed(self) -> None:
        strategy = Headroom(fallback=Compaction(threshold=10**9), post=self.headroom)
        items = (Item(Kind.USER, NOTES), Item(Kind.TOOL_CALL, 'sh({"cmd": "ls"})'), Item(Kind.TOOL_RESULT, "x" * 600))
        state = {"sent": 3, "last": _digest(items[-1]), "held": _digest(items[0])}
        request = Request(items, items, frozenset({0, 1, 3}), 1000, 200_000, force=True, state=state)

        class Summarizer:
            def summarize(self, cut: int, prompt: str) -> str:
                return "S"

        context = strategy.plan(request, Summarizer())
        self.assertEqual([(item.kind, item.text) for item in context.items],
                         [(Kind.USER, "[8400 chars]"), (Kind.SUMMARY, f"{SUMMARY_PREFIX}\nS")])  # (no longer the newest)
        self.assertEqual([body["messages"] for body in self.headroom.bodies], [[{"role": "user", "content": NOTES}]])

    def test_a_history_not_as_forwarded_is_compressed_whole_and_the_settings_go_along(self) -> None:
        items = (Item(Kind.USER, "fix the bug"), Item(Kind.TOOL_CALL, 'sh({"cmd": "cat app.log"})'),
                 Item(Kind.TOOL_RESULT, "x" * 600))
        request = Request(items, items, frozenset({0, 1, 3}), 1000, 200_000, state={"sent": 3, "last": "another"},
                          model="gpt-6-luna")
        context = Headroom(mode="ccr", post=self.headroom).plan(request, None)
        body = self.headroom.bodies[-1]
        self.assertEqual((body["model"], body["config"], body["token_budget"]),
                         ("gpt-6-luna", {"frozen_message_count": 0, "mode": "ccr"}, 200_000))
        self.assertEqual(context.items[-1].text, "[600 chars]")
        Headroom(model="claude-sonnet-4-5", post=self.headroom).plan(request, None)
        self.assertEqual(self.headroom.bodies[-1]["model"], "claude-sonnet-4-5")  # (`headroom_model`, where set)


class CodeModeTests(unittest.TestCase):
    """A Codex code-mode script's commands, as Headroom's proxy reads them, given to it as one."""

    def test_each_exec_command_of_a_script_as_headroom_reads_it(self) -> None:
        cases = {
            'await tools.exec_command({"cmd": "cat a.py", "workdir": "/p"})': ["cat a.py"],
            "await tools.exec_command({cmd: \"sed -n '1,80p' f.py\", workdir: \"/r\"})": ["sed -n '1,80p' f.py"],
            "await tools.exec_command({'cmd': 'nl -ba f.py'})": ["nl -ba f.py"],
            "await tools.exec_command({cmd: `ls -la`})": ["ls -la"],
            'await tools.exec_command({cmd: "printf \\"a\\\\tb\\""})': ['printf "a\\tb"'],
            "await tools.exec_command({cmd: c})": [None],  # (a variable)
            'await tools.exec_command({cmd: "c" + "at f"})': [None],  # (an expression)
            "await tools.exec_command({cmd: `cat ${f}`})": [None],
            'await tools.exec_command({cmd: "c\\x61t f"})': [None],  # (an escape it does not decode)
            'await tools.exec_command("cat f")': [None],  # (not an object)
            'await tools.exec_command({"workdir": "/p"})': [],
            "text('no command')": [],
        }
        for script, commands in cases.items():
            self.assertEqual(_commands(script), commands, script)

    def test_a_scripts_commands_go_as_one_a_read_where_one_of_them_is(self) -> None:
        many = ('await tools.exec_command({cmd: "cat > f <<\'EOF\'\\nhi\\nEOF"});\n'
                'await tools.exec_command({cmd: "cd /p && cat f"});\nawait tools.exec_command({cmd: x});')
        self.assertEqual(_calls(f"exec {many}"), [("exec", '{"cmd": "cd /p && cat f; cat"}')])  # (no heredoc; `cat`: possibly a read)
        self.assertEqual(_calls("exec text('hi')"), [("exec", "text('hi')")])  # (no command: as it is)
        self.assertEqual(_calls('exec_command {"cmd": "ls"}'), [("exec_command", '{"cmd": "ls"}')])
        self.assertEqual(_calls('Bash({"command": "grep exec_command( x"})'), [("Bash", '{"command": "grep exec_command( x"}')])


class PartsTests(unittest.TestCase):
    """The Responses codec gives a list output's parts, and keeps those a new text keeps."""

    def edit(self, *texts: str, new: str, media: tuple = ()) -> Any:
        wire = {"type": "custom_tool_call_output", "call_id": "c", "output": [{"type": "input_text", "text": t} for t in texts]}
        self.assertEqual(CODEC.classify(wire).parts, texts)
        return CODEC.edit(wire, new, media)["output"]

    def test_a_changed_part_goes_where_it_was_and_the_others_as_they_were(self) -> None:
        self.assertEqual(self.edit(HEADER, "log", "\n[exit_code=0]", new=f"{HEADER}\nLOG\n\n[exit_code=0]"),
                         [{"type": "input_text", "text": t} for t in (HEADER, "LOG", "\n[exit_code=0]")])
        self.assertEqual(self.edit("log", new="LOG"), [{"type": "input_text", "text": "LOG"}])
        self.assertEqual(self.edit(HEADER, "a", "b", new=f"{HEADER}\nA\nB"),  # (two changed side by side: one)
                         [{"type": "input_text", "text": t} for t in (HEADER, "A\nB")])
        wire = {"type": "function_call_output", "call_id": "c", "output": "log"}
        self.assertEqual((CODEC.classify(wire).parts, CODEC.edit(wire, "LOG", ())["output"]), ((), "LOG"))


if __name__ == "__main__":
    unittest.main()
