from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.protocols import AnthropicMessages, Gemini, OpenAIChat, OpenAIResponses
from relay.strategies import Compaction, ContextFolding
from relay.strategies.conversation import IGNORED
from relay.strategies.folding import BRANCH as BRANCH_MESSAGE, NESTED, RETURNED

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def run(i: int, command: str, output: str) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": output}]


def read(i: int) -> list[dict[str, Any]]:
    return run(i, f"cat file_{i}.txt", f"file {i} text\nCODE: c{i}")


BRANCH = run(10, 'echo "[branch] Read files :: read file_2 and file_3, report their codes"',
             "[branch] Read files :: read file_2 and file_3, report their codes")
RETURN = run(11, 'echo "[return] file_2: c2, file_3: c3"', "Output:\n[return] file_2: c2, file_3: c3")


class FoldingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine(ContextFolding())
        self.history = [msg("developer", "rules"), msg("user", "read the files"), *read(1)]

    def send(self, *items: dict[str, Any]) -> Any:
        return self.engine.prepare(CODEC, HARNESS, {"model": "m", "input": list(items), "tools": TOOLS},
                                   tenant="t", post=lambda r: (500, {}))

    def text(self, item: dict[str, Any]) -> str:
        return CODEC.classify(item).text

    def test_a_branch_opens_with_foldagents_message(self) -> None:
        sent = self.send(*self.history, *BRANCH).body["input"]
        self.assertTrue(self.text(sent[1]).startswith("## Branches"))  # the guidance, after the harness's own
        self.assertTrue(self.text(sent[-1]).startswith("ROLE CHANGE: `MODE: BRANCH`"))
        self.assertIn("read file_2 and file_3, report their codes", self.text(sent[-1]))
        self.assertEqual(sent[-2], BRANCH[0])  # the call, as it was

    def test_a_return_folds_the_branch_and_keeps_the_prefix(self) -> None:
        self.send(*self.history, *BRANCH)
        branch = [*self.history, *BRANCH, *read(2), *read(3)]
        self.send(*branch)
        sent = self.send(*branch, *RETURN)
        self.assertTrue(sent.compacted)
        items = sent.body["input"]
        self.assertEqual([items[0], *items[2:-1]], [*self.history, BRANCH[0]])  # untouched up to the branch call
        self.assertEqual(self.text(items[-1]), RETURNED.format(message="file_2: c2, file_3: c3"))
        self.assertEqual(items[-1]["call_id"], "c10")  # still the branch call's result
        self.assertNotIn("CODE: c2", json.dumps(items))
        # It holds: the main conversation goes on after the folded branch.
        later = self.send(*branch, *RETURN, *read(4)).body["input"]
        self.assertEqual(later[:-2], items)
        self.assertEqual(later[-2:], read(4))

    def test_a_branch_cannot_branch(self) -> None:
        nested = run(12, 'echo "[branch] Deeper :: more"', "[branch] Deeper :: more")
        sent = self.send(*self.history, *BRANCH, *read(2), *nested).body["input"]
        self.assertEqual(self.text(sent[-1]), NESTED)

    def test_signals_are_read_from_any_form_of_output(self) -> None:
        # Hermes reports a command's output as JSON; Gemini CLI repeats the command before it.
        branch = run(10, 'echo \\"[branch] Read files :: read file_2\\"', json.dumps({"output": "[branch] Read files :: read file_2\n"}))
        ret = run(11, 'echo "[return] file_2: c2"', 'Command: echo "[return] file_2: c2"\nOutput: [return] file_2: c2\nExit Code: 0')
        self.send(*self.history, *branch)
        items = self.send(*self.history, *branch, *read(2), *ret).body["input"]
        self.assertEqual(self.text(items[-1]), RETURNED.format(message="file_2: c2"))

    def test_a_return_keeps_every_line_of_its_message(self) -> None:
        # Claude Code reports `echo "[return] line\nline"` as the lines the echo printed.
        ret = run(11, 'echo "[return] file_2: c2\nfile_3: c3"', "[return] file_2: c2\nfile_3: c3\nnothing pending")
        self.send(*self.history, *BRANCH)
        items = self.send(*self.history, *BRANCH, *read(2), *read(3), *ret).body["input"]
        self.assertEqual(self.text(items[-1]), RETURNED.format(message="file_2: c2\nfile_3: c3\nnothing pending"))

    def test_a_branch_beside_other_calls_of_their_own_is_answered_that_it_was_ignored(self) -> None:
        # Codex: an item per parallel call, then their outputs; a grep beside the echo finds the signal too.
        grep = run(9, 'grep -rn "branch" notes.md', "notes.md:3:[branch] Old :: an old note")
        beside = [BRANCH[0], grep[0], grep[1], BRANCH[1]]
        sent = self.send(*self.history, *beside).body["input"]
        self.assertEqual(sent[-4:-1], [BRANCH[0], grep[0], grep[1]])
        self.assertEqual(self.text(sent[-1]), IGNORED.format(name="branch"))
        self.assertEqual(sent[-1]["call_id"], "c10")
        # Run again alone, it opens; the one ignored stays answered so.
        alone = run(12, 'echo "[branch] Read files :: read file_2"', "[branch] Read files :: read file_2")
        later = self.send(*self.history, *beside, *alone).body["input"]
        self.assertEqual(self.text(later[-3]), IGNORED.format(name="branch"))
        self.assertTrue(self.text(later[-1]).startswith("ROLE CHANGE"))

    def test_a_branch_beside_other_calls_is_ignored_and_the_model_told(self) -> None:
        # Claude Code answers parallel tool uses in one message, in its own order.
        codec = AnthropicMessages()

        def use(i: str, name: str, command: str) -> dict[str, Any]:
            return {"type": "tool_use", "id": i, "name": name, "input": {"command": command}}

        def result(i: str, text: str) -> dict[str, Any]:
            return {"type": "tool_result", "tool_use_id": i, "content": text}

        task = {"role": "user", "content": "investigate the files"}
        calls = {"role": "assistant", "content": [use("b", "Bash", 'echo "[branch] Read models :: read models.py"'),
                                                  use("x", "Read", "exceptions.py"), use("y", "Read", "models.py")]}
        results = {"role": "user", "content": [result("x", "EXCEPTIONS TEXT"), result("b", "[branch] Read models :: read models.py"),
                                               result("y", "MODELS TEXT")]}

        def send(*messages: dict[str, Any]) -> Any:
            return self.engine.prepare(codec, HARNESS, {"model": "m", "messages": list(messages), "tools": [{"name": "Bash"}]},
                                       tenant="a", post=lambda r: (500, {})).body["messages"]

        sent = send(task, calls, results)
        self.assertEqual(sent[-1]["content"][:3], results["content"])  # each result as it was
        self.assertEqual(sent[-1]["content"][3], {"type": "text", "text": IGNORED.format(name="branch")})
        # Told once: on the request after, the step is as the harness sent it; run alone, the echo opens a branch.
        alone = [{"role": "assistant", "content": [use("c", "Bash", 'echo "[branch] Read models :: read models.py"')]},
                 {"role": "user", "content": [result("c", "[branch] Read models :: read models.py")]}]
        later = send(task, calls, results, *alone)
        self.assertEqual(later[2], results)
        self.assertIn(BRANCH_MESSAGE.split("\n")[0], json.dumps(later[-1]))

    def test_a_signal_the_model_only_mentions_beside_its_calls_opens_nothing(self) -> None:
        # Claude Code (trace 394): the model explains, in the text beside two reads, why it cannot open
        # a branch in the same step; the reads answer with the files.
        codec = AnthropicMessages()
        task = {"role": "user", "content": "open a branch beside the reads"}
        calls = {"role": "assistant", "content": [
            {"type": "text", "text": '分支的 `echo "[branch] ..."` 必须单独发出，不能和别的调用放在同一步。'},
            {"type": "tool_use", "id": "x", "name": "Read", "input": {"file_path": "exceptions.py"}},
            {"type": "tool_use", "id": "y", "name": "Read", "input": {"file_path": "adapters.py"}}]}
        results = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "EXCEPTIONS TEXT"},
                                               {"type": "tool_result", "tool_use_id": "y", "content": "ADAPTERS TEXT"}]}
        sent = self.engine.prepare(codec, HARNESS, {"model": "m", "messages": [task, calls, results], "tools": [{"name": "Bash"}]},
                                   tenant="a", post=lambda r: (500, {})).body["messages"]
        self.assertEqual(sent[-1], results)
        # Chat Completions: one assistant message holds the text and the calls too.
        codec = OpenAIChat()
        messages = [{"role": "user", "content": "read the files"},
                    {"role": "assistant", "content": 'I will open a branch later with echo "[branch] Read :: read file_2".',
                     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "sh", "arguments": '{"cmd": "cat file_1.txt"}'}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "file 1 text"}]
        sent = self.engine.prepare(codec, HARNESS, {"model": "m", "messages": messages, "tools": [{}]}, tenant="a",
                                   post=lambda r: (500, {})).body["messages"]
        self.assertEqual(sent[-1], messages[-1])
        # The echo itself still opens one, the text beside it or not.
        messages[1] = {**messages[1], "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "sh", "arguments": json.dumps({"cmd": 'echo "[branch] Read :: read file_2"'})}}]}
        messages[2] = {"role": "tool", "tool_call_id": "c1", "content": "[branch] Read :: read file_2"}
        sent = self.engine.prepare(codec, HARNESS, {"model": "m", "messages": messages, "tools": [{}]}, tenant="a",
                                   post=lambda r: (500, {})).body["messages"]
        self.assertIn("ROLE CHANGE", json.dumps(sent[-1]))

    def test_a_return_after_a_branch_that_was_ignored_folds_nothing(self) -> None:
        # Claude Code: its own context (reminders) leads the history; the branch beside other calls
        # was ignored, so the return has no branch to fold.
        from relay.harnesses import ClaudeCode
        codec, harness = AnthropicMessages(), ClaudeCode()

        def use(i: str, command: str) -> dict[str, Any]:
            return {"type": "tool_use", "id": i, "name": "Bash", "input": {"command": command}}

        def result(i: str, text: str) -> dict[str, Any]:
            return {"type": "tool_result", "tool_use_id": i, "content": text}

        task = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nCodebase and user instructions are shown below. Be sure to adhere to these instructions.\n</system-reminder>"},
                                            {"type": "text", "text": "investigate the files"}]}
        calls = {"role": "assistant", "content": [use("x", "cat exceptions.py"), use("b", 'echo "[branch] Read models :: read models.py"'),
                                                  use("y", "cat models.py")]}
        results = {"role": "user", "content": [result("x", "EXCEPTIONS TEXT"), result("b", "[branch] Read models :: read models.py"),
                                               result("y", "MODELS TEXT")]}
        step = [{"role": "assistant", "content": [use("s", "cat more.py")]}, {"role": "user", "content": [result("s", "MORE TEXT")]}]
        ret = [{"role": "assistant", "content": [use("r", 'echo "[return] models: 1180 lines"')]},
               {"role": "user", "content": [result("r", "[return] models: 1180 lines")]}]

        def send(*messages: dict[str, Any]) -> Any:
            return self.engine.prepare(codec, harness, {"model": "m", "messages": list(messages), "tools": [{"name": "Bash"}]},
                                       tenant="a", post=lambda r: (500, {}))

        tokens = {"role": "system", "content": "<total_tokens>900000 tokens left</total_tokens>"}  # (after each step)
        send(task, tokens, calls, results, tokens)
        send(task, tokens, calls, results, tokens, *step, tokens)
        history = [task, tokens, calls, results, tokens, *step, tokens, *ret]
        sent = send(*history)
        self.assertFalse(sent.compacted)
        self.assertEqual(sent.body["messages"], history)

    def test_a_signal_in_another_tools_input_opens_nothing(self) -> None:
        # Claude Code (trace 458): the model hands a sub-agent a task that tells it how to branch; the
        # Agent call's input holds the signal, its result is the sub-agent's report.
        codec = AnthropicMessages()
        task = {"role": "user", "content": "use a sub-agent"}
        call = {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "Agent", "input": {
            "prompt": 'Open a branch per file: echo "[branch] Read models :: read models.py", then echo "[return] ...".'}}]}
        report = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "The agent's report."}]}
        sent = self.engine.prepare(codec, HARNESS, {"model": "m", "messages": [task, call, report], "tools": [{"name": "Bash"}]},
                                   tenant="a", post=lambda r: (500, {})).body["messages"]
        self.assertEqual(sent[-1], report)

    def test_an_echo_reported_without_its_output_still_signals(self) -> None:
        # A harness that reports no output for the echo: the call says it.
        sent = self.send(*self.history, *run(10, 'echo "[branch] Read files :: read file_2"', "")).body["input"]
        self.assertTrue(self.text(sent[-1]).startswith("ROLE CHANGE"))

    def test_gemini_a_branch_beside_other_calls_is_ignored(self) -> None:
        codec = Gemini()

        def call(command: str) -> dict[str, Any]:
            return {"functionCall": {"name": "run_shell_command", "args": {"command": command}}}

        def output(text: str) -> dict[str, Any]:
            return {"functionResponse": {"name": "run_shell_command", "response": {"output": text}}}

        task = {"role": "user", "parts": [{"text": "read the files"}]}
        calls = {"role": "model", "parts": [call("cat file_1.txt"), call('echo "[branch] Read :: read file_2"')]}
        results = {"role": "user", "parts": [output("file 1 text"), output("[branch] Read :: read file_2")]}
        sent = self.engine.prepare(codec, HARNESS, {"contents": [task, calls, results], "tools": [{}]}, tenant="g",
                                   post=lambda r: (500, {})).body["contents"][-1]["parts"]
        self.assertEqual(sent, [*results["parts"], {"text": IGNORED.format(name="branch")}])

    def test_gemini_folds_once(self) -> None:
        codec = Gemini()

        def call(command: str) -> dict[str, Any]:
            return {"role": "model", "parts": [{"functionCall": {"name": "run_shell_command", "args": {"command": command}}}]}

        def output(text: str) -> dict[str, Any]:
            return {"role": "user", "parts": [{"functionResponse": {"name": "run_shell_command", "response": {"output": text}}}]}

        task = {"role": "user", "parts": [{"text": "read the files"}]}
        history = [task, call('echo "[branch] Read :: read file_2"'), output("[branch] Read :: read file_2"),
                   call("cat file_2.txt"), output("file 2 text"), call('echo "[return] file_2: c2"'), output("[return] file_2: c2")]

        def send(*contents: dict[str, Any]) -> Any:
            return self.engine.prepare(codec, HARNESS, {"contents": list(contents), "tools": [{}]}, tenant="g", post=lambda r: (500, {}))

        send(*history[:3])
        folded = send(*history)
        self.assertTrue(folded.compacted)
        self.assertEqual(len(folded.body["contents"]), 3)
        self.assertIn("file_2: c2", json.dumps(folded.body["contents"][2]))
        self.assertFalse(send(*history, call("cat file_3.txt"), output("file 3 text")).compacted)  # done, not again

    def test_a_conversation_without_branches_is_left_alone(self) -> None:
        sent = self.send(*self.history, *read(2))
        self.assertFalse(sent.compacted)
        self.assertEqual(sent.body["input"][2:], [*self.history[1:], *read(2)])

    def test_compaction_holds_the_threshold_when_folding_does_not(self) -> None:
        summary = (200, {"status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "SUMMARY"}]}]})
        engine = Engine(ContextFolding(Compaction(threshold=2_000, min_gain=0)))
        long = [*self.history, *(item for i in range(2, 12) for item in run(i, f"cat big_{i}", "x" * 1_000))]
        sent = engine.prepare(CODEC, HARNESS, {"model": "m", "input": long, "tools": TOOLS}, tenant="t",
                              post=lambda r: summary)
        self.assertTrue(sent.compacted)
        self.assertIn("SUMMARY", json.dumps(sent.body))
        self.assertNotIn("x" * 1_000, json.dumps(sent.body))

    def test_a_fold_that_brings_the_context_under_the_threshold_is_enough(self) -> None:
        engine = Engine(ContextFolding(Compaction(threshold=2_000, min_gain=0)))
        steps = [item for i in range(2, 9) for item in run(i, f"cat big_{i}", "x" * 1_000)]
        branch = [*self.history, *BRANCH, *steps, *RETURN]
        sent = engine.prepare(CODEC, HARNESS, {"model": "m", "input": branch, "tools": TOOLS}, tenant="t",
                              post=lambda r: self.fail("no summary is asked for"))
        self.assertTrue(sent.compacted)
        self.assertNotIn("x" * 1_000, json.dumps(sent.body))
        self.assertIn(RETURNED.split(":")[0], json.dumps(sent.body))


if __name__ == "__main__":
    unittest.main()
