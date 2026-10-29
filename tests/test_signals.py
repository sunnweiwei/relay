"""What counts as a signal: a lone `echo` of it, run as a command, its output its own (`signalled`):
among parallel calls too, each call's result an item of its own."""

from __future__ import annotations

import json
import re
import unittest
from typing import Any

from relay.core.ir import Item, Kind
from relay.protocols import AnthropicMessages, Gemini, OpenAIChat, OpenAIResponses
from relay.strategies.conversation import signalled

SIGNAL = re.compile(r"\[(branch|return)\][ \t]*([^\n]*)")
ECHO = 'echo "[branch] Read models :: read models.py"'


def anthropic(name: str, args: dict[str, Any], output: Any, text: str = "") -> tuple[Item, Item]:
    codec = AnthropicMessages()
    call = {"role": "assistant", "content": [*([{"type": "text", "text": text}] if text else []),
                                             {"type": "tool_use", "id": "t", "name": name, "input": args}]}
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": output}]}
    return codec.classify(call), codec.classify(result)


def said(call: Item, result: Item) -> tuple[str, str] | None:
    found = signalled([call], [result], SIGNAL, whole=True)
    return found and found[:2]


class Mentioned(unittest.TestCase):
    """The four places a signal's text turns up without being one."""

    def test_in_the_models_text_beside_a_call(self) -> None:
        self.assertIsNone(said(*anthropic("Read", {"file_path": "a.py"}, "A TEXT", text=f"I cannot run `{ECHO}` here.")))

    def test_in_another_tools_input(self) -> None:  # a task handed to a sub-agent
        self.assertIsNone(said(*anthropic("Agent", {"prompt": f"Open a branch per file: {ECHO}, then return."}, "report")))

    def test_in_a_command_and_its_output(self) -> None:  # grep, cat of a file that says it
        self.assertIsNone(said(*anthropic("Bash", {"command": 'grep -n "\\[branch\\]" folding.py'},
                                          '57:SIGNAL = "[branch] ..."')))
        self.assertIsNone(said(*anthropic("Bash", {"command": "cat README.md"}, "run `echo \"[branch] x :: y\"`")))

    def test_in_a_file_written(self) -> None:
        self.assertIsNone(said(*anthropic("Write", {"file_path": "x.sh", "content": ECHO}, "File created successfully")))

    def test_in_an_echo_that_is_not_alone(self) -> None:  # (the guidance: "alone")
        self.assertIsNone(said(*anthropic("Bash", {"command": f"cd src && {ECHO}"}, "[branch] Read models :: read models.py")))
        self.assertIsNone(said(*anthropic("Bash", {"command": f'{ECHO} && echo "[return] done"'}, "[branch] Read\n[return] done")))


class Signalled(unittest.TestCase):
    """A lone echo, in each harness's form."""

    def test_anthropic(self) -> None:
        self.assertEqual(said(*anthropic("Bash", {"command": ECHO}, "[branch] Read models :: read models.py")),
                         ("branch", "Read models :: read models.py"))

    def test_beside_the_models_text_and_other_calls(self) -> None:
        codec = AnthropicMessages()
        body = {"messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "Opening a [branch]."},
            {"type": "tool_use", "id": "x", "name": "Read", "input": {"file_path": "a.py"}},
            {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": ECHO}}]},
            {"role": "user", "content": [  # (as Claude Code sends them: in the order they finished)
                {"type": "tool_result", "tool_use_id": "b", "content": "[branch] Read models :: read models.py"},
                {"type": "tool_result", "tool_use_id": "x", "content": "A TEXT"}]}]}
        call, *results = map(codec.classify, codec.items(body))
        self.assertEqual(signalled([call], results, SIGNAL, whole=True), ("branch", "Read models :: read models.py", 0))

    def test_parallel_calls_of_their_own(self) -> None:  # Codex: an item per call; a grep beside it that finds the signal
        codec = OpenAIResponses()
        calls = [codec.classify({"type": "function_call", "call_id": c, "name": "exec_command", "arguments": json.dumps({"cmd": cmd})})
                 for c, cmd in (("g", 'grep -rn "branch" .'), ("e", ECHO))]
        results = [codec.classify({"type": "function_call_output", "call_id": c, "output": out})
                   for c, out in (("g", "notes.md:3:[branch] Read tests :: read tests"), ("e", "[branch] Read models :: read models.py"))]
        self.assertEqual(signalled(calls, results, SIGNAL), ("branch", "Read models :: read models.py", 1))
        empty = [results[1], codec.classify({"type": "function_call_output", "call_id": "g", "output": ""})]
        self.assertEqual(signalled(calls, empty, SIGNAL), ("branch", "Read models :: read models.py", 0))
        self.assertIsNone(signalled(calls, [empty[1], empty[1]], SIGNAL))  # (whose empty output: not known)

    def test_responses_shell_with_an_argument_list(self) -> None:  # Codex
        codec = OpenAIResponses()
        call = codec.classify({"type": "function_call", "call_id": "c", "name": "shell",
                               "arguments": json.dumps({"command": ["bash", "-lc", ECHO]})})
        result = codec.classify({"type": "function_call_output", "call_id": "c",
                                 "output": json.dumps({"output": "[branch] Read models :: read models.py\n", "metadata": {}})})
        self.assertEqual(said(call, result), ("branch", "Read models :: read models.py"))

    def test_chat_completions(self) -> None:
        codec = OpenAIChat()
        call = codec.classify({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"cmd": ECHO})}}]})
        result = codec.classify({"role": "tool", "tool_call_id": "c",
                                 "content": f'Command: {ECHO}\nOutput: [branch] Read models :: read models.py\nExit Code: 0'})
        self.assertEqual(said(call, result), ("branch", "Read models :: read models.py"))

    def test_gemini(self) -> None:
        codec = Gemini()
        call = codec.classify({"role": "model", "parts": [{"functionCall": {"name": "run_shell_command", "args": {"command": ECHO}}}]})
        result = codec.classify({"role": "user", "parts": [{"functionResponse": {
            "name": "run_shell_command", "response": {"output": "[branch] Read models :: read models.py"}}}]})
        self.assertEqual(said(call, result), ("branch", "Read models :: read models.py"))

    def test_claude_codes_hook_transcript(self) -> None:  # relay/integrations/claude_code.py: "tool {input}"
        call = Item(Kind.TOOL_CALL, f"Bash {json.dumps({'command': ECHO})}")
        self.assertEqual(said(call, Item(Kind.TOOL_RESULT, "[branch] Read models :: read models.py")),
                         ("branch", "Read models :: read models.py"))

    def test_single_quotes_and_no_quotes(self) -> None:
        self.assertEqual(said(*anthropic("Bash", {"command": "echo '[return] done'"}, "[return] done")), ("return", "done"))
        self.assertEqual(said(*anthropic("Bash", {"command": "echo [return] done"}, "[return] done")), ("return", "done"))

    def test_a_message_of_several_lines(self) -> None:
        self.assertEqual(said(*anthropic("Bash", {"command": 'echo "[return] models:\n- a\n- b"'}, "[return] models:\n- a\n- b\n")),
                         ("return", "models:\n- a\n- b"))

    def test_reported_without_output_the_command_says_it(self) -> None:
        self.assertEqual(said(*anthropic("Bash", {"command": 'echo "[return] a \\"quoted\\" word"'}, "")),
                         ("return", 'a "quoted" word'))

    def test_output_that_is_not_the_echos_says_nothing(self) -> None:
        self.assertIsNone(said(*anthropic("Bash", {"command": ECHO}, "permission denied")))


if __name__ == "__main__":
    unittest.main()
