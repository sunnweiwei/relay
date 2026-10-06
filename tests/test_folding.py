from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.harnesses import Harness
from relay.protocols import Gemini, OpenAIResponses
from relay.strategies import ContextFolding
from relay.strategies.folding import NESTED, RETURNED

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


if __name__ == "__main__":
    unittest.main()
