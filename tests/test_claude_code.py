"""Relay compacts Claude Code's conversation as Claude Code compacts it itself."""

from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.core.local import Document, Local
from relay.harnesses import ClaudeCode
from relay.harnesses.claude_code import COMPACT_PROMPT, CONTINUED, RESUME, TRANSCRIPT
from relay.protocols import AnthropicMessages
from relay.strategies import Compaction

CODEC = AnthropicMessages()
INSTRUCTIONS = ("<system-reminder>\nCodebase and user instructions are shown below. Be sure to adhere to these instructions. "
                "IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.\n\n"
                "Contents of /p/CLAUDE.md (project instructions, checked into the codebase):\n\n{}\n</system-reminder>")
ATTRIBUTION = "<system-reminder>\nAttribution for git commits and pull requests you create from here on: none.\n</system-reminder>"
OPENING = ("# Environment\nYou have been invoked in the following environment: \n - Primary working directory: /p\n\n"
           "Available agent types for the Agent tool:\n- claude: any task\n\n"
           "The following skills are available for use with the Skill tool:\n\n- init: a CLAUDE.md\n\nToday's date is 2026-10-07.")
LOCAL = Local("claude_code", "s", transcript="/home/u/.claude/projects/-p/s.jsonl",
              instructions=(Document("/p/CLAUDE.md", "codename BETA\n", "project"),),
              files=(Document("/p/b.txt", "b1\nb2\n"), Document("/p/a.txt", "a1\n")))


def text(*blocks: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": block} for block in blocks]


def read(n: int, path: str) -> list[dict[str, Any]]:
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{n}", "name": "Read", "input": {"file_path": path}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": "x" * 800}]}]


class Upstream:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any]) -> tuple[int, Any]:
        self.requests.append(body)
        return 200, {"content": [{"type": "text", "text": "<analysis>a</analysis>\n\n<summary>\nS\n</summary>"}],
                     "stop_reason": "end_turn"}


class ClaudeCodeCompactionTests(unittest.TestCase):
    def compact(self, *history: dict[str, Any]) -> tuple[list[dict[str, Any]], Upstream]:
        upstream = Upstream()
        start = [{"role": "user", "content": text(INSTRUCTIONS.format("codename ALPHA"), ATTRIBUTION, "read the files")},
                 {"role": "system", "content": text(OPENING)}]
        exchange = Engine(Compaction(threshold=300, min_gain=0)).prepare(
            CODEC, ClaudeCode(), {"model": "m", "messages": [*start, *history]}, tenant="t", post=upstream, local=LOCAL)
        self.assertTrue(exchange.compacted)
        return exchange.body["messages"], upstream

    def test_at_a_turn_start(self) -> None:
        sent, upstream = self.compact(*read(1, "/p/a.txt"), {"role": "assistant", "content": "done"},
                                      {"role": "user", "content": text("more")})
        self.assertIn(COMPACT_PROMPT, json.dumps(upstream.requests[0]["messages"][-1]))  # its own prompt
        self.assertEqual([m["role"] for m in sent], ["user", "system", "assistant", "user", "system"])
        first = [block["text"] for block in sent[0]["content"]]
        # The instruction files as they read now, with the summary, in its own words.
        self.assertEqual(first, [INSTRUCTIONS.format("codename BETA") + "\n",
                                 f"{CONTINUED}Summary:\nS{TRANSCRIPT.format(path=LOCAL.transcript)}{RESUME}"])
        self.assertEqual(sent[1]["content"][0]["text"], "Today's date is 2026-10-07.")
        self.assertEqual(sent[2]["content"], "done")  # the model's last round, kept
        self.assertEqual(CODEC.classify(sent[3]).text, f"{ATTRIBUTION}\nmore")
        again = sent[4]["content"][0]["text"]
        self.assertTrue(again.startswith('Called the Read tool with the following input: {"file_path":"/p/b.txt"}\n'
                                         "Result of calling the Read tool:\n1\tb1\n2\tb2\n3\t\n\n"
                                         'Called the Read tool with the following input: {"file_path":"/p/a.txt"}'))
        self.assertTrue(again.endswith("Available agent types for the Agent tool:\n- claude: any task\n\n"
                                       "# Environment\nYou have been invoked in the following environment: \n"
                                       " - Primary working directory: /p"))
        self.assertNotIn("skills are available", json.dumps(sent))  # not announced again

    def test_mid_turn(self) -> None:
        total = {"role": "system", "content": text("<total_tokens>9000 tokens left</total_tokens>")}
        sent, _ = self.compact(*read(1, "/p/a.txt"), *read(2, "/p/b.txt"), total)
        self.assertEqual([m["role"] for m in sent], ["user", "system", "assistant", "user", "user", "system"])
        self.assertEqual(sent[2]["content"][0]["input"], {"file_path": "/p/b.txt"})  # the last call, with its result
        self.assertEqual(CODEC.classify(sent[4]).text, ATTRIBUTION)
        # The files it does not show, folded into the system message that ends the request.
        last = CODEC.classify(sent[5]).text
        self.assertTrue(last.startswith("<total_tokens>9000 tokens left</total_tokens>\n\nCalled the Read tool with the "
                                        'following input: {"file_path":"/p/a.txt"}'))
        self.assertNotIn("/p/b.txt", last)

    def test_without_a_report_the_request_shows_what_it_can(self) -> None:
        upstream, start = Upstream(), [{"role": "user", "content": text(INSTRUCTIONS.format("codename ALPHA"), "go")}]
        history = [*start, *read(1, "/p/a.txt"), {"role": "assistant", "content": "done"}, {"role": "user", "content": "more"}]
        sent = Engine(Compaction(threshold=100, min_gain=0)).prepare(
            CODEC, ClaudeCode(), {"model": "m", "messages": history}, tenant="t", post=upstream).body["messages"]
        blocks = [block["text"] for block in sent[0]["content"]]
        self.assertEqual(blocks[0], INSTRUCTIONS.format("codename ALPHA") + "\n")  # as it last announced them
        self.assertEqual(blocks[1], f"{CONTINUED}Summary:\nS{RESUME}")  # no transcript to point at


if __name__ == "__main__":
    unittest.main()
