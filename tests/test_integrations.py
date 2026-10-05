from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from starlette.testclient import TestClient

from relay import install, integrations
from relay.core.engine import Engine
from relay.core.ir import Context, Item, Kind
from relay.harnesses import HARNESSES
from relay.integrations.claude_code import HISTORY, PLUGIN, TAIL, ClaudeCodeHook
from relay.prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from relay.strategies import Compaction
from relay.transport import create_app


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "text": text, "toolUses": []}


def call(n: int) -> dict[str, Any]:
    return {"role": "assistant", "text": "", "toolUses": [{"tool_use_id": f"t{n}", "tool": "Read", "input": {"file": n}}]}


def result(n: int) -> dict[str, Any]:
    return {"role": "user", "text": "", "toolUses": [], "toolResults": [{"tool_use_id": f"t{n}", "text": f"CODE: c{n}"}]}


def reply(text: str) -> dict[str, Any]:
    return {"role": "assistant", "text": text, "toolUses": []}


class PathTests(unittest.TestCase):
    def test_auto_takes_the_harness_preferred_path(self) -> None:
        claude, pi = HARNESSES["claude_code"], HARNESSES["pi"]
        self.assertEqual(integrations.choose(claude).name, "proxy")
        self.assertEqual(integrations.choose(claude, "hook").name, "hook")
        with self.assertRaisesRegex(ValueError, "pi has no hook path"):
            integrations.choose(pi, "hook")


class ClaudeCodeInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        patcher = patch.dict(os.environ, {"RELAY_HOME": str(self.root / "relay"), "CLAUDE_CONFIG_DIR": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_plugin_is_enabled_and_asked_before_every_request(self) -> None:
        original = '{\n  "env": {\n    "ANTHROPIC_BASE_URL": "http://gateway"\n  },\n  "theme": "light"\n}\n'
        (self.root / "settings.json").write_text(original)
        install.install("claude_code", "http://127.0.0.1:9999", ClaudeCodeHook().installation("http://127.0.0.1:9999").settings)
        settings = json.loads((self.root / "settings.json").read_text())
        self.assertEqual(settings["enabledPlugins"], {"relay@relay": True})
        self.assertEqual(settings["extraKnownMarketplaces"]["relay"]["source"], {"source": "directory", "path": str(PLUGIN)})
        self.assertTrue((PLUGIN / ".claude-plugin" / "marketplace.json").exists())
        env = settings["env"]
        self.assertEqual((env["CLAUDE_CODE_ENABLE_FUNCTION_HOOKS"], env["RELAY_HOOK_URL"]), ("1", "http://127.0.0.1:9999"))
        self.assertEqual(env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"], "0.01")  # Claude Code's window is left as it is
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://gateway")
        install.uninstall("claude_code")
        self.assertEqual((self.root / "settings.json").read_text(), original)


class ClaudeCodeCompactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine(Compaction(threshold=30_000, min_gain=0), event_log=os.path.join(tempfile.mkdtemp(), "events.jsonl"))

    def run_hook(self, messages: list[dict[str, Any]], agent: str | None = None) -> tuple[dict, dict]:
        request = {"harness": "claude_code", "agent": agent, "trigger": "auto", "model": "m", "messages": messages,
                   "tokens": 40_000, "window": 200_000}  # past the strategy's 30k
        asked = ClaudeCodeHook().compact(self.engine, request)["summarize"]
        done = ClaudeCodeHook().compact(self.engine, {**request, "summaries": {asked["key"]: "the codes were c1 and c2"}})
        return asked, done

    def event(self) -> dict[str, Any]:
        return json.loads(Path(self.engine.event_log).read_text().splitlines()[-1])

    def test_mid_turn_the_unsent_results_follow_in_the_fork_prompt(self) -> None:
        messages = [user("read the files"), call(1), result(1), call(2), result(2)]
        asked, done = self.run_hook(messages)
        self.assertEqual(asked["fork"], f"{TAIL}\n\n[Tool result]: CODE: c2\n\n{SUMMARIZATION_PROMPT}")
        self.assertTrue(asked["complete"].startswith(f"{HISTORY}\n\n[User]: read the files\n\n[Tool call]: Read"))
        summary = f"{SUMMARY_PREFIX}\nthe codes were c1 and c2"
        self.assertEqual(done, {"messages": [{"ref": 0}, {"role": "user", "text": summary}]})
        event = self.event()
        self.assertEqual((event["path"], event["messages_before"], event["summary_requests"]), ("hook", 5, 1))
        self.assertEqual(event["head"], [{"ref": 0}, {"kind": "summary", "text": summary}])

    def test_at_a_turn_start_the_fork_is_codexs_summary_request(self) -> None:
        messages = [user("read the files"), call(1), result(1), reply("c1"), user("now the next one")]
        asked, done = self.run_hook(messages)
        self.assertEqual(asked["fork"], SUMMARIZATION_PROMPT)  # it sees exactly the history before the new turn
        self.assertEqual(done["messages"][0], {"ref": 0})
        self.assertEqual(done["messages"][2:], [{"ref": 4}])

    def test_a_sub_agent_is_summarized_from_a_rendering(self) -> None:
        asked, done = self.run_hook([user("read file 1"), call(1), result(1)], agent="a1")
        self.assertIsNone(asked["fork"])
        self.assertIn("[Tool result]: CODE: c1", asked["complete"])

    def test_injected_context_is_left_to_claude_code(self) -> None:
        reminder = user("<system-reminder>codename ALPHA</system-reminder>")
        asked, done = self.run_hook([reminder, user("read the files"), call(1), result(1)])
        self.assertNotIn("ALPHA", asked["complete"])
        self.assertEqual(done["messages"][0], {"ref": 1})

    def test_the_strategy_decides_on_the_size_of_the_next_request(self) -> None:
        messages = [user("read the files"), call(1), result(1)]
        hook = ClaudeCodeHook()
        below = {"messages": messages, "model": "m", "tokens": 29_990, "window": 200_000}
        self.assertEqual(hook.compact(self.engine, below), {"skip": "compaction: not now"})
        # The new tool result (not sent yet) tips it over the threshold.
        grown = {**below, "messages": [*messages[:2], {**messages[2], "toolResults": [
            {"tool_use_id": "t1", "text": "x" * 400}]}]}
        self.assertIn("summarize", hook.compact(self.engine, grown))
        # Without a reply yet (a resumed process), the transcript alone is the estimate.
        self.assertEqual(hook.compact(self.engine, {**below, "tokens": None}), {"skip": "compaction: not now"})

    def test_near_claude_codes_own_limit_the_strategy_is_forced(self) -> None:
        engine = Engine(Compaction(threshold=190_000, min_gain=0))
        request = {"messages": [user("read the files"), call(1), result(1)], "model": "m", "window": 200_000}
        self.assertEqual(ClaudeCodeHook().compact(engine, {**request, "tokens": 160_000})["skip"], "compaction: not now")
        self.assertIn("summarize", ClaudeCodeHook().compact(engine, {**request, "tokens": 168_000}))

    def test_a_tool_call_is_never_separated_from_its_result(self) -> None:
        @dataclass(frozen=True)
        class KeepsCalls(Compaction):
            def plan(self, request, summarizer):  # keeps a tool call without its result
                return Context((request.current[1],))

        engine = Engine(KeepsCalls(threshold=10))
        messages = [user("read"), call(1), result(1)]
        with self.assertRaisesRegex(ValueError, "separates a tool call"):
            ClaudeCodeHook().compact(engine, {"messages": messages, "tokens": 100})

    def test_the_strategys_state_is_kept_by_session_and_agent(self) -> None:
        seen = []

        @dataclass(frozen=True)
        class Counts:
            name = "counts"

            def fingerprint(self):
                return {}

            def plan(self, request, summarizer):
                seen.append(request.state)
                return Context(request.current, (request.state or 0) + 1)

        engine, messages = Engine(Counts()), [user("read"), call(1), result(1)]
        for session in ("s1", "s1", "s2"):
            self.assertIn("skip", ClaudeCodeHook().compact(engine, {"messages": messages, "session": session}))
        ClaudeCodeHook().compact(engine, {"messages": messages, "session": "s1", "agent": "a1"})
        self.assertEqual(seen, [None, 1, None, None])

    def test_what_only_the_proxy_can_carry_is_refused(self) -> None:
        @dataclass(frozen=True)
        class Annotates:
            answer: Any
            name = "annotates"

            def fingerprint(self):
                return {}

            def plan(self, request, summarizer):
                return self.answer(request)

        messages = [user("read"), call(1), result(1)]
        for answer, refused in [
            (lambda r: Context(r.current, notes=("size",)), "notes"),
            (lambda r: Context((Item(Kind.SYSTEM, "rules"), *r.current)), "system"),
            (lambda r: Context((r.current[0], *(replace(i, text="short") for i in r.current[1:]))), "change a message"),
        ]:
            with self.assertRaisesRegex(ValueError, refused):
                ClaudeCodeHook().compact(Engine(Annotates(answer)), {"messages": messages})


class EndpointTests(unittest.TestCase):
    def test_the_plugin_asks_for_summaries_then_gets_the_transcript(self) -> None:
        client = TestClient(create_app(Engine(Compaction(threshold=30_000, min_gain=0))))
        request = {"harness": "claude_code", "model": "m", "tokens": 40_000, "messages": [user("read"), call(1), result(1)]}
        asked = client.post("/relay/v1/compact", json=request).json()["summarize"]
        done = client.post("/relay/v1/compact", json={**request, "summaries": {asked["key"]: "c1"}}).json()
        self.assertEqual(done["messages"][0], {"ref": 0})
        self.assertEqual(client.post("/relay/v1/compact", json={"harness": "nope"}).status_code, 500)


if __name__ == "__main__":
    unittest.main()
