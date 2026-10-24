from __future__ import annotations

import json
import os
import re
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
from relay.strategies import MEM1, Compaction, ContextFolding
from relay.transport import create_app


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "text": text, "toolUses": []}


def call(n: int) -> dict[str, Any]:
    return {"role": "assistant", "text": "", "toolUses": [{"tool_use_id": f"t{n}", "tool": "Read", "input": {"file": n}}]}


def result(n: int) -> dict[str, Any]:
    return {"role": "user", "text": "", "toolUses": [], "toolResults": [{"tool_use_id": f"t{n}", "text": f"CODE: c{n}"}]}


def reply(text: str) -> dict[str, Any]:
    return {"role": "assistant", "text": text, "toolUses": []}


RELEASE = "relay.integrations.claude_code.installed_release"


class PathTests(unittest.TestCase):
    def test_auto_takes_the_preferred_path_that_runs_the_strategy(self) -> None:
        claude, pi = HARNESSES["claude_code"], HARNESSES["pi"]
        hook, https = integrations.HOOKS["claude_code"], integrations.HTTPS["claude_code"]
        with patch(RELEASE, return_value="2.1.309"):
            self.assertEqual(integrations.choose(claude).name, "https")  # (Claude Code's default)
            self.assertEqual(integrations.choose(claude, "auto", "clm").name, "https")
            self.assertEqual(integrations.choose(claude, "auto", "mem1").name, "proxy")  # (checked on neither)
        with patch(RELEASE, return_value="2.1.309"), patch.object(type(https), "unavailable", lambda self: "a proxy"):
            self.assertEqual(integrations.choose(claude).name, "hook")  # (the HTTPS proxy unavailable: the hook)
        with patch(RELEASE, return_value="2.1.309"), patch.object(type(https), "preferred", False):
            self.assertEqual(integrations.choose(claude).name, "hook")  # (once preferred: before the proxy)
            self.assertEqual(integrations.choose(claude, "auto", "folding").name, "hook")
            self.assertEqual(integrations.choose(claude, "auto", "clm").name, "proxy")  # (notes for one request)
            self.assertEqual(integrations.choose(claude, "auto", "ace").name, "proxy")  # (a playbook that grows)
            with patch.object(type(hook), "checked", {"compaction": "2.1.294"}):
                self.assertEqual(integrations.choose(claude, "auto", "folding").name, "proxy")  # (not checked on it)
                self.assertEqual(integrations.choose(claude, "hook", "folding").name, "hook")  # (named: taken)
        with patch(RELEASE, return_value="2.1.309"), patch.object(type(https), "preferred", False), \
                patch.object(type(hook), "preferred", False):
            self.assertEqual(integrations.choose(claude).name, "proxy")  # (a hook not preferred: when named)
            self.assertEqual(integrations.choose(claude, "hook").name, "hook")
            self.assertEqual(integrations.choose(claude, "proxy").name, "proxy")
            with self.assertRaisesRegex(ValueError, "clm needs notes, which the hook cannot carry"):
                integrations.choose(claude, "hook", "clm")
        with patch(RELEASE, return_value="2.1.250"), patch.object(type(https), "preferred", False):
            self.assertEqual(integrations.choose(claude).name, "proxy")
            with self.assertRaisesRegex(ValueError, "older than 2.1.294"):
                integrations.choose(claude, "hook")
        with patch(RELEASE, return_value=None):
            self.assertEqual(integrations.choose(claude).name, "proxy")
        with self.assertRaisesRegex(ValueError, "pi has no hook path"):
            integrations.choose(pi, "hook")

    def test_every_strategy_says_what_it_needs(self) -> None:
        from relay.strategies import STRATEGIES
        self.assertEqual(set(integrations.NEEDS), set(STRATEGIES))

    def test_a_hook_is_checked_only_with_what_it_carries_and_the_readme_says_so(self) -> None:
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        for name, hook in [*integrations.HOOKS.items(), *integrations.HTTPS.items()]:
            for strategy, releases in hook.checked.items():
                self.assertIsNone(integrations.uncarried(hook, strategy), strategy)
                self.assertRegex(readme, rf"\| `{name}` \| {hook.name} \| `{strategy}` \| {re.escape(releases)} \|")


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
        self.assertNotIn("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", env)  # (the plugin moves it, as a session starts)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://gateway")  # (Remote Control needs Anthropic's own)
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
        self.assertEqual(done, {"messages": [{"ref": 0}, {"role": "user", "text": summary, "toolUses": []}]})
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
        messages = [user("read"), call(1), result(1)]
        for answer, refused in [
            (lambda r: Context(r.current, notes=("size",)), "notes"),
            (lambda r: Context((Item(Kind.SYSTEM, "rules"), *r.current[:2])), "instructions are not the ones"),
            (lambda r: Context((r.current[0], replace(r.current[1], text="Read {}"), r.current[2])), "change a tool call"),
            (lambda r: Context((r.current[0], replace(r.current[1], kind=Kind.ASSISTANT), r.current[2])), "keeps a message"),
        ]:
            with self.assertRaisesRegex(ValueError, refused):
                ClaudeCodeHook().compact(Engine(Answers(answer)), {"messages": messages})

    def test_the_instructions_claude_code_got_and_new_text_for_a_result(self) -> None:
        messages = [user("read"), call(1), result(1), call(2), result(2)]

        def answer(r):  # the strategy's instructions, the first result shortened
            return Context((Item(Kind.SYSTEM, "rules"), *r.current[:2], replace(r.current[2], text="[c1, shortened]"),
                            *r.current[3:]))

        done = ClaudeCodeHook().compact(Engine(Answers(answer)), {"messages": messages, "instructions": "rules"})
        self.assertEqual(done["messages"], [
            {"ref": 0}, {"ref": 1},
            {"role": "user", "text": "", "toolUses": [], "toolResults": [{"tool_use_id": "t1", "text": "[c1, shortened]"}]},
            {"ref": 3}, {"ref": 4}])
        unchanged = ClaudeCodeHook().compact(Engine(Answers(lambda r: Context((Item(Kind.SYSTEM, "rules"), *r.current)))),
                                             {"messages": messages, "instructions": "rules"})
        self.assertEqual(unchanged, {"skip": "answers: not now"})  # (the instructions alone: nothing to compact)

    def test_the_history_grows_through_the_compactions(self) -> None:
        seen = []

        def answer(r):  # the first time, everything but the latest result goes
            seen.append((len(r.history), r.history[-1].text))
            return Context((Item(Kind.SUMMARY, "S"), r.current[-2], r.current[-1])) if len(seen) == 1 else None

        engine, messages = Engine(Answers(answer)), [user("read"), call(1), result(1), call(2), result(2)]
        done = ClaudeCodeHook().compact(engine, {"messages": messages, "session": "s"})
        after = [*({**messages[e["ref"]]} if "ref" in e else e for e in done["messages"]), call(3), result(3)]
        ClaudeCodeHook().compact(engine, {"messages": after, "session": "s"})
        self.assertEqual(seen, [(5, "CODE: c2"), (7, "CODE: c3")])  # (the two new items after the five seen)

    def test_what_a_resumed_session_repeats_goes_and_the_latest_result_stands(self) -> None:
        # (resuming, Claude Code gathers each copy compactions kept of a call, then each result it had)
        rewritten = {**result(1), "toolResults": [{"tool_use_id": "t1", "text": "[c1, shortened]"}]}
        messages = [user("read"), call(1), call(1), result(1), rewritten, reply("done"), user("next")]
        seen = []
        engine = Engine(Answers(lambda r: seen.append([i.text for i in r.current])))
        done = ClaudeCodeHook().compact(engine, {"messages": messages})
        self.assertEqual(seen, [["read", 'Read {"file": 1}', "[c1, shortened]", "done", "next"]])
        self.assertEqual(done["messages"], [{"ref": 0}, {"ref": 2}, {"ref": 4}, {"ref": 5}, {"ref": 6}])  # (written clean)

    def test_the_context_window_begins_after_each_compaction(self) -> None:
        bases = []

        def answer(r):  # compacts once, on the second request
            bases.append(r.base)
            return Context((Item(Kind.SUMMARY, "S"), *r.current[-2:])) if len(bases) == 2 else None

        engine, hook = Engine(Answers(answer)), ClaudeCodeHook()
        messages = [user("read"), call(1), result(1)]
        hook.compact(engine, {"messages": messages, "session": "s", "tokens": 20_000})
        done = hook.compact(engine, {"messages": [*messages, call(2), result(2)], "session": "s", "tokens": 25_000})
        after = [e if "ref" not in e else [*messages, call(2), result(2)][e["ref"]] for e in done["messages"]]
        hook.compact(engine, {"messages": [*after, call(3), result(3)], "session": "s", "tokens": 3_000})
        hook.compact(engine, {"messages": [*after, call(3), result(3), call(4), result(4)], "session": "s", "tokens": 4_000})
        self.assertEqual([b // 1000 for b in bases], [20, 20, 3, 3])  # (each the first request's estimate, ~k)

    def test_a_branch_is_written_as_its_echo_comes_in_and_folds_at_the_next_request(self) -> None:
        engine, hook = Engine(ContextFolding(Compaction(threshold=10**9))), ClaudeCodeHook()
        branch = {"role": "assistant", "text": "", "toolUses": [
            {"tool_use_id": "t1", "tool": "Bash", "input": {"command": 'echo "[branch] read :: read file 2"'}}]}
        messages = [user("read the files"), branch]
        written = hook.result(engine, {"messages": messages, "call": {"tool_use_id": "t1", "text": "[branch] read :: read file 2"}})
        self.assertIn("MODE: BRANCH", written["text"])
        echoed = {"role": "user", "text": "", "toolUses": [], "toolResults": [{"tool_use_id": "t1", "text": written["text"]}]}
        self.assertEqual(hook.compact(engine, {"messages": [*messages, echoed], "instructions": hook.instructions(engine)}),
                         {"skip": "folding: not now"})  # (nothing left to compact)
        self.assertEqual(hook.result(engine, {"messages": [*messages, echoed, call(2)],  # (not a signal)
                                              "call": {"tool_use_id": "t2", "text": "CODE: c2"}}), {})
        ret = {"role": "assistant", "text": "", "toolUses": [
            {"tool_use_id": "t3", "tool": "Bash", "input": {"command": 'echo "[return] c2"'}}]}
        later = [*messages, echoed, call(2), result(2), ret]
        self.assertEqual(hook.result(engine, {"messages": later, "call": {"tool_use_id": "t3", "text": "[return] c2"}}), {})
        returned = {"role": "user", "text": "", "toolUses": [], "toolResults": [{"tool_use_id": "t3", "text": "[return] c2"}]}
        folded = hook.compact(engine, {"messages": [*later, returned], "instructions": hook.instructions(engine)})
        self.assertEqual(folded["messages"][:2], [{"ref": 0}, {"ref": 1}])  # (the fold: a compaction, the steps gone)
        self.assertIn("Branch has finished its task", folded["messages"][2]["toolResults"][0]["text"])
        self.assertEqual(len(folded["messages"]), 3)

    def test_what_a_compaction_dropped_stays_dropped_when_a_resume_brings_it_back(self) -> None:
        calls = []

        def answer(r):  # the first time, the first read goes
            calls.append(r)
            return Context((r.current[0], *r.current[3:])) if len(calls) == 1 else None

        engine, hook = Engine(Answers(answer)), ClaudeCodeHook()
        messages = [{**m, "handle": f"row{n}"} for n, m in enumerate([user("read"), call(1), result(1), call(2), result(2)])]
        done = hook.compact(engine, {"messages": messages, "session": "s"})
        self.assertEqual(done["messages"], [{"ref": 0}, {"ref": 3}, {"ref": 4}])
        # Resumed: the history before the compaction is back, by its rows; the same call made anew is not that
        resumed = [*messages, {**reply("c1 c2"), "handle": "row5"}, {**user("next"), "handle": "row6"}]
        clean = hook.compact(engine, {"messages": resumed, "session": "s"})
        self.assertEqual(clean["messages"], [{"ref": 0}, {"ref": 3}, {"ref": 4}, {"ref": 5}, {"ref": 6}])
        self.assertEqual(hook.compact(engine, {"messages": [resumed[n] for n in (0, 3, 4, 5, 6)], "session": "s"}),
                         {"skip": "answers: not now"})

    def test_a_compaction_the_breaker_would_end_the_turn_for_is_held_unless_forced(self) -> None:
        engine = Engine(Compaction(threshold=30_000, min_gain=0))
        request = {"messages": [user("read the files"), call(1), result(1)], "model": "m", "window": 200_000, "hold": True}
        self.assertEqual(ClaudeCodeHook().compact(engine, {**request, "tokens": 40_000}),
                         {"skip": "compaction: held, two compactions in a row would end Claude Code's turn"})
        self.assertIn("summarize", ClaudeCodeHook().compact(engine, {**request, "tokens": 168_000}))  # (forced)

    def test_the_instructions_are_the_strategys_own(self) -> None:
        self.assertIn("internal state", ClaudeCodeHook().instructions(Engine(MEM1())))
        self.assertIn("branch", ClaudeCodeHook().instructions(Engine(ContextFolding())).lower())
        self.assertEqual(ClaudeCodeHook().instructions(Engine(Compaction())), "")


@dataclass(frozen=True)
class Answers:
    answer: Any
    name = "answers"

    def fingerprint(self):
        return {}

    def plan(self, request, summarizer):
        return self.answer(request)


class EndpointTests(unittest.TestCase):
    def test_the_plugin_asks_for_summaries_then_gets_the_transcript(self) -> None:
        client = TestClient(create_app(Engine(Compaction(threshold=30_000, min_gain=0))))
        request = {"harness": "claude_code", "model": "m", "tokens": 40_000, "messages": [user("read"), call(1), result(1)]}
        asked = client.post("/relay/v1/compact", json=request).json()["summarize"]
        done = client.post("/relay/v1/compact", json={**request, "summaries": {asked["key"]: "c1"}}).json()
        self.assertEqual(done["messages"][0], {"ref": 0})
        self.assertEqual(client.post("/relay/v1/compact", json={"harness": "nope"}).status_code, 500)

    def test_the_instructions_and_what_the_hook_cannot_run(self) -> None:
        self.assertIn("## Branches", TestClient(create_app(Engine(ContextFolding()))).post(
            "/relay/v1/instructions", json={"harness": "claude_code"}).json()["instructions"])
        self.assertEqual(TestClient(create_app(Engine(MEM1()))).post(  # (a new context every request)
            "/relay/v1/instructions", json={"harness": "claude_code"}).status_code, 409)
        from relay.strategies import ContextLanguageModel
        refused = TestClient(create_app(Engine(ContextLanguageModel()))).post(
            "/relay/v1/instructions", json={"harness": "claude_code"})
        self.assertEqual(refused.status_code, 409)
        self.assertIn("clm needs notes, which the hook cannot carry", refused.json()["error"]["message"])


if __name__ == "__main__":
    unittest.main()
