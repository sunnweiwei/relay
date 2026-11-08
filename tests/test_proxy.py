from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
import unittest.mock
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from starlette.responses import JSONResponse

from relay import Compaction, Engine, PrefixStore, ProxyConfig, create_app, providers
from relay.prompts import SUMMARY_PREFIX
from relay.transport import engine_from_settings
from relay.core.budget import InputBudget
from relay.core.ir import Context, Item, Kind, Request
from relay.strategies import ContextLanguageModel
from relay.strategies.base import Strategy, Summarizer
from tests.fakes import COMPACTION_MARKER, FakeUpstream, _prompt_tokens, local_upstreams, relay_settings, serve

NEVER = Compaction(threshold=10**9)
# A local input budget no test request reaches: it selects the bounded-evaluation recovery path.
try:
    GENEROUS: InputBudget | None = InputBudget(1_000_000)
except Exception:  # (tiktoken, the `budget` extra, or its encoding is unavailable)
    GENEROUS = None


@contextmanager
def relay(fake: FakeUpstream, strategy: Strategy = NEVER, api_key: str | None = None,
          input_budget: InputBudget | None = None) -> Iterator[tuple[str, Engine]]:
    with serve(fake.app) as upstream:
        engine = Engine(strategy, PrefixStore(), input_budget=input_budget)
        config = ProxyConfig(upstreams=local_upstreams(upstream, api_key))
        with serve(create_app(engine, config)) as url:
            yield url, engine


def responses_body(steps: int, *, stream: bool = False) -> dict[str, Any]:
    items: list[dict[str, Any]] = [
        {"type": "message", "role": "developer", "content": "rules"},
        {"type": "message", "role": "user", "content": "task"},
    ]
    for i in range(steps):
        items += [{"type": "function_call", "call_id": f"c{i}", "name": "shell", "arguments": "{}"},
                  {"type": "function_call_output", "call_id": f"c{i}", "output": "x" * 2_000}]
    return {"model": "gpt-5", "input": items, "stream": stream, "tools": [{"type": "function", "name": "shell"}]}


def messages_body(steps: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "task"}]
    for i in range(steps):
        messages += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 2_000}]},
        ]
    return {"model": "claude-x", "max_tokens": 1000, "stream": True, "messages": messages,
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}]}


def last_user_text(items: list[dict[str, Any]]) -> str:
    content = items[-1].get("content") or ""
    return content if isinstance(content, str) else content[0]["text"]


HEADERS = {"authorization": "Bearer tenant"}


class ProxyTests(unittest.TestCase):
    def test_streams_pass_through_byte_for_byte_and_record_usage(self) -> None:
        body = responses_body(1, stream=True)
        direct = FakeUpstream()
        with serve(direct.app) as upstream:
            expected = httpx.post(f"{upstream}/v1/responses", json=body).content
        fake = FakeUpstream()
        with relay(fake) as (url, engine):
            relayed = httpx.post(f"{url}/v1/responses", json=body, headers=HEADERS)
        self.assertEqual(relayed.status_code, 200)
        self.assertEqual(relayed.content, expected)
        self.assertEqual(len(engine.store), 1)  # the reported usage was recorded

    def test_compacts_a_responses_request(self) -> None:
        fake = FakeUpstream()
        with relay(fake, Compaction(threshold=1_000)) as (url, _):
            response = httpx.post(f"{url}/v1/responses", json=responses_body(3), headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        summary, main = fake.bodies("/v1/responses")
        self.assertIn(COMPACTION_MARKER, last_user_text(summary["input"]))
        self.assertEqual(summary["tool_choice"], "none")
        self.assertEqual([i["role"] for i in main["input"]], ["developer", "user", "user"])
        self.assertTrue(last_user_text(main["input"]).startswith(SUMMARY_PREFIX))

    def test_compacts_a_streaming_messages_request(self) -> None:
        fake = FakeUpstream()
        with relay(fake, Compaction(threshold=1_000)) as (url, _):
            response = httpx.post(f"{url}/v1/messages", json=messages_body(3), headers={"x-api-key": "tenant"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"message_stop", response.content)
        summary, main = fake.bodies("/v1/messages")
        self.assertEqual((summary["stream"], summary["tool_choice"]), (False, {"type": "none"}))
        self.assertEqual(len(main["messages"]), 2)
        self.assertTrue(last_user_text(main["messages"]).startswith(SUMMARY_PREFIX))

    def test_an_upstream_rate_limit_is_waited_out_where_set(self) -> None:
        class Limited(FakeUpstream):
            refusals = 1

            async def _responses(self, request: Any) -> Any:
                if self.refusals:
                    self.refusals -= 1
                    return JSONResponse({"detail": "Rate limit exceeded"}, 429)
                return await super()._responses(request)

        for wait, status in (("0", 429), ("10", 200)):
            with self.subTest(wait=wait), relay_settings(upstream_429_wait=wait), relay(Limited()) as (url, _):
                self.assertEqual(httpx.post(f"{url}/v1/responses", json=responses_body(1), headers=HEADERS).status_code, status)

    def test_context_overflow_is_compacted_and_retried(self) -> None:
        body = responses_body(6)
        fake = FakeUpstream(max_prompt_tokens=2_000)
        with relay(fake) as (url, _):
            response = httpx.post(f"{url}/v1/responses", json=body, headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        requests = fake.bodies("/v1/responses")
        self.assertEqual(requests[0]["input"], body["input"])  # rejected as too long
        summaries = [r for r in requests if COMPACTION_MARKER in last_user_text(r["input"])]
        self.assertGreater(len(summaries), 1)  # the summary request itself was trimmed to fit
        self.assertTrue(last_user_text(requests[-1]["input"]).startswith(SUMMARY_PREFIX))

    def test_bounded_clm_guard_does_not_block_explicit_overflow_recovery(self) -> None:
        if GENEROUS is None:
            self.skipTest("the bounded path needs the budget extra")
        fake = FakeUpstream(max_prompt_tokens=2_000)
        with tempfile.TemporaryDirectory() as directory:
            strategy = ContextLanguageModel(budget=4_000, reserve=2_000, directory=directory)
            with relay(fake, strategy, input_budget=GENEROUS) as (url, _):
                response = httpx.post(f"{url}/v1/responses", json=responses_body(10), headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        requests = fake.bodies("/v1/responses")
        main = [r for r in requests if COMPACTION_MARKER not in last_user_text(r["input"])]
        self.assertEqual(len(main), 2)  # first guarded request, then one recovered request
        self.assertIn("[context manager: overflow guard]", json.dumps(main[0]))
        self.assertIn(SUMMARY_PREFIX, json.dumps(main[1]))
        self.assertGreater(len(requests), len(main))

    def test_bounded_overflow_recovery_rejects_noops_and_never_retries_twice(self) -> None:
        if GENEROUS is None:
            self.skipTest("the bounded path needs the budget extra")
        for noop in (True, False):
            with self.subTest(noop=noop):
                forces: list[bool] = []

                class Rewriter:
                    name = "test-rewriter"

                    def fingerprint(self) -> dict[str, Any]:
                        return {"name": self.name}

                    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
                        forces.append(request.force)
                        if request.force and noop:
                            return None
                        return Context((*request.current, Item(Kind.ASSISTANT, "x" * 8_000)))

                fake = FakeUpstream(max_prompt_tokens=1_000)
                with relay(fake, Rewriter(), input_budget=GENEROUS) as (url, _):
                    response = httpx.post(f"{url}/v1/responses", json=responses_body(0), headers=HEADERS)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(forces, [False, True])
                self.assertEqual(len(fake.bodies("/v1/responses")), 1 if noop else 2)

    def test_the_limit_an_upstream_names_as_it_refuses_is_the_window_from_then_on(self) -> None:
        """(An account's limit below the model's window: one refusal, then compacted before it.)"""

        self.addCleanup(providers.REFUSED.clear)
        fake, headers = FakeUpstream(max_prompt_tokens=2_000), {"x-api-key": "tenant"}
        with relay(fake, Compaction()) as (url, _):  # (at 90% of the window: claude-x's 200k, until refused)
            for task in ("task", "another task"):  # (two conversations of the model)
                body = messages_body(6)
                body["messages"][0]["content"] = task
                response = httpx.post(f"{url}/v1/messages", json={**body, "stream": False}, headers=headers)
                self.assertEqual(response.status_code, 200)
        refused = [b for b in fake.bodies("/v1/messages") if _prompt_tokens(b) > 2_000]
        self.assertEqual([b["messages"][0]["content"] for b in refused], ["task"])

    def test_a_summary_request_whose_connection_drops_is_retried(self) -> None:
        fake = FakeUpstream()
        with serve(fake.app) as upstream:
            app = create_app(Engine(Compaction(threshold=1_000), PrefixStore()),
                             ProxyConfig(upstreams=local_upstreams(upstream)))
            with serve(app) as url:
                client, calls = app.state.summary_client, []
                original = client.post

                def flaky(*args: Any, **kwargs: Any) -> httpx.Response:
                    calls.append(1)
                    if len(calls) == 1:
                        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
                    return original(*args, **kwargs)

                client.post = flaky
                response = httpx.post(f"{url}/v1/responses", json=responses_body(3), headers=HEADERS, timeout=30)
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(last_user_text(fake.bodies("/v1/responses")[-1]["input"]).startswith(SUMMARY_PREFIX))

    def test_failed_compaction_forwards_the_original_request(self) -> None:
        body = responses_body(3)
        fake = FakeUpstream(fail_summaries=True)
        with relay(fake, Compaction(threshold=1_000)) as (url, _), self.assertLogs("relay", "WARNING"):
            response = httpx.post(f"{url}/v1/responses", json=body, headers=HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake.bodies("/v1/responses")[-1]["input"], body["input"])

    def test_a_request_relay_cannot_read_is_forwarded_as_it_is(self) -> None:
        body = {"model": "m", "messages": ["not a message"], "stream": False}
        fake = FakeUpstream()
        with relay(fake) as (url, _), self.assertLogs("relay", "WARNING"):
            httpx.post(f"{url}/v1/messages", json=body, headers={"x-claude-code-session-id": "s"})
        self.assertEqual(fake.bodies("/v1/messages")[-1], body)  # (the upstream's to answer)

    def test_other_routes_pass_through(self) -> None:
        fake = FakeUpstream()
        with relay(fake) as (url, _):
            self.assertEqual(httpx.get(f"{url}/v1/models").status_code, 404)
            httpx.get(f"{url}/api/hello")
        self.assertEqual([r["path"] for r in fake.requests], ["/v1/models", "/api/hello"])

    def test_the_trace_counts_items_as_the_harness_codec_reads_them(self) -> None:
        body = messages_body(1)
        body["messages"][-1]["content"].append({"type": "text", "text": "Current runtime context. cwd=/w"})
        fake = FakeUpstream()
        with tempfile.TemporaryDirectory() as tmp, serve(fake.app) as upstream:
            trace = Path(tmp) / "trace.jsonl"
            config = ProxyConfig(upstreams=local_upstreams(upstream), harness="deepseek_harness", trace=str(trace))
            with serve(create_app(Engine(NEVER, PrefixStore()), config)) as url:
                httpx.post(f"{url}/v1/messages", json=body, headers={"x-api-key": "tenant"})
            record = json.loads(trace.read_text().splitlines()[0])
        self.assertEqual((record["items"], record["sent"]), (4, 4))  # its tool results and its snapshot apart

    def test_a_leaving_harness_goes_as_it_sends_until_a_conversation_needs_relay(self) -> None:
        from relay import install, leaving
        install.install("generic", "http://127.0.0.1:8787", [])
        leaving.leave("generic")
        fake = FakeUpstream()
        with serve(fake.app) as upstream:
            config = ProxyConfig(upstreams=local_upstreams(upstream), harness="generic")
            engine = Engine(Compaction(threshold=1_000), PrefixStore(), window=100_000)
            with serve(create_app(engine, config)) as url:
                httpx.post(f"{url}/v1/responses", json=responses_body(3), headers=HEADERS)
                self.assertEqual(fake.bodies("/v1/responses"), [responses_body(3)])  # fits: as it was sent
                self.assertEqual(list(leaving.leaving()), [])  # (and no conversation holds it on Relay)
                self.assertEqual(install.installed(), [])

    def test_a_conversation_compacted_by_the_harness_after_a_strategy_change_hands_it_back(self) -> None:
        with relay_settings():
            from relay import install, leaving
            from relay.harnesses.claude_code import COMPACT_PROMPT
            install.install("claude_code", "http://127.0.0.1:8787", [])
            fake = FakeUpstream()
            long = messages_body(0)
            for i in range(4):  # ~640k tokens: past what goes direct even at Claude Code's 1M models' allowance
                long["messages"] += [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x " * 200_000}]}]
            headers = {"x-api-key": "tenant", "x-claude-code-session-id": "s"}
            with serve(fake.app) as upstream:
                config = ProxyConfig(upstreams=local_upstreams(upstream), harness="claude_code")
                engine = Engine(Compaction(threshold=1_000, min_gain=0), PrefixStore())
                with serve(create_app(engine, config)) as url:
                    httpx.post(f"{url}/v1/messages", json=long, headers=headers)  # used, compacted, under one strategy
                    leaving.leave("claude_code")
                    engine.strategy = Compaction(threshold=2_000, min_gain=0)  # then another (`relay strategy`)
                    self.assertEqual(len(leaving.needing("claude_code")), 1)
                    compact = {**long, "messages": [*long["messages"], {"role": "assistant", "content": "done"},
                                                    {"role": "user", "content": f"{COMPACT_PROMPT} the conversation"}]}
                    httpx.post(f"{url}/v1/messages", json=compact, headers=headers)  # its own `/compact`
                    sent = json.dumps(fake.bodies("/v1/messages")[-1])
                    self.assertLess(len(sent), 100_000)  # the context the model last saw, not the whole history
                    self.assertEqual(leaving.needing("claude_code"), [])  # handed back: the harness is restored
                    self.assertEqual(list(leaving.leaving()), [])
                    self.assertEqual(install.installed(), [])

    def test_a_leaving_claude_code_conversation_within_its_models_window_goes_direct(self) -> None:
        with relay_settings():
            from relay import install, leaving
            install.install("claude_code", "http://127.0.0.1:8787", [])
            leaving.leave("claude_code")
            fake = FakeUpstream()
            body = messages_body(0)
            body["messages"] += [  # ~216k tokens: past 90% of the catalog's 200k for Claude, within Claude Code's allowance
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Bash", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "x " * 270_000}]}]
            with serve(fake.app) as upstream:
                config = ProxyConfig(upstreams=local_upstreams(upstream), harness="claude_code")
                engine = Engine(Compaction(threshold=1_000, min_gain=0), PrefixStore())
                with serve(create_app(engine, config)) as url:
                    httpx.post(f"{url}/v1/messages", json={**body, "model": "claude-opus-5-5"},
                               headers={"x-api-key": "tenant", "x-claude-code-session-id": "s"})
            self.assertEqual(fake.bodies("/v1/messages")[-1]["messages"], body["messages"])  # as it was sent
            self.assertEqual(install.installed(), [])

    def test_a_conversation_the_upstream_refuses_direct_stays_on_relay(self) -> None:
        # Within Claude Code's allowance, but the model's real window is smaller: the refusal, not the
        # allowance, decides. It is managed from then on, and holds the harness on Relay.
        with relay_settings():
            from relay import install, leaving
            install.install("claude_code", "http://127.0.0.1:8787", [])
            leaving.leave("claude_code")
            fake = FakeUpstream(max_prompt_tokens=100_000)
            body = messages_body(0)
            for i in range(6):  # ~216k tokens as Relay estimates them (~135k as the upstream counts)
                body["messages"] += [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x " * 45_000}]}]
            more = [*body["messages"], {"role": "assistant", "content": [{"type": "tool_use", "id": "t9", "name": "Bash", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "ok"}]}]
            headers = {"x-api-key": "tenant", "x-claude-code-session-id": "s"}
            with serve(fake.app) as upstream:
                config = ProxyConfig(upstreams=local_upstreams(upstream), harness="claude_code")
                engine = Engine(Compaction(threshold=1_000, min_gain=0), PrefixStore())
                with serve(create_app(engine, config)) as url:
                    first = httpx.post(f"{url}/v1/messages", json={**body, "stream": False}, headers=headers)
                    second = httpx.post(f"{url}/v1/messages", json={**body, "messages": more, "stream": False}, headers=headers)
            self.assertEqual((first.status_code, second.status_code), (200, 200))
            whole = [b for b in fake.bodies("/v1/messages") if len(json.dumps(b)) > 500_000 and "tool_choice" not in b]
            self.assertEqual(len(whole), 1)  # sent direct once, refused; not again
            self.assertEqual(len(leaving.needing("claude_code")), 1)
            self.assertEqual(install.installed(), ["claude_code"])  # (still on Relay)

    def test_a_harness_newer_than_its_profile_was_checked_against_is_served_and_noted(self) -> None:
        fake = FakeUpstream()
        with relay(fake) as (url, _):
            for agent in ("codex_exec/0.163.0 (Debian 12.0.0; x86_64)", "codex_exec/9.0.0 (Debian 12.0.0; x86_64)"):
                with self.assertLogs("relay", "WARNING") if agent.startswith("codex_exec/9") else contextlib.nullcontext():
                    response = httpx.post(f"{url}/v1/responses", json=responses_body(0),
                                          headers={**HEADERS, "user-agent": agent, "originator": "codex_exec"})
                self.assertEqual(response.status_code, 200)
            status = httpx.get(f"{url}/relay/v1/status").json()
        self.assertEqual(status["harnesses"]["codex"], {"release": "9.0.0", "checked": "0.162.0–0.163.0"})

    def test_configured_api_keys_replace_client_credentials(self) -> None:
        fake = FakeUpstream()
        with relay(fake, api_key="secret") as (url, _):
            httpx.post(f"{url}/v1/responses", json=responses_body(0), headers=HEADERS)
            httpx.post(f"{url}/v1/messages", json=messages_body(0), headers={"x-api-key": "tenant"})
        openai, anthropic = (r["headers"] for r in fake.requests)
        self.assertEqual(openai["authorization"], "Bearer secret")
        self.assertEqual((anthropic["x-api-key"], "authorization" in anthropic), ("secret", False))

    def test_the_cache_settings_reach_the_engine_that_serves(self) -> None:
        with relay_settings(cache_max_entries=50000, cache_max_bytes=1000000, cache_ttl_seconds=86400,
                            cache_path="off"):  # (not in the user's home)
            store = engine_from_settings().store
        self.assertEqual((store.max_entries, store.max_bytes, store.ttl_seconds), (50_000, 1_000_000, 86_400.0))


if __name__ == "__main__":
    unittest.main()
