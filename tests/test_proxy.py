from __future__ import annotations

import os
import unittest
import unittest.mock
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from relay import Compaction, Engine, PrefixStore, ProxyConfig, create_app
from relay.prompts import SUMMARY_PREFIX
from relay.transport import engine_from_env
from tests.fakes import COMPACTION_MARKER, FakeUpstream, local_upstreams, serve

NEVER = Compaction(threshold=10**9)


@contextmanager
def relay(fake: FakeUpstream, strategy: Compaction = NEVER, api_key: str | None = None) -> Iterator[tuple[str, Engine]]:
    with serve(fake.app) as upstream:
        engine = Engine(strategy, PrefixStore())
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

    def test_configured_api_keys_replace_client_credentials(self) -> None:
        fake = FakeUpstream()
        with relay(fake, api_key="secret") as (url, _):
            httpx.post(f"{url}/v1/responses", json=responses_body(0), headers=HEADERS)
            httpx.post(f"{url}/v1/messages", json=messages_body(0), headers={"x-api-key": "tenant"})
        openai, anthropic = (r["headers"] for r in fake.requests)
        self.assertEqual(openai["authorization"], "Bearer secret")
        self.assertEqual((anthropic["x-api-key"], "authorization" in anthropic), ("secret", False))

    def test_the_cache_settings_reach_the_engine_that_serves(self) -> None:
        settings = {"RELAY_CACHE_MAX_ENTRIES": "50000", "RELAY_CACHE_MAX_BYTES": "1000000",
                    "RELAY_CACHE_TTL_SECONDS": "86400", "RELAY_CACHE_PATH": "off"}  # (not in the user's home)
        with unittest.mock.patch.dict(os.environ, settings):
            store = engine_from_env().store
        self.assertEqual((store.max_entries, store.max_bytes, store.ttl_seconds), (50_000, 1_000_000, 86_400.0))


if __name__ == "__main__":
    unittest.main()
