from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI

from relay import Compact, PrefixCheckpointCache
from relay.proxy import ProxyConfig, create_app
from relay.strategies.compact import CODEX_COMPACTION_PROMPT


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


class FakeInputTokens:
    def __init__(self, owner: "FakeManagementResponses") -> None:
        self.owner = owner

    def count(self, **request):
        self.owner.count_calls.append(request)
        input_items = request.get("input", [])
        is_summary = bool(
            input_items and input_items[-1].get("content") == CODEX_COMPACTION_PROMPT
        )
        return SimpleNamespace(input_tokens=1 if is_summary else self.owner.token_count)


class FakeManagementResponses:
    def __init__(self, token_count: int = 500) -> None:
        self.token_count = token_count
        self.count_calls: list[dict] = []
        self.create_calls: list[dict] = []
        self.input_tokens = FakeInputTokens(self)

    def create(self, **request):
        self.create_calls.append(request)
        last = request.get("input", [{}])[-1]
        if last.get("content") == CODEX_COMPACTION_PROMPT:
            return SimpleNamespace(output=[], output_text="portable summary")
        raise AssertionError("unexpected management request")


def response_json(body: dict) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": body["model"],
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "task result",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
    }


def sse_response(body: dict) -> bytes:
    response = response_json(body)
    item = response["output"][0]
    events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.in_progress",
            "sequence_number": 1,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 0,
            "item": item,
        },
        {
            "type": "response.completed",
            "sequence_number": 4,
            "response": response,
        },
    ]
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in events
    )


def parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    ]


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.upstream_calls: list[tuple[str, dict | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            self.upstream_calls.append((request.url.path, body))
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": []})
            assert body is not None
            if body.get("stream"):
                return httpx.Response(
                    200,
                    content=sse_response(body),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=response_json(body))

        self.transport = httpx.MockTransport(handler)
        self.config = ProxyConfig(
            upstream_base_url="https://upstream.test/v1",
            upstream_api_key="test-key",
        )

    async def test_json_create_is_intercepted_and_checkpoint_replays(self) -> None:
        management = FakeManagementResponses(token_count=500)
        app = create_app(
            Compact(compact_threshold=100),
            self.config,
            upstream_transport=self.transport,
            management_responses=management,
        )
        trajectory = [message("user", "long task")]
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                first = (
                    await client.post(
                        "/v1/responses",
                        json={
                            "model": "test",
                            "input": trajectory,
                            "store": False,
                            "context_management": [
                                {"type": "compaction", "compact_threshold": 100}
                            ],
                        },
                    )
                ).json()
                self.assertEqual(first["output"][0]["type"], "compaction")
                trajectory.extend(first["output"])
                trajectory.append(message("user", "continue"))
                management.token_count = 1
                second = await client.post(
                    "/v1/responses",
                    json={"model": "test", "input": trajectory, "store": False},
                )
                self.assertEqual(second.status_code, 200)

        first_upstream = self.upstream_calls[0][1]
        second_upstream = self.upstream_calls[1][1]
        assert first_upstream is not None and second_upstream is not None
        self.assertNotIn("context_management", first_upstream)
        self.assertFalse(
            any(item.get("type") == "compaction" for item in second_upstream["input"])
        )
        self.assertIn("portable summary", second_upstream["input"][1]["content"])
        self.assertEqual(second_upstream["input"][-1]["content"], "continue")

    async def test_cache_mode_is_transparent_and_reuses_an_exact_prefix(self) -> None:
        management = FakeManagementResponses(token_count=500)
        cache = PrefixCheckpointCache(secret=b"test-secret")
        config = ProxyConfig(
            upstream_base_url="https://upstream.test/v1",
            upstream_api_key="test-key",
            checkpoint_mode="cache",
        )
        app = create_app(
            Compact(compact_threshold=100),
            config,
            upstream_transport=self.transport,
            management_responses=management,
            checkpoint_cache=cache,
        )
        trajectory = [message("user", "long task")]
        headers = {"authorization": "Bearer tenant-a"}
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                first = (
                    await client.post(
                        "/v1/responses",
                        headers=headers,
                        json={
                            "model": "test",
                            "input": trajectory,
                            "store": False,
                        },
                    )
                ).json()
                self.assertFalse(
                    any(item.get("type") == "compaction" for item in first["output"])
                )
                trajectory.extend(first["output"])
                trajectory.append(message("user", "continue"))
                management.token_count = 1
                second = await client.post(
                    "/v1/responses",
                    headers=headers,
                    json={"model": "test", "input": trajectory, "store": False},
                )
                self.assertEqual(second.status_code, 200)

        second_upstream = self.upstream_calls[1][1]
        assert second_upstream is not None
        self.assertIn("portable summary", second_upstream["input"][1]["content"])
        self.assertEqual(second_upstream["input"][-1]["content"], "continue")
        self.assertEqual(cache.stats().hits, 1)

    async def test_cache_mode_requires_a_tenant_identity(self) -> None:
        config = ProxyConfig(
            upstream_base_url="https://upstream.test/v1",
            upstream_api_key="test-key",
            checkpoint_mode="cache",
        )
        app = create_app(
            Compact(),
            config,
            upstream_transport=self.transport,
            management_responses=FakeManagementResponses(),
            checkpoint_cache=PrefixCheckpointCache(secret=b"test-secret"),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                response = await client.post(
                    "/v1/responses",
                    json={"model": "test", "input": [message("user", "hello")]},
                )
        self.assertEqual(response.status_code, 400)
        self.assertIn("per-tenant Bearer", response.json()["error"]["message"])

    async def test_cache_mode_does_not_inject_sse_checkpoint_events(self) -> None:
        config = ProxyConfig(
            upstream_base_url="https://upstream.test/v1",
            upstream_api_key="test-key",
            checkpoint_mode="cache",
        )
        app = create_app(
            Compact(compact_threshold=100),
            config,
            upstream_transport=self.transport,
            management_responses=FakeManagementResponses(token_count=500),
            checkpoint_cache=PrefixCheckpointCache(secret=b"test-secret"),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                text = (
                    await client.post(
                        "/v1/responses",
                        headers={"authorization": "Bearer tenant-a"},
                        json={
                            "model": "test",
                            "input": [message("user", "long")],
                            "stream": True,
                        },
                    )
                ).text

        events = parse_sse(text)
        self.assertFalse(
            any(event.get("item", {}).get("type") == "compaction" for event in events)
        )
        self.assertFalse(
            any(
                item.get("type") == "compaction"
                for item in events[-1]["response"]["output"]
            )
        )

    async def test_pre_compaction_rewrites_sse_indices_and_completed_output(
        self,
    ) -> None:
        management = FakeManagementResponses(token_count=500)
        app = create_app(
            Compact(compact_threshold=100),
            self.config,
            upstream_transport=self.transport,
            management_responses=management,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                text = (
                    await client.post(
                        "/v1/responses",
                        json={
                            "model": "test",
                            "input": [message("user", "long")],
                            "stream": True,
                        },
                    )
                ).text
        events = parse_sse(text)
        marker_added = next(
            event
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "compaction"
        )
        task_added = next(
            event
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "message"
        )
        completed = events[-1]
        self.assertEqual(marker_added["output_index"], 0)
        self.assertEqual(task_added["output_index"], 1)
        self.assertEqual(completed["response"]["output"][0]["type"], "compaction")
        self.assertEqual(completed["response"]["output"][1]["type"], "message")
        self.assertEqual(
            [event["sequence_number"] for event in events],
            list(range(len(events))),
        )

    async def test_non_responses_endpoint_is_passed_through(self) -> None:
        app = create_app(
            Compact(),
            self.config,
            upstream_transport=self.transport,
            management_responses=FakeManagementResponses(),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                response = await client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"object": "list", "data": []})
        self.assertEqual(self.upstream_calls[0][0], "/v1/models")

    async def test_openai_sdk_only_needs_a_proxy_base_url(self) -> None:
        management = FakeManagementResponses(token_count=500)
        app = create_app(
            Compact(compact_threshold=100),
            self.config,
            upstream_transport=self.transport,
            management_responses=management,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as http_client:
                sdk = AsyncOpenAI(
                    api_key="agent-key",
                    base_url="http://proxy.test/v1",
                    http_client=http_client,
                )
                response = await sdk.responses.create(
                    model="test",
                    input=[message("user", "long")],
                    store=False,
                )
                self.assertEqual(response.output[0].type, "compaction")
                self.assertEqual(response.output[1].type, "message")

                stream = await sdk.responses.create(
                    model="test",
                    input=[message("user", "another long request")],
                    store=False,
                    stream=True,
                )
                events = [event async for event in stream]
                completed = events[-1]
                self.assertEqual(completed.type, "response.completed")
                self.assertEqual(completed.response.output[0].type, "compaction")
                self.assertEqual(completed.response.output[1].type, "message")


if __name__ == "__main__":
    unittest.main()
