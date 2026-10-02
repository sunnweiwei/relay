from __future__ import annotations

import json
import asyncio
import unittest
from unittest.mock import patch

import httpx

from relay.gemini_manager import GeminiManagementResponses
from relay.proxy import ProxyConfig, create_app
from relay.strategies.checkpoint import Checkpoint
from relay.strategies.sliding_window import SlidingWindow


class GeminiManagementResponsesTests(unittest.TestCase):
    def test_count_and_summary_use_native_gemini_and_same_key(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            self.assertEqual(request.headers["x-goog-api-key"], "test-key")
            payload = json.loads(request.content)
            self.assertIn("compaction", payload["contents"][0]["parts"][0]["text"])
            if request.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 123})
            self.assertTrue(request.url.path.endswith(":generateContent"))
            self.assertIn("systemInstruction", payload)
            return httpx.Response(200, json={"candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": "Marker: BLUE-ORCHID-731"}]},
            }]})

        manager = GeminiManagementResponses(
            "https://generativelanguage.googleapis.com",
            "test-key",
            "gemini-3.8-flash",
            transport=httpx.MockTransport(handler),
        )
        try:
            items = [{"role": "user", "content": "compaction"}]
            self.assertEqual(manager.input_tokens.count(input=items).input_tokens, 123)
            self.assertEqual(manager.create(input=items).output_text, "Marker: BLUE-ORCHID-731")
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                calls[0].url.path,
                "/v1beta/models/gemini-3.8-flash:countTokens",
            )
        finally:
            manager.close()

    def test_transient_503_retries_without_changing_request(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.content)
            if len(requests) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": "Summary"}]},
            }]})

        manager = GeminiManagementResponses(
            "https://generativelanguage.googleapis.com",
            "test-key",
            "gemini-3.8-flash",
            transport=httpx.MockTransport(handler),
        )
        try:
            with patch("relay.gemini_manager.time.sleep") as sleep:
                result = manager.create(input=[{"role": "user", "content": "hello"}])
            self.assertEqual(result.output_text, "Summary")
            self.assertEqual(requests[0], requests[1])
            sleep.assert_called_once()
        finally:
            manager.close()

    def test_exhausted_management_network_error_is_retryable_for_cli(self):
        attempts = []

        def unavailable(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            raise httpx.ConnectError("temporary network failure", request=request)

        manager = GeminiManagementResponses(
            "https://generativelanguage.googleapis.com", "test-key",
            "gemini-3.8-flash", transport=httpx.MockTransport(unavailable),
        )
        app = create_app(
            SlidingWindow(max_input_tokens=1000),
            ProxyConfig(upstream_base_url="https://generativelanguage.googleapis.com",
                        checkpoint_mode="cache", gemini_enabled=True),
            management_responses=manager,
        )

        async def run():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://relay.local",
                ) as client:
                    return await client.post(
                        "/v1beta/models/gemini-3.8-flash:generateContent",
                        headers={"x-goog-api-key": "test-key"},
                        json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
                    )

        try:
            with patch("relay.gemini_manager.time.sleep"):
                response = asyncio.run(run())
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"],
                             "relay_gemini_upstream_unavailable")
            self.assertEqual(len(attempts), 3)
        finally:
            manager.close()

    def test_native_route_uses_gemini_management_without_separate_credentials(self):
        requests = []

        def management(request: httpx.Request) -> httpx.Response:
            requests.append(("management", request))
            return httpx.Response(200, json={"totalTokens": 10})

        def upstream(request: httpx.Request) -> httpx.Response:
            requests.append(("upstream", request))
            return httpx.Response(200, json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": "OK"}]}
            }]})

        manager_transport = httpx.MockTransport(management)
        real_manager = GeminiManagementResponses

        def manager_factory(base_url, api_key, model):
            return real_manager(base_url, api_key, model, transport=manager_transport)

        app = create_app(
            SlidingWindow(max_input_tokens=1000),
            ProxyConfig(
                upstream_base_url="https://generativelanguage.googleapis.com",
                checkpoint_mode="cache",
                gemini_enabled=True,
            ),
            upstream_transport=httpx.MockTransport(upstream),
        )

        async def run():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://relay.local",
                ) as client:
                    return await client.post(
                        "/v1beta/models/gemini-3.8-flash:generateContent",
                        headers={"x-goog-api-key": "test-key"},
                        json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
                    )

        with patch("relay.proxy.GeminiManagementResponses", side_effect=manager_factory):
            response = asyncio.run(run())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([kind for kind, _ in requests], ["management", "upstream"])
        for _, request in requests:
            self.assertEqual(request.headers["x-goog-api-key"], "test-key")

    def test_checkpoint_replaces_old_native_history(self):
        sent = []
        manager_methods = []

        def management(request: httpx.Request) -> httpx.Response:
            manager_methods.append(request.url.path.rsplit(":", 1)[-1])
            if request.url.path.endswith(":generateContent"):
                return httpx.Response(200, json={"candidates": [{
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "Marker BLUE-ORCHID-731"}]},
                }]})
            prompt = json.loads(request.content)["contents"][0]["parts"][0]["text"]
            items = json.loads(prompt.split("Conversation items (JSON, in order):\n", 1)[1])
            tokens = sum(len(str(item.get("content", ""))) // 4 + 10 for item in items)
            return httpx.Response(200, json={"totalTokens": tokens})

        def upstream(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": "OK"}]}
            }]})

        manager_transport = httpx.MockTransport(management)
        real_manager = GeminiManagementResponses

        def manager_factory(base_url, api_key, model):
            return real_manager(base_url, api_key, model, transport=manager_transport)

        app = create_app(
            Checkpoint(checkpoint_threshold=50, context_threshold=120),
            ProxyConfig(
                upstream_base_url="https://generativelanguage.googleapis.com",
                checkpoint_mode="cache",
                gemini_enabled=True,
            ),
            upstream_transport=httpx.MockTransport(upstream),
        )

        async def run():
            contents = []
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://relay.local",
                ) as client:
                    for prompt in [
                        "Remember BLUE-ORCHID-731",
                        "middle " * 30,
                        "latest " * 30,
                    ]:
                        contents.append({"role": "user", "parts": [{"text": prompt}]})
                        response = await client.post(
                            "/v1beta/models/gemini-3.8-flash:generateContent",
                            headers={"x-goog-api-key": "test-key"},
                            json={"contents": contents},
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        contents.append(response.json()["candidates"][0]["content"])

        with patch("relay.proxy.GeminiManagementResponses", side_effect=manager_factory):
            asyncio.run(run())
        self.assertIn("countTokens", manager_methods)
        self.assertIn("generateContent", manager_methods)
        self.assertLess(len(sent[-1]["contents"]), 5)
        self.assertIn("Marker BLUE-ORCHID-731", json.dumps(sent[-1]))
