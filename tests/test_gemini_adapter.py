from __future__ import annotations

import json
from importlib.util import find_spec
import unittest
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from relay.gemini import GeminiInput
from relay.proxy import ProxyConfig, create_app
from relay.strategies.sliding_window import SlidingWindow
from tests.harness_e2e_support import HarnessUpstream, STRATEGIES, strategy_for
from tests.test_codex_e2e import _serve


class GeminiAdapterTests(unittest.TestCase):
    def history(self):
        return {"contents": [
            {"role": "user", "parts": [{"text": "Preserve BLUE-ORCHID-731"}]},
            {"role": "model", "parts": [{"text": "Starting"},
                {"functionCall": {"name": "read", "args": {"path": "a"}}, "thoughtSignature": "signed"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "read", "response": {"output": "data"}}}]},
            {"role": "model", "parts": [{"text": "Finished"}]},
            {"role": "user", "parts": [{"text": "Turn 4: latest exact request"}]},
        ], "tools": [{"functionDeclarations": [{"name": "read"}]}],
            "systemInstruction": {"parts": [{"text": "Follow instructions"}]},
            "generationConfig": {"temperature": 0.2}}

    def test_round_trip_preserves_parts_signatures_and_config(self):
        body = self.history()
        adapter = GeminiInput(body, "manager")
        self.assertEqual(adapter.render(adapter.request["input"]), body)
        calls = [i for i in adapter.request["input"] if i["type"] == "function_call"]
        results = [i for i in adapter.request["input"] if i["type"] == "function_call_output"]
        self.assertEqual(calls[0]["call_id"], results[0]["call_id"])

    def test_replacement_removes_old_history_instead_of_merging_it_back(self):
        adapter = GeminiInput(self.history(), "manager")
        result = adapter.render([{"type": "message", "role": "user", "content": "MEMORY"},
                                 adapter.request["input"][-1]])
        self.assertEqual(result["contents"], [
            {"role": "user", "parts": [{"text": "MEMORY"}]}, self.history()["contents"][-1]])

    def test_cli_result_id_can_follow_model_call_without_id(self):
        body = {"contents": [
            {"role": "model", "parts": [{"functionCall": {"name": "read", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {
                "id": "read__call_123", "name": "read", "response": {"output": "ok"}}}]},
        ]}
        adapter = GeminiInput(body, "manager")
        call, result = adapter.request["input"]
        self.assertEqual(call["call_id"], result["call_id"])
        self.assertEqual(adapter.render(adapter.request["input"]), body)

    def test_unsupported_parts_and_orphan_results_fail_closed(self):
        for part in [{"inlineData": {"mimeType": "image/png", "data": "AA"}},
                     {"functionResponse": {"name": "missing", "response": {}}}]:
            with self.subTest(part=part), self.assertRaises(ValueError):
                GeminiInput({"contents": [{"role": "user", "parts": [part]}]}, "manager")

    def test_native_requests_execute_strategies_and_preserve_response(self):
        manager = HarnessUpstream()
        captured = []
        payload = {"candidates": [{"content": {"role": "model", "parts": [{"text": "NATIVE_OK"}]}}]}

        async def native(request):
            captured.append((request.url.path, await request.json(), dict(request.headers)))
            if request.url.path.endswith(":streamGenerateContent"):
                return Response("data: " + json.dumps(payload) + "\n\n", media_type="text/event-stream")
            return JSONResponse(payload)

        upstream = Starlette(routes=[Route("/{path:path}", native, methods=["POST"])])
        with _serve(manager.app) as management_url, _serve(upstream) as native_url:
            for name in STRATEGIES:
                with self.subTest(strategy=name):
                    if name == "rlm" and not find_spec("rlm"):
                        self.skipTest("Install optional dependency: pip install -e '.[rlm]'")
                    strategy = strategy_for(name)
                    app = create_app(strategy, ProxyConfig(
                        upstream_base_url=native_url, checkpoint_mode="cache", gemini_enabled=True,
                        management_base_url=management_url + "/v1", management_api_key="fake-manager",
                        management_model="relay-test-model"))
                    with patch.object(strategy, "prepare", wraps=strategy.prepare) as prepare, _serve(app) as url:
                        native_calls_before = len(captured)
                        manager_calls_before = len(manager.manager_requests)
                        response = httpx.post(url + "/v1beta/models/gemini-test:generateContent",
                                              headers={"x-goog-api-key": "fake-tenant"}, json=self.history())
                    if name == "prolong":
                        self.assertEqual(response.status_code, 400)
                        self.assertIn("context conversion completed", response.text)
                        self.assertIn("context_management", response.text)
                        self.assertEqual(prepare.call_count, 1)
                        self.assertGreater(len(manager.manager_requests), manager_calls_before)
                        self.assertEqual(len(captured), native_calls_before)
                    else:
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.json(), payload)
                        self.assertEqual(prepare.call_count, 1)
                        path, body, headers = captured[-1]
                        self.assertEqual(path, "/v1beta/models/gemini-test:generateContent")
                        self.assertEqual(headers["x-goog-api-key"], "fake-tenant")
                        self.assertEqual(body["generationConfig"], self.history()["generationConfig"])

    def test_sse_body_is_preserved(self):
        async def native(request):
            return Response('data: {"candidates": []}\n\n', media_type="text/event-stream")
        manager = HarnessUpstream()
        upstream = Starlette(routes=[Route("/{path:path}", native, methods=["POST"])])
        with _serve(manager.app) as management_url, _serve(upstream) as native_url:
            app = create_app(strategy_for("sliding_window"), ProxyConfig(
                upstream_base_url=native_url, checkpoint_mode="cache", gemini_enabled=True,
                management_base_url=management_url + "/v1", management_api_key="fake",
                management_model="relay-test-model"))
            with _serve(app) as url:
                response = httpx.post(url + "/v1beta/models/gemini-test:streamGenerateContent?alt=sse",
                                      headers={"x-goog-api-key": "fake"}, json=self.history())
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.text, 'data: {"candidates": []}\n\n')

    def test_strategy_really_replaces_native_context(self):
        captured = []
        async def native(request):
            captured.append(await request.json())
            return JSONResponse({"candidates": []})
        manager = HarnessUpstream()
        upstream = Starlette(routes=[Route("/{path:path}", native, methods=["POST"])])
        with _serve(manager.app) as management_url, _serve(upstream) as native_url:
            app = create_app(SlidingWindow(max_input_tokens=60), ProxyConfig(
                upstream_base_url=native_url, checkpoint_mode="cache", gemini_enabled=True,
                management_base_url=management_url + "/v1", management_api_key="fake",
                management_model="relay-test-model"))
            with _serve(app) as url:
                response = httpx.post(url + "/v1beta/models/gemini-test:generateContent",
                                      headers={"x-goog-api-key": "fake"}, json=self.history())
        self.assertEqual(response.status_code, 200, response.text)
        sent = json.dumps(captured[0]["contents"])
        self.assertIn("Turn 4: latest exact request", sent)
        self.assertNotIn("BLUE-ORCHID-731", sent)
        self.assertNotIn("functionCall", sent)
        self.assertEqual(captured[0]["tools"], self.history()["tools"])

    def test_growing_native_history_reuses_strategy_state(self):
        manager = HarnessUpstream()
        captured = []
        async def native(request):
            captured.append(await request.json())
            return JSONResponse({"candidates": []})
        upstream = Starlette(routes=[Route("/{path:path}", native, methods=["POST"])])
        with _serve(manager.app) as management_url, _serve(upstream) as native_url:
            for name in STRATEGIES:
                if name == "prolong":
                    continue
                with self.subTest(strategy=name):
                    if name == "rlm" and not find_spec("rlm"):
                        self.skipTest("Install optional dependency: pip install -e '.[rlm]'")
                    strategy = strategy_for(name)
                    app = create_app(strategy, ProxyConfig(
                        upstream_base_url=native_url, checkpoint_mode="cache", gemini_enabled=True,
                        management_base_url=management_url + "/v1", management_api_key="fake-manager",
                        management_model="relay-test-model"))
                    with _serve(app) as url:
                        for turn in range(1, 5):
                            body = self.history()
                            for earlier in range(1, turn):
                                body["contents"].extend([
                                    {"role": "model", "parts": [{"text": f"Completed turn {earlier}"}]},
                                    {"role": "user", "parts": [{"text": f"Turn {earlier + 4}: next"}]},
                                ])
                            response = httpx.post(
                                url + "/v1beta/models/gemini-test:generateContent",
                                headers={"x-goog-api-key": "fake"}, json=body)
                            self.assertEqual(response.status_code, 200, response.text)
                            sent = json.dumps(captured[-1]["contents"])
                            if name == "rlm":
                                self.assertIn("Result from the Recursive Language Model", sent)
                            else:
                                self.assertIn(f"Turn {turn + 3}", sent)
