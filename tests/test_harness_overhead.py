"""Offline checks for actual sizing corrections and preserved evaluation content."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import httpx
import pytest

from relay.core.budget import InputBudget
from relay.core.engine import Engine
from relay.core.ir import Item, Kind
from relay.core.tokens import counting_with, item_tokens, structured_tokens
from relay.harnesses import browse_profile
from relay.transport.proxy import ProxyConfig, create_app
from tests.test_engine import body, msg, summary
from tests.test_input_budget import CODEC, ForcedManager


def reasoning(value: bytes = b"opaque" * 1000) -> dict:
    return {"type": "reasoning", "encrypted_content": base64.b64encode(value).decode(), "summary": []}


def response(items: list[dict], tokens: int = 127) -> dict:
    return {**summary("fixture")[1], "output": items,
            "usage": {"input_tokens": 100, "output_tokens": tokens,
                      "output_tokens_details": {"reasoning_tokens": tokens}}}


def test_reasoning_uses_observed_usage_in_both_strategy_and_final_request() -> None:
    policy, wire = InputBudget(1000), reasoning()
    item = CODEC.classify(wire)
    with pytest.raises(ValueError, match="unknown"):
        policy.count(CODEC, body(wire))
    records = policy.observe_response(response([wire]))
    assert records[0]["tokens"] == 127 and records[0]["response_items"] == 1
    assert "encrypted_content" not in json.dumps(records)
    with counting_with(policy.text_tokens, policy.opaque_tokens):
        assert item_tokens(item) == 127 + 8
        assert policy.count(CODEC, body(wire)) == item_tokens(item) + 3
    assert policy.opaque_tokens(item) != item.opaque // 4


def test_aggregate_reasoning_usage_remains_an_explicit_upper_bound_for_partial_histories() -> None:
    policy, a, b = InputBudget(1000), reasoning(b"a" * 100), reasoning(b"b" * 100)
    records = policy.observe_response(response([a, b], 20))
    assert all(r["response_items"] == 2 for r in records)
    assert policy.opaque_tokens(CODEC.classify(a)) == policy.opaque_tokens(CODEC.classify(b)) == 20
    assert policy.count(CODEC, body(a, b)) == 3 + 2 * (20 + 8)
    assert policy.count(CODEC, body(b)) == 3 + 20 + 8
    with pytest.raises(ValueError, match="Conflicting"):
        policy.observe_response(response([a], 10))


def test_zero_reasoning_usage_is_known_and_missing_usage_does_not_become_zero() -> None:
    policy, wire = InputBudget(1000), reasoning()
    assert policy.observe_response({"output": [wire]}) == []
    with pytest.raises(ValueError, match="unknown"):
        policy.opaque_tokens(CODEC.classify(wire))
    policy.observe_response(response([wire], 0))
    assert policy.opaque_tokens(CODEC.classify(wire)) == 0


def test_decoded_tool_fields_share_the_strategy_counter_without_changing_the_payload() -> None:
    policy = InputBudget(10000)
    declaration = {"type": "additional_tools", "role": "developer", "tools": [
        {"name": "exec", "description": 'line one\nline two\n"quoted"\\path\n' * 80,
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]}
    wire_before = json.dumps(declaration)
    classified = CODEC.classify(declaration)
    with counting_with(policy.text_tokens, policy.opaque_tokens):
        expected = structured_tokens(declaration, policy.text_tokens) + 8
        assert item_tokens(classified) == expected
        assert policy.count(CODEC, body(declaration)) == expected + 3
    assert json.dumps(declaration) == wire_before


def test_profile_preserves_tools_permissions_task_method_prompt_and_rejects_unknown_codex_prompt(monkeypatch) -> None:
    generic = browse_profile.CODING_PREFIX + " is to collaborate."
    monkeypatch.setattr(browse_profile, "CODING_SHA256", hashlib.sha256(generic.encode()).hexdigest())
    permissions = "<permissions instructions>Keep all deny paths and approval=never.</permissions instructions>"
    bundle = {"role": "developer", "content": [
        {"type": "input_text", "text": "<skills_instructions>large catalog</skills_instructions>"},
        {"type": "input_text", "text": permissions},
        {"type": "input_text", "text": "<recommended_plugins>unused</recommended_plugins>"},
        {"type": "input_text", "text": "<collaboration_mode>Keep mode settings.</collaboration_mode>"},
        {"type": "input_text", "text": "<multi_agent_mode>Do not spawn without explicit instructions.</multi_agent_mode>"},
        {"type": "input_image", "image_url": "untouched"}]}
    tool = {"type": "additional_tools", "role": "developer", "tools": [{"name": "exec"}]}
    generic_message = {"role": "developer", "content": [{"type": "input_text", "text": generic}]}
    request = body(tool, generic_message, bundle,
                   msg("developer", "Method-specific instructions"), msg("user", "Task and environment"))
    frozen = json.dumps(request)
    lean = browse_profile.apply_profile(request, browse_profile.PROFILE)
    assert json.dumps(request) == frozen
    assert lean["input"][0] == tool and lean["input"][-2:] == request["input"][-2:]
    assert lean["input"][2]["content"] == bundle["content"][1:2] + bundle["content"][3:]
    assert "large catalog" not in json.dumps(lean)
    assert browse_profile.BROWSE_INSTRUCTIONS in json.dumps(lean)
    assert browse_profile.apply_profile(lean, browse_profile.PROFILE) == lean
    other = body(msg("developer", "Unrelated developer text."))
    assert browse_profile.apply_profile(other, browse_profile.PROFILE) == other
    unknown = body({"role": "developer", "content": [
        {"type": "input_text", "text": generic + " Unrecognized extra rule."}]})
    with pytest.raises(ValueError, match="unrecognized Codex coding instructions"):
        browse_profile.apply_profile(unknown, browse_profile.PROFILE)
    assert browse_profile.apply_profile(request, None) is request


@pytest.mark.parametrize("streaming", [False, True])
def test_main_response_usage_is_available_before_the_next_request(streaming: bool) -> None:
    async def run() -> None:
        policy, wire = InputBudget(1000), reasoning()
        manager = ForcedManager("noop", [])
        app = create_app(Engine(manager, input_budget=policy), ProxyConfig())
        sent = []

        def handle(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            policy.check(CODEC, payload, "fake")
            sent.append(payload)
            output = response([wire, msg("assistant", "fixture")])
            if not streaming:
                return httpx.Response(200, json=output)
            # Exercise the fallback for a terminal event without repeated output.
            events = [{"type": "response.output_item.done", "item": wire},
                      {"type": "response.completed", "response": {**output, "output": []}}]
            content = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
            return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

        async with app.router.lifespan_context(app):
            await app.state.client.aclose()
            app.state.summary_client.close()
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            app.state.summary_client = httpx.Client(transport=httpx.MockTransport(handle))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                first = await client.post("/v1/responses", json={**body(msg("user", "task")), "stream": streaming})
                assert first.status_code == 200
                assert policy.opaque_tokens(CODEC.classify(wire)) == 127
                second = await client.post("/v1/responses", json={**body(msg("user", "task"), wire,
                    msg("assistant", "fixture"), msg("user", "continue")), "stream": streaming})
                assert second.status_code == 200 and len(sent) == 2
    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
def test_auxiliary_response_also_registers_reasoning_usage(streaming: bool) -> None:
    async def run() -> None:
        policy, wire = InputBudget(1000), reasoning()

        class Auxiliary:
            name = "auxiliary-test"
            def fingerprint(self): return {"name": self.name}
            def plan(self, request, lm):
                assert lm.complete([Item(Kind.USER, "auxiliary fixture")]) == "fixture"
                assert policy.opaque_tokens(CODEC.classify(wire)) == 127

        app = create_app(Engine(Auxiliary(), input_budget=policy), ProxyConfig())
        calls = []

        def handle(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            policy.check(CODEC, calls[-1], "fake")
            output = response([wire, {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "fixture"}]}]) if len(calls) == 1 else summary("done")[1]
            if streaming:
                return httpx.Response(200, content="data: " + json.dumps({
                    "type": "response.completed", "response": output}) + "\n\n",
                    headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=output)

        async with app.router.lifespan_context(app):
            await app.state.client.aclose()
            app.state.summary_client.close()
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            app.state.summary_client = httpx.Client(transport=httpx.MockTransport(handle))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                result = await client.post("/v1/responses", json=body(msg("user", "task")))
                assert result.status_code == 200 and len(calls) == 2
    asyncio.run(run())
