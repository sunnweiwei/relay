"""No network/model calls: bounded requests must reach a fake transport only after checks."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from relay.core.budget import (MEDIA_INPUT_BUDGET_COUNTER, BudgetExceeded,
                               InputAccountingError, InputBudget)
from relay.core.engine import Engine, _Summarizer
from relay.core.ir import Context, Item, Kind
from relay.core.tokens import approx_tokens, counting_with, item_tokens, truncate_middle
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from relay.transport.proxy import ProxyConfig, create_app
from relay.transport.proxy import engine_from_settings
from tests.fakes import relay_settings
from tests.test_engine import body, msg, step, summary

CODEC = OpenAIResponses()


def png_header(width: int, height: int) -> str:
    data = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    return base64.b64encode(data).decode()


def image_message(width: int, height: int, detail: str | None = "high", mime: str = "image/png") -> dict[str, Any]:
    image = {"type": "input_image", "image_url": f"data:{mime};base64,{png_header(width, height)}"}
    if detail is not None:
        image["detail"] = detail
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}, image]}


def summarizer(policy: InputBudget, history: list[dict], post: Any, *, instructions: str = "") -> _Summarizer:
    return _Summarizer(CODEC, body(*history), lambda cut: history[:cut], 1, post,
                       policy.input_limit, 4, instructions, input_budget=policy)


def test_full_summary_prompt_and_instructions_are_counted_before_dispatch() -> None:
    policy, sent = InputBudget(100), []
    history = [msg("developer", "rules"), msg("user", "task"), *step(1, 10)]
    lm = summarizer(policy, history, lambda b: sent.append(b) or summary("short"),
                    instructions="guidance " * 120)
    with pytest.raises(BudgetExceeded, match="smallest legal"):
        lm.summarize(len(history), "prompt " * 120)
    assert sent == []


def test_summary_chunks_preserve_every_tool_pair_and_include_carryover() -> None:
    policy, sent = InputBudget(300), []
    history = [msg("developer", "rules"), msg("user", "task")]
    for n in range(5):
        history.extend(step(n, 450))

    def post(request: dict) -> tuple[int, Any]:
        policy.check(CODEC, request, "test")
        assert not CODEC.orphans(CODEC.items(request))
        sent.append(request)
        return summary("Recorded evidence " * 15)

    lm = summarizer(policy, history, post, instructions="preserve all facts " * 8)
    assert lm.summarize(len(history), "Please summarize " * 8)
    assert len(sent) > 1
    output_ids = [i["call_id"] for request in sent for i in request["input"]
                  if i.get("type") == "function_call_output"]
    assert output_ids == [f"c{n}" for n in range(5)]
    assert "Recorded evidence" in json.dumps(sent[1])


def test_unsplittable_pair_is_rejected_without_sending_it() -> None:
    policy, sent = InputBudget(100), []
    history = [msg("developer", "rules"), msg("user", "task"), *step(1, 5000)]
    lm = summarizer(policy, history, lambda b: sent.append(b) or summary("short"))
    with pytest.raises(BudgetExceeded, match="smallest legal"):
        lm.summarize(len(history), "summarize")
    assert sent == []  # No spend on the prefix when a later group cannot fit.


def test_completion_path_enforces_same_limit_without_adding_a_fallback() -> None:
    policy, sent = InputBudget(100), []
    lm = summarizer(policy, [], lambda b: sent.append(b) or summary("short"))
    with pytest.raises(BudgetExceeded) as caught:
        lm.complete([Item(Kind.SYSTEM, "RLM rules"), Item(Kind.USER, "dense 中文 " * 100)])
    assert caught.value.scope == "completion"
    assert sent == []


def test_upstream_summary_overflow_retries_only_a_strictly_smaller_prefix() -> None:
    policy, sent = InputBudget(2000), []
    history = [msg("developer", "rules"), msg("user", "task"), *step(1, 500), *step(2, 500)]

    def post(request: dict) -> tuple[int, Any]:
        sent.append(request)
        if len(sent) == 1:
            return 400, {"error": {"code": "context_length_exceeded"}}
        return summary("short")

    lm = summarizer(policy, history, post)
    lm.summarize(len(history), "summarize")
    assert len(sent[1]["input"]) < len(sent[0]["input"])
    assert len(sent) == 3


def test_counter_scope_is_shared_with_strategy_helpers_and_restored() -> None:
    policy = InputBudget(1000)
    text = "中文 punctuation/123 " * 30
    legacy = approx_tokens(text)
    with counting_with(policy.text_tokens):
        assert approx_tokens(text) == policy.text_tokens(text)
        assert item_tokens(Item(Kind.USER, text)) == policy.text_tokens(text) + 8
        assert policy.text_tokens(truncate_middle(text, 25)) <= 25
    assert approx_tokens(text) == legacy


def test_bounded_worker_settings_enable_the_shared_counter() -> None:
    with relay_settings(input_budget_counter=InputBudget.name, input_budget_limit=23560):
        engine = engine_from_settings()
        assert engine.input_budget is not None
        assert engine.input_budget.input_limit == 23560


def test_media_counter_is_an_explicit_versioned_setting() -> None:
    with relay_settings(input_budget_counter=MEDIA_INPUT_BUDGET_COUNTER, input_budget_limit=23560):
        engine = engine_from_settings()
        assert engine.input_budget is not None
        assert engine.input_budget.name == MEDIA_INPUT_BUDGET_COUNTER
        assert engine.input_budget.fingerprint()["media_policy"]["kind"] == \
            "empirical_image_patch_upper_bound_v1"


def test_fixed_tool_schema_and_output_reserve_are_not_ignored() -> None:
    policy = InputBudget(2000, context_limit=1000, output_reserve=300)
    request = body(msg("user", "task"))
    request["input"].insert(0, {"type": "additional_tools", "role": "developer",
                               "tools": [{"description": "tool schema " * 500}]})
    with pytest.raises(BudgetExceeded):
        policy.check(CODEC, request, "main")
    request = body(msg("user", "task " * 100))
    request["max_output_tokens"] = 950
    with pytest.raises(BudgetExceeded) as caught:
        policy.check(CODEC, request, "main")
    assert caught.value.limit == 50


def test_summary_rejection_reports_the_effective_output_reserve_limit() -> None:
    policy = InputBudget(1000, context_limit=1000, output_reserve=100)
    history = [msg("developer", "rules"), msg("user", "task")]
    lm = summarizer(policy, history, lambda b: pytest.fail("must not dispatch"))
    lm.body["max_output_tokens"] = 980
    with pytest.raises(BudgetExceeded) as caught:
        lm.summarize(len(history), "summary prompt " * 20)
    assert caught.value.limit == 20


def test_provider_usage_does_not_rescale_local_planning() -> None:
    observations = []

    class Observe:
        name = "observe"
        def fingerprint(self): return {"name": self.name}
        def plan(self, request, lm): observations.append(request.tokens)

    policy = InputBudget(2000)
    engine = Engine(Observe(), input_budget=policy)
    request = body(msg("user", "task"), *step(1, 300))
    exchange = engine.prepare(CODEC, Harness(), request, tenant="test", post=lambda b: summary("unused"))
    engine.record(exchange, 1)  # deliberately incompatible accounting from a provider
    engine.prepare(CODEC, Harness(), request, tenant="test", post=lambda b: summary("unused"))
    assert observations == [policy.count(CODEC, request)] * 2


@dataclass
class ForcedManager:
    action: str
    forces: list[bool]
    name: str = "forced-test"

    def fingerprint(self): return {"name": self.name, "action": self.action}
    def plan(self, request, lm):
        self.forces.append(request.force)
        if self.action == "error":
            raise RuntimeError("manager broke")
        if request.force and self.action == "shrink":
            return Context((Item(Kind.USER, "task"), Item(Kind.SUMMARY, "short evidence")))
        if request.force and self.action == "auxiliary":
            lm.complete([Item(Kind.USER, "oversized " * 500)])
        return None


async def proxy_request(manager: Any, policy: InputBudget, request: dict, statuses: list[int] | None = None):
    app = create_app(Engine(manager, input_budget=policy), ProxyConfig())
    sent = []

    def handle(upstream: httpx.Request) -> httpx.Response:
        payload = json.loads(upstream.content)
        policy.check(CODEC, payload, "fake_transport")
        sent.append(payload)
        code = statuses.pop(0) if statuses else 200
        if code == 400:
            return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
        return httpx.Response(200, json=summary("done")[1])

    async with app.router.lifespan_context(app):
        await app.state.client.aclose()
        app.state.summary_client.close()
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        app.state.summary_client = httpx.Client(transport=httpx.MockTransport(handle))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            response = await client.post("/v1/responses", json=request)
    return response, sent


def test_local_main_overflow_recovers_before_the_first_upstream_dispatch() -> None:
    manager = ForcedManager("shrink", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(100), body(msg("user", "task"), *step(1, 5000))))
    assert response.status_code == 200
    assert manager.forces == [False, True]
    assert len(sent) == 1 and "short evidence" in json.dumps(sent[0])


@pytest.mark.parametrize("action", ["noop", "auxiliary"])
def test_unrecoverable_local_request_never_reaches_upstream(action: str) -> None:
    manager = ForcedManager(action, [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(100), body(msg("user", "task"), *step(1, 5000))))
    assert response.status_code == 400
    assert response.json()["error"]["source"] == "relay_input_budget"
    assert manager.forces == [False, True]
    assert sent == []


def test_manager_exception_cannot_forward_unmanaged_context() -> None:
    manager = ForcedManager("error", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(1000), body(msg("user", "task"))))
    assert response.status_code == 500 and sent == []


def test_local_and_provider_overflows_share_one_recovery_attempt() -> None:
    manager = ForcedManager("shrink", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(100),
        body(msg("user", "task"), *step(1, 5000)), [400]))
    assert response.status_code == 400
    assert manager.forces == [False, True] and len(sent) == 1
    assert response.json()["error"]["source"] == "relay_input_budget"


def test_repeated_provider_overflow_is_terminal_after_one_recovery() -> None:
    manager = ForcedManager("shrink", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(1000),
        body(msg("user", "task"), *step(1, 300)), [400, 400]))
    assert response.status_code == 400 and len(sent) == 2
    assert manager.forces == [False, True]
    assert response.json()["error"]["source"] == "relay_input_budget"


def test_provider_overflow_forced_request_is_rechecked_locally() -> None:
    manager = ForcedManager("auxiliary", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(100), body(msg("user", "task")), [400]))
    assert response.status_code == 400
    assert manager.forces == [False, True] and len(sent) == 1


def test_server_side_state_cannot_bypass_local_budget() -> None:
    manager = ForcedManager("noop", [])
    request = {**body(msg("user", "task")), "previous_response_id": "hidden-history"}
    response, sent = asyncio.run(proxy_request(manager, InputBudget(100), request))
    assert response.status_code == 400 and sent == []


def test_dispatch_tolerance_preserves_nominal_method_window_and_real_context_limit() -> None:
    policy = InputBudget(100, dispatch_slack=10)
    assert policy.input_limit == 100
    assert policy.limit_for({}) == 110
    assert policy.fingerprint()["dispatch_slack"] == 10
    observations = []

    class Observe:
        name = "observe-tolerance"
        def fingerprint(self): return {"name": self.name}
        def plan(self, request, lm): observations.append(request.window)

    request = body(msg("user", "word " * 94))
    assert 100 < policy.count(CODEC, request) <= 110
    Engine(Observe(), input_budget=policy).prepare(CODEC, Harness(), request,
        tenant="test", post=lambda _: summary("unused"))
    assert observations == [100]
    with pytest.raises(BudgetExceeded):
        policy.check(CODEC, body(msg("user", "word " * 110)), "main")
    bounded = InputBudget(100, context_limit=120, output_reserve=15, dispatch_slack=10)
    assert bounded.limit_for({}) == 105
    assert bounded.limit_for({"max_output_tokens": 30}) == 90


@pytest.mark.parametrize("width,height,detail,observed,bound", [
    (64, 64, None, 90, 133), (64, 64, "high", 90, 133),
    (512, 512, None, 393, 448), (512, 512, "high", 400, 448),
    (640, 640, None, 566, 628), (640, 640, "high", 573, 628),
    (768, 768, None, 777, 848), (768, 768, "high", 784, 848),
    (64, 768, "high", 166, 188),
    (1024, 1024, "high", None, 1408), (2048, 2048, "high", None, 5248),
])
def test_empirical_luna_media_counter_uses_conservative_bound(
        width: int, height: int, detail: str | None, observed: int | None, bound: int) -> None:
    policy = InputBudget(32000, name=MEDIA_INPUT_BUDGET_COUNTER)
    with_image = {**body(image_message(width, height, detail)), "model": "gpt-6-luna"}
    without_image = {**body(msg("user", "inspect")), "model": "gpt-6-luna"}
    delta = policy.count(CODEC, with_image) - policy.count(CODEC, without_image)
    assert delta == bound
    assert observed is None or observed <= delta
    assert policy.fingerprint()["media_policy"]["max_dimension"] == 2048
    item = CODEC.classify(with_image["input"][0])
    with counting_with(policy.text_tokens, policy.opaque_tokens, policy.media_item_tokens):
        assert item_tokens(item) == policy.item_tokens(item)


def test_media_input_is_rejected_outside_calibrated_model_detail_and_dimensions() -> None:
    media_policy = InputBudget(32000, name=MEDIA_INPUT_BUDGET_COUNTER)
    ordinary = InputBudget(32000)
    image = image_message(64, 64)
    with pytest.raises(InputAccountingError, match="not supported"):
        ordinary.count(CODEC, {**body(image), "model": "gpt-6-luna"})
    with pytest.raises(InputAccountingError, match="only for gpt-6-luna"):
        media_policy.count(CODEC, {**body(image), "model": "other-model"})
    with pytest.raises(InputAccountingError, match="high or automatic"):
        media_policy.count(CODEC, {**body(image_message(64, 64, "low")), "model": "gpt-6-luna"})
    with pytest.raises(InputAccountingError, match="supports dimensions only through 2048px"):
        media_policy.count(CODEC, {**body(image_message(2049, 64)), "model": "gpt-6-luna"})
    with pytest.raises(InputAccountingError, match="PNG images only"):
        media_policy.count(CODEC, {**body(image_message(64, 64, mime="image/jpeg")), "model": "gpt-6-luna"})


def test_media_counter_adds_two_images_and_preserves_each_in_the_shared_item_counter() -> None:
    policy = InputBudget(32000, name=MEDIA_INPUT_BUDGET_COUNTER)
    first = image_message(64, 64)["content"]
    image_part = first[-1]
    request = {**body({"type": "message", "role": "user", "content": [*first, image_part]}),
               "model": "gpt-6-luna"}
    item = CODEC.classify(request["input"][0])
    assert len(item.media) == 2
    assert policy.media_item_tokens(item) == 266
    with counting_with(policy.text_tokens, policy.opaque_tokens, policy.media_item_tokens):
        assert item_tokens(item) == policy.item_tokens(item)


def test_media_context_and_image_count_are_included_in_the_final_budget_check() -> None:
    policy = InputBudget(32000, name=MEDIA_INPUT_BUDGET_COUNTER)
    request = {**body(image_message(64, 64)), "model": "gpt-6-luna"}
    expected = policy.count(CODEC, request)
    seen = []

    class ObserveItemSize:
        name = "observe-media-item-size"

        def fingerprint(self):
            return {"name": self.name}

        def plan(self, req, lm):
            seen.append(item_tokens(req.current[-1]))

    Engine(ObserveItemSize(), input_budget=policy).prepare(
        CODEC, Harness(), request, tenant="test", post=lambda _: summary("unused"))
    assert seen == [policy.item_tokens(CODEC.classify(request["input"][0]))]
    response, sent = asyncio.run(proxy_request(ForcedManager("noop", []), policy, request))
    assert response.status_code == 200
    assert len(sent) == 1 and sent[0]["input"] == request["input"]
    assert policy.check(CODEC, sent[0], "offline_media_test") == expected


def test_unseen_reasoning_is_refused_or_estimated_as_set() -> None:
    item = Item(Kind.REASONING, "", opaque=4000, opaque_key="seen elsewhere")  # (a resumed conversation's)
    with pytest.raises(InputAccountingError):
        InputBudget(1000).opaque_tokens(item)
    assert InputBudget(1000, unknown_opaque="estimate").opaque_tokens(item) == 1000
