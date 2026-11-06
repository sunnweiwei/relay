from __future__ import annotations

import pytest

from relay.core.budget import BudgetExceeded, InputBudget
from relay.core.truncation import truncate_request
from relay.protocols import OpenAIResponses
from tests.test_engine import body, msg, step

CODEC = OpenAIResponses()


def test_fitting_request_is_returned_unchanged() -> None:
    policy = InputBudget(1000)
    request = body(msg("developer", "fixed rules"), msg("user", "initial task"))
    result, info = truncate_request(CODEC, request, policy, scope="main")
    assert result is request
    assert info["kept_indices"] == [0, 1]
    assert info["before_tokens"] == info["after_tokens"]


def test_system_task_and_schema_are_preserved_while_old_groups_drop() -> None:
    policy = InputBudget(70)
    request = body(msg("developer", "fixed rules"), msg("user", "initial task"),
                   msg("user", "old " * 80), msg("user", "new task"))
    request["tools"] = [{"type": "function", "name": "keep", "description": "schema"}]
    result, info = truncate_request(CODEC, request, policy, scope="main")
    assert result["input"][0] == request["input"][0]
    assert result["input"][-1] == request["input"][-1]
    assert result["tools"] == request["tools"]
    assert info["dropped_groups"] or info["truncated_text_items"]
    assert policy.count(CODEC, result) <= policy.limit_for(request)


def test_call_result_pair_stays_complete_and_ids_arguments_are_unchanged() -> None:
    policy = InputBudget(230)
    pair = step(3, 900)
    pair[0]["arguments"] = '{"path":"keep exactly"}'
    request = body(msg("user", "task"), *pair, msg("user", "latest"))
    result, info = truncate_request(CODEC, request, policy, scope="main")
    sent = CODEC.items(result)
    assert not CODEC.orphans(sent)
    call = next(item for item in sent if item.get("type") == "function_call")
    assert call["call_id"] == "c3" and call["arguments"] == '{"path":"keep exactly"}'
    assert info["after_tokens"] <= policy.limit_for(request)


def test_single_large_message_is_truncated_with_marker_and_valid_utf8() -> None:
    policy = InputBudget(90)
    text = "起" * 500 + " finish"
    request = body(msg("user", text))
    result, info = truncate_request(CODEC, request, policy, scope="completion", protected=set())
    output = CODEC.classify(result["input"][0]).text
    output.encode("utf-8")
    assert "[harness truncated" in output
    assert output.startswith("起") and output.endswith("finish")
    assert info["truncated_text_items"] == 1
    assert info["after_tokens"] <= policy.limit_for(request)


def test_assistant_output_text_truncates_without_dropping_its_group() -> None:
    policy = InputBudget(90)
    request = body(msg("user", "task"),
                   {"type": "message", "role": "assistant", "content": "answer " * 500})
    result, info = truncate_request(CODEC, request, policy, scope="completion", protected={0})
    assert info["dropped_groups"] == 0
    assert info["kept_indices"] == [0, 1]
    assert "[harness truncated" in result["input"][1]["content"][0]["text"]
    assert info["after_tokens"] <= policy.limit_for(request)


def test_impossible_fixed_input_is_rejected() -> None:
    policy = InputBudget(40)
    request = {"model": "m", "instructions": "fixed " * 100,
               "input": [msg("user", "task")],
               "tools": [{"description": "fixed schema " * 100}]}
    with pytest.raises(BudgetExceeded):
        truncate_request(CODEC, request, policy, scope="main")


def test_unpaired_tools_are_rejected_without_rewriting() -> None:
    request = body(msg("user", "task"), step(1, 10)[0])
    with pytest.raises(ValueError, match="unpaired"):
        truncate_request(CODEC, request, InputBudget(1000), scope="main")


def test_provider_target_can_be_stricter_than_local_limit() -> None:
    policy = InputBudget(1000)
    request = body(msg("user", "task"), msg("assistant", "old " * 100))
    target = 45
    result, info = truncate_request(CODEC, request, policy, scope="main", protected={0},
                                   target_limit=target)
    assert info["after_tokens"] <= target
    assert policy.count(CODEC, result) <= target
