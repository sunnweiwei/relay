from __future__ import annotations

import base64
import math
import struct
import zlib

import pytest

from evals.pilot.run import resolve_input_budget_counter
from relay.core.budget import (
    InputAccountingError,
    InputBudget,
    MEDIA_INPUT_BUDGET_COUNTER,
    PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER,
)
from relay.core.tokens import counting_with, item_tokens
from relay.protocols import OpenAIResponses

CODEC = OpenAIResponses()
def _png(width: int, height: int) -> bytes:
    """A valid grayscale PNG built in memory (no task data or external tool needed)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))


@pytest.fixture(scope="module")
def task_frames() -> dict[str, bytes]:
    # The original check used a frame of a Science task video; the counter reads only the
    # encoded header, so synthetic images of the same 1040x1392 size exercise the same path.
    return {"image/png": _png(1040, 1392), "image/jpeg": _minimal_jpeg(1040, 1392)}


def _body(payload: bytes, mime: str, *, detail: str | None = "high", model: str = "gpt-6-luna") -> dict:
    image = {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"}
    if detail is not None:
        image["detail"] = detail
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "private frame question"}, image,
        ]}],
    }


def _minimal_jpeg(width: int, height: int) -> bytes:
    # SOI, one valid SOF0 segment, then EOI. The parser needs only encoded metadata.
    sof_payload = b"\x08" + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00"
    segment = b"\xff\xc0" + struct.pack(">H", len(sof_payload) + 2) + sof_payload
    return b"\xff\xd8" + segment + b"\xff\xd9"


def test_png_and_jpeg_frames_are_counted_from_encoded_bytes(task_frames: dict[str, bytes]) -> None:
    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    records = {}
    for mime, payload in task_frames.items():
        request = _body(payload, mime)
        count = policy.count(CODEC, request)
        record = policy.media_diagnostics(CODEC, request)[0]
        assert record["declared_mime"] == mime
        assert record["detected_mime"] == mime
        assert record["decoded_byte_count"] == len(payload)
        assert record["width"] == 1040
        assert record["height"] == 1392
        expected = 128 + math.ceil(1.25 * math.ceil(1040 / 32) * math.ceil(1392 / 32))
        text_only = {**request, "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "private frame question"}
        ]}]}
        assert count - policy.count(CODEC, text_only) == expected
        item = CODEC.classify(request["input"][0])
        written = CODEC.write(item)
        roundtrip_image = next(part for part in written["content"] if part.get("type") == "input_image")
        assert roundtrip_image["image_url"].endswith(base64.b64encode(payload).decode("ascii"))
        with counting_with(policy.text_tokens, policy.opaque_tokens, policy.media_item_tokens):
            assert item_tokens(item) == policy.item_tokens(item)
        records[mime] = (record["width"], record["height"], expected)
    assert records["image/png"] == records["image/jpeg"]


def test_text_only_equivalence_and_opt_in_default() -> None:
    regular = InputBudget(10_000)
    media = InputBudget(10_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    body = {"model": "gpt-6-luna", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "same text"}]}
    ]}
    assert regular.count(CODEC, body) == media.count(CODEC, body)
    assert resolve_input_budget_counter(None, None, None) is None
    assert resolve_input_budget_counter(None, PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER, "1") == PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER
    with pytest.raises(ValueError):
        resolve_input_budget_counter(None, PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER, None)
    with pytest.raises(ValueError):
        resolve_input_budget_counter(None, "unknown", "1")
    with pytest.raises(ValueError):
        resolve_input_budget_counter(MEDIA_INPUT_BUDGET_COUNTER, PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER, "1")


@pytest.mark.parametrize("width,height", [(1, 1), (64, 768), (768, 768), (2048, 2048)])
def test_jpeg_dimensions_and_policy_boundaries(width: int, height: int) -> None:
    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    data = _minimal_jpeg(width, height)
    record = policy.media_diagnostics(CODEC, _body(data, "image/jpeg"))[0]
    assert (record["width"], record["height"]) == (width, height)
    assert policy.count(CODEC, _body(data, "image/jpeg")) > 0


def test_jpeg_over_limit_and_malformed_inputs_are_classified_without_payload_leaks() -> None:
    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    cases = [
        (_minimal_jpeg(2049, 64), "image/jpeg", "invalid_or_oversized_dimensions"),
        (b"\xff\xd8\xff\xc0\x00\x03", "image/jpeg", "invalid_image_header"),
        (b"not an image", "image/jpeg", "invalid_image_header"),
    ]
    for data, mime, category in cases:
        request = _body(data, mime)
        with pytest.raises(InputAccountingError) as caught:
            policy.count(CODEC, request)
        assert caught.value.category == category
        diagnostic = caught.value.media_diagnostics[0]
        assert diagnostic["error_category"] == category
        assert not ({"data", "url", "text", "question", "reasoning"} & diagnostic.keys())
        assert base64.b64encode(data).decode() not in repr(diagnostic)

    malformed = _body(_minimal_jpeg(64, 64), "image/jpeg")
    malformed["input"][0]["content"][-1]["image_url"] = "data:image/jpeg;base64,%%%"
    with pytest.raises(InputAccountingError) as caught:
        policy.count(CODEC, malformed)
    assert caught.value.category == "malformed_base64"
    assert caught.value.media_diagnostics[0]["decoded_byte_count"] is None


def test_encoded_image_size_limit_is_retained() -> None:
    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    oversized = {
        "model": "gpt-6-luna",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + "A" * (16 * 1024 * 1024 + 4),
             "detail": "high"}
        ]}],
    }
    with pytest.raises(InputAccountingError) as caught:
        policy.count(CODEC, oversized)
    assert caught.value.category == "encoded_image_too_large"
    assert caught.value.media_diagnostics[0]["decoded_byte_count"] is None


def test_mime_detail_model_and_remote_media_rejections_have_safe_categories(task_frames: dict[str, bytes]) -> None:
    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    valid = task_frames["image/jpeg"]
    checks = [
        (_body(valid, "image/png"), "mime_signature_mismatch"),
        (_body(valid, "image/jpeg", detail="low"), "unsupported_detail"),
        (_body(valid, "image/jpeg", model="other-model"), "unsupported_model"),
    ]
    for request, category in checks:
        with pytest.raises(InputAccountingError) as caught:
            policy.count(CODEC, request)
        assert caught.value.category == category
        assert caught.value.media_diagnostics
    # Remote media never needs the URL copied into diagnostics.
    remote = {"model": "gpt-6-luna", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_image", "image_url": "https://private.example/frame.png", "detail": "high"}
    ]}]}
    with pytest.raises(InputAccountingError) as caught:
        policy.count(CODEC, remote)
    assert caught.value.category == "non_inline_media"
    assert "private.example" not in repr(caught.value.media_diagnostics)


def test_strategy_item_counter_attaches_metadata_for_direct_media_failure() -> None:
    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    malformed = _body(b"broken", "image/jpeg")
    item = CODEC.classify(malformed["input"][0])
    with pytest.raises(InputAccountingError) as caught:
        policy.media_item_tokens(item)
    assert caught.value.media_diagnostics[0]["error_category"] == "invalid_image_header"
    assert "private frame question" not in repr(caught.value.media_diagnostics)


def test_proxy_failure_logs_only_metadata(task_frames: dict[str, bytes], caplog: pytest.LogCaptureFixture) -> None:
    import asyncio
    import logging

    from tests.test_input_budget import ForcedManager, proxy_request

    policy = InputBudget(100_000, name=PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)
    data = task_frames["image/jpeg"]
    request = _body(data, "image/png")
    caplog.set_level(logging.WARNING, logger="relay")
    response, sent = asyncio.run(proxy_request(ForcedManager("noop", []), policy, request))
    assert response.status_code == 500
    assert sent == []
    logs = caplog.text
    assert '"declared_mime":"image/png"' in logs
    assert '"detected_mime":"image/jpeg"' in logs
    assert '"width":1040' in logs and '"height":1392' in logs
    assert '"decoded_byte_count":' in logs
    assert "private frame question" not in logs
    assert base64.b64encode(data).decode("ascii") not in logs
