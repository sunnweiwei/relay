"""Shared scripted upstream, trace capture and Compact/cache invariants."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from relay import Compact, PrefixCheckpointCache
from relay.middleware import trajectory_digest
from relay.proxy import ProxyConfig, create_app
from relay.strategies.compact import CODEX_SUMMARY_PREFIX
from tests.harness_e2e_support import HarnessUpstream
from tests.test_codex_e2e import _is_summary_request, _text

HARNESSES = (
    "gemini-cli", "codex", "mini-swe", "opencode", "pi", "cursor", "claude-code"
)
THRESHOLD = 100


@dataclass
class Lookup:
    trajectory: list[dict[str, Any]]
    matched_items: int | None
    artifact: dict[str, Any] | None


@dataclass
class StoredCheckpoint:
    covered_items: int
    prefix_digest: str
    artifact_digest: str


class RecordingCache(PrefixCheckpointCache):
    def __init__(self) -> None:
        super().__init__(secret=b"compact-cache-contract")
        self.lookups: list[Lookup] = []
        self.stored: list[StoredCheckpoint] = []

    def match(self, partition, trajectory):
        match = super().match(partition, trajectory)
        self.lookups.append(Lookup(
            deepcopy(list(trajectory)),
            match.matched_items if match else None,
            deepcopy(match.artifact) if match else None,
        ))
        return match

    def put_prefixes(self, partition, trajectory, checkpoints):
        for depth, artifact in checkpoints:
            self.stored.append(StoredCheckpoint(
                depth,
                trajectory_digest(trajectory[:depth]),
                trajectory_digest([artifact]),
            ))
        return super().put_prefixes(partition, trajectory, checkpoints)


class StableCompactUpstream(HarnessUpstream):
    """Force one compaction, then keep the restored summary below threshold."""

    def __init__(self, *, tool_first: bool = False, mini_mode: bool = False,
                 tool_name: str | None = None) -> None:
        super().__init__(tool_name=tool_name)
        self.tool_first = tool_first
        self.mini_mode = mini_mode

    async def dispatch(self, request):
        if request.url.path == "/v1/responses/input_tokens":
            body = await request.json()
            with self.lock:
                self.count_requests.append(body)
            if _is_summary_request(body):
                count = 10
            elif any(CODEX_SUMMARY_PREFIX in _text(item)
                     for item in body.get("input", [])):
                count = 10
            else:
                count = 1000
            return JSONResponse({"object": "response.input_tokens", "input_tokens": count})
        return await super().dispatch(request)


def make_responses_relay(upstream_url: str, cache: RecordingCache):
    raw: list[dict[str, Any]] = []
    app = create_app(
        Compact(compact_threshold=THRESHOLD),
        ProxyConfig(
            upstream_base_url=f"{upstream_url}/v1",
            upstream_api_key="fake-upstream-key",
            checkpoint_mode="cache",
        ),
        checkpoint_cache=cache,
    )

    class CaptureRaw(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path == "/v1/responses":
                raw.append(deepcopy(await request.json()))
            return await call_next(request)

    app.add_middleware(CaptureRaw)
    return app, raw


def assert_contract(
    cache: RecordingCache,
    summary_requests: list[dict[str, Any]],
    raw_inputs: list[list[dict[str, Any]]],
    forwarded_inputs: list[list[dict[str, Any]]],
    *,
    latest_marker: str,
) -> None:
    assert len(raw_inputs) >= 2
    assert len(raw_inputs) == len(forwarded_inputs) == len(cache.lookups)
    assert cache.lookups[0].matched_items is None
    assert cache.lookups[1].matched_items == len(raw_inputs[0])
    assert cache.lookups[1].artifact is not None
    first_prefix = trajectory_digest(raw_inputs[0])
    restored = cache.lookups[1]
    assert any(
        checkpoint.covered_items == restored.matched_items
        and checkpoint.prefix_digest == first_prefix
        and checkpoint.artifact_digest == trajectory_digest([restored.artifact])
        for checkpoint in cache.stored
    ), "Resumed request did not restore the checkpoint saved for its exact history"
    assert len(summary_requests) == 1, "Old history was summarized again"
    assert cache.stats().hits >= len(raw_inputs) - 1

    first_summary = [
        _text(item) for item in forwarded_inputs[0]
        if CODEX_SUMMARY_PREFIX in _text(item)
    ]
    assert len(first_summary) == 1
    assert "relay checkpoint summary 1" in first_summary[0]
    for raw, forwarded in zip(raw_inputs[1:], forwarded_inputs[1:]):
        assert not any(CODEX_SUMMARY_PREFIX in _text(item) for item in raw)
        assert first_summary[0] in [_text(item) for item in forwarded]
        assert not any(item.get("type") == "compaction" for item in forwarded)
    for lookup in cache.lookups[1:]:
        assert lookup.matched_items is not None
        assert lookup.artifact is not None
        assert any(
            checkpoint.covered_items == lookup.matched_items
            and checkpoint.prefix_digest == trajectory_digest(
                lookup.trajectory[:lookup.matched_items]
            )
            and checkpoint.artifact_digest == trajectory_digest([lookup.artifact])
            for checkpoint in cache.stored
        ), "Cache hit did not return the artifact for the matching prefix"
    assert latest_marker in "\n".join(_text(item) for item in forwarded_inputs[-1])


def assert_responses_wire_contract(
    raw_requests: list[dict[str, Any]],
    forwarded_requests: list[dict[str, Any]],
    *, expected_stream: bool = True,
) -> None:
    """Check the Responses envelope on both sides of the Relay."""
    assert len(raw_requests) == len(forwarded_requests) >= 2
    for raw, forwarded in zip(raw_requests, forwarded_requests):
        assert bool(raw.get("stream")) is expected_stream
        assert bool(forwarded.get("stream")) is expected_stream
        for request in (raw, forwarded):
            assert not request.get("previous_response_id")
            assert not request.get("conversation")
        assert not any(
            isinstance(item, dict) and item.get("type") == "compaction"
            for item in forwarded.get("input", [])
        )
