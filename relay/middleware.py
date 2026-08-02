from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import zlib

from .strategies import ContextStrategy, PreparedInput


LOCAL_COMPACTION_PREFIX = "relay:v1:"


@dataclass(frozen=True)
class LocalCompactionItem:
    """Python-level analogue of the SDK's `ResponseCompactionItem`."""

    id: str
    encrypted_content: str
    type: str = "compaction"
    created_by: str = "relay"

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def item_dict(item: Any) -> dict[str, Any]:
    """Convert an SDK response item or a plain mapping into JSON data."""

    if isinstance(item, Mapping):
        return deepcopy(dict(item))
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    raise TypeError("Responses input must contain mappings or SDK response items")


def item_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("context-managed Responses calls require append-only list input")
    return [item_dict(item) for item in value]


def trajectory_digest(items: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(
        items, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def local_compaction_item(
    strategy: str, active_input: Sequence[dict[str, Any]]
) -> LocalCompactionItem:
    """Encode a Relay-local checkpoint in the Responses compaction-item shape."""

    payload = json.dumps(
        {
            "version": 1,
            "strategy": strategy,
            "active_input": item_list(active_input),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    encoded = base64.urlsafe_b64encode(zlib.compress(payload, level=9)).decode()
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return LocalCompactionItem(
        id=f"cmp_local_{digest}",
        encrypted_content=f"{LOCAL_COMPACTION_PREFIX}{encoded}",
    )


def decode_local_compaction(
    item: Mapping[str, Any], expected_strategy: str
) -> list[dict[str, Any]] | None:
    content = item.get("encrypted_content")
    if not isinstance(content, str) or not content.startswith(LOCAL_COMPACTION_PREFIX):
        return None
    try:
        encoded = content.removeprefix(LOCAL_COMPACTION_PREFIX)
        payload_bytes = zlib.decompress(base64.urlsafe_b64decode(encoded))
        payload = json.loads(payload_bytes)
    except Exception as exc:
        raise ValueError("invalid local compaction checkpoint") from exc
    expected_id = f"cmp_local_{hashlib.sha256(payload_bytes).hexdigest()[:24]}"
    if item.get("id") != expected_id:
        raise ValueError("local compaction checkpoint failed its integrity check")
    if payload.get("version") != 1:
        raise ValueError("unsupported local compaction checkpoint version")
    if payload.get("strategy") != expected_strategy:
        raise ValueError("local compaction checkpoint belongs to another strategy")
    return item_list(payload.get("active_input"))


@dataclass
class CompactResponse:
    """Attribute-compatible result for `responses.compact(...).output` workflows."""

    output: list[dict[str, Any]]
    id: str
    object: str = "response.compaction"
    created_at: int = 0
    usage: Any = None

    @classmethod
    def from_output(cls, output: Sequence[dict[str, Any]]) -> CompactResponse:
        canonical = item_list(output)
        digest = trajectory_digest(canonical)[:24]
        return cls(canonical, f"cmp_local_{digest}", created_at=int(time.time()))

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return asdict(self)


class ManagedResponse:
    """A normal SDK Response whose output may include a local compaction item."""

    def __init__(
        self,
        response: Any,
        output: Sequence[Any] | None = None,
    ) -> None:
        self.response = response
        self.output = list(output if output is not None else response.output)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        dump = getattr(self.response, "model_dump", None)
        if not callable(dump):
            raise AttributeError("wrapped response does not implement model_dump")
        value = dump(**kwargs)
        value["output"] = item_list(self.output)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)


class ManagedStream(Iterator[Any]):
    def __init__(self, stream: Any, finalize: Callable[[Any], ManagedResponse]) -> None:
        self._stream = stream
        self._iterator = iter(stream)
        self._finalize = finalize
        self.final_response: ManagedResponse | None = None

    def __iter__(self) -> ManagedStream:
        return self

    def __next__(self) -> Any:
        event = next(self._iterator)
        if getattr(event, "type", None) == "response.completed":
            managed = self._finalize(event.response)
            self.final_response = managed
            return _CompletedEvent(event, managed)
        return event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class ManagedStreamManager:
    def __init__(self, manager: Any, finalize: Callable[[Any], ManagedResponse]) -> None:
        self._manager = manager
        self._finalize = finalize
        self.stream: ManagedStream | None = None

    def __enter__(self) -> ManagedStream:
        self.stream = ManagedStream(self._manager.__enter__(), self._finalize)
        return self.stream

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._manager.__exit__(exc_type, exc, traceback)


class _CompletedEvent:
    """Preserve the SDK event surface while replacing its final response."""

    def __init__(self, event: Any, response: ManagedResponse) -> None:
        self._event = event
        self.response = response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._event, name)


class ContextEngine:
    """Transport-neutral request/response context transformation."""

    def __init__(self, strategy: ContextStrategy) -> None:
        self.strategy = strategy

    def prepare(self, responses: Any, request: dict[str, Any]) -> PreparedInput:
        if (
            request.get("previous_response_id") is not None
            or request.get("conversation") is not None
        ):
            raise ValueError(
                "this middleware uses stateless append-only input; do not combine it with "
                "previous_response_id or conversation"
            )
        raw = item_list(request.get("input"))
        active = self.restore(raw)
        return self.strategy.prepare(responses, dict(request), active)

    def restore(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index in range(len(raw) - 1, -1, -1):
            item = raw[index]
            if item.get("type") != "compaction":
                continue
            local = decode_local_compaction(item, self.strategy.name)
            if local is not None:
                return [*local, *deepcopy(raw[index + 1 :])]
            # Official encrypted compaction items are already canonical model input.
            return deepcopy(raw[index:])
        return deepcopy(raw)

    def upstream_request(
        self, request: dict[str, Any], prepared: PreparedInput
    ) -> dict[str, Any]:
        forwarded = dict(request)
        forwarded["input"] = deepcopy(prepared.input)
        forwarded.update(deepcopy(prepared.overrides))
        management = [
            deepcopy(item)
            for item in forwarded.get("context_management") or []
            if item.get("type") != "compaction"
        ]
        if management:
            forwarded["context_management"] = management
        else:
            forwarded.pop("context_management", None)
        return forwarded

    def finalize(
        self,
        responses: Any,
        request: dict[str, Any],
        prepared: PreparedInput,
        raw_output: Sequence[Any],
    ) -> list[Any]:
        output = item_list(raw_output)
        active = self.strategy.finish(
            responses,
            dict(request),
            deepcopy(prepared.input),
            deepcopy(output),
        )
        visible_output: list[Any] = list(raw_output)
        has_official_compaction = any(
            item.get("type") == "compaction" for item in output
        )
        if prepared.compacted and not has_official_compaction:
            visible_output = [
                local_compaction_item(self.strategy.name, prepared.input),
                *visible_output,
            ]
        elif active != [*prepared.input, *output] and not has_official_compaction:
            visible_output.append(local_compaction_item(self.strategy.name, active))
        return visible_output

    def compact(
        self, responses: Any, request: dict[str, Any]
    ) -> CompactResponse:
        if (
            request.get("previous_response_id") is not None
            or request.get("conversation") is not None
        ):
            raise ValueError("responses.compact requires an explicit input window")
        active = self.restore(item_list(request.get("input")))
        output = self.strategy.compact(responses, dict(request), active)
        return CompactResponse.from_output(output)


class ManagedResponses:
    """Drop-in wrapper for the synchronous OpenAI `client.responses` resource."""

    def __init__(self, responses: Any, strategy: ContextStrategy) -> None:
        self._responses = responses
        self.engine = ContextEngine(strategy)
        self.strategy = strategy

    def create(self, **request: Any) -> Any:
        prepared = self.engine.prepare(self._responses, request)
        forwarded = self.engine.upstream_request(request, prepared)
        response = self._responses.create(**forwarded)
        finalize = self._finalizer(request, prepared)
        if request.get("stream") is True:
            return ManagedStream(response, finalize)
        return finalize(response)

    def stream(self, **request: Any) -> ManagedStreamManager:
        prepared = self.engine.prepare(self._responses, request)
        forwarded = self.engine.upstream_request(request, prepared)
        manager = self._responses.stream(**forwarded)
        return ManagedStreamManager(manager, self._finalizer(request, prepared))

    def compact(self, **request: Any) -> CompactResponse:
        return self.engine.compact(self._responses, request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)

    def _finalizer(
        self, request: dict[str, Any], prepared: PreparedInput
    ) -> Callable[[Any], ManagedResponse]:
        def finalize(response: Any) -> ManagedResponse:
            visible = self.engine.finalize(
                self._responses,
                request,
                prepared,
                list(getattr(response, "output", ())),
            )
            return ManagedResponse(response, visible)

        return finalize


class ContextManagingOpenAI:
    """Wrap an existing synchronous `OpenAI` client without changing other APIs."""

    def __init__(self, client: Any, strategy: ContextStrategy | None = None) -> None:
        if strategy is None:
            from .strategies import Compact

            strategy = Compact()
        self._client = client
        self.responses = ManagedResponses(client.responses, strategy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
