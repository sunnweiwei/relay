from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .checkpoint_cache import PrefixCheckpointCache
from .middleware import ContextEngine, item_dict, item_list
from .strategies import ContextStrategy, strategy_from_env

_HOP_HEADERS = {"connection", "content-length", "host", "transfer-encoding"}
_TRANSFORMED_HEADERS = _HOP_HEADERS | {"content-encoding", "content-md5", "etag"}


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8787
    checkpoint_mode: str = "inline"

    @classmethod
    def from_env(cls) -> ProxyConfig:
        return cls(
            upstream_base_url=os.getenv(
                "RELAY_UPSTREAM_BASE_URL", "https://api.openai.com/v1"
            ),
            upstream_api_key=os.getenv("RELAY_UPSTREAM_API_KEY"),
            host=os.getenv("RELAY_HOST", "127.0.0.1"),
            port=int(os.getenv("RELAY_PORT", "8787")),
            checkpoint_mode=os.getenv("RELAY_CHECKPOINT_MODE", "inline"),
        )


def _headers(headers: Mapping[str, str], api_key: str | None) -> dict[str, str]:
    forwarded = {
        key: value for key, value in headers.items() if key.lower() not in _HOP_HEADERS
    }
    if api_key:
        forwarded["authorization"] = f"Bearer {api_key}"
    return forwarded


def _api_key(headers: Mapping[str, str], configured: str | None) -> str:
    if configured:
        return configured
    authorization = headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ValueError("an upstream API key or Bearer authorization is required")
    return token


def _cache_namespace(request: Request) -> str:
    authorization = request.headers.get("authorization")
    if not authorization:
        raise ValueError(
            "checkpoint cache mode requires per-tenant Bearer authentication"
        )
    return json.dumps(
        {
            "authorization": authorization,
            "organization": request.headers.get("openai-organization"),
            "project": request.headers.get("openai-project"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _upstream_url(base_url: str, request: Request) -> str:
    path = request.url.path
    if base_url.rstrip("/").endswith("/v1") and path.startswith("/v1/"):
        path = path[3:]
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}{query}"


def _response_headers(
    headers: Mapping[str, str], *, transformed: bool = False
) -> dict[str, str]:
    removed = _TRANSFORMED_HEADERS if transformed else _HOP_HEADERS
    return {key: value for key, value in headers.items() if key.lower() not in removed}


def _management_responses(
    config: ProxyConfig, request: Request
) -> tuple[Any, Callable[[], None]]:
    from openai import OpenAI

    client = OpenAI(
        api_key=_api_key(request.headers, config.upstream_api_key),
        base_url=config.upstream_base_url,
        organization=request.headers.get("openai-organization"),
        project=request.headers.get("openai-project"),
    )
    return client.responses, client.close


def _sse(event: str | None, data: str) -> bytes:
    lines = [] if event is None else [f"event: {event}"]
    lines.extend(f"data: {line}" for line in data.splitlines() or [""])
    return ("\n".join(lines) + "\n\n").encode()


async def _sse_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str | None, str]]:
    event: str | None = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)


async def _raw_body(response: httpx.Response) -> AsyncIterator[bytes]:
    if response.is_stream_consumed:
        yield response.content
        return
    async for chunk in response.aiter_raw():
        yield chunk


def _event_payload(
    event_type: str, item: dict[str, Any], index: int, sequence: int
) -> dict[str, Any]:
    return {
        "type": event_type,
        "output_index": index,
        "item": item,
        "sequence_number": sequence,
    }


async def _managed_sse(
    upstream: httpx.Response,
    engine: ContextEngine,
    responses: Any,
    request_body: dict[str, Any],
    prepared: Any,
    close_management: Callable[[], None],
) -> AsyncIterator[bytes]:
    pre_marker = (
        item_dict(marker) if (marker := engine.checkpoint_item(prepared)) else None
    )
    inserted_pre = False
    sequence_offset = 0
    try:
        async for event, data in _sse_events(upstream):
            if data == "[DONE]":
                yield _sse(event, data)
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                yield _sse(event, data)
                continue

            event_type = payload.get("type")
            if (
                pre_marker is not None
                and not inserted_pre
                and event_type
                in {
                    "response.output_item.added",
                    "response.completed",
                }
            ):
                sequence = int(payload.get("sequence_number", 0))
                for offset, kind in enumerate(
                    ("response.output_item.added", "response.output_item.done")
                ):
                    marker_event = _event_payload(
                        kind, pre_marker, 0, sequence + offset
                    )
                    yield _sse(kind, json.dumps(marker_event, separators=(",", ":")))
                sequence_offset += 2
                inserted_pre = True

            if isinstance(payload.get("sequence_number"), int):
                payload["sequence_number"] += sequence_offset
            if pre_marker is not None and isinstance(payload.get("output_index"), int):
                payload["output_index"] += 1

            if event_type != "response.completed":
                yield _sse(event, json.dumps(payload, separators=(",", ":")))
                continue

            response_data = payload.get("response") or {}
            raw_output = response_data.get("output") or []
            visible = await run_in_threadpool(
                engine.finalize,
                responses,
                request_body,
                prepared,
                raw_output,
            )
            visible_data = item_list(visible)
            prefix = 1 if pre_marker is not None else 0
            post_items = visible_data[prefix + len(raw_output) :]
            sequence = int(payload.get("sequence_number", 0))
            for item_offset, item in enumerate(post_items):
                index = prefix + len(raw_output) + item_offset
                for kind in ("response.output_item.added", "response.output_item.done"):
                    marker_event = _event_payload(kind, item, index, sequence)
                    yield _sse(kind, json.dumps(marker_event, separators=(",", ":")))
                    sequence += 1
            payload["sequence_number"] = sequence
            response_data["output"] = visible_data
            payload["response"] = response_data
            yield _sse(event, json.dumps(payload, separators=(",", ":")))
    finally:
        await upstream.aclose()
        await run_in_threadpool(close_management)


def create_app(
    strategy: ContextStrategy | None = None,
    config: ProxyConfig | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    management_responses: Any | None = None,
    checkpoint_cache: PrefixCheckpointCache | None = None,
) -> Starlette:
    config = config or ProxyConfig.from_env()
    engine = ContextEngine(
        strategy or strategy_from_env(),
        checkpoint_mode=config.checkpoint_mode,
        checkpoint_cache=checkpoint_cache,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        timeout = httpx.Timeout(60.0, read=None)
        app.state.upstream = httpx.AsyncClient(
            timeout=timeout, transport=upstream_transport
        )
        yield
        await app.state.upstream.aclose()

    async def dispatch(request: Request) -> Response:
        upstream_client: httpx.AsyncClient = request.app.state.upstream
        url = _upstream_url(config.upstream_base_url, request)
        headers = _headers(request.headers, config.upstream_api_key)
        body_bytes = await request.body()

        if request.method != "POST" or request.url.path != "/v1/responses":
            upstream = await upstream_client.send(
                upstream_client.build_request(
                    request.method, url, headers=headers, content=body_bytes
                ),
                stream=True,
            )
            return StreamingResponse(
                _raw_body(upstream),
                status_code=upstream.status_code,
                headers=_response_headers(upstream.headers),
                background=BackgroundTask(upstream.aclose),
            )

        close_management: Callable[[], None] = lambda: None
        try:
            body = json.loads(body_bytes)
            if management_responses is None:
                responses, close_management = await run_in_threadpool(
                    _management_responses, config, request
                )
            else:
                responses, close_management = management_responses, lambda: None
            namespace = (
                _cache_namespace(request)
                if engine.checkpoint_cache is not None
                else "local"
            )
            prepared = await run_in_threadpool(
                engine.prepare, responses, body, namespace
            )
            forwarded = engine.upstream_request(body, prepared)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            await run_in_threadpool(close_management)
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "code": "relay_error",
                    }
                },
                status_code=400,
            )

        upstream = await upstream_client.send(
            upstream_client.build_request("POST", url, headers=headers, json=forwarded),
            stream=bool(body.get("stream")),
        )
        if upstream.status_code >= 400:
            content = await upstream.aread()
            await upstream.aclose()
            await run_in_threadpool(close_management)
            return Response(
                content,
                status_code=upstream.status_code,
                headers=_response_headers(upstream.headers),
            )

        if body.get("stream") is True:
            return StreamingResponse(
                _managed_sse(
                    upstream,
                    engine,
                    responses,
                    body,
                    prepared,
                    close_management,
                ),
                media_type="text/event-stream",
                headers=_response_headers(upstream.headers, transformed=True),
            )

        try:
            data = upstream.json()
            data["output"] = item_list(
                await run_in_threadpool(
                    engine.finalize,
                    responses,
                    body,
                    prepared,
                    data.get("output") or [],
                )
            )
            return JSONResponse(
                data,
                status_code=upstream.status_code,
                headers=_response_headers(upstream.headers, transformed=True),
            )
        finally:
            await upstream.aclose()
            await run_in_threadpool(close_management)

    return Starlette(
        routes=[
            Route(
                "/{path:path}",
                dispatch,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            )
        ],
        lifespan=lifespan,
    )


def main() -> None:
    import uvicorn

    config = ProxyConfig.from_env()
    uvicorn.run(create_app(config=config), host=config.host, port=config.port)
