"""HTTP proxy in front of the model APIs.

Conversation endpoints (one per codec) go through the engine; every other request,
and every response body, passes through unchanged. Streaming responses are relayed
as they arrive while a tap reads the reported prompt tokens. The same server answers
harness hooks (`hooks.py`).
"""

from __future__ import annotations

import json
import logging
import os
import zlib
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..core.engine import Engine, Exchange
from ..core.store import PrefixStore
from ..harnesses import detect
from ..protocols import Codec, codec_for
from ..providers import Upstream, route, upstreams_from_settings
from ..strategies import STRATEGIES
from ..strategies.base import Strategy
from .hooks import hook_routes
from .. import settings

log = logging.getLogger("relay")

TENANT_HEADERS = ("authorization", "x-api-key", "x-goog-api-key", "api-key")
_RESPONSE_DROPPED = {"connection", "content-encoding", "content-length", "transfer-encoding"}


@dataclass(frozen=True)
class ProxyConfig:
    upstreams: dict[str, Upstream] = field(default_factory=upstreams_from_settings)
    harness: str | None = None  # force a harness profile instead of detecting it
    trace: str | None = None  # append every managed request (as received) to this JSONL file
    host: str = "127.0.0.1"
    port: int = 8787

    @classmethod
    def from_settings(cls) -> ProxyConfig:
        return cls(
            harness=settings.value("harness") or None,
            trace=settings.value("trace") or None,
            host=settings.value("host", "127.0.0.1"),
            port=int(settings.value("port", "8787")),
        )


def engine_from_settings() -> Engine:
    engine = Engine(strategy_from_settings(), PrefixStore.from_settings(), event_log=settings.value("event_log") or None)
    engine.window = window_from_settings()
    return engine


def strategy_from_settings() -> Strategy:
    name = settings.value("strategy", "compaction")
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}: one of {', '.join(STRATEGIES)}")
    return STRATEGIES[name].from_settings()


def window_from_settings() -> int | None:
    window = settings.value("context_window")
    return int(window) if window else None


def create_app(engine: Engine | None = None, config: ProxyConfig | None = None) -> Starlette:
    # An engine made here follows the settings file: a strategy changed there (`relay strategy`,
    # `relay set`) applies from the next request on, to every conversation.
    following = engine is None
    engine = engine or engine_from_settings()
    config = config or ProxyConfig.from_settings()
    seen = [settings.version()]

    def follow() -> None:
        if not following or (now := settings.version()) == seen[0]:
            return
        seen[0] = now
        try:
            engine.strategy, engine.window = strategy_from_settings(), window_from_settings()
            log.info("relay now runs %s", engine.strategy.name)
        except Exception:
            log.exception("could not apply the changed settings; %s goes on", engine.strategy.name)
    detect({}, config.harness)  # reject an unknown forced harness at startup

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None))
        app.state.summary_client = httpx.Client(timeout=httpx.Timeout(60.0, read=600.0))
        yield
        await app.state.client.aclose()
        app.state.summary_client.close()
        engine.store.flush()  # a store kept on disk: what changed since it was last written

    async def status(request: Request) -> Response:
        """What Relay runs now (`relay status`, and `relay setup` waiting for it)."""

        follow()
        return JSONResponse({"relay": "ok", "strategy": engine.strategy.name, "window": engine.window,
                             "following": following, "pid": os.getpid()})

    async def handle(request: Request) -> Response:
        if request.headers.get("upgrade"):  # e.g. Codex tries WebSockets first, then HTTP
            return JSONResponse({"error": {"message": "relay: protocol upgrades are not supported"}}, 426)
        client: httpx.AsyncClient = request.app.state.client
        codec = codec_for(request.url.path) if request.method == "POST" else None
        upstream = route(request.url.path, request.headers, config.upstreams)
        url = upstream.url(request.url.path, request.url.query)
        headers = upstream.headers(request.headers)
        content = await request.body()
        body = _json(_decompress(content, headers.get("content-encoding", ""))) if codec else None
        if codec is None or not isinstance(body, dict) or not codec.managed(body):
            return await _passthrough(client, request.method, url, headers, content)
        follow()
        plain = {k: v for k, v in headers.items() if k != "content-encoding"}  # for bodies Relay writes

        summary_client: httpx.Client = request.app.state.summary_client

        def trace(**record: Any) -> None:  # test instrumentation, enabled by the `trace` setting
            if config.trace:
                with open(config.trace, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        def post(payload: dict[str, Any]) -> tuple[int, Any]:  # Relay's own request (a summary)
            response = summary_client.post(url, headers=plain, json=payload)
            if _is_stream(response, payload):
                result = codec.stream_result(_events(response.content))
            else:
                result = _json(response.content)
                result = response.text[:2000] if result is None else result
            trace(path=request.url.path, summary_request=payload, summary_status=response.status_code, answer=result)
            return response.status_code, result

        harness = detect(request.headers, config.harness, request.url.path)

        def prepare(force: bool = False) -> Exchange:
            return engine.prepare(
                codec,
                harness,
                body,
                tenant=next((request.headers[h] for h in TENANT_HEADERS if h in request.headers), ""),
                post=post,
                force=force,
            )

        try:
            exchange = await run_in_threadpool(prepare)
        except Exception:
            log.exception("could not prepare the request; forwarding it unchanged")
            return await _passthrough(client, "POST", url, headers, content)

        items = harness.codec(codec).items  # as the engine counts them
        trace(path=request.url.path, user_agent=request.headers.get("user-agent"),
              items=len(items(body)), sent=len(items(exchange.body)),
              rewritten=exchange.body is not body, compacted=exchange.compacted, body=body,
              forwarded=exchange.body if exchange.body is not body else None,
              cache={"depth": exchange.depth, "covered": exchange.state.get("covered", 0), "diverged": exchange.diverged})

        try:
            if exchange.body is body:
                response = await _send(client, url, headers, content)
            else:
                response = await _send(client, url, plain, exchange.body)
            if response.status_code >= 400:
                error = await response.aread()
                await response.aclose()
                if not codec.is_overflow(response.status_code, _json(error)) or exchange.compacted:
                    return Response(error, response.status_code, _response_headers(response.headers))
                retry = await run_in_threadpool(prepare, True)  # compact now, then try once more
                if not retry.compacted:
                    return Response(error, response.status_code, _response_headers(response.headers))
                exchange = retry
                response = await _send(client, url, plain, exchange.body)
        except httpx.HTTPError as exc:
            return _unreachable(exc)
        streaming = _is_stream(response, exchange.body)

        def on_usage(tokens: int | None, cached: int | None) -> None:
            engine.record(exchange, tokens)
            trace(usage=tokens, cached=cached, status=response.status_code)

        return await _relay(response, codec, streaming, on_usage)

    return Starlette(
        routes=[*hook_routes(engine), Route("/relay/v1/status", status, methods=["GET"]),
                Route("/{path:path}", handle, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])],
        lifespan=lifespan,
    )


async def _send(client: httpx.AsyncClient, url: str, headers: dict[str, str], body: Any) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
    request = client.build_request("POST", url, headers=headers, content=content)
    return await client.send(request, stream=True)


async def _passthrough(
    client: httpx.AsyncClient, method: str, url: str, headers: dict[str, str], content: bytes
) -> Response:
    try:
        request = client.build_request(method, url, headers=headers, content=content)
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        return _unreachable(exc)
    return StreamingResponse(
        response.aiter_bytes(),
        response.status_code,
        _response_headers(response.headers),
        background=BackgroundTask(response.aclose),
    )


async def _relay(
    response: httpx.Response, codec: Codec, streaming: bool, on_usage: Callable[[int | None, int | None], None]
) -> Response:
    headers = _response_headers(response.headers)
    if not streaming:
        data = await response.aread()
        await response.aclose()
        payload = _json(data)
        on_usage(*((codec.usage(payload), codec.cached(payload)) if isinstance(payload, dict) else (None, None)))
        return Response(data, response.status_code, headers)

    tap = _UsageTap(codec)

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                tap.feed(chunk)
                yield chunk
        finally:
            await response.aclose()
            on_usage(tap.tokens, tap.cached)

    return StreamingResponse(stream(), response.status_code, headers)


class _UsageTap:
    """Reads prompt-token usage (and how much of it the cache served) from a server-sent event
    stream without altering it."""

    def __init__(self, codec: Codec) -> None:
        self.codec = codec
        self.pending = b""
        self.tokens: int | None = None
        self.cached: int | None = None

    def feed(self, chunk: bytes) -> None:
        *lines, self.pending = (self.pending + chunk).split(b"\n")
        for line in lines:
            if line.startswith(b"data:"):
                payload = _json(line[5:].strip())
                if isinstance(payload, dict) and (tokens := self.codec.usage(payload)) is not None:
                    self.tokens = tokens
                if isinstance(payload, dict) and (cached := self.codec.cached(payload)) is not None:
                    self.cached = cached


def _unreachable(exc: Exception) -> Response:
    return JSONResponse({"error": {"message": f"relay: upstream unreachable: {exc}"}}, 502)


def _decompress(content: bytes, encoding: str) -> bytes:
    """Request bodies may be compressed (Codex sends zstd to the ChatGPT backend)."""

    try:
        if encoding == "zstd":
            import zstandard

            return zstandard.ZstdDecompressor().decompressobj().decompress(content)
        if encoding in {"gzip", "deflate"}:
            return zlib.decompress(content, zlib.MAX_WBITS | 32)
    except Exception:
        return b""  # not readable: the request is passed through untouched
    return content


def _is_stream(response: httpx.Response, request: dict[str, Any]) -> bool:
    # Some backends (e.g. ChatGPT's Codex endpoint) stream without an SSE content type.
    streamed = "text/event-stream" in response.headers.get("content-type", "") or request.get("stream") is True
    return response.status_code < 400 and streamed


def _events(content: bytes) -> list[dict[str, Any]]:
    payloads = (_json(line[5:].strip()) for line in content.split(b"\n") if line.startswith(b"data:"))
    return [payload for payload in payloads if isinstance(payload, dict)]


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _RESPONSE_DROPPED}


def _json(content: bytes) -> Any:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
