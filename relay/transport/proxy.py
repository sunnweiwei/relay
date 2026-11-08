"""HTTP proxy in front of the model APIs.

Conversation endpoints (one per codec) go through the engine; every other request,
and every response body, passes through unchanged. Streaming responses are relayed
as they arrive while a tap reads the reported prompt tokens. The same server answers
harness hooks (`hooks.py`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
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
from ..core.budget import (MEDIA_INPUT_BUDGET_COUNTER, PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER,
                           BudgetExceeded, InputBudget, InputAccountingError)
from ..core.store import PrefixStore
from .. import leaving
from ..harnesses import Harness, detect
from ..providers import context_window, refused, window_of
from ..protocols import Codec, codec_for
from ..protocols.base import Body, error_message
from ..providers import Upstream, route, upstreams_from_settings
from ..strategies import STRATEGIES
from ..strategies.base import Strategy
from . import outbound
from .hooks import hook_routes
from .. import intercept
from .. import private, service, settings

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
    intercept_port: int | None = None  # Relay as an HTTPS proxy too, here (`relay.intercept`)

    @classmethod
    def from_settings(cls) -> ProxyConfig:
        intercept = settings.value("intercept_port")
        return cls(
            harness=settings.value("harness") or None,
            trace=settings.value("trace") or None,
            host=settings.value("host", "127.0.0.1"),
            port=int(settings.value("port", "8787")),
            intercept_port=int(intercept) if intercept else None,
        )


def engine_from_settings() -> Engine:
    engine = Engine(strategy_from_settings(), PrefixStore.from_settings(), event_log=settings.value("event_log") or None,
                    input_budget=budget_from_settings(), harness_profile=settings.value("harness_profile") or None)
    engine.window = window_from_settings()
    return engine


def budget_from_settings() -> InputBudget | None:
    counter = settings.value("input_budget_counter")
    if not counter:
        return None
    if counter not in {InputBudget.name, MEDIA_INPUT_BUDGET_COUNTER,
                       PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER}:
        raise ValueError(f"Unsupported input budget counter: {counter}")
    limit = settings.value("input_budget_limit")
    if not limit:
        raise ValueError("input_budget_limit is required for a strict counter")
    context = settings.value("budget_context_limit")
    return InputBudget(int(limit), int(context) if context else None,
                       int(settings.value("budget_output_reserve", "0")),
                       name=counter,
                       dispatch_slack=int(settings.value("budget_dispatch_slack", "0")),
                       overflow_fallback=settings.value("overflow_fallback", "error"),
                       unknown_opaque=settings.value("budget_unknown_opaque", "error"))


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
            strategy, window, budget = strategy_from_settings(), window_from_settings(), budget_from_settings()
            if budget and engine.input_budget and budget.fingerprint() == engine.input_budget.fingerprint():
                budget = engine.input_budget  # Preserve observed reasoning usage across settings reloads.
            engine.strategy, engine.window, engine.input_budget = strategy, window, budget
            engine.harness_profile = settings.value("harness_profile") or None
            log.info("relay now runs %s", engine.strategy.name)
        except Exception:
            log.exception("could not apply the changed settings; %s goes on", engine.strategy.name)
    detect({}, config.harness)  # reject an unknown forced harness at startup
    releases: dict[str, str] = {}  # each harness's release, as its latest request said it
    last: dict[str, float] = {}  # when each harness's latest request came
    running = service.code()

    def release(harness: Harness, headers: Mapping[str, str]) -> None:
        if version := harness.version(headers):
            noted(harness.name, version)

    def noted(name: str, version: str) -> None:
        if releases.get(name) != version:
            releases[name] = version
            if (harness := detect({}, name)).newer(version):
                log.warning("%s %s is newer than the releases its profile was checked against (%s); Relay serves it "
                            "as the newest of them", name, version, harness.verified)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        app.state.client = outbound.async_client(timeout=httpx.Timeout(60.0, read=None))
        app.state.summary_client = outbound.client(timeout=httpx.Timeout(60.0, read=600.0))
        tunnel = await intercept.serve(config.host, config.intercept_port, config.port) if config.intercept_port else None
        yield
        if tunnel is not None:
            tunnel.close()
        await app.state.client.aclose()
        app.state.summary_client.close()
        engine.store.flush()  # a store kept on disk: what changed since it was last written
        leaving.flush()

    async def status(request: Request) -> Response:
        """What Relay runs now (`relay status`, and `relay setup` waiting for it)."""

        follow()
        leaving.flush()  # (`relay status`, `relay uninstall` read what it noted)
        return JSONResponse({"relay": "ok", "strategy": engine.strategy.name, "window": engine.window,
                             "following": following, "pid": os.getpid(), "code": running, "last": last,
                             "harnesses": {name: {"release": version, "checked": detect({}, name).verified}
                                           for name, version in releases.items()}})

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
        if isinstance(body, dict) and "thread" in body:
            # A server-side thread (Claude Code on Anthropic's own endpoint, through Relay's HTTPS
            # proxy) would leave the history with the server and send Relay only what is new. Relay
            # answers as a server without threads does: Claude Code sends the turn again without one,
            # and keeps the session (on that model) without them, its whole conversation every time.
            return JSONResponse({"type": "error", "error": {
                "type": "invalid_request_error", "message": "relay: server-side threads are not supported through Relay",
                "details": {"error_code": "thread_unsupported_request"}}}, 400)
        if codec is None or not isinstance(body, dict) or not codec.managed(body):
            if engine.input_budget and codec is not None:
                return JSONResponse({"error": {"code": "unsupported_budget_request",
                    "message": "A bounded evaluation requires a fully visible, stateless request"}}, 400)
            return await _passthrough(client, request.method, url, headers, content)
        follow()
        plain = {k: v for k, v in headers.items() if k != "content-encoding"}  # for bodies Relay writes

        summary_client: httpx.Client = request.app.state.summary_client

        def trace(**record: Any) -> None:  # test instrumentation, enabled by the `trace` setting
            if config.trace:
                private.append(config.trace, json.dumps(record, ensure_ascii=False) + "\n")

        def post(payload: dict[str, Any]) -> tuple[int, Any]:  # Relay's own request (a summary)
            if engine.input_budget:
                engine.input_budget.check(codec, payload, "auxiliary")
            try:
                response = summary_client.post(url, headers=plain, json=payload)
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                # The upstream dropped the connection mid-response: as retryable as its 503.
                log.warning("Relay's own request lost its connection: %s", exc)
                trace(path=request.url.path, summary_request=payload, summary_error=f"{type(exc).__name__}: {exc}")
                return 503, {"error": {"message": f"relay: upstream connection lost: {exc}"}}
            if _is_stream(response, payload):
                result = codec.stream_result(_events(response.content))
            else:
                result = _json(response.content)
                result = response.text[:2000] if result is None else result
            if engine.input_budget and isinstance(result, dict) and response.status_code < 400:
                engine.input_budget.observe_response(result)
            trace(path=request.url.path, summary_request=payload, summary_status=response.status_code, answer=result,
                  **({"budget_tokens": engine.input_budget.count(codec, payload),
                      "budget_policy": engine.input_budget.fingerprint()} if engine.input_budget else {}))
            return response.status_code, result

        harness = detect(request.headers, config.harness, request.url.path)
        release(harness, request.headers)
        last[harness.name] = time.time()
        # The model's own window (not Relay's threshold): what fits once the harness goes direct.
        window = harness.window(body.get("model")) or context_window(body.get("model")) or engine.window
        window = window_of(body.get("model"), max(window or 0, harness.direct_window or 0) or None)
        departing = harness.name in leaving.leaving() and window  # (`relay uninstall`: see relay/leaving.py)

        def prepare(force: bool = False, fallback_limit: int | None = None) -> Exchange:
            return engine.prepare(
                codec,
                harness,
                body,
                tenant=next((request.headers[h] for h in TENANT_HEADERS if h in request.headers), ""),
                post=post,
                force=force,
                fallback_limit=fallback_limit,
                on_fallback=lambda record: trace(overflow_fallback=record),
                direct_within=(lambda name: leaving.direct_within(harness.name, name, window)) if departing and not force
                else None,
            )

        def noted(exchange: Exchange) -> None:
            leaving.seen(harness.name, exchange.lasting or exchange.conversation, exchange.title, exchange.whole, window,
                         exchange.compacting, session=harness.session(request.headers))
            if departing and (restored := leaving.settle()):
                log.info("no conversation of %s needs Relay any more: its configuration is restored", ", ".join(restored))

        def budget_failure(error: BudgetExceeded) -> Response:
            trace(budget_rejection=error.payload()["error"], recovery_attempted=True)
            return JSONResponse(error.payload(), 400)

        def log_accounting_failure(error: InputAccountingError, phase: str) -> None:
            diagnostics = getattr(error, "media_diagnostics", None)
            if diagnostics:
                log.warning("media accounting diagnostic %s", json.dumps({
                    "category": error.category, "phase": phase, "images": diagnostics,
                }, sort_keys=True, separators=(",", ":")))
            log.exception("input accounting failed %s", phase)

        recovered = False
        try:
            try:
                exchange = await run_in_threadpool(prepare)
            except BudgetExceeded as error:
                trace(budget_rejection=error.payload()["error"], recovery_attempted=False)
                recovered = True
                exchange = await run_in_threadpool(prepare, True)
        except BudgetExceeded as error:
            return budget_failure(error)
        except httpx.TimeoutException:
            if not engine.input_budget:
                log.exception("could not prepare the request; forwarding it unchanged")
                return await _passthrough(client, "POST", url, headers, content)
            log.exception("upstream auxiliary request timed out")
            return JSONResponse({"error": {"code": "upstream_request_timeout",
                "message": "An auxiliary model request timed out"}}, 504)
        except InputAccountingError as error:
            log_accounting_failure(error, "before dispatch")
            return JSONResponse({"error": {"code": "input_accounting_failed",
                "message": "The input counter could not account for the request"}}, 500)
        except Exception:
            if engine.input_budget:
                log.exception("bounded context preparation failed; no request was forwarded")
                return JSONResponse({"error": {"code": "context_management_failed",
                    "message": "Context management failed before dispatch; see the Relay log"}}, 500)
            log.exception("could not prepare the request; forwarding it unchanged")
            return await _passthrough(client, "POST", url, headers, content)

        items = harness.codec(codec).items  # as the engine counts them

        def trace_exchange(exchange: Exchange, recovered: bool) -> None:
            trace(path=request.url.path, user_agent=request.headers.get("user-agent"),
                  items=len(items(body)), sent=len(items(exchange.body)), direct=exchange.direct,
                  rewritten=exchange.body is not body, compacted=exchange.compacted, body=body,
                  forwarded=exchange.body if exchange.body is not body else None,
                  **({"budget_tokens": exchange.estimate, "budget_policy": engine.input_budget.fingerprint(),
                      "recovery_attempted": recovered} if engine.input_budget else {}),
                  cache={"depth": exchange.depth, "covered": exchange.state.get("covered", 0), "diverged": exchange.diverged})

        trace_exchange(exchange, recovered)

        try:
            if engine.input_budget:
                engine.input_budget.check(codec, exchange.body, "main")
            # (`upstream_429_wait`: seconds Relay itself waits out an upstream rate limit before passing
            # it on, for an evaluation whose harness gives up on a short burst of them; off by default)
            wait, delay, waited = float(settings.value("upstream_429_wait", "0") or 0), 2.0, 0.0
            while True:
                if exchange.body is body:
                    response = await _send(client, url, headers, content)
                else:
                    response = await _send(client, url, plain, exchange.body)
                if response.status_code != 429 or waited >= wait:
                    break
                await response.aclose()
                log.info("upstream 429: sending again in %.0f s", delay)
                await asyncio.sleep(delay)
                waited, delay = waited + delay, min(delay * 2, 30.0)
            if response.status_code >= 400:
                error = await response.aread()
                await response.aclose()
                log.warning("upstream refused %s (%d): %s", request.url.path, response.status_code, error[:400])
                overflow = codec.is_overflow(response.status_code, refusal := _json(error))
                if overflow and (limit := refused(body.get("model"), error_message(refusal)[1])):
                    log.info("%s is refused past %d tokens: Relay counts that as its window", body.get("model"), limit)
                if engine.input_budget:
                    # A bounded evaluation makes at most one forced retry, including after an
                    # ordinary edit or guard, and reports a second rejection as a budget failure.
                    if recovered and overflow:
                        return budget_failure(BudgetExceeded(exchange.estimate, engine.input_budget.limit_for(exchange.body),
                            "main", "The provider rejected input after the one permitted recovery attempt."))
                    if recovered or not overflow:
                        return Response(error, response.status_code, _response_headers(response.headers))
                    target = (int(exchange.estimate * 0.9)
                              if engine.input_budget.overflow_fallback == "truncate" else None)
                    retry = await run_in_threadpool(prepare, True, target)  # compact, then bounded fallback
                    if not retry.compacted or retry.body == exchange.body:
                        return budget_failure(BudgetExceeded(retry.estimate, engine.input_budget.limit_for(retry.body),
                            "main", "The method did not produce a changed context after provider rejection."))
                else:
                    if not overflow or exchange.compacted:
                        return Response(error, response.status_code, _response_headers(response.headers))
                    if exchange.direct:  # sent as going direct would, and too long: it does not fit direct
                        await run_in_threadpool(leaving.refused, harness.name, exchange.lasting or exchange.conversation)
                    retry = await run_in_threadpool(prepare, True)  # compact now, then try once more
                    if not retry.compacted:
                        return Response(error, response.status_code, _response_headers(response.headers))
                exchange = retry
                if engine.input_budget:
                    engine.input_budget.check(codec, exchange.body, "main")
                    trace_exchange(exchange, True)
                response = await _send(client, url, plain, exchange.body)
                if engine.input_budget and response.status_code >= 400:
                    retried_error = await response.aread()
                    if codec.is_overflow(response.status_code, _json(retried_error)):
                        await response.aclose()
                        return budget_failure(BudgetExceeded(exchange.estimate, engine.input_budget.limit_for(exchange.body),
                            "main", "The provider rejected input after the one permitted recovery attempt."))
        except BudgetExceeded as error:
            return budget_failure(error)
        except httpx.TimeoutException as exc:
            if not engine.input_budget:
                return _unreachable(exc)
            log.exception("upstream request timed out during recovery")
            return JSONResponse({"error": {"code": "upstream_request_timeout",
                "message": "A model request timed out during recovery"}}, 504)
        except InputAccountingError as error:
            log_accounting_failure(error, "during recovery")
            return JSONResponse({"error": {"code": "input_accounting_failed",
                "message": "The input counter could not account for recovery"}}, 500)
        except httpx.HTTPError as exc:
            return _unreachable(exc)
        except Exception:
            if not engine.input_budget:
                raise
            log.exception("bounded overflow recovery failed; no unmanaged retry was sent")
            return JSONResponse({"error": {"code": "context_management_failed",
                "message": "Context recovery failed before dispatch; see the Relay log"}}, 500)
        try:  # (once the upstream took it: a conversation going direct may yet turn out not to fit)
            if response.status_code < 400:
                await run_in_threadpool(noted, exchange)
        except Exception:
            log.exception("could not note the conversation")
        streaming = _is_stream(response, exchange.body)

        def on_usage(tokens: int | None, cached: int | None) -> None:
            engine.record(exchange, tokens)
            trace(usage=tokens, cached=cached, status=response.status_code)

        def on_response(payload: Body) -> None:
            if engine.input_budget:
                records = engine.input_budget.observe_response(payload)
                if records:
                    trace(reasoning_usage=records)

        return await _relay(response, codec, streaming, on_usage, on_response if engine.input_budget else None)

    return Starlette(
        routes=[*hook_routes(engine, follow, noted), Route("/relay/v1/status", status, methods=["GET"]),
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
    if response.status_code >= 400:  # (an error is short: kept in the log, for whoever looks into it)
        data = await response.aread()
        await response.aclose()
        log.warning("upstream refused %s %s (%d): %s", method, url.split("?")[0], response.status_code, data[:400])
        return Response(data, response.status_code, _response_headers(response.headers))
    return StreamingResponse(
        response.aiter_bytes(),
        response.status_code,
        _response_headers(response.headers),
        background=BackgroundTask(response.aclose),
    )


async def _relay(
    response: httpx.Response, codec: Codec, streaming: bool, on_usage: Callable[[int | None, int | None], None],
    on_response: Callable[[Body], None] | None = None,
) -> Response:
    headers = _response_headers(response.headers)
    if response.status_code >= 400:  # (an error is short: kept in the log, for whoever looks into it)
        data = await response.aread()
        await response.aclose()
        log.warning("upstream refused %s (%d): %s", response.request.url.path, response.status_code, data[:400])
        on_usage(None, None)
        return Response(data, response.status_code, headers)
    if not streaming:
        data = await response.aread()
        await response.aclose()
        payload = _json(data)
        if on_response and isinstance(payload, dict):
            on_response(payload)
        on_usage(*((codec.usage(payload), codec.cached(payload)) if isinstance(payload, dict) else (None, None)))
        return Response(data, response.status_code, headers)

    tap = _UsageTap(codec, on_response)

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

    def __init__(self, codec: Codec, on_response: Callable[[Body], None] | None = None) -> None:
        self.codec = codec
        self.pending = b""
        self.tokens: int | None = None
        self.cached: int | None = None
        self.on_response = on_response
        self.reasoning_items: list[Body] = []

    def feed(self, chunk: bytes) -> None:
        *lines, self.pending = (self.pending + chunk).split(b"\n")
        for line in lines:
            if line.startswith(b"data:"):
                payload = _json(line[5:].strip())
                if self.on_response and isinstance(payload, dict):
                    if payload.get("type") == "response.output_item.done":
                        item = payload.get("item") or {}
                        if item.get("type") == "reasoning":
                            self.reasoning_items.append(item)
                    if payload.get("type") in {"response.completed", "response.incomplete"}:
                        final = payload.get("response") or {}
                        self.on_response({**final, "output": final.get("output") or self.reasoning_items})
                        self.reasoning_items = []
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
