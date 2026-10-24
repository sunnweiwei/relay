"""Relay's endpoints for harness hooks: a harness compacting its own history asks Relay's
strategy what the history should become, what instructions its system prompt gets, and
what a tool's result should say as it comes in (`relay.integrations`)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..core.engine import Engine
from ..integrations import HOOKS, uncarried

log = logging.getLogger("relay")


def hook_routes(engine: Engine, follow: Callable[[], None], noted: Callable[[str, str], None]) -> list[Route]:
    """`follow` applies changed settings; `noted` records the harness's release its hook reports."""

    async def answer(request: Request, kind: str) -> Response:
        try:
            follow()
            payload = await request.json()
            name = payload.get("harness")
            hook = HOOKS[name]
            if payload.get("version"):
                noted(name, payload["version"])
            if why := uncarried(hook, engine.strategy.name):  # (`relay strategy` moves the harness to the proxy)
                notice = (f"Relay: {why}, so it runs through the proxy, for new sessions; this one "
                          f"compacts on its own")
                return JSONResponse({"error": {"message": f"relay: {why}; it runs through the proxy"},
                                     "notice": notice}, 409)
            if kind == "instructions":
                return JSONResponse({"instructions": await run_in_threadpool(hook.instructions, engine)})
            if kind == "result":
                return JSONResponse(await run_in_threadpool(hook.result, engine, payload))
            return JSONResponse(await run_in_threadpool(hook.compact, engine, payload))
        except Exception as error:  # the harness then compacts as it would without Relay
            log.warning("hook %s failed", kind, exc_info=True)
            return JSONResponse({"error": {"message": f"relay: {error!r:.300}"}}, 500)

    async def compact(request: Request) -> Response:
        return await answer(request, "compaction")

    async def instructions(request: Request) -> Response:
        return await answer(request, "instructions")

    async def result(request: Request) -> Response:
        return await answer(request, "result")

    return [Route("/relay/v1/compact", compact, methods=["POST"]),
            Route("/relay/v1/instructions", instructions, methods=["POST"]),
            Route("/relay/v1/result", result, methods=["POST"])]
