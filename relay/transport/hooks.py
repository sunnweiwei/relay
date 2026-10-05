"""Relay's endpoint for harness hooks: a harness compacting its own history asks Relay's
strategy what the history should become (`relay.integrations`)."""

from __future__ import annotations

import logging

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..core.engine import Engine
from ..integrations import HOOKS

log = logging.getLogger("relay")


def hook_routes(engine: Engine) -> list[Route]:
    async def compact(request: Request) -> Response:
        try:
            payload = await request.json()
            hook = HOOKS[payload.get("harness")]
            return JSONResponse(await run_in_threadpool(hook.compact, engine, payload))
        except Exception as error:  # the harness then compacts as it would without Relay
            log.warning("hook compaction failed", exc_info=True)
            return JSONResponse({"error": {"message": f"relay: {error!r:.300}"}}, 500)

    return [Route("/relay/v1/compact", compact, methods=["POST"])]
