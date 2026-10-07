"""Relay's endpoints for a harness's side: a harness compacting its own history asks Relay's
strategy what the history should become (`relay.integrations`); a harness reports what it knows
of its session that its requests do not show (`relay.core.local`)."""

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

    async def local(request: Request) -> Response:
        try:
            report = await request.json()
            if not isinstance(report, dict):
                raise ValueError("a report is a JSON object")
            local = engine.locals.update(report)
            return JSONResponse({"ok": True, "files": bool(local.files)})  # (a restarted Relay has none to keep)
        except Exception as error:
            log.warning("could not keep a harness's report", exc_info=True)
            return JSONResponse({"error": {"message": f"relay: {error!r:.300}"}}, 400)

    return [Route("/relay/v1/compact", compact, methods=["POST"]), Route("/relay/v1/local", local, methods=["POST"])]
