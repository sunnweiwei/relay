"""Exercise live Harness adapters against a local API, without a billable call."""
from __future__ import annotations

import json

import pytest
from starlette.responses import JSONResponse, Response

from tests.test_codex_e2e import _is_summary_request, _response, _serve, _sse, _text

from experiments.compact_cache.checks import StableCompactUpstream
from experiments.compact_cache.providers import ModelRoute


class LiveWireUpstream(StableCompactUpstream):
    async def dispatch(self, request):
        if request.url.path == "/v1/responses/input_tokens":
            body = await request.json()
            self.count_requests.append(body)
            contents = " ".join(_text(item) for item in body.get("input", []))
            if _is_summary_request(body) or "Another language model started" in contents:
                count = 100
            elif ("Turn 2" in contents or
                  any(item.get("type") == "function_call_output"
                      for item in body.get("input", []))):
                count = 2000
            else:
                count = 1000
            return JSONResponse({"object": "response.input_tokens",
                                 "input_tokens": count})
        if request.url.path == "/v1/responses":
            body = await request.json()
            if not _is_summary_request(body) and not self.mini_mode:
                self.main_requests.append(body)
                contents = " ".join(_text(item) for item in body.get("input", []))
                reply = "BLUE-ORCHID-731" if "Turn 3" in contents else "ACK"
                response_id = f"resp_bridge_{len(self.main_requests)}"
                if body.get("stream"):
                    return Response(_sse(body, reply, response_id),
                                    media_type="text/event-stream")
                return JSONResponse(_response(body, reply, response_id))
        response = await super().dispatch(request)
        if self.mini_mode and request.url.path == "/v1/responses":
            body = response.body.replace(
                b"relay-mini-ok", b"BLUE-ORCHID-731"
            )
            headers = dict(response.headers)
            headers.pop("content-length", None)
            return Response(
                content=body,
                status_code=response.status_code,
                headers=headers,
            )
        return response


@pytest.mark.parametrize("harness", (
    "codex", "opencode", "pi", "mini-swe", "gemini-cli", "claude-code"))
def test_live_route_reaches_local_model_api(harness: str, monkeypatch) -> None:
    upstream = LiveWireUpstream(mini_mode=harness == "mini-swe")
    with _serve(upstream.app) as url:
        route = ModelRoute("openai", "gpt-6-luna", f"{url}/v1", "local-test")
        monkeypatch.setenv("RELAY_LIVE_TESTS", "1")
        if harness in {"codex", "opencode", "pi", "mini-swe"}:
            from experiments.compact_cache import live_responses as module
            monkeypatch.setattr(module, "route_for", lambda model: route)
            record = module.run(harness, "gpt-6-luna")
        elif harness == "gemini-cli":
            from experiments.compact_cache import live_gemini as module
            monkeypatch.setattr(module, "route_for", lambda model: route)
            record = module.run("gpt-6-luna")
        else:
            from experiments.compact_cache import live_claude as module
            monkeypatch.setattr(module, "route_for", lambda model: route)
            record = module.run("gpt-6-luna")

    assert record["task_requests"] >= 2
    assert record["summary_calls"] >= 1, {
        "task_requests": record["task_requests"],
        "count_calls": record.get("count_calls"),
        "stages": record["stages"],
    }
    assert upstream.main_requests
    calibration = record["threshold_calibration"]
    if harness in {"pi", "mini-swe", "gemini-cli", "claude-code"}:
        assert (calibration["first_input_tokens"]
                < calibration["compact_threshold"]
                < calibration["second_input_tokens"])
    else:
        assert calibration["compact_threshold"] > calibration["largest_atomic_summary_tokens"]
    assert record["relay_requests"] >= record["task_model_requests"]
    assert record["turn_traces"]
    assert all(request.get("max_output_tokens", 0) <= 512
               for request in upstream.main_requests)
    assert record.get("budgeted_task_input_tokens", 0) <= 24_000
    assert record["status"] == "pass", json.dumps(record, indent=2)
