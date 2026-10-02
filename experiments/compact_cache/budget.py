"""Hard token and request bounds for a single billable compatibility run."""
from __future__ import annotations

import json
from typing import Any

import httpx
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

MAX_TASK_REQUESTS = 4
MAX_INPUT_PER_REQUEST = 12_000
MAX_TOTAL_TASK_INPUT = 24_000
MAX_TASK_OUTPUT = 512
MAX_SUMMARY_OUTPUT = 256


class TaskBudget:
    def __init__(self, client: OpenAI, *, max_requests: int = MAX_TASK_REQUESTS,
                 max_total_input: int = MAX_TOTAL_TASK_INPUT,
                 max_input_per_request: int = MAX_INPUT_PER_REQUEST,
                 max_output: int = MAX_TASK_OUTPUT) -> None:
        self.client = client
        self.max_requests = max_requests
        self.max_total_input = max_total_input
        self.max_input_per_request = max_input_per_request
        self.max_output = max_output
        self.requests = 0
        self.input_tokens = 0

    def reserve(self, body: dict[str, Any]) -> int:
        if self.requests >= self.max_requests:
            raise ValueError("task request budget exceeded")
        fields = {key: body[key] for key in ("model", "input", "instructions", "reasoning")
                  if key in body}
        tokens = self.client.responses.input_tokens.count(**fields).input_tokens
        if tokens > self.max_input_per_request:
            raise ValueError(f"task input {tokens} exceeds {self.max_input_per_request} tokens")
        if self.input_tokens + tokens > self.max_total_input:
            raise ValueError("cumulative task input token budget exceeded")
        self.requests += 1
        self.input_tokens += tokens
        return tokens

    def bounded(self, body: dict[str, Any], *, output: int = MAX_TASK_OUTPUT) -> dict[str, Any]:
        self.reserve(body)
        cap = min(output, self.max_output)
        return {**body, "max_output_tokens": min(
            int(body.get("max_output_tokens") or cap), cap)}


class BoundedResponsesTransport(httpx.AsyncBaseTransport):
    """Cap the task call after Relay compaction, immediately before the API."""

    def __init__(self, budget: TaskBudget) -> None:
        self.budget = budget
        self.inner = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses":
            return await self.inner.handle_async_request(request)
        body = json.loads(request.content)
        try:
            bounded = await run_in_threadpool(self.budget.bounded, body)
        except ValueError as exc:
            return httpx.Response(429, json={"error": {"message": str(exc)}}, request=request)
        content = json.dumps(bounded).encode()
        headers = dict(request.headers)
        headers.pop("content-length", None)
        capped = httpx.Request(request.method, request.url, headers=headers, content=content,
                               extensions=request.extensions)
        return await self.inner.handle_async_request(capped)

    async def aclose(self) -> None:
        await self.inner.aclose()
