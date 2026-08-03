from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, GeneratedCheckpoint, PreparedInput
from .compact import _protected_prefix

# Official implementation:
# https://github.com/alexzhang13/rlm/tree/72d6940142ddfb84ee6be573dc999a37e633e671
OFFICIAL_RLM_VERSION = "0.1.3"
OFFICIAL_RLM_COMMIT = "72d6940142ddfb84ee6be573dc999a37e633e671"

RLM_ROOT_PROMPT = """Determine the next assistant turn for the coding-agent request stored in `context`.

Inspect the request and its complete append-only trajectory before answering. The result
will be passed to a Responses API renderer that has the same instructions and tool schemas.
Return a self-contained proposal for exactly the next assistant turn:

- If the task needs a tool, identify the exact tool name and arguments.
- If the task can finish, provide the exact final response.
- Preserve identifiers, paths, commands, constraints, and relevant tool results exactly.

Use the RLM REPL and recursive LM calls to analyze the context. Do not merely summarize the
trajectory, and do not discuss this instruction in the final proposal."""

RLM_HANDOFF_PREFIX = "Result from the Recursive Language Model:\n"
_RENDER_INSTRUCTION = """

Produce the next assistant response now. Follow the original instructions and available
tool schemas. If the proposal calls for a tool, emit the corresponding native tool call.
Do not mention the Recursive Language Model or this handoff."""


@dataclass
class RLM(BaseStrategy):
    """Run the official Recursive Language Model before rendering one agent turn."""

    manager_model: str | None = None
    max_depth: int = 1
    max_iterations: int = 30
    environment: str = "local"
    max_timeout: float | None = None
    max_tokens: int | None = None
    orchestrator: bool = True
    name: str = field(default="rlm", init=False)

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_timeout is not None and self.max_timeout <= 0:
            raise ValueError("max_timeout must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not self.environment:
            raise ValueError("environment is required")

    @classmethod
    def from_env(cls) -> RLM:
        return cls(
            manager_model=os.getenv("RELAY_RLM_MODEL") or None,
            max_depth=int(os.getenv("RELAY_RLM_MAX_DEPTH", "1")),
            max_iterations=int(os.getenv("RELAY_RLM_MAX_ITERATIONS", "30")),
            environment=os.getenv("RELAY_RLM_ENVIRONMENT", "local"),
            max_timeout=_optional_float("RELAY_RLM_MAX_TIMEOUT"),
            max_tokens=_optional_int("RELAY_RLM_MAX_TOKENS"),
            orchestrator=_environment_bool("RELAY_RLM_ORCHESTRATOR", True),
        )

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint is not None:
            raise ValueError("rlm does not use checkpoint artifacts")
        return deepcopy(trajectory)

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        active = self.materialize(trajectory, checkpoint)
        rendered = self._run(responses, request, active)
        return PreparedInput(rendered, compacted=rendered != active)

    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self._run(responses, request, active)

    def cache_scope(self) -> dict[str, Any]:
        return {
            "official_rlm_commit": OFFICIAL_RLM_COMMIT,
            "official_rlm_version": OFFICIAL_RLM_VERSION,
            "manager_model": self.manager_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "environment": self.environment,
            "max_timeout": self.max_timeout,
            "max_tokens": self.max_tokens,
            "orchestrator": self.orchestrator,
            "persistent": False,
            "compaction": False,
            "prompt_version": 1,
        }

    def _run(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        model = self.manager_model or request.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("a model is required for RLM")

        runtime = _official_runtime(
            backend="openai",
            backend_kwargs=_backend_kwargs(responses, model),
            environment=self.environment,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_timeout=self.max_timeout,
            max_tokens=self.max_tokens,
            orchestrator=self.orchestrator,
            persistent=False,
            compaction=False,
        )
        try:
            result = runtime.completion(
                _request_context(request, trajectory),
                root_prompt=RLM_ROOT_PROMPT,
            )
            proposal = getattr(result, "response", result)
            if not isinstance(proposal, str) or not proposal.strip():
                raise RuntimeError("official RLM returned no response")
        finally:
            runtime.close()

        return [
            *_protected_prefix(trajectory),
            {
                "type": "message",
                "role": "user",
                "content": f"{RLM_HANDOFF_PREFIX}{proposal.strip()}{_RENDER_INSTRUCTION}",
            },
        ]


def _official_runtime(**kwargs: Any) -> Any:
    try:
        from rlm import RLM as OfficialRLM
    except ImportError as exc:
        raise RuntimeError(
            "RLM requires the official package: pip install 'relay[rlm]'"
        ) from exc
    return OfficialRLM(**kwargs)


def _backend_kwargs(responses: Any, model: str) -> dict[str, Any]:
    values: dict[str, Any] = {"model_name": model}
    client = getattr(responses, "_client", None)
    if client is None:
        return values

    api_key = getattr(client, "api_key", None)
    if isinstance(api_key, str) and api_key:
        values["api_key"] = api_key
    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        values["base_url"] = str(base_url)
    for key in ("organization", "project"):
        value = getattr(client, key, None)
        if isinstance(value, str) and value:
            values[key] = value
    return values


def _request_context(
    request: dict[str, Any], trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    ignored = {
        "background",
        "context_management",
        "conversation",
        "input",
        "previous_response_id",
        "store",
        "stream",
    }
    return {
        **{
            key: deepcopy(value) for key, value in request.items() if key not in ignored
        },
        "input": deepcopy(trajectory),
    }


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value in {None, ""} else float(value)


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value in {None, ""} else int(value)


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
