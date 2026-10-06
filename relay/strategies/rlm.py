"""RLM (Recursive Language Models, Zhang, Kraska and Khattab; github.com/alexzhang13/rlm) as a Relay
strategy: the conversation becomes a variable in a Python REPL, and the model works out the agent's
next step by writing code over it and calling itself on parts of it.

Each request runs the authors' package (`rlms`, `pip install 'relay[rlm]'`): the conversation (the
user's messages, the agent's, its tool calls and their results, as `{"role", "content"}` dicts) is
the REPL's `context`, and the root prompt asks for the agent's next step. The RLM cannot run the
harness's tools, so its answer is all the task's model sees of the conversation, after the
harness's own instructions and context: that model, with the harness's tools, takes the step. Every
model call, the root's and the sub-calls', goes to the task's own model through the harness's
upstream and credentials.

`persistent` is the authors' multi-turn mode: a conversation's REPL stays across its requests, each
request adds only what is new as the next `context_k`, and the work on earlier ones stays in
`history_k` and in the variables the model kept, so a request reads what changed rather than the
whole conversation again. Without it (`rlm`), as the authors run a fresh query, every request
works over the whole conversation. The `local` environment runs the model's code in Relay's
process; `RELAY_RLM_ENVIRONMENT=docker` uses the authors' Docker environment instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..core.ir import LABELS, Context, Item, Kind, Request
from .base import Summarizer

PROMPT = """Work out the next step of the coding agent whose conversation is in `context`: a list of messages, oldest first (the user's requests, the agent's replies, its tool calls and their results). Submit through `answer` an instruction for the model that takes the step, which has the agent's instructions and tools but not this conversation, so make it self-contained: either exactly one next action (the tool to use and its arguments: the command to run, the file to read or change, with exact paths and content) or, if the task is done, the exact final reply to the user. Keep paths, identifiers, values and the user's constraints exactly as they are."""
PERSISTENT = """

`context` (`context_0`) holds the conversation when you were first asked; each later `context_k` holds only what was added since (the agent's step as it was taken, its result, any new message from the user). Your work on each is in `history_k`, and your variables are still defined."""
HANDOFF = """Result from the Recursive Language Model:
{answer}

Produce the next assistant response now. Follow the original instructions and available tool schemas. If the result calls for a tool, emit the corresponding native tool call. Do not mention the Recursive Language Model or this handoff."""
SESSIONS = 32  # persistent REPLs kept, the least recently used dropped first
ROLES = {"system": Kind.SYSTEM, "assistant": Kind.ASSISTANT}


@dataclass(frozen=True)
class RLM:
    persistent: bool = False
    max_depth: int = 1  # the authors' default: sub-calls are plain model calls
    max_iterations: int = 30
    environment: str = "local"
    name: str = "rlm"
    sessions: OrderedDict = field(default_factory=OrderedDict, compare=False, repr=False)  # persistent: by conversation

    @classmethod
    def from_env(cls) -> RLM:
        return cls(max_depth=int(os.getenv("RELAY_RLM_MAX_DEPTH", "1")),
                   max_iterations=int(os.getenv("RELAY_RLM_MAX_ITERATIONS", "30")),
                   environment=os.getenv("RELAY_RLM_ENVIRONMENT", "local"))

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "persistent": self.persistent, "max_depth": self.max_depth,
                "max_iterations": self.max_iterations, "environment": self.environment, "prompt": PROMPT,
                "handoff": HANDOFF}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        conversation = [{"role": LABELS.get(item.kind, item.kind.value), "content": item.text}
                        for item in request.history if item.text]
        runtime, context = self._session(request.conversation, conversation) if self.persistent else (self._new(), conversation)
        runtime.backend_kwargs = {"client": _TaskModel(summarizer)}  # this request's upstream, root and sub-calls alike
        answer = runtime.completion(context, root_prompt=PROMPT + (PERSISTENT if self.persistent else "")).response
        if not answer or not answer.strip():
            raise RuntimeError("the RLM gave no answer")
        return Context((Item(Kind.SUMMARY, HANDOFF.format(answer=answer.strip())),))

    def _new(self) -> Any:
        return _official().RLM(backend="relay", backend_kwargs={}, environment=self.environment, max_depth=self.max_depth,
                               max_iterations=self.max_iterations, persistent=self.persistent)

    def _session(self, conversation: str, items: list[dict[str, str]]) -> tuple[Any, list[dict[str, str]]]:
        """The conversation's REPL and what it has not seen; a new one where the history changed."""

        known = self.sessions.pop(conversation, None)
        if known and known[2] == _digest(items[: known[1]]):
            runtime, new = known[0], items[known[1]:]
        else:
            if known:
                known[0].close()
            runtime, new = self._new(), items
        self.sessions[conversation] = (runtime, len(items), _digest(items))
        while len(self.sessions) > SESSIONS:
            self.sessions.popitem(last=False)[1][0].close()
        return runtime, new


@dataclass(frozen=True)
class PersistentRLM(RLM):
    persistent: bool = True
    name: str = "rlm_persistent"


class _TaskModel:
    """The RLM's model client: the task's own model, through the harness's upstream."""

    model_name = "task"

    def __init__(self, lm: Summarizer) -> None:
        self.lm, self.calls = lm, 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        self.calls += 1
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self.lm.complete([Item(ROLES.get(m["role"], Kind.USER), str(m["content"])) for m in messages])

    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        return await asyncio.to_thread(self.completion, prompt, model)

    def get_last_usage(self) -> Any:
        from rlm.core.types import ModelUsageSummary

        return ModelUsageSummary(1, 0, 0)

    def get_usage_summary(self) -> Any:
        from rlm.core.types import ModelUsageSummary, UsageSummary

        return UsageSummary({self.model_name: ModelUsageSummary(self.calls, 0, 0)})


def _official() -> Any:
    """The authors' package, its clients joined by Relay's (backend "relay": the client given)."""

    try:
        import rlm.core.rlm as core
    except ImportError as exc:
        raise RuntimeError("the RLM strategies need the RLM authors' package: pip install 'relay[rlm]'") from exc
    if not getattr(core.get_client, "relay", False):
        official = core.get_client

        def get_client(backend: str, kwargs: dict[str, Any]) -> Any:
            return kwargs["client"] if backend == "relay" else official(backend, kwargs)

        get_client.relay = True  # type: ignore[attr-defined]
        core.get_client = get_client
    return core


def _digest(items: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest()
