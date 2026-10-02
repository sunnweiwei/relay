"""OpenAI Responses API (`POST /v1/responses`), as spoken by Codex and the OpenAI SDK."""

from __future__ import annotations

import json
from typing import Any

from ..core.ir import Item, Kind
from .base import OVERFLOW_PHRASES, Body, WireItem, canonical_json, error_message

_ROLES = {
    "system": Kind.SYSTEM,
    "developer": Kind.SYSTEM,
    "user": Kind.USER,
    "assistant": Kind.ASSISTANT,
}
_TEXT_PARTS = {"input_text", "output_text", "text", "summary_text", "reasoning_text"}


class OpenAIResponses:
    name = "openai_responses"
    paths = ("/responses",)

    def managed(self, body: Body) -> bool:
        return (
            isinstance(body.get("input"), list)
            and body.get("previous_response_id") is None
            and body.get("conversation") is None
        )

    def items(self, body: Body) -> list[WireItem]:
        return list(body.get("input") or [])

    def with_items(self, body: Body, items: list[WireItem]) -> Body:
        return {**body, "input": items}

    def classify(self, item: WireItem) -> Item:
        kind = item.get("type", "message")
        if kind != "message" and item.get("role") in {"system", "developer"}:
            return Item(Kind.SYSTEM, _string(item))  # e.g. Codex's `additional_tools` declarations
        if kind == "message":
            text, media = _content_text(item.get("content"))
            return Item(_ROLES.get(item.get("role"), Kind.OTHER), text, media=media)
        if kind == "reasoning":
            return Item(Kind.REASONING, _content_text(item.get("summary"))[0])
        if kind == "compaction":  # opaque server-side compaction
            return Item(Kind.SUMMARY)
        if kind.endswith("_output"):
            return Item(Kind.TOOL_RESULT, _output_text(item.get("output")))
        if kind.endswith("_call"):
            fields = (item.get(key) for key in ("name", "arguments", "input", "action"))
            return Item(Kind.TOOL_CALL, " ".join(_string(value) for value in fields if value))
        return Item(Kind.OTHER, _string(item))

    def canonical(self, item: WireItem) -> bytes:
        """A reasoning item is its encrypted content: resumed sessions drop its display summary
        (OpenClaw) or write absent content back as null or [] (Codex `exec resume`)."""

        if item.get("type") == "reasoning":
            item = {k: v for k, v in item.items() if k != "summary" and not (k == "content" and not v)}
        return canonical_json(item)

    def boundaries(self, items: list[WireItem]) -> frozenset[int]:
        """Never separate a call from its output, or reasoning (and the commentary
        that may follow it) from the item it precedes."""

        legal, pending = {0}, set()
        for index, item in enumerate(items):
            kind, call_id = item.get("type", "message"), item.get("call_id")
            if call_id and kind.endswith("_output"):
                pending.discard(call_id)
            elif call_id and kind.endswith("_call"):
                pending.add(call_id)
            if not pending and kind != "reasoning" and item.get("phase") != "commentary":
                legal.add(index + 1)
        return frozenset(legal)

    def arrange(self, head: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        return head

    def user_message(self, text: str) -> WireItem:
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

    def preamble(self, body: Body) -> str:
        return _string(body.get("instructions") or "") + _string(body.get("tools") or "")

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        request = {key: value for key, value in body.items() if key != "background"}
        request.update(input=[*items, self.user_message(prompt)], store=False)
        if request.get("tools") or any(item.get("type") == "additional_tools" for item in items):
            request["tool_choice"] = "none"
        if isinstance(request.get("text"), dict):  # no structured-output schema for the summary
            request["text"] = {k: v for k, v in request["text"].items() if k != "format"}
        return request

    def stream_result(self, events: list[Body]) -> Body:
        final = next(
            (e.get("response") for e in reversed(events) if e.get("type", "").startswith("response.")
             and isinstance(e.get("response"), dict) and e["response"].get("status") != "in_progress"),
            None,
        ) or {}
        done = [e["item"] for e in events if e.get("type") == "response.output_item.done"]
        return {**final, "output": final.get("output") or done}

    def output_text(self, payload: Body) -> str:
        parts = [
            _content_text(item.get("content"))[0]
            for item in payload.get("output") or []
            if item.get("type") == "message" and item.get("role") == "assistant"
        ]
        return "\n".join(part for part in parts if part).strip() or str(
            payload.get("output_text") or ""
        ).strip()

    def finished(self, payload: Body) -> bool:
        return payload.get("status") == "completed"

    def usage(self, payload: Body) -> int | None:
        if payload.get("type") in {"response.completed", "response.incomplete"}:
            payload = payload.get("response") or {}
        usage = payload.get("usage")
        tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        return tokens if isinstance(tokens, int) else None

    def is_overflow(self, status: int, payload: Any) -> bool:
        code, message = error_message(payload)
        return status == 400 and (
            code == "context_length_exceeded" or any(p in message for p in OVERFLOW_PHRASES)
        )


def _content_text(content: Any) -> tuple[str, bool]:
    if isinstance(content, str):
        return content, False
    texts, media = [], False
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") in _TEXT_PARTS and isinstance(part.get("text"), str):
            texts.append(part["text"])
        else:
            media = True
    return "\n".join(texts), media


def _output_text(output: Any) -> str:
    return output if isinstance(output, str) else _content_text(output)[0]


def _string(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
