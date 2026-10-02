"""Responses-shaped counting and summarization backed by the Gemini API."""
from __future__ import annotations

import json
import time
from random import uniform
from types import SimpleNamespace
from typing import Any

import httpx


class GeminiManagementUnavailable(RuntimeError):
    """A retryable Gemini management call remained unavailable after retries."""


class GeminiManagementResponses:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"x-goog-api-key": api_key},
            timeout=120.0,
            transport=transport,
        )
        self.input_tokens = self

    @staticmethod
    def _prompt(items: list[dict[str, Any]], instructions: str | None) -> str:
        prefix = f"Instructions: {instructions}\n\n" if instructions else ""
        return prefix + "Conversation items (JSON, in order):\n" + json.dumps(
            items, ensure_ascii=False, separators=(",", ":")
        )

    def _post(self, model: str, method: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(3):
            try:
                response = self.client.post(
                    f"v1beta/models/{model}:{method}", json=body
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in {408, 429, 500, 502, 503, 504}:
                    raise ValueError(
                        f"Gemini management {method} returned HTTP {status}"
                    ) from exc
                if attempt == 2:
                    raise GeminiManagementUnavailable(
                        f"Gemini management {method} remained unavailable (HTTP {status})"
                    ) from exc
            except httpx.RequestError as exc:
                if attempt == 2:
                    raise GeminiManagementUnavailable(
                        f"Gemini management {method} request failed after retries"
                    ) from exc
            time.sleep(2**attempt + uniform(0, 0.25))
        raise AssertionError("unreachable")

    def count(
        self,
        *,
        input: list[dict[str, Any]],
        model: str | None = None,
        instructions: str | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        prompt = self._prompt(input, instructions)
        data = self._post(
            model or self.model,
            "countTokens",
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        )
        return SimpleNamespace(input_tokens=int(data["totalTokens"]))

    def create(
        self,
        *,
        input: list[dict[str, Any]],
        model: str | None = None,
        instructions: str | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        prompt = self._prompt(input, instructions)
        data = self._post(
            model or self.model,
            "generateContent",
            {
                "systemInstruction": {"parts": [{"text":
                    "You summarize conversation history for a context checkpoint. "
                    "Follow the final compaction instruction in the supplied items. "
                    "Keep it under 80 words and return only the summary, without a preamble."
                }]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
            },
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini management model returned no candidate")
        candidate = candidates[0]
        if candidate.get("finishReason") != "STOP":
            raise RuntimeError(
                f"Gemini management model stopped with {candidate.get('finishReason')}"
            )
        parts = candidate.get("content", {}).get("parts") or []
        output = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part.get("text"), str) and not part.get("thought")
        ).strip()
        if not output:
            raise RuntimeError("Gemini management model returned no text")
        return SimpleNamespace(output_text=output)

    def close(self) -> None:
        self.client.close()
