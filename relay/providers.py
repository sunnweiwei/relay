"""Upstream endpoints and model facts.

Relay forwards each request in the protocol the harness spoke, so a provider is an
endpoint for that protocol: OpenAI, xAI and OpenRouter all serve the Responses API;
Anthropic, OpenRouter and many hosted models serve the Messages API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .install import mounts

# Not forwarded: hop-by-hop headers, and accept-encoding so that Relay always reads
# (and returns) an uncompressed body.
_DROPPED_HEADERS = {
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "transfer-encoding",
}

# (model-name prefix, context window in tokens); the first match wins. Vendor prefixes
# such as "openai/" or "anthropic/" (OpenRouter) are ignored.
CONTEXT_WINDOWS = (
    ("gpt-5", 272_000),
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("claude", 200_000),
    ("gemini", 1_048_576),
    ("grok-4", 256_000),
    ("grok-code", 256_000),
)


def context_window(model: object) -> int | None:
    if not isinstance(model, str):
        return None
    name = model.rsplit("/", 1)[-1].lower()
    return next((size for prefix, size in CONTEXT_WINDOWS if name.startswith(prefix)), None)


@dataclass(frozen=True)
class Upstream:
    """A provider endpoint: requests under the local `mount` path go to `base_url`."""

    base_url: str
    mount: str = ""
    api_key: str | None = None  # replaces the harness's own credentials when set
    key_header: str = "authorization"  # "authorization" (Bearer) or "x-api-key"

    def url(self, path: str, query: str = "") -> str:
        if self.mount and path.startswith(self.mount):
            path = path[len(self.mount) :]
        return f"{self.base_url.rstrip('/')}{path}" + (f"?{query}" if query else "")

    def headers(self, incoming: Mapping[str, str]) -> dict[str, str]:
        headers = {k: v for k, v in incoming.items() if k.lower() not in _DROPPED_HEADERS}
        if self.api_key:
            for name in ("authorization", "x-api-key"):
                headers.pop(name, None)
            value = self.api_key if self.key_header == "x-api-key" else f"Bearer {self.api_key}"
            headers[self.key_header] = value
        return headers


def route(path: str, headers: Mapping[str, str], upstreams: Mapping[str, Upstream]) -> Upstream:
    """The upstream for a request: an installed mount (`/up/<id>/...`), else by path."""

    if path.startswith("/up/"):
        mount = "/".join(path.split("/", 3)[:3])
        if origin := mounts().get(mount):
            return Upstream(origin, mount)
    if path.startswith("/backend-api/codex/"):
        return upstreams["chatgpt"]
    if path.startswith(("/v1/messages", "/api/")) or "anthropic-version" in headers:
        return upstreams["anthropic"]
    if path.startswith(("/v1beta/", "/v1alpha/")) or "x-goog-api-key" in headers:
        return upstreams["gemini"]
    return upstreams["openai"]


def upstreams_from_env() -> dict[str, Upstream]:
    anthropic_url = os.getenv("RELAY_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    return {
        "openai": Upstream(
            os.getenv("RELAY_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "/v1",
            os.getenv("RELAY_OPENAI_API_KEY") or None,
        ),
        "chatgpt": Upstream(  # Codex signed in with ChatGPT
            os.getenv("RELAY_CHATGPT_BASE_URL", "https://chatgpt.com/backend-api/codex"),
            "/backend-api/codex",
        ),
        "gemini": Upstream(os.getenv("RELAY_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")),
        "anthropic": Upstream(
            anthropic_url,
            "",
            os.getenv("RELAY_ANTHROPIC_API_KEY") or None,
            "x-api-key" if "api.anthropic.com" in anthropic_url else "authorization",
        ),
    }
