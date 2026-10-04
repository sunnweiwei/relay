"""OpenAI Codex CLI (Responses API).

Codex resends its history verbatim, so no canonicalization is needed. It injects
environment and instruction context as user messages; these markers mirror
`CONTEXTUAL_USER_FRAGMENT_MATCHERS` in codex-rs/core/src/context.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..core.ir import Item, Kind
from ..install import Setting
from ..prompts import SUMMARY_PREFIX
from ..protocols.base import Codec, WireItem
from .base import NEVER, Harness, State

CONTEXT_MARKERS = (
    "<environment_context>",
    "<user_instructions>",
    "# AGENTS.md instructions",
    "<user_shell_command>",
    "<turn_aborted>",
    "<subagent_notification>",
    "<agent_message_board_notification>",
    "<codex_internal_context",
    "<goal_context>",
)

# Sections in the order Codex renders them: the developer bundle (world-state sections in
# registration order, `<model_switch>` first and `<recommended_plugins>` last), separate
# developer messages, then the user context message.
BUNDLE = (
    "model_switch",
    "skills_instructions",
    "context_window_guidance",
    "realtime_conversation",
    "permissions instructions",
    "collaboration_mode",
    "persistent_mode",
    "environments_instructions",
    "apps_instructions",
    "plugins_instructions",
    "tools",
    "model_catalog",
    "recommended_plugins",
)
SEPARATE = ("multi_agent_role", "multi_agent_mode", "managed_developer_instructions")
USER_CONTEXT = ("agents_md", "environment_context")

# (replacement notice, removal notice) of the sections Codex replaces with a notice.
NOTICES = {
    "agents_md": (
        "These AGENTS.md instructions replace all previously provided AGENTS.md instructions.",
        "The previously provided AGENTS.md instructions no longer apply.",
    ),
    "persistent_mode": (
        "These persistent-mode instructions replace all previously provided persistent-mode instructions.",
        "The previously provided persistent-mode instructions no longer apply.",
    ),
    "context_window_guidance": (
        "This context-window guidance replaces all previously provided context-window guidance.",
        "The previously provided context-window guidance no longer applies.",
    ),
    "managed_developer_instructions": (
        "These managed developer instructions replace all previously provided managed developer instructions.",
        "The previously provided managed developer instructions no longer apply.",
    ),
}

_TAG = re.compile(r"\s*<([A-Za-z][^<>\n]*)>")

# Approved command prefixes inside `<permissions instructions>`, and the update Codex writes
# when the user approves new ones (codex-rs/protocol/src/models.rs `format_allow_prefixes`).
APPROVED_PREFIXES = "## Approved command prefixes\nThe following prefix rules have already been approved: "
PREFIX_SAVED = "Approved command prefix saved:\n"
MAX_RENDERED_PREFIXES = 100
MAX_PREFIX_TEXT_CHARS = 5000
PREFIXES_TRUNCATED = "...\n[Some commands were truncated]"
# What follows the approval text inside `<permissions instructions>`
# (codex-rs/prompts/src/permissions_instructions.rs).
AUTO_REVIEW = "\n\n`approvals_reviewer` is `auto_review`"
AFTER_APPROVAL = (" The writable root", "## Denied filesystem reads", "Additional permission paths/globs are omitted.",
                  "</permissions instructions>")


class Codex(Harness):
    name = "codex"

    def matches(self, headers: Mapping[str, str]) -> bool:
        # App-server clients name themselves in `originator`; every Codex request carries its turn metadata.
        return headers.get("originator", "").startswith("codex") or headers.get(
            "user-agent", ""
        ).startswith("codex") or "x-codex-turn-metadata" in headers

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and item.text.lstrip().startswith(CONTEXT_MARKERS):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def compacting(self, codec: Codec, items: list[WireItem]) -> bool:
        """Codex compacts on the server: its request ends with a `compaction_trigger` item, which
        the API requires to stay last (so no summary of Relay's may be made from it)."""

        return bool(items) and items[-1].get("type") == "compaction_trigger"

    def state(self, codec: Codec, items: list[WireItem]) -> State:
        """Re-render the initial context from the context updates in the history, like Codex:
        its per-request prefix stays first, the rendering joins the context block, and the
        updates folded into it are superseded."""

        rendering = rebuild(items) if codec.name == "openai_responses" else None
        if rendering is None:
            return super().state(codec, items)

        def kept(index: int, kind: Kind) -> Item:
            return replace(self.refine(codec.classify(items[index])), ref=index, kind=kind)

        return State(
            (
                *(kept(index, Kind.SYSTEM) for index in rendering.pinned),
                *(kept(part, Kind.CONTEXT) if isinstance(part, int)
                  else Item(Kind.CONTEXT, codec.classify(part).text, wire=json.dumps(part, ensure_ascii=False))
                  for part in rendering.context),
            ),
            rendering.updates,
        )

    def settings(self) -> list[Setting]:
        """Re-point the built-in OpenAI provider, so Codex keeps its own login and its
        ChatGPT-backend features (the wrapped URL keeps the `/backend-api/codex` path)."""

        home = Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser()
        auth = home / "auth.json"
        chatgpt = auth.exists() and json.loads(auth.read_text()).get("auth_mode") == "chatgpt"
        default = "https://chatgpt.com/backend-api/codex" if chatgpt else "https://api.openai.com/v1"
        # Codex compacts at its limit, capped at 90% of the context window it assumes: both out of
        # reach (its context-left meter then reads near 100%).
        return [Setting(home / "config.toml", ("openai_base_url",), endpoint=default),
                Setting(home / "config.toml", ("model_auto_compact_token_limit",), NEVER),
                Setting(home / "config.toml", ("model_context_window",), NEVER)]

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        provider = (
            'model_providers.relay={name="Relay",'
            f'base_url="{relay_url}/v1",env_key="OPENAI_API_KEY",'
            'wire_api="responses",supports_websockets=false}'
        )
        return ["codex", "-c", 'model_provider="relay"', "-c", provider, *args], {}


@dataclass(frozen=True)
class Rendering:
    """Codex's initial context for a compaction: request items by index, or new items."""

    pinned: tuple[int, ...]  # per-request prefix (`additional_tools`, base instructions)
    context: tuple[int | WireItem, ...]  # the re-rendered initial context
    updates: frozenset[int]  # the context updates in the history it folds in


def section(text: str) -> str | None:
    """The world-state section a context fragment renders, if it is marked."""

    if text.startswith("# AGENTS.md instructions"):
        return "agents_md"
    match = _TAG.match(text)
    if match is None:
        return None
    name = match.group(1)  # e.g. "permissions instructions", or a tag with attributes
    return name.split()[0] if "=" in name else name


def is_context(item: WireItem) -> bool:
    """A message Codex wrote to give the model context rather than conversation."""

    parts = _parts(item)
    if parts is None:
        return False
    role = item.get("role")
    return role in {"developer", "system"} or (
        role == "user" and bool(parts) and parts[0].lstrip().startswith(CONTEXT_MARKERS)
    )


def rebuild(items: list[WireItem]) -> Rendering | None:
    """Codex's initial context for a compaction of `items`, or None if it cannot be found."""

    pinned = _prefix(items)
    start = end = len(pinned)
    while end < len(items) and is_context(items[end]):
        end += 1
    agent = [i for i, item in enumerate(items) if _is_agent(item)]
    if end == start or not agent:
        return None
    render = [_Message.from_item(i, items[i]) for i in range(start, end)]

    last_agent = agent[-1]
    pending = any(_is_user_turn(item) for item in items[last_agent + 1 :])
    model_switch, updates = None, set()
    for index in range(end, len(items)):
        if not is_context(items[index]):
            continue
        folded = True  # every part is a world-state section (not an event such as <turn_aborted>)
        for text in _parts(items[index]) or []:
            key = section(text)
            if key == "model_switch":
                # Rendered only at the start of the turn that switched models.
                model_switch = text if pending and index > last_agent else None
            elif key in BUNDLE or key in SEPARATE or key in USER_CONTEXT:
                _apply(render, key, text)
            elif text.startswith(PREFIX_SAVED):
                _approve(render, _prefix_list(text[len(PREFIX_SAVED):]) or [])
            else:
                folded = False
        if folded:
            updates.add(index)
    if model_switch is not None:
        _apply(render, "model_switch", model_switch)
    context = tuple(message.wire() for message in render if message.parts)
    return Rendering(tuple(pinned), context, frozenset(updates))


@dataclass
class _Message:
    role: str
    parts: list[dict[str, Any]]
    ref: int | None = None  # the request item it still equals, if unchanged
    keys: list[str | None] = field(default_factory=list)

    @classmethod
    def from_item(cls, index: int, item: WireItem) -> _Message:
        content = item["content"]
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else list(content)
        return cls(item["role"], parts, index, [section(p["text"]) for p in parts])

    def set(self, position: int, text: str) -> None:
        if self.parts[position]["text"] != text:
            self.parts[position] = {**self.parts[position], "text": text}
            self.ref = None

    def insert(self, position: int, key: str, text: str) -> None:
        self.parts.insert(position, {"type": "input_text", "text": text})
        self.keys.insert(position, key)
        self.ref = None

    def remove(self, position: int) -> None:
        del self.parts[position], self.keys[position]
        self.ref = None

    def wire(self) -> int | WireItem:
        if self.ref is not None:
            return self.ref
        return {"type": "message", "role": self.role, "content": self.parts}


def _apply(render: list[_Message], key: str, text: str) -> None:
    replacement, removal = NOTICES.get(key, (None, None))
    removed = removal is not None and removal in text
    if replacement is not None:
        text = text.replace(f"{replacement}\n\n", "", 1)
    for message in render:
        if key in message.keys:
            position = message.keys.index(key)
            if removed:
                message.remove(position)
            else:
                message.set(position, text)
            return
    if not removed:
        _insert(render, key, text)


def _approve(render: list[_Message], added: list[list[str]]) -> None:
    """Merge newly approved command prefixes into the permissions section's prefix list."""

    for message in render:
        if "permissions instructions" not in message.keys:
            continue
        position = message.keys.index("permissions instructions")
        text = message.parts[position]["text"]
        start = text.find(APPROVED_PREFIXES)
        if start < 0:
            message.set(position, _first_prefixes(text, added))
            return
        start += len(APPROVED_PREFIXES)
        lines = text[start:].split("\n")
        count = next((i for i, line in enumerate(lines) if not line.startswith("- [")), len(lines))
        current = _prefix_list("\n".join(lines[:count]))
        if current is None or lines[count:count + 2] == PREFIXES_TRUNCATED.split("\n"):
            return  # a truncated list cannot be re-rendered with the prefixes it left out
        listed = "\n".join(lines[:count])
        merged = _format_prefixes({tuple(p) for p in current} | {tuple(p) for p in added})
        message.set(position, text[:start] + merged + text[start + len(listed):])
        return


def _first_prefixes(text: str, added: list[list[str]]) -> str:
    """Add the approved prefix list to a permissions text that had none, where Codex puts it:
    last in the approval text, which only on-request and granular policies give a list."""

    if "prefix_rule" not in text and "Approval policy is `granular`" not in text:
        return text
    listed = APPROVED_PREFIXES + _format_prefixes({tuple(p) for p in added})
    review = text.find(AUTO_REVIEW)
    if review >= 0:
        return f"{text[:review]}\n\n{listed}{text[review:]}"
    end = min((i for i in (text.find(marker) for marker in AFTER_APPROVAL) if i >= 0), default=-1)
    if end <= 0 or text[end - 1] != "\n":
        return text
    return f"{text[:end]}\n\n{listed}\n{text[end:]}"


def _prefix_list(text: str) -> list[list[str]] | None:
    """Prefixes rendered one per line as `- ["git", "status"]`."""

    prefixes = []
    for line in text.split("\n"):
        try:
            prefix = json.loads(line[2:]) if line.startswith("- [") else None
        except json.JSONDecodeError:
            prefix = None
        if not isinstance(prefix, list) or not all(isinstance(token, str) for token in prefix):
            return None
        prefixes.append(prefix)
    return prefixes


def _format_prefixes(prefixes: set[tuple[str, ...]]) -> str:
    """Codex's `format_allow_prefixes`: shortest first, at most 100 entries and 5000 chars."""

    ordered = sorted(prefixes, key=lambda p: (len(p), sum(len(t.encode()) for t in p), [t.encode() for t in p]))
    text = "\n".join(
        "- [" + ", ".join(json.dumps(token, ensure_ascii=False) for token in prefix) + "]"
        for prefix in ordered[:MAX_RENDERED_PREFIXES]
    )
    truncated = len(ordered) > MAX_RENDERED_PREFIXES or len(text) > MAX_PREFIX_TEXT_CHARS
    return text[:MAX_PREFIX_TEXT_CHARS] + PREFIXES_TRUNCATED if truncated else text


def _insert(render: list[_Message], key: str, text: str) -> None:
    """Add a section the window's first rendering did not have, where Codex renders it."""

    if key in BUNDLE:
        bundle = next((m for m in render if m.role == "developer" and set(m.keys) & set(BUNDLE)), None)
        if bundle is None:
            bundle = _Message("developer", [])
            render.insert(0, bundle)
        rank = BUNDLE.index(key)
        position = next(
            (i for i, k in enumerate(bundle.keys) if k in BUNDLE and BUNDLE.index(k) > rank),
            len(bundle.keys),
        )
        bundle.insert(position, key, text)
    elif key in SEPARATE:
        rank = SEPARATE.index(key)
        position = next(
            (i for i, m in enumerate(render)
             if m.role == "user" or any(k in SEPARATE and SEPARATE.index(k) > rank for k in m.keys)),
            len(render),
        )
        message = _Message("developer", [])
        message.insert(0, key, text)
        render.insert(position, message)
    else:
        user = next((m for m in render if m.role == "user" and set(m.keys) & set(USER_CONTEXT)), None)
        if user is None:
            user = _Message("user", [])
            render.append(user)
        position = 0 if key == "agents_md" else len(user.keys)
        user.insert(position, key, text)


def _prefix(items: list[WireItem]) -> list[int]:
    """Codex's per-request prefix: `additional_tools` and the base instructions after it."""

    if not items or items[0].get("type") != "additional_tools":
        return []
    parts = _parts(items[1]) if len(items) > 1 else None
    if items[1:2] and items[1].get("role") == "developer" and parts and len(parts) == 1 and section(parts[0]) is None:
        return [0, 1]
    return [0]


def _parts(item: WireItem) -> list[str] | None:
    """The texts of a plain text message, or None for anything else."""

    if item.get("type", "message") != "message":
        return None
    content = item.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return None
    texts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "input_text"]
    return texts if len(texts) == len(content) and all(isinstance(t, str) for t in texts) else None


def _is_agent(item: WireItem) -> bool:
    kind = item.get("type", "message")
    return (kind == "message" and item.get("role") == "assistant") or kind == "reasoning" or kind.endswith(
        ("_call", "_output")
    )


def _is_user_turn(item: WireItem) -> bool:
    parts = _parts(item)
    return (
        item.get("type", "message") == "message"
        and item.get("role") == "user"
        and not is_context(item)
        and not (parts and parts[0].startswith(SUMMARY_PREFIX))
    )
