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
    "git_attribution",  # extensions' sections follow the built-in ones
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
# How a message between agents begins (multi-agent v2; codex-rs/protocol/src/protocol.rs), and which
# of them Codex's compaction keeps (codex-rs/core/src/compact_remote_v2.rs).
AGENT_MESSAGE = re.compile(r"Message Type: (\w+)\nTask name: (\S+)\nSender: (\S+)")

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

# `<environment_context>` (codex-rs/core/src/context/world_state/environment.rs): an update lists
# only the environments that changed, then every turn-wide value as it is now.
ENV_OPEN, ENV_CLOSE = "<environment_context>", "</environment_context>"
ENV_VALUES = ("<cwd>", "<status>", "<error>", "<shell>")  # one environment's values
_ENVIRONMENT = re.compile(r'    <environment id="([^"]*)"(?: primary="(true|false)")?( status="unavailable" /)?>\n')
_REMOVED = re.compile(r'  <(?:shell_version|current_date) status="unavailable" />\n')
# Its `<subagents>` lists the live child threads, but a change to them alone renders no update:
# the multi-agent (v1) tools that spawn, close and resume them show the changes since.
_SUBAGENTS = re.compile(r"  <subagents>\n((?:    .*\n)*)  </subagents>\n")
AGENT_TOOLS = ("spawn_agent", "close_agent", "resume_agent")


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
        if item.kind is Kind.USER and (message := AGENT_MESSAGE.match(item.text)):
            # As Codex's compaction keeps them: an agent's assignment stays, while a finished agent's
            # answer and a sub-agent's progress report are conversation to summarize.
            kind, recipient, sender = message.groups()
            if kind == "FINAL_ANSWER" or (kind in ("MESSAGE", "CHANNEL_POST") and sender.startswith(f"{recipient}/")):
                return replace(item, kind=Kind.OTHER)
        return super().refine(item)

    def compacting(self, codec: Codec, items: list[WireItem]) -> bool:
        """Codex compacts on the server: its request ends with a `compaction_trigger` item, which
        the API requires to stay last (so no summary of Relay's may be made from it)."""

        return bool(items) and items[-1].get("type") == "compaction_trigger"

    def state(self, codec: Codec, items: list[WireItem]) -> State:
        """Re-render the initial context from the context updates in the history, like Codex:
        its per-request prefix stays first, and the rendering joins the context block; mid-turn,
        the updates written after the model's last response come after it."""

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
                *(replace(self.refine(codec.classify(items[index])), ref=index) for index in rendering.trailing),
            ),
        )

    def place(self, head: tuple[Item, ...], state: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        """Like Codex, except that mid-turn the updates newer than everything the head keeps (those
        that open the next step) follow the compacted history."""

        refs = [item.ref for item in head if item.ref is not None]
        newer = tuple(item for item in state if mid_turn and refs and item.ref is not None and item.ref > max(refs))
        return (*super().place(head, tuple(item for item in state if item not in newer), mid_turn), *newer)

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
    trailing: tuple[int, ...] = ()  # mid-turn, the updates that follow the compacted history


def section(text: str) -> str | None:
    """The world-state section a context fragment renders, if it is marked."""

    if text.startswith("# AGENTS.md instructions"):
        return "agents_md"
    match = _TAG.match(text)
    if match is None:
        return None
    name = match.group(1)  # e.g. "permissions instructions", or a tag with attributes
    return name.split()[0] if "=" in name else name


def _foldable(text: str) -> bool:
    key = section(text)
    return key in BUNDLE or key in SEPARATE or key in USER_CONTEXT or text.startswith(PREFIX_SAVED)


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
    # Mid-turn, Codex compacts with the state it built before the model's last response; the
    # updates written after that response open the next step, after the compacted history.
    settled = len(items) if pending else _last_response(items, last_agent)
    model_switch, trailing, environment = None, [], start
    for index in range(end, len(items)):
        if not is_context(items[index]):
            continue
        if index > settled and all(_foldable(text) for text in _parts(items[index]) or []):
            trailing.append(index)
            continue
        for text in _parts(items[index]) or []:
            key = section(text)
            if key == "environment_context":
                environment = index
            if key == "model_switch":
                # Rendered only at the start of the turn that switched models.
                model_switch = text if pending and index > last_agent else None
            elif key in BUNDLE or key in SEPARATE or key in USER_CONTEXT:
                _apply(render, key, text)
            elif text.startswith(PREFIX_SAVED):
                _approve(render, _prefix_list(text[len(PREFIX_SAVED):]) or [])
    if model_switch is not None:
        _apply(render, "model_switch", model_switch)
    _follow_subagents(render, items, environment, settled)
    context = tuple(message.wire() for message in render if message.parts)
    return Rendering(tuple(pinned), context, tuple(trailing))


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
            elif key == "environment_context":
                message.set(position, _environment_update(message.parts[position]["text"], text))
            else:
                message.set(position, text)
            return
    if not removed:
        _insert(render, key, text)


@dataclass
class _Environments:
    """An `<environment_context>`: each environment's (primary, values) by id, None for the single
    environment rendered without one (or None if unavailable), then the turn-wide values."""

    environments: dict[str | None, tuple[str | None, list[str]] | None]
    rest: str

    @classmethod
    def parse(cls, text: str) -> _Environments | None:
        if not (text.startswith(ENV_OPEN + "\n") and text.endswith(ENV_CLOSE)):
            return None
        lines = text[len(ENV_OPEN) + 1 : -len(ENV_CLOSE)].splitlines(keepends=True)
        environments: dict[str | None, tuple[str | None, list[str]] | None] = {}
        i = 0
        if lines[:1] == ["  <environments>\n"]:
            i = 1
            while i < len(lines) and lines[i] != "  </environments>\n":
                match = _ENVIRONMENT.fullmatch(lines[i])
                if match is None:
                    return None
                i += 1
                if match.group(3):
                    environments[match.group(1)] = None
                    continue
                values = []
                while i < len(lines) and lines[i].startswith("      "):
                    values.append(lines[i][6:])
                    i += 1
                if lines[i : i + 1] != ["    </environment>\n"]:
                    return None
                environments[match.group(1)] = (match.group(2), values)
                i += 1
            if i == len(lines):
                return None
            i += 1
        else:
            values = []
            while i < len(lines) and lines[i].startswith(tuple(f"  {tag}" for tag in ENV_VALUES)):
                values.append(lines[i][2:])
                i += 1
            if values:
                environments[None] = (None, values)
        return cls(environments, "".join(lines[i:]))

    def render(self) -> str | None:
        """Codex's full rendering: a single available environment without an id, else all of them."""

        listed = sorted(self.environments.items(), key=lambda entry: (entry[0] or "").encode())
        body = ""
        if len(listed) == 1 and not any(v.startswith("<status>") for v in listed[0][1][1]):
            body = "".join(f"  {value}" for value in listed[0][1][1])
        elif listed:
            body = "  <environments>\n"
            for id, (primary, values) in listed:
                if id is None or (len(listed) > 1 and primary is None):
                    return None
                attribute = f' primary="{primary}"' if len(listed) > 1 else ""
                body += f'    <environment id="{id}"{attribute}>\n'
                body += "".join(f"      {value}" for value in values) + "    </environment>\n"
            body += "  </environments>\n"
        return f"{ENV_OPEN}\n{body}{self.rest}{ENV_CLOSE}"


def _environment_update(text: str, update: str) -> str:
    """The environment context after an update: unchanged environments carried over, the changed
    ones and the turn-wide values (`<subagents>` too) taken from the update."""

    current, new = _Environments.parse(text), _Environments.parse(update)
    if current is None or new is None:
        return update
    if None in new.environments:
        environments = new.environments  # the single environment, rendered in full
    elif not new.environments:
        environments = current.environments
    else:  # a listing that replaces a single environment names every environment
        environments = {id: env for id, env in current.environments.items() if id is not None}
        environments.update(new.environments)
    available = {id: env for id, env in environments.items() if env is not None}
    rendered = _Environments(available, _REMOVED.sub("", new.rest)).render()
    return update if rendered is None else rendered


def _last_response(items: list[WireItem], last_agent: int) -> int:
    """Where the model's last response (its reasoning, messages and calls) begins."""

    def produced(item: WireItem) -> bool:
        return _is_agent(item) and not item.get("type", "").endswith("_output")

    start = next((i for i in range(last_agent, -1, -1) if produced(items[i])), 0)
    while start > 0 and produced(items[start - 1]):
        start -= 1
    return start


def _follow_subagents(render: list[_Message], items: list[WireItem], since: int, settled: int) -> None:
    """Apply the child threads spawned, closed and resumed after the last environment context by
    calls made before `settled`. A Codex process that resumes the session starts without the
    threads left open, which the history does not show."""

    message = next((m for m in render if "environment_context" in m.keys), None)
    if message is None:
        return
    position = message.keys.index("environment_context")
    text = message.parts[position]["text"]
    parsed = _Environments.parse(text)
    if parsed is None:
        return
    listed = _SUBAGENTS.search(parsed.rest)
    agents = [line[4:-1] for line in listed.group(1).splitlines(keepends=True)] if listed else []
    calls = {item.get("call_id"): item for item in items[:settled]
             if item.get("type") == "function_call" and item.get("name") in AGENT_TOOLS
             and item.get("namespace") in (None, "multi_agent_v1")}
    nicknames: dict[str, str | None] = {}
    changed = False
    for index, item in enumerate(items):
        call = calls.get(item.get("call_id")) if item.get("type") == "function_call_output" else None
        result, arguments = (_json(_output_text(item)), _json(call.get("arguments"))) if call else (None, None)
        if not (isinstance(result, dict) and isinstance(arguments, dict)):
            continue
        name = call["name"]
        agent = result.get("agent_id") if name == "spawn_agent" else arguments.get(
            "target" if name == "close_agent" else "id")
        if not isinstance(agent, str) or (name == "close_agent" and "previous_status" not in result) or (
                name == "resume_agent" and "status" not in result):
            continue
        if name == "spawn_agent":
            nicknames[agent] = result.get("nickname")
        if index < since:
            continue
        present = [line for line in agents if line == f"- {agent}" or line.startswith(f"- {agent}: ")]
        if name == "close_agent" and present:
            agents = [line for line in agents if line not in present]
            changed = True
        elif name != "close_agent" and not present:
            nickname = nicknames.get(agent)
            agents.append(f"- {agent}: {nickname}" if nickname else f"- {agent}")
            changed = True
    if not changed:
        return
    block = "".join(f"    {line}\n" for line in agents)
    block = f"  <subagents>\n{block}  </subagents>\n" if agents else ""
    rest = parsed.rest[: listed.start()] + block + parsed.rest[listed.end():] if listed else parsed.rest + block
    # Only the turn-wide values change: the environments stay as written (unavailable ones too).
    message.set(position, text[: len(text) - len(ENV_CLOSE) - len(parsed.rest)] + rest + ENV_CLOSE)


def _output_text(item: WireItem) -> str | None:
    output = item.get("output")
    if isinstance(output, list):
        texts = [part.get("text") for part in output if isinstance(part, dict)]
        return "".join(texts) if all(isinstance(t, str) for t in texts) else None
    return output if isinstance(output, str) else None


def _json(text: Any) -> Any:
    try:
        return json.loads(text) if isinstance(text, str) else None
    except json.JSONDecodeError:
        return None


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
