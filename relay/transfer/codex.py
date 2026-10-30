"""Codex's sessions: rollout files, `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<local time>-<id>.jsonl`
(CODEX_HOME, else ~/.codex), one JSON entry a line: `session_meta` first, then `response_item`s (the
conversation as its model reads it: Responses API items), `compacted` where Codex compacted it (the
history from there on), and events for its interface (`event_msg`).

Its tools in the standard space: exec_command and shell (shell; change where the command is
apply_patch, which Codex runs itself), apply_patch (change), view_image (read, the image), and in
its code mode `exec`, a script, where the script calls one of those once. Written,
a session is its meta and the conversation's items (`codex resume` takes it so), each result as
Codex reports one, and the items its interface shows a resumed conversation by.
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import shlex
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.ir import Kind, Media
from ..harnesses.codex import Codex
from ..prompts import SUMMARY_PREFIX
from ..protocols import OpenAIResponses
from . import Call, Message, Session, Written, made, moved, note

PROFILE, CODEC = Codex(), OpenAIResponses()
VERSION = "0.165.0"  # the release whose sessions are written (resumed so by 0.165)
HEREDOC = re.compile(r"\s*apply_patch\s*<<-?\s*['\"]?(\w+)['\"]?\n(.*)\n\1\s*", re.S)
CALLS = ("function_call", "custom_tool_call", "local_shell_call")
TOOLS = {"exec_command", "view_image"}
HOW = "commands, file reads (as cat or sed) and file changes (as apply_patch) as exec_command calls, images as view_image"


class _Codex:
    def find(self, session: str) -> Path:
        path = Path(session).expanduser()
        if path.suffix == ".jsonl" and path.exists():
            return path
        found = sorted(p for folder in ("sessions", "archived_sessions") for p in (home() / folder).rglob(f"rollout-*-{session}.jsonl"))
        if not found:
            raise ValueError(f"no Codex session {session} in {home() / 'sessions'}")
        return found[0]

    def read(self, path: Path) -> Session:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = next((row.get("payload") or {} for row in rows if row.get("type") == "session_meta"), {})
        history: list[tuple[dict[str, Any], str]] = []  # the items its model reads, and when each came
        for row in rows:
            payload = row.get("payload") or {}
            if row.get("type") == "response_item":
                history.append((payload, row.get("timestamp", "")))
            elif row.get("type") == "compacted":  # its compaction: what it kept, then its summary
                kept = payload.get("replacement_history")
                history = [(item, row.get("timestamp", "")) for item in kept] if isinstance(kept, list) else [
                    *((item, at) for item, at in history if _typed(item) is not None),
                    ({"type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": f"{SUMMARY_PREFIX}\n{payload.get('message', '')}"}]}, row.get("timestamp", ""))]
        turns: list[dict[str, Any]] = []
        calls: dict[str, dict[str, Any]] = {}  # call_id -> the call, its result once read
        for item, at in history:
            kind = item.get("type")
            if kind == "message" and item.get("role") == "assistant":
                text = "".join(part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict))
                turns.append({"role": "assistant", "text": [text], "calls": [], "time": at, "answered": False})
            elif kind == "message" and item.get("role") == "user":
                if (view := _typed(item)) is not None:
                    turns.append({"role": "user", "text": [view.text], "media": view.media, "time": at})
            elif kind in CALLS or kind == "web_search_call":
                last = turns[-1] if turns else None
                if not (last and last["role"] == "assistant" and not last["answered"]):  # (a response of its own)
                    turns.append(last := {"role": "assistant", "text": [], "calls": [], "time": at, "answered": False})
                call = {"item": item, "turn": last, "result": {} if kind == "web_search_call" else None}  # (no output item)
                last["calls"].append(call)
                calls[item.get("call_id") or ""] = call
            elif isinstance(kind, str) and kind.endswith("_call_output"):
                if (call := calls.get(item.get("call_id", ""))) is not None:
                    call["result"] = item
                    call["turn"]["answered"] = True
        messages = []
        for turn in turns:  # (a call no output answers, as when the session stopped there, goes)
            done = tuple(_standard(call["item"], call["result"]) for call in turn.get("calls", []) if call["result"] is not None)
            text = "\n\n".join(t for t in turn["text"] if t.strip())
            if text or done or turn.get("media"):
                messages.append(Message(turn["role"], text, tuple(turn.get("media", ())), done, turn["time"]))
        model = next((row["payload"].get("model", "") for row in reversed(rows) if row.get("type") == "turn_context"), "")
        return Session("codex", meta.get("id", ""), meta.get("cwd", ""), tuple(messages), model, meta.get("timestamp", ""))

    def write(self, session: Session, cwd: str) -> Written:
        sid, now = _uuid7(), datetime.now().astimezone()
        path = home() / "sessions" / now.strftime("%Y/%m/%d") / f"rollout-{now.strftime('%Y-%m-%dT%H-%M-%S')}-{sid}.jsonl"
        rows = [_row("session_meta", {"id": sid, "session_id": sid, "timestamp": _iso(), "cwd": cwd,
                                      "originator": "codex_cli_rs", "cli_version": VERSION, "source": "cli",
                                      "model_provider": _provider(), "history_mode": "paginated",
                                      "context_window": {"window_id": _uuid7()}})]
        if session.harness != "codex":  # (its own kind of note, first, as its own instructions are: the model reads it,
            # its interface does not show it)
            rows.append(_row("response_item", {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": moved(session, "codex", HOW, TOOLS)}]}))
        turn: dict[str, Any] = {}  # the turn its interface shows the messages in: from a user message to the next
        notes = 0

        def shown(item: dict[str, Any], at: str) -> None:
            if not turn:
                begin(at)
            ms = _ms(at)
            rows.append(_row("event_msg", {"type": "item_completed", "thread_id": sid, "turn_id": turn["id"], "item": item,
                                           "started_at_ms": ms, "completed_at_ms": ms}, at))

        def begin(at: str) -> None:
            turn.update(id=_uuid7(), at=at, said=None)
            rows.append(_row("event_msg", {"type": "task_started", "turn_id": turn["id"], "root_turn_id": turn["id"],
                                           "turn_attribution": {"turn_id": turn["id"], "turn_trigger": "user", "parent_turn_id": None,
                                                                "initiating_agent_path": None, "root_turn_id": turn["id"]},
                                           "started_at": _ms(at) // 1000, "model_context_window": None,
                                           "collaboration_mode_kind": "default"}, at))

        def end(at: str) -> None:
            if turn:
                rows.append(_row("event_msg", {"type": "task_complete", "turn_id": turn["id"], "root_turn_id": turn["id"],
                                               "last_agent_message": turn["said"], "started_at": _ms(turn["at"]) // 1000,
                                               "completed_at": _ms(at) // 1000, "duration_ms": max(0, _ms(at) - _ms(turn["at"]))}, at))
                turn.clear()

        for message in session.messages:
            at = message.time or _iso()
            if message.role == "user":
                end(at)
                begin(at)
                content = [*([{"type": "input_text", "text": message.text}] if message.text else []),
                           *({"type": "input_image", "image_url": _url(m)} for m in message.media)]
                rows.append(_row("response_item", {"type": "message", "role": "user", "content": content}, at))
                shown({"type": "UserMessage", "id": _uuid7(), "content": [{"type": "text", "text": message.text, "text_elements": []}]}, at)
                continue
            texts, items, outputs, views = [message.text] if message.text else [], [], [], []
            for call in message.calls:
                if (native := _native(call, cwd)) is None:
                    texts.append(note(call, session.harness))
                    notes += 1
                    continue
                for name, given, output, view in native:
                    if call.tool == "change" and session.harness != "codex" and isinstance(output, str):
                        output = f"{output.rstrip(chr(10))}\n{made(call, session.harness)}"
                    call_id = _id()
                    items.append({"type": "function_call", "name": name, "arguments": json.dumps(given, ensure_ascii=False),
                                  "call_id": call_id})
                    outputs.append({"type": "function_call_output", "call_id": call_id, "output": output})
                    views.append({**view, "id": call_id})
            if texts:
                text = "\n\n".join(texts)
                rows.append(_row("response_item", {"type": "message", "role": "assistant",
                                                   "content": [{"type": "output_text", "text": text}]}, at))
                shown({"type": "AgentMessage", "id": _id("msg_"), "content": [{"type": "Text", "text": text}]}, at)
                turn["said"] = text
            rows += [_row("response_item", item, at) for item in (*items, *outputs)]
            for view in views:
                shown(view, at)
        end(session.messages[-1].time if session.messages and session.messages[-1].time else _iso())
        for ordinal, row in enumerate(rows):
            row["ordinal"] = ordinal
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return Written(sid, path, f"cd {shlex.quote(cwd)} && codex resume {sid}", notes)


def home() -> Path:
    return Path(os.getenv("CODEX_HOME") or "~/.codex").expanduser()


def _typed(item: dict[str, Any]) -> Any:
    """A user message as its user typed it (its summary, for Codex's own), None for Codex's own context."""

    view = PROFILE.refine(CODEC.classify(item))
    if view.kind is Kind.SUMMARY:
        return view
    return PROFILE.typed(view) if view.kind is Kind.USER else None


def _standard(item: dict[str, Any], result: dict[str, Any]) -> Call:
    """One of its calls, in the standard space."""

    kind, name = item.get("type"), item.get("name") or item.get("type", "")
    if kind == "function_call":
        try:
            given: Any = json.loads(item.get("arguments") or "{}")
        except ValueError:
            given = item.get("arguments")
    else:
        given = item.get("input") if kind == "custom_tool_call" else item.get("action") or {}
    raw = result.get("output", "")
    parts = raw if isinstance(raw, list) else []
    text = raw if isinstance(raw, str) else "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "input_text")
    media = tuple(Media("image", *_data(p.get("image_url", ""))) for p in parts if isinstance(p, dict) and p.get("type") == "input_image")
    if name == "exec" and kind == "custom_tool_call" and isinstance(given, str):  # (its code mode: a script)
        ran = re.match(r"Script (completed|failed)[^\n]*\n(?:Wall time[^\n]*\n)?Output:\n", text)
        output, failed = (text[ran.end():], ran.group(1) == "failed") if ran else (text, False)
        base = {"name": name, "input": given, "output": output, "error": failed, "exit": None, "media": media}
        if (scripted := _scripted(given)) is None:
            return Call("", {}, **base)
        tool, value = scripted
        if tool == "apply_patch" or HEREDOC.fullmatch(value):
            return Call("change", {"files": _files(HEREDOC.fullmatch(value).group(2) if tool != "apply_patch" else value)}, **base)
        return Call("shell", {"command": value}, **base)
    output, code = _output(text)
    base = {"name": name, "input": given, "output": output, "error": bool(code), "exit": code, "media": media}
    command = given.get("cmd") if isinstance(given, dict) and isinstance(given.get("cmd"), str) else None
    if command is None and isinstance(given, dict) and isinstance(given.get("command"), list):  # shell, local_shell_call
        argv = [str(a) for a in given["command"]]
        command = argv[2] if argv[:2] in (["bash", "-lc"], ["/bin/bash", "-lc"], ["sh", "-c"]) and len(argv) > 2 else shlex.join(argv)
        if argv[:1] == ["apply_patch"] and len(argv) > 1:
            return Call("change", {"files": _files(argv[1])}, **base)
    if name == "apply_patch" and isinstance(given, str):
        return Call("change", {"files": _files(given)}, **base)
    if command is not None:
        patch = HEREDOC.fullmatch(command)
        if patch:
            return Call("change", {"files": _files(patch.group(2))}, **base)
        return Call("shell", {"command": command}, **base)
    if name == "view_image" and isinstance(given, dict):
        return Call("read", {"path": given.get("path", ""), "start": 1, "lines": None}, **base)
    return Call("", {}, **base)


def _native(call: Call, cwd: str) -> list[tuple[str, dict[str, Any], Any, dict[str, Any]]] | None:
    """A call in the standard space as its tools: (tool, arguments, output, the item its interface
    shows) for each, None for no counterpart."""

    if call.tool == "shell":
        code = call.exit if call.exit is not None else 1 if call.error else 0
        return [("exec_command", {"cmd": call.args["command"]}, _exec(call.output, code),
                 _command(call.args["command"], cwd, call.output, code))]
    if call.tool == "read":
        path = _relative(call.args["path"], cwd)
        if call.media and not call.error:
            images = [{"type": "input_image", "image_url": _url(m), "detail": "high"} for m in call.media]
            return [("view_image", {"path": path}, images, {"type": "ImageView", "path": f"file://{_absolute(path, cwd)}"})]
        start, lines = call.args.get("start") or 1, call.args.get("lines")
        command = f"cat {shlex.quote(path)}" if start == 1 and not lines else (
            f"sed -n '{start},{start + lines - 1 if lines else '$'}p' {shlex.quote(path)}")
        code = 1 if call.error else 0
        return [("exec_command", {"cmd": command}, _exec(call.output, code), _command(command, cwd, call.output, code, path))]
    if call.tool != "change":
        return None
    native, patch, changed, shown = [], [], {}, {}  # (shown: the files as its interface shows them changed)
    for file in call.args["files"]:
        path = _relative(file["path"], cwd)
        if file["kind"] == "update" and "content" in file and not file.get("hunks"):  # (written over, its lines unknown)
            command = f"cat > {shlex.quote(path)} <<'EOF'\n{file['content'].rstrip(chr(10))}\nEOF"
            native.append(("exec_command", {"cmd": command}, _exec("", 0), _command(command, cwd, "", 0)))
            continue
        if file["kind"] == "add":
            patch += [f"*** Add File: {path}", *(f"+{line}" for line in file.get("content", "").splitlines())]
            shown[_absolute(path, cwd)] = {"type": "add", "content": file.get("content", "")}
        elif file["kind"] == "delete":
            patch.append(f"*** Delete File: {path}")
            shown[_absolute(path, cwd)] = {"type": "delete", "content": ""}
        else:
            patch += [f"*** Update File: {path}", *([f"*** Move to: {_relative(file['to'], cwd)}"] if file.get("to") else [])]
            hunks = file.get("hunks") or [[*(f"-{line}" for line in old.split("\n")), *(f"+{line}" for line in new.split("\n"))]
                                          for old, new, _ in file.get("edits", [])]
            for hunk in hunks:
                patch += ["@@", *hunk]
            shown[_absolute(path, cwd)] = {"type": "update", "unified_diff": _diff(hunks, file.get("starts")),
                                           "move_path": _absolute(file["to"], cwd) if file.get("to") else None}
        changed[_relative(file.get("to") or file["path"], cwd)] = {"add": "A", "delete": "D"}.get(file["kind"], "M")
    if patch:
        body = "\n".join(["*** Begin Patch", *patch, "*** End Patch"])
        end = next(f"EOF{n or ''}" for n in range(100) if f"\nEOF{n or ''}\n" not in f"\n{body}\n")
        command = f"apply_patch <<'{end}'\n{body}\n{end}"
        said = call.output if call.error else "Success. Updated the following files:\n" + "".join(
            f"{mark} {path}\n" for path, mark in changed.items())
        output = (f"Wall time: 0.0000 seconds\nOutput:\nExit code: {1 if call.error else 0}\nWall time: 0 seconds\n"
                  f"Output:\n{said}")
        show = {"type": "FileChange", "changes": shown, "status": "failed" if call.error else "completed", "stdout": said, "stderr": ""}
        native.insert(0, ("exec_command", {"cmd": command}, output, show))
    return native or None


def _diff(hunks: list[list[str]], starts: list[int] | None) -> str:
    """Hunks as a unified diff, each with its header (at line 1 where its start is not known)."""

    diff, moved = [], 0
    for n, hunk in enumerate(hunks):
        old, new = sum(line[:1] in " -" for line in hunk), sum(line[:1] in " +" for line in hunk)
        start = starts[n] if starts and n < len(starts) else 1
        diff += [f"@@ -{start},{old} +{start + moved},{new} @@", *hunk]
        moved += new - old
    return "".join(f"{line}\n" for line in diff)


def _output(text: str) -> tuple[str, int | None]:
    """A command's output as Codex reports it, without its header (`Chunk ID: …`, `Wall time: …`,
    `Process exited with code N`, … then `Output:`; apply_patch's, nested), and its exit code."""

    code = None
    for _ in range(2):
        header = re.match(r"((?:[A-Z][\w ]*: [^\n]*\n|Process exited with code -?\d+\n)*)Output:\n", text)
        if not header:
            break
        text = text[header.end():]
        if found := re.search(r"(?:Process exited with code|Exit code:) (-?\d+)", header.group(1)):
            code = int(found.group(1))
            break
    if code is None and text.startswith("{"):  # (its older shell tool: JSON, with metadata)
        try:
            reported = json.loads(text)
            text, code = reported.get("output", text), (reported.get("metadata") or {}).get("exit_code")
        except (ValueError, AttributeError):
            pass
    return text, code


def _scripted(script: str) -> tuple[str, str] | None:
    """The one call a code-mode script makes of exec_command (its command) or apply_patch (its
    patch), None for any other script."""

    called = re.findall(r"\btools\.(\w+)\s*\(", script)
    if len(called) != 1 or called[0] not in ("exec_command", "apply_patch"):
        return None
    after = script[re.search(r"\btools\.\w+\s*\(", script).end():]
    if called[0] == "exec_command":
        found = re.match(r"\s*\{[^}]*?\bcmd\s*:\s*", after, re.S)
        after = after[found.end():] if found else ""
    literal = re.match(r'\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|`((?:[^`\\$]|\\.|\$(?!\{))*)`)', after, re.S)
    if not literal:
        return None
    double, single, template = literal.groups()
    if template is not None:  # (no ${...} in it: its text as written)
        return called[0], re.sub(r"\\([`\\$])", r"\1", template)
    quoted = double if double is not None else single.replace("\\'", "'").replace('"', '\\"')
    try:
        return called[0], json.loads(f'"{quoted}"')
    except ValueError:
        return None


def _files(patch: str) -> list[dict[str, Any]]:
    """The files an apply_patch patch changes, in the standard space."""

    files: list[dict[str, Any]] = []
    for line in patch.splitlines():
        if found := re.match(r"\*\*\* (Add|Update|Delete) File: (.+)", line):
            files.append({"path": found.group(2).strip(), "kind": found.group(1).lower()})
            files[-1].update({"add": {"content": ""}, "update": {"hunks": []}}.get(files[-1]["kind"], {}))
        elif not files or line.startswith(("*** Begin Patch", "*** End Patch", "*** End of File")):
            continue
        elif line.startswith("*** Move to: "):
            files[-1]["to"] = line[len("*** Move to: "):].strip()
        elif files[-1]["kind"] == "add":
            files[-1]["content"] += (line[1:] if line.startswith("+") else line) + "\n"
        elif files[-1]["kind"] == "update" and line.startswith("@@"):
            files[-1]["hunks"].append([])
        elif files[-1]["kind"] == "update" and line[:1] in (" ", "-", "+", ""):
            if not files[-1]["hunks"]:
                files[-1]["hunks"].append([])
            files[-1]["hunks"][-1].append(line or " ")
    return files


def _exec(output: str, code: int) -> str:
    return (f"Chunk ID: {secrets.token_hex(3)}\nWall time: 0.0000 seconds\nProcess exited with code {code}\n"
            f"Original token count: {max(1, len(output) // 4)}\nOutput:\n{output}")


def _command(command: str, cwd: str, output: str, code: int, read: str = "") -> dict[str, Any]:
    """A command as its interface shows one (`read`: the file it reads, where it does)."""

    parsed = {"type": "read", "cmd": command, "name": os.path.basename(read), "path": read} if read else {"type": "unknown", "cmd": command}
    return {"type": "CommandExecution", "process_id": None, "command": ["/bin/bash", "-lc", command], "cwd": f"file://{cwd}",
            "parsed_cmd": [parsed], "source": "unified_exec_startup", "status": "completed" if code == 0 else "failed",
            "aggregated_output": output, "exit_code": code, "duration": {"secs": 0, "nanos": 0}}


def _ms(at: str) -> int:
    try:
        return int(datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return int(time.time() * 1000)


def _row(kind: str, payload: dict[str, Any], at: str = "") -> dict[str, Any]:
    return {"timestamp": at or _iso(), "type": kind, "payload": payload}


def _provider() -> str:
    """The model provider Codex is set up with, which a resumed session goes on with."""

    try:
        return tomllib.loads((home() / "config.toml").read_text(encoding="utf-8")).get("model_provider") or "openai"
    except (OSError, ValueError):
        return "openai"


def _relative(path: str, cwd: str) -> str:
    absolute = _absolute(path, cwd)
    inside = os.path.commonpath([absolute, cwd]) == os.path.normpath(cwd) if os.path.isabs(cwd) else False
    return os.path.relpath(absolute, cwd) if inside else absolute


def _absolute(path: str, cwd: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(cwd, path))


def _url(media: Media) -> str:
    return media.url or f"data:{media.mime or 'image/png'};base64,{media.data}"


def _data(url: str) -> tuple[str, str, str]:
    found = re.match(r"data:([^;]+);base64,(.*)", url, re.S)
    return (found.group(1), found.group(2), "") if found else ("", "", url)


def _id(prefix: str = "call_") -> str:
    return prefix + secrets.token_urlsafe(18).replace("-", "").replace("_", "")[:24]


def _uuid7() -> str:
    """A time-ordered id, as Codex gives its sessions."""

    stamp, rand = int(time.time() * 1000), random.getrandbits(74)
    return str(uuid.UUID(int=(stamp << 80) | (0x7 << 76) | ((rand >> 62) << 64) | (0b10 << 62) | (rand & ((1 << 62) - 1))))


def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


HARNESS = _Codex()
