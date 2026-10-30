"""Claude Code's sessions: `<config>/projects/<its cwd, a name>/<id>.jsonl` (config:
CLAUDE_CONFIG_DIR, else ~/.claude), one JSON entry a line. The conversation is its user and
assistant entries, each linked to the one before by `parentUuid` (attachments, request records and
the like among them); an assistant message may be written in several entries, one per block, and
each tool result is an entry of its own, with what the tool reported (`toolUseResult`).

Its tools in the standard space: Bash (shell), Read (read), Edit, MultiEdit and Write (change).
Written, a session is what Claude Code writes of a conversation's messages (`--resume` takes it so,
without its other entries), with each result as the tool reports one.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.ir import Kind, Media
from ..harnesses.claude_code import ClaudeCode
from ..protocols import AnthropicMessages
from . import Call, Message, Session, Written, made, moved, note

PROFILE, CODEC = ClaudeCode(), AnthropicMessages()
VERSION = "2.1.309"  # the release whose sessions are written (resumed so by 2.1.309 and 2.1.312)
CURRENT = " (file state is current in your context — no need to Read it back)"
UPDATED = "The file {path} has been updated successfully." + CURRENT
TOOLS, HOW = {"Bash", "Read", "Edit", "Write"}, "commands as Bash, file reads as Read, file changes as Edit and Write"


class _ClaudeCode:
    def find(self, session: str) -> Path:
        path = Path(session).expanduser()
        if path.suffix == ".jsonl" and path.exists():
            return path
        found = sorted((home() / "projects").glob(f"*/{session}.jsonl"))
        if not found:
            raise ValueError(f"no Claude Code session {session} in {home() / 'projects'}")
        return found[0]

    def read(self, path: Path) -> Session:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_uuid = {row["uuid"]: row for row in rows if row.get("uuid")}
        said = [row for row in rows if row.get("type") in ("user", "assistant") and not row.get("isSidechain")]
        chain, row = [], said[-1] if said else None  # (the latest branch: from its last message back)
        while row is not None:
            chain.append(row)
            row = by_uuid.get(row.get("parentUuid"))
        turns: list[dict[str, Any]] = []
        calls: dict[str, dict[str, Any]] = {}  # tool_use id -> the call, its result once read
        for row in reversed(chain):
            if row.get("type") not in ("user", "assistant") or row.get("isMeta") or row.get("isSidechain"):
                continue
            message, time = row.get("message") or {}, row.get("timestamp", "")
            content = message.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [
                block for block in content or [] if isinstance(block, dict)]
            if row["type"] == "assistant":
                last = turns[-1] if turns else None
                if not (last and last["role"] == "assistant" and last["id"] and last["id"] == message.get("id")):
                    turns.append(last := {"role": "assistant", "id": message.get("id"), "text": [], "calls": [], "time": time})
                last["text"] += [block["text"] for block in blocks if block.get("type") == "text" and block.get("text")]
                for block in blocks:
                    if block.get("type") == "tool_use":
                        calls[block["id"]] = {"name": block.get("name", ""), "input": block.get("input") or {}}
                        last["calls"].append(calls[block["id"]])
                continue
            for block in blocks:
                if block.get("type") == "tool_result" and (call := calls.get(block.get("tool_use_id"))) is not None:
                    call["result"] = (block, row.get("toolUseResult"))
            typed = [block for block in blocks if block.get("type") != "tool_result"]
            if not typed:
                continue
            item = PROFILE.refine(CODEC.classify({"role": "user", "content": typed}))
            if item.kind is Kind.SUMMARY:  # (its own compaction's summary: what its model read)
                turns.append({"role": "user", "text": [item.text], "media": item.media, "time": time})
            elif item.kind is Kind.USER and (view := PROFILE.typed(item)) is not None:
                turns.append({"role": "user", "text": [view.text], "media": view.media, "time": time})
        messages = []
        for turn in turns:  # (a call no result answers, as when the session stopped there, goes)
            done = tuple(_standard(call["name"], call["input"], *call["result"]) for call in turn.get("calls", []) if "result" in call)
            text = "\n\n".join(t for t in turn["text"] if t.strip())
            if text or done or turn.get("media"):
                messages.append(Message(turn["role"], text, tuple(turn.get("media", ())), done, turn["time"]))
        first = said[0] if said else {}
        model = next((row["message"].get("model", "") for row in reversed(said) if row["type"] == "assistant"), "")
        return Session("claude_code", first.get("sessionId", path.stem), first.get("cwd", ""), tuple(messages), model,
                       first.get("timestamp", ""))

    def write(self, session: Session, cwd: str) -> Written:
        sid, folder = str(uuid.uuid4()), project(cwd)
        rows: list[dict[str, Any]] = []
        notes = 0

        def entry(kind: str, message: dict[str, Any], time: str, **more: Any) -> str:
            row = {"parentUuid": rows[-1]["uuid"] if rows else None, "isSidechain": False, "userType": "external",
                   "entrypoint": "cli", "cwd": cwd, "sessionId": sid, "version": VERSION, "type": kind,
                   "message": message, "uuid": str(uuid.uuid4()), "timestamp": time or _now(), **more}
            rows.append(row)
            return row["uuid"]

        if session.harness != "claude_code":  # (its own kind of note: the model reads it, its interface does not show it;
            # first, with the first message, as at the end it would have Claude Code answer it "No response requested.")
            entry("user", {"role": "user", "content": f"<system-reminder>\n{moved(session, 'claude_code', HOW, TOOLS)}\n"
                                                      "</system-reminder>"}, session.messages[0].time if session.messages else "",
                  isMeta=True)
        for message in session.messages:
            if message.role == "user":
                content = [*([{"type": "text", "text": message.text}] if message.text else []), *map(_image, message.media)]
                entry("user", {"role": "user", "content": message.text if not message.media else content}, message.time)
                continue
            uses, results, texts = [], [], [message.text] if message.text else []
            for call in message.calls:
                if (native := _native(call, cwd)) is None:
                    texts.append(note(call, session.harness))
                    notes += 1
                    continue
                for name, given, answer, error, reported in native:
                    if call.tool == "change" and session.harness != "claude_code" and isinstance(answer, str):
                        answer = f"{answer}\n{made(call, session.harness)}".lstrip("\n")
                    uses.append({"type": "tool_use", "id": _id("toolu_"), "name": name, "input": given})
                    results.append((uses[-1]["id"], answer, error, reported))
            content = [*({"type": "text", "text": text} for text in texts), *uses]
            if not content:
                continue
            # (the model, where it was its own: one it does not know it would try to restore, and say so)
            model = {"model": session.model} if session.harness == "claude_code" and session.model else {}
            asked = entry("assistant", {"id": _id("msg_"), "type": "message", "role": "assistant", **model, "content": content,
                                        "stop_reason": "tool_use" if uses else "end_turn", "stop_sequence": None,
                                        "usage": {"input_tokens": 0, "output_tokens": 0}}, message.time)
            for use, answer, error, reported in results:
                block = {"type": "tool_result", "tool_use_id": use, "content": answer, **({"is_error": True} if error else {})}
                entry("user", {"role": "user", "content": [block]}, message.time, toolUseResult=reported,
                      sourceToolAssistantUUID=asked)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{sid}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return Written(sid, path, f"cd {shlex.quote(cwd)} && claude --resume {sid}", notes)


def home() -> Path:
    return Path(os.getenv("CLAUDE_CONFIG_DIR") or "~/.claude").expanduser()


def project(cwd: str) -> Path:
    """Where Claude Code keeps the sessions started in `cwd`: its name, every character but a
    letter or a digit a hyphen, and past 200 characters cut, with a hash of the whole (its own)."""

    units = [int.from_bytes(pair, "little") for pair in _pairs(cwd.encode("utf-16-le"))]
    name = "".join(chr(unit) if chr(unit) in string.ascii_letters + string.digits else "-" for unit in units)
    if len(name) > 200:
        hashed = 0
        for unit in units:
            hashed = (hashed * 31 + unit) & 0xFFFFFFFF
        hashed = abs(hashed - (1 << 32) if hashed >= 1 << 31 else hashed)
        digits = ""
        while True:
            hashed, digit = divmod(hashed, 36)
            digits = (string.digits + string.ascii_lowercase)[digit] + digits
            if not hashed:
                break
        name = f"{name[:200]}-{digits}"
    return home() / "projects" / name


def _standard(name: str, given: dict[str, Any], block: dict[str, Any], reported: Any) -> Call:
    """One of its calls, in the standard space."""

    content = block.get("content")
    parts = [content] if isinstance(content, str) else [part for part in content or [] if isinstance(part, dict)]
    output = "\n".join(part if isinstance(part, str) else part.get("text", "") for part in parts
                       if isinstance(part, str) or part.get("type") == "text")
    media = tuple(m for part in parts if isinstance(part, dict) and (m := _media(part)) is not None)
    error = bool(block.get("is_error"))
    base = {"name": name, "input": given, "output": output, "error": error, "media": media}
    if name == "Bash":
        code = re.match(r"Exit code (-?\d+)\n?", output) if error else None
        return Call("shell", {"command": given.get("command", "")}, **{**base, "output": output[code.end():] if code else output},
                    exit=int(code.group(1)) if code else None if error else 0)
    if name == "Read":
        file = reported.get("file") if isinstance(reported, dict) and isinstance(reported.get("file"), dict) else {}
        text = file.get("content") if isinstance(file.get("content"), str) else "\n".join(
            re.sub(r"^\s*\d+\t", "", line, count=1) for line in output.split("\n"))
        args = {"path": given.get("file_path", ""), "start": given.get("offset") or 1, "lines": given.get("limit")}
        return Call("read", args, **{**base, "output": output if error or media else text})
    # (the lines it changed, as it reports them: what a harness that patches by line writes)
    patches = [patch for patch in reported.get("structuredPatch") or [] if isinstance(patch, dict)
               and isinstance(patch.get("lines"), list)] if isinstance(reported, dict) else []
    hunks = [patch["lines"] for patch in patches]
    lined = {"starts": [patch.get("oldStart") for patch in patches]} if hunks and all(p.get("oldStart") for p in patches) else {}
    if name in ("Edit", "MultiEdit"):
        edits = given.get("edits") if name == "MultiEdit" else [given]
        replaced = [(e.get("old_string", ""), e.get("new_string", ""), bool(e.get("replace_all"))) for e in edits or []]
        file = {"path": given.get("file_path", ""), "kind": "update", "edits": replaced, **({"hunks": hunks, **lined} if hunks else {})}
        return Call("change", {"files": [file]}, **base)
    if name == "Write":
        created = isinstance(reported, dict) and reported.get("type") == "create"
        file = {"path": given.get("file_path", ""), "kind": "add" if created else "update", "content": given.get("content", ""),
                **({"hunks": hunks, **lined} if hunks and not created else {})}
        return Call("change", {"files": [file]}, **base)
    return Call("", {}, **base)


def _native(call: Call, cwd: str) -> list[tuple[str, dict[str, Any], Any, bool, Any]] | None:
    """A call in the standard space as its tools: (tool, input, result content, is_error, what the
    tool reports) for each, None for no counterpart."""

    if call.tool == "shell":
        code = call.exit if call.exit is not None else 1 if call.error else 0
        if call.error or code:
            answer = f"Exit code {code}\n{call.output}"
            return [("Bash", {"command": call.args["command"]}, answer, True, f"Error: {answer}")]
        reported = {"stdout": call.output, "stderr": "", "interrupted": False, "isImage": False, "noOutputExpected": False}
        return [("Bash", {"command": call.args["command"]}, call.output, False, reported)]
    if call.tool == "read":
        path, start, lines = _absolute(call.args["path"], cwd), call.args.get("start") or 1, call.args.get("lines")
        given = {"file_path": path, **({"offset": start} if start != 1 else {}), **({"limit": lines} if lines else {})}
        if call.error:
            return [("Read", given, call.output, True, f"Error: {call.output}")]
        if call.media:
            image = call.media[0]
            return [("Read", given, [_image(m) for m in call.media], False,
                     {"type": "image", "file": {"base64": image.data, "type": image.mime}})]
        numbered = "\n".join(f"{start + n}\t{line}" for n, line in enumerate(call.output.split("\n")))
        count = call.output.count("\n") + 1
        reported = {"type": "text", "file": {"filePath": path, "content": call.output, "numLines": count,
                                             "startLine": start, "totalLines": start + count - 1}}
        return [("Read", given, numbered, False, reported)]
    if call.tool != "change":
        return None
    native: list[tuple[str, dict[str, Any], Any, bool, Any]] = []
    for file in call.args["files"]:
        path = _absolute(file["path"], cwd)
        if file["kind"] == "delete":
            native.append(("Bash", {"command": f"rm {shlex.quote(path)}"}, "", False, {"stdout": "", "stderr": ""}))
            continue
        if file.get("to"):
            to = _absolute(file["to"], cwd)
            native.append(("Bash", {"command": f"mv {shlex.quote(path)} {shlex.quote(to)}"}, "", False, {"stdout": "", "stderr": ""}))
            path = to
        if "content" in file:
            created = file["kind"] == "add"
            answer = f"File created successfully at: {path}{CURRENT}" if created else UPDATED.format(path=path)
            native.append(("Write", {"file_path": path, "content": file["content"]}, answer, False,
                           {"type": "create" if created else "update", "filePath": path, "content": file["content"],
                            "structuredPatch": [], "originalFile": None, "userModified": False}))
            continue
        hunks, starts = file.get("hunks") or [], file.get("starts") or []
        edits = file.get("edits") or [(_side(hunk, " -"), _side(hunk, " +"), False) for hunk in hunks]
        patches = _patches(hunks, starts)  # (what its interface shows the change by)
        for n, (old, new, everywhere) in enumerate(edits):
            if not old:  # (an insertion with no line around it to find it by: said as text)
                return None
            shown = patches if file.get("edits") else patches[n:n + 1]
            native.append(("Edit", {"file_path": path, "old_string": old, "new_string": new, "replace_all": everywhere},
                           UPDATED.format(path=path), False,
                           {"filePath": path, "oldString": old, "newString": new, "originalFile": None,
                            "structuredPatch": shown if n == 0 or not file.get("edits") else [], "userModified": False,
                            "replaceAll": everywhere}))
    if call.error and native:  # (it failed: the first of them says so, as the tool said it)
        name, given = native[0][:2]
        return [(name, given, call.output, True, f"Error: {call.output}")]
    return native or None


def _patches(hunks: list[list[str]], starts: list[int]) -> list[dict[str, Any]]:
    """Hunks as it reports a change (at line 1 where a hunk's start is not known)."""

    patches, moved = [], 0
    for n, hunk in enumerate(hunks):
        old, new = sum(line[:1] in " -" for line in hunk), sum(line[:1] in " +" for line in hunk)
        start = starts[n] if n < len(starts) else 1
        patches.append({"oldStart": start, "oldLines": old, "newStart": start + moved, "newLines": new, "lines": hunk})
        moved += new - old
    return patches


def _side(hunk: list[str], kept: str) -> str:
    """A hunk's text before (" -") or after (" +") it applies."""

    return "\n".join(line[1:] for line in hunk if line[:1] in kept)


def _absolute(path: str, cwd: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(cwd, path))


def _media(block: dict[str, Any]) -> Media | None:
    source = block.get("source") or {}
    if block.get("type") != "image" or not isinstance(source, dict):
        return None
    return Media("image", source.get("media_type", ""), source.get("data", ""), source.get("url", ""))


def _image(media: Media) -> dict[str, Any]:
    source = {"type": "url", "url": media.url} if media.url and not media.data else {
        "type": "base64", "media_type": media.mime or "image/png", "data": media.data}
    return {"type": "image", "source": source}


def _id(prefix: str) -> str:
    return prefix + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _pairs(data: bytes) -> list[bytes]:
    return [data[n:n + 2] for n in range(0, len(data), 2)]


HARNESS = _ClaudeCode()
