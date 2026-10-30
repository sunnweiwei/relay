"""A conversation moved from one harness to another (`relay transfer`), apart from context
management, which it neither uses nor changes.

A harness's own record of a conversation (its session file) is read into the standard space
below, and written from it the way the other harness records a conversation of its own, so the
other resumes it as if it had held it all along. Each harness maps its own record and its own
tools to and from the standard space (`relay/transfer/<harness>.py`), never to another harness.

What moves is the conversation as its model last saw it: what the user typed (and its images),
the model's text, and its tool calls with what they answered, in order, the calls of one response
together, from the harness's latest compaction on (its summary first). Each call is written as the
other harness's own tool, its result as that tool reports one. What does not move: what a harness
writes itself (its system prompt, instruction files, environment, reminders; the other writes its
own), the model's reasoning (encrypted or signed for the model that wrote it), and calls without a
counterpart, which become text saying what was run and what it answered.

Where the move is not exact, the model is told, so that what it reads does not mislead it: the
conversation starts with a note of the harness's own kind (Claude Code's meta reminder, Codex's
developer message: read by the model, not shown) saying where it came from, how its calls are
written here, and which of the tools it used there are not here (`moved`); a call written as
another kind of tool here (a patch as edits, an edit as a patch) says, in its result, which tool
made it there (`made`); a call without a counterpart says it was made there, with a tool not here
(`note`). A call written as the same kind of tool (a command, a file read, an image viewed) says
nothing more.

The standard tools (`Call.tool`, its `args`):

- `shell`: a command, `{command}`; the result is its output (stdout and stderr as the model read
  them) and `exit`, its exit code where known.
- `read`: a file read, `{path, start, lines}` (from line `start`, 1-based; `lines` of them, None:
  to the end); the result is the text read, without line numbers, or the file as `media`.
- `change`: files changed, `{files: [...]}`, each `{path, kind}` with `kind` "add" (`content`),
  "update" or "delete". An update has `hunks` (unified-diff hunks, each a list of lines starting
  with " ", "-" or "+"), `edits` ([(old, new, all)]: exact replacements) or `content` (the whole new
  text), as many of them as the harness knows (a harness writes the one its tools take), `starts`
  (the line each hunk starts at, where known), and `to` where the file moved; the result is what
  the tool said.
- "" (none): no counterpart; `name` and `input` say what was called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..core.ir import Media


@dataclass(frozen=True)
class Call:
    """A tool call and what it answered."""

    tool: str  # what it does in the standard space ("" for nothing there)
    args: dict[str, Any]  # its arguments there
    name: str  # the tool, as the harness named it
    input: Any  # and its input, as the harness gave it
    output: str = ""  # the result's text, as the model read it, without the harness's wrapping
    error: bool = False
    exit: int | None = None  # a command's exit code, where known
    media: tuple[Media, ...] = ()


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    text: str = ""
    media: tuple[Media, ...] = ()
    calls: tuple[Call, ...] = ()  # an assistant message's tool calls, each with what it answered
    time: str = ""  # ISO 8601, where known


@dataclass(frozen=True)
class Session:
    harness: str
    id: str
    cwd: str
    messages: tuple[Message, ...]
    model: str = ""
    time: str = ""  # when it started


@dataclass(frozen=True)
class Written:
    """A session written for a harness: its id, its file, and the command that resumes it."""

    id: str
    path: Path
    command: str
    notes: int = field(default=0)  # calls written as text, without a counterpart


class Harness(Protocol):
    def find(self, session: str) -> Path:
        """The harness's session file for `session` (an id, or a path)."""

    def read(self, path: Path) -> Session: ...

    def write(self, session: Session, cwd: str) -> Written:
        """`session` as a session of this harness's own, in `cwd`."""


def transfer(source: str, session: str, target: str, cwd: str | None = None) -> tuple[Session, Written]:
    """`session` of the harness `source` written as a session of `target` (in `cwd`, else its own)."""

    harnesses = HARNESSES
    if source not in harnesses or target not in harnesses:
        raise ValueError(f"transfer works between {', '.join(sorted(harnesses))}")
    read = harnesses[source].read(harnesses[source].find(session))
    return read, harnesses[target].write(read, cwd or read.cwd)


NAMES = {"claude_code": "Claude Code", "codex": "Codex"}


def note(call: Call, source: str) -> str:
    """A call without a counterpart, as text: where it was made, with what, and what it answered."""

    given = call.input if isinstance(call.input, str) else json.dumps(call.input, ensure_ascii=False)
    answer = call.output + (f" [{len(call.media)} image(s)]" if call.media else "")
    return (f"[Moved from {NAMES.get(source, source)}: there I called its tool `{call.name}`, which is not available here, "
            f"with: {given}\nIt answered{' (an error)' if call.error else ''}: {answer}]")


def made(call: Call, source: str) -> str:
    """What a call written as another kind of tool says of the tool that made it."""

    return f"(Moved from {NAMES.get(source, source)}: made there with its `{call.name}` tool.)"


def moved(session: Session, target: str, how: str, tools: set[str]) -> str:
    """The note a moved conversation starts with: where it came from, how its calls are written here
    (`how`), and the tools it used there that are not here (`tools`: the target's own)."""

    source, model = NAMES.get(session.harness, session.harness), f" (its model: {session.model})" if session.model else ""
    gone = sorted({call.name for message in session.messages for call in message.calls} - tools)
    return (f"The conversation that follows was moved here from {source}{model} with `relay transfer`. Its tool calls are written "
            f"as {NAMES.get(target, target)}'s own: {how}; calls with no tool like them here are described in text."
            + (f" Tools it used in {source} are not available here ({', '.join(f'`{name}`' for name in gone)}): use the "
               f"tools you have now." if gone else ""))


from . import claude_code, codex  # noqa: E402 (each maps its harness to and from the space above)

HARNESSES: dict[str, Harness] = {"claude_code": claude_code.HARNESS, "codex": codex.HARNESS}
