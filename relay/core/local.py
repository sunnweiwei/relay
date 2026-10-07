"""What a harness knows of its session that its requests do not show.

Where `relay install` can extend a harness (Claude Code: a plugin; Kimi Code: hooks), the
harness's side reports its session before each model request (`POST /relay/v1/local`): where it
keeps the whole transcript, its instruction files as they are now, the files it read most
recently, its plan, the skills it ran and its background tasks, leaving out what did not change
since its last report. A harness that compacts by itself re-reads these after its compaction; a
profile placing the harness's state after Relay's compaction can then do the same. Snapshots are
kept in memory by harness, session and agent, the latest one each.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class Document:
    path: str = ""  # a file's path, or a skill's name
    content: str = ""
    kind: str = ""  # what it is to the harness (an instruction file: "project", "user", "local", ...)


@dataclass(frozen=True)
class Task:
    id: str = ""
    kind: str = ""  # "agent", "shell"
    description: str = ""
    status: str = ""


@dataclass(frozen=True)
class Local:
    harness: str = ""
    session: str = ""
    agent: str | None = None  # a sub-agent's id; None for the main conversation
    opening: str = ""  # the conversation's first user message, by which its requests are found
    transcript: str | None = None  # where the harness keeps the whole conversation
    cwd: str | None = None
    instructions: tuple[Document, ...] = ()  # instruction files as they read now
    files: tuple[Document, ...] = ()  # read most recently, newest first, as the harness would re-read them
    plan: Document | None = None
    skills: tuple[Document, ...] = ()  # run in this session
    tasks: tuple[Task, ...] = ()  # running in the background
    extra: Mapping[str, Any] = field(default_factory=dict)  # anything else the harness's side reports

    @classmethod
    def from_json(cls, data: Mapping[str, Any], previous: Local | None = None) -> Local:
        """A report; what it leaves out (unchanged since its last one) is taken from `previous`."""

        if previous is not None:
            current = cls.from_json(data)
            fields = {name: getattr(current if name in data else previous, name)
                      for name in ("opening", "transcript", "cwd", "instructions", "files", "plan", "skills", "tasks")}
            return cls(current.harness, current.session, current.agent, **fields,
                       extra={**previous.extra, **current.extra})

        def documents(key: str) -> tuple[Document, ...]:
            return tuple(Document(str(d.get("path", "")), str(d.get("content", "")), str(d.get("kind", "")))
                         for d in data.get(key) or [] if isinstance(d, Mapping))

        plan = data.get("plan")
        known = {"harness", "session", "agent", "opening", "transcript", "cwd", "instructions", "files", "plan", "skills", "tasks"}
        return cls(
            harness=str(data.get("harness", "")),
            session=str(data.get("session", "")),
            agent=data.get("agent") or None,
            opening=str(data.get("opening") or ""),
            transcript=data.get("transcript") or None,
            cwd=data.get("cwd") or None,
            instructions=documents("instructions"),
            files=documents("files"),
            plan=Document(str(plan.get("path", "")), str(plan.get("content", ""))) if isinstance(plan, Mapping) else None,
            skills=documents("skills"),
            tasks=tuple(Task(str(t.get("id", "")), str(t.get("kind", "")), str(t.get("description", "")),
                             str(t.get("status", ""))) for t in data.get("tasks") or [] if isinstance(t, Mapping)),
            extra={k: v for k, v in data.items() if k not in known},
        )


class LocalStore:
    """The latest snapshot of each harness session and agent."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str, str | None], Local] = OrderedDict()
        self._lock = Lock()

    def update(self, data: Mapping[str, Any]) -> Local:
        """Keep a report, over the last one on the same conversation."""

        key = (str(data.get("harness", "")), str(data.get("session", "")), data.get("agent") or None)
        with self._lock:
            local = self._entries[key] = Local.from_json(data, self._entries.get(key))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return local

    def find(self, harness: str, session: str, opening: str) -> Local | None:
        """The report on the conversation a request opening so belongs to (a sub-agent's or the
        main one's: requests do not say which)."""

        with self._lock:
            found = [local for (h, s, _), local in self._entries.items() if (h, s) == (harness, session)]
        # The request's first message ends as the conversation's does (a harness may render its
        # own context ahead of it differently); the longest such opening is the closest match.
        opening = opening.strip()
        matching = [local for local in found if local.opening.strip() and opening.endswith(local.opening.strip()[-200:])]
        return max(matching, key=lambda local: len(local.opening.strip()), default=None) or next(
            (local for local in found if local.agent is None), None)
