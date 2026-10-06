"""Context Language Models (Shao et al., 2026) as a Relay strategy: the model edits its own context.

Before every request the conversation the model will see is mirrored to a file, and the model
edits that file with its ordinary tools; on the next request the edited file becomes its context.
The model-facing text follows the paper's harness (facebookresearch/context-language-models): its
"Managing your context" system-prompt section, the size readout ending every request, nudges at
25%, 50% and 75% of the budget and an urgent one on every request past 90% of the limit (the budget
less a 2,048-token reserve), and its default edit gate, which refuses an edit that grows the context
past the limit. As there, the system prompt and the original task are protected: they stay out of
the file. The file's format follows pi-clm, the authors' adaptation of CLM to a coding harness,
because Relay's turns carry structure the paper's plain-text turns do not: a metadata line, then
one `[[CTX_TURN ...]]` block per turn, each bound to its turn by an id, so that untouched blocks
stay the original items (tool calls, reasoning and images intact). An edited user, assistant or
tool-result turn keeps its place with the new text (a result still answers its call; its images
stay); any other edited block, and every new one (`id=new-*`, any role label), becomes a user-role
note labelled with that role; removed blocks are gone; the document's order is followed. A tool
call kept without its result, or a result without its call, is told as a note when the request is
sent (Relay does that for every strategy, and only for those items). A file with no block headers
replaces everything after the task with one note. The harness's own context (instructions,
environment, reminders) stays out of the file and where the harness sent it. Notes and nudges say
they come from the context manager, and the urgent nudge asks for the task to go on after the
edit: a chat harness ends the turn on the first reply without a tool call, where the paper's loop
ran until the task was submitted. A steering document (`RELAY_CLM_STEERING`, the paper's in-context
instruction) adds a policy of the user's to the guidance; the nudges can be turned off, as in the
paper's steering runs (`RELAY_CLM_NUDGES=off`). Requests that offer no tools
(a title, a quota check) are left alone.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core.ir import LABELS, Context, Item, Kind, Request, note
from ..core.tokens import approx_tokens
from .base import Summarizer

# The paper's "Managing your context" section (clm_agent/prompts.yaml), with the file's block headers.
GUIDANCE = r"""## Managing your context

**Goal: maximize task success** — keep going until the task is done (no penalty for extra turns).
Your context budget is {budget}; each request shows your current size.

Your conversation is mirrored to `{path}` (refreshed before every request). **Free up context by
editing that file** — replace stale regions (big outputs, dead ends, superseded notes) with a
concise, specific summary. A bloated transcript wastes budget and dulls your reasoning, so compact
as it grows.

**Locate text with code — never paste or retype it** (context is long: retyping wastes tokens and
usually mis-matches). Blocks are numbered `[[CTX_TURN … index=1 …]]`, `[[CTX_TURN … index=2 …]]`, …
in order (your system prompt and the original task are hidden and protected for you, so the first
block you can edit is 1). Match a block by its header, a passage by a short unique first/last line,
or slice on the headers:
    python3 - <<'PY'
    import re; p="{path}"; s=open(p).read()
    # collapse block 7 (its header -> the next header); its text is never retyped:
    s=re.sub(r"(\[\[CTX_TURN [^\]]* index=7 [^\]]*\]\]).*?(?=\n\[\[CTX_TURN|\Z)",
             r"\1\n[grep done: parser.py:142 drops quoted commas; fix=csv.reader]", s, flags=re.S)
    open(p,"w").write(s)
    PY

**Rules:** leave the user's messages as they are: they say what the user wants; compact your own turns
and tool output. Don't `cat` this file (its text is already in your context). Keep its first line and the
`[[CTX_TURN …]]` header of any block you keep — emptying a block's text drops that block. To add a
block, copy a current header with a unique `id=new-NAME` and a role such as `notes`. Each request
says whether your last edit was applied. Those receipts, the size readout and CONTEXT BUDGET NUDGEs
come from the context manager, not the user: act on them, then go on with the user's request.

**Compact cheaply** — an edit forces everything *after* it to be re-read, so cost grows with how
much text FOLLOWS the edit:
- **Batch**: one large compaction beats many small edits.
- **Mind what's below your edit** — it all gets re-read, so don't compact a small early region while
  a long, still-useful tail sits beneath it (that re-reads the whole tail for little gain). Keep a
  useful tail; if it's much larger than what you'd compact, wait and compact head + tail together —
  UNLESS you expect to hit the context limit soon, then compact now.
- **Be generous in the summary**: the tail is re-read regardless, so a detailed replacement is
  essentially free."""
# The paper's budget nudges (utils/budget.py), for coding as well as search.
NOTE_CONTRACT = ("When you write a replacement note, COPY facts forward from the text you are replacing "
                 "(quote them): every RULED OUT candidate with its reason and the words 'do not retry'; the exact "
                 "queries/commands already tried; exact values marked VERIFIED or UNVERIFIED; and a NEXT line.")
NUDGES = {
    0.25: "No action needed. Before your next few steps, make sure your notes record which commands and files you "
          "already tried.",
    0.5: "Finish the unit of work in flight, then tidy ONCE. " + NOTE_CONTRACT,
    0.75: "close to the limit. Compact settled spans now — but do NOT wipe: edits keeping under 25% of the region they "
          "touch are usually followed by re-doing the deleted work. " + NOTE_CONTRACT,
}
URGENT = 0.9  # of the limit: nudged on every request past it
STEERING = "## Your context-management policy\n\nFollow this policy where it differs from the advice above:"
META = "[[LIVE_CONTEXT version=1 revision={revision} document={document}]]"
COMMENT = "# Edit block bodies, delete blocks or add new ones. Keep the first line and the header of every block you keep."
HEADER = "[[CTX_TURN document={document} index={index} role={role} id={id}]]"
HEADER_LINE = re.compile(r"^\[\[CTX_TURN ([^\]\n]*)\]\]\s*$", re.M)  # (escaped inside bodies)
ATTRIBUTE = re.compile(r"(document|index|role|id)=([\w-]+)")
LABEL = re.compile(r"\[context role=([A-Za-z][\w-]*)\]\n")  # how a note carries its role
STRUCTURAL = re.compile(r"^(\\*)(\[\[(?:CTX_TURN|LIVE_CONTEXT) )", re.M)  # escaped inside bodies
NO_TEXT = "[no text: encrypted or media content]"


@dataclass(frozen=True)
class ContextLanguageModel:
    budget: int | None = None  # tokens; None: the model's context window
    reserve: int = 2048  # generation headroom: the limit is the budget less this
    directory: str = "/tmp/.live_ctx"  # where the files live: the model's tools must reach it
    steering: str = ""  # a context-management policy in words, added to the guidance (the paper's s)
    nudges: bool = True  # the budget nudges; the size readout stays
    name: str = "clm"

    @classmethod
    def from_env(cls) -> ContextLanguageModel:
        budget, steering = os.getenv("RELAY_CLM_BUDGET"), os.getenv("RELAY_CLM_STEERING")
        return cls(budget=int(budget) if budget else None, reserve=int(os.getenv("RELAY_CLM_RESERVE", "2048")),
                   directory=os.getenv("RELAY_CLM_DIR", "/tmp/.live_ctx"),
                   steering=Path(steering).read_text(encoding="utf-8").strip() if steering else "",
                   nudges=os.getenv("RELAY_CLM_NUDGES", "on").lower() not in ("0", "off", "false"))

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "budget": self.budget, "reserve": self.reserve, "directory": self.directory,
                "guidance": GUIDANCE, "nudges": NUDGES if self.nudges else None, "steering": self.steering}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        state, path = request.state or {}, Path(self.directory) / f"{request.conversation}.md"
        budget = self.budget or request.window or 0
        limit = max(budget - self.reserve, 0)
        # The system prompt is not in the request; the original task is protected as well.
        task = next((n + 1 for n, item in enumerate(request.current) if item.kind is Kind.USER), 0)
        items, tokens, notes = list(request.current), request.tokens, []
        if state and path.exists() and _digest(edited := path.read_text(encoding="utf-8")) != state["digest"]:
            applied, tokens, note = _apply(edited, request, task, state, limit)
            items, notes = applied or items, [note]
        revision = state.get("revision", 0) + (items != list(request.current))
        text = _render(items[task:], _document(request.conversation, revision), revision)
        _write(path, text)
        nudged = state.get("nudged", [])
        if budget:
            nudged, nudge = self._nudge(tokens, budget, limit, nudged) if self.nudges else (nudged, None)
            notes += [nudge] if nudge else []
            over = f" — OVER; compact {path} now" if tokens > limit else ""
            notes.append(f"[context: ~{tokens}/{limit} tokens{over}]")
        else:
            notes.append(f"[context: ~{tokens} tokens]")
        guidance = GUIDANCE.format(path=path, budget=f"{budget} tokens" if budget else "not set")
        guidance = Item(Kind.SYSTEM, f"{guidance}\n\n{STEERING}\n\n{self.steering}" if self.steering else guidance)
        return Context((guidance, *items),
                       {"revision": revision, "digest": _digest(text), "ids": _ids(items[task:]), "nudged": nudged},
                       tuple(notes))

    def _nudge(self, tokens: int, budget: int, limit: int, nudged: list[float]) -> tuple[list[float], str | None]:
        """The paper's nudges: urgent on every request past 90% of the limit, else the highest tier
        newly crossed; a tier re-arms once the context drops below it."""

        nudged = [f for f in nudged if tokens >= int(budget * f)]
        if tokens >= int(URGENT * limit):
            return nudged, (f"CONTEXT BUDGET NUDGE (URGENT): You are at {tokens}/{limit} tokens — over "
                            f"{round(URGENT * 100)}% of your context limit. Compact your context THIS TURN, before anything "
                            f"else: remove stale regions now, then continue the task.")
        crossed = [f for f in NUDGES if tokens >= int(budget * f)]
        if not (fire := [f for f in crossed if f not in nudged]):
            return nudged, None
        top = max(fire)
        size = f"context is at ~{round(top * 100)}% of your {budget}-token budget ({tokens} tokens)"
        return sorted({*nudged, *crossed}), f"CONTEXT BUDGET NUDGE: {size}{' —' if top > 0.5 else '.'} {NUDGES[top]}"


def _apply(edited: str, request: Request, task: int, state: dict[str, Any],
           limit: int) -> tuple[list[Item] | None, int, str]:
    """The conversation with the edited file in place of what it was written from (the protected
    task kept in front), its size, and the model's receipt, worded as the paper's harness words it."""

    current, before = request.current, request.tokens
    snapshot, end = current[task: task + len(state["ids"])], task + len(state["ids"])
    if _ids(snapshot) != state["ids"] or end not in request.boundaries:
        return None, before, "[context file: edit NOT applied — the conversation changed after the file was written.]"
    document = _document(request.conversation, state["revision"])
    meta = META.format(revision=state["revision"], document=document)
    # A header is read by its attributes (any order; `index` is descriptive); one that cannot be
    # read is refused, never taken for text.
    blocks = []
    for header in HEADER_LINE.finditer(edited):
        attributes = dict(ATTRIBUTE.findall(header.group(1)))
        if "id" not in attributes or attributes.get("document", document) != document:
            return None, before, (f"[context file: edit NOT applied — this header cannot be read: {header.group(0)[:120]} "
                                  f"(a header needs an id, and document={document} if it names one).]")
        blocks.append((header, attributes))
    head: list[Item] = []
    if not blocks:  # plain text: everything after the task becomes one note
        body = _unescape("\n".join(line for line in edited.splitlines() if line not in (meta, COMMENT))).strip()
        if not body:
            return None, before, "[context file: edit NOT applied — the file is empty.]"
        head.append(note("notes", body))
    else:
        if edited.lstrip().startswith("[[LIVE_CONTEXT ") and not edited.lstrip().startswith(meta):  # (headers name the version too)
            return None, before, f"[context file: edit NOT applied — keep its first line exactly as it was: {meta}]"
        by_id = {id_: n for n, id_ in enumerate(state["ids"])}
        preamble = "\n".join(line for line in edited[: blocks[0][0].start()].splitlines() if line not in (meta, COMMENT))
        if preamble.strip():
            head.append(note("notes", _unescape(preamble).strip()))
        seen: set[str] = set()
        for n, (header, attributes) in enumerate(blocks):
            id_ = attributes["id"]
            body = _unescape(edited[header.end(): blocks[n + 1][0].start() if n + 1 < len(blocks) else len(edited)]).strip()
            if id_ in seen or not (id_ in by_id or id_.startswith("new-")):
                return None, before, f"[context file: edit NOT applied — block id {id_} is {'repeated' if id_ in seen else 'unknown'}.]"
            seen.add(id_)
            if not body:
                continue  # removed
            item = snapshot[by_id[id_]] if id_ in by_id else None
            role = attributes.get("role") or (_shown(item)[0] if item is not None else "notes")
            if item is not None and (role, body) == _shown(item):
                head.append(item)
            elif item is not None and role == _shown(item)[0]:
                head.append(replace(item, text=body))  # new text, same turn: a result still answers its call
            else:
                head.append(note(role, body))
    # As it will be sent: what the protocol cannot keep where it stands (a call whose result was
    # removed) is told as a note, the rest stays as written.
    items = list(request.sendable((*current[:task], *head, *current[end:])))
    after = before - _size(current) + _size(items)
    if limit and after > before and after > limit:  # the paper's default gate: an edit must fit
        return None, before, (f"[context file: edit REJECTED — it GREW context ~{before}->{after} tokens, so it was NOT "
                              f"applied (still ~{before}). An edit must FIT the {limit}-token limit: you likely "
                              f"duplicated/appended content — replace stale text with a SHORTER summary instead.]")
    if after > before:
        receipt = (f"[context file: edit applied but it GREW context ~{before}->{after} tokens (it fits, so it was kept). "
                   f"If you meant to condense, you likely duplicated content instead of replacing it.]")
    elif limit and after > limit:
        receipt = (f"[context file: edit applied — context ~{before}->{after} tokens, but STILL OVER the ~{limit}-token "
                   f"limit. Compact more NOW (delete stale turns/outputs).]")
    else:
        receipt = f"[context file: edit applied — context ~{before}->{after} tokens, {len(items) - len(current) + end} turns]"
    return items, after, receipt


def _render(items: list[Item], document: str, revision: int) -> str:
    blocks = [f"{HEADER.format(document=document, index=n, role=role, id=id_)}\n{_escape(body)}"
              for n, (item, id_) in enumerate(zip(items, _ids(items)), start=1) for role, body in [_shown(item)]]
    return "\n\n".join([META.format(revision=revision, document=document), COMMENT, *blocks]) + "\n"


def _shown(item: Item) -> tuple[str, str]:
    """The role label and text a block shows for an item."""

    if item.ref is None and item.kind is Kind.USER and (label := LABEL.match(item.text)):
        return label.group(1), item.text[label.end():].strip()
    return LABELS.get(item.kind, item.kind.value), item.text.strip() or NO_TEXT


def _ids(items: list[Item] | tuple[Item, ...]) -> list[str]:
    """Each block's id: its position and a digest of its item, stable while the item is."""

    return [f"{n}-{hashlib.sha256(f'{i.kind.value}{i.ref}{i.text}'.encode()).hexdigest()[:12]}"
            for n, i in enumerate(items, start=1)]


def _document(conversation: str, revision: int) -> str:
    return hashlib.sha256(f"{conversation}:{revision}".encode()).hexdigest()[:16]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _size(items: list[Item] | tuple[Item, ...]) -> int:
    return sum(approx_tokens(item.text) for item in items)


def _escape(body: str) -> str:
    return STRUCTURAL.sub(lambda m: f"\\{m.group(1)}{m.group(2)}", body)


def _unescape(body: str) -> str:
    return STRUCTURAL.sub(lambda m: f"{m.group(1)[1:]}{m.group(2)}", body)


def _write(path: Path, text: str) -> None:
    """Atomically, readable only by the user (it holds the conversation); never committed, should
    the directory be inside a repository (where a sandboxed harness's tools can reach it)."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not (ignore := path.parent / ".gitignore").exists():
        ignore.write_text("*\n", encoding="utf-8")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
