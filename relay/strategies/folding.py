"""Context Folding (FoldAgent, Sun et al.) as a Relay strategy: the agent branches off a sub-task and
returns from it with a message, and the branch's steps fold away.

As in FoldAgent (github.com/sunnweiwei/FoldAgent), a branch inherits the whole conversation and is
told it is a branch ("ROLE CHANGE: `MODE: BRANCH`" with its task); its steps follow; when it
returns, the main conversation keeps only the branch call, answered by "Branch has finished its
task, the returned message is: ...". The harness's own tools stand in for FoldAgent's `branch` and
`return`: the agent runs `echo "[branch] <description> :: <task>"` and `echo "[return] <message>"`
(the guidance says how), and the echo's result is rewritten in place, to the branch message when it
opens and to the returned message when it folds. Everything before the branch call stays as it was,
so the prompt cache holds through it. A branch cannot branch (FoldAgent's rule); a call to do so is
answered by its message to return first. The conversation itself says which branches are open or
done, so the strategy keeps no state. Fixed-threshold compaction is the fallback, as in AutoCompact:
when the context as folded still reaches the compaction threshold (the agent did not branch, or a
branch grew past it), the conversation is compacted as `compaction` compacts it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.ir import Context, Item, Kind, Request
from ..core.tokens import item_tokens
from .base import Summarizer
from .compaction import Compaction
from .conversation import signalled, step

GUIDANCE = """## Branches (context folding)

For a focused sub-task (exploring code, reading a set of files, trial and error), open a branch:
run the shell command `echo "[branch] <3-5 word description> :: <task: objectives and what to
report>"`, alone (no other tool call beside it). You keep everything you know; work on the sub-task
only. When it is done, or you cannot go further, run `echo "[return] <message>"`, alone (no other
tool call beside it): a comprehensive message with the outcome, every exact value found, files
changed, and what is still pending. Then the branch's steps are folded away and only your message
stays, so put everything that matters into it. A branch cannot
open another branch."""
# FoldAgent's messages (agents/prompts.py BRANCH_MESSAGE, agents/fold_agent.py).
BRANCH = """ROLE CHANGE: `MODE: BRANCH`

You are now a branch. You have been assigned a specific task by MAIN as shown above and inherit their full context and understanding of the problem.

Your role is to focus exclusively on the assigned task. Return immediately after completing it, and never perform actions beyond the specified task. Focus exclusively on the assigned task.

Your final response must be clear and compact, while faithfully and comprehensively capture:
* Outcome of your assigned task.
* Any files modified, created, or deleted; any environment changes or commands that affected the system.
* Key insight: Important discoveries about the codebase, problem patterns, architecture understanding, or technical details that future phases need to know
* What you assumed from previous context, what future phases should know, any unresolved questions or potential issues
* Any additional notes you would like to inform MAIN about.

Assigned task ({description}):
{task}

* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return to return (`echo "[return] <message>"`), do not perform action beyond the assigned sub task."""
RETURNED = "Branch has finished its task, the returned message is:\n\n{message}"
NESTED = "You are in branch mode and cannot branch task or finish the task. Use return to go back to the main agent."
SIGNAL = re.compile(r"\[(branch|return)\][ \t]*([^\n]*)")  # an echoed signal, to the end of its line


@dataclass(frozen=True)
class ContextFolding:
    fallback: Compaction = field(default_factory=Compaction)  # the fixed-threshold limit, if folding does not hold it
    name: str = "folding"

    @classmethod
    def from_settings(cls) -> ContextFolding:
        return cls(Compaction.from_settings())

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "fallback": self.fallback.fingerprint(), "guidance": GUIDANCE, "branch": BRANCH}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        items, changed, opened = list(request.current), False, None  # opened: the open branch's result, its step's end
        n = 0
        while (found := step(items, n)) is not None:
            start, first, end = found
            results = items[first:end]
            if (marked := [first + k for k, result in enumerate(results) if BRANCH[:20] in result.text]):
                opened = marked[0], end  # opened earlier, still open
            elif any(RETURNED[:20] in result.text for result in results):
                opened = None
            if (signal := _signal(items[start:first], results)) is None:
                n = end
                continue
            kind, text, k = signal
            echo = first + k
            if kind == "branch" and opened is not None:
                items[echo] = replace(items[echo], text=NESTED)
            elif kind == "branch":
                description, _, task = text.partition("::")
                items[echo] = replace(items[echo], text=BRANCH.format(description=description.strip(), task=task.strip()))
                opened = echo, end
            elif opened is not None:  # the branch returns: its steps and the return call fold away
                items[opened[0]] = replace(items[opened[0]], text=RETURNED.format(message=text))
                del items[opened[1]:end]
                end, opened = opened[1], None
            else:  # a return with no branch open: left as it is
                n = end
                continue
            changed = True
            n = end
        guidance = Item(Kind.SYSTEM, GUIDANCE)
        folded = items if changed else list(request.current)
        tokens = request.tokens - sum(map(item_tokens, request.current)) + sum(map(item_tokens, folded))
        if (context := self.fallback.plan(replace(request, tokens=tokens), summarizer)) is not None:
            return Context((guidance, *context.items))  # (the summary covers the branches' steps too)
        return Context((guidance, *folded))


def _signal(response: list[Item], results: list[Item]) -> tuple[str, str, int] | None:
    """A branch signal, what it says and which result is the echo's (see `signalled`); None once
    its output is rewritten."""

    if any(mark[:20] in result.text for result in results for mark in (BRANCH, RETURNED, NESTED)):
        return None
    return signalled(response, results, SIGNAL, whole=True)
