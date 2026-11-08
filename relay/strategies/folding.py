"""Context Folding (FoldAgent, Sun et al.) as a Relay strategy: the agent branches off a sub-task and
returns from it with a message, and the branch's steps fold away.

As in FoldAgent (github.com/sunnweiwei/FoldAgent, `agents/fold_agent.py`), with its prompts verbatim:
the system prompt gets its `<MULTI_AGENT>` block, its `branch` and `return` tools and the MAIN
directive of its user prompt (without the SWE phase list). The harness's own shell stands in for the
tools, and only their call syntax is mapped: the agent runs `echo "[branch] <description> :: <prompt>"`
and `echo "[return] <message>"`, each alone. A branch inherits the whole conversation; the branch
call's result becomes FoldAgent's branch message ("ROLE CHANGE: `MODE: BRANCH`"), and the task is
the call's, shown above it. While a branch is open, each of its observations ends with FoldAgent's
branch-mode reminder (after the step's last result; where one result answers several calls, in a
note on the request it ends), and a branch call is refused with FoldAgent's message, the reminder
after it. A branch closes when it returns, and the call's result becomes "Branch has finished its
task, the returned message is: ...", or when the model ends its turn without returning (an answer
with no tool call), and it becomes FoldAgent's text for a branch that stops without a return,
"Branch has finished its task. The last message was: ..." with that answer. Then the branch's own
steps fold away; everything else, the user's messages included, stays, and nothing before the
branch call changes, so the prompt cache holds through it. As in FoldAgent, a branch must return
512 tokens before the context's limit, counted over the whole context (FoldAgent counts from its
prompt, main's steps before the branch included): past it, the branch's last step is rolled back,
FoldAgent's prompt to return is added, and the model's next response closes the branch (its
returned message, else that response as the last message). At most 10 branches stand in the
context (FoldAgent's `max_session` in its evaluations); the 11th is refused with its message. The
conversation itself says which branches are open or done, so the strategy keeps no state.

The context's limit is Relay's addition: FoldAgent ends the episode when main's context is full,
which a harness cannot. Here it is the compaction threshold: with no branch open, a context that
still reaches it is compacted as `compaction` compacts it (its settings), and a branch must return
512 tokens before it (114,688 tokens at the default 90% of a 128K window).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.ir import Context, Item, Kind, Request
from ..core.tokens import item_tokens
from .base import Summarizer
from .compaction import Compaction
from .conversation import RESPONSE, calls, echoed, ignored, signalled

# FoldAgent's prompts, verbatim but for the call syntax: CODE_SYSTEM_PROMPT_BRANCH's <MULTI_AGENT>
# (agents/prompts.py:544-556), branch_tool() as convert_tools_to_description renders it, with
# TOOL_PROMPT's opening (agents/tool_spec.py:240-283), and CODE_USER_PROMPT_BRANCH's MAIN directive
# (agents/prompts.py:597, 638).
GUIDANCE = """<MULTI_AGENT>
You operate in one of two modes: MAIN or BRANCH. Your current role will be clearly stated at the beginning of each turn. You must follow ONLY the instructions for your assigned role.

### `MODE: MAIN`
* As MAIN, you are the orchestrator who identifies sub-tasks that involve trial-and-error or exploration (such as code exploration, test creation, fix implementation, or verification) and branches them.
* MAIN does not conduct specific work. MAIN focuses on planning, orchestration, and verification.
* When branching, provide concise, clear, specific objectives and indicate what information needs to be preserved.

### **`MODE: BRANCH`**
* As a BRANCH, you are the executor. You should focus intensively on your assigned task; do not perform any action beyond the assigned task.
* Think thoroughly about the subtask, and work hard to effectively and comprehensively complete the assigned subtask.
* Your returned response should faithfully capture the subtask results, any state changes to the repository or environment, key insights about the codebase, and critical information that subsequent phases will need.
</MULTI_AGENT>

You have access to the following functions:

---- BEGIN FUNCTION #1: branch ----
Description: Create a sub-branch to execute a sub-task.
Parameters:
  (1) description (string, required): A concise 3-5 word identifier for the sub-task  (e.g., "Narrow down last 7 overs stats", "Code Exploration", "Test Creation").
  (2) prompt (string, required): Clear, compact task prompt: state objectives and critical info to preserve in the response. Be brief and informative.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: return ----
Description: Finish the interaction when the sub task is complete OR if the assistant cannot proceed further with the task.
Parameters:
  (1) message (string, required): A comprehensive message describing sub task outcome.
---- END FUNCTION #2 ----

If you choose to call one of these functions, run it as a shell command in the following format, alone (no other tool call beside it):

echo "[branch] <description> :: <prompt>"
echo "[return] <message>"

Now you are MAIN, you will orchestrate the solution by managing strategic phases directly and branching execution phases.

Now you are in `MODE: MAIN`"""
BRANCH = '''ROLE CHANGE: `MODE: BRANCH`

You are now a branch. You have been assigned a specific task by MAIN as shown above and inherit their full context and understanding of the problem.

Your role is to focus exclusively on the assigned task. Return immediately after completing it, and never perform actions beyond the specified task. Focus exclusively on the assigned task.

Your final response must be clear and compact, while faithfully and comprehensively capture:
* Outcome of your assigned task.
* Any files modified, created, or deleted; any environment changes or commands that affected the system.
* Key insight: Important discoveries about the codebase, problem patterns, architecture understanding, or technical details that future phases need to know
* What you assumed from previous context, what future phases should know, any unresolved questions or potential issues
* Any additional notes you would like to inform MAIN about.

When you have completed your assigned task, use the return tool to formally return control to MAIN:

echo "[return] Your message to MAIN here — include branch results, state changes, key insights, dependencies, and notes."'''  # (prompts.py:812-831)
# agents/fold_agent.py: the reminder after each observation in a branch (:173), the answer to a branch
# call in it (:171), the prompt to return at the branch limit (:172), the texts a closed branch's call
# is answered with (:183, :189) and the answer to a branch past the cap (:151).
REMINDER = ("* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you "
            'complete the assigned sub task, use `echo "[return] <message>"` to return, do not perform action beyond '
            "the assigned sub task.")
NESTED = "You are in branch mode and cannot branch task or finish the task. Use the `return` tool to go back to the main agent."
FORCED = ("The context limit has been exceeded for the branch. Please finish the sub task directly and clearly state the "
          "progress made and the pending jobs of the sub task. Only summarize the sub task progress, using the return tool.")
RETURNED = "Branch has finished its task, the returned message is:\n\n{message}"
UNRETURNED = "Branch has finished its task. The last message was:\n\n{message}"
LIMIT = "You've already reached the limit of {count} branch calls. Continue working independently."
MAX_SESSION = 10  # branches (scripts/eval_swe.py, eval_bc.py: max_session = val_max_session = 10)
RESERVE = 512  # a branch returns this many tokens before the context's limit (agents/utils.py:412-415)
SIGNAL = re.compile(r"\[(branch|return)\][ \t]*([^\n]*)")  # an echoed signal, to the end of its line
OWN = RESPONSE | {Kind.TOOL_RESULT}  # what a branch's steps are made of


@dataclass(frozen=True)
class ContextFolding:
    fallback: Compaction = field(default_factory=Compaction)  # the context's limit (Relay's addition)
    name: str = "folding"

    @classmethod
    def from_settings(cls) -> ContextFolding:
        return cls(Compaction.from_settings())

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "fallback": self.fallback.fingerprint(), "guidance": GUIDANCE, "branch": BRANCH,
                "reminder": REMINDER, "forced": FORCED, "max_session": MAX_SESSION, "reserve": RESERVE}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        items, notes = list(request.current), ()
        opened = last = None  # the open branch (its result, its step's end, its description), and its last step
        forced, count, n = False, 0, 0  # told to return; the branches the context holds
        while (found := _response(items, n)) is not None:
            start, first, end = found
            forced = forced or (opened is not None and any(map(_forced, items[n:start])))
            response, results = items[start:first], items[first:end]
            called, n = any(item.kind is Kind.TOOL_CALL for item in response), end
            if (called and not results) or (not called and (opened is None or not _said(response, Kind.ASSISTANT))):
                continue  # (calls not answered yet; an answer outside a branch)
            mark = next((mark for mark in (BRANCH, RETURNED, NESTED, LIMIT) if any(mark[:20] in r.text for r in results)), None)
            if mark is BRANCH:  # opened on an earlier request, still open
                opened, last, forced, count = (first, end, _description(response)), None, False, count + 1
                continue
            if mark is RETURNED:  # (a branch closed)
                count += 1
                continue
            signal = signalled(response, results, SIGNAL, whole=True) if called and mark is None else None
            lone = signal[0] if signal and signal[3] else None
            if opened is not None and (not called or forced or lone == "return"):  # the branch closes
                text = (RETURNED.format(message=signal[1]) if lone == "return" else
                        UNRETURNED.format(message=_said(response, Kind.ASSISTANT, Kind.TOOL_CALL)))
                n, opened, last, forced = _close(items, opened[0], opened[1], end, text), None, None, False
                continue
            if signal and not signal[3]:  # beside other calls
                notes += ignored(items, None if signal[2] is None else first + signal[2], end, signal[0])
            elif lone == "branch" and opened is not None:  # (its only call: the one result)
                items[first] = replace(items[first], text=NESTED)
            elif lone == "branch" and count >= MAX_SESSION:
                items[first] = replace(items[first], text=LIMIT.format(count=count))
            elif lone == "branch":
                items[first] = replace(items[first], text=BRANCH)
                opened, last, count = (first, end, _description(response)), None, count + 1
                continue
            if opened is not None:  # a step of the branch's own (a return with no branch open is left as it is)
                notes += _remind(items, response, first, end, REMINDER.format(description=opened[2]))
                last = start, end
        forced = forced or (opened is not None and any(map(_forced, items[n:])))  # (told, not answered yet)
        tokens = request.tokens - sum(map(item_tokens, request.current)) + sum(map(item_tokens, items))
        guidance = Item(Kind.SYSTEM, GUIDANCE)
        if opened is not None and not request.force:
            if not forced and self.fallback.due(replace(request, tokens=tokens + RESERVE))[0]:
                if last is not None:  # FoldAgent rolls the branch's last step back first (agents/utils.py:446-447)
                    notes = () if last[1] == len(items) else notes  # (the notes were that step's)
                    del items[last[0]:last[1]]
                items.append(Item(Kind.USER, FORCED))
            return Context((guidance, *items), notes=notes)
        if (context := self.fallback.plan(replace(request, tokens=tokens), summarizer)) is not None:
            return Context((guidance, *context.items), notes=notes)  # (the summary covers the branches' steps too)
        return Context((guidance, *items), notes=notes)


def _response(items: list[Item], n: int) -> tuple[int, int, int] | None:
    """The first model response from `n` on, and the results answering it: where it starts, where
    its results start and where they end."""

    while n < len(items) and items[n].kind not in RESPONSE:
        n += 1
    if n == len(items):
        return None
    start = n
    while n < len(items) and items[n].kind in RESPONSE:
        n += 1
    end = n
    while end < len(items) and items[end].kind is Kind.TOOL_RESULT:
        end += 1
    return start, n, end


def _close(items: list[Item], result: int, start: int, end: int, text: str) -> int:
    """The open branch closes: its call's result (`result`) says `text`, and its own steps (from
    `start`, to `end`) and the prompt to return fold away; anything else, as the user's messages,
    stays. Where the conversation goes on."""

    items[result] = replace(items[result], text=text)
    kept = [item for item in items[start:end] if item.kind not in OWN and not _forced(item)]
    items[start:end] = kept
    return start + len(kept)


def _remind(items: list[Item], response: list[Item], first: int, end: int, reminder: str) -> tuple[str, ...]:
    """The reminder after a branch step's observation: at the end of its last result, or, where one
    result answers several calls (Anthropic Messages, Gemini), in a note on the request it ends."""

    if end - first == 1 and sum(len(calls(item.text)[0]) for item in response if item.kind is Kind.TOOL_CALL) > 1:
        return (reminder,) if end == len(items) else ()
    if not items[end - 1].text.endswith(reminder):
        items[end - 1] = replace(items[end - 1], text=f"{items[end - 1].text}\n{reminder}")
    return ()


def _description(response: list[Item]) -> str:
    """A branch's description, from its call."""

    return (echoed(response, SIGNAL) or "[branch]")[len("[branch]"):].partition("::")[0].strip()


def _said(response: Sequence[Item], *kinds: Kind) -> str:
    return "\n\n".join(item.text.strip() for item in response if item.kind in kinds and item.text.strip())


def _forced(item: Item) -> bool:
    return item.kind is Kind.USER and item.text == FORCED
