"""ACE (Agentic Context Engineering, Zhang et al., arXiv 2510.04618) as a Relay strategy: an evolving
playbook of strategies and lessons, kept across conversations.

As in ACE's agent setup (github.com/ace-agent/ace-appworld, online adaptation without ground truth,
`ACE_online_no_GT`), the playbook goes in the agent's context on every request, in the words of ACE's
generator prompt. A task ends when the agent ends it (AppWorld's `complete_task()`; here: a turn
the agent closes with a reply and no tool call, before the user's next message). Then the
reflector diagnoses that task's whole trajectory, and the curator turns its reflection into
additions to the playbook: `ADD` operations only, into ACE's sections, merged by ACE's own rules
(`playbook.py`: a fresh id per bullet, appended to its section, else to OTHERS). The prompts are
ACE's (`experiments/prompts/appworld_react_{reflector_no_gt,curator}_prompt.txt`, Apache-2.0),
the reflector's naming a coding agent where it named an AppWorld one; the playbook starts with
ACE's sections and none of AppWorld's own bullets.

Two differences come from where Relay sits. Relay sees a turn end only when the conversation's
next request comes, so a task is reflected on then (that request waits for it), and a
conversation's last turn is never. And ACE does not bound a task's context (AppWorld's agent
trims old outputs itself): past the compaction threshold, the conversation is compacted as
`compaction` compacts it, which leaves the playbook as it is.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import settings
from ..core.ir import Context, Item, Kind, Request
from .base import Summarizer
from .compaction import Compaction

log = logging.getLogger("relay")
PROMPTS = Path(__file__).parents[1] / "prompts"
REFLECTOR = (PROMPTS / "ace_reflector.md").read_text(encoding="utf-8")
CURATOR = (PROMPTS / "ace_curator.md").read_text(encoding="utf-8")
# ACE's generator prompt, around the playbook (appworld_react_generator_prompt.txt).
PLACED = ("You are also provided with a curated playbook of strategies, API-specific information, common mistakes, "
          "and proven solutions to help you solve the task effectively.\n\n"
          "**Playbook**: - Read the **Playbook** first, then execute the task by explicitly leveraging each relevant "
          "section:\n### PLAYBOOK BEGIN\n{playbook}\n### PLAYBOOK END")
# ACE's sections (appworld_initial_playbook.txt), without AppWorld's own bullets.
INITIAL = "\n\n".join(["## STRATEGIES AND HARD RULES", "## APIs TO USE FOR SPECIFIC INFORMATION",
                       "## USEFUL CODE SNIPPETS AND TEMPLATES", "## COMMON MISTAKES AND CORRECT STRATEGIES",
                       "## PROBLEM-SOLVING HEURISTICS AND WORKFLOWS", "## VERIFICATION CHECKLIST",
                       "## TROUBLESHOOTING AND PITFALLS:", "## OTHERS"]) + "\n"
SECTIONS = {"strategies_and_hard_rules", "apis_to_use_for_specific_information", "useful_code_snippets_and_templates",
            "common_mistakes_and_correct_strategies", "problem_solving_heuristics_and_workflows",
            "verification_checklist", "troubleshooting_and_pitfalls", "others"}  # the curator's (adaptation_react.py)
SLUGS = {"strategies_and_hard_rules": "shr", "hard_rules": "hr", "strategies_and_insights": "si",
         "apis_to_use_for_specific_information": "api", "useful_code_snippets_and_templates": "code",
         "code_snippets_and_templates": "code", "common_mistakes_and_correct_strategies": "cms",
         "common_mistakes_to_avoid": "err", "problem_solving_heuristics_and_workflows": "psw",
         "problem_solving_heuristics": "prob", "verification_checklist": "vc", "troubleshooting_and_pitfalls": "ts",
         "others": "misc", "meta_strategies": "meta"}  # utils.get_section_slug
ROLES = {Kind.USER: "user", Kind.ASSISTANT: "assistant", Kind.TOOL_CALL: "assistant", Kind.TOOL_RESULT: "user",
         Kind.SUMMARY: "user"}  # (as ACE's messages: the agent's code and the environment's output)
SEEN = "See full conversation history below"
KEPT = 500  # tasks remembered as reflected on, per conversation


@dataclass(frozen=True)
class ACE:
    playbook: str = ""  # the playbook's file; "": in Relay's directory
    fallback: Compaction = field(default_factory=Compaction)  # the window, which ACE leaves to the agent
    name: str = "ace"

    @classmethod
    def from_settings(cls) -> ACE:
        return cls(settings.value("ace_playbook", ""), Compaction.from_settings())

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "playbook": self.playbook, "fallback": self.fallback.fingerprint(),
                "prompts": hashlib.sha256((REFLECTOR + CURATOR + PLACED).encode()).hexdigest()[:16]}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        state = request.state or {}
        reflected = list(state.get("reflected", []))
        path = Path(self.playbook).expanduser() if self.playbook else settings.home() / "ace_playbook.md"
        for task in _ended(request.current):
            key = _digest(task)
            if key in reflected:
                continue
            reflected.append(key)
            try:
                _learn(path, task, summarizer)
            except Exception:  # (the agent goes on with the playbook as it was)
                log.warning("ace: reflecting on a task failed; the playbook is unchanged", exc_info=True)
        playbook = Item(Kind.SYSTEM, PLACED.format(playbook=_read(path).strip()))
        compacted = self.fallback.plan(request, summarizer)
        items = compacted.items if compacted is not None else request.current
        return Context((playbook, *items), {"reflected": reflected[-KEPT:]})


def _ended(items: tuple[Item, ...]) -> list[tuple[Item, ...]]:
    """The tasks the agent ended: from a user message to the agent's reply before the next one."""

    tasks, start = [], None
    for n, item in enumerate(items):
        if item.kind is Kind.USER:
            if start is not None and n and items[n - 1].kind is Kind.ASSISTANT:
                tasks.append(items[start:n])
            start = n
    return tasks


def _digest(task: tuple[Item, ...]) -> str:
    return hashlib.sha256("\0".join(f"{i.kind.value}:{i.text}" for i in task).encode()).hexdigest()[:16]


def _history(task: tuple[Item, ...]) -> str:
    """ACE's "FULL CONVERSATION HISTORY" of a task."""

    lines = [f"[{n}] {ROLES.get(item.kind, 'user').upper()}: {item.text}\n" for n, item in enumerate(task)
             if item.kind in ROLES]
    return "\n\n=== FULL CONVERSATION HISTORY ===\n" + "\n".join(lines)


def _learn(path: Path, task: tuple[Item, ...], summarizer: Summarizer) -> None:
    """ACE's curator_call: the reflector on the task, then the curator's additions merged."""

    history, playbook = _history(task), _read(path)
    reflector = (REFLECTOR.replace("{{ground_truth_code}}", "").replace("{{test_report}}", "")
                 .replace("{{generated_code}}", SEEN).replace("{{generated_rationale}}", SEEN)
                 .replace("{{spec_or_api_docs}}", SEEN).replace("{{execution_error}}", SEEN)
                 .replace("{{playbook}}", playbook or "N/A").replace("{{previous_reflection}}", "N/A"))
    reflection = summarizer.complete([Item(Kind.USER, reflector + history)])
    question = next((item.text for item in task if item.kind is Kind.USER), "")
    curator = CURATOR.format(initial_generated_code=SEEN, final_generated_code=SEEN, guidebook=reflection,
                             current_playbook=playbook, question_context=question, gt=None)
    answer = summarizer.complete([Item(Kind.USER, curator + history)])
    operations = _operations(answer)
    with _locked(path):
        current = _read(path)  # (another conversation may have added to it meanwhile)
        updated = _apply(current, operations, _next_id(current))
        path.write_text(updated, encoding="utf-8")
    log.info("ace: %d bullets added to %s", len(operations), path)


def _operations(answer: str) -> list[dict[str, Any]]:
    """The curator's operations, validated as ACE validates them (else none)."""

    found = _json(answer)
    if not isinstance(found, dict) or not isinstance(found.get("reasoning"), str) \
            or not isinstance(found.get("operations"), list):
        raise ValueError(f"the curator's answer is not ACE's JSON: {answer[:200]!r}")
    operations = []
    for op in found["operations"]:
        if not isinstance(op, dict) or op.get("type") != "ADD" or not {"type", "section", "content"} <= op.keys():
            raise ValueError(f"not an ADD operation: {op!r}")  # (ACE rejects them all then)
        if _normalized(str(op["section"])) in SECTIONS:
            operations.append(op)
    return operations


def _json(text: str) -> Any:
    """ACE's extract_json_from_text: the whole answer, a ```json block, or the first balanced object."""

    for candidate in (text.strip(), *re.findall(r"```json\s*(.*?)\s*```", text, re.S | re.I)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for start in (n for n, char in enumerate(text) if char == "{"):
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            continue
    return None


def _normalized(section: str) -> str:
    return section.strip().lower().replace(" ", "_").replace("&", "and").rstrip(":")


def _next_id(playbook: str) -> int:
    """ACE's get_next_global_id: one past the highest bullet number."""

    numbers = [int(m.group(1)) for line in playbook.splitlines() if (m := re.match(r"\s*\[[^\]]*-(\d+)\]", line))]
    return max(numbers, default=0) + 1


def _apply(playbook: str, operations: list[dict[str, Any]], next_id: int) -> str:
    """ACE's apply_curator_operations: each addition at the end of its section, else of OTHERS."""

    lines = playbook.strip().split("\n")
    sections = {_normalized(line.strip()[2:]) for line in lines if line.strip().startswith("##")}
    adds: list[tuple[str, str]] = []
    for op in operations:
        section = _normalized(str(op.get("section", "general")))
        section = section if section in sections or section == "general" else "others"
        slug = SLUGS.get(section, section[:4])
        adds.append((section, f"[{slug}-{next_id:05d}] {op.get('content', '')}"))
        next_id += 1
    out, current = [], None
    for line in lines:
        if line.strip().startswith("##"):
            out += [bullet for section, bullet in adds if section == current]
            adds = [(s, b) for s, b in adds if s != current]
            current = _normalized(line.strip()[2:])
        out.append(line)
    out += [bullet for section, bullet in adds if section == current]
    adds = [(s, b) for s, b in adds if s != current]
    if adds:  # (no section of theirs: under OTHERS)
        at = next((n for n, line in enumerate(out) if line.strip() == "## OTHERS"), len(out) - 1)
        out[at + 1:at + 1] = [bullet for _, bullet in adds]
    return "\n".join(out) + "\n"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return INITIAL


class _locked:
    """The playbook's file, one conversation at a time (they share it)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.path.with_suffix(self.path.suffix + ".lock"), "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
