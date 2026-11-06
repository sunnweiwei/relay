"""The Harbor agents context_bench.py runs, imported by Harbor from its own environment (ContextBench, and for
`oracle_clm` the CLM package, installed there). Its first section holds the paper modes' prompts and tools; that
section needs only the standard library, and osworld.py runs it on its own for OSWorld's harnesses.

- RelayCodexStream: ContextBench's CodexStream (Codex, one `codex exec resume` per operation) through a Relay of
  the trial's own, with Codex's home sealed from the model's commands;
- OracleBase, OracleSummary: the paper's own base loop (ContextBenchAgent) and Codex's own compaction (CodexStream),
  on the same model, at the endpoint the driver names;
- PaperAgent: the paper's harness modes (ACM, Self-Compact, folding, RLM; their harness is unreleased) on
  ContextBench's loop, with the methods' own prompts and tools (the paper modes section);
- RelayChatAgent: Relay's strategies on that same loop, a Relay in front of that endpoint.

Every agent keeps ContextBench's context snapshots out of the task container until it is done (PrivateSnapshots).
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import contextbench.harbor.wrap as _wrap
import openai
from contextbench import stream as _proto
from contextbench.harbor.agents import ContextBenchAgent
from contextbench.harbor.agents.base import BASH_TOOL, NO_TOOL_CALL_TEXT
from contextbench.harbor.turn_source import TurnSource
from contextbench.harbor.wrap import CodexStream, _codex_items  # (the unpatched converter; it is patched below)
from contextbench.tokens import count_tokens, record_snapshot

sys.path.insert(0, str(Path(__file__).parent))
from context_bench import NEVER, OVERHEAD, RESERVE, WINDOW, with_skill  # noqa: E402

# ====================================================================== the paper modes
# The prompts and tools the CLM paper's harness modes (ACM, Self-Compact, context folding) gave the model: ACM's and
# Self-Compact's read from Relay's strategies (the methods' own, verbatim), FoldAgent's from its code. osworld.py
# runs this section on its own for OSWorld's harnesses, so it needs nothing beyond the standard library.
REPO = Path(__file__).resolve().parents[1]


def constants(module: str, *names: str) -> tuple[Any, ...]:
    """String constants of a Relay strategy, read from its source (the methods' own prompts, verbatim)
    without importing Relay into Harbor's environment."""

    tree = ast.parse((REPO / "relay" / "strategies" / f"{module}.py").read_text(encoding="utf-8"))
    found = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}
    return tuple(found[name] for name in names)


ACM_SUMMARY, ACM_RECALL = constants("acm", "SUMMARY_INSTRUCTION", "QUERY_MEMORY_PROMPT")
SC_RUBRIC, SC_SUMMARIZER, SC_QUESTION, SC_RESEARCH, SC_CONTINUE = constants(
    "selfcompact", "RUBRIC", "SUMMARIZER", "QUESTION", "RESEARCH", "CONTINUE")
# FoldAgent's branch message (agents/prompts.py, BRANCH_MESSAGE), verbatim.
FOLD_BRANCH = """ROLE CHANGE: `MODE: BRANCH`

You are now a branch. You have been assigned a specific task by MAIN as shown above and inherit their full context and understanding of the problem.

Your role is to focus exclusively on the assigned task. Return immediately after completing it, and never perform actions beyond the specified task. Focus exclusively on the assigned task.

Your final response must be clear and compact, while faithfully and comprehensively capture:
* Outcome of your assigned task.
* Any files modified, created, or deleted; any environment changes or commands that affected the system.
* Key insight: Important discoveries about the codebase, problem patterns, architecture understanding, or technical details that future phases need to know
* What you assumed from previous context, what future phases should know, any unresolved questions or potential issues
* Any additional notes you would like to inform MAIN about.

When you have completed your assigned task, use the return tool to formally return control to MAIN:

<function=return>
<parameter=message>
Your message to MAIN here — include branch results, state changes, key insights, dependencies, and notes.
</parameter>
</function>"""
FOLD_OVER = ("The context limit has been exceeded for the branch. Please finish the sub task directly and clearly "
             "state the progress made and the pending jobs of the sub task. Only summarize the sub task progress, "
             "using the return tool.")  # (FoldAgent's, for a branch past the limit)
BRANCH_STEPS = 40


def function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


# The tools the paper's harness gave ACM and folding, as its skills describe them; the descriptions are the
# methods' own (ACM's src/tools.py, FoldAgent's agents/tool_spec.py), ACM's range worded as the skill has it.
MANAGE_TOOL = function("manage_context", (
    "Compress your working memory. Takes everything since your previous manage_context call (or since the task "
    "message, on the first call), up to and including the turn in which you call it, and replaces it with one "
    "message \"[summary_id: N] <summary>\". The system prompt and the task message are never included. The "
    "original messages are saved as summary N (retrievable later with query_memory(summary_id, query)).\n\n"
    "This tool takes no arguments."), {}, [])
QUERY_TOOL = function("query_memory", (
    "Retrieve detailed information from a previously compressed summary. Each manage_context call produces a "
    "summary prefixed with [summary_id: N] and saves the original messages as summary_N. This tool loads them and "
    "uses an LLM to extract information matching your query from the original (uncompressed) content.\n\n"
    "DO NOT CALL THIS TOOL until at least one manage_context call has produced a summary_id."), {
    "summary_id": {"type": "integer", "description": "The summary_id (as shown in the [summary_id: N] prefix of a "
                   "prior manage_context summary) whose original content should be searched."},
    "query": {"type": "string", "description": "What specific information to extract from the original messages "
              "of that summary."}}, ["summary_id", "query"])
BRANCH_TOOL = function("branch", "Create a sub-branch to execute a sub-task.", {
    "description": {"type": "string", "description": 'A concise 3-5 word identifier for the sub-task  (e.g., '
                    '"Narrow down last 7 overs stats", "Code Exploration", "Test Creation").'},
    "prompt": {"type": "string", "description": "Clear, compact task prompt: state objectives and critical info to "
               "preserve in the response. Be brief and informative."}}, ["description", "prompt"])
RETURN_TOOL = function("return", "Finish the interaction when the sub task is complete OR if the assistant cannot "
                        "proceed further with the task.", {
    "message": {"type": "string", "description": "A comprehensive message describing sub task outcome."}}, ["message"])
THINK = re.compile(r"<think>.*?</think>", re.S)
VERDICT = re.compile(r"\b(C1|C2|C3|N1)\b(?:\s+[A-Z-]+)?[^A-Za-z0-9\n]*([YN])\b")


def fires(answer: str) -> bool:
    """Self-Compact's rule: compress iff C1 = C2 = C3 = Y and N1 = N (a missing answer is N)."""

    verdict: dict[str, str] = {}
    for label, value in VERDICT.findall(answer or ""):
        verdict.setdefault(label, value)
    return all(verdict.get(k) == "Y" for k in ("C1", "C2", "C3")) and verdict.get("N1", "N") == "N"


def render_message(message: dict[str, Any]) -> str:
    calls = " ".join(f"{(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})"
                     for c in message.get("tool_calls") or [])
    return (str(message.get("content") or "") + (f" [tool_call] {calls}" if calls else "")).strip()


# ====================================================================== the agents

# Codex keeps every operation in its home (the rollout `codex exec resume` reads), where models found and grepped
# them: the model's commands run as the host's user (which also owns CLM's mounted file), and Codex's home is root's.
BASH_WRAPPER = """#!/bin/bash.real
if [ "$(cat /proc/$PPID/comm 2>/dev/null)" = "codex" ]; then
  exec env HOME=/tmp/agent-home setpriv --reuid={uid} --regid={gid} --clear-groups /bin/bash.real "$@"
fi
exec /bin/bash.real "$@"
"""


SEALED = "/root/cb_codex"  # the prompt and Codex's event log of each turn, root's alone


async def _seal(agent: Any, environment: Any, home: str) -> None:
    await agent._cb_put_text(environment, BASH_WRAPPER.format(uid=os.getuid(), gid=os.getgid()), "/tmp/bash.wrapper")
    result = await environment.exec(
        f"set -e; [ -e /bin/bash.real ] || cp /bin/bash /bin/bash.real; install -m 755 /tmp/bash.wrapper /bin/bash; "
        f"rm -f /tmp/bash.wrapper; mkdir -p /tmp/agent-home; chown {os.getuid()}:{os.getgid()} /tmp/agent-home; "
        f"chmod 700 {home}; mkdir -p {SEALED}; chmod 700 {SEALED}", timeout_sec=60)
    if result.return_code != 0:
        raise RuntimeError(f"could not seal Codex's home: {result.stdout} {result.stderr}")


def _flat(item: dict[str, Any]) -> dict[str, Any]:
    """A tool result's output as its text: Codex sends it as a list of parts, which ContextBench's
    converter would write out as the list's repr (newlines and quotes escaped, so nothing matches)."""

    output = item.get("output")
    if item.get("type") in ("function_call_output", "custom_tool_call_output") and isinstance(output, list):
        return {**item, "output": "".join(str(p.get("text") or "") for p in output if isinstance(p, dict))}
    return item


# CodexStream's own snapshots, from Codex's rollout, are read through _flat too.
_wrap._codex_items = (lambda convert: lambda items: convert([_flat(i) if isinstance(i, dict) else i
                                                            for i in items or []]))(_wrap._codex_items)


class PrivateSnapshots:
    """The context snapshots kept out of the container while the agent runs: ContextBench writes them
    under the agent's log directory, which is mounted in the task container, and a model found and
    grepped them for values its context no longer held. They are moved there when the agent is done,
    before the verifier (which reads them there) runs. (The trial's own directory is not mounted.)"""

    def cb_snapshot(self, messages: list[Any], kind: str = "sent", extra: dict[str, Any] | None = None) -> int:
        src = self.cb_turn_source
        tokens, _ = count_tokens(messages)
        self.cb_peak_tokens = max(getattr(self, "cb_peak_tokens", 0), tokens)
        step = getattr(self, "_cb_snap_step", 0)
        record_snapshot(self._private, step, list(messages),
                        {"kind": kind, "n_ops_delivered": src.n_delivered if src else 0,
                         "n_ops_total": src.n_total if src else 0, "budget": self.cb_budget,
                         "reserve": self.cb_reserve, **(extra or {})})
        self._cb_snap_step = step + 1
        return tokens

    @property
    def _private(self) -> Path:
        return self.logs_dir.parent / "private"

    async def cb_setup(self, environment: Any, *, budget: int | None = None, reserve: int | None = None,
                       **load_kwargs: Any) -> Any:
        """CBStreamMixin's, with the stream's own record (`interactive_stream.json`: every command that
        released an operation, batches pasted to disk included) kept out of the container too."""

        if budget is not None:
            self.cb_budget = int(budget)
        if reserve is not None:
            self.cb_reserve = int(reserve)
        self._private.mkdir(parents=True, exist_ok=True)
        self.cb_source = await TurnSource.load(environment, logs_dir=self._private, **load_kwargs)
        self._cb_snap_step = 0
        self.cb_peak_tokens = 0
        return self.cb_source

    async def _turn(self, environment: Any, prompt: str, n: int) -> list[dict[str, Any]]:
        """CodexStream's turn, its prompt and Codex's event log kept in a directory only root reads (it
        tees them under /logs/agent, where every command's earlier output stays readable to the model);
        the log is read back as root and kept with the snapshots."""

        work = PurePosixPath(SEALED)
        await self._cb_put_text(environment, prompt, str(work / "prompt.txt"))
        model = (self.model_name or "").split("/")[-1]
        head = f"codex exec resume {shlex.quote(self.thread_id)}" if self.thread_id else "codex exec"
        log = work / f"codex-turn-{n:04d}.jsonl"
        await environment.exec(
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            f"{head} {self.COMPAT_FLAGS if self.cb_compat else ''} --dangerously-bypass-approvals-and-sandbox "
            f"--skip-git-repo-check --model {shlex.quote(model)} --json "
            f'-- "$(cat {work / "prompt.txt"})" > {log} 2>&1 </dev/null',
            env=self._env(), timeout_sec=self.cb_turn_timeout)
        text = (await environment.exec(f"cat {log}", timeout_sec=300)).stdout or ""
        (self._private / "logs").mkdir(parents=True, exist_ok=True)
        (self._private / "logs" / log.name).write_text(text, encoding="utf-8")
        events = []
        for line in text.splitlines():
            if line.strip().startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for event in events:
            if event.get("type") == "thread.started" and event.get("thread_id"):
                self.thread_id = event["thread_id"]
        return events

    def _publish(self) -> None:
        for name, target in (("context_snapshots", self.logs_dir / "context_snapshots"), ("logs", self.logs_dir)):
            source = self._private / name
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                for path in source.iterdir():
                    shutil.move(str(path), target / path.name)
        for path in self._private.glob("*.json"):
            shutil.move(str(path), self.logs_dir / path.name)
        shutil.rmtree(self._private, ignore_errors=True)


if hasattr(CodexStream, "options_model"):  # (Harbor 0.24, the Codex arms'; the CLM package's 0.16.1 has none)
    class RelayCodexOptions(CodexStream.options_model):  # (Harbor takes only declared kwargs)
        relay_url: str = ""
        relay_trace: str = ""
        codex_auth: str = ""
        skill_text: str = ""
        family: str = ""
        context_budget_tokens: int = WINDOW
        context_budget_reserve_tokens: int = RESERVE

    class RelayCodexStream(PrivateSnapshots, CodexStream):
        """ContextBench's CodexStream, signed in with ChatGPT and sent through a Relay: the snapshots and
        the budget wall are what Relay forwarded."""

        options_model = RelayCodexOptions

        def __init__(self, *args: Any, relay_url: str = "", relay_trace: str = "", codex_auth: str = "",
                     skill_text: str = "", family: str = "", context_budget_tokens: int = WINDOW,
                     context_budget_reserve_tokens: int = RESERVE, **kwargs: Any):
            super().__init__(*args, context_budget_tokens=context_budget_tokens,
                             context_budget_reserve_tokens=context_budget_reserve_tokens, **kwargs)
            self.relay_url, self.relay_trace, self.codex_auth = relay_url, Path(relay_trace), codex_auth
            self.skill_text = Path(skill_text).read_text(encoding="utf-8") if skill_text else ""
            self.family = family
            self.trace_at = 0
            self.last: list[dict[str, Any]] = []

        def _env(self) -> dict[str, str]:
            return {"CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix()}  # (ChatGPT's login, no API key)

        async def _configure(self, environment: Any) -> None:
            model = (self.model_name or "").split("/")[-1]
            toml = "\n".join([
                "include_permissions_instructions = false", "include_apps_instructions = false",
                "include_collaboration_mode_instructions = false", "include_environment_context = false",
                f"model = {json.dumps(model)}",
                f"model_context_window = {NEVER}", f"model_auto_compact_token_limit = {NEVER}",
                f"openai_base_url = {json.dumps(self.relay_url)}",
                f"stream_idle_timeout_ms = {int(self.cb_stream_idle_ms)}", "",
                "[features]", "shell_snapshot = false  # (its snapshot is written by a command, which cannot write Codex's home)", ""])
            home = self._REMOTE_CODEX_HOME.as_posix()
            await environment.exec(f"mkdir -p {home} {self._WORK} /logs/agent", env=self._env(), timeout_sec=60)
            await self._cb_put_text(environment, toml, f"{home}/config.toml")
            await self._cb_put_text(environment, Path(self.codex_auth).read_text(), f"{home}/auth.json")
            await _seal(self, environment, home)

        def _forwarded(self) -> list[list[dict[str, Any]]]:
            """Every request Relay forwarded since the last look, as messages."""

            if not self.relay_trace.exists():
                return []
            with self.relay_trace.open("rb") as stream:
                stream.seek(self.trace_at)
                data = stream.read()
            end = data.rfind(b"\n") + 1  # (a record still being written waits for the next look)
            self.trace_at += end
            out = []
            for line in data[:end].splitlines():
                record = json.loads(line)
                if "items" not in record or not isinstance(record.get("body"), dict):
                    continue  # (Relay's own summary requests, usage lines)
                body = record.get("forwarded") or record["body"]
                system = [{"role": "system", "content": body["instructions"]}] if body.get("instructions") else []
                out.append(system + _codex_items([_flat(item) for item in body.get("input") or []]))
            return out

        async def _session_messages(self, environment: Any) -> list[dict[str, Any]]:
            sent = self._forwarded()
            usable = self.cb_budget - self.cb_reserve
            for messages in sent[:-1]:
                tokens = self.cb_snapshot(messages, "sent")
                if tokens > usable:
                    return messages  # (the caller snapshots it again and calls the wall)
            self.last = sent[-1] if sent else self.last
            return self.last

        async def run(self, instruction: str, environment: Any, context: Any) -> None:
            try:
                await super().run(with_skill(instruction, self.skill_text, self.family), environment, context)
            finally:
                self._publish()


class OracleBase(PrivateSnapshots, ContextBenchAgent):
    """ContextBench's append-only loop, with the paper's base skill in its place (its snapshots out of the
    model's reach while it runs, as every other arm's: a model once read its own back from /logs/agent)."""

    def __init__(self, *args: Any, skill_text: str = "", family: str = "", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.skill_text = Path(skill_text).read_text(encoding="utf-8") if skill_text else ""
        self.family = family

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        try:
            await super().run(with_skill(instruction, self.skill_text, self.family), environment, context)
        finally:
            self._publish()

class OracleSummary(PrivateSnapshots, CodexStream):
    """ContextBench's CodexStream signed in with ChatGPT, Codex compacting itself: as text (through a
    provider of its own, Codex summarizes locally with its own prompt, where the OpenAI provider
    compacts remotely into an encrypted item no snapshot can read) once the request reaches 75% of the
    budget as ContextBench counts it (Codex's count includes its tool definitions, OVERHEAD)."""

    def __init__(self, *args: Any, codex_auth: str = "", skill_text: str = "", family: str = "", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.codex_auth = codex_auth
        self.skill_text = Path(skill_text).read_text(encoding="utf-8") if skill_text else ""
        self.family = family

    def _env(self) -> dict[str, str]:
        return {"CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix()}

    async def _configure(self, environment: Any) -> None:
        model = (self.model_name or "").split("/")[-1]
        toml = "\n".join([
            "include_permissions_instructions = false", "include_apps_instructions = false",
            "include_collaboration_mode_instructions = false", "include_environment_context = false",
            f"model = {json.dumps(model)}", 'model_provider = "chatgpt"',
            f"model_context_window = {WINDOW + OVERHEAD}",
            f"model_auto_compact_token_limit = {int(0.75 * WINDOW) + OVERHEAD}",
            f"stream_idle_timeout_ms = {int(self.cb_stream_idle_ms)}", "",
            "[model_providers.chatgpt]", 'name = "ChatGPT"', 'base_url = "https://chatgpt.com/backend-api/codex"',
            'wire_api = "responses"', "requires_openai_auth = true", "",
            "[features]", "shell_snapshot = false  # (its snapshot is written by a command, which cannot write Codex's home)", ""])
        home = self._REMOTE_CODEX_HOME.as_posix()
        await environment.exec(f"mkdir -p {home} {self._WORK} /logs/agent", env=self._env(), timeout_sec=60)
        await self._cb_put_text(environment, toml, f"{home}/config.toml")
        await self._cb_put_text(environment, Path(self.codex_auth).read_text(), f"{home}/auth.json")
        await _seal(self, environment, home)

    async def _session_messages(self, environment: Any) -> list[dict[str, Any]]:
        """CodexStream's reading of Codex's rollout (what the model saw, after its compactions), but
        read as root through exec: CodexStream copies it under /logs/agent, where the model could read it."""

        home = self._REMOTE_CODEX_HOME.as_posix()
        name = f"*{self.thread_id}*" if self.thread_id else "*"
        result = await environment.exec(f"cat \"$(ls -t {home}/sessions/*/*/*/rollout-{name}.jsonl | head -1)\"",
                                        timeout_sec=300)
        messages: list[dict[str, Any]] = []
        system: list[dict[str, Any]] = []
        for line in (result.stdout or "").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, payload = record.get("type"), record.get("payload") or {}
            if kind == "session_meta":
                base = payload.get("base_instructions")
                text = base.get("text") if isinstance(base, dict) else base
                system = [{"role": "system", "content": str(text)}] if text else system
            elif kind == "compacted":
                replacement = payload.get("replacement_history")
                messages = _wrap._codex_items(replacement) if isinstance(replacement, list) else [
                    {"role": "user", "content": str(payload.get("message") or "")}]
            elif kind == "response_item":
                messages += _wrap._codex_items([payload])
        return system + messages

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        try:
            await super().run(with_skill(instruction, self.skill_text, self.family), environment, context)
        finally:
            self._publish()


class _ChatAgent(PrivateSnapshots, ContextBenchAgent):
    """ContextBenchAgent with the tools a call offers chosen per call, and the paper's skill in place."""

    def __init__(self, *args: Any, skill_text: str = "", family: str = "", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.skill_text = Path(skill_text).read_text(encoding="utf-8") if skill_text else ""
        self.family = family

    async def execute(self, environment: Any, command: str) -> tuple[int, str]:
        try:
            return await super().execute(environment, command)
        except RuntimeError as error:  # (this Harbor's timeout; ContextBench expects asyncio's, and goes on)
            if "timed out" not in str(error):
                raise
            return 124, self._format_result(124, "", f"command timed out after {self.command_timeout}s")

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self._model, "messages": messages, "max_tokens": self.max_tokens}
        if tools is None or tools:
            params["tools"] = tools or [BASH_TOOL]
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._get_client().chat.completions.create(**params)
                break
            except openai.BadRequestError:
                raise
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                    openai.RateLimitError):
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        self._n_llm_calls += 1
        self._usage["prompt_tokens"] += response.usage.prompt_tokens
        self._usage["completion_tokens"] += response.usage.completion_tokens
        message = response.choices[0].message
        out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if calls := [c.model_dump() for c in (message.tool_calls or [])]:
            out["tool_calls"] = calls
        return out

    def _finish_run(self, context: Any, src: Any, step: int, started: float) -> None:
        self.cb_finish(None, self.messages)
        (self.logs_dir / "usage.json").write_text(json.dumps(
            {**self._usage, "n_llm_calls": self._n_llm_calls, "n_commands": self._n_commands, "n_turns": step,
             "peak_context_tokens": self.cb_peak_tokens, "wall_s": round(time.monotonic() - started, 1),
             "terminated_by": src.terminated_by, "budget": self.budget, "reserve": self.reserve}, indent=2) + "\n")
        context.n_input_tokens = self._usage["prompt_tokens"]
        context.n_output_tokens = self._usage["completion_tokens"]
        context.metadata = {"n_turns": step, "terminated_by": src.terminated_by, "n_ops_delivered": src.n_delivered,
                            "n_ops_total": src.n_total, "peak_context_tokens": self.cb_peak_tokens}

class PaperAgent(_ChatAgent):
    """The paper's harness modes on ContextBench's loop, as their skills describe them (the harness itself
    is not released): every bash result ends with "[context: ~N/M tokens]";

    - base: append-only.
    - acm: tools manage_context and query_memory. manage_context replaces everything from the previous call
      (at first, from after the task) through the calling turn with one message "[summary_id: N] <summary>",
      written by a separate call with ACM's summarizer prompt over the messages as "[role] text" blocks; the
      originals are kept on the host (not in the sandbox); query_memory answers from them with ACM's recall
      prompt. Summaries stay.
    - self_compact: past 37% of the budget, a probe (Self-Compact's rubric, a user message) every second
      command, in place of what that command would have released (the model answers, then runs the marker
      again, as its skill says); the model answers in text, and Y/Y/Y/N fires the summarizer (Self-Compact's, with the question,
      its <summary> taken), after which the context is the system prompt, the task, the summary as the
      assistant's ("Here is my research so far:") and Self-Compact's continue message.
    - folding: tool branch(description, prompt) runs a sub-agent on a copy of the context (tools bash and
      return, FoldAgent's branch message), whose turns fold away: the call's result is "Branch <description>
      finished. Returned summary: <message>". Operations go to the main context only; a marker run in a
      branch releases nothing; only the main context is snapshot. No compaction behind it.
    The budget wall is ContextBenchAgent's."""

    def __init__(self, *args: Any, mode: str = "base", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.boundary, self.summaries, self.since, self.probing = 2, 0, 0, False

    def _gauge(self, messages: list[dict[str, Any]], content: str) -> str:
        tokens, _ = count_tokens([*messages, {"role": "tool", "content": content}])
        return f"{content}\n[context: ~{tokens}/{self.budget} tokens]"

    def _tools(self) -> list[dict[str, Any]]:
        return {"acm": [BASH_TOOL, MANAGE_TOOL, QUERY_TOOL], "folding": [BASH_TOOL, BRANCH_TOOL]}.get(self.mode, [BASH_TOOL])

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        src = await self.cb_setup(environment, budget=self.budget, reserve=self.reserve)
        self.task = with_skill(instruction, self.skill_text, self.family)
        if self.mode == "rlm":
            return await self._run_rlm(src, context)
        self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.task}]
        self.messages += self.cb_user_turns(self.cb_initial_turns())
        step, started = 0, time.monotonic()
        try:
            while not self.cb_done:
                if step >= self.max_steps:
                    self.cb_finish("max_steps")
                    break
                if self.cb_over_budget(self.messages):
                    break
                try:
                    message = await self.complete(self.messages, self._tools())
                except openai.BadRequestError as exc:
                    if any(w in str(exc).lower() for w in ("context", "too long", "maximum")):
                        self.cb_finish("budget_wall")
                        break
                    raise
                step += 1
                calls = message.get("tool_calls") or []
                if self.probing:  # (Self-Compact: this reply answers the probe)
                    self.probing = False
                    if not calls:
                        self.messages.append(message)
                        if fires(message.get("content") or ""):
                            await self._self_compact()
                        continue
                if not calls:
                    message.pop("tool_calls", None)
                    self.messages.append(message)
                    note = (_proto.EXPLAIN_NON_STANDALONE_TEXT if _proto.mentions_advance_marker(message.get("content") or "")
                            else NO_TOOL_CALL_TEXT)
                    self.messages.append({"role": "user", "content": note})
                    continue
                call = calls[0]
                message["tool_calls"] = [call]
                self.messages.append(message)
                function = call["function"]
                try:
                    args = json.loads(function["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                name, command, rc = function["name"], str(args.get("command") or ""), 0
                if name == "bash" and command:
                    rc, content = await self.execute(environment, command)
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "content": self._gauge(self.messages, content)})
                    self.since += 1
                    if (self.mode == "self_compact" and self.since >= 2
                            and count_tokens(self.messages)[0] > 0.37 * self.budget):
                        # The probe comes in place of what this step would release: the model answers
                        # it, then runs the marker again to resume the stream (its skill).
                        self.messages.append({"role": "user", "content": SC_RUBRIC})
                        self.probing, self.since = True, 0
                    else:
                        self.messages += self.cb_user_turns(self.cb_pending(command, rc, turn=step))
                elif name == "manage_context" and self.mode == "acm":
                    await self._manage(len(self.messages) - 1, call)
                elif name == "query_memory" and self.mode == "acm":
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "content": await self._recall(args)})
                elif name == "branch" and self.mode == "folding":
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "content": await self._branch(environment, args)})
                else:
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content":
                                          f"error: no tool `{name}`; call one of: "
                                          + ", ".join(t["function"]["name"] for t in self._tools())})
        except asyncio.CancelledError:
            self.cb_finish("timeout")
            raise
        finally:
            self._finish_run(context, src, step, started)
            self._publish()

    async def _run_rlm(self, src: Any, context: Any) -> None:
        """RLM as the paper's harness runs it (rlm_hh, its skill): the authors' package (`rlms`) with its
        Docker REPL, system prompt and per-turn prompt ("Turn i/N:"), its ```repl``` blocks executed and their
        outputs (20,000 characters each at most) returned as the next user message, as `RLM.completion`
        does. One conversation for the whole stream: the first operation is `context`, every later one is
        appended to it; the model submits a reply with `answer["content"]` and `answer["ready"] = True`, the
        reply goes into the transcript as the assistant's, and the next operation is appended. No bash."""

        from rlm.clients import get_client
        from rlm.core.lm_handler import LMHandler
        from rlm.core.types import CodeBlock, QueryMetadata, RLMIteration
        from rlm.environments.docker_repl import DockerREPL
        from rlm.utils.parsing import find_code_blocks, format_iteration
        from rlm.utils.prompts import RLM_SYSTEM_PROMPT, build_rlm_system_prompt, build_user_prompt

        handler = LMHandler(get_client("openai", {"model_name": self._model, "base_url": self.api_base,
                                                  "api_key": self.api_key}))
        handler.start()
        stream = "\n".join(self.cb_initial_turns())
        repl = await asyncio.to_thread(DockerREPL, image="python:3.12-slim", persistent=True,
                                       lm_handler_address=(handler.host, handler.port), context_payload=stream)
        self.messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, QueryMetadata(stream), root_prompt=self.task)
        step, turn, started = 0, 0, time.monotonic()
        try:
            while not self.cb_done:
                if step >= self.max_steps:
                    self.cb_finish("max_steps")
                    break
                self.messages.append(build_user_prompt(None, turn))
                if self.cb_over_budget(self.messages):
                    break
                try:
                    response = (await self.complete(self.messages, []))["content"] or ""
                except openai.BadRequestError as exc:
                    if any(w in str(exc).lower() for w in ("context", "too long", "maximum")):
                        self.cb_finish("budget_wall")
                        break
                    raise
                step, turn = step + 1, turn + 1
                blocks = [CodeBlock(code=code, result=await asyncio.to_thread(repl.execute_code, code))
                          for code in find_code_blocks(response)]
                self.messages += format_iteration(RLMIteration(prompt=None, response=response, code_blocks=blocks))
                reply = next((b.result.final_answer for b in blocks if b.result.final_answer is not None), None)
                if reply is None:
                    continue
                self.messages.append({"role": "assistant", "content": reply})
                new = "".join(f"\n{text}" for text in self.cb_pending("echo READY_FOR_NEXT_OP", 0, turn=step))
                await asyncio.to_thread(repl.execute_code, 'answer["content"] = ""; answer["ready"] = False\n'
                                        f"context_0 = context_0 + {new!r}; context = context_0")
                turn = 0
        except asyncio.CancelledError:
            self.cb_finish("timeout")
            raise
        finally:
            await asyncio.to_thread(repl.cleanup)
            handler.stop()
            self._finish_run(context, src, step, started)
            self._publish()

    async def _self_compact(self) -> None:
        answer = (await self.complete([*self.messages, {"role": "user", "content": SC_SUMMARIZER
                                       + SC_QUESTION.format(question=self.task)}], []))["content"]
        found = re.search(r"<summary>\s*(.*?)\s*</summary>", THINK.sub("", answer or ""), re.S | re.I)
        summary = (found.group(1) if found else THINK.sub("", answer or "")).strip()
        if summary:
            self.cb_snapshot(list(self.messages), "turn")
            self.messages = [*self.messages[:2], {"role": "assistant", "content": SC_RESEARCH + summary},
                             {"role": "user", "content": SC_CONTINUE}]

    async def _manage(self, at: int, call: dict[str, Any]) -> None:
        dropped = self.messages[self.boundary:at + 1]
        if len(dropped) <= 1:
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content":
                                  "Error: nothing to compress. There are no messages between your previous "
                                  "manage_context call and this one."})
            return
        number = self.summaries + 1
        originals = [{"role": m.get("role"), "content": render_message(m)} for m in dropped]
        (self.logs_dir.parent / "acm").mkdir(exist_ok=True)
        (self.logs_dir.parent / "acm" / f"summary_{number}.json").write_text(json.dumps(originals, indent=2))
        conversation = "\n\n".join(f"[{m['role']}] {m['content']}" for m in originals)
        answer = THINK.sub("", (await self.complete([{"role": "user", "content": ACM_SUMMARY.format(
            question=self.task, conversation=conversation)}], []))["content"] or "").strip()
        memory = re.search(r"<memory>(.*?)</memory>", answer, re.S | re.I)
        text = memory.group(1).strip() if memory and len(memory.group(1).strip()) >= 50 else answer
        self.cb_snapshot(list(self.messages), "turn")
        self.messages[self.boundary:at + 1] = [{"role": "user", "content": f"[summary_id: {number}] {text}"}]
        self.boundary += 1
        self.summaries = number

    async def _recall(self, args: dict[str, Any]) -> str:
        number, query = args.get("summary_id"), str(args.get("query") or "").strip()
        path = self.logs_dir.parent / "acm" / f"summary_{number}.json"
        if not query or not path.exists():
            return f"Error: summary_id {number} not found." if query else "Error: missing required parameters (summary_id, query)."
        history = "\n".join(f"[{m['role']}] {m['content']}" for m in json.loads(path.read_text()))
        answer = (await self.complete([{"role": "user", "content": ACM_RECALL.format(
            summary_id=number, history=history, query=query)}], []))["content"]
        return f"[query_memory: summary_id={number}]\n\n{THINK.sub('', answer or '').strip()}"

    async def _branch(self, environment: Any, args: dict[str, Any]) -> str:
        description, prompt = str(args.get("description") or ""), str(args.get("prompt") or "")
        branch = [*self.messages[:-1], {"role": "user", "content": f"{prompt}\n\n{FOLD_BRANCH}"}]
        usable, warned, said = self.budget - self.reserve, False, ""
        for _ in range(BRANCH_STEPS):
            if count_tokens(branch)[0] > usable:
                if warned:
                    break
                branch.append({"role": "user", "content": FOLD_OVER})
                warned = True
            message = await self.complete(branch, [BASH_TOOL, RETURN_TOOL])
            said = message.get("content") or said
            calls = message.get("tool_calls") or []
            if not calls:
                branch += [message, {"role": "user", "content": "Use the return tool to return to MAIN."}]
                continue
            call = calls[0]
            message["tool_calls"] = [call]
            branch.append(message)
            function = call["function"]
            try:
                inner = json.loads(function["arguments"] or "{}")
            except json.JSONDecodeError:
                inner = {}
            if function["name"] == "return":
                said = str(inner.get("message") or said)
                break
            command = str(inner.get("command") or "")
            content = (await self.execute(environment, command))[1] if function["name"] == "bash" and command \
                else "error: call bash or return"
            branch.append({"role": "tool", "tool_call_id": call["id"], "content": self._gauge(branch, content)})
        return f"Branch {description} finished. Returned summary: {said or '(nothing)'}"

class RelayChatAgent(_ChatAgent):
    """ContextBenchAgent through a Relay: the snapshots and the budget wall are what Relay forwarded (its
    `trace`); each bash result ends with the readout of the last request Relay sent, as the paper's harness
    gave every arm one, except where the strategy gives its own (ACM, CLM: `gauge=off`)."""

    def __init__(self, *args: Any, relay_trace: str = "", gauge: str = "on", settle: str = "off", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.relay_trace, self.gauge, self.settle = Path(relay_trace), gauge == "on", settle == "on"
        self.ending: tuple[str | None, list[Any] | None] | None = None
        self.trace_at, self.sent_tokens, self.walled = 0, 0, False
        self.held: tuple[list[dict[str, Any]], int] = ([], 0)  # (Relay's last forwarded messages, of how many of ours)

    def cb_over_budget(self, messages: list[Any]) -> bool:
        return self.walled  # (checked on what Relay sent, after each request)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        message = await super().complete(messages, tools)
        usable = self.budget - self.reserve
        for sent, ours in self._forwarded():
            self.held = (sent, ours)
            self.sent_tokens = self.cb_snapshot(sent, "sent")
            if self.sent_tokens > usable and not self.walled:
                self.walled = True
                self.cb_source.terminate("budget_wall")
        return message

    def _forwarded(self) -> list[list[dict[str, Any]]]:
        if not self.relay_trace.exists():
            return []
        with self.relay_trace.open("rb") as stream:
            stream.seek(self.trace_at)
            data = stream.read()
        end = data.rfind(b"\n") + 1
        self.trace_at += end
        return [((record.get("forwarded") or record["body"]).get("messages") or [], len(record["body"].get("messages") or []))
                for line in data[:end].splitlines() if "items" in (record := json.loads(line))
                and isinstance(record.get("body"), dict)]

    def cb_finish(self, reason: str | None = None, messages: list[Any] | None = None) -> None:
        # The final context is what Relay holds: its last forwarded request, then what came after it here
        # (the last reply, its results), not this loop's own unmanaged list. With `settle` (CLM) the end waits
        # for `run`: the model's last edit to its context file is applied only by the next request.
        if self.settle and self.ending is None:
            self.ending = (reason, None if messages is None else list(messages))
            return
        sent, ours = self.held
        if messages is not None and ours:
            messages = [*sent, *messages[ours:]]
        super().cb_finish(reason, messages)

    async def execute(self, environment: Any, command: str) -> tuple[int, str]:
        rc, content = await super().execute(environment, command)
        if self.gauge:
            tokens = self.sent_tokens + count_tokens([{"role": "tool", "content": content}])[0]
            content = f"{content}\n[context: ~{tokens}/{self.budget} tokens]"
        return rc, content

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        try:
            await super().run(with_skill(instruction, self.skill_text, self.family), environment, context)
        finally:
            if self.ending is not None:
                await self._settle(*self.ending)
            self._publish()

    async def _settle(self, reason: str | None, messages: list[Any] | None) -> None:
        """One more request through Relay, its answer unused, so the final context is the one Relay would send
        next: CLM's original harness syncs the file after every command, the last one included (ClmAgent's
        final snapshot), where Relay reads the file when the next request comes. Not after a budget wall."""
        if messages and not self.walled and messages[-1].get("role") != "assistant":
            await self.complete(messages)
        self.settle = False
        self.cb_finish(reason, messages)
