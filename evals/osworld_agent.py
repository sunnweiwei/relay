"""The OSWorld agent osworld.py runs: OSWorld's Qwen agent over native function calling, and the context-management
methods' own harnesses on it. Run by osworld.py in the OSWorld checkout's environment, as OSWorld's
`scripts/python/run_multienv_qwen.py` with this agent in place of QwenAgent (same options):

    cd OSWORLD && PYTHONPATH=RELAY/evals .venv/bin/python RELAY/evals/osworld_agent.py [run_multienv_qwen options]

Configuration comes from the environment (the runner forks its workers, which inherit it):

    OSW_MODE      unset: the scaffold as it is (its own screenshot folding); else a MethodClient mode (below)
    OSW_STRATEGY  for `relay`: Relay's strategy (which tools the model gets: ACM's and folding's echo shell, CLM's bash)
    OSW_BUDGET    the context budget in tokens (the wall is the budget less 2,048, the model's own count)
    OSW_LOG_DIR   where method_log.json goes (the method's calls, the wall)
    OSW_CTX_DIR, OSW_SANDBOX   relay with Relay's clm: the directory Relay writes the context file to, the sandbox

## Function calling

The Qwen scaffold writes `computer_use` calls as XML text and reads them back from the model's text.
Models served through APIs that only act through declared tools (gpt-5.6-sol on the ChatGPT backend
answers the XML format with an apology and no action) get the same scaffold here with the wire format
changed and nothing else: the agent keeps its XML internally (parsing, translation to pyautogui,
screenshot folding, summaries, terminal judging stay QwenAgent's), and `FunctionCallClient` converts
each request on its way out and each reply on its way in:

- the system prompt's `<tools>` block and XML instructions become one declared `computer_use` function;
- an assistant turn's XML calls become its `tool_calls`, and the `<tool_response>` parts that QwenAgent
  puts in the next user turn become one `tool` message per call, the observation (screenshot) in the
  last one; a call left without a response (a user answer or a recovery prompt comes next) gets
  "success", as QwenAgent's own filler says;
- the reply's tool calls are written back as XML blocks after its text.

Requests whose system prompt declares no tools (summaries, the terminal judge) go out unchanged.

## The methods (MethodClient)

`MethodClient` stands between FunctionCallQwenAgent and the model. The agent sends its whole history every
step (every screenshot: `--image_max` and `--history_n` out of reach, no context rescue); the client keeps a
transcript of its own, appends what is new in each request (the results of the last GUI calls: their
screenshot), applies its method, and asks the model until the answer holds GUI actions, which go back to the
agent. Non-GUI tools (manage_context, query_memory, branch, return, the echo shell) are answered here and never
reach the agent. Modes, after the paper modes on ContextBench (context_bench_agent.py's PaperAgent, whose prompts
and tools they share: osworld.paper_modes):

- relay: the transcript is sent as is; a Relay in front of the model manages it (its strategy). An echo-only
  `bash` tool is offered when the strategy signals through the shell (ACM, folding): `echo "[manage_context]"`
  answers with its text, which Relay rewrites; any other command answers that there is no terminal.
- none: append-only (the budget wall ends the run).
- self_compact: past 37% of the budget, every second GUI step, Self-Compact's rubric as a user message; the
  model answers in text, and C1=C2=C3=Y, N1=N fires the summarizer (Self-Compact's, with the question, its
  <summary> taken); the transcript becomes system, task, the summary as the assistant's ("Here is my research so
  far:") and the continue message, with the current screenshot.
- acm: tools manage_context and query_memory (ACM's descriptions and prompts). manage_context replaces everything
  from the previous call (at first, from after the task) through the calling turn with "[summary_id: N] ...";
  originals kept here, query_memory answers from them with ACM's recall prompt, as text (ACM's release is text
  only, so images stand as placeholders).
- folding: tools branch and return (FoldAgent's). branch answers with FoldAgent's branch message and the steps
  that follow belong to the branch; return folds them: the branch call's result becomes "Branch <description>
  finished. Returned summary: <message>". A branch past the budget is told to return (FoldAgent's message).
- relay with Relay's `clm`: a `bash` tool in a sandbox container holding only the directory Relay writes the
  context file to (no network), since the scaffold has no shell of its own for the model to edit it with.

Every arm: each GUI tool result ends with "[context: ~N/M tokens]" (the paper's readout) except where the method
gives its own (relay ACM, CLM); the wall is the model's own count of the request (its reported prompt tokens):
past budget - reserve, the run ends (the agent gets a terminate action).
"""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from mm_agents.qwen import QwenAgent
from mm_agents.qwen.client import Completion, QwenClient, _field, _text
from mm_agents.qwen.parser import parse_tool_calls

from osworld import paper_modes  # (evals/ on PYTHONPATH, as osworld.py runs this)

(ACM_RECALL, ACM_SUMMARY, BRANCH_TOOL, FOLD_BRANCH, FOLD_OVER, MANAGE_TOOL, QUERY_TOOL, RETURN_TOOL,
 SC_CONTINUE, SC_QUESTION, SC_RESEARCH, SC_RUBRIC, SC_SUMMARIZER, fires) = paper_modes(
    "ACM_RECALL", "ACM_SUMMARY", "BRANCH_TOOL", "FOLD_BRANCH", "FOLD_OVER", "MANAGE_TOOL", "QUERY_TOOL", "RETURN_TOOL",
    "SC_CONTINUE", "SC_QUESTION", "SC_RESEARCH", "SC_RUBRIC", "SC_SUMMARIZER", "fires")

TOOLS = re.compile(r"# Tool\s*<tools>\s*(\{.*?\})\s*</tools>\s*", re.S)
XML_FORMAT = re.compile(r"When using the tool, emit complete XML blocks.*?</tool_call>\s*", re.S)
FUNCTION_NOTE = "Act only through the `computer_use` function. Each call performs one action.\n\n"
WIRE_RULES = {  # (the XML rules, said of function calls)
    "- Required parameters must be present and no text may follow a tool call.\n": "- Required parameters must be present.\n",
    "- Optional reasoning may appear before the first tool call.\n": "- Optional brief reasoning may accompany the calls.\n",
    "- Return each action as a separate complete <tool_call>...</tool_call> block; do not return an actions array.\n":
        "- Return each action as a separate computer_use call; do not return an actions array.\n",
}
RESPONSE_OPEN, RESPONSE_CLOSE = "<tool_response>\n", "\n</tool_response>"


_JPEG: dict[int, str] = {}


def _jpeg(url: str) -> str:
    """A PNG data URL as a JPEG one (same pixels' size, so the same image tokens; ~10x fewer bytes: a full
    history carries every screenshot, about 1.3 MB each as PNG)."""

    if not url.startswith("data:image/png;base64,"):
        return url
    key = hash(url)
    if key not in _JPEG:
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        _JPEG[key] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        if len(_JPEG) > 2000:
            _JPEG.pop(next(iter(_JPEG)))
    return _JPEG[key]


def _images_to_jpeg(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    return [{**p, "image_url": {**p["image_url"], "url": _jpeg(p["image_url"]["url"])}} if p.get("type") == "image_url" else p
            for p in content]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content or [] if p.get("type") == "text")


def _xml(arguments: dict[str, Any]) -> str:
    lines = ["<tool_call>", "<function=computer_use>"]
    for name, value in arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"<parameter={name}>{rendered}</parameter>")
    return "\n".join([*lines, "</function>", "</tool_call>"])


def _responses(content: Any) -> list[list[dict[str, Any]]] | None:
    """The tool responses of a user turn QwenAgent built with `_tool_response`, each as its parts; None if it is not one."""

    if not isinstance(content, list) or not content or content[0].get("text") != RESPONSE_OPEN:
        return None
    out: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for part in content:
        text = part.get("text") if part.get("type") == "text" else None
        if text == RESPONSE_OPEN:
            current = []
        elif text in (RESPONSE_CLOSE, RESPONSE_CLOSE + "\n"):
            out.append(current or [])
            current = None
        elif current is not None:
            current.append(part)
    return out


def to_wire(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """QwenAgent's messages as function-calling messages, and the declared tool (None: no tools declared)."""

    system = _text_of(messages[0].get("content")) if messages and messages[0].get("role") == "system" else ""
    found = TOOLS.search(system)
    if not found:
        return messages, None
    tool = json.loads(found.group(1))
    prompt = XML_FORMAT.sub(FUNCTION_NOTE, TOOLS.sub("", system), count=1)
    for old, new in WIRE_RULES.items():
        prompt = prompt.replace(old, new)
    out: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
    pending: list[str] = []  # the last assistant turn's call ids still waiting for a response
    for index, message in enumerate(messages[1:], start=1):
        role, content = message.get("role"), message.get("content")
        if role == "assistant":
            text = _text_of(content)
            calls = parse_tool_calls(text)
            prose = text[: calls[0].start].strip() if calls else text.strip()
            ids = [f"call_{index}_{n}" for n in range(len(calls))]
            wire: dict[str, Any] = {"role": "assistant", "content": prose or None}
            if calls:
                wire["tool_calls"] = [{"id": i, "type": "function", "function": {
                    "name": "computer_use", "arguments": json.dumps(c.parameters, ensure_ascii=False)}}
                    for i, c in zip(ids, calls)]
            out.append(wire)
            pending = ids
            continue
        responses = _responses(content) if role == "user" else None
        if responses is not None and pending:
            for n, call_id in enumerate(pending):
                parts = responses[n] if n < len(responses) else [{"type": "text", "text": "success"}]
                # (a tool message's images: an endpoint on the Responses API takes them as the call's output)
                body: Any = parts if any(p.get("type") == "image_url" for p in parts) else _text_of(parts) or "success"
                out.append({"role": "tool", "tool_call_id": call_id, "content": body})
            pending = []
            continue
        for call_id in pending:  # (answered by something else: a user reply, a recovery prompt)
            out.append({"role": "tool", "tool_call_id": call_id, "content": "success"})
        pending = []
        out.append({"role": role, "content": content})
    for call_id in pending:
        out.append({"role": "tool", "tool_call_id": call_id, "content": "success"})
    return [{**m, "content": _images_to_jpeg(m.get("content"))} for m in out], tool


class FunctionCallClient(QwenClient):
    """QwenClient sending declared tools and reading tool calls (no streaming)."""

    def complete(self, payload: dict[str, Any]) -> Completion:
        messages, tool = to_wire(payload["messages"])
        if tool is None:
            return super().complete({**payload, "stream": False})
        retryable = self._retryable_exceptions()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._openai_client().chat.completions.create(
                    model=payload.get("model", self.model), messages=messages, tools=[tool],
                    max_tokens=payload.get("max_tokens", 32768))
                message = response.choices[0].message
                calls = []
                for call in _field(message, "tool_calls") or []:
                    function = _field(call, "function")
                    try:
                        arguments = json.loads(_field(function, "arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    if _field(function, "name") == "computer_use" and isinstance(arguments, dict):
                        calls.append(_xml(arguments))
                text = _text(_field(message, "content")).strip()
                return Completion("", "\n".join(filter(None, [text, *calls])))
            except retryable as exc:
                last_error = exc
                self.logger.warning("API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(min(self.retry_sleep_cap, self.retry_sleep_base * max(1, attempt)))
        raise last_error


class FunctionCallQwenAgent(QwenAgent):
    """QwenAgent whose client speaks native function calling."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        old = self.client
        mode = os.environ.get("OSW_MODE", "")
        if mode:
            self.client = MethodClient(
                model=old.model, base_url=old.base_url, api_key=old.api_key, stream=False,
                max_retries=old.max_retries, retry_sleep_base=old.retry_sleep_base, retry_sleep_cap=old.retry_sleep_cap,
                request_timeout=old.request_timeout, generation_timeout=old.generation_timeout, top_k=old.top_k,
                logger=old.logger, mode=mode, strategy=os.environ.get("OSW_STRATEGY", ""),
                budget=int(os.environ["OSW_BUDGET"]), log_dir=os.environ["OSW_LOG_DIR"])
            return
        self.client = FunctionCallClient(
            model=old.model, base_url=old.base_url, api_key=old.api_key, stream=False,
            max_retries=old.max_retries, retry_sleep_base=old.retry_sleep_base, retry_sleep_cap=old.retry_sleep_cap,
            request_timeout=old.request_timeout, generation_timeout=old.generation_timeout, top_k=old.top_k,
            logger=old.logger)


BASH_REAL = {"type": "function", "function": {
    "name": "bash", "description": "Run one bash command in a sandbox shell (it holds your context file; it cannot "
    "see or control the desktop). Use it only to manage your context.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
THINK = re.compile(r"<think>.*?</think>", re.S)
BASH_ECHO = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command. Only `echo` is available here (there is no terminal).",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
NO_TERMINAL = "error: there is no terminal; only `echo` runs here. Act on the desktop with computer_use."
IMAGE_TOKENS = 2786  # (Relay's estimate of a 1920x1088 screenshot; the model counts ~2476)
INNER_LIMIT = 8  # non-GUI rounds in one step before the step gives up


def tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                total += IMAGE_TOKENS if p.get("type") == "image_url" else math.ceil(len(str(p.get("text") or "")) / 4)
        else:
            total += math.ceil(len(str(content or "")) / 4)
        total += math.ceil(len(json.dumps(m.get("tool_calls") or [])) / 4) + 4
    return total


def render(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join("[screenshot]" if p.get("type") == "image_url" else str(p.get("text") or "") for p in content)
    calls = " ".join(f"{c['function']['name']}({c['function']['arguments']})"
                     for c in message.get("tool_calls") or [])
    return (str(content or "") + (f" [tool_call] {calls}" if calls else "")).strip()


class MethodClient(FunctionCallClient):
    def __init__(self, *args: Any, mode: str = "none", strategy: str = "", budget: int = 65536, reserve: int = 2048,
                 log_dir: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mode, self.strategy, self.budget, self.reserve = mode, strategy, budget, reserve
        self.log_dir = Path(log_dir)
        self.reset_state()

    def reset_state(self) -> None:
        self.transcript: list[dict[str, Any]] = []
        self.seen = 0  # agent wire messages already in the transcript
        self.task = ""
        self.walled = False
        self.peak = 0
        self.since, self.summaries, self.boundary = 0, 0, 2
        self.originals: dict[int, list[dict[str, str]]] = {}
        self.branch: tuple[int, str] | None = None  # (index of the branch call's result, description)
        self.warned = False  # (FoldAgent's over-budget message, once a branch)
        self.log: list[dict[str, Any]] = []

    def _tools(self, computer: dict[str, Any]) -> list[dict[str, Any]]:
        extra = {"acm": [MANAGE_TOOL, QUERY_TOOL], "folding": [BRANCH_TOOL, *([RETURN_TOOL] if self.branch else [])]}
        tools = [computer, *extra.get(self.mode, [])]
        if self.mode == "relay" and self.strategy in ("acm", "folding"):
            tools.append(BASH_ECHO)
        if self.mode == "relay" and self.strategy == "clm":
            tools.append(BASH_REAL)
        return tools

    def _sandbox(self, command: str) -> str:
        """One command in the trial's sandbox container (OSW_SANDBOX), the context directory its cwd."""

        try:
            done = subprocess.run(["docker", "exec", "-w", os.environ["OSW_CTX_DIR"], os.environ["OSW_SANDBOX"],
                                   "bash", "-lc", command],
                                  capture_output=True, text=True, timeout=60)
            out, code = (done.stdout or "") + (done.stderr or ""), done.returncode
        except subprocess.TimeoutExpired:
            out, code = "command timed out after 60s", 124
        if len(out) > 10_000:
            out = out[:5000] + "\n... The output of your last command was too long. ...\n" + out[-5000:]
        return f"exit code {code}\n{out}".strip()

    def _call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> Any:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._openai_client().chat.completions.create(**kwargs)
            except self._retryable_exceptions() as exc:
                last = exc
                self.logger.warning("API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(min(self.retry_sleep_cap, self.retry_sleep_base * max(1, attempt)))
        raise last

    def _text(self, messages: list[dict[str, Any]], max_tokens: int = 4096) -> str:
        response = self._call(messages, None, max_tokens)
        return THINK.sub("", response.choices[0].message.content or "").strip()

    def _gauge(self) -> None:
        """The readout after the newest GUI result (the paper's), unless the method has its own."""

        if self.mode == "relay" and self.strategy in ("acm", "clm"):
            return
        for m in reversed(self.transcript):
            if m.get("role") == "tool":
                note = {"type": "text", "text": f"[context: ~{tokens(self.transcript)}/{self.budget} tokens]"}
                m["content"] = [*m["content"], note] if isinstance(m["content"], list) else f"{m['content']}\n{note['text']}"
                return

    def _record(self, kind: str, **data: Any) -> None:
        self.log.append({"kind": kind, "step": self.seen, **data})
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "method_log.json").write_text(json.dumps(self.log, indent=1))

    def complete(self, payload: dict[str, Any]) -> Completion:
        wire, computer = to_wire(payload["messages"])
        if computer is None:  # (summaries, the terminal judge: as they are)
            return super().complete(payload)
        if self.walled:
            return Completion("", _xml({"action": "terminate", "status": "failure"}))
        if not self.transcript or len(wire) < self.seen:  # (a new task)
            self.reset_state()
            self.transcript, self.seen = copy.deepcopy(wire), len(wire)
            self.task = next((m.get("content") for m in wire if m.get("role") == "user"), "")
            self.task = " ".join(str(p.get("text") or "") for p in self.task) if isinstance(self.task, list) else str(self.task)
        else:
            # (the model's last answer comes back as the agent's copy, under the ids its results use)
            self.transcript += copy.deepcopy(wire[self.seen:])
            self.seen = len(wire)
            self.since += 1
        self._gauge()
        if self.mode == "self_compact":
            self._probe()
        if self.mode == "folding" and self.branch and not self.warned and tokens(self.transcript) > self.budget - self.reserve:
            self.transcript.append({"role": "user", "content": FOLD_OVER})
            self.warned = True
        for _ in range(INNER_LIMIT):
            response = self._call(self.transcript, self._tools(computer), payload.get("max_tokens", 32768))
            used = response.usage.prompt_tokens
            self.peak = max(self.peak, used)
            if used > self.budget - self.reserve:
                self.walled = True
                self._record("budget_wall", prompt_tokens=used)
                return Completion("", _xml({"action": "terminate", "status": "failure"}))
            message = response.choices[0].message
            text = (message.content or "").strip()
            calls = [c.model_dump() for c in message.tool_calls or []]
            gui = [c for c in calls if c["function"]["name"] == "computer_use"]
            other = [c for c in calls if c["function"]["name"] != "computer_use"]
            if other:  # one non-GUI call at a time, answered here
                call = other[0]
                self.transcript.append({"role": "assistant", "content": text or None, "tool_calls": [call]})
                self._answer(call)
                continue
            # (not kept here: the agent rebuilds this answer from its XML and sends it back with its results)
            args = []
            for c in gui:
                try:
                    a = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    a = {}
                if isinstance(a, dict):
                    args.append(_xml(a))
            return Completion("", "\n".join(filter(None, [text, *args])))
        self._record("inner_limit")
        return Completion("", "")

    def _answer(self, call: dict[str, Any]) -> None:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        result = f"error: no tool `{name}`"
        if name == "bash" and any(t is BASH_REAL for t in self._tools({})):
            command = str(args.get("command") or "")
            result = self._sandbox(command)
        elif name == "bash":
            command = str(args.get("command") or "").strip()
            found = re.fullmatch(r"""echo\s+(["']?)(.*)\1""", command, re.S)
            result = found.group(2) if found else NO_TERMINAL
        elif name == "manage_context" and self.mode == "acm":
            result = self._manage(len(self.transcript) - 1)
            if result is None:
                return
        elif name == "query_memory" and self.mode == "acm":
            result = self._recall(args)
        elif name == "branch" and self.mode == "folding":
            if self.branch:
                result = "You are in branch mode and cannot branch task or finish the task. Use return to go back to the main agent."
            else:
                description = str(args.get("description") or "")
                self.transcript.append({"role": "tool", "tool_call_id": call["id"], "content": FOLD_BRANCH.split("\n\nWhen you have")[0]
                                        + f"\n\nAssigned task ({description}):\n{args.get('prompt') or ''}"})
                self.branch = (len(self.transcript) - 1, description)
                self._record("branch", description=description)
                return
        elif name == "return" and self.mode == "folding" and self.branch:
            at, description = self.branch
            message = str(args.get("message") or "")
            del self.transcript[at + 1:]
            self.transcript[at]["content"] = f"Branch {description} finished. Returned summary: {message or '(nothing)'}"
            self.branch, self.warned = None, False
            self._record("return", description=description, chars=len(message))
            return
        self.transcript.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    def _manage(self, at: int) -> str | None:
        """ACM: compress from the boundary through the calling turn (at: the call's message)."""

        dropped = self.transcript[self.boundary:at + 1]
        if len(dropped) <= 1:
            return ("Error: nothing to compress. There are no messages between your previous manage_context call "
                    "and this one.")
        number = self.summaries + 1
        originals = [{"role": m.get("role"), "content": render(m)} for m in dropped]
        self.originals[number] = originals
        conversation = "\n\n".join(f"[{m['role']}] {m['content']}" for m in originals)
        answer = self._text([{"role": "user", "content": ACM_SUMMARY.format(question=self.task, conversation=conversation)}])
        memory = re.search(r"<memory>(.*?)</memory>", answer, re.S | re.I)
        text = memory.group(1).strip() if memory and len(memory.group(1).strip()) >= 50 else answer
        self.transcript[self.boundary:at + 1] = [{"role": "user", "content": f"[summary_id: {number}] {text}"}]
        self.boundary += 1
        self.summaries = number
        self._record("manage_context", summary_id=number, dropped=len(dropped), chars=len(text))
        return None

    def _recall(self, args: dict[str, Any]) -> str:
        number, query = args.get("summary_id"), str(args.get("query") or "").strip()
        if not query or number not in self.originals:
            return f"Error: summary_id {number} not found." if query else "Error: missing required parameters (summary_id, query)."
        history = "\n".join(f"[{m['role']}] {m['content']}" for m in self.originals[number])
        answer = self._text([{"role": "user", "content": ACM_RECALL.format(summary_id=number, history=history, query=query)}])
        self._record("query_memory", summary_id=number)
        return f"[query_memory: summary_id={number}]\n\n{answer}"

    def _probe(self) -> None:
        """Self-Compact: the rubric past 37% of the budget, every second step; Y/Y/Y/N summarizes."""

        if self.since < 2 or tokens(self.transcript) <= 0.37 * self.budget:
            return
        self.since = 0
        answer = self._text([*self.transcript, {"role": "user", "content": SC_RUBRIC}], 2048)
        fired = fires(answer)
        self._record("probe", fired=fired)
        if not fired:
            return
        summary = self._text([*self.transcript, {"role": "user", "content": SC_SUMMARIZER + SC_QUESTION.format(question=self.task)}])
        found = re.search(r"<summary>\s*(.*?)\s*</summary>", summary, re.S | re.I)
        summary = (found.group(1) if found else summary).strip()
        if not summary:
            return
        screen = next((p for m in reversed(self.transcript) if isinstance(m.get("content"), list)
                       for p in m["content"] if p.get("type") == "image_url"), None)
        self.transcript = [*self.transcript[:2], {"role": "assistant", "content": SC_RESEARCH + summary},
                           {"role": "user", "content": [{"type": "text", "text": SC_CONTINUE + "\n\nThe current screen:"},
                                                        *([screen] if screen else [])]}]
        self._record("compact", chars=len(summary))


def main() -> int:
    """OSWorld's Qwen runner with this agent: run from the OSWorld checkout."""

    root = Path.cwd()
    sys.path[:0] = [str(root), str(root / "scripts" / "python")]
    import run_multienv_qwen as runner

    import osworld_agent  # (the class from this module by its name, not __main__'s copy, for the forked workers)

    runner.QwenAgent = osworld_agent.FunctionCallQwenAgent  # (_new_agent looks it up here; workers are forked)
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
