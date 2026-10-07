"""Codex's context after compaction, Relay against Codex itself, with the real binary.

Each scenario runs three times against the scripted fake upstream, in the same directories
with a fresh Codex home: without compaction to measure the request sizes, with Codex's own
compaction, and through Relay with Codex's compaction off. The first request after a compaction
in the last two runs must match, ids and the summary text aside (subagents are numbered in the
order they were spawned, and listed in any order: Codex lists them from a hash map). Thresholds
sit between measured sizes, so both compact in the same turn and at the same kind of point
(mid-turn or turn start).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import shutil
import struct
import subprocess
import tempfile
import termios
import threading
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from relay import Compaction, Engine, PrefixStore, ProxyConfig, create_app
from relay.prompts import SUMMARY_PREFIX
from tests.fakes import COMPACTION_MARKER, FakeUpstream, local_upstreams, serve

MODEL, SWITCHED = "gpt-5.1-codex", "gpt-5.5"
COMMAND = "seq 1 6000"  # about 7k tokens of tool output per call
ESCALATE = {"sandbox_permissions": "require_escalated", "justification": "Run seq outside the sandbox?",
            "prefix_rule": ["seq"]}
OFF = 10**9
AGENTS = {"A": "# Project notes (v1)\nKeep answers short.\n", "B": "# Project notes (v2)\nCite file paths.\n"}
ANSI = re.compile(r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")


@dataclass
class Turn:
    prompt: str
    cwd: str = "A"
    model: str = MODEL
    before: Callable[[Path], None] | None = None  # runs in the test root before the turn starts
    env: dict[str, str] = field(default_factory=dict)


def _text(item: dict[str, Any]) -> str:
    content = item.get("content")
    return content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict))


def _tokens(body: dict[str, Any]) -> int:
    return len(json.dumps(body)) // 4  # what the fake reports as prompt tokens


def _conversation(bodies: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    """The model requests of the conversation of `task` (not its summaries or side requests)."""

    return [body for body in bodies if body.get("tools") and any(task in _text(i) for i in body["input"])
            and COMPACTION_MARKER not in _text(body["input"][-1])]


def _compacted(bodies: list[dict[str, Any]], task: str, nth: int = 1) -> list[dict[str, Any]] | None:
    """The input of the first model request after the n-th compaction in the conversation of `task`."""

    summaries = [i for i, body in enumerate(bodies) if COMPACTION_MARKER in _text(body["input"][-1])
                 and any(task in _text(item) for item in body["input"])]
    if len(summaries) < nth:
        return None
    return next((body["input"] for body in _conversation(bodies[summaries[nth - 1] + 1:], task)
                 if any(_text(item).startswith(SUMMARY_PREFIX) for item in body["input"])), None)


SPAWNED = re.compile(r'\{"agent_id":"([^"]+)","nickname":(?:"([^"]*)"|null)\}')


def _normal(items: list[dict[str, Any]], bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spawned: dict[str, str | None] = {}
    for body in bodies:
        for item in body["input"]:
            if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
                spawned.update((m[1], m[2]) for m in SPAWNED.finditer(item["output"]) if m[1] not in spawned)
    text = json.dumps([{k: v for k, v in item.items() if k != "id"} for item in items])
    for n, (agent, nickname) in enumerate(spawned.items()):
        text = text.replace(agent, f"agent-{n}")
        text = re.sub(rf"\b{re.escape(nickname)}\b", f"nickname-{n}", text) if nickname else text
    text = re.sub(r"(<subagents>\\n)(.*?)(\\n  </subagents>)",
                  lambda m: m[1] + "\\n".join(sorted(m[2].split("\\n"))) + m[3], text)
    return [{**item, "content": "<summary>"} if _text(item).startswith(SUMMARY_PREFIX) else item
            for item in json.loads(text)]


def spawn(message: str) -> dict[str, Any]:
    return {"namespace": "multi_agent_v1", "name": "spawn_agent", "input": {"message": message}}


def wait(agent: str) -> dict[str, Any]:
    return {"namespace": "multi_agent_v1", "name": "wait_agent", "input": {"targets": [agent], "timeout_ms": 60000}}


def close(agent: str) -> dict[str, Any]:
    return {"namespace": "multi_agent_v1", "name": "close_agent", "input": {"target": agent}}


def delegating(steps: dict[str, list[Callable[[list[str]], dict[str, Any]]]]):
    """A fake script: in a user turn whose prompt contains a key, its calls, each made from the ids
    of the agents spawned so far, then shell commands; agents (prompts starting "AGENT") answer
    at once."""

    def script(step: int, turns: list[tuple[str, str, int]]) -> dict[str, Any] | str | None:
        prompts = [text for role, text, _ in turns if role == "user" and not text.lstrip().startswith(("<", "#"))]
        if prompts and prompts[0].startswith("AGENT"):
            return "done"
        calls = next((calls for key, calls in steps.items() if prompts and key in prompts[-1]), [])
        agents = [m[1] for role, text, _ in turns if role == "tool" for m in SPAWNED.finditer(text)]
        return calls[step](agents) if step < len(calls) else None

    return script


class Bench:
    """Fixed directories and a Codex home, reset before every run."""

    def __init__(self, root: Path) -> None:
        self.root, self.home = root, root / "home"

    def reset(self, base_url: str, limit: int) -> None:
        for path in (self.home, self.root / "A", self.root / "B"):
            for _ in range(50):  # a Codex process that just exited may still be writing there
                shutil.rmtree(path, ignore_errors=True)
                if not path.exists():
                    break
                time.sleep(0.1)
        self.home.mkdir()
        for name, notes in AGENTS.items():
            work = self.root / name
            work.mkdir()
            (work / "AGENTS.md").write_text(notes)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
        trusted = "".join(f'[projects."{self.root / name}"]\ntrust_level = "trusted"\n' for name in AGENTS)
        (self.home / "config.toml").write_text(
            f'model = "{MODEL}"\nmodel_provider = "test"\nmodel_auto_compact_token_limit = {limit}\n'
            f'[model_providers.test]\nname = "Test"\nbase_url = "{base_url}/v1"\nenv_key = "RELAY_TEST_KEY"\n'
            f'wire_api = "responses"\nsupports_websockets = false\n{trusted}')

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CODEX", "OPENAI"))}
        return {**env, "HOME": str(self.home), "CODEX_HOME": str(self.home), "RELAY_TEST_KEY": "tenant",
                "NO_COLOR": "1", "TERM": "xterm-256color"}


def exec_turns(bench: Bench, fake: FakeUpstream, turns: list[Turn], limits: list[int]) -> list[int]:
    """`codex exec`, then `codex exec resume` for each later turn; request counts after each turn."""

    thread, counts = None, []
    for turn, limit in zip(turns, limits):
        if turn.before:
            turn.before(bench.root)
        options = ["--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
                   "-m", turn.model, "-c", f"model_auto_compact_token_limit={limit}"]
        command = ["codex", "exec", *options, turn.prompt] if thread is None else \
            ["codex", "exec", "resume", *options, thread, turn.prompt]
        result = subprocess.run(command, cwd=bench.root / turn.cwd, env={**bench.env(), **turn.env},
                                stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr[-2000:]
        thread = thread or next(json.loads(line)["thread_id"] for line in result.stdout.splitlines()
                                if '"thread.started"' in line)
        counts.append(len(fake.bodies("/v1/responses")))
    return counts


def app_server_turns(bench: Bench, fake: FakeUpstream, turns: list[Turn], limits: list[int]) -> list[int]:
    """One `codex app-server` thread (on-request, workspace-write); command approvals are
    accepted with the proposed prefix rule, as a user choosing "don't ask again" would."""

    process = subprocess.Popen(["codex", "app-server", "-c", f"model_auto_compact_token_limit={limits[0]}"],
                               cwd=bench.root / turns[0].cwd, env=bench.env(), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    replies: dict[int, dict] = {}
    finished: list[dict] = []
    changed = threading.Condition()

    def send(message: dict) -> None:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def read() -> None:
        for line in process.stdout:
            message = json.loads(line)
            if message.get("method") == "item/commandExecution/requestApproval":
                amendment = message["params"].get("proposedExecpolicyAmendment")
                decision = {"acceptWithExecpolicyAmendment": {"execpolicy_amendment": amendment}} \
                    if amendment else "accept"
                send({"jsonrpc": "2.0", "id": message["id"], "result": {"decision": decision}})
            elif "id" in message and "method" not in message:
                with changed:
                    replies[message["id"]] = message
                    changed.notify_all()
            elif message.get("method") == "turn/completed":
                with changed:
                    finished.append(message)
                    changed.notify_all()

    def call(rid: int, method: str, params: dict) -> dict:
        send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        with changed:
            assert changed.wait_for(lambda: rid in replies, 60), method
        return replies[rid]["result"]

    threading.Thread(target=read, daemon=True).start()
    try:
        call(1, "initialize", {"clientInfo": {"name": "relay-test", "version": "0"}})
        send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        thread = call(2, "thread/start", {"cwd": str(bench.root / turns[0].cwd), "model": turns[0].model,
                                          "approvalPolicy": "on-request", "sandbox": "workspace-write"})["thread"]
        counts = []
        for number, turn in enumerate(turns, start=1):
            if turn.before:
                turn.before(bench.root)
            call(2 + number, "turn/start", {"threadId": thread["id"],
                                            "input": [{"type": "text", "text": turn.prompt, "text_elements": []}]})
            with changed:
                assert changed.wait_for(lambda: len(finished) >= number, 300), f"turn {number}"
            counts.append(len(fake.bodies("/v1/responses")))
        return counts
    finally:
        process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()


def tui_turns(bench: Bench, fake: FakeUpstream, turns: list[Turn], limits: list[int]) -> list[int]:
    """The interactive TUI in a pseudo-terminal: prompts are pasted, approvals answered with `p`
    (approve and don't ask again for the proposed prefix); a turn ends with `task_complete`."""

    fd, terminal = os.openpty()
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 160, 0, 0))
    process = subprocess.Popen(  # setsid -c: the pseudo-terminal becomes Codex's controlling terminal
        ["setsid", "-c", "codex", "-m", turns[0].model, "-s", "workspace-write", "-a", "on-request",
         "-c", f"model_auto_compact_token_limit={limits[0]}"],
        cwd=bench.root / turns[0].cwd, env=bench.env(), stdin=terminal, stdout=terminal, stderr=terminal)
    os.close(terminal)
    screen = ""

    def pump(seconds: float) -> None:
        nonlocal screen
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            try:
                data = os.read(fd, 65536).decode(errors="replace")
            except OSError:
                return
            if "\x1b[6n" in data:  # answer the terminal queries a TUI makes
                os.write(fd, b"\x1b[1;1R")
            if "\x1b[c" in data:
                os.write(fd, b"\x1b[?62;22c")
            for code in ("10", "11"):
                if f"\x1b]{code};?" in data:
                    os.write(fd, f"\x1b]{code};rgb:ffff/ffff/ffff\x1b\\".encode())
            screen = re.sub(r"\s+", " ", screen + ANSI.sub(" ", data))[-20_000:]

    def approve() -> None:
        nonlocal screen
        if "Would you like to run the following command" in screen:
            pump(1.5)  # let the whole overlay render before choosing
            os.write(fd, b"p" if "don't ask again for commands that start with" in screen else b"y")
            screen = ""

    def completed() -> int:
        logs = sorted((bench.home / "sessions").rglob("*.jsonl"))  # by time: the session, then its subagents
        return logs[0].read_text().count('"task_complete"') if logs else 0

    try:
        pump(8)
        counts = []
        for number, turn in enumerate(turns, start=1):
            if turn.before:
                turn.before(bench.root)
            os.write(fd, b"\x1b[200~" + turn.prompt.encode() + b"\x1b[201~")
            pump(1)
            os.write(fd, b"\r")
            deadline = time.monotonic() + 300
            while completed() < number and time.monotonic() < deadline:
                pump(0.5)
                approve()
            assert completed() >= number, f"turn {number} did not finish: {screen[-500:]}"
            counts.append(len(fake.bodies("/v1/responses")))
        return counts
    finally:
        process.kill()
        process.wait()
        os.close(fd)


@unittest.skipUnless(shutil.which("codex"), "codex is not installed")
class CodexContextEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="relay-codex-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.bench = Bench(self.root)

    def run_once(self, driver, turns, fake_args, limits=None, threshold=None):
        """One run; returns the fake's request bodies and the request counts after each turn."""

        fake = FakeUpstream(**fake_args)
        with serve(fake.app) as upstream:
            if threshold is None:
                self.bench.reset(upstream, (limits or [OFF])[0])
                counts = driver(self.bench, fake, turns, limits or [OFF] * len(turns))
            else:
                engine = Engine(Compaction(threshold=threshold, min_gain=0), PrefixStore())
                with serve(create_app(engine, ProxyConfig(upstreams=local_upstreams(upstream)))) as relay:
                    self.bench.reset(relay, OFF)
                    counts = driver(self.bench, fake, turns, [OFF] * len(turns))
        return fake.bodies("/v1/responses"), counts

    def compare(self, driver, turns, fake_args, *, turn_start: bool = False, at: int | None = None, nth: int = 1):
        """Calibrate, run Codex's compaction and Relay's, and compare the first request after the
        n-th compaction: mid-way through turn 2, at its start, or before request `at` of the task."""

        task = turns[0].prompt
        bodies, counts = self.run_once(driver, turns, fake_args)
        if at is not None:
            before, after = (_tokens(body) for body in _conversation(bodies, task)[at - 1 : at + 1])
            threshold = before + 2 * (after - before) // 3
            limits = [threshold] * len(turns)
        else:
            mine = [b for b in bodies if any(task in _text(i) for i in b["input"])]
            last1, first2, second2 = (_tokens(mine[i]) for i in (counts[0] - 1, counts[0], counts[0] + 1))
            if turn_start:  # Codex: compact before turn 2 (per-process limits); Relay: on its first request
                limits, threshold = [OFF, last1 - 1], (last1 + first2) // 2
            else:  # both compact mid-way through turn 2, after its first tool result; Relay's own
                # estimate of a request with large context updates runs a little above the fake's count
                threshold = first2 + 2 * (second2 - first2) // 3
                limits = [threshold] * len(turns)
        native, _ = self.run_once(driver, turns, fake_args, limits=limits)
        relayed, _ = self.run_once(driver, turns, fake_args, threshold=threshold)
        codex_view, relay_view = _compacted(native, task, nth), _compacted(relayed, task, nth)
        self.assertIsNotNone(codex_view, "Codex did not compact")
        self.assertIsNotNone(relay_view, "Relay did not compact")
        self.assertEqual(_normal(relay_view, relayed), _normal(codex_view, native))
        return relay_view, relayed

    def directory_agents_and_model_turns(self) -> list[Turn]:
        return [Turn("List the files.", cwd="A"), Turn("List them again.", cwd="B", model=SWITCHED)]

    def test_exec_resume_after_changing_directory_agents_md_and_model_mid_turn(self) -> None:
        view, _ = self.compare(exec_turns, self.directory_agents_and_model_turns(),
                               {"tool_calls": 5, "command": COMMAND})
        text = json.dumps(view)
        self.assertIn("Project notes (v2)", text)
        self.assertNotIn("replace all previously provided AGENTS.md", text)
        self.assertNotIn("<model_switch>", text)

    def test_exec_resume_after_changing_directory_agents_md_and_model_at_turn_start(self) -> None:
        view, _ = self.compare(exec_turns, self.directory_agents_and_model_turns(),
                               {"tool_calls": 5, "command": COMMAND}, turn_start=True)
        text = json.dumps(view)
        self.assertIn("Project notes (v2)", text)
        self.assertIn("<model_switch>", text)

    def test_exec_resume_after_the_date_changes(self) -> None:
        # Codex dates by TZ (its timezone is the system's): the update names the date, not the
        # working directory and shell, as when a session runs past midnight.
        turns = [Turn("List the files.", env={"TZ": "Pacific/Kiritimati"}),
                 Turn("List them again.", env={"TZ": "Pacific/Pago_Pago"})]
        view, relayed = self.compare(exec_turns, turns, {"tool_calls": 5, "command": COMMAND})
        updates = [_text(item) for body in relayed for item in body["input"]
                   if _text(item).startswith("<environment_context>") and "<cwd>" not in _text(item)]
        self.assertTrue(updates)
        text = json.dumps(view)
        self.assertIn("<cwd>", text)
        self.assertIn(re.search(r"<current_date>.*?</current_date>", updates[-1]).group(), text)

    def test_app_server_with_approved_command_prefixes(self) -> None:
        view, _ = self.compare(app_server_turns, [Turn("Count to two thousand."), Turn("Do it again.")],
                               {"tool_calls": 3, "command": COMMAND, "arguments": ESCALATE})
        self.assertIn('approved: - [\\"seq\\"]', json.dumps(view))

    @unittest.skipUnless(shutil.which("setsid"), "setsid is not installed")
    def test_tui_with_approved_command_prefixes_and_agents_md_edited_between_turns(self) -> None:
        edit = lambda root: (root / "A" / "AGENTS.md").write_text(AGENTS["B"])  # noqa: E731
        view, _ = self.compare(tui_turns, [Turn("Count to two thousand."), Turn("Do it again.", before=edit)],
                               {"tool_calls": 3, "command": COMMAND, "arguments": ESCALATE})
        text = json.dumps(view)
        self.assertIn('approved: - [\\"seq\\"]', text)
        self.assertIn("Project notes (v1)", text)  # a running TUI keeps the AGENTS.md it loaded

    def test_concurrent_sessions_editing_the_same_directory(self) -> None:
        other = "Count in the other session."

        def sessions(bench: Bench, fake: FakeUpstream, turns: list[Turn], limits: list[int]) -> list[int]:
            """Session A's first turn while session B runs in the same directory and AGENTS.md changes,
            then A's second turn; returns the request count of A's first turn."""

            env, work = bench.env(), bench.root / "A"
            options = ["--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-m", MODEL]
            a = subprocess.Popen(["codex", "exec", *options, "-c", f"model_auto_compact_token_limit={limits[0]}",
                                  turns[0].prompt], cwd=work, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            deadline = time.monotonic() + 60
            while not fake.bodies("/v1/responses") and time.monotonic() < deadline:
                time.sleep(0.1)
            (work / "AGENTS.md").write_text(AGENTS["B"])  # edited while A's turn runs
            subprocess.run(["codex", "exec", *options, other], cwd=work, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=300, check=True)
            stdout, _ = a.communicate(timeout=300)
            thread = next(json.loads(line)["thread_id"] for line in stdout.splitlines() if '"thread.started"' in line)
            mine = [b for b in fake.bodies("/v1/responses") if any(turns[0].prompt in _text(i) for i in b["input"])]
            result = subprocess.run(["codex", "exec", "resume", *options, "-c",
                                     f"model_auto_compact_token_limit={limits[1]}", thread, turns[1].prompt],
                                    cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
            assert result.returncode == 0, result.stderr[-2000:]
            return [len(mine)]

        turns = [Turn("Count to two thousand."), Turn("Do it again.")]
        view, relayed = self.compare(sessions, turns, {"tool_calls": 3, "command": f"sleep 1; {COMMAND}"})
        self.assertIn("Project notes (v2)", json.dumps(view))  # A's second turn picked up B's edit
        for body in relayed:  # one Relay, two sessions: neither sees the other's conversation
            texts = [_text(item) for item in body["input"]]
            self.assertFalse(any(other in t for t in texts) and any(turns[0].prompt in t for t in texts))

    def test_exec_with_agents_spawned_waited_on_and_closed(self) -> None:
        # Codex lists its open agents in the environment context, though a change to them alone
        # renders no update.
        calls = [lambda agents, n=n: spawn(f"AGENT {n}") for n in range(5)]
        calls += [lambda agents: wait(agents[0]), lambda agents: close(agents[1]), lambda agents: close(agents[3])]
        view, relayed = self.compare(exec_turns, [Turn("Delegate the counting.")],
                                     {"tool_calls": 10, "command": COMMAND, "script": delegating({"Delegate": calls})},
                                     at=9)
        listed = re.findall(r"- (agent-\d+)", json.dumps(_normal(view, relayed)))
        self.assertEqual(sorted(listed), ["agent-0", "agent-2", "agent-4"])

    def test_exec_resume_with_an_agent_swapped_after_the_date_changes(self) -> None:
        # The update for the new date lists the agents open at the time.
        script = delegating({"Delegate": [lambda agents: spawn("AGENT A"), lambda agents: wait(agents[0])],
                             "Swap": [lambda agents: close(agents[0]), lambda agents: spawn("AGENT B"),
                                      lambda agents: wait(agents[1])]})
        turns = [Turn("Delegate the counting.", env={"TZ": "Pacific/Kiritimati"}),
                 Turn("Swap the agents.", env={"TZ": "Pacific/Pago_Pago"})]
        view, relayed = self.compare(exec_turns, turns, {"tool_calls": 6, "command": COMMAND, "script": script}, at=10)
        self.assertEqual(re.findall(r"- (agent-\d+)", json.dumps(_normal(view, relayed))), ["agent-1"])

    def test_exec_resume_with_everything_changing_and_two_compactions(self) -> None:
        script = delegating({"Delegate": [lambda agents: spawn("AGENT A"), lambda agents: wait(agents[0])]})
        turns = [Turn("Delegate the counting.", env={"TZ": "Pacific/Kiritimati"}),
                 Turn("Count in B.", cwd="B", model=SWITCHED, env={"TZ": "Pacific/Pago_Pago"}),
                 Turn("Count in A.", env={"TZ": "Pacific/Pago_Pago"})]
        self.compare(exec_turns, turns, {"tool_calls": 4, "command": COMMAND, "script": script}, at=5, nth=2)

    def test_app_server_with_a_command_approved_in_the_compacting_step(self) -> None:
        # Codex compacts with the state from before the approval; the saved prefix follows.
        view, _ = self.compare(app_server_turns, [Turn("Count to six thousand.")],
                               {"tool_calls": 3, "command": COMMAND, "arguments": ESCALATE}, at=1)
        self.assertEqual(_text(view[-1]), 'Approved command prefix saved:\n- ["seq"]')

    @unittest.skipUnless(shutil.which("setsid"), "setsid is not installed")
    def test_tui_with_an_agent_approvals_and_agents_md_edited_between_turns(self) -> None:
        edit = lambda root: (root / "A" / "AGENTS.md").write_text(AGENTS["B"])  # noqa: E731
        script = delegating({"Delegate": [lambda agents: spawn("AGENT A"), lambda agents: wait(agents[0])]})
        view, relayed = self.compare(
            tui_turns, [Turn("Delegate the counting."), Turn("Do it again.", before=edit)],
            {"tool_calls": 4, "command": COMMAND, "arguments": ESCALATE, "script": script}, at=6)
        self.assertEqual(re.findall(r"- (agent-\d+)", json.dumps(_normal(view, relayed))), ["agent-0"])


if __name__ == "__main__":
    unittest.main()
