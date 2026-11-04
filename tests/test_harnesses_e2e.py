"""Real harness binaries driven through Relay against the fake upstream."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from relay import Compaction, Engine, PrefixStore, ProxyConfig, create_app
from relay import intercept
from relay.integrations import ClaudeCodeHook
from relay.prompts import SUMMARY_PREFIX
from relay.protocols import AnthropicMessages, OpenAIResponses
from relay.strategies import ContextFolding, Headroom, Jev
from relay.strategies.jev import STATE_CONTEXT
from tests.fakes import COMPACTION_MARKER, FINAL_TEXT, FakeUpstream, local_upstreams, serve


def _env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC", "CODEX", "OPENAI"))}
    return {**env, "HOME": str(home), "CODEX_HOME": str(home), "NO_COLOR": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"}


def _commands(harness: str, work: Path) -> list[list[str]]:
    if harness == "codex":
        common = ["--json", "--skip-git-repo-check"]
        return [["codex", "exec", *common, "--sandbox", "workspace-write", "--cd", str(work), "List the files."],
                ["codex", "exec", "resume", "--last", *common, "Do it again."]]
    common = ["--output-format", "json", "--permission-mode", "bypassPermissions"]
    return [["claude", "-p", "List the files.", *common], ["claude", "-p", "Do it again.", "--continue", *common]]


def _text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


class HarnessEndToEndTests(unittest.TestCase):
    """Two turns with enough tool output to force several compactions."""

    cases = {"codex": ("/v1/responses", "input", 11_500), "claude": ("/v1/messages", "messages", 16_500)}

    def run_harness(self, harness: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        path, key, threshold = self.cases[harness]
        fake = FakeUpstream(tool_calls=6, command="seq 1 300")
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            config = ProxyConfig(upstreams=local_upstreams(upstream))
            engine = Engine(Compaction(threshold=threshold), PrefixStore())
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            env = _env(home)
            with serve(create_app(engine, config)) as relay:
                if harness == "codex":
                    (home / "config.toml").write_text(
                        'model = "gpt-5.1-codex"\nmodel_provider = "relay"\n[model_providers.relay]\n'
                        f'name = "Relay"\nbase_url = "{relay}/v1"\nenv_key = "RELAY_TEST_KEY"\n'
                        'wire_api = "responses"\nsupports_websockets = false\n')
                    env["RELAY_TEST_KEY"] = "tenant"
                else:
                    env.update(ANTHROPIC_BASE_URL=relay, ANTHROPIC_API_KEY="tenant")
                for command in _commands(harness, work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)

        bodies = fake.bodies(path)
        # (the prompt, which Claude Code's system messages may follow)
        summaries = [b for b in bodies if COMPACTION_MARKER in _text([i for i in b[key] if i.get("role") != "system"][-1])]
        main = [b for b in bodies if b not in summaries]
        first = bodies.index(summaries[0])
        self.assertGreaterEqual(len(summaries), 2)
        self.assertLess(len(summaries), len(main) / 2)  # rewrites are reused, not recomputed
        for body in bodies[first + 1 :]:
            if body not in summaries:
                self.assertTrue(any(_text(i).startswith(SUMMARY_PREFIX) for i in body[key]))

    def test_codex(self) -> None:
        self.run_harness("codex")

    def test_claude_code(self) -> None:
        self.run_harness("claude")


class ClaudeCodeHookTests(unittest.TestCase):
    """Claude Code on its hook: its requests go to its own endpoint (here the fake), and Relay's
    strategy decides through the plugin, before every request."""

    def run_claude(self, strategy: Any, fake: FakeUpstream) -> list[dict[str, Any]]:
        if shutil.which("claude") is None:
            self.skipTest("claude is not installed")
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            (home / ".claude").mkdir(parents=True), work.mkdir()
            with serve(create_app(Engine(strategy, PrefixStore()), ProxyConfig(upstreams={}))) as relay:
                config: dict[str, Any] = {}
                for setting in ClaudeCodeHook().installation(relay).settings:
                    config.setdefault(setting.key[0], {})[setting.key[1]] = setting.value
                (home / ".claude" / "settings.json").write_text(json.dumps(config))
                env = {**_env(home), "ANTHROPIC_BASE_URL": upstream, "ANTHROPIC_API_KEY": "tenant"}
                self.compactions = []  # Claude Code's own record of its compactions, after each turn
                for command in _commands("claude", work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)
                    self.compactions.append(sum(line.count('"compact_boundary"') for path in
                                                (home / ".claude" / "projects").rglob("*.jsonl")
                                                for line in path.read_text().splitlines()))
        return [b for b in fake.bodies("/v1/messages") if b.get("tools")]

    def test_compaction_through_the_hook(self) -> None:
        bodies = self.run_claude(Compaction(threshold=16_500), FakeUpstream(tool_calls=6, command="seq 1 300"))
        compacted = [b for b in bodies if any(SUMMARY_PREFIX in _text(m) for m in b["messages"])]
        self.assertTrue(compacted)  # (the summary, Claude Code's own model's, in Claude Code's transcript)
        first = bodies.index(compacted[0])
        for body in bodies[first:]:  # (turn 2 too: resumed, Claude Code would bring the history back)
            self.assertIn(body, compacted)
        self.assertTrue(all(_size(b) < 16_500 * 4 + 40_000 for b in bodies))  # (its own context besides)

    def test_the_strategys_instructions_end_the_system_prompt(self) -> None:
        bodies = self.run_claude(ContextFolding(), FakeUpstream(tool_calls=2))
        self.assertTrue(bodies)
        for body in bodies:
            self.assertIn("## Branches (context folding)", body["system"][-1]["text"][-3_000:])

    def test_a_result_the_strategy_rewrites_and_a_branch_that_folds(self) -> None:
        def bash(command: str) -> dict[str, Any]:
            return {"name": "Bash", "input": {"command": command, "description": "Run"}}

        def script(steps: int, turns: list) -> Any:
            sent = json.dumps(fake.requests[-1]["body"]["messages"])
            if "Branch has finished its task" in sent:
                return FINAL_TEXT
            if "MODE: BRANCH" in sent:
                return bash('echo "[return] the numbers run from 1 to 50"') if "\\n50" in sent else bash("seq 1 50")
            return bash('echo "[branch] count numbers :: print 1 to 50"')

        fake = FakeUpstream(tool_calls=99, script=script)
        bodies = self.run_claude(ContextFolding(), fake)
        sent = [json.dumps(b["messages"]) for b in bodies]
        self.assertTrue(any("MODE: BRANCH" in s for s in sent))  # (the branch's echo, rewritten)
        last = sent[-1]
        self.assertIn("Branch has finished its task, the returned message is", last)
        self.assertNotIn("\\n49\\n50", last)  # (the branch's steps, folded away)
        self.assertEqual(self.compactions[0], 1)  # (the branch's echo was written as it came in; the fold compacted)


class ClaudeCodeHookSwitchTests(unittest.TestCase):
    """One open Claude Code session, the strategy changed between its turns (`relay strategy`)."""

    def test_a_strategy_changed_mid_session_compacts_and_is_told(self) -> None:
        if shutil.which("claude") is None:
            self.skipTest("claude is not installed")
        # (below Claude Code 2.1.309's own request with two files read, which is smaller than 2.1.294's)
        engine = Engine(Compaction(threshold=14_000), PrefixStore())
        fake = FakeUpstream(tool_calls=6, command="seq 1 300")
        turns = "".join(json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
                        for text in ("List the files.", "Do it again."))

        def switch(request: dict[str, Any]) -> None:  # (between the turns, as `relay strategy folding` would)
            if "Do it again." in json.dumps(request.get("body")) and engine.strategy.name == "compaction":
                engine.strategy = ContextFolding(Compaction(threshold=14_000))

        record = fake._record
        fake._record = lambda request, body: (switch({"body": body}), record(request, body))[1]
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            (home / ".claude").mkdir(parents=True), work.mkdir()
            with serve(create_app(engine, ProxyConfig(upstreams={}))) as relay:
                config: dict[str, Any] = {}
                for setting in ClaudeCodeHook().installation(relay).settings:
                    config.setdefault(setting.key[0], {})[setting.key[1]] = setting.value
                (home / ".claude" / "settings.json").write_text(json.dumps(config))
                env = {**_env(home), "ANTHROPIC_BASE_URL": upstream, "ANTHROPIC_API_KEY": "tenant"}
                result = subprocess.run(["claude", "-p", "--input-format", "stream-json", "--output-format",
                                         "stream-json", "--verbose", "--permission-mode", "bypassPermissions"],
                                        cwd=work, env=env, input=turns, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(engine.strategy.name, "folding")
        bodies = [b for b in fake.bodies("/v1/messages") if b.get("tools")]
        second = next(n for n, b in enumerate(bodies) if "Do it again." in json.dumps(b))
        compacted = [n for n, b in enumerate(bodies) if any(SUMMARY_PREFIX in _text(m) for m in b["messages"])]
        self.assertTrue([n for n in compacted if n > second + 1], "turn 2 never compacted")  # (folding's fallback)
        toasts = [json.loads(line) for line in result.stdout.splitlines() if '"ui_toast"' in line]
        self.assertTrue(any("Relay now runs folding" in t.get("text", "") for t in toasts), toasts)


class ClaudeCodeHttpsTests(unittest.TestCase):
    """Claude Code through Relay as its HTTPS proxy: its base URL stays api.anthropic.com, Relay
    opens those connections and manages each request as through the proxy (here the fake answers
    for api.anthropic.com); Claude Code's own transcript is never touched."""

    def test_requests_are_compacted_and_claude_codes_transcript_is_its_own(self) -> None:
        if shutil.which("claude") is None:
            self.skipTest("claude is not installed")
        fake = FakeUpstream(tool_calls=6, command="seq 1 300")
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            (home / ".claude").mkdir(parents=True), work.mkdir()
            os.environ["RELAY_HOME"] = str(Path(root, "relay"))  # (the CA, in this test's Relay directory)
            # (below Claude Code's own request with a few results: on Anthropic's endpoint it is smaller)
            engine = Engine(Compaction(threshold=12_000), PrefixStore())
            with serve(create_app(engine, ProxyConfig(upstreams=local_upstreams(upstream)))) as relay:
                from tests.test_intercept import _listener
                proxy, _ = _listener(lambda: intercept.serve("127.0.0.1", 0, int(relay.rsplit(":", 1)[1])))
                (home / ".claude" / "settings.json").write_text(json.dumps({"env": {
                    "HTTPS_PROXY": f"http://127.0.0.1:{proxy}", "NODE_EXTRA_CA_CERTS": str(intercept.ca_file())}}))
                env = {**_env(home), "ANTHROPIC_API_KEY": "tenant"}  # (no ANTHROPIC_BASE_URL)
                for command in _commands("claude", work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)
            transcript = "".join(path.read_text() for path in (home / ".claude" / "projects").rglob("*.jsonl"))
        requests = [r for r in fake.requests if isinstance(r.get("body"), dict) and r["body"].get("tools")]
        self.assertTrue(requests)
        self.assertIn("prompt-caching-scope", requests[0]["headers"].get("anthropic-beta", ""))  # (Anthropic's own)
        self.assertTrue(any(any(SUMMARY_PREFIX in _text(m) for m in r["body"]["messages"]) for r in requests))
        self.assertFalse([r for r in requests if "thread" in r["body"]])  # (no server-side thread: Relay sees it all)
        self.assertNotIn("compact_boundary", transcript)  # (Claude Code never compacted: it saw nothing)
        self.assertNotIn(SUMMARY_PREFIX, transcript)


class JevEndToEndTests(unittest.TestCase):
    """`jev` with each harness: Jev (here a fake that keeps every call and no result in full) is asked
    about the older calls, and the requests that follow carry their results cut to the library's head
    and note (where that frees too little, compaction compacts instead)."""

    def run_harness(self, harness: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        path, key, _ = HarnessEndToEndTests.cases[harness]
        # (above the harness's own request with a few calls: with only the newest, pinned, there is
        # nothing to ask about, and too little to free falls back to compaction)
        threshold, asked = {"codex": 13_500, "claude": 16_500}[harness], []

        async def answer(request: Request) -> JSONResponse:
            body = await request.json()
            asked.append({**body, "authorization": request.headers.get("authorization")})
            return JSONResponse({"answers": {name: {"noul": 0.9 if name.startswith("call_") else 0.1}
                                             for name in body["questions"]}})

        fake = FakeUpstream(tool_calls=10, command="seq 1 300")
        jev_app = Starlette(routes=[Route("/v1/systemone", answer, methods=["POST"])])
        with serve(fake.app) as upstream, serve(jev_app) as jev, tempfile.TemporaryDirectory() as root:
            strategy = Jev(trigger=Compaction(threshold=threshold), fallback=Compaction(threshold=threshold),
                           api_key="jev-key", url=f"{jev}/v1/systemone")
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            env = _env(home)
            with serve(create_app(Engine(strategy, PrefixStore()), ProxyConfig(upstreams=local_upstreams(upstream)))) as relay:
                if harness == "codex":
                    (home / "config.toml").write_text(
                        'model = "gpt-5.1-codex"\nmodel_provider = "relay"\n[model_providers.relay]\n'
                        f'name = "Relay"\nbase_url = "{relay}/v1"\nenv_key = "RELAY_TEST_KEY"\n'
                        'wire_api = "responses"\nsupports_websockets = false\n')
                    env["RELAY_TEST_KEY"] = "tenant"
                else:
                    env.update(ANTHROPIC_BASE_URL=relay, ANTHROPIC_API_KEY="tenant")
                for command in _commands(harness, work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)
        self.assertTrue(asked)
        self.assertEqual({(a["model"], a["authorization"], a["state"]["context"]) for a in asked},
                         {("jev-latest", "Bearer jev-key", STATE_CONTEXT)})
        cut = [body for b in fake.bodies(path) if "[fast-jev-compaction truncated" in (body := json.dumps(b[key]))]
        self.assertTrue([body for body in cut if SUMMARY_PREFIX not in body])  # (Jev's context, no summary)

    def test_codex(self) -> None:
        self.run_harness("codex")

    def test_claude_code(self) -> None:
        self.run_harness("claude")


class HeadroomEndToEndTests(unittest.TestCase):
    """`headroom` with each harness and a real Headroom (`headroom proxy`, in the test image): the log
    each command prints goes compressed, its ERROR line kept, and each request forwards what the ones
    before it forwarded as it was (the prompt cache holds)."""

    LOG = ('for i in $(seq 1 400); do echo "2026-11-04 10:00:00 INFO request $i served in 12ms"; done; '
           'echo "2026-11-04 10:07:00 ERROR disk full: CODE amber"')

    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("headroom") is None:
            raise unittest.SkipTest("Headroom is not installed")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        cls.url, cls.home = f"http://127.0.0.1:{port}", tempfile.TemporaryDirectory()
        env = {**os.environ, "HOME": cls.home.name, "HEADROOM_TELEMETRY": "off", "HEADROOM_BEACON": "off", "DO_NOT_TRACK": "1"}
        cls.headroom = subprocess.Popen(["headroom", "proxy", "--port", str(port)], env=env,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):
            try:
                httpx.get(f"{cls.url}/health", timeout=2)
                return
            except httpx.HTTPError:
                time.sleep(0.2)
        cls.tearDownClass()
        raise RuntimeError("Headroom did not start")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.headroom.terminate()
        cls.headroom.wait(timeout=30)
        cls.home.cleanup()

    def run_harness(self, harness: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        path, key, _ = HarnessEndToEndTests.cases[harness]
        fake = FakeUpstream(tool_calls=4, command=self.LOG)
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            env = _env(home)
            with serve(create_app(Engine(Headroom(url=self.url), PrefixStore()), ProxyConfig(upstreams=local_upstreams(upstream)))) as relay:
                if harness == "codex":
                    (home / "config.toml").write_text(
                        'model = "gpt-5.1-codex"\nmodel_provider = "relay"\n[model_providers.relay]\n'
                        f'name = "Relay"\nbase_url = "{relay}/v1"\nenv_key = "RELAY_TEST_KEY"\n'
                        'wire_api = "responses"\nsupports_websockets = false\n')
                    env["RELAY_TEST_KEY"] = "tenant"
                else:
                    env.update(ANTHROPIC_BASE_URL=relay, ANTHROPIC_API_KEY="tenant")
                for command in _commands(harness, work):
                    result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)
        codec = OpenAIResponses() if harness == "codex" else AnthropicMessages()
        sent = [[codec.canonical(item) for item in body[key]] for body in fake.bodies(path)]
        last = json.dumps(fake.bodies(path)[-1][key])
        self.assertIn("lines omitted", last)
        self.assertIn("ERROR disk full: CODE amber", last)
        self.assertNotIn("request 200 served", last)
        for n, items in enumerate(sent):  # (of the same conversation: the same first item)
            if earlier := [before for before in sent[:n] if before[:1] == items[:1] and len(before) <= len(items)]:
                longest = max(earlier, key=len)
                self.assertEqual(items[: len(longest)], longest, f"request {n} changed what was forwarded")

    def test_codex(self) -> None:
        self.run_harness("codex")

    def test_claude_code(self) -> None:
        self.run_harness("claude")


class PiEndToEndTests(unittest.TestCase):
    """pi (1.4) declares its tools and its reasoning in the input, not beside it: once compacted,
    its requests keep both, and the model goes on calling tools."""

    def test_a_compacted_request_keeps_the_tools_and_the_reasoning_pi_declared(self) -> None:
        if shutil.which("pi") is None:
            self.skipTest("pi is not installed")
        fake = FakeUpstream(tool_calls=6, script=[{"name": "bash", "input": {"command": "seq 1 300"}}])
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            (home / ".pi/agent").mkdir(parents=True), work.mkdir()
            engine = Engine(Compaction(threshold=3_000), PrefixStore())
            with serve(create_app(engine, ProxyConfig(upstreams=local_upstreams(upstream)))) as relay:
                (home / ".pi/agent/models.json").write_text(json.dumps({"providers": {"openai": {"baseUrl": f"{relay}/v1"}}}))
                (home / ".pi/agent/settings.json").write_text(json.dumps({"compaction": {"enabled": False}}))
                env = {**_env(home), "OPENAI_API_KEY": "tenant"}
                for resume in ([], ["-c"]):
                    result = subprocess.run(["pi", *resume, "-p", "--model", "openai/gpt-6-luna", "--thinking", "high",
                                             "List the files."], cwd=work, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    self.assertIn(FINAL_TEXT, result.stdout)
        bodies = fake.bodies("/v1/responses")
        compacted = [b for b in bodies if any(SUMMARY_PREFIX in _text(i) for i in b["input"])]
        self.assertGreater(len(compacted), 2)
        configured = any(i.get("type") == "configuration_update" for i in bodies[0]["input"])  # (since 1.4)
        for body in compacted:
            self.assertTrue(OpenAIResponses().offers_tools(body))
            if configured:
                self.assertIn({"type": "configuration_update", "reasoning": {"effort": "high"}}, body["input"])
        self.assertTrue(any(i.get("type") == "function_call" for b in compacted for i in b["input"]))  # (still calling tools)


def _size(body: dict[str, Any]) -> int:
    return len(json.dumps(body))


class RelayRunTests(unittest.TestCase):
    """`relay run <harness>` points the harness at a private proxy."""

    def run_cli(self, harness: str, args: list[str], path: str) -> None:
        if shutil.which(harness) is None:
            self.skipTest(f"{harness} is not installed")
        fake = FakeUpstream()
        with serve(fake.app) as upstream, tempfile.TemporaryDirectory() as root:
            home, work = Path(root, "home"), Path(root, "work")
            home.mkdir(), work.mkdir()
            relay_home = Path(root, "relay")  # Relay's settings: its upstreams are the fake
            relay_home.mkdir()
            (relay_home / "settings.json").write_text(json.dumps({"openai_base_url": f"{upstream}/v1",
                                                                  "anthropic_base_url": upstream}))
            env = {**_env(home), "RELAY_HOME": str(relay_home), "OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
            result = subprocess.run([sys.executable, "-m", "relay.cli", "run", harness, "--", *args],
                                    cwd=work, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn(FINAL_TEXT, result.stdout)
        self.assertTrue(fake.bodies(path))

    def test_codex(self) -> None:
        self.run_cli("codex", ["exec", "--json", "--skip-git-repo-check", "-m", "gpt-5.1-codex", "Hi."],
                     "/v1/responses")

    def test_claude_code(self) -> None:
        self.run_cli("claude", ["-p", "Hi.", "--output-format", "json"], "/v1/messages")


if __name__ == "__main__":
    unittest.main()
