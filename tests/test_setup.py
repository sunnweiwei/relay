"""Relay's settings and its command line: settings a running Relay follows, and `relay setup`,
which points a harness at Relay only once Relay answers."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from starlette.testclient import TestClient

from relay import cli, install, leaving, service, settings
from relay.transport import create_app
from tests.fakes import FakeUpstream, local_upstreams, relay_settings, serve


class SettingsTests(unittest.TestCase):
    def test_values_are_json_where_they_parse_and_text_otherwise(self) -> None:
        self.assertEqual(settings.parse("clm_budget=26000"), ("clm_budget", 26000))
        self.assertEqual(settings.parse("clm_nudges=false"), ("clm_nudges", False))
        self.assertEqual(settings.parse("clm_dir=/tmp/ctx"), ("clm_dir", "/tmp/ctx"))
        with self.assertRaises(ValueError):
            settings.parse("clm_budget")

    def test_saved_settings_are_read_as_text_and_unset_with_none(self) -> None:
        with relay_settings(clm_budget=26000, clm_nudges=False):
            self.assertEqual((settings.value("clm_budget"), settings.value("clm_nudges")), ("26000", "false"))
            settings.save({"clm_budget": None})
            self.assertIsNone(settings.value("clm_budget"))
            self.assertEqual(settings.value("strategy", "compaction"), "compaction")


class FollowingTests(unittest.TestCase):
    def test_a_running_relay_follows_the_strategy_from_the_next_request(self) -> None:
        fake = FakeUpstream(tool_calls=0)
        with relay_settings(strategy="compaction", cache_path="off"), serve(fake.app) as upstream:
            from relay.transport import ProxyConfig

            body = {"model": "m", "input": [{"type": "message", "role": "user", "content": "task"}],
                    "tools": [{"type": "function", "name": "sh", "parameters": {}}], "stream": False}
            with TestClient(create_app(config=ProxyConfig(upstreams=local_upstreams(upstream)))) as client:
                self.assertEqual(client.get("/relay/v1/status").json()["strategy"], "compaction")
                client.post("/v1/responses", json=body)
                settings.save({"strategy": "clm"})
                client.post("/v1/responses", json=body)  # the same conversation, its next request
                self.assertEqual(client.get("/relay/v1/status").json()["strategy"], "clm")
        first, second = (json.dumps(b) for b in fake.bodies("/v1/responses"))
        self.assertNotIn("## Managing your context", first)
        self.assertIn("## Managing your context", second)

    def test_settings_it_cannot_apply_leave_the_strategy_running(self) -> None:
        with relay_settings(strategy="compaction", cache_path="off"):
            with TestClient(create_app()) as client:
                settings.save({"strategy": "no such strategy"})
                with self.assertLogs("relay", "ERROR"):
                    self.assertEqual(client.get("/relay/v1/status").json()["strategy"], "compaction")


class SetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.codex = self.root / "codex"
        self.codex.mkdir()
        (self.codex / "config.toml").write_text('model = "m"\n')
        patcher = mock.patch.dict(os.environ, {"RELAY_HOME": str(self.root / "relay"), "CODEX_HOME": str(self.codex)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_the_harness_is_left_alone_when_relay_does_not_start(self) -> None:
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "taken", return_value=False), \
                mock.patch.object(service, "start", return_value="process"), \
                mock.patch.object(service, "wait", return_value=None):
            code, out = self.run_cli("setup", "codex", "--strategy", "clm")
        self.assertEqual(code, 1)
        self.assertIn("left as it was", out)
        self.assertEqual((self.codex / "config.toml").read_text(), 'model = "m"\n')
        self.assertEqual(install.installed(), [])
        self.assertEqual(settings.value("strategy"), "clm")  # (kept for the next try)

    def test_setup_starts_relay_then_points_the_harness_at_it(self) -> None:
        calls = []
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "taken", return_value=False), \
                mock.patch.object(service, "start", side_effect=lambda: calls.append("start") or "systemd"), \
                mock.patch.object(service, "wait", side_effect=lambda *a, **k: calls.append(("wait", a)) or {"strategy": "clm"}):
            code, out = self.run_cli("setup", "codex", "--strategy", "clm", "--set", "clm_budget=40000")
        self.assertEqual(code, 0, out)
        self.assertEqual(calls, ["start", ("wait", ("clm",))])
        self.assertEqual(install.installed(), ["codex"])
        self.assertIn("127.0.0.1:8787/up/codex-0", (self.codex / "config.toml").read_text())
        self.assertEqual((settings.value("strategy"), settings.value("clm_budget")), ("clm", "40000"))
        self.assertIn("one already open goes on as it was", out)

    def test_uninstalling_every_harness_leaves_relay_running_but_not_started_at_login(self) -> None:
        claude = self.root / "claude"
        released = []
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(claude)}), \
                mock.patch.object(service, "release", side_effect=lambda: released.append(1) or "systemd"):
            install.install("codex", "http://127.0.0.1:8787", [])
            install.install("claude_code", "http://127.0.0.1:8787",
                            cli.HARNESSES["claude_code"].starter(service.start_command()))  # (as `relay setup` sets it)
            code, out = self.run_cli("uninstall", "codex")
            self.assertEqual((code, released), (0, []))  # (Claude Code still goes through Relay)
            code, out = self.run_cli("uninstall", "--all")
        self.assertEqual((code, install.installed(), released), (0, [], [1]))
        self.assertIn("no longer started at login", out)
        self.assertNotIn("relay start", json.dumps(json.loads((claude / "settings.json").read_text())))  # (its hook went)
        with self.assertRaises(SystemExit):
            self.run_cli("stop")  # (there is no stopping Relay: uninstalling is going back)

    def test_setup_says_how_to_leave(self) -> None:
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "taken", return_value=False), \
                mock.patch.object(service, "start", return_value="systemd"), \
                mock.patch.object(service, "wait", return_value={"strategy": "compaction"}):
            code, out = self.run_cli("setup", "codex")
        self.assertIn("To leave Relay: `relay uninstall codex` (or ask the agent to run it)", out)

    def test_a_conversation_recorded_with_a_smaller_window_is_judged_by_the_harnesss_now(self) -> None:
        # Recorded before Claude Code's allowance (Relay 25f6507: the catalog's 200k).
        install.install("claude_code", "http://127.0.0.1:8787", [])
        (self.root / "relay").mkdir(parents=True, exist_ok=True)
        (self.root / "relay" / "conversations.json").write_text(json.dumps({"old": {
            "harness": "claude_code", "title": "fix it", "last": time.time(), "window": 200_000, "whole": 182_000}}))
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "release"):
            code, out = self.run_cli("uninstall", "claude")
        self.assertEqual((code, install.installed()), (0, []))  # it fits direct: restored at once

    def test_an_old_record_is_handed_back_by_the_harnesss_own_compaction(self) -> None:
        # Recorded before conversations were named apart from the strategy: its compaction, under
        # the new name, hands it back by the harness and what its user first typed.
        install.install("claude_code", "http://127.0.0.1:8787", [])
        (self.root / "relay").mkdir(parents=True, exist_ok=True)
        (self.root / "relay" / "conversations.json").write_text(json.dumps({"old": {
            "harness": "claude_code", "title": "fix it", "last": time.time(), "window": 300_000, "whole": 400_000}}))
        with mock.patch.object(service, "status", return_value=None):
            self.run_cli("uninstall", "claude")
        self.assertEqual(len(leaving.needing("claude_code")), 1)
        leaving.seen("claude_code", "new", "fix it", 400_000, 300_000, True)  # its own /compact, through Relay
        self.assertEqual(leaving.needing("claude_code"), [])

    def test_uninstall_gives_the_command_that_compacts_the_conversation_and_asks_the_agent_to_ask(self) -> None:
        install.install("claude_code", "http://127.0.0.1:8787", [])
        leaving.seen("claude_code", "c1", "fix it", 400_000, 300_000, False, session="d6f1-session")
        with mock.patch.object(service, "status", return_value=None):
            code, out = self.run_cli("uninstall", "claude")
        self.assertIn('`claude --resume d6f1-session -p "/compact"`', out)
        last = out.strip().splitlines()[-1]
        self.assertIn("ask", last.lower())
        self.assertIn("A, B or C", last)

    def test_a_harness_with_long_conversations_leaves_once_they_no_longer_need_relay(self) -> None:
        install.install("codex", "http://127.0.0.1:8787", [])
        leaving.seen("codex", "c1", "fix the flaky test", 300_000, 128_000, False)  # grew past the window
        leaving.seen("codex", "c2", "a short one", 20_000, 128_000, False)
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "release") as release:
            code, out = self.run_cli("uninstall", "codex")
            self.assertEqual((code, install.installed(), list(leaving.leaving())), (0, ["codex"], ["codex"]))
            self.assertIn('"fix the flaky test" — ~300k tokens for a 128k window', out)
            self.assertNotIn("a short one", out)
            self.assertIn("`relay uninstall codex --now`", out)
            # The last lines say it is not done, and what finishes it (an agent puts them to its user).
            self.assertIn("Not finished: codex still goes through Relay", out)
            self.assertEqual([line.strip()[:2] for line in out.strip().splitlines()[-4:-1]], ["A.", "B.", "C."])
            code, out = self.run_cli("status")
            self.assertIn("leaving Relay (uninstall not finished): codex, restored by itself once", out)
            self.assertIn("harnesses set up: none", out)
            leaving.seen("codex", "c1", "fix the flaky test", 300_000, 128_000, True)  # its own /compact, through Relay
            code, out = self.run_cli("status")
            self.assertIn("codex: no conversation needs Relay any more; its configuration is restored", out)
            self.assertEqual((install.installed(), leaving.leaving()), ([], {}))
            release.assert_called_once()

    def test_leaving_ends_a_week_after_last_use_or_now_or_when_set_up_again(self) -> None:
        install.install("codex", "http://127.0.0.1:8787", [])
        leaving.seen("codex", "c1", "old work", 300_000, 128_000, False, now=time.time() - 8 * 86400)
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "release"):
            code, out = self.run_cli("uninstall", "codex")  # (used more than a week ago: nothing holds it)
            self.assertIn("uninstalled codex", out)
            install.install("codex", "http://127.0.0.1:8787", [])
            leaving.seen("codex", "c2", "new work", 300_000, 128_000, False)
            self.run_cli("uninstall", "codex")
            self.assertEqual(list(leaving.leaving()), ["codex"])
            leaving.stay("codex")  # (`relay setup codex` again)
            self.assertEqual(leaving.leaving(), {})
            code, out = self.run_cli("uninstall", "codex", "--now")
            self.assertIn("uninstalled codex", out)
            self.assertIn("fails direct", out)
            self.assertEqual(install.installed(), [])

    def test_a_port_something_else_holds_is_given_up_for_a_free_one(self) -> None:
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "taken", return_value=True), \
                mock.patch.object(service, "free_port", return_value=48123), mock.patch.object(service, "start", return_value="process"), \
                mock.patch.object(service, "wait", return_value={"strategy": "compaction"}):
            code, out = self.run_cli("setup", "codex")
        self.assertEqual(code, 0, out)
        self.assertEqual(settings.value("port"), "48123")
        self.assertIn("127.0.0.1:48123/up/codex-0", (self.codex / "config.toml").read_text())
        with mock.patch.object(service, "status", return_value=None), mock.patch.object(service, "taken", return_value=True):
            code, out = self.run_cli("setup", "codex", "--set", "port=9")  # (one the user chose: their call)
        self.assertEqual(code, 1)
        self.assertIn("choose another port", out)

    def test_changing_the_strategy_reaches_the_running_relay_without_a_restart(self) -> None:
        with mock.patch.object(service, "status", return_value={"strategy": "compaction"}), \
                mock.patch.object(service, "restart") as restart, \
                mock.patch.object(service, "wait", return_value={"strategy": "folding"}) as wait:
            code, out = self.run_cli("strategy", "folding")
        self.assertEqual(code, 0, out)
        restart.assert_not_called()
        wait.assert_called_once_with("folding", seconds=15)
        self.assertIn("every conversation", out)

    def test_claude_code_set_up_on_its_hook_goes_through_the_path_its_strategy_runs_on(self) -> None:
        claude = self.root / "claude"
        claude.mkdir()

        def config() -> dict:
            return json.loads((claude / "settings.json").read_text())

        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(claude)}), \
                mock.patch("relay.integrations.claude_code.installed_release", return_value="2.1.309"), \
                mock.patch.object(service, "status", return_value={"strategy": "compaction"}), \
                mock.patch.object(service, "wait", side_effect=lambda name, **_: {"strategy": name}):
            code, out = self.run_cli("setup", "claude", "--strategy", "compaction", "--via", "hook")
            self.assertEqual(code, 0, out)
            self.assertIn("installed claude_code (hook)", out)
            self.assertEqual(config()["enabledPlugins"], {"relay@relay": True})
            self.assertNotIn("ANTHROPIC_BASE_URL", config()["env"])  # (Remote Control stays on)
            code, out = self.run_cli("strategy", "clm")  # (notes for one request: the proxy's)
            self.assertEqual(code, 0, out)
            self.assertIn("claude_code now goes through Relay as its HTTPS proxy (clm needs notes, which the hook "
                          "cannot carry)", out)  # (named, the hook cannot; the HTTPS proxy keeps Remote Control)
            self.assertNotIn("Remote Control is off", out)
            self.assertEqual(config()["env"]["HTTPS_PROXY"], f"http://127.0.0.1:{settings.value('intercept_port')}")
            self.assertNotIn("ANTHROPIC_BASE_URL", config()["env"])
            self.assertNotIn("enabledPlugins", config())
            code, out = self.run_cli("strategy", "folding")
            self.assertIn("claude_code now goes through its hook (it runs folding)", out)
            self.assertNotIn("ANTHROPIC_BASE_URL", config()["env"])
            code, out = self.run_cli("status")
            self.assertIn("harnesses set up: claude_code (hook)", out)
            code, out = self.run_cli("uninstall", "claude")
            self.assertEqual(config().get("env", {}), {})

    def test_settings_read_at_start_restart_the_service(self) -> None:
        with mock.patch.object(service, "status", return_value={"strategy": "compaction"}), \
                mock.patch.object(service, "record", return_value={"manager": "systemd"}), \
                mock.patch.object(service, "restart") as restart, \
                mock.patch.object(service, "wait", return_value={"strategy": "compaction"}):
            self.assertEqual(self.run_cli("set", "cache_ttl_seconds=60")[0], 0)
        restart.assert_called_once()

    def test_an_unknown_strategy_is_refused_before_anything_changes(self) -> None:
        code, out = self.run_cli("set", "strategy=nope")
        self.assertEqual(code, 1)
        self.assertIn("unknown strategy", out)
        self.assertIsNone(settings.value("strategy"))

    def test_help_is_the_guide(self) -> None:
        code, out = self.run_cli("help")
        self.assertEqual(code, 0)
        self.assertIn("relay setup codex --strategy clm", out)
        self.assertIn("For a coding agent asked to set Relay up", out)


class StarterTests(unittest.TestCase):
    """What `relay setup` adds to a harness so that it starts Relay itself, and takes back."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        patcher = mock.patch.dict(os.environ, {"RELAY_HOME": str(self.root / "relay"), "HOME": str(self.root),
                                               "CLAUDE_CONFIG_DIR": str(self.root / "claude"),
                                               "KIMI_CODE_HOME": str(self.root / "kimi"),
                                               "PI_CODING_AGENT_DIR": str(self.root / "pi")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_start_command_says_nothing_and_succeeds_while_relay_runs(self) -> None:
        with mock.patch.object(service, "address", return_value="http://127.0.0.1:9"):
            command = service.start_command()
        self.assertIn(">/dev/null 2>&1", command)  # (a session-start hook's output reaches the model)
        self.assertTrue(command.endswith("; true"))

    def test_claude_codes_hook_follows_the_users_and_goes_with_uninstall(self) -> None:
        from relay.harnesses import HARNESSES
        from relay.harnesses.claude_code import settings_file

        own = {"hooks": [{"type": "command", "command": "echo mine"}]}
        settings_file().parent.mkdir(parents=True)
        settings_file().write_text(json.dumps({"hooks": {"SessionStart": [own]}}))
        install.install("claude_code", "http://relay", HARNESSES["claude_code"].starter("relay-start"))
        hooks = json.loads(settings_file().read_text())["hooks"]["SessionStart"]
        self.assertEqual((hooks[0], hooks[1]["hooks"][0]["command"]), (own, "relay-start"))
        install.uninstall("claude_code")
        self.assertEqual(json.loads(settings_file().read_text())["hooks"]["SessionStart"], [own])

    def test_a_plugin_file_is_written_with_the_command_and_removed(self) -> None:
        from relay.harnesses import HARNESSES

        install.install("pi", "http://relay", HARNESSES["pi"].starter("relay-start"))
        plugin = self.root / "pi" / "extensions" / "relay-start.ts"
        self.assertIn('const COMMAND = "relay-start";', plugin.read_text())
        install.uninstall("pi")
        self.assertFalse(plugin.exists())

    def test_kimi_codes_hook_is_valid_toml_after_the_users(self) -> None:
        import tomllib

        from relay.harnesses import HARNESSES

        config = self.root / "kimi" / "config.toml"
        config.parent.mkdir()
        config.write_text('hooks = [{ event = "Stop", command = "echo mine" }]\n')
        install.install("kimi_code", "http://relay", HARNESSES["kimi_code"].starter("relay-start"))
        hooks = tomllib.loads(config.read_text())["hooks"]
        self.assertEqual([h["event"] for h in hooks], ["Stop", "UserPromptSubmit"])
        install.uninstall("kimi_code")
        self.assertEqual(tomllib.loads(config.read_text())["hooks"], [{"event": "Stop", "command": "echo mine"}])

    def test_kimi_codes_tables_named_with_a_slash_are_found_and_kept_toml(self) -> None:
        # (reported: its model `luna/gpt-6-luna` got a second, unquoted table, and Kimi Code read no config)
        import tomllib

        from relay.harnesses import HARNESSES
        from relay.harnesses.base import NEVER

        config = self.root / "kimi" / "config.toml"
        config.parent.mkdir()
        original = ('default_model = "luna/gpt-6-luna"\n\n[providers."luna.example"]\ntype = "openai"\n'
                    'base_url = "https://luna.example/v1"\n\n[models."luna/gpt-6-luna"]\nprovider = "luna.example"\n'
                    'max_context_size = 400000\n')
        config.write_text(original)
        install.install("kimi_code", "http://relay", HARNESSES["kimi_code"].settings())
        written = tomllib.loads(config.read_text())  # (valid TOML)
        self.assertEqual(config.read_text().count("[models."), 1)
        self.assertEqual(written["models"]["luna/gpt-6-luna"]["max_context_size"], NEVER)
        self.assertTrue(written["providers"]["luna.example"]["base_url"].startswith("http://relay/up/"))
        install.uninstall("kimi_code")
        self.assertEqual(config.read_text(), original)
        # A new table's name is quoted.
        install.install("kimi_code", "http://relay", [install.Setting(config, ("models", "sol/gpt-6", "max_context_size"), 1)])
        self.assertEqual(tomllib.loads(config.read_text())["models"]["sol/gpt-6"], {"max_context_size": 1})
        self.assertIn('[models."sol/gpt-6"]', config.read_text())
        install.uninstall("kimi_code")

    def test_kimi_codes_config_an_earlier_relay_broke_is_repaired_on_uninstall(self) -> None:
        # What `relay setup` wrote before: the same table again, unquoted, after the user's own.
        import tomllib

        config = self.root / "kimi" / "config.toml"
        config.parent.mkdir()
        original = '[models."luna/gpt-6-luna"]\nprovider = "luna"\nmax_context_size = 400000\n'
        config.write_text(original + "\n[models.luna/gpt-6-luna]\nmax_context_size = 9007199254740992\n")
        key = ["models", "luna/gpt-6-luna", "max_context_size"]
        install.state_file().parent.mkdir(parents=True, exist_ok=True)
        install.state_file().write_text(json.dumps({"kimi_code": [{"path": str(config), "key": key, "previous": None}]}))
        install.uninstall("kimi_code")
        self.assertEqual(config.read_text(), original)
        self.assertEqual(tomllib.loads(config.read_text())["models"]["luna/gpt-6-luna"]["max_context_size"], 400000)


class ServiceTests(unittest.TestCase):
    def test_the_service_runs_relay_and_restarts_it(self) -> None:
        with relay_settings():
            unit, agent = service.unit(), service.agent()
        self.assertIn("-m relay.cli serve", unit)
        self.assertIn("Restart=always", unit)
        self.assertIn("<key>KeepAlive</key><true/>", agent)


if __name__ == "__main__":
    unittest.main()
