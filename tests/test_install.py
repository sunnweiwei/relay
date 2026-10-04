from __future__ import annotations

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from relay import install
from relay.harnesses import HARNESSES
from relay.install import Setting
from relay.providers import route, upstreams_from_env


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        patcher = patch.dict(os.environ, {"RELAY_HOME": str(self.root / "relay")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_endpoints_are_wrapped_and_restored_in_every_format(self) -> None:
        files = {
            "a.json": '{"env": {"BASE": "https://gateway.example/anthropic"}, "keep": 1}\n',
            "b.toml": 'top = "x"\n\n[providers.kimi]\ntype = "kimi"\nbase_url = "https://api.kimi.com/coding/v1"\n',
            "c.yaml": "GOOSE_PROVIDER: openai\nextensions:\n  developer:\n    enabled: true\n",
            "d.env": "OTHER=1\n",
        }
        for name, text in files.items():
            (self.root / name).write_text(text)
        settings = [
            Setting(self.root / "a.json", ("env", "BASE"), endpoint="https://api.anthropic.com"),
            Setting(self.root / "b.toml", ("providers", "kimi", "base_url"), endpoint="unused"),
            Setting(self.root / "c.yaml", ("OPENAI_HOST",), endpoint="https://api.openai.com"),
            Setting(self.root / "d.env", ("GOOGLE_GEMINI_BASE_URL",), endpoint="https://generativelanguage.googleapis.com"),
            Setting(self.root / "a.json", ("prune",), value=False),
        ]
        install.install("h", "http://127.0.0.1:8787", settings)

        self.assertEqual(json.loads((self.root / "a.json").read_text())["env"]["BASE"],
                         "http://127.0.0.1:8787/up/h-0/anthropic")
        self.assertIn('base_url = "http://127.0.0.1:8787/up/h-1/coding/v1"', (self.root / "b.toml").read_text())
        self.assertIn("OPENAI_HOST: http://127.0.0.1:8787/up/h-2\n", (self.root / "c.yaml").read_text())
        self.assertIn("GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8787/up/h-3", (self.root / "d.env").read_text())
        self.assertEqual(install.mounts()["/up/h-1"], "https://api.kimi.com")
        upstream = route("/up/h-0/anthropic/v1/messages", {}, upstreams_from_env())
        self.assertEqual(upstream.url("/up/h-0/anthropic/v1/messages"), "https://gateway.example/anthropic/v1/messages")

        install.uninstall("h")
        for name, text in files.items():
            self.assertEqual((self.root / name).read_text().strip(), text.strip() if name != "a.json"
                             else json.dumps(json.loads(text), indent=2))
        self.assertEqual(install.mounts(), {})

    def test_harness_auto_compaction_is_turned_off_and_restored(self) -> None:
        """Relay compacts instead: each harness's own trigger is switched off (or set beyond reach)
        in its own format, next to what the user had, and uninstall puts it back."""

        files = {
            "codex/config.toml": 'model = "gpt-6-luna"\n\n[mcp_servers.docs]\ncommand = "docs"\n',
            "kimi/config.toml": '[providers.test]\ntype = "openai_responses"\n\n[models.test]\nprovider = "test"\n'
                                'max_context_size = 200000\n',
            "hermes/config.yaml": "model:\n  default: gpt-6-luna\n  provider: openai-api\n",
            "dsh/cordis.patch.yml": "- id: compaction-basic\n  config:\n    thresholdRatio: 0.7\n",
        }
        for name, text in files.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text)
        env = {"CODEX_HOME": "codex", "KIMI_CODE_HOME": "kimi", "HERMES_HOME": "hermes", "DSH_HOME": "dsh"}
        with patch.dict(os.environ, {key: str(self.root / value) for key, value in env.items()}):
            for name in ("codex", "kimi_code", "hermes", "deepseek_harness"):
                install.install(name, "http://127.0.0.1:8787", HARNESSES[name].settings())
            codex = tomllib.loads((self.root / "codex/config.toml").read_text())
            self.assertEqual((codex["model_auto_compact_token_limit"], codex["mcp_servers"]), (10**9, {"docs": {"command": "docs"}}))
            self.assertIn("max_context_size = 1000000000", (self.root / "kimi/config.toml").read_text())
            self.assertIn("compression:\n  enabled: false", (self.root / "hermes/config.yaml").read_text())
            self.assertIn("thresholdRatio: 0.7\n    auto: false", (self.root / "dsh/cordis.patch.yml").read_text())
            for name in ("deepseek_harness", "hermes", "kimi_code", "codex"):
                install.uninstall(name)
        for name, text in files.items():
            self.assertEqual((self.root / name).read_text(), text, name)

    def test_nested_yaml_and_list_items(self) -> None:
        patch_file = self.root / "cordis.patch.yml"
        text = (
            "- id: agent-default-model\n"
            "  config:\n"
            "    provider: openai\n"
            "- id: llm-pi-ai\n"
            "  config:\n"
            "    providers:\n"
            "      openai:\n"
            "        apiKeyEnv: OPENAI_API_KEY\n"
            "      acme:\n"
            "        baseURL: https://gw.acme.example/v1  # gateway\n"
        )
        patch_file.write_text(text)
        routes = ("[id=llm-pi-ai]", "config", "providers")
        self.assertEqual(install.yaml_keys(patch_file, routes), ["openai", "acme"])
        self.assertEqual(install.read(patch_file, (*routes, "acme", "baseURL")), "https://gw.acme.example/v1")
        settings = [
            Setting(patch_file, (*routes, "openai", "baseURL"), endpoint="https://api.openai.com/v1"),
            Setting(patch_file, (*routes, "acme", "baseURL"), endpoint=""),
            Setting(patch_file, ("[id=llm-deepseek]", "config", "baseURL"), endpoint="https://api.deepseek.com/anthropic"),
        ]
        install.install("d", "http://relay", settings)
        self.assertEqual(install.read(patch_file, (*routes, "openai", "baseURL")), "http://relay/up/d-0/v1")
        self.assertEqual(install.read(patch_file, (*routes, "acme", "baseURL")), "http://relay/up/d-1/v1")
        self.assertTrue(patch_file.read_text().endswith(
            "- id: llm-deepseek\n  config:\n    baseURL: http://relay/up/d-2/anthropic\n"))
        self.assertEqual(install.mounts(), {"/up/d-0": "https://api.openai.com", "/up/d-1": "https://gw.acme.example",
                                            "/up/d-2": "https://api.deepseek.com"})

        install.uninstall("d")
        restored = patch_file.read_text()
        self.assertEqual(restored, text.replace("  # gateway", ""))  # inline comments on edited lines go

    def test_json_list_items(self) -> None:
        models = self.root / "models.json"
        text = {"models": [{"id": "a", "url": "https://gw.example/v1/chat/completions"}, {"id": "b"}]}
        models.write_text(json.dumps(text))
        install.install("w", "http://relay", [
            Setting(models, ("models", "[id=a]", "url"), endpoint=""),
            Setting(models, ("models", "[id=new]", "url"), endpoint="https://api.openai.com/v1/chat/completions"),
        ])
        entries = json.loads(models.read_text())["models"]
        self.assertEqual(entries[0]["url"], "http://relay/up/w-0/v1/chat/completions")
        self.assertEqual(entries[2], {"id": "new", "url": "http://relay/up/w-1/v1/chat/completions"})
        install.uninstall("w")
        self.assertEqual(json.loads(models.read_text()), text)

    def test_uninstall_removes_containers_it_created(self) -> None:
        config = self.root / "config.json"
        config.write_text('{"agents": {}}')
        install.install("n", "http://relay", [Setting(config, ("providers", "gemini", "apiBase"), endpoint="https://x.example")])
        self.assertEqual(json.loads(config.read_text())["providers"]["gemini"]["apiBase"], "http://relay/up/n-0")
        install.uninstall("n")
        self.assertEqual(json.loads(config.read_text()), {"agents": {}})
