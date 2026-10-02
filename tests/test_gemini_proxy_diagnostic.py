"""Diagnose native Gemini protocol routing; passes do NOT mean strategies ran."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tests.test_codex_e2e import _serve
from tests.harness_e2e_support import STRATEGIES, clean_env, relay_for, strategy_for


class GeminiProxyDiagnosticTests(unittest.TestCase):
    def fake_upstream(self):
        captured = []

        async def dispatch(request):
            body = await request.json()
            captured.append((request.url.path, body))
            if request.url.path.endswith(":countTokens"):
                return JSONResponse({"totalTokens": 20})
            payload = {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "RELAY_GEMINI_NATIVE_OK"}]}, "finishReason": "STOP", "index": 0}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5,
                                  "totalTokenCount": 25}}
            if request.url.path.endswith(":streamGenerateContent"):
                return Response("data: " + json.dumps(payload) + "\n\n",
                                media_type="text/event-stream")
            return JSONResponse(payload)

        return Starlette(routes=[Route("/{path:path}", dispatch, methods=["POST"])]), captured

    def test_real_cli_uses_native_protocol_and_bypasses_strategy(self):
        binary = os.environ.get("RELAY_GEMINI_BIN") or shutil.which("gemini")
        if not binary:
            self.skipTest("Install Gemini CLI or set RELAY_GEMINI_BIN")
        upstream, captured = self.fake_upstream()
        strategy = strategy_for("sliding_window")
        with tempfile.TemporaryDirectory() as directory, _serve(upstream) as upstream_url:
            root = Path(directory)
            env = clean_env(root)
            config_dir = root / "gemini-config"
            config_dir.mkdir()
            settings_dir = config_dir / ".gemini"
            settings_dir.mkdir()
            shutil.copyfile(Path(__file__).parent / "fixtures" / "gemini_proxy_settings.json",
                            settings_dir / "settings.json")
            env.update({"GEMINI_CLI_HOME": str(config_dir), "GEMINI_API_KEY": "fake-local-key",
                        "GEMINI_CLI_TRUST_WORKSPACE": "true",
                        "GOOGLE_GEMINI_BASE_URL": "", "GOOGLE_GENAI_API_VERSION": "v1beta"})
            relay, _, _ = relay_for(strategy, upstream_url)
            with patch.object(strategy, "prepare", wraps=strategy.prepare) as prepare, _serve(relay) as relay_url:
                env["GOOGLE_GEMINI_BASE_URL"] = relay_url
                result = subprocess.run([binary, "-p", "Reply only RELAY_GEMINI_NATIVE_OK; do not use tools.",
                                         "-m", "gemini-2.5-flash", "--output-format", "json"],
                                        cwd=root, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RELAY_GEMINI_NATIVE_OK", result.stdout)
            self.assertTrue(any(path.endswith(":streamGenerateContent") for path, _ in captured))
            self.assertTrue(any("contents" in body for _, body in captured))
            self.assertEqual(prepare.call_count, 0, "Native requests bypass context management")

    def test_native_route_bypasses_every_strategy(self):
        upstream, _ = self.fake_upstream()
        with _serve(upstream) as upstream_url:
            for name in STRATEGIES:
                with self.subTest(strategy=name):
                    strategy = strategy_for(name)
                    relay, _, _ = relay_for(strategy, upstream_url)
                    with patch.object(strategy, "prepare", wraps=strategy.prepare) as prepare, _serve(relay) as relay_url:
                        response = httpx.post(relay_url + "/v1beta/models/gemini-2.5-flash:generateContent",
                                              json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(prepare.call_count, 0)


if __name__ == "__main__":
    unittest.main()
