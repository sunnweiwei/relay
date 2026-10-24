"""Relay as an HTTPS proxy (`relay.intercept`): it opens api.anthropic.com's connections, with a
CA that can vouch for nothing else, and tunnels every other one as it is."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from relay import install, intercept
from relay.integrations import ClaudeCodeHttps
from tests.fakes import serve


def _listener(handler) -> tuple[int, threading.Thread]:
    """An asyncio server of `handler` on a free port, in a thread of its own."""

    ready, box = threading.Event(), {}

    def run() -> None:
        async def main() -> None:
            server = await handler()
            box["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            async with server:
                await server.serve_forever()
        asyncio.run(main())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(10)
    return box["port"], thread


class InterceptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        patcher = patch.dict(os.environ, {"RELAY_HOME": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_ca_vouches_for_api_anthropic_com_alone_and_its_keys_are_the_owners(self) -> None:
        from cryptography import x509
        ca = x509.load_pem_x509_certificate(intercept.ca_file().read_bytes())
        constraints = ca.extensions.get_extension_for_class(x509.NameConstraints)
        self.assertTrue(constraints.critical)
        self.assertEqual([n.value for n in constraints.value.permitted_subtrees], ["api.anthropic.com"])
        for name in ("ca.key", "leaf.key"):
            self.assertEqual(stat.S_IMODE((intercept.folder() / name).stat().st_mode), 0o600)
        before = (intercept.folder() / "leaf.pem").read_bytes()
        intercept.ensure()
        self.assertEqual((intercept.folder() / "leaf.pem").read_bytes(), before)  # (made once)

    def test_api_anthropic_com_is_opened_for_relay_and_another_host_tunnelled(self) -> None:
        seen = []

        async def answer(request: Request) -> JSONResponse:
            seen.append((request.url.path, request.headers.get("host")))
            return JSONResponse({"through": "relay"})

        echoed = []

        async def echo():  # another host: what it is sent comes back as it was (no TLS opened)
            async def handle(reader, writer):
                data = await reader.read(100)
                echoed.append(data)
                writer.write(data)
                await writer.drain()
                writer.close()
            return await asyncio.start_server(handle, "127.0.0.1", 0)

        with serve(Starlette(routes=[Route("/{path:path}", answer, methods=["GET", "POST"])])) as relay:
            endpoint = int(relay.rsplit(":", 1)[1])
            proxy, _ = _listener(lambda: intercept.serve("127.0.0.1", 0, endpoint))
            trusted = ssl.create_default_context(cafile=str(intercept.ca_file()))
            with httpx.Client(proxy=f"http://127.0.0.1:{proxy}", verify=trusted) as client:
                self.assertEqual(client.post("https://api.anthropic.com/v1/messages", json={}).json(), {"through": "relay"})
            self.assertEqual(seen, [("/v1/messages", "api.anthropic.com")])
            other, _ = _listener(echo)

            async def tunnel() -> bytes:
                reader, writer = await asyncio.open_connection("127.0.0.1", proxy)
                writer.write(f"CONNECT 127.0.0.1:{other} HTTP/1.1\r\nHost: 127.0.0.1:{other}\r\n\r\n".encode())
                await writer.drain()
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"\x16\x03\x01raw bytes")
                await writer.drain()
                return await reader.read(100)

            self.assertEqual(asyncio.run(tunnel()), b"\x16\x03\x01raw bytes")
            self.assertEqual(echoed, [b"\x16\x03\x01raw bytes"])

    def test_any_path_it_opens_goes_to_anthropic(self) -> None:
        from relay.providers import route, upstreams_from_settings
        upstreams = upstreams_from_settings()
        for path in ("/v1/messages", "/v1/environments/bridge", "/v1/sessions/s1/events/stream", "/v1/code/sessions"):
            self.assertIs(route(path, {"host": "api.anthropic.com"}, upstreams), upstreams["anthropic"], path)  # (Remote Control's)

    def test_a_server_side_thread_is_declined_as_a_server_without_threads_does(self) -> None:
        from relay import Compaction, Engine, ProxyConfig, create_app
        from starlette.testclient import TestClient
        from tests.fakes import FakeUpstream, local_upstreams
        fake = FakeUpstream(tool_calls=0)
        with serve(fake.app) as upstream:
            app = create_app(Engine(Compaction()), ProxyConfig(upstreams=local_upstreams(upstream)))
            with TestClient(app) as client:
                body = {"model": "m", "max_tokens": 10, "thread": {"type": "create"}, "stream": False,
                        "messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Bash", "input_schema": {}}]}
                refused = client.post("/v1/messages", json=body, headers={"anthropic-version": "2023-06-01"})
        self.assertEqual(refused.status_code, 400)  # (Claude Code then sends the turn again, without threads)
        self.assertEqual(refused.json()["error"]["details"]["error_code"], "thread_unsupported_request")
        self.assertEqual(fake.bodies("/v1/messages"), [])  # (nothing reached the upstream)

    def test_relay_forgets_the_proxy_it_inherits_that_is_its_own(self) -> None:
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:9100", "HTTP_PROXY": "http://corp:3128"}):
            intercept.forget_own(9100)
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertEqual(os.environ["HTTP_PROXY"], "http://corp:3128")


class ClaudeCodeHttpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        patcher = patch.dict(os.environ, {"RELAY_HOME": str(self.root / "relay"), "CLAUDE_CONFIG_DIR": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root / "relay").mkdir()
        (self.root / "relay" / "settings.json").write_text(json.dumps({"intercept_port": 9100}))

    def test_claude_code_keeps_its_endpoint_and_trusts_relays_ca(self) -> None:
        original = '{\n  "env": {\n    "ANTHROPIC_API_KEY": "k"\n  }\n}\n'
        (self.root / "settings.json").write_text(original)
        install.install("claude_code", "http://127.0.0.1:8787", ClaudeCodeHttps().installation("http://127.0.0.1:8787").settings)
        env = json.loads((self.root / "settings.json").read_text())["env"]
        self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:9100")
        self.assertEqual(env["NODE_EXTRA_CA_CERTS"], str(intercept.folder() / "ca.pem"))
        self.assertNotIn("ANTHROPIC_BASE_URL", env)  # (Remote Control needs Anthropic's own)
        install.uninstall("claude_code")
        self.assertEqual((self.root / "settings.json").read_text(), original)

    def test_the_users_cas_are_kept_beside_relays(self) -> None:
        theirs = self.root / "corp.pem"
        theirs.write_text("-----BEGIN CERTIFICATE-----\nCORP\n-----END CERTIFICATE-----\n")
        with patch.dict(os.environ, {"NODE_EXTRA_CA_CERTS": str(theirs)}):
            settings = ClaudeCodeHttps().installation("http://127.0.0.1:8787").settings
        bundle = Path(next(s.value for s in settings if s.key[-1] == "NODE_EXTRA_CA_CERTS"))
        self.assertIn("CORP", bundle.read_text())
        self.assertIn(intercept.ca_file().read_text().strip(), bundle.read_text())

    def test_a_proxy_claude_code_already_goes_through_is_not_replaced(self) -> None:
        self.assertIsNone(ClaudeCodeHttps().unavailable())
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://corp:3128"}):
            self.assertIn("already goes through a proxy (http://corp:3128", ClaudeCodeHttps().unavailable())
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:9100"}):
            self.assertIsNone(ClaudeCodeHttps().unavailable())  # (Relay's own)


if __name__ == "__main__":
    unittest.main()
