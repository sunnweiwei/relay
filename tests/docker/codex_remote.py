"""A stand-in for ChatGPT's Remote Control service and the app the user drives Codex from.

    python3 codex_remote.py serve          the service, on https://127.0.0.1:8900 (`chatgpt_base_url`)
    python3 codex_remote.py say "<text>"   the user sends a message in the app; prints the reply

Codex reaches Remote Control at `chatgpt_base_url` and the model at `openai_base_url`, so Relay
sits only on the model path. Only the Remote Control service is replaced: Codex's other ChatGPT
backend calls (account and workspace lookup before each turn, plugins, analytics) pass through to
chatgpt.com unchanged. The workspace lookup accepts only an HTTPS backend, so the service has a
certificate from a test CA that Codex alone trusts (with the system's). The first message starts
`codex app-server --remote-control` (what `codex remote-control` runs), which enrolls here and
opens the service's WebSocket; the app is one client on it, as in the protocol of
codex-rs/app-server-transport/src/transport/remote_control. The app starts a thread, sends each
message as a turn, and after each turn reads the thread back as the app shows it (~/thread.json).
Every message the app receives is kept in ~/app.jsonl.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from aiohttp import ClientSession, WSMsgType, web

PORT, APP_PORT = 8900, 8901  # the service; where `say` reaches the app
CHATGPT = "https://chatgpt.com"
SERVICE = "/backend-api/wham/remote/control/server"
ENROLLMENT = {"server_id": "srv_relay_test", "environment_id": "env_relay_test",
              "remote_control_token": "relay-test-token", "expires_at": "2999-01-01T00:00:00Z"}
CLIENT, STREAM = "app", "app-stream-1"
HOME = Path.home()
TLS = HOME / "chatgpt-standin"


class App:
    """The user's app: a client of the app-server, through the service's WebSocket."""

    def __init__(self) -> None:
        self.ws: web.WebSocketResponse | None = None
        self.seq = itertools.count()  # this client's message sequence
        self.ids = itertools.count(1)
        self.pending: dict[int, asyncio.Future] = {}
        self.chunks: dict[int, str] = {}
        self.ready = asyncio.Event()  # connected, initialized, thread started
        self.thread = ""
        self.done: asyncio.Queue = asyncio.Queue()  # turn/completed notifications
        self.replies: list[str] = []

    async def send(self, message: dict) -> None:
        envelope = {"type": "client_message", "client_id": CLIENT, "stream_id": STREAM,
                    "seq_id": next(self.seq), "message": message}
        await self.ws.send_json(envelope)

    async def request(self, method: str, params: dict) -> dict:
        id = next(self.ids)
        self.pending[id] = asyncio.get_running_loop().create_future()
        await self.send({"id": id, "method": method, "params": params})
        return await self.pending[id]

    async def start(self) -> None:
        await self.request("initialize", {"clientInfo": {"name": "relay-remote-test", "version": "0.1.0"}})
        await self.send({"method": "initialized"})
        if not self.thread:
            started = await self.request("thread/start", {"cwd": "/project", "approvalPolicy": "never",
                                                          "sandbox": "danger-full-access"})
            self.thread = started["thread"]["id"]
        else:  # the app-server reconnected: pick the thread up again
            await self.request("thread/resume", {"threadId": self.thread})
        self.ready.set()

    async def receive(self, envelope: dict) -> None:
        kind = envelope.get("type")
        if kind == "server_message_chunk":  # a large message, in base64 pieces
            self.chunks[envelope["segment_id"]] = envelope["message_chunk_base64"]
            if len(self.chunks) < envelope["segment_count"]:
                return
            data = b"".join(base64.b64decode(self.chunks[i]) for i in sorted(self.chunks))
            self.chunks.clear()
            message = json.loads(data)
        elif kind == "server_message":
            message = envelope["message"]
        else:
            return
        await self.ws.send_json({"type": "ack", "client_id": CLIENT, "stream_id": envelope["stream_id"],
                                 "seq_id": envelope["seq_id"]})
        with (HOME / "app.jsonl").open("a") as log:
            log.write(json.dumps(message) + "\n")
        if "method" not in message:  # a response
            future = self.pending.pop(message.get("id"), None)
            if future and not future.done():
                if "error" in message:
                    future.set_exception(RuntimeError(json.dumps(message["error"])))
                else:
                    future.set_result(message.get("result") or {})
        elif "id" in message:  # the app-server asks the user something; approvals are off
            await self.send({"id": message["id"], "error": {"code": -32601, "message": "not supported by this app"}})
        elif message["method"] == "item/completed" and message["params"]["item"].get("type") == "agentMessage":
            self.replies.append(message["params"]["item"]["text"])
        elif message["method"] == "turn/completed":
            await self.done.put(message["params"])

    async def say(self, text: str) -> str:
        await self.ready.wait()
        self.replies.clear()
        await self.request("turn/start", {"threadId": self.thread, "input": [{"type": "text", "text": text}]})
        await self.done.get()
        shown = await self.request("thread/read", {"threadId": self.thread, "includeTurns": True})
        (HOME / "thread.json").write_text(json.dumps(shown, indent=1))
        return self.replies[-1] if self.replies else ""


def certificate() -> ssl.SSLContext:
    """The service's certificate, from a CA in the bundle that Codex is given."""

    def openssl(*args: str) -> None:
        subprocess.run(["openssl", *args], cwd=TLS, check=True, capture_output=True)

    TLS.mkdir(exist_ok=True)
    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", "/CN=Relay test CA",
            "-keyout", "ca.key", "-out", "ca.pem")
    openssl("req", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1", "-keyout", "service.key",
            "-out", "service.csr")
    (TLS / "service.ext").write_text("subjectAltName=IP:127.0.0.1\nbasicConstraints=CA:FALSE\n")
    openssl("x509", "-req", "-in", "service.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
            "-days", "2", "-extfile", "service.ext", "-out", "service.pem")
    (TLS / "bundle.pem").write_text(Path("/etc/ssl/certs/ca-certificates.crt").read_text() + (TLS / "ca.pem").read_text())
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(TLS / "service.pem", TLS / "service.key")
    return context


def serve() -> None:
    app = App()
    codex: list[asyncio.subprocess.Process] = []

    async def enroll(request: web.Request) -> web.Response:  # also the token refresh
        return web.json_response(ENROLLMENT)

    async def connect(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        app.ws, app.seq = ws, itertools.count()
        asyncio.create_task(app.start())
        async for frame in ws:
            if frame.type == WSMsgType.TEXT:
                await app.receive(json.loads(frame.data))
        app.ready.clear()
        return ws

    async def say(request: web.Request) -> web.Response:
        if not codex:  # the user turns Remote Control on
            log = open(HOME / "app-server.log", "ab")
            codex.append(await asyncio.create_subprocess_exec(
                "codex", "app-server", "--remote-control", "--listen", "off", stdout=log, stderr=log,
                env={**os.environ, "CODEX_CA_CERTIFICATE": str(TLS / "bundle.pem")}))
        return web.Response(text=await app.say(await request.text()))

    async def chatgpt(request: web.Request) -> web.Response:  # the rest of the ChatGPT backend
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        async with ClientSession() as session, session.request(
                request.method, CHATGPT + request.path_qs, headers=headers, data=await request.read()) as reply:
            print(request.method, request.path, reply.status, flush=True)
            return web.Response(status=reply.status, body=await reply.read(),  # decoded: Codex asks for no encoding
                                content_type=reply.content_type)

    server = web.Application(client_max_size=2**26)
    server.router.add_post(SERVICE + "/enroll", enroll)
    server.router.add_post(SERVICE + "/refresh", enroll)
    server.router.add_get(SERVICE, connect)
    server.router.add_post("/say", say)
    server.router.add_route("*", "/{tail:.*}", chatgpt)

    async def run() -> None:
        runner = web.AppRunner(server)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", PORT, ssl_context=certificate()).start()
        await web.TCPSite(runner, "127.0.0.1", APP_PORT).start()
        await asyncio.Event().wait()

    asyncio.run(run())


def say(text: str) -> None:
    for _ in range(100):  # the service may still be starting
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{APP_PORT}/say", text.encode(), timeout=900) as reply:
                print(reply.read().decode())
                return
        except urllib.error.URLError as error:
            if not isinstance(error.reason, ConnectionRefusedError):
                raise
        time.sleep(0.2)


if __name__ == "__main__":
    serve() if sys.argv[1] == "serve" else say(sys.argv[2])
