"""HTTP transport for isolated Science sidecars, using private Unix sockets.

The sidecar's public HTTP API is unchanged. A host gateway holds a randomly
assigned port and authenticates the URL prefix before forwarding requests.
Sidecar files and the Unix transport never enter the agent's filesystem.
This module uses only the standard library so it can run inside task images.
"""
from __future__ import annotations

import argparse
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import socket
from socketserver import ThreadingMixIn
import subprocess
import threading
from typing import Any, Callable
from urllib.parse import urlsplit


HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate",
                         "proxy-authorization", "te", "trailer", "transfer-encoding",
                         "upgrade", "host", "content-length"})


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path | str, *, timeout: float = 330) -> None:
        super().__init__("localhost", timeout=timeout)
        self.path = str(path)

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        # Experiment paths exceed sockaddr_un's 108-byte path limit. Resolve
        # through an open directory rather than creating shared short symlinks.
        descriptor = os.open(str(Path(self.path).parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.sock.connect(f"/proc/self/fd/{descriptor}/{Path(self.path).name}")
        finally:
            os.close(descriptor)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class UnixHTTPServer(ThreadedHTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self) -> None:
        path = Path(self.server_address)
        descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            socket.socket.bind(self.socket, f"/proc/self/fd/{descriptor}/{path.name}")
        finally:
            os.close(descriptor)
        self.server_name, self.server_port = "localhost", 0


Route = Callable[[str], tuple[http.client.HTTPConnection, str] | None]


def proxy_handler(route: Route) -> type[BaseHTTPRequestHandler]:
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def forward(self) -> None:
            selected = route(self.path)
            if selected is None:
                self.send_error(403, "Unknown service or invalid access token")
                return
            connection, path = selected
            try:
                # Official clients send Content-Length, including zero for GET.
                # Do not silently reinterpret a chunked request as an empty body.
                if self.headers.get("Transfer-Encoding"):
                    self.send_error(501, "Chunked request bodies are unsupported")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    raise ValueError("negative request size")
                body = self.rfile.read(length) if length else None
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
                connection.request(self.command, path, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
            except (OSError, ValueError, http.client.HTTPException):
                self.send_error(502, "Sidecar service transport failed")
            finally:
                connection.close()

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = forward

        def log_message(self, *_args: Any) -> None:
            # The request URL contains a per-run credential.
            return

    return Proxy


def gateway(token: str, sockets: dict[str, Path]) -> ThreadedHTTPServer:
    """Bind port zero once, retaining the socket through the service lifetime."""
    def route(path: str) -> tuple[http.client.HTTPConnection, str] | None:
        parsed = urlsplit(path)
        parts = parsed.path.split("/", 3)
        if len(parts) < 3 or not hmac.compare_digest(parts[1], token):
            return None
        target = sockets.get(parts[2])
        if target is None:
            return None
        forwarded = "/" + (parts[3] if len(parts) == 4 else "")
        if parsed.query:
            forwarded += "?" + parsed.query
        return UnixConnection(target), forwarded

    return ThreadedHTTPServer(("127.0.0.1", 0), proxy_handler(route))


def serve_sidecar(config: dict[str, Any]) -> int:
    """Supervise an unchanged trusted service inside its private namespace."""
    servers: list[ThreadedHTTPServer] = []
    socket_path = Path(config["socket"])
    if socket_path.exists():
        raise FileExistsError(socket_path)
    port = int(config["port"])
    local = UnixHTTPServer(str(socket_path), proxy_handler(
        lambda path: (http.client.HTTPConnection("127.0.0.1", port, timeout=330), path)))
    servers.append(local)
    socket_path.chmod(0o600)
    # For the protein task, the unchanged submission service calls the assay
    # at its official hostname and port. Its local peer proxy uses only the
    # other sidecar's Unix socket, not the host's network namespace.
    for peer in config.get("peers", []):
        path = Path(peer["socket"])
        servers.append(ThreadedHTTPServer(("127.0.0.1", int(peer["port"])),
            proxy_handler(lambda target, path=path: (UnixConnection(path), target))))
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    child = subprocess.Popen(config["command"], cwd=config["workdir"])

    def stop(_signum: int, _frame: Any) -> None:
        child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return child.wait()
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        socket_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return serve_sidecar(json.loads(args.config.read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
