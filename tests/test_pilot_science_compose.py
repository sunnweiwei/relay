"""Actual transport checks for independent Science service lifetimes."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
import http.client
import json
from pathlib import Path
import threading

from evals.pilot.science_sidecar_transport import (
    ThreadedHTTPServer, UnixHTTPServer, gateway, proxy_handler,
)
from evals.pilot.science_compose import TaskServices


def test_concurrent_service_gateways_preserve_payload_and_isolate_tokens(tmp_path: Path) -> None:
    servers = []
    endpoints = []
    try:
        for index in range(2):
            class Echo(BaseHTTPRequestHandler):
                def do_POST(self):
                    data = self.rfile.read(int(self.headers["Content-Length"]))
                    payload = json.dumps({"path": self.path, "body": data.decode(),
                                          "job": self.server.job}).encode()
                    self.send_response(201)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *_args):
                    return

            backend = ThreadedHTTPServer(("127.0.0.1", 0), Echo)
            backend.job = index
            path = tmp_path / ("long-experiment-directory-" * 5) / str(index) / "bridge.sock"
            path.parent.mkdir(parents=True)
            unix = UnixHTTPServer(str(path), proxy_handler(
                lambda target, port=backend.server_port: (
                    http.client.HTTPConnection("127.0.0.1", port), target)))
            public = gateway(f"token-{index}", {"broker": path})
            servers.extend((backend, unix, public))
            for server in (backend, unix, public):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            endpoints.append(public.server_port)
        assert endpoints[0] != endpoints[1]

        def request(index: int, token: str):
            connection = http.client.HTTPConnection("127.0.0.1", endpoints[index], timeout=5)
            connection.request("POST", f"/{token}/broker/act?case=example", b'{"action":"observe"}')
            response = connection.getresponse()
            result = response.status, response.read()
            connection.close()
            return result

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(request, i % 2, f"token-{i % 2}") for i in range(12)]
            for i, future in enumerate(futures):
                status, payload = future.result()
                assert status == 201
                assert json.loads(payload) == {"path": "/act?case=example",
                    "body": '{"action":"observe"}', "job": i % 2}
        assert request(0, "token-1")[0] == 403
        assert request(1, "token-0")[0] == 403
    finally:
        for server in reversed(servers):
            server.shutdown()
            server.server_close()


def test_service_qualified_artifacts_come_from_their_own_filesystem(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('''artifacts = [
      {source="/root/results/answer.json", service="main"},
      {source="/var/log/actions.json", service="broker"}]
      [task]
      name="terminal-bench-science/longitudinal-clinical-agent"
    ''')
    manager = TaskServices(task, tmp_path / "images", tmp_path / "services", Path("/unused"))
    manager.output.mkdir()
    for name, relative, value in [("main", "root/results/answer.json", "answer"),
                                   ("broker", "var/log/actions.json", "authoritative")]:
        root = tmp_path / name
        manager.roots[name] = root
        target = root / relative
        target.parent.mkdir(parents=True)
        target.write_text(value)
    forged = manager.roots["main"] / "var/log/actions.json"
    forged.parent.mkdir(parents=True)
    forged.write_text("forged")
    records = manager.freeze(tmp_path / "frozen")
    assert (tmp_path / "frozen/var/log/actions.json").read_text() == "authoritative"
    assert [x["service"] for x in records] == ["main", "broker"]
