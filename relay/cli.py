"""Relay command line.

  relay serve                       run the proxy
  relay install <harness>           point the harness's own config at the proxy
  relay uninstall <harness>         restore the harness's config
  relay run <harness> [args...]     run a harness through a private proxy
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time

import uvicorn

from . import install
from .harnesses import HARNESSES
from .transport import ProxyConfig, create_app

ALIASES = {"claude": "claude_code", "dsh": "deepseek_harness", "mini": "mini_swe"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relay", description=__doc__)
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="run the proxy (default)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    harnesses = sorted({*HARNESSES, *ALIASES} - {"generic"})
    for name, text in (("install", "point a harness at Relay"), ("uninstall", "restore a harness")):
        command = commands.add_parser(name, help=text)
        command.add_argument("harness", choices=harnesses)
        command.add_argument("--url", help="Relay's address (default: RELAY_HOST/RELAY_PORT)")
    run = commands.add_parser("run", help="run a harness through a private proxy")
    run.add_argument("harness", choices=harnesses)
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the harness")
    options = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("RELAY_LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(message)s")

    config = ProxyConfig.from_env()
    if options.command in {"install", "uninstall"}:
        harness = HARNESSES[ALIASES.get(options.harness, options.harness)]
        try:
            if options.command == "uninstall":
                paths = sorted(set(install.uninstall(harness.name)))
            else:
                settings = harness.settings()
                install.install(harness.name, options.url or f"http://{config.host}:{config.port}", settings)
                paths = sorted({s.path for s in settings})
        except ValueError as error:
            print(f"relay {options.command}: {error}", file=sys.stderr)
            return 1
        print(f"{options.command}ed {harness.name}: " + ", ".join(map(str, paths)))
        return 0
    if options.command == "run":
        return _run(config, ALIASES.get(options.harness, options.harness), options.args)
    host = getattr(options, "host", None) or config.host
    port = getattr(options, "port", None) or config.port
    uvicorn.run(create_app(config=config), host=host, port=port, ws="none")
    return 0


def _run(config: ProxyConfig, harness: str, args: list[str]) -> int:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_app(config=config), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.01)
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    command, env = HARNESSES[harness].launch(url, [a for a in args if a != "--"])
    try:
        return subprocess.run(command, env={**os.environ, **env}, check=False).returncode
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
