"""Relay command line.

  relay serve                       run the proxy (and the endpoint harness hooks call)
  relay install <harness> [--via]   set the harness up to run Relay's strategy: through the
                                    proxy, or through a hook the harness offers
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

from . import install, integrations
from .harnesses import HARNESSES, Harness
from .transport import ProxyConfig, create_app

ALIASES = {"claude": "claude_code", "dsh": "deepseek_harness", "mini": "mini_swe"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relay", description=__doc__)
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="run the proxy (default)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    harnesses = sorted({*HARNESSES, *ALIASES} - {"generic"})
    install_ = commands.add_parser("install", help="set a harness up to run Relay's strategy")
    uninstall_ = commands.add_parser("uninstall", help="restore a harness")
    for command in (install_, uninstall_):
        command.add_argument("harness", choices=harnesses)
    install_.add_argument("--url", help="Relay's address (default: RELAY_HOST/RELAY_PORT)")
    install_.add_argument("--via", choices=["auto", "proxy", "hook"], default="auto",
                          help="integration path (default: the harness's preferred one)")
    run = commands.add_parser("run", help="run a harness through a private proxy")
    run.add_argument("harness", choices=harnesses)
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the harness")
    options = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("RELAY_LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(message)s")

    config = ProxyConfig.from_env()
    if options.command == "install":
        url = options.url or f"http://{config.host}:{config.port}"
        return _install(HARNESSES[ALIASES.get(options.harness, options.harness)], options.via, url)
    if options.command == "uninstall":
        return _uninstall(HARNESSES[ALIASES.get(options.harness, options.harness)])
    if options.command == "run":
        return _run(config, ALIASES.get(options.harness, options.harness), options.args)
    host = getattr(options, "host", None) or config.host
    port = getattr(options, "port", None) or config.port
    uvicorn.run(create_app(config=config), host=host, port=port, ws="none")
    return 0


def _install(harness: Harness, via: str, url: str) -> int:
    try:
        path = integrations.choose(harness, via)
        setup = path.installation(url)
        install.install(harness.name, url, setup.settings)
    except ValueError as error:
        print(f"relay install: {error}", file=sys.stderr)
        return 1
    print(f"installed {harness.name} ({path.name}): " + ", ".join(sorted({str(s.path) for s in setup.settings})))
    for note in setup.notes:
        print(f"  note: {note}")
    return 0


def _uninstall(harness: Harness) -> int:
    try:
        paths = install.uninstall(harness.name)
    except ValueError as error:
        print(f"relay uninstall: {error}", file=sys.stderr)
        return 1
    print(f"uninstalled {harness.name}: " + ", ".join(sorted({str(path) for path in paths})))
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
