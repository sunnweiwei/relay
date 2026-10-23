"""Relay command line (`relay help` for the guide).

  relay setup <harness> [--strategy S] [--set name=value ...] [--autostart]
                                    save the settings, start Relay, point the harness at it
  relay strategy [S] [--set ...]    show or change the strategy; the running Relay follows at once
  relay settings | set name=value ... | unset name ...
  relay status                      whether Relay runs, its strategy, the harnesses set up
  relay uninstall <harness> | --all restore the harness's config (the last one: Relay is no longer
                                    started at login; it runs on for conversations still open)
  relay start                       make sure Relay runs (harness hooks run it as a session starts)
  relay serve                       run Relay in the foreground (what the service runs)
  relay install <harness> [--via]   only point a harness at Relay (through the proxy, or a hook)
  relay run <harness> [args...]     run a harness through a private proxy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn

from . import install, integrations, leaving, service, settings
from .harnesses import HARNESSES, Harness
from .strategies import STRATEGIES
from .transport import ProxyConfig, create_app

ALIASES = {"claude": "claude_code", "dsh": "deepseek_harness", "mini": "mini_swe"}
GUIDE = Path(__file__).with_name("HELP.md")


# Settings read only as Relay starts: `relay set` restarts the service for them.
AT_START = {"host", "port", "harness", "trace", "event_log", "log_level", "openai_base_url", "chatgpt_base_url",
            "anthropic_base_url", "gemini_base_url", "openai_api_key", "anthropic_api_key", "cache_path",
            "cache_secret", "cache_ttl_seconds", "cache_max_entries", "cache_max_bytes", "cache_max_nodes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    harnesses = sorted({*HARNESSES, *ALIASES} - {"generic"})
    setup = commands.add_parser("setup", help="save the settings, start Relay, point a harness at it")
    setup.add_argument("harness", choices=harnesses)
    strategy = commands.add_parser("strategy", help="show or change the strategy")
    strategy.add_argument("name", nargs="?", choices=sorted(STRATEGIES))
    for command in (setup, strategy):
        command.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a setting (repeatable)")
    setup.add_argument("--strategy", choices=sorted(STRATEGIES))
    setup.add_argument("--via", choices=["auto", "proxy", "hook"], default="auto",
                       help="integration path (default: the harness's preferred one)")
    setup.add_argument("--autostart", action="store_true",
                       help="register Relay with the system's service manager (start at login, restart if it exits)")
    commands.add_parser("settings", help="every setting written")
    assign = commands.add_parser("set", help="change settings")
    assign.add_argument("assignments", nargs="+", metavar="NAME=VALUE")
    unset = commands.add_parser("unset", help="return settings to their defaults")
    unset.add_argument("names", nargs="+", metavar="NAME")
    commands.add_parser("status", help="whether Relay runs, its strategy, the harnesses set up")
    commands.add_parser("start", help="make sure Relay runs")
    commands.add_parser("help", help="the guide")
    serve = commands.add_parser("serve", help="run Relay in the foreground")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    install_ = commands.add_parser("install", help="only point a harness at Relay")
    uninstall_ = commands.add_parser("uninstall", help="restore a harness")
    install_.add_argument("harness", choices=harnesses)
    uninstall_.add_argument("harness", choices=harnesses, nargs="?")
    uninstall_.add_argument("--all", action="store_true", help="every harness set up")
    uninstall_.add_argument("--now", action="store_true",
                            help="restore it now, even if conversations that grew through Relay would then fail")
    install_.add_argument("--url", help="Relay's address (default: the host and port settings)")
    install_.add_argument("--via", choices=["auto", "proxy", "hook"], default="auto",
                          help="integration path (default: the harness's preferred one)")
    run = commands.add_parser("run", help="run a harness through a private proxy")
    run.add_argument("harness", choices=harnesses)
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the harness")
    options = parser.parse_args(argv)
    logging.basicConfig(level=settings.value("log_level", "INFO"), format="%(asctime)s %(name)s %(message)s")
    harness = HARNESSES[ALIASES.get(options.harness, options.harness)] if getattr(options, "harness", None) else None

    try:
        if options.command in (None, "help"):
            print(GUIDE.read_text(encoding="utf-8"))
            return 0
        if options.command == "setup":
            return _setup(harness, options.strategy, options.set, options.via, options.autostart)
        if options.command == "strategy":
            return _strategy(options.name, options.set)
        if options.command == "settings":
            print(json.dumps(settings.load(), indent=2, sort_keys=True))
            return 0
        if options.command in ("set", "unset"):
            changes = dict(map(settings.parse, options.assignments)) if options.command == "set" else {}
            return _change(changes, tuple(getattr(options, "names", ())))
        if options.command == "status":
            return _status()
        if options.command == "start":
            _settle()
            now = service.ensure()
            print(f"Relay runs {now['strategy']} at {service.address()}" if now else
                  f"Relay did not start: see {service.log_file()}", file=sys.stdout if now else sys.stderr)
            return 0 if now else 1
    except (ValueError, RuntimeError) as error:
        print(f"relay {options.command}: {error}", file=sys.stderr)
        return 1
    config = ProxyConfig.from_settings()
    if options.command == "install":
        return _install(harness, options.via, options.url or service.address())
    if options.command == "uninstall":
        if options.all == bool(harness):
            print("relay uninstall: name a harness, or --all", file=sys.stderr)
            return 1
        names = install.installed() if options.all else [harness.name]
        return max([_uninstall(HARNESSES[name], options.now) for name in names], default=0)
    if options.command == "run":
        return _run(config, ALIASES.get(options.harness, options.harness), options.args)
    host = getattr(options, "host", None) or config.host
    port = getattr(options, "port", None) or config.port
    uvicorn.run(create_app(config=config), host=host, port=port, ws="none")
    return 0


def _setup(harness: Harness, strategy: str | None, assignments: list[str], via: str, autostart: bool) -> int:
    """The settings saved, Relay running with them, then the harness pointed at it (and, where it
    runs a command as a session starts, made to start Relay)."""

    changes = dict(map(settings.parse, assignments)) | ({"strategy": strategy} if strategy else {})
    if changes.get("strategy", "compaction") not in STRATEGIES:
        raise ValueError(f"unknown strategy {changes['strategy']!r}: one of {', '.join(sorted(STRATEGIES))}")
    restart = bool(AT_START & set(changes))
    settings.save(changes)
    wanted = settings.value("strategy", "compaction")
    moved = service.status() is None and service.taken()  # (something else there: not Relay)
    if moved:
        if "port" in changes:
            raise ValueError(f"something other than Relay listens at {service.address()}: choose another port")
        busy, port = service.address(), service.free_port()
        settings.save({"port": port})
        print(f"something other than Relay listens at {busy}: Relay listens on port {port}")
    managed = service.record().get("manager") in ("systemd", "launchd")
    if autostart and not managed:
        kind = service.start(autostart=True)
    elif service.status() is None:
        kind = service.start() if not managed else (service.restart() or service.record()["manager"])
    elif restart and service.record():
        kind = service.restart() or service.record()["manager"]
    else:
        kind = None  # (running, and following the settings)
    if kind:
        print(f"started Relay ({'as a ' + kind + ' service' if kind != 'process' else 'as a process of its own'})")
        if autostart and kind == "process":
            print("  note: no service manager here: Relay runs as a process of its own")
    if service.wait(wanted) is None:
        print(f"relay setup: Relay did not answer at {service.address()} running {wanted}; the harness is left as it was."
              f"\n{_tail(service.log_file())}", file=sys.stderr)
        return 1
    print(f"Relay runs {wanted} at {service.address()}")
    starter = harness.starter(service.start_command())
    leaving.stay(harness.name)  # (set up again while leaving: it stays)
    if moved and harness.name in install.installed():
        install.uninstall(harness.name)  # (pointed at the old port)
    if harness.name not in install.installed() and _install(harness, via, service.address(), starter):
        return 1
    _move(wanted)
    if not starter and service.record().get("manager") == "process":
        print(f"  note: {harness.name} cannot start Relay itself: after this machine restarts, run `relay start` "
              f"(or set Relay up with --autostart)")
    print(f"{harness.name} goes through Relay: conversations started from now on use {wanted}; one already "
          f"open goes on as it was until a new one is started (`relay help`, \"Which conversations change\")")
    return 0


def _strategy(name: str | None, assignments: list[str]) -> int:
    if name is None and not assignments:
        now = service.status()
        print(f"strategy: {settings.value('strategy', 'compaction')}"
              + (f" (Relay runs {now['strategy']})" if now else " (Relay is not running)"))
        print("strategies: " + ", ".join(sorted(STRATEGIES)))
        return 0
    return _change(dict(map(settings.parse, assignments)) | ({"strategy": name} if name else {}), ())


def _change(changes: dict, removed: tuple[str, ...]) -> int:
    """Settings changed, and the running Relay following them."""

    if (name := changes.get("strategy")) is not None and name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}: one of {', '.join(sorted(STRATEGIES))}")
    settings.save(changes, removed)
    wanted = settings.value("strategy", "compaction")
    if "strategy" in changes:
        _move(wanted)
    if service.status() is None:
        print(f"saved; Relay is not running (`relay setup <harness>` starts it)")
        return 0
    if AT_START & {*changes, *removed} and service.record():
        service.restart()
    if service.wait(wanted, seconds=15) is None:
        print(f"saved, but the running Relay does not run {wanted}: see `relay status` and {service.log_file()}",
              file=sys.stderr)
        return 1
    print(f"Relay runs {wanted}, for every conversation from its next request")
    return 0


def _move(strategy: str) -> None:
    """Each harness set up on the path that runs `strategy`: the one it was installed with (`auto`,
    the preferred path that runs it), or else the preferred one that does."""

    for name in install.installed():
        if name in leaving.leaving():
            continue
        harness, known, now = HARNESSES[name], integrations.recorded(name) or {}, _path(name)
        wanted = known.get("via", now)
        try:
            target = integrations.choose(harness, wanted, strategy)
        except ValueError as error:
            why, target = str(error).split(";")[0], integrations.choose(harness, "auto", strategy)
        else:
            why = None
        if target.name == now:
            continue
        preferred = integrations.paths(harness)
        why = why or next((integrations.unfit(path, strategy) for path in preferred[:preferred.index(target)]), None)
        install.uninstall(name)
        if _install(harness, target.name, service.address(), harness.starter(service.start_command()), quiet=True):
            continue
        integrations.remember(name, wanted, target.name)
        print(f"  {name} now goes through the {target.name} ({why or f'it runs {strategy}'}): conversations started "
              "from now on; one already open goes on as it was")
        if name == "claude_code" and target.name == "proxy":
            print("  note: through the proxy, Claude Code's Remote Control is off (it needs Anthropic's own endpoint)")


def _status() -> int:
    now, entry = service.status(), service.record()
    print(f"Relay: {'running at ' + service.address() + ', strategy ' + now['strategy'] if now else 'not running'}")
    kept = {"process": f"a process of its own (pid {entry.get('pid')})", "systemd": "a systemd user service",
            "launchd": "a launchd agent"}
    print(f"kept running by: {kept.get(entry.get('manager'), 'nothing (started by hand?)')}")
    _settle()
    departing = leaving.leaving()
    print("harnesses set up: " + (", ".join(f"{n} ({_path(n)})" for n in install.installed() if n not in departing)
                                  or "none"))
    for name, seen in ((now or {}).get("harnesses") or {}).items():  # (as their requests said it)
        newer = HARNESSES[name].newer(seen["release"])
        print(f"  {name} {seen['release']}: profile checked against {seen['checked'] or 'no release'}"
              + (" (newer: served as the newest of those)" if newer else ""))
    for name in departing:
        waiting = leaving.needing(name)
        print(f"leaving Relay: {name}, restored by itself once {'this conversation does' if len(waiting) == 1 else 'these do'}"
              f" not need Relay any more (its own `/compact` in it, or a week after its last use):")
        for entry in waiting:
            print(f"  · {_described(entry)}")
    for name, when in leaving.restored().items():
        if name not in departing and name not in install.installed():
            print(f"left Relay: {name}, restored by itself {_ago(when)} (no conversation needed Relay any more); "
                  "it talks to its provider directly")
    print(f"settings: {settings.path()}")
    return 0 if now else 1


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _install(harness: Harness, via: str, url: str, extra: list | None = None, quiet: bool = False) -> int:
    strategy = settings.value("strategy", "compaction")
    try:
        path = integrations.choose(harness, via, strategy)
        setup = path.installation(url)
        install.install(harness.name, url, [*setup.settings, *(extra or [])])
    except ValueError as error:
        print(f"relay install: {error}", file=sys.stderr)
        return 1
    integrations.remember(harness.name, via, path.name)
    if quiet:
        return 0
    passed = [f"{other.name}: {why}" for other in integrations.paths(harness)[:integrations.paths(harness).index(path)]
              if via == "auto" and (why := integrations.unfit(other, strategy))]
    print(f"installed {harness.name} ({path.name}): " + ", ".join(sorted({str(s.path) for s in setup.settings})))
    for why in passed:
        print(f"  not the {why}")
    for note in setup.notes:
        print(f"  note: {note}")
    return 0


def _uninstall(harness: Harness, now: bool = False) -> int:
    """The harness's configuration restored, unless conversations that grew past the model's window
    through Relay would fail direct: it then leaves in two steps (relay/leaving.py)."""

    name = harness.name
    if name not in install.installed():
        print(f"relay uninstall: {name} is not installed", file=sys.stderr)
        return 1
    service.status()  # (a running Relay writes down the conversations it served)
    waiting = leaving.needing(name)
    if waiting and not now:
        leaving.leave(name)
        one = len(waiting) == 1
        print(f"{name} is leaving Relay. Its conversations now go as {name} sends them, as if direct, except "
              f"{'this one' if one else 'these'}: grown past the model's window through Relay, "
              f"{'it' if one else 'they'} would fail direct, so Relay keeps managing {'it' if one else 'them'}:")
        for entry in waiting:
            print(f"  · {_described(entry)}")
        print(f"To keep {'it' if one else 'one'}, run {name}'s own `/compact` in it: then it fits, and goes direct too.")
        print(f"Once no conversation needs Relay (or a week after the last use of each), {name}'s configuration is "
              f"restored by itself. `relay status` shows what is left; `relay uninstall {name} --now` restores it now.")
        return 0
    leaving.stay(name)
    paths = install.uninstall(name)
    integrations.forget(name)
    print(f"uninstalled {name}: " + ", ".join(sorted({str(path) for path in paths})))
    for entry in waiting:  # (--now)
        print(f"  note: this one grew past the model's window through Relay and fails direct: {_described(entry)}")
    if not install.installed() and service.release() in ("systemd", "launchd"):
        print("  no harness goes through Relay now: it is no longer started at login, and the one running goes on "
              "for the conversations still open")
    return 0


def _path(name: str) -> str:
    """The path the harness was installed on (before Relay recorded it: its hook's setting, or the proxy)."""

    if known := (integrations.recorded(name) or {}).get("path"):
        return known
    return "hook" if ("env", "RELAY_HOOK_URL") in {key for _, key in install.written(name)} else "proxy"


def _ago(when: float) -> str:
    ago = max(0, time.time() - when)
    return ("just now" if ago < 60 else f"{ago / 60:.0f} minutes ago" if ago < 3600
            else f"{ago / 3600:.0f} hours ago" if ago < 86400 else f"{ago / 86400:.0f} days ago")


def _described(entry: dict) -> str:
    return (f"\"{entry.get('title') or '(no text)'}\" — ~{entry['whole'] / 1000:.0f}k tokens for a "
            f"{entry['window'] / 1000:.0f}k window, last used {_ago(entry['last'])}")


def _settle() -> None:
    """Restore the leaving harnesses none of whose conversations needs Relay any more."""

    for name in leaving.settle():
        print(f"{name}: no conversation needs Relay any more; its configuration is restored")


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
