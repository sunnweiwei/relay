"""Relay as a process of the user's: started by `relay setup` and `relay start`; once no harness goes
through it, no longer started at login (it runs on until the machine restarts).

A harness pointed at Relay reaches its model only through Relay, so Relay must be running. By
default it runs as a process of its own (a daemon: its pid and its log in Relay's directory),
outside the session that started it, and nothing is registered with the system. A harness that
runs a command as a session starts (Claude Code's SessionStart hook) runs `relay start`, which
returns at once while Relay answers and starts it otherwise, so a machine's restart costs
nothing. `relay setup --autostart` registers Relay with the system's service manager instead
(a systemd user unit on Linux, a launchd agent on macOS), which starts it at login and again if it
exits. What runs it is recorded in Relay's directory (`service.json`).

A Relay that stops ends the requests still open after `GRACEFUL` seconds (a long stream; Remote
Control keeps one open), and a restart makes sure the old process is gone before the new one takes
its ports. A Relay whose process runs but does not answer is unreachable, not stopped: from a
sandbox that keeps commands off the network (Codex's), or stuck; nothing is restarted then.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import settings

NAME = "relay"
LABEL = "com.relay.serve"  # launchd
PACKAGE = Path(__file__).resolve().parent.parent  # where `python -m relay.cli` finds Relay
GRACEFUL = 3.0  # seconds a stopping Relay waits for the requests still open, then ends them
STOPPING = GRACEFUL + 5.0  # seconds a Relay asked to stop gets before it is made to


def command() -> list[str]:
    return [sys.executable, "-m", "relay.cli", "serve"]


def log_file() -> Path:
    return settings.home() / "relay.log"


def address() -> str:
    return f"http://{settings.value('host', '127.0.0.1')}:{settings.value('port', '8787')}"


def status(timeout: float = 2.0) -> dict[str, Any] | None:
    """What the running Relay reports (`/relay/v1/status`), or None if none answers."""

    try:
        with urllib.request.urlopen(f"{address()}/relay/v1/status", timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, ValueError):
        return None


def taken() -> bool:
    """Whether something listens where Relay should (Relay or not)."""

    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((settings.value("host", "127.0.0.1"), int(settings.value("port", "8787")))) == 0


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((settings.value("host", "127.0.0.1"), 0))
        return probe.getsockname()[1]


def wait(strategy: str | None = None, seconds: float = 30.0) -> dict[str, Any] | None:
    """The running Relay's status once it answers (and runs `strategy`, if given)."""

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (now := status()) is not None and (strategy is None or now.get("strategy") == strategy):
            return now
        time.sleep(0.3)
    return None


def process() -> int | None:
    """The process Relay runs as, as recorded, while it runs: whether it answers or not."""

    entry = record()
    try:
        if entry.get("manager") == "systemd":
            shown = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", f"{NAME}.service"],
                                   capture_output=True, text=True).stdout.strip()
            pid = int(shown) if shown.isdigit() else 0
        elif entry.get("manager") == "launchd":
            shown = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"], capture_output=True, text=True).stdout
            pid = int(found.group(1)) if (found := re.search(r"^\s*pid = (\d+)", shown, re.M)) else 0
        else:
            pid = entry.get("pid") or 0
    except OSError:
        pid = 0
    return pid if pid and _alive(pid) else None


def unreachable(pid: int) -> str:
    """What to do about a Relay whose process `pid` runs but does not answer."""

    return (f"Relay's process {pid} runs, but does not answer at {address()} from here. Run in a sandbox (as Codex runs "
            f"commands)? Run this again outside it. If Relay does not answer there either, it is stuck: end process "
            f"{pid} (kill {pid}), then run `relay start`.")


def _alive(pid: int) -> bool:
    """Whether `pid` is a Relay that runs (one this process may not signal, as from a sandbox, does)."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:  # (and not another process that got its pid since)
        return command()[2].encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True  # (no /proc: macOS)


def manager() -> str:
    """What can keep Relay running here: "systemd", "launchd" or "process"."""

    if sys.platform == "darwin" and shutil.which("launchctl"):
        return "launchd"
    if shutil.which("systemctl") and subprocess.run(["systemctl", "--user", "show-environment"],
                                                     capture_output=True).returncode == 0:
        return "systemd"
    return "process"


def record() -> dict[str, Any]:
    try:
        return json.loads((settings.home() / "service.json").read_text())
    except (OSError, ValueError):
        return {}


def _remember(entry: dict[str, Any] | None) -> None:
    path = settings.home() / "service.json"
    if entry is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, indent=2) + "\n")


def unit() -> str:
    """The systemd user unit."""

    return (f"[Unit]\nDescription=Relay, context management for coding agents\n\n[Service]\n"
            f"ExecStart={' '.join(command())}\nWorkingDirectory={PACKAGE}\nRestart=always\nRestartSec=2\n"
            f"StandardOutput=append:{log_file()}\nStandardError=append:{log_file()}\n\n"
            f"[Install]\nWantedBy=default.target\n")


def agent() -> str:
    """The launchd agent."""

    arguments = "".join(f"<string>{part}</string>" for part in command())
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>'
            f"<key>Label</key><string>{LABEL}</string><key>ProgramArguments</key><array>{arguments}</array>"
            f"<key>WorkingDirectory</key><string>{PACKAGE}</string><key>RunAtLoad</key><true/>"
            f"<key>KeepAlive</key><true/><key>StandardOutPath</key><string>{log_file()}</string>"
            f"<key>StandardErrorPath</key><string>{log_file()}</string></dict></plist>\n")


def start(autostart: bool = False) -> str:
    """Run Relay: a daemon of its own, or with `autostart` under the service manager (started now
    and at every login); which one."""

    kind = manager() if autostart else "process"
    log_file().parent.mkdir(parents=True, exist_ok=True)
    if kind == "systemd":
        path = Path("~/.config/systemd/user/relay.service").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit())
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", f"{NAME}.service"])
        _run(["systemctl", "--user", "restart", f"{NAME}.service"])  # (an older one: the current settings)
        _remember({"manager": kind, "path": str(path)})
    elif kind == "launchd":
        path = Path(f"~/Library/LaunchAgents/{LABEL}.plist").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(agent())
        domain = f"gui/{os.getuid()}"
        subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True)
        subprocess.run(["launchctl", "enable", f"{domain}/{LABEL}"], capture_output=True)  # (if released before)
        _run(["launchctl", "bootstrap", domain, str(path)])
        _remember({"manager": kind, "path": str(path)})
    else:
        with _locked():  # (two sessions starting at once start one Relay)
            _daemon()
    return kind


def _daemon() -> None:
    """Relay as a process of its own, outside the caller's session and its signals (lock held)."""

    stop_process()
    _rotate()
    with log_file().open("ab") as log:
        process = subprocess.Popen(command(), cwd=PACKAGE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    _remember({"manager": "process", "pid": process.pid})


def ensure(seconds: float = 20.0) -> dict[str, Any] | None:
    """Relay running: at once if it answers, else started as it was set up (its status, or None)."""

    if (now := status(timeout=1.0)) is not None:
        return now
    if (pid := process()) is not None:  # (starting, or unreachable from here)
        if (now := wait(seconds=5)) is None:
            raise RuntimeError(unreachable(pid))
        return now
    entry = record()
    if entry.get("manager") in ("systemd", "launchd"):
        restart()
    else:
        with _locked():
            if status(timeout=1.0) is None:  # (not started meanwhile by another session)
                _daemon()
    return wait(seconds=seconds)


def start_command() -> str:
    """A shell command that makes sure Relay runs, cheap while it does, and says nothing (a
    harness's hook: some add what a session-start hook prints to the model's context)."""

    probe = f"import urllib.request as u; u.urlopen('{address()}/relay/v1/status', timeout=1)"
    start = f"cd {shlex.quote(str(PACKAGE))} && {' '.join(map(shlex.quote, command()[:-1]))} start"
    return f'{shlex.quote(sys.executable)} -c "{probe}" >/dev/null 2>&1 || ({start}) >/dev/null 2>&1; true'


def restart() -> None:
    """Start Relay again with its current settings (where it listens, its upstreams, its store)."""

    kind = record().get("manager")
    if kind == "systemd":
        _run(["systemctl", "--user", "restart", f"{NAME}.service"])
    elif kind == "launchd":
        _run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LABEL}"])
    else:
        start()


def release() -> str | None:
    """No harness goes through Relay any more: the service manager stops starting it at login. The
    one running goes on, for the conversations still open on its address; what had kept it running."""

    kind = record().get("manager")
    if kind == "systemd":
        subprocess.run(["systemctl", "--user", "disable", f"{NAME}.service"], capture_output=True)
    elif kind == "launchd":
        subprocess.run(["launchctl", "disable", f"gui/{os.getuid()}/{LABEL}"], capture_output=True)
    return kind


def stop_process() -> None:
    """The process Relay was started as, stopped: asked, then made to (one waiting on requests that
    stay open holds the HTTPS proxy's port). RuntimeError if it cannot be."""

    pid = record().get("pid")
    if not pid or not _alive(pid):
        return
    for sent, seconds in ((signal.SIGTERM, STOPPING), (signal.SIGKILL, 5.0)):
        try:
            os.kill(pid, sent)
        except ProcessLookupError:
            return
        except PermissionError:
            raise RuntimeError(f"Relay's process {pid} cannot be stopped from here (a sandbox?): run this outside it")
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not _alive(pid):
                return
            time.sleep(0.1)
    raise RuntimeError(f"Relay's process {pid} did not stop")


def _rotate(limit: int = 5 * 1024 * 1024) -> None:
    """Keep the log small: past `limit`, the next start begins a new one (the last one kept)."""

    if log_file().exists() and log_file().stat().st_size > limit:
        log_file().replace(log_file().with_suffix(".log.1"))


@contextmanager
def _locked() -> Iterator[None]:
    settings.home().mkdir(parents=True, exist_ok=True)
    with (settings.home() / "start.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _run(arguments: list[str]) -> None:
    result = subprocess.run(arguments, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(arguments)}: {(result.stderr or result.stdout).strip()[:300]}")
