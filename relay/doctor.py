"""`relay doctor`: whether the harnesses set up really go through Relay, and Relay through to their
providers. A harness that left Relay fails silently: its conversations go on, unmanaged. Its config
was rewritten by another tool (cc-switch rewrites Claude Code's settings.json whole as the user
switches provider), the shell sets a variable its config sets (Gemini CLI, Goose and mini-swe-agent
take the shell's first), or Relay runs older code than is installed; and Relay's own connections may
fail where the harness's did not (a TLS-inspecting proxy whose root only the OS store holds).
Exit 0: all well; 1: warnings; 2: problems."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from . import install, service, settings
from .transport import outbound

LEVELS = ("ok", "warning", "problem")


def run() -> int:
    worst = 0

    def say(level: int, text: str) -> None:
        nonlocal worst
        worst = max(worst, level)
        print(f"{LEVELS[level]:8} {text}")

    now = service.status()
    if now is None:
        pid = service.process()
        say(2, f"Relay is unreachable: {service.unreachable(pid)}" if pid else "Relay is not running: `relay start`")
    else:
        say(0, f"Relay runs {now['strategy']} at {service.address()}")
        if now.get("code") and now["code"] != service.code():
            say(1, "Relay runs older code than is installed (it was updated since it started): `relay restart`")
    names = install.installed()
    if not names:
        say(1, "no harness is set up: `relay setup <harness>`")
    origins: set[str] = set()
    for name in names:
        records = install.records(name)
        origins |= {record["origin"] for record in records if "origin" in record}
        if off := install.drifted(name):
            files = ", ".join(sorted({record["path"] for record in off}))
            say(2, f"{name} no longer goes through Relay: {files} was rewritten since (cc-switch, say, rewrites Claude "
                   f"Code's settings.json whole): `relay setup {name}` points it at Relay again, keeping what is there now "
                   f"as what `relay uninstall` restores")
        else:
            when = ((now or {}).get("last") or {}).get(name)
            say(0, f"{name} points at Relay" + (f"; its latest request came {ago(when)}" if when
                                                 else "; no request of it since Relay started"))
        for variable, value in _shadowed(records):
            say(2, f"{name}: this shell sets {variable}={value}, which {name} takes before its config's: started from "
                   f"here, it does not go through Relay (unset it, or set it to the config's)")
    for origin in sorted(origins):
        level, text = _reach(origin)
        say(level, text)
    path = settings.path()
    if path.exists() and path.stat().st_mode & 0o077 and any(name.endswith("_api_key") for name in settings.load()):
        say(1, f"{path} holds API keys and others on this machine can read it: `chmod 600 {path}`")
    return worst


def _shadowed(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Variables of the shell that a harness takes before what Relay wrote: settings named as
    variables (a .env file's, Goose's), as the harness's config holds them now."""

    found = []
    for record in records:
        key = tuple(record["key"])
        if len(key) == 1 and key[0].isupper() and (value := os.environ.get(key[0])) is not None:
            if value != install.read(Path(record["path"]), key):
                found.append((key[0], value))
    return found


def _reach(origin: str) -> tuple[int, str]:
    """Whether Relay's connections reach a provider (from this shell, with Relay's trust)."""

    try:
        with outbound.client(timeout=10) as client:
            client.head(origin)
    except httpx.ConnectError as error:
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            return 2, (f"{origin}: Relay does not trust its certificate. Behind a TLS-inspecting proxy, install its root in "
                       "the system's store or name its bundle in SSL_CERT_FILE where Relay runs")
        return 2, f"{origin} is unreachable from here ({error})"
    except httpx.HTTPError as error:
        return 1, f"{origin}: {type(error).__name__} ({error})"
    return 0, f"{origin} reachable"


def ago(when: float) -> str:
    ago = max(0, time.time() - when)
    return ("just now" if ago < 60 else f"{ago / 60:.0f} minutes ago" if ago < 3600
            else f"{ago / 3600:.0f} hours ago" if ago < 86400 else f"{ago / 86400:.0f} days ago")
