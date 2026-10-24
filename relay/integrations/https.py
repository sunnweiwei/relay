"""Claude Code through Relay as its HTTPS proxy (`relay.intercept`).

`relay install claude --via https` leaves Claude Code's base URL as it is and sets, in its own
settings, `HTTPS_PROXY` (Relay's proxy) and `NODE_EXTRA_CA_CERTS` (Relay's CA, which can vouch
for api.anthropic.com alone). Claude Code still reaches api.anthropic.com, so Remote Control and
what Claude Code keeps for Anthropic's own endpoint stay on, while every request is managed as
through the proxy: its transcript is never touched, and every strategy runs. The commands Claude
Code runs inherit the proxy, as an organization's proxy is inherited: Relay tunnels their
connections as they are (only api.anthropic.com's are opened).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .. import intercept, settings
from ..harnesses.claude_code import settings_file
from ..install import Setting
from . import claude_code
from .base import Installation, Proxy

SINCE = "2.1.294"  # the oldest release of Claude Code the HTTPS proxy was checked with


class ClaudeCodeHttps:
    name = "https"
    carries = Proxy.carries  # (the requests are managed as through the proxy)
    # Checked with Claude (`check.py --via https --strategy S claude_code:claude`); `acm` and `mem1` met
    # Anthropic's refusals (reasoning_extraction) on its own endpoint, `ace` fails on Claude through
    # the proxy too.
    checked: dict[str, str] | None = {s: "2.1.294–2.1.309" for s in ("compaction", "folding", "clm", "autocompact", "prolong",
                                                            "selfcompact")}
    # Before the hook and the proxy: Claude Code keeps its endpoint (Remote Control) and its own
    # transcript, every strategy runs as through the proxy (sub-agents included), and nothing of
    # Claude Code's compaction (its once-a-session system prompt, its thrashing breaker) is in the way.
    preferred = True

    def installation(self, relay_url: str) -> Installation:
        path, port = settings_file(), settings.value("intercept_port")
        if not port:
            raise ValueError("Relay's HTTPS proxy has no port (`relay setup` gives it one)")
        trusted = str(intercept.ca_file())
        theirs = os.environ.get("NODE_EXTRA_CA_CERTS") or _env(path).get("NODE_EXTRA_CA_CERTS")
        if theirs and Path(theirs).expanduser().exists():  # (the user's CAs, and Relay's: one file)
            bundle = intercept.folder() / "bundle.pem"
            bundle.write_bytes(Path(theirs).expanduser().read_bytes().rstrip() + b"\n" + Path(trusted).read_bytes())
            trusted = str(bundle)
        return Installation([Setting(path, ("env", "HTTPS_PROXY"), f"http://127.0.0.1:{port}"),
                             Setting(path, ("env", "NODE_EXTRA_CA_CERTS"), trusted)],
                            notes=("the commands Claude Code runs go through Relay's proxy too (it tunnels them as "
                                   "they are)",))

    def unavailable(self) -> str | None:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            return "Relay's HTTPS proxy needs the cryptography package"
        if (release := claude_code.installed_release()) is None:
            return "Claude Code was not found here (the HTTPS proxy needs its release)"
        if claude_code._release(release) < claude_code._release(SINCE):
            return f"Claude Code {release} is older than {SINCE}, the oldest release the HTTPS proxy was checked with"
        own = settings.value("intercept_port")
        for source, env in (("the environment", os.environ), ("Claude Code's settings", _env(settings_file()))):
            value = next((env[v] for v in intercept.PROXY_VARS if env.get(v)), "")
            if value and not (own and value.rstrip("/").endswith(f"127.0.0.1:{own}")):
                return f"Claude Code already goes through a proxy ({value}, in {source}), which Relay's does not chain to"
        return None


def _env(path: Path) -> dict[str, str]:
    try:
        return dict(json.loads(path.read_text()).get("env") or {})
    except (OSError, ValueError, AttributeError):
        return {}
