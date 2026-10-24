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
from .base import Installation, Proxy


class ClaudeCodeHttps:
    name = "https"
    carries = Proxy.carries  # (the requests are managed as through the proxy)
    checked: dict[str, str] | None = {}  # none yet: taken when named
    preferred = False

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
