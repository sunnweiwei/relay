"""The SForge agent EdgeBench's Relay arms run: Codex with a Relay of its own in its container.

SForge gives the agent a work container and no way to mount a host directory into it
(`backend/docker_backend.py` takes no volumes), so a Relay on the host could serve the
proxy but could not share files with the agent. `clm` and `prolong` need exactly that
sharing: the model edits the context file its own tools reach. Running Relay inside the
work container gives it the agent's user and filesystem, and gives each task its own
store, at the cost of installing Relay there as the container starts.

Codex's own compaction is replaced rather than out-competed. `model_auto_compact_token_limit`
cannot lift its trigger for a model slug Codex does not know: the fallback metadata is a
272k window, config values are clamped to it (`models-manager/src/model_info.rs`), and the
trigger is always 90% of the window (`protocol/src/openai_models.rs`). A model catalog
(`model_catalog_json`, replacing the bundled one for this process) can, because the window
itself comes from it — `edgebench.py --codex-prompt` writes one. Where Codex knows the slug,
NEVER is enough and the catalog can be left out, as `terminal_bench.py` does.

Configuration arrives through SFORGE_AGENT_EXTRA_ENV:

    RELAY_STRATEGY   a strategy name, or "off" for Codex alone (the `native` arm)
    RELAY_SETTINGS   further Relay settings as JSON, merged last
    RELAY_ASSETS     where the driver serves the Relay wheel and the catalog
    RELAY_WHEEL      the wheel's filename there
    RELAY_CATALOG    the catalog's filename there, or "off" to leave Codex's own window alone
    PIP_WHEEL        a pip wheel's filename there, for an image whose Python has no pip
    PIP_INDEX        a package index, where the default one is slow or blocked
"""

from __future__ import annotations

import json
import shlex

from sforge.harness.agent.codex import CodexAgent

PORT = 8787
HOME = "/home/agent/.relay"
CATALOG = "/home/agent/codex_model_catalog.json"
# Relay asks for Python 3.11 only for tomllib, and the work images carry 3.10 or no Python
# at all; tomli stands in for it. The specifiers are quoted: bash reads `>` as a redirection.
DEPS = " ".join(shlex.quote(dep) for dep in (
    "starlette>=0.47.0", "uvicorn>=0.35.0", "httpx>=0.28.0",
    "zstandard>=0.23", "cryptography>=42", "tomli",
))


def settings(env: dict[str, str]) -> dict[str, object]:
    """The settings.json Relay reads, with the paths a strategy shares with the agent
    under the agent's own home, where its tools can read and write them."""

    values: dict[str, object] = {
        "port": PORT,
        "strategy": env.get("RELAY_STRATEGY", "compaction"),
        "openai_base_url": env["RELAY_UPSTREAM"],
        "cache_path": f"{HOME}/store.json",
        "event_log": f"{HOME}/events.jsonl",
        # `freed` counts an item's text and not its opaque part, so a context that is
        # mostly encrypted reasoning can sit above the threshold and never compact.
        "compact_min_gain": 0,
        "clm_dir": "/home/agent/.live_ctx",
        "acm_dir": "/home/agent/.acm",
        "prolong_log": "/home/agent/.prolong/log.jsonl",
        "ace_playbook": f"{HOME}/ace_playbook.md",
    }
    if window := env.get("RELAY_CONTEXT_WINDOW"):
        values["context_window"] = int(window)
    if extra := env.get("RELAY_SETTINGS"):
        values.update(json.loads(extra))
    return values


class CodexRelayAgent(CodexAgent):

    name = "codex-relay"

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        super().augment_env(env, model)
        # Codex is pointed at the Relay beside it; Relay is pointed where the run was.
        env["RELAY_UPSTREAM"] = env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        env.setdefault("RELAY_ASSETS", "http://host.docker.internal:8899")
        env.setdefault("RELAY_WHEEL", "relay-0.1.0-py3-none-any.whl")
        env.setdefault("RELAY_CATALOG", "codex_model_catalog.json")
        env.setdefault("RELAY_STRATEGY", "compaction")
        env["RELAY_HOME"] = HOME
        if env["RELAY_STRATEGY"] != "off":
            env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"

    def __init__(self, config) -> None:  # noqa: ANN001 (as Agent.__init__ takes it)
        super().__init__(config)
        self.install_cmds = [*CodexAgent.install_cmds, *self._setup()]

    def _setup(self) -> list[str]:
        env = self._config.agent_extra_env
        index = f"-i {shlex.quote(env['PIP_INDEX'])}" if env.get("PIP_INDEX") else ""
        cmds = []

        if env.get("RELAY_CATALOG", "codex_model_catalog.json") != "off":
            cmds.append(
                f'curl -fsSL -o {CATALOG} "$RELAY_ASSETS/$RELAY_CATALOG"'
            )
        if env.get("RELAY_STRATEGY", "compaction") == "off":
            return cmds

        values = json.dumps(settings({**env, "RELAY_UPSTREAM": "__UPSTREAM__"}))
        return cmds + [
            # A C++ work image carries no Python, and a Debian one marks the Python it
            # carries externally managed (PEP 668), where pip refuses to install anything.
            # Dropping that marker beats a flag the older pip in the Ubuntu images rejects;
            # the container is thrown away with the task.
            'command -v python3 >/dev/null || sudo -n sh -c '
            '"apt-get update -qq && apt-get install -y -qq python3-minimal python3-pip" || true; '
            'sudo -n rm -f "$(python3 -c '
            "'import sysconfig; print(sysconfig.get_path(\"stdlib\"))'"
            ')/EXTERNALLY-MANAGED" 2>/dev/null || true; '
            # An image can carry a Python with no pip and an apt mirror it cannot reach, so
            # pip comes from the driver's own wheel, run straight out of the zip, as a last resort.
            'python3 -m pip -V >/dev/null 2>&1 || python3 -m ensurepip --user >/dev/null 2>&1 || true; '
            'if ! python3 -m pip -V >/dev/null 2>&1 && [ -n "${PIP_WHEEL:-}" ]; then '
            '  mkdir -p /tmp/pip-boot && '
            '  curl -fsSL -o "/tmp/pip-boot/$PIP_WHEEL" "$RELAY_ASSETS/$PIP_WHEEL" && '
            '  python3 "/tmp/pip-boot/$PIP_WHEEL/pip" install -q --no-index "/tmp/pip-boot/$PIP_WHEEL"; fi; '
            'python3 -V && python3 -m pip -V',
            f'python3 -m pip install -q {index} --ignore-requires-python {DEPS}',
            'mkdir -p /tmp/relay-wheel && '
            'curl -fsSL -o "/tmp/relay-wheel/$RELAY_WHEEL" "$RELAY_ASSETS/$RELAY_WHEEL" && '
            'python3 -m pip install -q --ignore-requires-python --no-deps "/tmp/relay-wheel/$RELAY_WHEEL"',
            'python3 -c "import tomllib" 2>/dev/null || '
            'printf "from tomli import *\\nfrom tomli import load, loads\\n" '
            '> "$(python3 -c \'import site;print(site.getusersitepackages())\')/tomllib.py"',
            f'mkdir -p {HOME} && python3 - <<\'PY\'\n'
            f'import json, os\n'
            f'settings = json.loads({values!r})\n'
            f'settings["openai_base_url"] = os.environ["RELAY_UPSTREAM"]\n'
            f'json.dump(settings, open("{HOME}/settings.json", "w"), indent=1, sort_keys=True)\n'
            f'print("relay settings:", json.dumps(settings, sort_keys=True))\n'
            f'PY',
            # Relay outlives this exec: the container stays up for the whole run.
            f'cd /home/agent && nohup python3 -m relay.cli serve > {HOME}/serve.log 2>&1 & '
            f'for _ in $(seq 1 60); do '
            f'  if curl -fsS -m 2 http://127.0.0.1:{PORT}/relay/v1/status 2>/dev/null; then echo; exit 0; fi; '
            f'  sleep 1; done; '
            f'echo "Relay did not start:"; tail -40 {HOME}/serve.log; exit 1',
        ]

    def format_run_cmd(self, prompt_path: str, *, model: str | None = None, cwd: str = "",
                       internet: bool = True, resume: bool = False) -> str:
        cmd = super().format_run_cmd(prompt_path, model=model, cwd=cwd,
                                     internet=internet, resume=resume)
        if self._config.agent_extra_env.get("RELAY_CATALOG", "codex_model_catalog.json") != "off":
            cmd = cmd.replace("codex exec", f"codex exec -c model_catalog_json={shlex.quote(CATALOG)}", 1)
        return cmd
