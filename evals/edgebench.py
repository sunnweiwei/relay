r"""EdgeBench through Relay: Codex on day-scale tasks under each strategy.

Setup (from the Relay checkout; host Python 3.11-3.13 and Docker required)::

    pip install -e '.[dev]' pip
    pip install 'sforge==1.1.0'  # or: pip install -e /path/to/EdgeBench
    sforge fetch-tasks edgebench
    sforge pull

This adapter targets SForge 1.1.0's Docker backend and its stock Codex 0.130.0
agent. DIR below is the EdgeBench bundle containing tasks/ and with work/judge
images loaded. From that bundle directory, start `sforge serve` in a separate
terminal. Configure SForge's credentials and endpoint as for a normal run,
using SFORGE_AGENT_API_KEY and SFORGE_AGENT_API_BASE_URL.

A short smoke run, launched from the Relay checkout::

    python evals/edgebench.py --bundle DIR --out runs/edgebench-smoke \
        --tasks borden_source_inversion git_rewrite_in_zig \
        --arms codex clm --timeout 1800 --stagger 0 --arm-delay 0 \
        --model YOUR_MODEL --codex-prompt /path/to/codex-0.130.0/codex-rs/models-manager/prompt.md
    python evals/edgebench.py --report runs/edgebench-smoke --bundle DIR \
        --csv runs/curves.csv --price-in 0.10 --price-out 0.50

Every arm runs the same tasks. Defaults are all five arms and 12 hours per task:

    codex       Stock Codex behavior, including its own compaction (baseline).
    native      Codex alone, with its compaction trigger moved out of reach.
    compaction  Relay's Codex-style compaction.
    clm         The model edits its own context file through Relay.
    folding     Relay context folding.

Relay runs inside each work container: SForge provides no host-directory mounts,
and CLM needs the agent's own tools to reach its context files. edgebench_agent.py
installs Relay from a wheel this driver builds and serves. Every task has its own
Relay process and store.

Every arm except codex requires --codex-prompt from the matching Codex release.
For an unknown model slug, Codex's fallback window is 272k and its compaction
trigger remains capped at 90% of that window despite larger config values. The
model catalog preserves the fallback base instructions and raises the window to
put that trigger out of reach; --window independently controls Relay's window.
The catalog schema and fallback metadata are tied to Codex 0.130.0; recheck them
before changing SForge's pinned agent version.

The asset server defaults to the Linux Docker bridge, 172.17.0.1:8899. Configure
--asset-bind with a local listen address and --asset-host with a hostname reachable
from containers when needed. Docker Desktop may need --asset-bind 0.0.0.0
--asset-host host.docker.internal; SForge's host network isolation must also support
the environment. Only the Relay wheel, pip bootstrap wheel and model catalog are
served. --pip-index optionally selects the container package index; dependencies
still require package-index access during setup.

Tasks retain EdgeBench's internet restrictions by default. --enable-internet
overrides them, making results incomparable with standard EdgeBench results.
Network isolation requires SForge's usual host iptables/sudo setup.

Use a new --out directory for each run. run.json records the tasks and arms;
each arm has sforge.log and per-task snapshots of Relay events/logs, CLM files,
and Codex rollout sessions. Periodic copies run every minute; a separate final/
snapshot is taken before Docker cleanup and preferred by reports. Abrupt
termination can leave incomplete snapshots. Keep the judge reports under
DIR/logs/runs/<run-id>-<arm>/ to regenerate reports.

The report prints best score out of 100, collected cost, input tokens and strategy
event counts. --csv writes each score/usage event with cumulative totals; hours
are relative to the task's first collected event. Token usage differences Codex's
cumulative totals per session, avoiding duplicate token_count events. Missing
rollouts are flagged and excluded from collected cost, while their numeric CSV
usage/cost fields remain zero. --price-in and --price-out are USD per million
tokens; supply your provider's rates. All input, including cached input, is charged
at the input rate. cached_tokens in the CSV allows discounted cost recalculation.

SForge must be importable by this interpreter. The launcher registers codex-relay
through SForge's private factory._REGISTRY and wraps DockerBackend.cleanup_container
inside its child process. Upstream interface changes may require adapter updates;
no SForge files are patched.

Local checks (SForge-dependent tests skip when it is not installed)::

    python -m pytest tests/test_edgebench.py -q

These cover accounting, reporting, shell syntax, SForge registration and cleanup
order. A real two-task smoke run is still needed to verify Docker setup and model
calls end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOST = "172.17.0.1"  # the Docker bridge, where the work containers reach this machine
ARMS = ("codex", "native", "compaction", "clm", "folding")
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # USD per token, as the model's card gives them
# The window the catalog hands the model. 100M is the value these arms actually ran with;
# the point is only that 90% of it is unreachable, so Codex never compacts.
CATALOG_WINDOW = 100_000_000
HOME_RELAY = "/home/agent/.relay"  # where the agent keeps Relay, to collect from
# Codex 0.130's fallback metadata, field for field (models-manager/src/model_info.rs), so a
# catalog entry changes the window and nothing else about how the agent behaves.
FALLBACK = {
    "description": None, "supported_reasoning_levels": [], "shell_type": "default",
    "visibility": "none", "supported_in_api": True, "priority": 99,
    "additional_speed_tiers": [], "service_tiers": [], "availability_nux": None,
    "upgrade": None, "supports_reasoning_summaries": False, "default_reasoning_summary": "auto",
    "support_verbosity": False, "default_verbosity": None, "apply_patch_tool_type": None,
    "web_search_tool_type": "text", "truncation_policy": {"mode": "bytes", "limit": 10000},
    "supports_parallel_tool_calls": False, "supports_image_detail_original": False,
    "effective_context_window_percent": 95, "experimental_supported_tools": [],
    "input_modalities": ["text", "image"], "supports_search_tool": False,
}


def catalog(out: Path, slug: str, prompt: Path, window: int = CATALOG_WINDOW) -> Path:
    """A model catalog that replaces Codex's bundled one, to put its compaction out of reach.

    `prompt` is the Codex release's `codex-rs/models-manager/prompt.md`, which is what its
    fallback metadata uses as the base instructions; the bundled catalog's own entries carry
    a different text, and taking one of those would quietly change the agent's system prompt.
    """

    entry = {"slug": slug, "display_name": slug, "base_instructions": prompt.read_text(encoding="utf-8"),
             "context_window": window, "max_context_window": window, **FALLBACK}
    path = out / "codex_model_catalog.json"
    path.write_text(json.dumps({"models": [entry]}, ensure_ascii=False), encoding="utf-8")
    return path


def assets(out: Path, port: int, bind: str = HOST) -> ThreadingHTTPServer:
    """Serve the Relay wheel and the catalog to the work containers, which reach this
    machine as host.docker.internal (SForge maps it to the host gateway)."""

    subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO), "-w", str(out), "--no-deps", "-q"],
                   check=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "pip<26", "-d", str(out),
                    "--only-binary=:all:", "--no-deps", "-q"], check=True)
    server = ThreadingHTTPServer((bind, port),
                                 partial(SimpleHTTPRequestHandler, directory=str(out)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def wheel(out: Path) -> str:
    """The wheel just built: pip never cleans this directory, so an older one may sit beside it."""

    return max(out.glob("relay-*.whl"), key=lambda p: p.stat().st_mtime).name


def arm_run(arm: str, out: Path, options: argparse.Namespace, assets_dir: Path) -> subprocess.Popen:
    """One SForge run. The agent is registered in-process rather than installed into
    SForge, so an EdgeBench checkout stays as it came."""

    extra = {"RELAY_STRATEGY": "off" if arm in ("codex", "native") else arm,
             "RELAY_ASSETS": f"http://{options.asset_host}:{options.asset_port}",
             "RELAY_WHEEL": wheel(assets_dir),
             "PIP_WHEEL": next(assets_dir.glob("pip-*.whl")).name,
             # `codex` is the arm that keeps Codex's own compaction, so it is the one
             # that does not take the catalog.
             "RELAY_CATALOG": "off" if arm == "codex" else "codex_model_catalog.json"}
    if options.window:
        extra["RELAY_CONTEXT_WINDOW"] = str(options.window)
    if options.pip_index:
        extra["PIP_INDEX"] = options.pip_index

    argv = ["sforge", "run", "--backend", "docker", "--agent", "codex-relay", "--model", options.model,
            "--effort", options.effort, "--timeout", str(options.timeout),
            "--eval-interval", str(options.eval_interval),
            "--stagger", str(options.stagger),
            "--run-id", f"{options.run_id}-{arm}", "--task", *options.tasks]
    if options.submission_cooldown is not None:
        argv += ["--submission-cooldown", str(options.submission_cooldown)]
    if options.enable_internet:
        argv += ["--enable-internet"]
    launcher = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "from evals.edgebench_agent import CodexRelayAgent;"
        "from sforge.harness.agent import factory;"
        "factory._REGISTRY['codex-relay'] = CodexRelayAgent;"
        "from evals.edgebench import collect_before_cleanup;"
        "from pathlib import Path;"
        "collect_before_cleanup(%r, Path(%r));"
        "from sforge.cli import main;"
        "sys.argv = %r;"
        "sys.exit(main())" % (str(REPO), f"{options.run_id}-{arm}", str(out), argv)
    )
    inherited = os.environ.get("SFORGE_AGENT_EXTRA_ENV", "")
    env = {**os.environ, "SFORGE_AGENT_EXTRA_ENV": ",".join(
        filter(None, [inherited, *(f"{k}={v}" for k, v in extra.items())]))}
    with (out / "sforge.log").open("a") as log:
        return subprocess.Popen([sys.executable, "-c", launcher], cwd=options.bundle,
                                stdout=log, stderr=log, env=env)


def collect_container(name: str, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    for source in (f"{HOME_RELAY}/events.jsonl", f"{HOME_RELAY}/serve.log",
                   "/home/agent/.live_ctx", "/home/agent/.codex/sessions"):
        final = into / Path(source).name
        staging = into / f".{final.name}.new"
        shutil.rmtree(staging, ignore_errors=True)
        if subprocess.run(["docker", "cp", f"{name}:{source}", str(staging)],
                          capture_output=True, timeout=120).returncode:
            continue
        if staging.is_dir():
            shutil.rmtree(final, ignore_errors=True)
        staging.replace(final)


def collect_before_cleanup(run_id: str, out: Path) -> None:
    """Process-local wrapper: preserve the final request before SForge removes the container.

    Final snapshots are separate from periodic copies so the two cannot overwrite each other.
    No SForge files are changed. This depends on its DockerBackend lifecycle interface.
    """
    from sforge.harness.backend.docker_backend import DockerBackend
    cleanup = DockerBackend.cleanup_container

    def preserving_cleanup(self, handle, logger=None):
        try:
            if handle and (match := re.fullmatch(r"sforge\.run\.(.+)\." + re.escape(run_id), handle.name)):
                collect_container(handle.name, out / match.group(1) / "final")
        except Exception as exc:
            print(f"Final collection failed for {run_id}: {exc}", file=sys.stderr)
        finally:
            cleanup(self, handle, logger)

    DockerBackend.cleanup_container = preserving_cleanup


def collect(run_ids: list[str], bundle: Path, out: Path, stop: threading.Event) -> None:
    """Copy what dies with the container: Relay's events and log, the strategy's own files,
    and Codex's rollout, which is where the tokens a task spent are recorded."""

    while not stop.wait(60):
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True).stdout.split()
        for name in names:
            match = re.fullmatch(r"sforge\.run\.(.+)\.(" + "|".join(map(re.escape, run_ids)) + r")", name)
            if not match:
                continue
            task, run_id = match.group(1), match.group(2)
            into = out / run_id.rsplit("-", 1)[-1] / task
            try:
                collect_container(name, into)
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"Collection failed for {name}: {exc}", file=sys.stderr)



def scores(task_dir: Path, rescale) -> list[tuple[float, float]]:
    """(when, score out of 100) for every submission, including a game task's own rounds."""

    from sforge.harness.score_rescale import rescale_score
    out = []
    for report in task_dir.glob("submissions/*/report.json"):
        data = json.loads(report.read_text())
        score = data.get("score_0_100")
        if score is None and data.get("score") is not None:
            score = rescale_score(rescale, data["score"]) if rescale else data["score"]
        if score is not None:
            out.append((float(data.get("submitted_at") or report.stat().st_mtime), float(score)))
    for result in task_dir.glob("submissions/*/game_result.json"):
        data = json.loads(result.read_text())
        if (peak := data.get("peak_score")) is not None:
            out.append((result.stat().st_mtime, float(peak)))
    return sorted(out)


def usage(sessions: Path) -> list[tuple[float, int, int, int]]:
    """(when, input, cached, output) spent since the event before, from Codex's own rollout.

    Codex records `token_count` twice per request with the same `last_token_usage`, so the
    per-request field cannot be summed; `total_token_usage` is cumulative and rises once per
    request, and a session's totals start again at zero when a run resumes into a new one.
    """

    from datetime import datetime
    out = []
    for rollout in sorted(sessions.rglob("rollout-*.jsonl")):
        before = (0, 0, 0)
        for line in rollout.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            payload = event.get("payload") or {}
            if payload.get("type") != "token_count" or not payload.get("info"):
                continue
            total = (payload["info"] or {}).get("total_token_usage") or {}
            if not total:
                continue
            now = (total.get("input_tokens", 0), total.get("cached_input_tokens", 0),
                   total.get("output_tokens", 0))
            if now == before:
                continue
            when = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
            out.append((when, *(max(0, a - b) for a, b in zip(now, before))))
            before = now
    return sorted(out)


def rows(out: Path, bundle: Path, run_id: str, arm: str, task: str,
         price_in: float, price_out: float) -> list[dict]:
    """A row per event, each carrying the best score so far and what has been spent."""

    spec = json.loads((bundle / "tasks" / f"{task}.json").read_text())["judge"]
    from sforge.harness.score_rescale import parse_rescale_spec
    task_dir = bundle / "logs" / "runs" / f"{run_id}-{arm}" / task
    events = [{"t": t, "score": s} for t, s in scores(task_dir, parse_rescale_spec(spec.get("rescale")))]
    collected = out / arm / task
    sessions = collected / "final" / "sessions"
    if not sessions.exists():
        sessions = collected / "sessions"
    events += [{"t": t, "in": i, "cached": c, "out": o} for t, i, c, o in usage(sessions)]
    events.sort(key=lambda e: e["t"])
    if not events:
        return []
    start = events[0]["t"]
    best = tokens_in = tokens_out = cached = 0
    result = []
    for event in events:
        best = max(best, event.get("score", 0))
        tokens_in += event.get("in", 0)
        cached += event.get("cached", 0)
        tokens_out += event.get("out", 0)
        result.append({"arm": arm, "task": task, "hours": round((event["t"] - start) / 3600, 5),
                       "score": event.get("score", ""), "best_score": round(best, 4),
                       "input_tokens": tokens_in, "cached_tokens": cached, "output_tokens": tokens_out,
                       "cost_usd": round(tokens_in * price_in + tokens_out * price_out, 6)})
    return result


def report(out: Path, bundle: Path, csv_path: Path | None,
           price_in: float, price_out: float) -> None:
    # The run ids the arms used, as the run that made this directory recorded them.
    run_id = json.loads((out / "run.json").read_text())["run_id"] if (out / "run.json").exists() else out.name
    arms = [a for a in ARMS if (out / a).exists()]
    metadata = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    tasks = sorted(set(metadata.get("tasks", [])) | {
        p.name for a in arms for p in (out / a).iterdir() if p.is_dir()})
    everything: list[dict] = []
    missing: list[str] = []
    print("| task | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for task in tasks:
        cells = []
        for arm in arms:
            task_rows = rows(out, bundle, run_id, arm, task, price_in, price_out)
            everything += task_rows
            if not task_rows:
                cells.append("—")
                continue
            last = task_rows[-1]
            if not last["input_tokens"]:
                # Nothing was collected out of the container, so the cost column is not zero,
                # it is unknown; say so rather than print a dollar sign over missing data.
                missing.append(f"{arm}/{task}")
            acted = out / arm / task / "events.jsonl"
            if (acted.parent / "final" / acted.name).exists():
                acted = acted.parent / "final" / acted.name
            changed = len(acted.read_text().splitlines()) if acted.exists() else 0
            cost_text = f"${last['cost_usd']:.2f}" if last["input_tokens"] else "cost unknown"
            cells.append(f"{last['best_score']:.2f} ({cost_text}, "
                         f"{last['input_tokens'] / 1e6:.2f}M in, changed {changed})")
        print(f"| {task} | " + " | ".join(cells) + " |")
    for arm in arms:
        mine = [r for r in everything if r["arm"] == arm]
        best = {t: max((r["best_score"] for r in mine if r["task"] == t), default=0) for t in tasks}
        cost = sum(r["cost_usd"] for t in tasks for r in [max((x for x in mine if x["task"] == t),
                   key=lambda x: x["hours"], default={"cost_usd": 0})])
        print(f"{arm}: mean {sum(best.values()) / max(1, len(tasks)):.2f} over {len(tasks)} tasks, "
              f"${cost:.2f} collected cost")
    if missing:
        print(f"no rollout collected for {', '.join(missing)}: their tokens and cost read as zero")
    if csv_path and everything:
        import csv
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(everything[0]))
            writer.writeheader()
            writer.writerows(everything)
        print(f"{len(everything)} rows in {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, help="the EdgeBench bundle SForge runs from")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, help="report on a run's OUT instead")
    parser.add_argument("--csv", type=Path, help="write the report's rows, for the curves")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--tasks", nargs="+", required=False, default=[])
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=43200, help="per task, as EdgeBench sets it")
    parser.add_argument("--eval-interval", type=int, default=1800)
    parser.add_argument("--submission-cooldown", type=int, default=None)
    parser.add_argument("--window", type=int, default=None, help="Relay's threshold; its own by default")
    parser.add_argument("--codex-prompt", type=Path, default=None, dest="catalog",
                        help="the Codex release's models-manager/prompt.md, from which the catalog "
                             "that disables Codex's own compaction is built; every arm but `codex` needs it")
    parser.add_argument("--catalog-window", type=int, default=CATALOG_WINDOW,
                        help="the window that catalog gives the model (its trigger is 90%% of it)")
    parser.add_argument("--asset-port", type=int, default=8899,
                        help="where the work containers fetch the wheel and the catalog")
    parser.add_argument("--asset-bind", default=HOST, help="local interface serving run assets")
    parser.add_argument("--asset-host", default=HOST, help="asset hostname reachable from work containers")
    parser.add_argument("--arm-delay", type=float, default=150, help="seconds between launching arms")
    parser.add_argument("--stagger", type=int, default=600,
                        help="seconds SForge spreads an arm's task launches over")
    parser.add_argument("--enable-internet", action="store_true",
                        help="override every task's own `internet: false`. SForge enforces that "
                             "setting with host iptables and needs passwordless sudo for it; "
                             "without either, runs are not comparable with EdgeBench's own numbers")
    parser.add_argument("--pip-index", default=None, help="a package index for the work containers")
    parser.add_argument("--run-id", default=f"relay-{time.time_ns()}",
                        help="SForge run prefix; unique by default to keep judge logs separate")
    parser.add_argument("--price-in", type=float, default=PRICE_IN * 1e6,
                        help="USD per million input tokens, for the cost column")
    parser.add_argument("--price-out", type=float, default=PRICE_OUT * 1e6,
                        help="USD per million output tokens")
    options = parser.parse_args()

    if options.report:
        if not options.bundle:
            parser.error("--bundle is required to read the runs the report describes")
        report(options.report, options.bundle, options.csv,
               options.price_in / 1e6, options.price_out / 1e6)
        return 0
    if not (options.bundle and options.out and options.tasks):
        parser.error("--bundle, --out and --tasks are required to run")
    if not options.catalog and set(options.arms) - {"codex"}:
        # Without it Codex compacts itself at 90% of the window it assumes, under every arm,
        # and an arm would measure Relay and Codex together. Better to refuse than to drift.
        parser.error("--codex-prompt is required for every arm but `codex` (see the module docstring)")

    options.bundle = options.bundle.expanduser().resolve()
    options.out = options.out.expanduser().resolve()
    if options.catalog:
        options.catalog = options.catalog.expanduser().resolve()
        if not options.catalog.is_file():
            parser.error("--codex-prompt must name an existing file")
    if not options.bundle.is_dir():
        parser.error("--bundle must name an existing directory")
    if (options.out / "run.json").exists():
        parser.error("--out already contains a run; use a new directory")
    options.out.mkdir(parents=True, exist_ok=True)
    (options.out / "run.json").write_text(json.dumps(
        {"run_id": options.run_id, "arms": options.arms, "tasks": options.tasks,
         "model": options.model, "effort": options.effort, "timeout": options.timeout}, indent=1))
    assets_dir = options.out / "assets"
    assets_dir.mkdir(exist_ok=True)
    if options.catalog:
        catalog(assets_dir, options.model, options.catalog, options.catalog_window)

    server = assets(assets_dir, options.asset_port, options.asset_bind)
    stop = threading.Event()
    run_ids = [f"{options.run_id}-{arm}" for arm in options.arms]
    collector = threading.Thread(target=collect, args=(run_ids, options.bundle, options.out, stop), daemon=True)
    collector.start()
    jobs = []
    try:
        for n, arm in enumerate(options.arms):
            if n:
                time.sleep(options.arm_delay)
            (options.out / arm).mkdir(exist_ok=True)
            jobs.append(arm_run(arm, options.out / arm, options, assets_dir))
        return max((abs(job.wait()) for job in jobs), default=0)
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            try:
                job.wait(timeout=30)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()
        stop.set()
        collector.join(timeout=5)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
