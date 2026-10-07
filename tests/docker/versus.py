"""Each harness's own compaction against Relay's, on a deterministic model (reader.py).

Both runs play the same session in the harness-test image: turn 1 reads four files, the
instruction files change, and turn 2 reads four more. In the native run the harness compacts
itself: between the turns with its own command (`manual`, where it has one headless), or when
its own trigger, set low, fires (`auto`). In the Relay run, `relay install` turns that off and
Relay compacts the same request (its threshold set just above the request before it). The model
reads the same files and writes the same summary in both, so the first request after the
compaction should match, summary wrapper aside; the report shows both, item by item, and their
differences.

python3 tests/docker/versus.py [--scenario manual|auto] [harness ...]     (reports in output/versus/)
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests" / "docker")]
import check  # noqa: E402

from relay.harnesses import HARNESSES  # noqa: E402
from relay.protocols import codec_for  # noqa: E402

# The spec each harness runs on: the protocol it speaks natively, so no gateway sits in between.
# (Codex: tests/test_codex_context_e2e.py; `codex exec` has no `/compact`.)
SPECS = {"claude_code": "claude_code:claude", "gemini_cli": "gemini_cli:gemini",
         "pi": "pi:gpt", "opencode": "opencode:gpt", "kilo": "kilo:gpt", "crush": "crush:gpt", "goose": "goose:gpt",
         "kimi_code": "kimi_code:gpt", "hermes": "hermes:gpt", "nanobot": "nanobot:gpt",
         "deepseek_harness": "deepseek_harness:gpt", "workbuddy": "workbuddy:gemini"}
PORT = 9100
READER = f"http://127.0.0.1:{PORT}"
ORIGINS = ("https://api.openai.com", "https://api.anthropic.com", "https://generativelanguage.googleapis.com")
SETUP = f"""(cd /relay && python3 -m tests.docker.reader {PORT} ~/reader.jsonl > ~/reader.log 2>&1 &)
until python3 -c 'import socket; socket.create_connection(("127.0.0.1", {PORT}))' 2>/dev/null; do sleep 0.2; done
""" + check.TO_OPENROUTER.replace("{openrouter}", READER)
FILES = [f"file_{n}.txt" for n in range(1, 9)]
SEEN: dict[tuple, list[dict]] = {}  # what Relay traced in each session
PROMPT1 = f"TASK: {' '.join(FILES[:4])} Read each of these files in full, one per tool call, then reply."
PROMPT2 = f"Background notes, not needed for the task:\n{check.NOTES}\n\nTASK: {' '.join(FILES[4:])} Read them too."
# Headless, between the turns (the others take the command for a prompt; WorkBuddy's and Gemini
# CLI's next turn resumes the whole history).
MANUAL = {"claude_code": '"/compact"', "nanobot": '"/compact"', "kilo": "--command compact"}
# Where check.py's early triggers miss this session, or keep what the harness would not by default
# (Relay compacts as the harness does with its defaults): Claude Code compacts on its own only
# when the model rejects a request; pi's window for the model is 272k (it keeps its default 20k);
# Hermes never compacts below 64k tokens (its files are larger); dsh keeps a fifth of its threshold.
AUTO = {**check.EARLY, "claude_code": "",
        "pi": check.JSON_MERGE.format("~/.pi/agent/settings.json", "compaction", '{"reserveTokens": 242000}'),
        "hermes": "sed -i 's/^model:/model:\\n  context_length: 65000/' ~/.hermes/config.yaml",
        "deepseek_harness": "printf -- '- id: compaction-basic\\n  config:\\n    thresholdRatio: 0.06\\n    retainRatio: 0.012\\n' "
                            ">> ~/.dsh/cordis.patch.yml"}
LINES = {"pi": 300, "hermes": 600}  # lines per file, where requests must grow faster
READER_ENV = {"claude_code": {"READER_LIMIT": "34000"}}
# `mid`: where the harness compacts within a turn only when the model rejects a request (Claude
# Code), in turn 2's tool loop.
MID = {"claude_code": {"READER_LIMIT": "36000"}}
TOOLS = {"claude_code": "Read"}  # where native compaction treats files read with it apart
KEYS = "OPENAI_API_KEY=sk-reader\nANTHROPIC_API_KEY=sk-ant-reader\nGEMINI_API_KEY=reader-key\n"
OUT = REPO / "output" / "versus"


def session(harness: str, scenario: str, relay: int | None) -> list[dict]:
    """Play the session; `relay` is Relay's threshold (None: the harness compacts itself)."""

    name = SPECS[harness]
    spec = check.SPECS[name]
    root = Path(tempfile.mkdtemp(prefix=f"versus-{harness}-{scenario}-{'relay' if relay else 'native'}-", dir=os.getenv("TMPDIR")))
    home, project = root / "home", root / "project"
    home.mkdir()
    project.mkdir()
    for n, file in enumerate(FILES, start=1):
        lines = [f"file {n} line {i}: the quick brown fox jumps over the lazy dog" for i in range(1, LINES.get(harness, 120) + 1)]
        (project / file).write_text("\n".join([*lines, f"CODE: {check.CODES[n - 1]}"]) + "\n")
    for file in check.STATE_FILES:
        (project / file).write_text("# Project instructions\n\nThe project codename is ALPHA.\n")
    reader = {"auto": READER_ENV, "mid": MID}.get(scenario, {}).get(harness, {})
    env = {**reader, "READER_TOOL": TOOLS.get(harness, ""),
           "PROMPT": PROMPT1, "PROMPT2": PROMPT2, "RELAY_COMPACT_THRESHOLD": str(relay or 10**9),
           "RELAY_COMPACT_MIN_GAIN": "0"}
    compact = ""
    if scenario == "manual" and not relay:
        command = spec.resume.replace('"$PROMPT"', MANUAL[harness])
        compact = f"cd /project && timeout 300 {command} > ~/compact.out 2>&1 < /dev/null"
    check.container(name, root, env, f"""
        cd /project && timeout 300 {spec.command} > ~/harness.out 2> ~/harness.err < /dev/null
        sed -i s/ALPHA/BETA/ {" ".join(check.STATE_FILES)}
        {compact}
        export PROMPT="$PROMPT2"
        cd /project && timeout 300 {spec.resume} > ~/harness2.out 2> ~/harness2.err < /dev/null
    """, timeout=1200, install=relay is not None, setup="\n".join([SETUP.format(harness=harness, origins=ORIGINS), AUTO[harness] if scenario in ("auto", "mid") else ""]))
    log, trace = home / "reader.jsonl", home / "trace.jsonl"
    SEEN[(harness, scenario, relay)] = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def summarized(record: dict) -> bool:
    answer = record.get("answer") or []
    return answer[:1] == ["text"] and str(answer[1]).startswith("READER SUMMARY")


def conversation(record: dict) -> bool:
    body = record.get("body") or {}
    answered = (record.get("answer") or [""])[0] != "overflow"
    return bool(body.get("tools")) and answered and not summarized(record) and codec_for(record["path"]) is not None


def after_compaction(records: list[dict]) -> dict | None:
    """The first request of the conversation after its (first) summary."""

    at = next((i for i, r in enumerate(records) if summarized(r)), None)
    return None if at is None else next((r for r in records[at + 1:] if conversation(r)), None)


NOISE = [(re.compile(p), r) for p, r in (
    (r"\b(call|fc|msg|resp|toolu|chatcmpl)[_-][0-9A-Za-z]+", r"\1_ID"),
    (r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])", "UUID"),
    (r"\b\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?Z?)?", "DATE"),
    (r"/tmp/[\w./-]+", "/tmp/PATH"),
    (r"READER SUMMARY: task=[^;]*; done=\d+\.", "<SUMMARY>"),
    (r"cc_version=[\w.]+", "cc_version=VERSION"),  # Claude Code hashes its request into this
    (r'"id": "(\w+?)_\d{10,}_\d+"', r'"id": "\1_ID"'),  # Gemini CLI's call ids, by time
    (r"Process Group PGID: \d+", "Process Group PGID: N"),  # and the process its command ran in
)]


def render(record: dict, harness: str) -> list[str]:
    """The request as the model sees it: its preamble, then each item by kind."""

    codec, profile = codec_for(record["path"]), HARNESSES[harness]
    body = record["body"]
    lines = [f"PREAMBLE: {codec.preamble(body)}"]
    for wire in codec.items(body):
        item = profile.refine(codec.classify(wire))
        lines.append(f"{item.kind.value.upper()}: {item.text.strip()}")
    text = "\n".join(lines)
    for pattern, replacement in NOISE:
        text = pattern.sub(replacement, text)
    return text.splitlines()


def compare(harness: str, scenario: str) -> str:
    label = f"{harness} {scenario}"
    native = session(harness, scenario, None)
    native_view = after_compaction(native)
    if native_view is None:
        return f"{label}: the harness did not compact (or nothing after it); {len(native)} requests"
    first = next(i for i, r in enumerate(native) if summarized(r))
    # The harness compacted as the model rejected a request: so does Relay, on that same request.
    # Else Relay's threshold falls between its own counts of the last request the model answered
    # before the harness compacted and the next one (as a session through Relay that never
    # compacts counts them).
    threshold = 10**9
    if not any((r.get("answer") or [""])[0] == "overflow" for r in native[:first]):
        answered = sum(conversation(r) for r in native[:first])
        session(harness, scenario, 10**9)
        counts = [t["tokens"] for t in SEEN[(harness, scenario, 10**9)] if "tokens" in t and (t.get("body") or {}).get("tools")]
        if not 0 < answered < len(counts):
            return f"{label}: could not place Relay's threshold ({answered} answered, {len(counts)} counted)"
        threshold = (counts[answered - 1] + counts[answered]) // 2
    relayed = session(harness, scenario, threshold)
    relay_view = after_compaction(relayed)
    if relay_view is None:
        return f"{label}: Relay did not compact at {threshold}; {len(relayed)} requests"
    a, b = render(native_view, harness), render(relay_view, harness)
    diff = list(difflib.unified_diff(a, b, "native", "relay", lineterm="", n=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{harness}-{scenario}.txt").write_text("\n".join(["### native", *a, "", "### relay", *b, "", "### diff", *diff]))
    (OUT / f"{harness}-{scenario}.json").write_text(json.dumps({"native": native, "relay": relayed}))
    changed = sum(line[:1] in "+-" for line in diff[2:])
    return f"{label}: {'SAME' if not diff else f'{changed} lines differ'} (threshold {threshold})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("harnesses", nargs="*", default=list(SPECS))
    parser.add_argument("--scenario", choices=["manual", "auto", "mid"], action="append")
    args = parser.parse_args()
    runs = [(h, s) for s in args.scenario or ["manual", "auto", "mid"] for h in args.harnesses
            if s == "auto" or h in {"manual": MANUAL, "mid": MID}[s]]
    keys = Path(tempfile.mkdtemp(dir=os.getenv("TMPDIR"))) / "keys"
    keys.write_text(KEYS)
    os.environ["RELAY_TEST_KEYS"] = str(keys)
    os.environ.pop("RELAY_CHECK_OPENROUTER", None)
    with ThreadPoolExecutor(len(runs)) as pool:
        for line in pool.map(lambda run: compare(*run), runs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
