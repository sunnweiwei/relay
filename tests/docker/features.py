"""What a harness tells its model besides the conversation, and whether Relay's compaction keeps it.

    RELAY_TEST_KEYS=keys.env python tests/docker/features.py claude_code:gpt [gemini_cli:gpt ...]
    RELAY_TEST_KEYS=keys.env python tests/docker/features.py --native crush:gpt   (it compacts itself)
    python tests/docker/features.py --report HOME [...]     (again, from a run's home)

One session per harness, through Relay recording only (it never compacts), that exercises what
the harness injects: instruction files (changed between turns), a todo list, a sub-agent, a skill,
an MCP server with instructions (tests/docker/probe_mcp.py), a file it read changed under it.
Then the harness compacts itself (its own command, where its headless mode has one; with
`--native`, its automatic compaction, turned back on low, after a turn that reads six more files)
and is asked what it knows.

The report, from the recorded requests:
- every kind of content the harness wrote into the main conversation, as Relay's profile reads
  it (system, context, state key; user messages carrying injected blocks are flagged);
- what the harness put back after compacting itself (its notion of its state), against what Relay
  takes as its state before that point, and what Relay's own compaction of that request keeps:
  markers (the instructions' codename, the skill, the MCP server's instructions, the todo steps)
  found in injected content (not in what the user typed) of each, the request's own system prompt
  (`instructions`, `system`, `systemInstruction`) included.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check import CODES, NATIVE_COMPACT, REPO, SPECS, STATE_FILES, compact_turn, container, self_compactions  # noqa: E402

sys.path.insert(0, str(REPO))
from relay.core.engine import Engine  # noqa: E402
from relay.core.ir import AGENT_KINDS, Kind  # noqa: E402
from relay.core.tokens import approx_tokens  # noqa: E402
from relay.harnesses import detect  # noqa: E402
from relay.protocols import codec_for  # noqa: E402
from relay.strategies import Compaction  # noqa: E402
from tests.test_replay import SUMMARIES  # noqa: E402

TURNS = (
    "This project has instruction files: what project codename do they give? Use your todo or task-tracking tool, "
    "if you have one, to plan two steps named exactly 'TANGO: read file_1.txt' and 'UNIFORM: read file_2.txt'. "
    "Then read file_1.txt in full. If you have a tool from an MCP server named probe, call its probe_word tool. "
    "If you have a skill named relay-probe, use it to find the probe word. Reply with the codename, the CODE at the "
    "end of file_1.txt, the MCP tool's word (or NONE) and the probe word (or NONE).",
    "If you can delegate work to a sub-agent, have one read file_2.txt and report its CODE; otherwise read it "
    "yourself. Mark your todo steps done if you have them. Reply with that CODE and the project codename as your "
    "project instructions state it now.",
    "Do not read files or call tools. From what you can see, reply: the project codename as your instructions "
    "state it now; the skills you have; the MCP servers you have and what their instructions say; your todo list.",
)
# `--native`: before the last turn the session grows (six more files), and the harness's own automatic
# compaction, turned back on and low, compacts it.
GROW = ("Read file_3.txt to file_8.txt in the current directory one at a time, each in full, and reply with their "
        "CODEs in order.")
TYPED = ("This project has instruction files", "If you can delegate", "Do not read files", "Read file_3.txt to file_8",
         *NATIVE_COMPACT.values())
SKILL = """---
name: relay-probe
description: Gives the probe word. Use it when asked for the probe word.
---
The probe word is QUARTZ.
"""
SKILL_DIRS = ("~/.claude/skills", "~/.agents/skills", "~/.codex/skills", "~/.gemini/skills", "~/.config/opencode/skills",
              "~/.config/kilo/skills", "~/.pi/agent/skills", "~/.workbuddy/skills", "~/.codebuddy/skills",
              "~/.kimi-code/skills", "~/.hermes/skills", "~/.config/crush/skills", "~/.config/goose/skills",
              "/project/skills")
SERVER = ["python3", "/relay/tests/docker/probe_mcp.py"]
MCP_JSON = """python3 -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
c = json.loads(p.read_text()) if p.exists() else {}; n = c
for k in sys.argv[2].split(".")[:-1]: n = n.setdefault(k, {})
n[sys.argv[2].split(".")[-1]] = json.loads(sys.argv[3]); p.write_text(json.dumps(c))' """
SERVE = " ".join(SERVER)
MCP = {  # the user's MCP server, as each harness adds one
    "claude_code": f"claude mcp add -s user probe -- {SERVE}",
    "codex": f"codex mcp add probe -- {SERVE}",
    "gemini_cli": f"gemini mcp add -s user --trust probe {SERVE}",
    "workbuddy": f"CODEBUDDY_CONFIG_DIR=$HOME/.workbuddy codebuddy mcp add -s user probe -- {SERVE}",
    "opencode": MCP_JSON + f"~/.config/opencode/opencode.json mcp.probe '{json.dumps({'type': 'local', 'command': SERVER})}'",
    "kilo": MCP_JSON + f"~/.config/kilo/config.json mcp.probe '{json.dumps({'type': 'local', 'command': SERVER})}'",
    "crush": MCP_JSON + f"~/.config/crush/crush.json mcp.probe '{json.dumps({'type': 'stdio', 'command': 'python3', 'args': SERVER[1:]})}'",
    "openclaw": f"openclaw mcp set probe '{json.dumps({'command': 'python3', 'args': SERVER[1:]})}'",
    "goose": "printf '  probe:\\n    enabled: true\\n    name: probe\\n    type: stdio\\n    cmd: python3\\n    args: [%s]\\n"
             "    timeout: 300\\n' " + SERVER[1] + " >> ~/.config/goose/config.yaml",
    "kimi_code": MCP_JSON + f"~/.kimi-code/mcp.json mcpServers.probe '{json.dumps({'command': 'python3', 'args': SERVER[1:]})}'",
    "hermes": f"printf 'mcp_servers:\\n  probe:\\n    command: python3\\n    args: [{SERVER[1]}]\\n' >> ~/.hermes/config.yaml",
    "nanobot": MCP_JSON + f"~/.nanobot/config.json tools.mcpServers.probe '{json.dumps({'command': 'python3', 'args': SERVER[1:]})}'",
}
# Harnesses whose headless mode has no compaction command: their automatic trigger, low, after `relay install`.
AUTO = {"gemini_cli": MCP_JSON + "~/.gemini/settings.json model.compressionThreshold 0.05"}
YAML_SET = """python3 -c 'import yaml, pathlib, sys; p = pathlib.Path(sys.argv[1]).expanduser(); c = yaml.safe_load(p.read_text()) or {}; n = c
for k in sys.argv[2].split(".")[:-1]: n = n.setdefault(k, {})
n[sys.argv[2].split(".")[-1]] = yaml.safe_load(sys.argv[3]); p.write_text(yaml.safe_dump(c))' """
# `--native`: each harness's automatic compaction on again, at about 15k tokens, after `relay install`.
NATIVE_AUTO = {
    "gemini_cli": MCP_JSON + "~/.gemini/settings.json model.compressionThreshold 0.01",
    "pi": MCP_JSON + "~/.pi/agent/settings.json compaction '{\"enabled\": true, \"reserveTokens\": 255000, \"keepRecentTokens\": 2000}'",
    "opencode": MCP_JSON + "~/.config/opencode/opencode.json compaction.auto true && " + MCP_JSON
                + "~/.config/opencode/opencode.json provider.openai.models.gpt-6-luna '{\"limit\": {\"context\": 20000, \"output\": 4000}}'",
    "kilo": MCP_JSON + "~/.config/kilo/config.json compaction.auto true && " + MCP_JSON
            + "~/.config/kilo/config.json provider.openai.models.gpt-6-luna '{\"limit\": {\"context\": 20000, \"output\": 4000}}'",
    "crush": MCP_JSON + "~/.config/crush/crush.json options.disable_auto_summarize false && "
             "sed -i 's/\"context_window\": 200000/\"context_window\": 20000/' ~/.config/crush/crush.json",
    "openclaw": MCP_JSON + "~/.openclaw/openclaw.json agents.defaults.compaction.enabled true && " + MCP_JSON
                + "~/.openclaw/openclaw.json models.providers.openai.models "
                  "'[{\"id\": \"gpt-6-luna\", \"name\": \"gpt-6-luna\", \"contextWindow\": 24000, \"maxTokens\": 4000}]'",
    "kimi_code": "sed -i -e 's/^max_context_size = .*/max_context_size = 22000/' -e 's/^max_output_size = .*/max_output_size = 4000/' "
                 "~/.kimi-code/config.toml",
    "hermes": YAML_SET + "~/.hermes/config.yaml compression '{enabled: true, threshold: 0.25}' && " + YAML_SET
              + "~/.hermes/config.yaml model.context_length 64000",
    "deepseek_harness": "printf -- '- id: compaction-basic\\n  config:\\n    auto: true\\n    thresholdRatio: 0.05\\n"
                        "    retainRatio: 0.02\\n' >> ~/.dsh/cordis.patch.yml",
}
MARKERS = {"codename": "BETA", "old codename": "ALPHA", "skill": "relay-probe", "MCP instructions": "MERCURY",
           "todo": "UNIFORM", "file change": "VIOLET", "cwd": "/project"}


def run(name: str, native: bool = False) -> Path:
    spec, harness = SPECS[name], name.split(":")[0]
    root = Path(tempfile.mkdtemp(prefix=f"relay-features-{harness}-"))
    home, project = root / "home", root / "project"
    home.mkdir()
    project.mkdir()
    for n, code in enumerate(CODES, start=1):
        lines = [f"file {n} line {i}: the quick brown fox jumps over the lazy dog" for i in range(1, 41 if n < 3 else 121)]
        (project / f"file_{n}.txt").write_text("\n".join([*lines, f"CODE: {code}"]) + "\n")
    for file in STATE_FILES:
        (project / file).write_text("# Project instructions\n\nThe project codename is ALPHA.\n")
    skills = " && ".join(f"mkdir -p {d}/relay-probe && cat > {d}/relay-probe/SKILL.md <<'MD'\n{SKILL}MD" for d in SKILL_DIRS)

    def turn(n: int, prompt: str) -> str:
        command = spec.command if n == 1 else spec.resume
        return f"export PROMPT={json.dumps(prompt)} && cd /project && timeout 600 {command} > ~/turn{n}.out 2> ~/turn{n}.err < /dev/null"

    compact = NATIVE_COMPACT.get(harness)
    turns = "\n".join([
        NATIVE_AUTO.get(harness, "") if native else AUTO.get(harness, ""),
        turn(1, TURNS[0]),
        "sed -i s/ALPHA/BETA/ " + " ".join(STATE_FILES) + " && echo 'file 1 was edited: VIOLET' >> /project/file_1.txt",
        turn(2, TURNS[1]),
        turn(5, GROW) if native else "",
        compact_turn(harness, spec, "~/turn3.out") if compact and harness not in AUTO and not (native and harness in NATIVE_AUTO) else "",
        turn(4, TURNS[2]),
    ])
    env = {"compact_growth": str(10**9), "compact_min_gain": "0"}  # Relay records; it never compacts
    container(name, root, env, turns, timeout=3600, setup=f"{skills}\n{MCP.get(harness, '')}")
    return home


def typed(text: str) -> bool:
    """Whether a user message is one of the session's prompts (after a timestamp or a wrapping tag)."""

    return any(prompt in text[:300] for prompt in TYPED[:4])


def injected(harness, item) -> str:
    """What the harness wrote of an item: all of a system or context item, the injected blocks of a
    user message."""

    if item.kind in (Kind.SYSTEM, Kind.CONTEXT):
        return item.text
    if item.kind is Kind.USER:
        return "\n".join(m.group(0) for pattern in harness.injected for m in pattern.finditer(item.text))
    return ""


def short(text: str, n: int = 130) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def report(home: Path) -> None:
    requests = [r for r in map(json.loads, (home / "trace.jsonl").open()) if isinstance(r.get("body"), dict)]
    if not requests:
        print(f"== {home}: no requests")
        return
    codec = codec_for(requests[0]["path"])
    harness = detect({}, path=requests[0]["path"])
    views = [[harness.refine(codec.classify(w)) for w in codec.items(r["body"])] for r in requests]
    main = [n for n, (request, view) in enumerate(zip(requests, views))  # (helpers, as title writers, have no tools)
            if request["body"].get("tools") and any(i.kind is Kind.USER and typed(harness.visible(i.text)) for i in view)]
    print(f"== {harness.name}: {len(requests)} requests, {len(main)} in the main conversation  ({home})")

    seen: dict[tuple, list] = {}
    for n in main:
        for item in views[n]:
            if item.kind in AGENT_KINDS or item.kind is Kind.SUMMARY:
                continue
            text = injected(harness, item)
            said = harness.visible(item.text) if item.kind is Kind.USER else ""
            if item.kind is Kind.USER and not text:
                if not any(t in said for t in TYPED):
                    seen.setdefault(("user?", "", short(said, 70)), [0, said])[0] += 1
                continue
            label = "user+inj" if item.kind is Kind.USER else item.kind.value
            key = (label, harness.state_key(item) or "", short(re.sub(r"\d+", "N", text), 70))
            seen.setdefault(key, [0, text])[0] += 1
    print("  what the harness wrote (kind, state key, count):")
    for (kind, key, _), (count, text) in seen.items():
        print(f"    {kind:9} {key:16} x{count:<4} {short(text)}")

    compactions = [int(line.split(":")[0]) for line in self_compactions(requests)]
    native = next((n for n in compactions if n in main or n > main[0]), None) if main else None
    pre = max((n for n in main if native is None or n < native), default=None)
    post = next((n for n in main if native is not None and n > native and n not in compactions), None)
    if pre is None:
        return

    def marks(texts: list[str], body: dict) -> str:
        joined = "\n".join([*texts, *(json.dumps(body.get(k)) for k in ("instructions", "system", "systemInstruction"))])
        return " ".join(f"{k}={'Y' if v in joined else '-'}" for k, v in MARKERS.items())

    pre_view = views[pre]
    state = harness.state(codec, codec.items(requests[pre]["body"]))
    print(f"  Relay's state at request {pre}:")
    for item in state.items:
        print(f"    {item.kind.value:9} {short(item.text)}")

    engine = Engine(Compaction(threshold=1, min_gain=0))
    for n in main:
        if n > pre:
            break
        exchange = engine.prepare(codec, harness, requests[n]["body"], tenant="t",
                                  post=lambda request: (200, SUMMARIES[codec.name]))
        engine.record(exchange, approx_tokens(json.dumps(exchange.body)))
    sent = [harness.refine(codec.classify(w)) for w in codec.items(exchange.body)]
    print(f"  Relay's compaction of request {pre} (compacted: {exchange.compacted}):")
    print("    " + " ".join(LAYOUT(i) for i in sent))

    print("  markers in what the harness wrote:")
    print(f"    before ({pre}):          {marks([injected(harness, i) for i in pre_view], requests[pre]["body"])}")
    if post is not None:
        after = [injected(harness, i) for i in views[post]]
        print(f"    harness compacted ({post}): {marks(after, requests[post]['body'])}")
        print("    its request after compacting itself:")
        for item in views[post]:
            if item.kind in AGENT_KINDS:
                break
            print(f"      {item.kind.value:9} {short(item.text)}")
    else:
        print("    (the harness did not compact itself)")
    print(f"    Relay compacted:        {marks([injected(harness, i) for i in sent], exchange.body)}")


def LAYOUT(item) -> str:  # noqa: N802
    return {Kind.SYSTEM: "S", Kind.CONTEXT: "C", Kind.USER: "U", Kind.SUMMARY: "Y"}.get(item.kind, ".")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--report", nargs="*", default=[], help="report on these run homes")
    parser.add_argument("--native", action="store_true", help="grow the session and let the harness compact itself")
    options = parser.parse_args()
    for home in [*map(Path, options.report), *(run(name, options.native) for name in options.names)]:
        report(home)
    return 0


if __name__ == "__main__":
    sys.exit(main())
