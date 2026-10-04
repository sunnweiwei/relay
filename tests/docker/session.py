"""A user's working session through Relay, inside Docker: a real repository, several turns.

    python tests/docker/session.py codex:gpt-api --repo PATH [--window 64000 | --baseline]

PATH is a checkout of more-itertools with a bug planted in `windowed` (the last, padded
window is dropped when step < n). Over the harness's own resume the user asks for a tour of
the code base, a fix for the reported bug, then (after adding a changelog rule to the
project's instructions) a small feature for `chunked`, and finally a summary of the session.
Relay compacts at 90% of a small context window, so a session compacts several times;
`--baseline` runs the same session through Relay without compacting, for comparison;
`--transition` has Relay join the session late, drop out for a turn, or restart (losing its
state) before every turn, while the harness keeps resending its whole history.

Afterwards, hidden checks in the container: the bug is fixed, the feature works, its stub,
tests and changelog entry exist (the changelog rule arrived after earlier compactions), the
suite passes, and the summary recalls both changes. Relay's own records give the
compactions, cache hits, divergences and rejected calls; the trace gives the cost.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
import zlib
from pathlib import Path

from check import SPECS, container, self_compactions

TURNS = [
    "I just cloned this repo (more-itertools) and I'm new to it. Give me a short tour: how the package is "
    "organized, where functions are defined, how the type stubs and docs relate to the code, and how to run "
    "the tests. Look around as much as you need, then keep the answer brief.",
    "A user reported a bug: `list(windowed([1, 2, 3, 4, 5, 6], 3, step=2))` returns `[(1, 2, 3), (3, 4, 5)]`, "
    "but the last window should be padded and included: `(5, 6, None)`. Please find the cause, fix it, add a "
    "regression test, and run the test suite.",
    "Next, a small feature: let `chunked` take an optional `fillvalue`. When it is given, pad the last chunk "
    "with it up to length n (like `grouper` does), e.g. `list(chunked([1, 2, 3, 4, 5], 2, fillvalue=0))` gives "
    "`[[1, 2], [3, 4], [5, 0]]`. Combining it with `strict=True` should raise ValueError. Update the docstring, "
    "the type stub and the tests, and follow the project's instructions.",
    "Thanks! Before I open a PR: summarize everything we changed in this session (which files, and why), and "
    "confirm that the full test suite passes.",
]
# `--tools`: the same work, with the requests coming from the team's ticket tracker (an MCP
# server), a mockup to look at (an image) and a release to look up (web search).
TOOL_TURNS = [
    "I just cloned this repo (more-itertools) and I'm new to it. Give me a short tour of how it is organized and "
    "how to run the tests. Our designer also left a mockup at mockup.png in the repo root: open it with your "
    "image viewing tool and tell me what shape and color it shows.",
    "Please handle ticket T-101 from our tracker (read it with the `tickets` MCP tool): find the cause, fix it, "
    "add a regression test, and run the test suite.",
    "Next, ticket T-102 (read it with the `tickets` tool); update the docstring, the type stub and the tests, and "
    "follow the project's instructions. If you have a web search tool, also look up the latest more-itertools "
    "release on PyPI and tell me its version.",
    "Thanks! Before I open a PR: which tickets did we handle and what changed for each (which files, and why)? "
    "What did the mockup show? And confirm that the full test suite passes.",
]
TICKETS = r'''
import json, sys
TICKETS = {
    "T-101": "Bug: list(windowed([1, 2, 3, 4, 5, 6], 3, step=2)) returns [(1, 2, 3), (3, 4, 5)], but the last window "
             "should be padded and included: (5, 6, None).",
    "T-102": "Feature: let chunked take an optional fillvalue. When it is given, pad the last chunk with it up to "
             "length n (like grouper does): list(chunked([1, 2, 3, 4, 5], 2, fillvalue=0)) gives [[1, 2], [3, 4], "
             "[5, 0]]. Combining it with strict=True should raise ValueError.",
}
TOOL = {"name": "get_ticket", "description": "Read a ticket from the team's tracker by its id, e.g. T-101.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}
for line in sys.stdin:  # MCP over stdio: one JSON-RPC message per line
    message = json.loads(line)
    method, params = message.get("method"), message.get("params") or {}
    if "id" not in message:
        continue  # a notification
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                  "serverInfo": {"name": "tickets", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        ticket = TICKETS.get(str((params.get("arguments") or {}).get("id", "")).upper())
        result = {"content": [{"type": "text", "text": ticket or "No such ticket."}], "isError": ticket is None}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
'''
MCP = {  # the user's harness, with the tracker connected (and web search on, where it is a setting)
    "codex": """cat >> ~/.codex/config.toml <<'TOML'
web_search = "live"
[mcp_servers.tickets]
command = "python3"
args = ["/home/agent/tickets.py"]
TOML""",
    "claude_code": """python3 -c 'import json, pathlib; p = pathlib.Path.home() / ".claude.json"; c = json.loads(p.read_text()); \\
c["mcpServers"] = {"tickets": {"type": "stdio", "command": "python3", "args": ["/home/agent/tickets.py"]}}; p.write_text(json.dumps(c))'""",
}


# `--harness-compact N`: the harness's own auto-compaction fires at about N tokens, as it would
# for a user who left it on; below Relay's trigger the harness compacts first.
HARNESS_COMPACT = {
    "codex": "echo 'model_auto_compact_token_limit = {limit}' > ~/limit.toml && cat ~/.codex/config.toml >> ~/limit.toml "
             "&& mv ~/limit.toml ~/.codex/config.toml",
    "claude_code": "export CLAUDE_CODE_AUTO_COMPACT_WINDOW={limit} CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=100",
}


def square_png(size: int = 96, low: int = 24, high: int = 72) -> bytes:
    """A red square on white."""

    rows = b"".join(b"\0" + b"".join(b"\xd0\x20\x20" if low <= x < high and low <= y < high else b"\xff\xff\xff"
                                      for x in range(size)) for y in range(size))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))  # noqa: E731
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


# `--branch`: after turn 3 the user forks the session to ask something else (and, in Claude
# Code, rewinds it to the end of turn 2 on another branch), then goes on with the original:
# each branch must find the compactions of the history it shares.
FORK = ("A different direction, just to compare: how would `chunked` look if, instead of `fillvalue`, it took a "
        "`pad=True` flag that pads with None? Explain in a few sentences; don't change any files.")
REWIND = ("Before any new feature: which test did you add for the `windowed` fix, and what exactly does it check? "
          "Answer briefly; don't change any files.")
SESSION_ID = {  # the session the turns so far belong to
    "codex": "$(ls -t ~/.codex/sessions/*/*/*/rollout-*.jsonl | head -1 | grep -o '[0-9a-f-]\\{36\\}.jsonl$' | cut -c1-36)",
    "claude_code": "$(basename $(ls -t ~/.claude/projects/*/*.jsonl | head -1) .jsonl)",
}
BRANCH = {  # (fork the session, rewind it to message $AT on a new branch, resume the original), by harness
    "codex": ('codex exec --skip-git-repo-check fork $SID "$PROMPT"', None,
              'codex exec --skip-git-repo-check resume $SID "$PROMPT"'),
    "claude_code": ('claude --resume $SID --fork-session {args}', 'claude --resume $SID --resume-session-at $AT --fork-session {args}',
                    'claude --resume $SID {args}'),
}
# The last message of turn 2 in a Claude Code session file (its line count was saved after turn 2).
TURN2_END = """AT=$(head -n $(cat ~/turn2.lines) ~/.claude/projects/*/$SID.jsonl | python3 -c 'import json, sys
print([e for e in map(json.loads, sys.stdin) if e.get("type") == "assistant"][-1]["uuid"])')"""


INSTRUCTIONS = ("# Notes for coding agents\n\n- Run the test suite with `python3 -m unittest` from the repository "
                "root.\n- Keep changes small and in the existing code style.\n")
RULE = "- Record every user-visible change in the Unreleased section at the top of `docs/versions.rst`.\n"
FILES = ("AGENTS.md", "CLAUDE.md")
# What happens to Relay before a turn: the user installs it late, removes it for a while, or it
# restarts and loses its state, while the harness keeps (and resends) the whole history.
PORT_OPEN = "python3 -c 'import socket; socket.create_connection((\"127.0.0.1\", 8787))' 2>/dev/null"
RELAY = {
    "install": "cd /relay && python3 -m relay.cli install {harness} >> ~/relay.log 2>&1",
    "uninstall": "cd /relay && python3 -m relay.cli uninstall {harness} >> ~/relay.log 2>&1",
    "restart": f"""kill $(cat ~/relay.pid) && while {PORT_OPEN}; do sleep 0.2; done
        cd /relay
        python3 -m relay.cli serve >> ~/relay.log 2>&1 & echo $! > ~/relay.pid
        until {PORT_OPEN}; do sleep 0.2; done""",
}
TRANSITIONS = {
    "late-join": {3: "install"},  # turns 1 and 2 talk to the provider directly
    "drop-out": {3: "uninstall", 4: "install"},
    "restart": {2: "restart", 3: "restart", 4: "restart"},
}

VERDICT = r'''
import json, os, subprocess, sys
sys.path.insert(0, "/project")
import more_itertools as mi

def added(*paths):  # lines the session added to these paths, new files included
    subprocess.run(["git", "add", "-A", "-N"], cwd="/project")
    diff = subprocess.run(["git", "diff", "HEAD", "--", *paths], cwd="/project", capture_output=True, text=True).stdout
    return "\n".join(line for line in diff.splitlines() if line.startswith("+"))

def strict_conflict():
    try:
        list(mi.chunked([1, 2, 3], 2, strict=True, fillvalue=0))
    except ValueError:
        return True
    return False

checks = {
    "windowed fixed": lambda: list(mi.windowed([1, 2, 3, 4, 5, 6], 3, step=2)) == [(1, 2, 3), (3, 4, 5), (5, 6, None)]
        and list(mi.windowed([1, 2, 3, 4, 5], 4, step=3)) == [(1, 2, 3, 4), (4, 5, None, None)]
        and list(mi.windowed(range(7), 3, step=2)) == [(0, 1, 2), (2, 3, 4), (4, 5, 6)]
        and list(mi.windowed([1, 2, 3, 4, 5], 2, step=3)) == [(1, 2), (4, 5)],
    "chunked fillvalue": lambda: list(mi.chunked([1, 2, 3, 4, 5], 2, fillvalue=0)) == [[1, 2], [3, 4], [5, 0]]
        and list(mi.chunked([1, 2, 3, 4], 2, fillvalue=0)) == [[1, 2], [3, 4]]
        and list(mi.chunked([1, 2, 3], 2)) == [[1, 2], [3]],
    "strict with fillvalue raises": strict_conflict,
    "stub updated": lambda: "fillvalue" in added("more_itertools/more.pyi"),
    "tests added": lambda: "windowed" in added("tests") and "fillvalue" in added("tests"),
    "changelog entry (rule added before turn 3)": lambda: "chunked" in added("docs/versions.rst"),
    "suite passes": lambda: open("/home/agent/suite.status").read().strip() == "0",
    "summary recalls both changes": lambda: all(word in open("/home/agent/turn4.out", errors="replace").read()
                                                for word in ("windowed", "chunked")),
}
if os.path.exists("/home/agent/tickets.py"):  # `--tools`: the tickets and the mockup are remembered too
    checks["summary recalls tickets and mockup"] = lambda: all(
        word in open("/home/agent/turn4.out", errors="replace").read().lower() for word in ("t-101", "t-102", "red", "square"))
results = {}
for name, check in checks.items():
    try:
        results[name] = bool(check())
    except Exception as error:
        results[name] = f"{type(error).__name__}: {error}"[:200]
print(json.dumps(results))
'''


def run(name: str, repo: Path, window: int, transition: str | None = None, tools: bool = False,
        branch: bool = False, harness_compact: int | None = None) -> dict:
    spec, harness, relay = SPECS[name], name.split(":")[0], TRANSITIONS.get(transition or "", {})
    prompts = TOOL_TURNS if tools else TURNS
    if branch:  # the turns after the first resume this session by id: a fork may be the most recent
        args = spec.command[spec.command.index("-p "):] if harness == "claude_code" else ""
        fork, rewind, resume = (command and command.format(args=args) for command in BRANCH[harness])
    root = Path(tempfile.mkdtemp(prefix=f"session-{name.replace(':', '-')}-", dir=os.getenv("TMPDIR")))
    home, project = root / "home", root / "project"
    home.mkdir()
    shutil.copytree(repo, project, ignore=shutil.ignore_patterns(".git"))
    for file in FILES:
        (project / file).write_text(INSTRUCTIONS)
    (home / "verdict.py").write_text(VERDICT)
    if tools:
        (home / "tickets.py").write_text(TICKETS)
        (project / "mockup.png").write_bytes(square_png())
    env = {f"PROMPT{n}": text for n, text in enumerate(prompts, start=1)} | {"RELAY_CONTEXT_WINDOW": str(window),
                                                                           "FORK": FORK, "REWIND": REWIND}
    turns = [f"""{MCP[harness] if tools else ""}
        {HARNESS_COMPACT[harness].format(limit=harness_compact) if harness_compact else ""}
        git config --global user.name dev && git config --global user.email dev@example.com
        cd /project && git init -q && git add -A && git commit -qm 'Import more-itertools'"""]
    for n in range(1, len(prompts) + 1):
        command = spec.command if n == 1 else resume if branch else spec.resume
        if branch and n == 4:
            turns.append(f"""export PROMPT="$FORK" && cd /project && timeout 600 {fork} > ~/fork.out 2> ~/fork.err < /dev/null
        {f'{TURN2_END} && export PROMPT="$REWIND" && timeout 600 {rewind} > ~/rewind.out 2> ~/rewind.err < /dev/null' if rewind else ""}""")
        turns.append(f"""{RELAY[relay[n]].format(harness=harness) if n in relay else ""}
        {f"printf '%s' '{RULE}' | tee -a {' '.join('/project/' + file for file in FILES)} > /dev/null" if n == 3 else ""}
        export PROMPT="$PROMPT{n}" && start=$(date +%s)
        cd /project && timeout 900 {command} > ~/turn{n}.out 2> ~/turn{n}.err < /dev/null
        echo {n} $start $(date +%s) >> ~/times
        {f"SID={SESSION_ID[harness]}" if branch and n == 1 else ""}
        {f"cat ~/.claude/projects/*/$SID.jsonl | wc -l > ~/turn2.lines" if branch and n == 2 and harness == "claude_code" else ""}""")
    turns.append("""cd /project && python3 -m unittest -q > ~/suite.out 2>&1; echo $? > ~/suite.status
        python3 ~/verdict.py > ~/verdict.json 2> ~/verdict.err""")
    container(name, root, env, "\n        ".join(turns), timeout=900 * len(prompts) + 600,
              install=transition != "late-join")
    return {"transition": transition, "tools": tools, "branch": branch, "harness_compact": harness_compact,
            **report(name, home, window)}


def report(name: str, home: Path, window: int) -> dict:
    def lines(file: str) -> list[str]:
        path = home / file
        return path.read_text(errors="replace").splitlines() if path.exists() else []

    trace = [json.loads(line) for line in lines("trace.jsonl")]
    events = [json.loads(line) for line in lines("events.jsonl")]
    times = [tuple(map(int, line.split())) for line in lines("times")]
    requests = [r for r in trace if "body" in r]
    log = "\n".join(lines("relay.log"))
    verdict = json.loads((home / "verdict.json").read_text() or "{}") if (home / "verdict.json").exists() else {}
    return {
        "name": name,
        "window": window,
        "home": str(home),
        "verdict": verdict,
        "passed": bool(verdict) and all(value is True for value in verdict.values()),
        "turn_seconds": [end - start for _, start, end in times],
        "requests": len(requests),
        "prompt_tokens": sum(r.get("usage") or 0 for r in trace if "status" in r),
        # Relay's own records: compactions per turn, requests that found a stored one, divergences.
        "compactions": [sum(start <= e["time"] <= end for e in events) for _, start, end in times],
        "tokens_before": [e["tokens_before"] for e in events],
        "summary_requests": [e.get("summary_requests") for e in events],  # more than one: a history caught up in pieces
        "cache_hits": sum(1 for r in requests if r.get("cache", {}).get("covered") and not r["compacted"]),
        "diverged": sum(1 for r in requests if r.get("cache", {}).get("diverged") is not None),
        "failed_compactions": log.count("failed; forwarding"),
        "harness_compactions": log.count("/responses/compact ") + len(self_compactions(requests)),  # the harness's own
        "rejected": sum(r["status"] >= 400 for r in trace if "status" in r),
        "answers": [" ".join(lines(f"turn{n}.out"))[-300:] for n in range(1, len(TURNS) + 1)],
        # `--branch`: each branch's first request and the stored compaction it found
        "branches": {name: {"cache": first.get("cache"), "compacted": first["compacted"],
                            "answer": " ".join(lines(f"{name}.out"))[-300:]}
                     for name, prompt in (("fork", FORK), ("rewind", REWIND))
                     if (first := next((r for r in requests if prompt[:60] in json.dumps(r["body"])), None))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", choices=sorted(SPECS))
    parser.add_argument("--repo", type=Path, required=True, help="the more-itertools checkout with the planted bug")
    parser.add_argument("--window", type=int, default=64_000, help="context window Relay compacts at 90%% of")
    parser.add_argument("--baseline", action="store_true", help="never compact (a window no session reaches)")
    parser.add_argument("--transition", choices=sorted(TRANSITIONS), help="Relay joins late, drops out, or restarts")
    parser.add_argument("--tools", action="store_true", help="tickets from an MCP server, an image, web search")
    parser.add_argument("--branch", action="store_true", help="fork (and rewind) the session after turn 3")
    parser.add_argument("--harness-compact", type=int, help="the harness's own auto-compaction at about this many tokens")
    options = parser.parse_args()
    for name in options.names:
        result = run(name, options.repo.resolve(), 10**9 if options.baseline else options.window, options.transition,
                     options.tools, options.branch, options.harness_compact)
        print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
