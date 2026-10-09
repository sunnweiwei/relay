"""Check real harnesses through Relay, inside Docker, compacting every few thousand tokens.

    python tests/docker/check.py pi:gpt [--growth N]       one harness × model
    python tests/docker/check.py --matrix [names...]       several; prints a results table
    python tests/docker/check.py --via hook claude_code:gpt   the harness compacts through Relay's hook
    python tests/docker/check.py --strategy clm codex:gpt     the model edits its own context (CLM)
    python tests/docker/check.py --strategy acm codex:gpt     one of the CLM paper's baselines (selfcompact,
                                                              autocompact, acm, mem1, rlm, rlm_persistent)

Each run gets a throwaway HOME: the harness's own config is written there (the "user's
existing setup"), `relay install <harness>` edits it, Relay runs in the background, and
the harness is used as a user would. Host configs are never mounted. Codex and
`claude_code:login` use a copy of the host's login that cannot refresh (see `copy_login`);
everything else uses API keys from the file named by RELAY_TEST_KEYS. Requires the image
built from tests/docker/Dockerfile.

The task takes two turns of one session. Turn 1 reads file_1..4 one tool call at a time
and reports their codes; turn 2 (the harness's own continue/resume) brings a long note,
which starts a new turn above the compaction budget, then reads file_5..8 and reports all
eight codes without reading the first four again. Relay compacts after every GROWTH tokens
added to the context, so a run compacts mid-turn and at the start of a turn, several times.

Checks, for every compaction: the summary contains every code read so far (information
survives repeated compaction); the new context has Codex's layout (system, recent user
messages verbatim, the initial context above the last user message and the summary last
mid-turn, or re-injected after the summary at a turn start) and keeps all of the initial
context; and every later request of the conversation that Relay forwarded carries exactly
the latest summary and none of the summarized tool output. Also: no file was read twice, both answers list the right codes in
order, no compaction failed, and the upstream accepted every model call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from relay.core.ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind  # noqa: E402
from relay.harnesses import detect  # noqa: E402
from relay.prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX  # noqa: E402
from relay.protocols import codec_for  # noqa: E402
from relay.strategies.selfcompact import RUBRIC, SUMMARIZER  # noqa: E402

IMAGE = "relay-harness-test"
CODES = ["amber", "birch", "cobalt", "dune", "ember", "fjord", "garnet", "harbor"]
SEPARATOR = r"[\s,`'\"]+(?:and\s+[`'\"]?)?"  # between codes in an answer (some models quote them, or end with "and")
RULES = ("strictly one after another: request one file, wait until you have seen its content, then "
         "request the next. Never request two files at once, and read each file in full "
         "(for example `cat file_1.txt`; no head, tail or grep). Each file ends with a line starting "
         "with 'CODE:'.")
PROMPT = (f"Read the files file_1.txt to file_4.txt in the current directory, {RULES} After reading "
          f"them, reply with their 4 codes in order, comma-separated, and nothing else.")
PROMPT_ALL = (f"Read the files file_1.txt to file_8.txt in the current directory, {RULES} After reading "
              f"them, reply with their 8 codes in order, comma-separated, and nothing else.")  # one-turn harnesses
NOTES = "\n".join(f"note {i}: the slow grey cat sleeps beside the warm stove all afternoon" for i in range(330))
DELEGATE = ("If you can delegate work to a sub-agent, have one read file_5.txt and report its CODE instead "
            "of reading it yourself. ")
FINISH = ("Do not read file_1.txt to file_4.txt again. Then reply with the codes of all 8 files (file_1.txt "
          "to file_8.txt) in order, comma-separated, and on a last line the project codename as your project "
          "instructions state it now.")
PROMPT2 = f"Background notes, not needed for the task:\n{NOTES}\n\nNow read file_5.txt to file_8.txt, {RULES} {DELEGATE}{FINISH}"
# `--clm-edit`: the model edits its context as told (CLM's mechanism, whatever it would choose itself),
# then answers from what it sees.
EDIT = ("Read file_1.txt, file_2.txt and file_3.txt in the current directory, one file per tool call, each in full. "
        "Then, in one edit of your context file (see \"Managing your context\"): delete the tool call that read "
        "file_2.txt and its result; replace the text of the block holding file_1.txt's output with exactly "
        "`MASKED file_1: CODE amber`; and add a block with `id=new-marker` and role `notes` whose text is exactly "
        "`MARKER 7319`. Change nothing else.")
EDIT_ASK = ("reply with exactly three lines: the text of your marker block; what your context now shows as the output "
            "of reading file_1.txt; and the CODE of file_2.txt if your context still shows its output, otherwise NONE.")
EDIT_PROMPTS = (f"{EDIT} Then reply with the CODE of file_3.txt.",
                f"Do not read any file and do not edit your context. Just {EDIT_ASK}")
EDIT_ONE = f"{EDIT} After the edit, {EDIT_ASK}"  # one-turn harnesses
# `--strategy folding`: the agent reads two files in a branch and returns with their codes.
FOLD = ("Read file_1.txt in the current directory, in full. Then open a branch (see \"Branches\") to read "
        "file_2.txt and file_3.txt, one file per tool call, each in full, and return with their CODEs. Back in "
        "the main conversation, read file_4.txt in full.")
FOLD_ASK = ("reply with exactly two lines: the CODEs of file_1.txt to file_4.txt in order, comma-separated; then YES if "
            "your context still shows the full text of file_2.txt, otherwise NO.")
FOLD_PROMPTS = (f"{FOLD} Then reply with the CODEs of file_1.txt to file_4.txt in order, comma-separated.",
                f"Do not read any file. Just {FOLD_ASK}")
FOLD_ONE = f"{FOLD} Then {FOLD_ASK}"
# `--strategy prolong`: each file also names an animal, which turn 1 does not ask for (so summaries
# drop it) and turn 2 does, without the files: the agent finds it in PRO-LONG's log.
ANIMALS = ["heron", "otter", "lynx", "ibex", "marten", "wombat", "bison", "gecko"]
RECALL = (f"Background notes, not needed for the task:\n{NOTES}\n\nBesides its CODE, each file you read earlier "
          "had a line `file N animal: <animal>`. Without opening file_1.txt to file_4.txt again, find the animal on "
          "that line of file_2.txt, and reply with that animal only.")
# The CLM paper's baselines. `--strategy selfcompact` and `--strategy mem1` run the default task
# (turn 2 without the sub-agent): Self-Compact under a window small enough that its rubric is asked,
# MEM1 with only the newest internal state to carry the codes to the end.
BASELINE_WINDOW = 40_000
# `--strategy autocompact`: the agent compacts after a reading phase; its last turn before stays.
AUTO = ("Read file_1.txt, file_2.txt and file_3.txt in the current directory, one file per tool call, each in full. "
        "That ends the reading phase: now compact your context (see \"Compaction\"). Then read file_4.txt in full.")
AUTO_PROMPTS = (f"{AUTO} Then reply with the CODEs of file_1.txt to file_4.txt in order, comma-separated.",
                f"Do not read any file. Just {FOLD_ASK}")
AUTO_ONE = f"{AUTO} Then reply with the CODEs of file_1.txt to file_4.txt in order, comma-separated."
# `--strategy acm`: the agent compresses what it read, then recalls an animal from the saved originals.
ACM_TASK = ("Read file_1.txt, file_2.txt and file_3.txt in the current directory, one file per tool call, each in "
            "full. Then compress your context with manage_context (see \"Context memory\"). Then read file_4.txt in full.")
ACM_ASK = ("Each file you read had a line `file N animal: <animal>`. Without opening any file, use query_memory to "
           "find the animal on that line of file_2.txt, then reply with that animal only.")
ACM_PROMPTS = (f"{ACM_TASK} Then reply with the CODEs of file_1.txt to file_4.txt in order, comma-separated.", ACM_ASK)
ACM_ONE = f"{ACM_TASK} Then: {ACM_ASK}"
# `--strategy rlm|rlm_persistent`: an RLM works out every step; turn 2 asks for the codes from the conversation.
RLM_TASK = ("Read file_1.txt, file_2.txt and file_3.txt in the current directory, one file per tool call, each in full. "
            "Then reply with their CODEs in order, comma-separated.")
RLM_PROMPTS = (RLM_TASK, "Do not read any file. Reply with the CODEs of file_1.txt to file_3.txt in order, comma-separated.")
# `--native-compact`: the agent itself reads on, so the conversation after the harness's summary compacts.
DIRECT2 = f"Background notes, not needed for the task:\n{NOTES}\n\nNow read file_5.txt to file_8.txt, {RULES} {FINISH}"
# `--subagent`: turn 2 hands three files to one sub-agent, so a sub-agent compacts too.
SUBAGENT2 = (f"Background notes, not needed for the task:\n{NOTES}\n\nNow have one sub-agent read file_5.txt, "
             f"file_6.txt and file_7.txt for you, {RULES} It must report their three CODEs. Then read file_8.txt "
             f"yourself. Do not read file_1.txt to file_4.txt again. Reply with the codes of all 8 files "
             f"(file_1.txt to file_8.txt) in order, comma-separated, and on a last line the project codename as "
             f"your project instructions state it now.")

# `--harness-compaction`: the user's configuration makes the harness compact itself at about 20k
# tokens; `relay install` must turn that off (Relay itself never compacts in this scenario).
JSON_MERGE = """python3 -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
c = json.loads(p.read_text()) if p.exists() else {{}}; n = c
for k in sys.argv[2].split(".")[:-1]: n = n.setdefault(k, {{}})
n[sys.argv[2].split(".")[-1]] = json.loads(sys.argv[3]); p.write_text(json.dumps(c))' {} {} '{}'"""
EARLY = {
    "codex": "printf 'model_context_window = 16000\\n' >> ~/.codex/config.toml",
    "claude_code": "export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=10",
    # (the forced compaction at 92% of 47k; at 100% it is out of this task's reach)
    "workbuddy": "python3 -c 'import json, pathlib; "
                 "p = pathlib.Path.home() / \".workbuddy/models.json\"; c = json.loads(p.read_text()); "
                 "c[\"models\"][0][\"maxInputTokens\"] = 47000; p.write_text(json.dumps(c))'",
    "gemini_cli": JSON_MERGE.format("~/.gemini/settings.json", "model.compressionThreshold", "0.02"),
    "pi": JSON_MERGE.format("~/.pi/agent/settings.json", "compaction", '{"reserveTokens": 252000, "keepRecentTokens": 4000}'),
    "opencode": JSON_MERGE.format("~/.config/opencode/opencode.json", "provider.openai.models.gpt-6-luna",
                                  '{"limit": {"context": 28000, "input": 24000, "output": 4000}}'),
    "kilo": JSON_MERGE.format("~/.config/kilo/config.json", "provider.openai.models.gpt-6-luna",
                              '{"limit": {"context": 28000, "input": 24000, "output": 4000}}'),
    "crush": "sed -i 's/\"context_window\": 200000/\"context_window\": 30000/' ~/.config/crush/crush.json",
    "goose": "echo 'GOOSE_AUTO_COMPACT_THRESHOLD: 0.01' >> ~/.config/goose/config.yaml",
    "openclaw": JSON_MERGE.format("~/.openclaw/openclaw.json", "models.providers.openai.models",
                                  '[{"id": "gpt-6-luna", "name": "gpt-6-luna", "contextWindow": 32000, "maxTokens": 8000}]'),
    "kimi_code": "sed -i 's/^max_context_size = 200000/max_context_size = 30000/' ~/.kimi-code/config.toml",
    "hermes": "sed -i 's/^model:/model:\\n  context_length: 30000/' ~/.hermes/config.yaml",
    "nanobot": JSON_MERGE.format("~/.nanobot/config.json", "agents.defaults.contextWindowTokens", "30000"),
    "deepseek_harness": "printf -- '- id: compaction-basic\\n  config:\\n    thresholdRatio: 0.06\\n    retainRatio: 0.02\\n' "
                        ">> ~/.dsh/cordis.patch.yml",
}
# How harnesses ask for their own summary (each words it differently; Codex and Claude Code are
# recognized by their profiles): WorkBuddy, pi, OpenCode/Kilo, Crush, Kimi Code, nanobot, Gemini
# CLI, DeepSeek Harness, Hermes Agent, Goose.
COMPACT_ASK = re.compile(r"structured summary|<conversation>|summary of our conversation|run out of context|"
                         r"compact replacement checkpoint|state_snapshot|compaction engine|summarization agent|"
                         r"summarize the conversation|\\[User\\]:", re.I)

# `--native-compact`: the harness compacts its own history between the turns, as a user would,
# with this command sent through its own continue/resume.
NATIVE_COMPACT = {"claude_code": "/compact", "codex": "/compact", "gemini_cli": "/compress", "pi": "/compact",
                  "opencode": "/compact", "kilo": "/compact", "crush": "/compact", "openclaw": "/compact",
                  "goose": "/compact", "kimi_code": "/compact", "hermes": "/compress", "nanobot": "/compact",
                  "deepseek_harness": "/compact", "workbuddy": "/compact"}

# `--probe`: what a harness keeps as state. Instruction files hold a codename that changes
# between the turns; turn 2 asks for a sub-agent. Relay only records (no compaction).
STATE_FILES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "CRUSH.md", "CODEBUDDY.md", ".goosehints")
PROBE = (
    "This project has instruction files. What project codename do they give? Use your todo or "
    "task-tracking tool, if you have one, to plan two steps: read file_1.txt, then read file_2.txt. "
    "Then read file_1.txt. Reply with the codename and the CODE at the end of file_1.txt.",
    "If you can delegate work to a sub-agent, have one read file_2.txt and report its CODE; otherwise "
    "read it yourself. Reply with that CODE and the project codename as your project instructions "
    "state it now.",
)

GPT, GEMINI, CLAUDE = "gpt-6-luna", "gemini-3.8-flash", "claude-sonnet-5-5"
GEMINI_OPENAI = "https://generativelanguage.googleapis.com/v1beta/openai"
GROWTH = 5_000  # each file adds about 2k tokens
# `--via hook`: the harness asks Relay before every request, and Relay's strategy compacts at this
# size. Relay is off the model path, so a second Relay that never compacts records it, in front of
# whatever endpoint the user's configuration names.
HOOK_THRESHOLD = 30_000
# `--strategy clm`: the model edits its own context through a mirror file; this budget is below what
# the task needs unedited (its reminders and readout are what make the model edit).
CLM_BUDGET = 26_000
RECORDER = f"""UPSTREAM=$(python3 -c 'import json, pathlib; p = pathlib.Path("~/.claude/settings.json").expanduser()
print((json.loads(p.read_text()) if p.exists() else {{}}).get("env", {{}}).get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))')
(cd /relay && RELAY_PORT=8788 RELAY_STRATEGY=compaction RELAY_COMPACT_GROWTH=1000000000 RELAY_HARNESS=claude_code RELAY_ANTHROPIC_BASE_URL=$UPSTREAM \
  python3 -m relay.cli serve > ~/recorder.log 2>&1 &)
until python3 -c 'import socket; socket.create_connection(("127.0.0.1", 8788))' 2>/dev/null; do sleep 0.2; done
{JSON_MERGE.format("~/.claude/settings.json", "env.ANTHROPIC_BASE_URL", '"http://127.0.0.1:8788"')}"""


@dataclass
class Spec:
    command: str  # run in /project; $PROMPT holds the task
    resume: str  # the same session's next turn, as the harness continues one ("": no resume, one turn)
    keys: dict[str, str] = field(default_factory=dict)  # container env var -> key in the keys file
    setup: str = ""  # shell run before `relay install`: the user's existing configuration
    login: bool = False  # copy the host's login instead of using API keys


def json_file(path: str, value: object) -> str:
    return f"mkdir -p $(dirname {path}) && cat > {path} <<'JSON'\n{json.dumps(value, indent=1)}\nJSON"


def crush_config(provider: str, kind: str, url: str, model: str, key: str) -> str:
    return json_file("~/.config/crush/crush.json", {
        "providers": {provider: {"type": kind, "base_url": url, "api_key": f"${key}",
                                 "models": [{"id": model, "name": model, "context_window": 200000,
                                             "default_max_tokens": 8000}]}},
        "models": {"large": {"provider": provider, "model": model},
                   "small": {"provider": provider, "model": model}},
        "permissions": {"allowed_tools": ["view", "ls", "bash", "glob", "grep"]},
    })


def kimi_config(kind: str, model: str, key: str) -> str:
    return f"""mkdir -p ~/.kimi-code && cat > ~/.kimi-code/config.toml <<'TOML'
default_model = "test"
[providers.test]
type = "{kind}"
api_key_env = "{key}"
[models.test]
provider = "test"
model = "{model}"
max_context_size = 200000
TOML"""


def goose_config(provider: str, model: str) -> str:
    return f"""mkdir -p ~/.config/goose && cat > ~/.config/goose/config.yaml <<'YAML'
GOOSE_PROVIDER: {provider}
GOOSE_MODEL: {model}
extensions:
  developer:
    enabled: true
    name: developer
    type: builtin
    bundled: true
    timeout: 300
YAML"""


def hermes_config(provider: str, model: str, base_url: str) -> str:  # as `hermes model` writes it
    return f"""mkdir -p ~/.hermes && cat > ~/.hermes/config.yaml <<'YAML'
model:
  default: {model}
  provider: {provider}
  base_url: {base_url}
YAML"""


def workbuddy_models(model: str, url: str, key: str, **extra: object) -> str:
    return json_file("~/.workbuddy/models.json", {"models": [{
        "id": model, "name": model, "vendor": "OpenAI", "apiKey": f"${{{key}}}", "url": url,
        "supportsToolCall": True, "supportsImages": False, **extra,
    }], "availableModels": [model]})


# WorkBuddy is a desktop app; its engine also ships as the CodeBuddy CLI, run here on ~/.workbuddy.
WORKBUDDY = 'env CODEBUDDY_CONFIG_DIR=$HOME/.workbuddy codebuddy {}-p -y --disallowedTools EnterPlanMode --model {} "$PROMPT"'  # headless: a plan cannot be approved
WORKBUDDY_GPT = f"{GPT} --effort medium"


def mini_config(model: str) -> str:
    return f"""mkdir -p ~/.config/mini-swe-agent && cat > ~/.config/mini-swe-agent/.env <<'ENV'
MSWEA_CONFIGURED=true
MSWEA_MODEL_NAME={model}
MSWEA_COST_TRACKING=ignore_errors
ENV"""


# One turn (mini cannot continue a session); mini only finishes through its submission command.
MINI = ('mini -y -l 0 -c mini.yaml -c agent.confirm_exit=false {} -t "$PROMPT Submit them by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo <codes>" '
        '; python3 -c "import json; print(json.load(open(\'/home/agent/.config/mini-swe-agent/last_mini_run.traj.json\'))'
        '[\'info\'][\'submission\'])"')


def nanobot_config(provider: str, model: str, key: str) -> str:
    # nanobot only knows to drop `temperature` for gpt-6 when a reasoning effort is set
    defaults = {"model": model, "provider": provider, "workspace": "/project", "reasoningEffort": "medium"}
    return json_file("~/.nanobot/config.json", {
        "agents": {"defaults": defaults},
        "providers": {provider: {"apiKey": f"${{{key}}}"}},  # resolved by nanobot at load time
    })


def dsh_patch(provider: str, model: str, key: str) -> str:
    return f"""mkdir -p ~/.dsh && cat > ~/.dsh/cordis.patch.yml <<'YAML'
- id: agent-default-model
  config:
    provider: {provider}
    model: {model}
    reasoningEffort: medium
- id: llm-pi-ai
  config:
    providers:
      {provider}:
        apiKeyEnv: {key}
YAML"""


LITELLM = f"""cat > ~/litellm.yaml <<'YAML'
model_list:
  - model_name: {GPT}
    litellm_params: {{model: openai/{GPT}, api_key: os.environ/OPENAI_API_KEY}}
litellm_settings: {{drop_params: true}}
YAML
litellm --config ~/litellm.yaml --port 4000 > ~/litellm.log 2>&1 &
until python3 -c 'import socket; socket.create_connection(("127.0.0.1", 4000))' 2>/dev/null; do sleep 0.5; done"""
# The user already routes the harness through LiteLLM, which translates its protocol to
# OpenAI; `relay install` then wraps that gateway.


CLAUDE_ARGS = '-p "$PROMPT" --permission-mode bypassPermissions --model'
REMOTE = "/relay/tests/docker/codex_remote.py"
ONBOARDED = """echo '{"hasCompletedOnboarding": true}' > ~/.claude.json"""
GEMINI_AUTH = json_file("~/.gemini/settings.json", {"security": {"auth": {"selectedType": "gemini-api-key"}}})
OPENCLAW = json_file("~/.openclaw/openclaw.json", {"agents": {"defaults": {"workspace": "/project"}}})
DSH_SESSION = '"$(basename "$(ls -td ~/.dsh/sessions/*/session-* | head -1)")"'
OPENAI_KEY, GEMINI_KEY = {"OPENAI_API_KEY": "OPENAI_API_KEY"}, {"GEMINI_API_KEY": "GEMINI_API_KEY"}


def spec(command: str, resume: str, keys: dict[str, str] | None = None, setup: str = "", login: bool = False) -> Spec:
    return Spec(command, resume, keys or {}, setup, login)


SPECS = {
    "codex:gpt": spec(
        'codex exec --skip-git-repo-check "$PROMPT"', 'codex exec --skip-git-repo-check resume --last "$PROMPT"',
        setup=f"""mkdir -p ~/.codex && printf 'model = "{GPT}"\\nmodel_reasoning_effort = "medium"\\nsandbox_mode = "danger-full-access"\\n' > ~/.codex/config.toml""",
        login=True,
    ),
    # Remote Control: the user drives Codex from the ChatGPT app. codex_remote.py stands in for the
    # service and the app, so nothing is enrolled with the account; Relay is on the model path only.
    "codex:remote": spec(
        f'python3 {REMOTE} say "$PROMPT"', f'python3 {REMOTE} say "$PROMPT"',
        setup=f"""mkdir -p ~/.codex && printf 'model = "{GPT}"\\nmodel_reasoning_effort = "medium"\\nsandbox_mode = "danger-full-access"\\napproval_policy = "never"\\nchatgpt_base_url = "https://127.0.0.1:8900/backend-api/"\\n' > ~/.codex/config.toml
        python3 {REMOTE} serve > ~/remote.log 2>&1 &""",
        login=True,
    ),
    # API key with reasoning items in the history: `exec resume` rewrites them (PR #2's finding).
    "codex:gpt-api": spec(
        'codex exec --skip-git-repo-check "$PROMPT"', 'codex exec --skip-git-repo-check resume --last "$PROMPT"',
        OPENAI_KEY,
        f"""mkdir -p ~/.codex && printf 'model = "{GPT}"\nmodel_reasoning_effort = "medium"\nsandbox_mode = "danger-full-access"\n' > ~/.codex/config.toml
        printenv OPENAI_API_KEY | codex login --with-api-key > /dev/null""",
    ),
    "claude_code:claude": spec(f"claude {CLAUDE_ARGS} {CLAUDE}", f"claude -c {CLAUDE_ARGS} {CLAUDE}",
                               {"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"}, ONBOARDED),
    "claude_code:login": spec(f"claude {CLAUDE_ARGS} {CLAUDE}", f"claude -c {CLAUDE_ARGS} {CLAUDE}",
                              setup=ONBOARDED, login=True),
    "claude_code:gpt": spec(
        f"claude {CLAUDE_ARGS} {GPT}", f"claude -c {CLAUDE_ARGS} {GPT}", OPENAI_KEY,
        f"""{LITELLM}
        export ANTHROPIC_API_KEY=sk-litellm ANTHROPIC_SMALL_FAST_MODEL={GPT} ANTHROPIC_DEFAULT_HAIKU_MODEL={GPT}
        {ONBOARDED}
        {json_file("~/.claude/settings.json", {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000"}})}""",
    ),
    "gemini_cli:gpt": spec(
        f'gemini -p "$PROMPT" --yolo --skip-trust -m {GPT}', f'gemini -r latest -p "$PROMPT" --yolo --skip-trust -m {GPT}',
        OPENAI_KEY,
        f"""{LITELLM}
        export GEMINI_API_KEY=sk-litellm
        mkdir -p ~/.gemini && echo GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:4000 > ~/.gemini/.env
        {GEMINI_AUTH}""",
    ),
    "gemini_cli:gemini": spec(f'gemini -p "$PROMPT" --yolo --skip-trust -m {GEMINI}',
                              f'gemini -r latest -p "$PROMPT" --yolo --skip-trust -m {GEMINI}', GEMINI_KEY, GEMINI_AUTH),
    "pi:gpt": spec(f'pi -p --model openai/{GPT} --thinking medium "$PROMPT"',
                   f'pi -c -p --model openai/{GPT} --thinking medium "$PROMPT"', OPENAI_KEY),
    "pi:gemini": spec(f'pi -p --model google/{GEMINI} "$PROMPT"', f'pi -c -p --model google/{GEMINI} "$PROMPT"', GEMINI_KEY),
    "opencode:gpt": spec(f'opencode run --model openai/{GPT} "$PROMPT"', f'opencode run -c --model openai/{GPT} "$PROMPT"',
                         OPENAI_KEY),
    "opencode:gemini": spec(f'opencode run --model google/{GEMINI} "$PROMPT"',
                            f'opencode run -c --model google/{GEMINI} "$PROMPT"',
                            {"GOOGLE_GENERATIVE_AI_API_KEY": "GEMINI_API_KEY"}),
    "kilo:gpt": spec('kilo run "$PROMPT"', 'kilo run -c "$PROMPT"', OPENAI_KEY,
                     json_file("~/.config/kilo/config.json", {"model": f"openai/{GPT}"})),
    "kilo:gemini": spec('kilo run "$PROMPT"', 'kilo run -c "$PROMPT"', {"GOOGLE_GENERATIVE_AI_API_KEY": "GEMINI_API_KEY"},
                        json_file("~/.config/kilo/config.json", {"model": f"google/{GEMINI}"})),
    "crush:gpt": spec('crush run --quiet "$PROMPT"', 'crush run -C --quiet "$PROMPT"', OPENAI_KEY,
                      crush_config("openai", "openai", "https://api.openai.com/v1", GPT, "OPENAI_API_KEY")),
    "crush:gemini": spec('crush run --quiet "$PROMPT"', 'crush run -C --quiet "$PROMPT"', GEMINI_KEY,
                         crush_config("gemini", "gemini", "https://generativelanguage.googleapis.com", GEMINI,
                                      "GEMINI_API_KEY")),
    "openclaw:gpt": spec(*[f'openclaw agent --local --session-id relay-check --model openai/{GPT} -m "$PROMPT"'] * 2,
                         OPENAI_KEY, OPENCLAW),
    "openclaw:gemini": spec(*[f'openclaw agent --local --session-id relay-check --model google/{GEMINI} -m "$PROMPT"'] * 2,
                            GEMINI_KEY, OPENCLAW),
    "goose:gpt": spec('goose run -n relay -t "$PROMPT"', 'goose run -n relay --resume -t "$PROMPT"', OPENAI_KEY,
                      goose_config("openai", GPT)),
    "goose:gemini": spec('goose run -n relay -t "$PROMPT"', 'goose run -n relay --resume -t "$PROMPT"',
                         {"GOOGLE_API_KEY": "GEMINI_API_KEY"}, goose_config("google", GEMINI)),
    "kimi_code:gpt": spec('kimi -p "$PROMPT"', 'kimi -c -p "$PROMPT"', OPENAI_KEY,
                          kimi_config("openai_responses", GPT, "OPENAI_API_KEY")),
    # Kimi Code asks gemini-3.8-flash for thinking level MINIMAL, which that model rejects.
    "kimi_code:gemini": spec('kimi -p "$PROMPT"', 'kimi -c -p "$PROMPT"', GEMINI_KEY,
                             kimi_config("google-genai", "gemini-2.5-flash-lite", "GEMINI_API_KEY")),
    "hermes:gpt": spec('hermes chat -q "$PROMPT" --yolo', 'hermes chat --continue -q "$PROMPT" --yolo', OPENAI_KEY,
                       hermes_config("openai-api", GPT, "https://api.openai.com/v1")),
    "hermes:gemini": spec('hermes chat -q "$PROMPT" --yolo', 'hermes chat --continue -q "$PROMPT" --yolo', GEMINI_KEY,
                          hermes_config("gemini", GEMINI, "https://generativelanguage.googleapis.com/v1beta")),
    "nanobot:gpt": spec(*['nanobot agent -s relay --no-markdown -m "$PROMPT"'] * 2, OPENAI_KEY,
                        nanobot_config("openai", GPT, "OPENAI_API_KEY")),
    "nanobot:gemini": spec(*['nanobot agent -s relay --no-markdown -m "$PROMPT"'] * 2, GEMINI_KEY,
                           nanobot_config("gemini", GEMINI, "GEMINI_API_KEY")),
    "deepseek_harness:gpt": spec('dsh --profile headless "$PROMPT"', f'dsh --profile headless --session-id {DSH_SESSION} "$PROMPT"',
                                 OPENAI_KEY, dsh_patch("openai", GPT, "OPENAI_API_KEY")),
    "deepseek_harness:gemini": spec('dsh --profile headless "$PROMPT"',
                                    f'dsh --profile headless --session-id {DSH_SESSION} "$PROMPT"',
                                    GEMINI_KEY, dsh_patch("google", GEMINI, "GEMINI_API_KEY")),
    # gpt-6 takes tools on chat/completions only without reasoning, and WorkBuddy speaks nothing else:
    # LiteLLM turns its requests into the Responses API, where the model reasons as in Codex.
    "workbuddy:gpt": spec(WORKBUDDY.format("", WORKBUDDY_GPT), WORKBUDDY.format("-c ", WORKBUDDY_GPT), OPENAI_KEY,
                          f"""{LITELLM}
                          {workbuddy_models(GPT, "http://127.0.0.1:4000/v1/chat/completions", "OPENAI_API_KEY",
                                            supportsReasoning=True)}"""),
    "workbuddy:gemini": spec(WORKBUDDY.format("", GEMINI), WORKBUDDY.format("-c ", GEMINI), GEMINI_KEY,
                             workbuddy_models(GEMINI, f"{GEMINI_OPENAI}/chat/completions", "GEMINI_API_KEY")),
    "mini_swe:gpt": spec(MINI.format("-c model.model_class=litellm_response"), "", OPENAI_KEY,
                         mini_config(f"openai/{GPT}")),
    "mini_swe:gemini": spec(MINI.format(""), "", GEMINI_KEY, mini_config(f"gemini/{GEMINI}")),
}


def run(name: str, growth: int | None = None, retried: bool = False, probe: bool = False,
        subagent: bool = False, window: int | None = None, native: bool = False, selfcompact: bool = False,
        via: str = "proxy", lines: int = 120, strategy: str = "compaction", steering: str | None = None,
        clm_edit: bool = False) -> dict:
    spec, harness = SPECS[name], name.split(":")[0]
    root = Path(tempfile.mkdtemp(prefix=f"relay-{name.replace(':', '-')}-", dir=os.getenv("TMPDIR")))
    home, project = root / "home", root / "project"
    home.mkdir()
    project.mkdir()
    for n, code in enumerate(CODES, start=1):
        text = [f"file {n} line {i}: the quick brown fox jumps over the lazy dog" for i in range(1, lines + 1)]
        if strategy in ("prolong", "acm"):
            text.insert(lines // 2, f"file {n} animal: {ANIMALS[n - 1]}")
        (project / f"file_{n}.txt").write_text("\n".join([*text, f"CODE: {code}"]) + "\n")
    for file in STATE_FILES:  # the harness's own instructions, changed between the turns
        (project / file).write_text("# Project instructions\n\nThe project codename is ALPHA.\n")

    # Compact after every GROWTH new tokens, whatever the harness's fixed prompt overhead.
    # No min_gain guard (Codex has none): turn 2 must start with a compaction even when turn 1
    # ended right after one.
    first, second = PROBE if probe else (PROMPT if spec.resume else PROMPT_ALL, SUBAGENT2 if subagent else DIRECT2 if native else PROMPT2)
    if clm_edit:
        first, second = EDIT_PROMPTS if spec.resume else (EDIT_ONE, "")
    if strategy == "folding":
        first, second = FOLD_PROMPTS if spec.resume else (FOLD_ONE, "")
    if strategy == "prolong":
        second = RECALL
    if strategy in ("selfcompact", "mem1") and spec.resume:
        second = DIRECT2
    if strategy == "autocompact":
        first, second = AUTO_PROMPTS if spec.resume else (AUTO_ONE, "")
    if strategy == "acm":
        first, second = ACM_PROMPTS if spec.resume else (ACM_ONE, "")
    if strategy.startswith("rlm"):
        first, second = RLM_PROMPTS if spec.resume else (RLM_TASK, "")
    # Growth (every GROWTH new tokens) by default; `window` instead uses Relay's own trigger,
    # 90% of a context window of that many tokens (and 95% always).
    trigger = {"RELAY_CONTEXT_WINDOW": str(window)} if window else {
        "RELAY_COMPACT_GROWTH": str(10**9 if probe or selfcompact else growth or GROWTH)}
    if via == "hook":
        trigger = {"RELAY_COMPACT_THRESHOLD": str(HOOK_THRESHOLD)}
    if strategy == "folding":
        trigger = {"RELAY_STRATEGY": "folding"}
    if strategy == "prolong":  # compaction inside, as usual, and the log where the harness's tools reach it
        trigger |= {"RELAY_STRATEGY": "prolong", "RELAY_PROLONG_LOG": "/project/.prolong/log.jsonl"}
    if strategy in ("selfcompact", "autocompact", "acm", "mem1", "rlm", "rlm_persistent"):  # (AutoCompact's fallback: 90% of the window)
        trigger = {"RELAY_STRATEGY": strategy, "RELAY_ACM_DIR": "/project/.acm"}
        trigger |= {"RELAY_CONTEXT_WINDOW": str(window or BASELINE_WINDOW)} if strategy == "selfcompact" else {}
    if strategy == "clm":  # the files in HOME, where the run can be inspected
        trigger = {"RELAY_STRATEGY": "clm", "RELAY_CLM_BUDGET": str(CLM_BUDGET), "RELAY_CLM_DIR": "/project/.live_ctx"}
        if clm_edit:  # no budget pressure: the edit is the task
            trigger = {k: v for k, v in trigger.items() if k != "RELAY_CLM_BUDGET"} | {"RELAY_CLM_NUDGES": "off"}
        if steering:
            trigger |= {"RELAY_CLM_STEERING": f"/relay/tests/docker/steering/{steering}.md",
                        "RELAY_CLM_NUDGES": "on" if POLICIES[steering].nudges else "off"}
    env = {"PROMPT": first, "PROMPT2": second, **trigger, "RELAY_COMPACT_MIN_GAIN": "0"}
    turn = 1800 if strategy.startswith("rlm") else 600  # an RLM works out every step before the model takes it
    container(name, root, env, f"""
        cd /project && timeout {turn} {spec.command} > ~/harness.out 2> ~/harness.err < /dev/null
        {f'export PROMPT="{NATIVE_COMPACT[harness]}" && cd /project && timeout 300 {spec.resume} > ~/compact.out 2>&1 < /dev/null' if native else ""}
        export PROMPT="$PROMPT2"
        sed -i s/ALPHA/BETA/ {" ".join(STATE_FILES)}
        {f"cd /project && timeout {turn} {spec.resume} > ~/harness2.out 2> ~/harness2.err < /dev/null" if spec.resume else ""}
    """, timeout=2 * turn + 300, setup="\n".join([EARLY[harness] if selfcompact else "", RECORDER if via == "hook" else ""]),
              via=via)
    if probe:
        return {"name": name, "home": str(home)}
    if via == "hook":
        return evaluate_hook(name, home)
    if strategy == "clm":
        return evaluate_clm_edit(name, home) if clm_edit else evaluate_clm(name, home, steering)
    if strategy == "folding":
        return evaluate_folding(name, home)
    if strategy == "prolong":
        return evaluate_prolong(name, home)
    if strategy in BASELINES:
        return BASELINES[strategy](name, home)
    # Relay's own trigger compacts less often; after the harness compacted itself, turn 2 starts small.
    result = evaluate(name, home, need=(0, 0) if window else (2, 0) if native else None, selfcompact=selfcompact)
    if not result["passed"] and result["transient"] and result["requests"] < 3 and not retried:
        time.sleep(60)  # rate-limited before the task got going: try once more
        return run(name, growth, retried=True, subagent=subagent, window=window, native=native, selfcompact=selfcompact,
                   via=via, lines=lines, strategy=strategy, steering=steering, clm_edit=clm_edit)
    return result


def container(name: str, root: Path, env: dict[str, str], turns: str, timeout: int = 1500, install: bool = True,
              setup: str = "", via: str = "proxy") -> None:
    """Run `turns` (shell) on a user's machine, in the image: HOME is root/home and /project is
    root/project; the harness is configured as the user had it (`Spec.setup`), Relay serves (its
    process id in ~/relay.pid) and, unless `install` is false, `relay install` points the
    harness at it. Relay's events and trace land in HOME."""

    spec, harness = SPECS[name], name.split(":")[0]
    home, project = root / "home", root / "project"
    env = {**env, "RELAY_EVENT_LOG": "/home/agent/events.jsonl", "RELAY_TRACE": "/home/agent/trace.jsonl"}
    passwd = root / "passwd"  # some harnesses look the user up (os.userInfo)
    passwd.write_text(f"root:x:0:0::/root:/bin/bash\nagent:x:{os.getuid()}:{os.getgid()}::/home/agent:/bin/bash\n")
    mounts = [f"{REPO}:/relay:ro", f"{home}:/home/agent", f"{project}:/project", f"{passwd}:/etc/passwd:ro"]
    if spec.keys:
        keys = dict(line.split("=", 1) for line in Path(os.environ["RELAY_TEST_KEYS"]).read_text().split() if "=" in line)
        # Spread Gemini quota over GEMINI_API_KEY, GEMINI_API_KEY_2, ... when several are given.
        gemini = sorted(k for k in keys if k.startswith("GEMINI_API_KEY"))
        alias = {"GEMINI_API_KEY": gemini[sum(map(ord, name)) % len(gemini)]} if gemini else {}
        env.update({var: keys[alias.get(key, key)] for var, key in spec.keys.items()})
    if spec.login and harness == "codex":
        copy_login("~/.codex/auth.json", home / ".codex/auth.json", "tokens", "refresh_token")
    if spec.login and harness == "claude_code":
        copy_login("~/.claude/.credentials.json", home / ".claude/.credentials.json", "claudeAiOauth", "refreshToken")
    if harness == "codex":  # the standalone package: codex plus its helper binaries
        mounts.append(f"{Path(shutil.which('codex')).resolve().parents[1]}:/opt/codex:ro")
        env["PATH"] = "/opt/codex/bin:/opt/codex/codex-path:/usr/local/bin:/usr/bin:/bin"
    if harness == "claude_code":
        mounts.append(f"{Path(shutil.which('claude')).resolve()}:/usr/local/bin/claude:ro")

    script = f"""
        cd /relay
        python3 -m relay.cli serve > ~/relay.log 2>&1 & echo $! > ~/relay.pid
        until python3 -c 'import socket; socket.create_connection(("127.0.0.1", 8787))' 2>/dev/null; do sleep 0.2; done
        cd ~ && {spec.setup}
        {setup}
        {f"cd /relay && python3 -m relay.cli install {harness} --via {via} >> ~/relay.log 2>&1" if install else ""}
        {turns}
    """
    docker = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-w", "/project",
              "-e", "HOME=/home/agent", "-e", "PYTHONPATH=/relay"]
    for mount in mounts:
        docker += ["-v", mount]
    for var, value in env.items():
        docker += ["-e", f"{var}={value}"]
    subprocess.run([*docker, IMAGE, "bash", "-c", script], check=False, timeout=timeout)



def copy_login(source: str, target: Path, *refresh: str) -> None:
    """Copy the host's login with its refresh token voided: the container uses the access token
    but cannot refresh it, which would rotate the token the host (and its Remote Control) holds."""

    login = json.loads(Path(source).expanduser().read_text())
    login[refresh[0]][refresh[1]] = "voided-by-relay-tests"
    target.parent.mkdir()
    target.write_text(json.dumps(login))
    target.chmod(0o600)


CALL_ID = re.compile(r'"(?:tool_use_id|call_id|tool_call_id|id)":"([^"]+)"')


def same_call(first: str, later: str) -> bool:
    """Whether two tool results answer the same call: a resent request, or one the harness added
    text to (Claude Code appends its compaction prompt to the last tool result)."""

    ids = set(CALL_ID.findall(first))
    return bool(ids & set(CALL_ID.findall(later))) if ids else first == later


def files_read(codec, harness, items: list) -> list[int]:
    """The files the request's newest item read: a tool result with their content, from a call
    naming them (not Claude Code re-attaching files after compacting itself, nor an agent
    reading a sub-agent's transcript, nor a report quoting a CODE)."""

    view = [harness.refine(codec.classify(item)) for item in items]
    if view[-1].kind is not Kind.TOOL_RESULT:
        return []
    newest = codec.canonical(items[-1]).decode()
    ids, calls = set(CALL_ID.findall(newest)), [n for n, item in enumerate(view) if item.kind is Kind.TOOL_CALL]
    own = [n for n in calls if ids & set(CALL_ID.findall(codec.canonical(items[n]).decode()))] or calls[-1:]
    asked = " ".join(view[n].text for n in own)
    return [n for n in range(1, 9) if f"file {n} line 1:" in newest and f"file_{n}" in asked]


def self_compactions(requests: list[dict]) -> list[str]:
    """Requests in which the harness compacts its own history: those its profile recognizes, and
    those whose last message asks for a summary (each harness words its own)."""

    found = []
    for n, request in enumerate(requests):
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        items = codec.items(request["body"])
        last = harness.refine(codec.classify(items[-1])) if items else None
        text = harness.visible(last.text) if last else ""  # not what the harness injects (skills, memory)
        asks = last and last.kind is Kind.USER and COMPACT_ASK.search(text) \
            and not re.match(r"\W*(Read the files|Background notes)", text)
        if items and (harness.compacting(codec, items) or asks):
            found.append(f"{n}: {text[:70]!r}")
    return found


def results(request: dict) -> list:
    """The tool results a forwarded request carries."""

    codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
    items = (harness.refine(codec.classify(item)) for item in codec.items(request.get("forwarded") or request["body"]))
    return [item for item in items if item.kind is Kind.TOOL_RESULT]


def name_of(keys: list[bytes]) -> tuple[bytes, ...]:
    """A conversation, named as the engine names it: by its first items that are not slots."""

    return tuple([key for key in keys if key not in (b"\0system", b"\0context")][:3])


LAYOUT = {Kind.SYSTEM: "S", Kind.CONTEXT: "C", Kind.USER: "U", Kind.SUMMARY: "Y"}
MARK = json.dumps(SUMMARY_PREFIX)[1:61]  # how every summary starts inside a JSON body


ASKED = tuple(json.dumps(prompt[:80])[1:-1] for prompt in (SUMMARIZATION_PROMPT, RUBRIC, SUMMARIZER))  # a summary's prompt (Codex's, Self-Compact's) inside a JSON body


def own_words(summary: str) -> str:
    """A summary's own words as they appear inside a JSON body (its start is the fixed prefix)."""

    return json.dumps(summary)[1:-1][-200:]
MODEL_CALLS = ("/responses", "/chat/completions", "/messages", ":generateContent", ":streamGenerateContent")


def evaluate(name: str, home: Path, need: tuple[int, int] | None = None, selfcompact: bool = False) -> dict:
    """Check a run; `need` is the (mid-turn, turn-start) compactions it must show, at least one
    (with `selfcompact`, none: the harness must not compact itself either)."""

    def text(file: str) -> str:
        path = home / file
        return path.read_text(errors="replace") if path.exists() else ""

    def records(file: str) -> list[dict]:
        return [json.loads(line) for line in text(file).splitlines()]

    events, trace = records("events.jsonl"), records("trace.jsonl")
    requests = [r for r in trace if "body" in r]
    two_turns = name not in SPECS or bool(SPECS[name].resume)
    log, answers = text("relay.log"), [text("harness.out").strip(), text("harness2.out").strip()][: 1 + two_turns]
    problems: list[str] = []

    # Every compaction: Codex's layout, nothing lost from the summarized history.
    compacting = [r for r in requests if r["compacted"]]
    if len(compacting) != len(events):
        problems.append(f"{len(events)} compactions logged, {len(compacting)} requests compacted")
    starts = mids = 0
    # Conversations besides the task: sub-agents, title generation, background calls.
    starts_of = {tuple(detect({}, path=r["path"]).identity(codec_for(r["path"]), codec_for(r["path"]).items(r["body"]))[:2])
                 for r in requests}
    others = len(starts_of) - 1
    resume_reused = None  # did the first turn-start compaction build on the stored rewrite?
    replaced: list[tuple[str, list[str]]] = []  # (summary, codes it replaced) per compaction
    for k, (event, request) in enumerate(zip(events, compacting), start=1):
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        raw = codec.items(request["body"])
        while raw and harness.volatile(harness.refine(codec.classify(raw[-1]))):
            raw = raw[:-1]  # regenerated on every request; Relay leaves it out, as here

        def kind(entry: dict) -> Kind:
            return Kind(entry["kind"]) if "kind" in entry else harness.refine(codec.classify(raw[entry["ref"]])).kind

        head, covered = split(event["head"], kind, len(raw))
        layout = "".join(LAYOUT.get(kind(entry), "?") for entry in head)
        kept = {e["ref"] for e in head if "ref" in e}
        first_action = next((i for i in range(len(raw)) if kind({"ref": i}) in AGENT_KINDS), len(raw))
        initial = {i for i in range(first_action) if kind({"ref": i}) in CONTEXT_KINDS}
        start = covered < len(raw)
        if start and codec.name == "anthropic_messages":  # no legal place for a system message there
            initial = {i for i in initial if kind({"ref": i}) is not Kind.SYSTEM}
        rendered = harness.state(codec, raw)  # the harness's state now: all of it must be in the new context
        if rendered.items:
            wires = [json.loads(e["wire"]) for e in head if "wire" in e]
            expected = [i for i in rendered.items
                        if (i.ref is None or i.ref < covered)  # later state is still in the tail
                        and not (start and codec.name == "anthropic_messages" and i.kind is Kind.SYSTEM)]
            missing = [i.ref if i.ref is not None else i.text[:60] for i in expected
                       if (i.ref not in kept if i.ref is not None else json.loads(i.wire) not in wires)]
            if missing:
                problems.append(f"compaction {k}: re-rendered initial context {missing!r:.200} missing")
        elif dropped := initial - kept:
            problems.append(f"compaction {k}: initial context {sorted(dropped)} dropped")
        starts, mids = starts + start, mids + (not start)
        if start and resume_reused is None:
            resume_reused = event["items_before"] < len(raw)
        if harness.name == "codex":  # developer sections join Codex's context block
            pattern = r"S*U*Y[SC]*" if start else r"S*U*[SC]*U?Y"
        else:  # Anthropic: system last
            pattern = r"S*U*YC*" if start else r"S*U*C*U?YS*"
        if not re.fullmatch(pattern, layout):
            problems.append(f"compaction {k}: layout {layout} is not Codex's")
        summary = next((e["text"] for e in head if e.get("kind") == "summary"), "")
        read = [c for c in CODES if f"CODE: {c}" in json.dumps(raw[:covered])]
        if lost := [c for c in read if c not in summary.lower()]:
            problems.append(f"compaction {k}: the summary lost {lost} of the {len(read)} codes read")
        users = {i for i in range(covered) if harness.refine(codec.classify(raw[i])).kind is Kind.USER}
        if dropped := users - kept:
            problems.append(f"compaction {k}: user messages {sorted(dropped)} not kept verbatim")
        replaced.append((summary, read))

    # Every later request of each conversation (the task's, a sub-agent's): its latest summary,
    # none of what that summary replaced.
    current: dict[tuple, tuple[str, list[str]]] = {}
    latest = iter(replaced)
    for n, request in enumerate(requests):
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        if harness.compacting(codec, codec.items(request["body"])):
            continue  # the harness summarizing its own history passes through as sent
        conversation = name_of(harness.identity(codec, codec.items(request["body"])))
        if request["compacted"] and (compaction := next(latest, None)):
            current[conversation] = compaction
        if conversation not in current:
            continue  # before its first compaction, or a request that never compacts (titles and the like)
        summary, read = current[conversation]
        sent = json.dumps(request.get("forwarded") or request["body"])
        if request.get("forwarded") is None:
            problems.append(f"request {n}: forwarded without the compacted context")
        elif sent.count(MARK) != 1 or own_words(summary) not in sent:
            problems.append(f"request {n}: carries {sent.count(MARK)} summaries, not just the latest")
        elif leaked := [c for c in read if any(f"CODE: {c}" in item.text for item in results(request))]:
            problems.append(f"request {n}: summarized tool output {leaked} still sent")  # (summaries may quote it)

    # Relay's own record of each request's cache decision: a stored compaction that no longer
    # applies means the harness rewrote history in a way its profile's identity does not see through.
    caches = [r["cache"] for r in requests if "cache" in r]
    for n, cache in enumerate(caches):
        if cache.get("diverged") is not None:
            problems.append(f"request {n}: left its conversation's stored compaction at item {cache['diverged']}")
    hits = sum(1 for r in requests if r.get("cache", {}).get("covered") and not r["compacted"])

    # The harness's latest instructions (BETA after the edit) must reach the model.
    for n, request in enumerate(requests):
        if not request.get("forwarded"):
            continue
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        told = [harness.refine(codec.classify(item)) for item in codec.items(request["body"])]
        injected = " ".join(item.text for item in told if item.kind in {Kind.SYSTEM, Kind.CONTEXT, Kind.USER})
        if "codename is BETA" in injected and "codename is BETA" not in json.dumps(request["forwarded"]):
            problems.append(f"request {n}: the latest instructions (BETA) were dropped")
    need_mid, need_start = need or (2, int(two_turns))
    own = self_compactions(requests)
    if selfcompact and own:
        problems.append(f"the harness compacted itself {len(own)}×: {own[:2]}")
    if not selfcompact and (mids < need_mid or starts < need_start or not mids + starts):
        problems.append(f"compacted {mids}× mid-turn and {starts}× at a turn start (need {need_mid} and {need_start})")
    # A conversation reading a file it already read (agents splitting work may overlap: not a re-read).
    first_read: dict[tuple, str] = {}  # (conversation, file) -> the tool result that returned it
    for request in requests:
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        items = codec.items(request["body"])
        conversation, newest = name_of(harness.identity(codec, items)), codec.canonical(items[-1]).decode()
        if reread := [n for n in files_read(codec, harness, items)
                      if not same_call(first_read.setdefault((conversation, n), newest), newest)]:
            problems.append(f"files read again: {[f'file_{n}.txt' for n in reread]}")
    for turn, (answer, codes) in enumerate(zip(answers, (CODES[:4], CODES) if two_turns else (CODES,)), start=1):
        if not re.search(SEPARATOR.join(codes), answer.lower()):
            problems.append(f"turn {turn} answer is wrong: {answer[-120:]!r}")
    if "failed; forwarding" in log or "could not prepare" in log:
        problems.append("a compaction failed (see relay.log)")
    upstream = [line for line in log.splitlines()  # not probes, e.g. Hermes looking for an Ollama server
                if "httpx HTTP Request: POST" in line and any(path in line for path in MODEL_CALLS)]
    transient = [line for line in upstream if '"HTTP/1.1 429' in line or '"HTTP/1.1 5' in line]
    rejected = sum('"HTTP/1.1 4' in line for line in upstream if line not in transient)
    rejected += sum(r["status"] >= 400 for r in trace if "status" in r)
    if rejected:
        problems.append(f"{rejected} model calls rejected")
    return {
        "name": name,
        "passed": not problems,
        "requests": len(requests),
        "mid_turn": mids,
        "turn_start": starts,
        "tokens_before": [e["tokens_before"] for e in events],
        "resume_reused": resume_reused,
        "other_conversations": others,
        "cache_hits": hits,  # requests that found their conversation's stored compaction (Relay's record)
        "self_compactions": len(own),
        "sub_compactions": sum("CODE: amber" not in json.dumps(r["body"]) for r in compacting),  # e.g. in sub-agents
        # The codename the harness itself last sent (Kimi Code keeps a session's instructions).
        "told": next((c for r in reversed(requests) for c in ("BETA", "ALPHA") if f"codename is {c}" in json.dumps(r["body"])), None),
        "codename": next((c for c in ("BETA", "ALPHA") if c in answers[-1].split("\n")[-1] or c in answers[-1][-80:]), None),
        "transient": len(transient),  # rate limits / overloads the harness retried
        "problems": problems,
        "answers": [answer[-80:] for answer in answers],
        "errors": (text("harness.err")[-300:] + text("harness2.err")[-300:]) if problems else "",
        "home": str(home),
    }


def evaluate_hook(name: str, home: Path) -> dict:
    """Check a run in which the harness compacted through Relay's hook (`--via hook`), against
    what the model received (the recorder's trace). Every compaction: Codex's replacement layout
    (the user messages, then the summary), and a summary that kept every code the model had read
    before it; the requests after it carry exactly that summary and none of the tool output it
    replaced. Also: the harness never compacted on its own (no hook failure fell back to it),
    no file was read twice, both answers are right, and every model call was accepted."""

    def records(file: str) -> list[dict]:
        path = home / file
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    events, trace = records("events.jsonl"), records("trace.jsonl")
    requests = [r for r in trace if "body" in r]
    answers = [(home / f).read_text(errors="replace").strip() if (home / f).exists() else "" for f in ("harness.out", "harness2.out")]
    log = (home / "relay.log").read_text(errors="replace") if (home / "relay.log").exists() else ""
    problems: list[str] = []

    def view(request: dict) -> list:
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        return [harness.refine(codec.classify(item)) for item in codec.items(request["body"])]

    def task(request: dict) -> str:  # a conversation, by its first user message (kept by every compaction)
        return next((harness_visible(i.text) for i in view(request) if i.kind is Kind.USER), "")[:200]

    def codes(items: list) -> set[str]:
        return {c for c in CODES for i in items if i.kind is Kind.TOOL_RESULT and f"CODE: {c}" in i.text}

    mids = starts = 0
    for k, event in enumerate(events, start=1):
        last = max((n for n, e in enumerate(event["head"]) if e.get("kind") == "summary"), default=len(event["head"]))
        layout = "".join("Y" if e.get("kind") == "summary" else "U" for e in event["head"][: last + 1])
        if not re.fullmatch(r"U*Y", layout):
            problems.append(f"compaction {k}: layout {layout} is not Codex's")
        if event["tokens_before"] < HOOK_THRESHOLD and not event.get("forced"):
            problems.append(f"compaction {k}: at {event['tokens_before']} tokens, below the strategy's {HOOK_THRESHOLD}")
        summary = next((e["text"] for e in event["head"] if e.get("kind") == "summary"), "")
        start = any("ref" in e for e in event["head"][last + 1:])  # the new turn follows the summary
        starts, mids = starts + start, mids + (not start)
        first = next((n for n, r in enumerate(requests) if own_words(summary) in json.dumps(r["body"])), None)
        if first is None:
            problems.append(f"compaction {k}: its summary never reached the model")
            continue
        conversation = task(requests[first])
        before = [r for r in requests[:first] if task(r) == conversation]
        read = codes([i for r in before for i in view(r)])
        if lost := sorted(c for c in read if c not in summary.lower()):
            problems.append(f"compaction {k}: the summary lost {lost} of the {len(read)} codes read")
        for n in range(first, len(requests)):  # later compactions replace the summary, never bring back the output
            if task(requests[n]) != conversation or any(a in json.dumps(requests[n]["body"]) for a in ASKED):
                continue  # another conversation, or a fork making a summary (and its continuations)
            sent = json.dumps(requests[n]["body"])
            if sent.count(MARK) != 1:
                problems.append(f"request {n}: carries {sent.count(MARK)} summaries, not just the latest")
            elif leaked := sorted(codes(view(requests[n])) & read):
                problems.append(f"request {n}: summarized tool output {leaked} still sent")
    own = self_compactions(requests)
    if own:
        problems.append(f"the harness compacted on its own {len(own)}×: {own[:2]}")
    if not events:
        problems.append("the harness never compacted through the hook")
    first_read: dict[tuple, str] = {}
    for request in requests:
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        items, newest = codec.items(request["body"]), codec.canonical(codec.items(request["body"])[-1]).decode()
        if reread := [n for n in files_read(codec, harness, items)
                      if not same_call(first_read.setdefault((task(request), n), newest), newest)]:
            problems.append(f"files read again: {[f'file_{n}.txt' for n in reread]}")
    for turn, (answer, wanted) in enumerate(zip(answers, (CODES[:4], CODES)), start=1):
        if not re.search(SEPARATOR.join(wanted), answer.lower()):
            problems.append(f"turn {turn} answer is wrong: {answer[-120:]!r}")
    if "hook compaction failed" in log:
        problems.append("a hook compaction failed (see relay.log)")
    if rejected := sum(r["status"] >= 400 for r in trace if "status" in r):
        problems.append(f"{rejected} model calls rejected")
    return {
        "name": f"{name} (hook)", "passed": not problems, "requests": len(requests), "mid_turn": mids, "turn_start": starts,
        "resume_reused": None, "cache_hits": 0, "other_conversations": len({task(r) for r in requests}) - 1,
        "self_compactions": len(own), "sub_compactions": sum(bool(e.get("agent")) for e in events),
        "told": next((c for r in reversed(requests) for c in ("BETA", "ALPHA") if f"codename is {c}" in json.dumps(r["body"])), None),
        "codename": next((c for c in ("BETA", "ALPHA") if c in answers[-1][-80:]), None),
        "tokens_before": [e["tokens_before"] for e in events], "transient": 0, "problems": problems,
        "answers": [a[-80:] for a in answers], "errors": "", "home": str(home),
    }


def evaluate_clm(name: str, home: Path, steering: str | None = None) -> dict:
    """Check a run with the CLM strategy (`--strategy clm`): the model edited its own context
    through the mirror, and every accepted edit took effect exactly. On the request it was read
    back and every later request of that conversation until the next edit, the turns it removed
    are gone, the ones it kept are the original items, and its notes are there; the harness's own
    items stay ahead of the conversation; every request that offers tools carries the strategy's
    instructions and ends with the size readout. Also: both answers are right and every model call
    was accepted. Re-reads are counted, not failed: a model may drop a file's text and read it
    again. With a steering brief (`--steering`), every edit must also have the shape the brief asks
    for (`POLICIES`); the answers are reported, not judged, as a policy may lose what the task needs."""

    def records(file: str) -> list[dict]:
        path = home / file
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    events, trace = records("events.jsonl"), records("trace.jsonl")
    shapes: list[Shape] = []
    requests = [r for r in trace if "body" in r]
    answers = [(home / f).read_text(errors="replace").strip() if (home / f).exists() else "" for f in ("harness.out", "harness2.out")]
    problems: list[str] = []
    edited = [n for n, r in enumerate(requests) if r["compacted"]]
    sizes = [int(n) for r in requests for n in re.findall(r"\[context: ~(\d+)/", json.dumps(r.get("forwarded") or {}))[-1:]]
    if not events and max(sizes, default=0) > CLM_BUDGET - 2048:  # Relay's readout (some gateways report no usage)
        problems.append("the model never edited its context, past its limit")
    if len(edited) != len(events):
        problems.append(f"{len(events)} edits logged, {len(edited)} requests rewritten")
    for k, (event, at) in enumerate(zip(events, edited), start=1):
        codec, harness = codec_for(requests[at]["path"]), detect({}, path=requests[at]["path"])
        raw = codec.items(requests[at]["body"])
        kept = {e["ref"] for e in event["head"] if "ref" in e and "text" not in e}
        retold = {e["ref"]: e["text"] for e in event["head"] if "ref" in e and "text" in e}  # kept with new text
        removed = [n for n in range(event["covered"]) if n not in kept and n not in retold
                   and harness.refine(codec.classify(raw[n])).kind not in CONTEXT_KINDS]
        notes = [e["text"] for e in event["head"] if "ref" not in e and "wire" not in e]
        refs = [e["ref"] for e in event["head"] if "ref" in e]
        if refs != sorted(refs):  # the harness's items where it sent them, the turns in their order
            problems.append(f"edit {k}: request items out of order: {refs}")
        # What the edit did to the turns the mirror showed: those before this request's own items.
        identity = [codec.canonical(item) for item in raw]
        seen = max((len(earlier) for r in requests[:at]
                    if (earlier := [codec.canonical(item) for item in codec.items(r["body"])]) == identity[: len(earlier)]),
                   default=0)
        conversation = [n for n in range(event["covered"]) if harness.refine(codec.classify(raw[n])).kind not in CONTEXT_KINDS]
        # The model's own turns after the protected task (the user's are left to the user).
        mirrored = [n for n in conversation[1:] if n < seen and harness.refine(codec.classify(raw[n])).kind is not Kind.USER]
        survivors = [n for n in mirrored if n in kept or n in retold]
        shapes.append(Shape(event["tokens_before"], len(mirrored), len([n for n in mirrored if n in removed]),
                            sorted({harness.refine(codec.classify(raw[n])).kind.value for n in retold}), len(retold),
                            len(notes), not removed or max(removed) < min(survivors, default=len(raw))))
        until = next((n for n in edited if n > at), len(requests))
        later = [r for r in requests[at:until] if codec.items(r["body"])[: event["covered"]] == raw[: event["covered"]]]
        for r in later:
            sent = codec.items(r.get("forwarded") or r["body"])
            sent.append(without_note(sent[-1]))  # Anthropic and Gemini join the notes to the last user turn
            sent += [without_guidance(item) for item in sent if "## Managing your context" in json.dumps(item)]
            if gone := [n for n in kept if raw[n] not in sent]:
                problems.append(f"edit {k}: kept turns {gone} not forwarded as they were")
            since = codec.items(r["body"])[len(raw):]  # lookalikes the harness sent later (Hermes's empty replies)
            if back := [n for n in removed if sent.count(raw[n]) > sum(raw[m] == raw[n] for m in kept) + since.count(raw[n])]:
                problems.append(f"edit {k}: removed turns {back} forwarded again")
            if lost := [t[:40] for t in [*notes, *retold.values()] if json.dumps(t)[1:-1] not in json.dumps(sent)]:
                problems.append(f"edit {k}: notes or new text {lost} missing")
            if gone or back or lost:
                break
    for n, r in enumerate(requests):
        if r["body"].get("tools") and not ("## Managing your context" in json.dumps(r.get("forwarded") or {})
                                           and "[context: ~" in json.dumps((r.get("forwarded") or {}))):
            problems.append(f"request {n}: without the strategy's instructions or readout")
            break
    turns = 2 if SPECS[name].resume else 1
    wrong = [f"turn {turn} answer is wrong: {answer[-120:]!r}"
             for turn, (answer, wanted) in enumerate(zip(answers[:turns], (CODES[:4], CODES)), start=1)
             if not re.search(SEPARATOR.join(wanted), answer.lower())]
    if not steering:
        problems += wrong
    else:
        policy = POLICIES[steering]
        if not shapes:
            problems.append(f"policy {steering}: the model never edited its context")
        problems += [f"policy {steering}: edit {k} at {shape.tokens} tokens is {shape}" for k, shape in enumerate(shapes, 1)
                     if policy.edit and not policy.edit(shape)]
        if policy.edits and len(shapes) < policy.edits:
            problems.append(f"policy {steering}: {len(shapes)} edits, at least {policy.edits} expected")
        if steering == "backup":
            backups = [f for d in [*home.rglob("compaction_backup"), *home.parent.joinpath("project").rglob("compaction_backup")]
                       for f in d.rglob("*") if f.is_file()]
            if len(backups) < len(shapes):
                problems.append(f"policy backup: {len(backups)} backups for {len(shapes)} edits")
    if rejected := sum(r["status"] >= 400 for r in trace if "status" in r):
        problems.append(f"{rejected} model calls rejected")
    refused = sum(re.search(r"edit (NOT applied|REJECTED)", json.dumps(r.get("forwarded") or {})) is not None
                  for r in requests)
    usage = [r["usage"] for r in trace if r.get("usage")]
    rereads = 0
    first_read: dict[int, str] = {}
    for request in requests:
        codec, harness = codec_for(request["path"]), detect({}, path=request["path"])
        items = codec.items(request["body"])
        newest = codec.canonical(items[-1]).decode()
        rereads += sum(not same_call(first_read.setdefault(n, newest), newest) for n in files_read(codec, harness, items))
    return {
        "name": f"{name} (clm{f', {steering}' if steering else ''})", "passed": not problems, "requests": len(requests),
        "edits": len(events), "refused_edits": refused, "rereads": rereads, "prompt_tokens": [f"{u // 1000}k" for u in usage],
        "edit_shapes": [f"{s.tokens // 1000}k: {s}" for s in shapes] if steering else None,
        "task": ("answers right" if not wrong else "; ".join(wrong)) if steering else None,
        "answers": [a[-80:] for a in answers], "problems": problems, "home": str(home),
    }


def evaluate_clm_edit(name: str, home: Path) -> dict:
    """Check a `--clm-edit` run, in four parts: an edit was applied (or the model never wrote the
    file, or Relay refused what it wrote); Relay applied it faithfully (`evaluate_clm`'s checks of
    every edit); the context then is what the model was asked for, on every later request (file_2's
    call and output gone, file_1's output rewritten in place, its call kept, the marker there); and
    the model, answering from it, sees it so (the marker, the rewritten output, file_2's CODE gone)."""

    def records(file: str) -> list[dict]:
        path = home / file
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    trace = records("trace.jsonl")
    requests = [r for r in trace if "body" in r and "MARKER 7319" in json.dumps(r["body"])  # the task's conversation
                and codec_for(r["path"]).offers_tools(r["body"])]  # (not a title made from it)
    resumable = bool(SPECS[name].resume)
    answers = [(home / f).read_text(errors="replace").strip() if (home / f).exists() else ""
               for f in ("harness.out", "harness2.out")[: 1 + resumable]]

    def items(r: dict) -> list[Item]:
        codec, harness = codec_for(r["path"]), detect({}, path=r["path"])
        return [harness.refine(codec.classify(i)) for i in codec.items(r.get("forwarded") or r["body"])]

    wrote = any(".live_ctx" in i.text for r in requests for i in items(r) if i.kind is Kind.TOOL_CALL)
    refused = [m.group(0)[:160] for r in requests  # Relay's receipts end a request (a model may read Relay's code)
               if (m := re.search(r"\[context file: edit (NOT applied|REJECTED)[^\]]*", items(r)[-1].text))]
    edited = [n for n, r in enumerate(requests) if r["compacted"]]
    applied = "yes" if edited else "no: " + ("Relay refused it" if refused else "the model wrote the file, Relay saw no change"
                                             if wrote else "the model never wrote the context file")
    faithful = [p for p in evaluate_clm(name, home)["problems"] if p.startswith("edit ")]

    def reads(text: str, n: int) -> bool:  # a call reading file_n alone
        return f"file_{n}.txt" in text and all(f"file_{m}" not in text for m in (1, 2, 3) if m != n)

    asked: list[str] = []
    for n in range(edited[-1] if edited else len(requests), len(requests)):
        seen = items(requests[n])
        own = [i for i in seen if i.kind is Kind.TOOL_CALL and ".live_ctx" not in i.text]  # not the edit's own calls
        output = lambda k: any(i.kind is Kind.TOOL_RESULT and i.text.count(f"file {k} line ") >= 50 for i in seen)  # noqa: E731
        asked = [what for what, bad in [
            ("file_2's call is there", any(reads(i.text, 2) for i in own)),
            ("file_2's output is there", output(2)),
            ("file_1's output is there", output(1)),
            ("file_1's output is not rewritten in place",
             not any(i.kind is Kind.TOOL_RESULT and "MASKED file_1: CODE amber" in i.text for i in seen)),
            ("file_1's call is gone", not any(reads(i.text, 1) for i in own)),
            ("the marker is missing", not any(i.kind is Kind.USER and "MARKER 7319" in i.text
                                              and "Change nothing else" not in i.text for i in seen)),
        ] if bad]
        if asked:
            asked = [f"request {n}: {'; '.join(asked)}"]
            break
    answer = answers[-1].lower()
    sees = "marker 7319" in answer and "masked file_1" in answer and "none" in answer
    problems = [*([] if edited else [f"no edit applied: {applied[4:]}"]), *faithful, *asked,
                *([] if sees else [f"the model does not see the edited context: {answers[-1][-160:]!r}"])]
    if rejected := sum(r["status"] >= 400 for r in trace if "status" in r):
        problems.append(f"{rejected} model calls rejected")
    return {"name": f"{name} (clm edit)", "passed": not problems, "requests": len(requests), "edits": len(edited),
            "applied": applied, "faithful": not faithful, "as_asked": bool(edited) and not asked, "sees": sees,
            "refused_edits": refused, "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


def evaluate_folding(name: str, home: Path) -> dict:
    """Check a `--strategy folding` run: the agent opened a branch (its echo answered by FoldAgent's
    branch message) and returned; from the fold on, every request of the conversation keeps the
    branch call answered by the returned message (with file_2's and file_3's CODEs), and neither
    file's text, while file_1 and file_4, read in the main conversation, stay; the request after the
    fold reads its prefix from the upstream's cache; and the model, answering from it, has the CODEs
    and no longer file_2's text."""

    def records(file: str) -> list[dict]:
        path = home / file
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    trace = records("trace.jsonl")
    records_ = [r for r in trace if "body" in r or "usage" in r]
    requests = [r for r in trace if "body" in r and "open a branch" in json.dumps(r["body"])
                and codec_for(r["path"]).offers_tools(r["body"])]
    resumable = bool(SPECS[name].resume)
    answers = [(home / f).read_text(errors="replace").strip() if (home / f).exists() else ""
               for f in ("harness.out", "harness2.out")[: 1 + resumable]]

    def items(r: dict) -> list[Item]:
        codec, harness = codec_for(r["path"]), detect({}, path=r["path"])
        return [harness.refine(codec.classify(i)) for i in codec.items(r.get("forwarded") or r["body"])]

    problems: list[str] = []
    opened = any("ROLE CHANGE: `MODE: BRANCH`" in i.text for r in requests for i in items(r))
    folds = [n for n, r in enumerate(requests) if r["compacted"]
             and any("Branch has finished its task" in i.text for i in items(r))]
    if not opened:
        problems.append("no branch opened")
    if not folds:
        problems.append("no branch returned")
    for n in range(folds[0] if folds else len(requests), len(requests)):
        seen = items(requests[n])
        if any("ROLE CHANGE: `MODE: BRANCH`" in i.text for i in seen):
            continue  # inside another branch, its own steps are there
        returned = [i for i in seen if i.kind is Kind.TOOL_RESULT and "Branch has finished its task" in i.text]
        full = lambda k: any(i.kind is Kind.TOOL_RESULT and i.text.count(f"file {k} line ") >= 50 for i in seen)  # noqa: E731
        wrong = [what for what, bad in [
            ("the returned message is missing", not returned),
            ("the returned message lacks file_2's or file_3's CODE",
             returned and not ("birch" in returned[-1].text.lower() and "cobalt" in returned[-1].text.lower())),
            ("file_2's text is there", full(2)), ("file_3's text is there", full(3)),
            ("file_1's text is gone", not full(1)),
        ] if bad]
        if wrong:
            problems.append(f"request {n}, after the fold: {'; '.join(wrong)}")
            break
    cache = None  # the request after the fold: how much of its prompt the upstream's cache served
    if folds and folds[0] + 1 < len(requests):
        after = requests[folds[0] + 1]
        usage = next((r for r in records_[records_.index(after) + 1:] if "usage" in r), {})
        if usage.get("usage") and usage.get("cached") is not None:
            cache = f"{usage['cached']}/{usage['usage']}"
            if usage["cached"] < 0.5 * usage["usage"]:
                problems.append(f"after the fold the cache served only {cache} prompt tokens")
    answer = answers[-1].lower()
    if not re.search(r"amber[\s,]+birch[\s,]+cobalt[\s,]+dune", answer):
        problems.append(f"the CODEs are wrong: {answers[-1][-120:]!r}")
    if resumable and not re.search(r"\bno\b", answer):
        problems.append(f"the model still sees file_2's text, it says: {answers[-1][-120:]!r}")
    if rejected := sum(r["status"] >= 400 for r in trace if "status" in r):
        problems.append(f"{rejected} model calls rejected")
    return {"name": f"{name} (folding)", "passed": not problems, "requests": len(requests), "folds": len(folds),
            "cache_after_fold": cache, "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


def evaluate_prolong(name: str, home: Path) -> dict:
    """Check a `--strategy prolong` run: the log holds every event of the session once (the user's
    prompts, file_2's text with its animal, the replies) and no read of itself; the context was
    compacted, with the PRO-LONG skill on every request; in turn 2 the agent searched the log rather
    than reading file_2 again, and named file_2's animal."""

    def records(file: str) -> list[dict]:
        path = home / file
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    trace, events = records("trace.jsonl"), records("events.jsonl")
    log = [json.loads(line) for line in (home.parent / "project/.prolong/log.jsonl").read_text().splitlines()] \
        if (home.parent / "project/.prolong/log.jsonl").exists() else []
    requests = [r for r in trace if "body" in r and codec_for(r["path"]).offers_tools(r["body"])]
    answers = [(home / f).read_text(errors="replace").strip() if (home / f).exists() else "" for f in ("harness.out", "harness2.out")]
    problems: list[str] = []
    texts = [json.dumps(entry["content"]) for entry in log]
    types = {entry["type"] for entry in log}
    if not log:
        problems.append("no log")
    elif missing := {"user_prompt", "tool_call", "tool_result", "assistant_message"} - types:
        problems.append(f"the log has no {sorted(missing)}")
    if log and not any("file 2 animal: otter" in t for t in texts):
        problems.append("file_2's text is not in the log")
    if len(texts) != len(set(texts)) and sum(t.count("Read the files file_1.txt") for t in texts) > 1:
        problems.append("events recorded twice")
    if any(".prolong/log.jsonl" in t for t in texts):
        problems.append("a read of the log was recorded")
    if not events:
        problems.append("the context was never compacted")
    if not all("## PRO-LONG memory" in json.dumps(r.get("forwarded") or {}) for r in requests):
        problems.append("a request went without the PRO-LONG skill")
    later = [r for r in requests if "that line of file_2.txt" in json.dumps(r["body"])]
    turn2 = [codec_for(later[-1]["path"]).classify(x) for x in codec_for(later[-1]["path"]).items(later[-1]["body"])] \
        if later else []
    asked = next((n for n, i in enumerate(turn2) if "that line of file_2.txt" in i.text), len(turn2))
    calls = [i.text for i in turn2[asked:] if i.kind is Kind.TOOL_CALL]  # turn 2's
    searched = any(".prolong" in c for c in calls)
    if not searched:
        problems.append("the agent never searched the log")
    rest = turn2[asked:]
    if any(c.kind is Kind.TOOL_CALL and ".prolong" not in c.text and r.kind is Kind.TOOL_RESULT and "file 2 line 1:" in r.text
           for c, r in zip(rest, rest[1:])):  # (a search of the log may print file_2's text; reading the file is the rest)
        problems.append("the agent read file_2.txt again")
    if "otter" not in answers[1].lower():
        problems.append(f"turn 2 answer is wrong: {answers[1][-120:]!r}")
    if rejected := sum(r["status"] >= 400 for r in trace if "status" in r):
        problems.append(f"{rejected} model calls rejected")
    summaries = [e2["text"] for e in events for e2 in e["head"] if e2.get("kind") == "summary"]
    return {"name": f"{name} (prolong)", "passed": not problems, "requests": len(requests), "compactions": len(events),
            "log_entries": len(log), "summaries_kept_the_animal": any("otter" in s.lower() for s in summaries),
            "searched_the_log": searched, "answers": [a[-80:] for a in answers], "problems": problems, "home": str(home)}


def baseline_run(name: str, home: Path) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """A baseline run's trace, events, the model's requests on the task, and the harness's answers."""

    def records(file: str) -> list[dict]:
        path = home / file
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    trace = records("trace.jsonl")
    requests = [r for r in trace if "body" in r and "file_1.txt" in json.dumps(r["body"])
                and codec_for(r["path"]).offers_tools(r["body"])]
    answers = [(home / f).read_text(errors="replace").strip() if (home / f).exists() else ""
               for f in ("harness.out", "harness2.out")[: 1 + bool(SPECS[name].resume)]]
    return trace, records("events.jsonl"), requests, answers


def seen(r: dict) -> list[Item]:
    """What the model saw on a traced request."""

    codec, harness = codec_for(r["path"]), detect({}, path=r["path"])
    return [harness.refine(codec.classify(i)) for i in codec.items(r.get("forwarded") or r["body"])]


def full(items: list[Item], k: int) -> bool:
    """file_k's text is there, in full."""

    return any(i.kind is Kind.TOOL_RESULT and i.text.count(f"file {k} line ") >= 50 for i in items)


def rejected(trace: list[dict]) -> list[str]:
    return [f"{n} model calls rejected"] if (n := sum(r["status"] >= 400 for r in trace if "status" in r)) else []


ALL_CODES = re.compile(SEPARATOR.join(CODES))
FOUR_CODES = re.compile(SEPARATOR.join(CODES[:4]))


def evaluate_selfcompact(name: str, home: Path) -> dict:
    """Check a `--strategy selfcompact` run: Relay asked the model Self-Compact's rubric, and at least
    once it fired and the summary followed (the rubric and the summary in one change); the request
    it changed holds the user's messages and the summary in place of the trajectory (no tool result
    left); and the model, going on from its own summary, had all 8 CODEs at the end."""

    trace, events, requests, answers = baseline_run(name, home)
    probes = [r for r in trace if "summary_request" in r and "You are about to decide whether to compress"
              in json.dumps(r["summary_request"])]
    verdicts = [" ".join(re.findall(r"\b(?:C1|C2|C3|N1)\b[^A-Za-z0-9\n]*[YN]\b", codec_for(r["path"]).output_text(r["answer"])))
                for r in probes if isinstance(r.get("answer"), dict)]
    fired = [e for e in events if e.get("summary_requests", 0) >= 2]
    problems = [] if probes else ["the rubric was never asked"]
    if not fired:
        problems.append(f"the rubric never fired ({len(events)} forced summaries)")
    for r in [r for r in requests if r["compacted"]]:
        if any(i.kind is Kind.TOOL_RESULT for i in seen(r)):
            problems.append("a tool result survived the summary")
            break
    if not ALL_CODES.search(answers[-1].lower()):
        problems.append(f"the CODEs are wrong: {answers[-1][-120:]!r}")
    problems += rejected(trace)
    return {"name": f"{name} (selfcompact)", "passed": not problems, "requests": len(requests), "probes": len(probes),
            "verdicts": verdicts, "summaries": [e.get("summary_requests") for e in events],
            "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


def evaluate_autocompact(name: str, home: Path) -> dict:
    """Check a `--strategy autocompact` run: the agent compacted (its echo); from then on every
    request shows the task, the turn before the call (file_3's read) and the call answered by an
    "# Auto Context Summary", and neither file_1's nor file_2's text; the model, answering from it,
    has the CODEs and no longer file_2's text."""

    trace, events, requests, answers = baseline_run(name, home)
    problems = [] if events else ["the agent never compacted"]
    first = next((n for n, r in enumerate(requests) if r["compacted"]), len(requests))
    for n in range(first, len(requests)):
        items = seen(requests[n])
        at = next((k for k, i in enumerate(items) if i.kind is Kind.TOOL_RESULT and "# Auto Context Summary" in i.text), None)
        kept = items[:at]  # what stayed of the history the summary replaced (reads after it are the model's own)
        wrong = [what for what, bad in [
            ("the compact call is not answered by the summary", at is None),
            ("file_1's text is kept", full(kept, 1)), ("file_2's text is kept", full(kept, 2)),
            ("file_3's read (the last turn) is gone", not full(kept, 3)),
        ] if bad]
        if wrong:
            problems.append(f"request {n}, after the compaction: {'; '.join(wrong)}")
            break
    answer = answers[-1].lower()
    if not FOUR_CODES.search(answer):
        problems.append(f"the CODEs are wrong: {answers[-1][-120:]!r}")
    if len(answers) > 1 and not re.search(r"\bno\b", answer):  # (asked in turn 2 alone, not beside the task)
        problems.append(f"the model still sees file_2's text, it says: {answers[-1][-120:]!r}")
    problems += rejected(trace)
    return {"name": f"{name} (autocompact)", "passed": not problems, "requests": len(requests),
            "compactions": len(events), "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


def evaluate_acm(name: str, home: Path) -> dict:
    """Check a `--strategy acm` run: manage_context compressed the three reads (its echo answered by
    "[summary_id: 1] ...", their originals saved), and from then on no request shows their text;
    query_memory recalled file_2's animal from the saved originals; every request after a tool
    result ends with the context size; the model has the CODEs and the animal."""

    trace, events, requests, answers = baseline_run(name, home)
    saved = sorted((home.parent / "project/.acm").glob("*/summary_1.json"))
    first = next((n for n, r in enumerate(requests) if any(i.kind is Kind.TOOL_RESULT and "[summary_id: 1]" in i.text
                                                           for i in seen(r))), None)
    problems = [] if first is not None else ["manage_context never compressed"]
    if not saved or "file 2 animal: otter" not in saved[0].read_text():
        problems.append("the originals were not saved")
    for n in range(len(requests) if first is None else first, len(requests)):
        items = seen(requests[n])
        at = next(k for k, i in enumerate(items) if i.kind is Kind.TOOL_RESULT and "[summary_id: 1]" in i.text)
        if kept := [k for k in (1, 2, 3) if full(items[:at], k)]:
            problems.append(f"request {n}, after the summary: file_{kept[0]}'s text is kept")
            break
    recalls = [i.text for r in requests for i in seen(r) if i.kind is Kind.TOOL_RESULT
               and "[query_memory: summary_id=" in i.text]
    if not recalls:
        problems.append("query_memory never answered")
    elif "otter" not in recalls[-1].lower():
        problems.append(f"the recall missed the animal: {recalls[-1][-120:]!r}")
    if any("[CURRENT CONTEXT TOKEN: " not in json.dumps(r.get("forwarded") or {}) for r in requests
           if any(i.kind is Kind.TOOL_RESULT for i in seen(r))):
        problems.append("a request after a tool result went without the context size")
    if len(answers) > 1 and not FOUR_CODES.search(answers[0].lower()):
        problems.append(f"turn 1 CODEs are wrong: {answers[0][-120:]!r}")
    if "otter" not in answers[-1].lower():
        problems.append(f"the last answer is wrong: {answers[-1][-120:]!r}")
    problems += rejected(trace)
    return {"name": f"{name} (acm)", "passed": not problems, "requests": len(requests), "changes": len(events),
            "recalls": len(recalls), "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


def evaluate_mem1(name: str, home: Path) -> dict:
    """Check a `--strategy mem1` run: every request carries MEM1's guidance; once the model wrote an
    internal state, each request shows nothing of its work from before the newest one (the user's
    messages stay); and the model, carrying the CODEs only in its internal state, had all 8 at the end."""

    trace, events, requests, answers = baseline_run(name, home)
    problems, states = [], 0
    if not all("## Memory" in json.dumps(r.get("forwarded") or {}) for r in requests):
        problems.append("a request went without the guidance")
    for n, r in enumerate(requests):
        items = seen(r)
        marks = [k for k, i in enumerate(items) if i.kind in (Kind.ASSISTANT, Kind.TOOL_CALL) and "<IS>" in i.text]
        states += bool(marks)
        if marks and any(i.kind is Kind.TOOL_RESULT for i in items[: marks[-1]]):
            problems.append(f"request {n}: a result from before the newest internal state is there")
            break
    if states < 2:
        problems.append(f"the model wrote an internal state on {states} requests")
    if not ALL_CODES.search(answers[-1].lower()):
        problems.append(f"the CODEs are wrong: {answers[-1][-120:]!r}")
    problems += rejected(trace)
    return {"name": f"{name} (mem1)", "passed": not problems, "requests": len(requests), "with_state": states,
            "cuts": len(events), "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


def evaluate_rlm(name: str, home: Path, persistent: bool = False) -> dict:
    """Check a `--strategy rlm` run (`rlm_persistent`: the authors' multi-turn mode): before every
    step of the agent the RLM ran (its own requests to the task's model), and the model taking the
    step saw the RLM's result instead of the conversation (no file's text); persistent, later runs
    found the earlier contexts in the REPL; and the agent's answers are right."""

    trace, events, requests, answers = baseline_run(name, home)
    runs = [r for r in trace if "summary_request" in r
            and "You are a Recursive Language Model" in json.dumps(r["summary_request"])]
    handed = [r for r in requests if "Result from the Recursive Language Model" in json.dumps(r.get("forwarded") or {})]
    problems = [] if runs else ["the RLM never ran"]
    if len(handed) != len(requests):
        problems.append(f"{len(requests) - len(handed)} of {len(requests)} steps went without the RLM's result")
    if any(full(seen(r), k) for r in handed for k in (1, 2, 3)):
        problems.append("a step saw a file's text")
    later = [r for r in runs if "contexts available (context_0 through" in json.dumps(r["summary_request"])]
    if persistent and not later:
        problems.append("no run found the earlier contexts")
    for turn, answer in enumerate(answers, start=1):
        if not re.search(SEPARATOR.join(CODES[:3]), answer.lower()):
            problems.append(f"turn {turn} answer is wrong: {answer[-120:]!r}")
    problems += rejected(trace)
    return {"name": f"{name} ({'rlm_persistent' if persistent else 'rlm'})", "passed": not problems,
            "requests": len(requests), "rlm_requests": len(runs), "runs_with_earlier_contexts": len(later),
            "answers": [a[-100:] for a in answers], "problems": problems, "home": str(home)}


BASELINES = {"selfcompact": evaluate_selfcompact, "autocompact": evaluate_autocompact, "acm": evaluate_acm,
             "mem1": evaluate_mem1, "rlm": evaluate_rlm,
             "rlm_persistent": lambda name, home: evaluate_rlm(name, home, persistent=True)}


@dataclass(frozen=True)
class Shape:
    """What one CLM edit did to the model's own turns the mirror showed (after the protected task)."""

    tokens: int  # the context's size when the edit was read back
    mirrored: int  # turns in the mirror
    removed: int  # of them, removed
    retold_kinds: list[str]  # kinds of the turns kept with new text
    retold: int
    notes: int  # notes in the new context (the model's own blocks)
    oldest_first: bool  # every removed turn older than every surviving one

    def __str__(self) -> str:
        return (f"-{self.removed}/{self.mirrored} removed, {self.retold} retold {self.retold_kinds or ''}, "
                f"{self.notes} notes{', oldest first' if self.oldest_first else ''}").replace(" , ", ", ")


@dataclass(frozen=True)
class Policy:
    """A steering brief (tests/docker/steering/NAME.md) and the shape each of its edits must have."""

    edit: Callable[[Shape], bool] | None = None
    edits: int = 1  # at least this many
    nudges: bool = False  # the budget nudges on (off where the brief says when to act)


POLICIES = {
    # One handoff summary in place of all the mirror showed, and only past 20k tokens.
    "compaction": Policy(lambda s: s.notes >= 1 and s.removed >= 0.9 * s.mirrored and s.tokens >= 19_000),
    # The oldest turns gone, nothing written, nothing retold.
    "sliding_window": Policy(lambda s: s.notes == 0 and s.retold == 0 and s.removed >= 1 and s.oldest_first),
    # Tool results retold, nothing removed or added.
    "masking": Policy(lambda s: s.removed == 0 and s.notes == 0 and s.retold >= 1 and s.retold_kinds == ["tool_result"]),
    # One memory block in place of all the mirror showed, after each file.
    "memory": Policy(lambda s: s.notes == 1 and s.removed >= 0.9 * s.mirrored, edits=3),
    # A backup before every edit (counted from the files); edits come from the default nudges.
    "backup": Policy(nudges=True),
}


def without_guidance(item: dict) -> dict:
    """A system message without the strategy's guidance (Chat Completions joins it to the first)."""

    content = item.get("content")
    if isinstance(content, str):
        return {**item, "content": content.split("\n\n## Managing your context")[0]}
    return {**item, "content": content[:-1]} if isinstance(content, list) and content else item


def without_note(item: dict) -> dict:
    """A message without its last part, where a note was joined to it."""

    key = next((k for k in ("content", "parts") if isinstance(item.get(k), list) and item[k]), None)
    return {**item, key: item[key][:-1]} if key else item


def split(entries: list[dict], kind: Callable[[dict], Kind], length: int) -> tuple[list[dict], int]:
    """A compaction's stored context as (head, covered): the entries before the history it keeps
    (up to the summary, and the harness's state placed with it), and where that history begins."""

    last = max((n for n, e in enumerate(entries) if e.get("kind") == "summary"), default=-1)
    covered = min((e["ref"] for e in entries[last + 1:] if "ref" in e and kind(e) not in CONTEXT_KINDS), default=length)
    return [e for e in entries if not ("ref" in e and e["ref"] >= covered)], covered


def harness_visible(text: str) -> str:
    return detect({}, "claude_code").visible(text)


def show(result: dict) -> None:
    print(f"\n== {result['name']}: {'PASS' if result['passed'] else 'FAIL'}  ({result['home']})")
    for key, value in result.items():
        if key not in {"name", "passed", "home"} and (value or key not in {"problems", "errors"}):
            print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"harness:model, from {', '.join(SPECS)}")
    parser.add_argument("--matrix", action="store_true", help="run every spec (or the given ones)")
    parser.add_argument("--growth", type=int, help=f"tokens between compactions (default {GROWTH})")
    parser.add_argument("--probe", action="store_true", help="record what the harness keeps as state")
    parser.add_argument("--subagent", action="store_true", help="turn 2 hands three files to a sub-agent")
    parser.add_argument("--window", type=int, help="use Relay's own trigger for a context window of this size")
    parser.add_argument("--native-compact", action="store_true", help="the harness compacts itself between the turns")
    parser.add_argument("--file-lines", type=int, default=120, help="lines per file (each ~18 tokens)")
    parser.add_argument("--strategy", default="compaction",
                        choices=["compaction", "clm", "folding", "prolong", "selfcompact", "autocompact", "acm", "mem1",
                                 "rlm", "rlm_persistent"],
                        help="Relay's strategy (clm: the model edits its own context)")
    parser.add_argument("--steering", help="with --strategy clm: a brief from tests/docker/steering (compaction, "
                        "sliding_window, masking, memory, backup), the model told how to manage its context")
    parser.add_argument("--clm-edit", action="store_true", help="with --strategy clm: the model edits its context as "
                        "told (delete, rewrite, add) and answers from what it then sees")
    parser.add_argument("--via", choices=["proxy", "hook"], default="proxy",
                        help="install through this path (hook: the harness compacts through Relay's hook)")
    parser.add_argument("--harness-compaction", action="store_true",
                        help="the harness's own auto-compaction set to fire early; it must stay off")
    options = parser.parse_args()
    names = options.names or (list(SPECS) if options.matrix else parser.error("name a spec or use --matrix"))
    results = []
    for name in names:
        results.append(run(name, options.growth, probe=options.probe, subagent=options.subagent, window=options.window,
                           native=options.native_compact, selfcompact=options.harness_compaction, via=options.via,
                           lines=options.file_lines, strategy=options.strategy, steering=options.steering,
                           clm_edit=options.clm_edit))
        if options.probe:
            print(f"== {name}: probe recorded  ({results[-1]['home']})")
        else:
            show(results[-1])
    if options.probe:
        return 0
    if len(results) > 1:
        print("\n| harness:model | mid-turn | turn start | requests | result |")
        print("|---|---|---|---|---|")
        for r in results:
            print(f"| {r['name']} | {r.get('mid_turn', '-')} | {r.get('turn_start', '-')} | {r['requests']} "
                  f"| {'PASS' if r['passed'] else 'FAIL: ' + '; '.join(r['problems'])} |")
    out = Path(os.getenv("TMPDIR", "/tmp")) / "relay-matrix-results.jsonl"
    with out.open("a") as stream:
        for r in results:
            stream.write(json.dumps(r) + "\n")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
