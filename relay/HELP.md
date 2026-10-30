# Relay

Relay manages what a coding agent's model sees. It sits between the agent (Codex, Claude Code,
Gemini CLI, pi, OpenCode and others: a "harness") and the model's API, and on every request
its strategy decides the context: Codex's compaction, a context the model edits itself (CLM),
and more. The harness, its login and its provider stay as they are.

## Installing

    curl -fsSL https://raw.githubusercontent.com/sunnweiwei/relay/main/install.sh | sh

This gives Relay an environment of its own: uv's, where uv is installed, or else a virtual
environment in `~/.local/share/relay`. It puts the `relay` command in `~/.local/bin` and, if that
folder is not on PATH, adds it to `~/.profile` (and `~/.bashrc` or `~/.zshrc`) for shells started
afterwards. From a
checkout, run `sh install.sh .`. Running the installer again upgrades Relay. It needs Python
3.11 or newer, or uv.

## Setting it up

    relay setup codex --strategy clm

This does three things, in order:

1. It saves the strategy (and any `--set name=value`) in Relay's settings.
2. It starts Relay as a process of its own, a daemon. Its pid and log are in
   `~/.config/relay/`. Nothing is registered with the system.
3. Once Relay answers, it points the harness at it. `relay install` edits only the harness's
   own config file, and records the old values.

If Relay does not start, the harness is left untouched. If something else already listens on
Relay's port (8787 by default), setup picks a free port and saves it. Relay compacts at the
context window the harness itself gives the model where it can tell (Codex: its model catalog),
and from its own table otherwise; `context_window` overrides both. The harness names are `codex`,
`claude`, `gemini_cli`, `pi`, `opencode`, `kilo`, `crush`, `openclaw`, `goose`, `kimi_code`,
`hermes`, `nanobot`, `dsh`, `workbuddy` and `mini`.

### Claude Code: its own hook, or the proxy

Claude Code (2.1.294 or newer) goes through Relay as its HTTPS proxy by default (the way an
organization's proxy is set): it keeps talking to api.anthropic.com, so Remote Control (from the
claude.ai or mobile app, `/remote-control`, `claude remote-control`) keeps working, as do the
features Claude Code keeps for Anthropic's own endpoint, and Relay manages each request as through
the proxy, its own history untouched. Relay makes a certificate authority of its own for it, which
only Claude Code trusts and which can vouch for api.anthropic.com alone. The commands Claude Code
runs go through the same proxy (tunnelled as they are). Where Claude Code already goes through a
proxy, setup takes its plugin hook instead (Remote Control kept, fewer strategies); a strategy
checked on neither (`acm`, `ace`, `mem1`, the RLMs) runs through the plain proxy, where Remote
Control does not start. `relay strategy` moves Claude Code between them, for the conversations
started from then on; `relay status` shows the path; `--via https|hook|proxy` names one.
On the hook, a strategy changed while a session is open decides its compactions from the next
request, but its instructions (folding's, PRO-LONG's) reach the model from a new session, and one
the hook cannot carry leaves that session to compact on its own; Claude Code shows a notice when
either happens.

### Keeping Relay running

A harness pointed at Relay reaches its model only through Relay. Where the harness can run a
command before its first model request, `relay setup` adds one. It runs `relay start`, which
returns at once while Relay answers and starts it otherwise, so a machine's restart costs nothing.
The command says nothing, and it never sees or changes a conversation. `relay uninstall` removes
it with the rest.

| Harness | What runs `relay start` |
| --- | --- |
| Claude Code, WorkBuddy / CodeBuddy, Gemini CLI | a `SessionStart` hook in its settings |
| Kimi Code | a `UserPromptSubmit` hook (each prompt) |
| pi | an extension (each agent run) |
| OpenCode, Kilo | a plugin (each model request) |
| Hermes Agent | a plugin, enabled in `plugins.enabled` (each model request) |
| Codex, DeepSeek Harness, OpenClaw, Crush, Goose, nanobot, mini-swe-agent | nothing: after a restart, run `relay start` |

Codex runs a hook only once its user has trusted it in Codex, and Relay does not trust hooks on
anyone's behalf. DeepSeek Harness runs its hooks in its sandbox, where nothing can start Relay.
Hermes's shell hooks need consent too, so its hook is a plugin.

`relay setup <harness> --autostart` registers Relay with the system's service manager instead:
a systemd user unit on Linux, a launchd agent on macOS. That starts Relay at login and again if
it exits, for any harness. Once no harness goes through Relay (`relay uninstall` of the last one),
it is no longer started at login; the one running goes on for conversations still open.

## Changing the strategy

    relay strategy                 # the strategy now, and the strategies there are
    relay strategy folding         # use Context Folding from now on
    relay strategy clm --set clm_budget=40000

The running Relay picks the change up with the next request, for every conversation it serves,
including the ones in progress. Each conversation starts under the new strategy from its full
history as the harness has it. Switching back finds the earlier strategy's context again.

## Which conversations change

| When | What changes |
| --- | --- |
| Relay already set up, strategy changed | Every conversation, from its next request (one that moves Claude Code to the other path: new conversations) |
| First `relay setup` | New conversations. Codex, Claude Code and the other harnesses read their config as a conversation starts: one already open goes on as it was, without Relay, until a new one is started. `codex exec resume` starts again on every turn, so it changes from its next turn. |
| `relay uninstall <harness>` | New conversations go to the provider directly again: at once, or, while conversations that grew past the window through Relay still need it, as if direct through Relay until those are compacted (see "Checking and undoing"). One already open keeps going through Relay, which keeps running for it (until the machine restarts). |

## For a coding agent asked to set Relay up

1. Install it with the line under "Installing" (or `sh install.sh <checkout>`). This needs
   network access. Use `~/.local/bin/relay` if that folder is not on PATH.
2. If you are Claude Code and the user uses Remote Control, tell them which strategies keep it
   (see "Claude Code: its own hook, or the proxy") before choosing one that runs through the proxy.
   Run `relay setup <your harness> --strategy <name>` outside your sandbox: it writes the
   harness's config in the home directory and starts Relay. Ask the user to approve it
   (in Codex, a command that runs with escalated permissions). Run it even when the harness's
   configuration looks empty or unusual. Relay knows where each harness keeps its settings,
   records each value it changes, and `relay uninstall` puts them back.
   If your harness's sandbox or command policy refuses the installer or `relay`, and asking the
   user to approve is not possible (some refuse in headless runs, as DeepSeek Harness's sandbox
   and Gemini CLI's command checker did in our tests), tell the user to run the installer and
   `relay setup` in a terminal. Do the same if Relay stops as soon as your command ends.
   OpenClaw ends every process a command started; there, `relay setup --autostart` in a
   terminal keeps Relay running.
3. Run `relay status` and check that it reports the strategy.
4. Tell the user that the strategy applies to conversations started from now on, and that
   this one goes on as before. If Relay was already set up, the change applies from the next
   request. Do not restart the harness or its app server: that can drop its connections.

To change the strategy later, run `relay strategy <name>`. It needs no restart.

## Strategies

| Strategy | What the model sees | Settings |
| --- | --- | --- |
| `compaction` (default) | Codex's local compaction: past a threshold, a summary and the recent user messages | `compact_ratio` (0.9 of the window), `compact_threshold`, `compact_growth`, `retain_user_tokens` (20000), `compact_min_gain` (0.1) |
| `clm` | Context Language Models: the model edits its own context in a file, with its own tools | `clm_budget`, `clm_reserve` (2048), `clm_dir` (/tmp/.live_ctx, a place the harness can write), `clm_nudges` (on), `clm_guard` (on: past the budget, the oldest new tool outputs are set aside in files), `clm_steering` |
| `folding` | Context Folding: the agent branches off a sub-task, and the branch's steps fold away when it returns | the `compaction` settings, as its fallback |
| `prolong` | PRO-LONG: compaction, plus a session log the agent can search | `prolong_log` (/tmp/.prolong/log.jsonl) |
| `selfcompact` | Self-Compact: a rubric decides when the model summarizes its own trajectory | |
| `autocompact` | AutoCompact: the model decides when to compact | the `compaction` settings, as its fallback |
| `acm` | ACM: the model compresses its context when it decides to, and recalls what it compressed | `acm_dir` (/tmp/.acm) |
| `ace` | ACE: a playbook of lessons from the tasks the agent ended, kept across conversations, with compaction for the window | `ace_playbook` (`ace_playbook.md` in Relay's directory), the `compaction` settings |
| `mem1` | MEM1: the model carries its memory in a state it rewrites every turn | |
| `rlm`, `rlm_persistent` | RLM: the conversation becomes a variable in a Python REPL the model works over (runs model-written code) | `rlm_environment` (local), `rlm_max_depth` (1), `rlm_max_iterations` (30) |

The thresholds count the model's context window. Relay knows the windows of common models;
for another one, set `context_window`.

## Settings

    relay settings                 # every setting written
    relay set context_window=200000 clm_budget=40000
    relay unset clm_budget

Values are JSON where they parse as JSON (`26000`, `true`), text otherwise. Most settings apply
from the next request. Some only apply when Relay starts, and `relay set` restarts Relay
for them:

- where Relay listens: `host` (127.0.0.1), `port` (8787);
- its upstreams: `openai_base_url`, `chatgpt_base_url`, `anthropic_base_url`, `gemini_base_url`,
  and `openai_api_key`, `anthropic_api_key` for requests that carry no key;
- its store: `cache_path` (off keeps it in memory), `cache_ttl_seconds`, `cache_max_entries`,
  `cache_max_bytes`.

## Checking and undoing

    relay status                   # running?, its strategy, what keeps it running, the harnesses set up
    relay start                    # make sure Relay runs (what the harnesses' hooks run)
    relay uninstall codex          # put the harness's config back as it was (--all: every harness)

While a harness points at Relay, it reaches its model only through Relay. If a model request
fails because nothing answers at Relay's address, run `relay start`. If that does not help, read
the log (`~/.config/relay/relay.log`), or `relay uninstall` the harness.

`relay status` says one of three things (and exits with 0, 3 or 1):

- **running**: Relay answers, with its strategy.
- **unreachable**: Relay's process runs but does not answer from where the command ran. A sandbox
  that keeps commands off the network (Codex runs commands in one) does that: run it outside the
  sandbox. If Relay does not answer there either, it is stuck: end the process it names, then run
  `relay start`. `relay start`, `relay setup` and `relay set` never restart an unreachable Relay.
- **not running**: no Relay process; `relay start` starts it.

Going back to the harness on its own: `relay uninstall <harness>` (there is no stopping Relay by
hand: a harness still pointed at it would fail), not changing its endpoint by hand (its own
auto-compaction would stay off).

Relay never shortened the harness's own history, so a conversation that grew past the model's window
through Relay would fail direct. `relay uninstall` therefore restores the harness at once only when
none did; otherwise the harness leaves in two steps:

1. It stays pointed at Relay, which forwards each conversation as the harness sends it (as if
   direct) except those that would not fit, which it keeps managing. `relay uninstall` names them.
2. To keep one, run the harness's own `/compact` in it (Hermes: `/compress`): its compaction goes
   through Relay, gets Relay's context, and the harness keeps the summary as its own history, so
   the conversation fits from then on.

Once no conversation needs Relay (or a week after the last use of each), the harness's
configuration is restored by itself; `relay status` shows what is left, then that it was restored.
`relay uninstall <harness> --now` restores it at once (those conversations then fail direct);
`relay setup <harness>` keeps it on Relay. An agent running `relay uninstall` for its user: when it
ends with "Not finished", say so, and ask the user which of its choices to take (`/compact` in the
conversation it names, waiting, or `--now`) rather than stopping there. The handing back was checked for Claude Code, Codex
(`/compact` in its TUI or app), Goose, nanobot, WorkBuddy, Kilo and pi; OpenCode, Crush, Hermes,
Kimi Code, DeepSeek Harness, OpenClaw and Gemini CLI compact from their TUI only, which the checks
cannot drive; mini-swe-agent has no compaction of its own (start a new conversation).

## Other commands

    relay serve                    # run Relay in the foreground (what its daemon runs)
    relay install <harness>        # only point a harness at Relay (`--via proxy|hook`: that path)
    relay run codex -- <args>      # one run through a private, temporary Relay
