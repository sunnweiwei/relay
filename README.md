<p align="center">
  <img src="assets/relay.gif" alt="Relay" width="1000">
</p>

# Relay

Relay is a context-management layer between agent harnesses and model APIs. It
runs as a local proxy: the harness keeps its normal append-only loop and sends its
whole history on every request; Relay decides what the model actually sees and
forwards that instead. One strategy written against Relay's protocol-neutral view
therefore works for every supported harness and model.

The first strategy is **Codex-style context compaction**: when the prompt reaches a
threshold, the history is summarized with Codex's own compaction prompt and replaced
by the initial context, the most recent user messages, and the summary.

## Quickstart

```bash
pip install -e .
relay serve &              # http://127.0.0.1:8787
relay install codex        # or: claude, gemini_cli, pi, opencode, kilo, crush,
                           #     openclaw, kimi_code, goose
codex                      # use the harness exactly as before
relay uninstall codex      # restore its configuration
```

`relay install` edits the harness's own configuration file. It wraps whatever endpoint
the harness is configured with (or its default): the setting becomes
`http://127.0.0.1:8787/up/<id><original path>`, and Relay forwards `/up/<id>/...` to the
original host. Logins, API keys, gateways and provider choices keep working; previous
values are recorded in `~/.config/relay/installed.json` for `relay uninstall`.

| Harness | Setting written by `relay install` |
| --- | --- |
| Codex | `openai_base_url` in `~/.codex/config.toml` (built-in provider; ChatGPT login or API key) |
| Claude Code | `env.ANTHROPIC_BASE_URL` in `~/.claude/settings.json` |
| Gemini CLI | `GOOGLE_GEMINI_BASE_URL` in `~/.gemini/.env` (API-key mode) |
| pi | `providers.{openai,anthropic,google}.baseUrl` in `~/.pi/agent/models.json` |
| OpenCode / Kilo | `provider.{openai,anthropic,google}.options.baseURL` in `opencode.json` / `kilo/config.json`, plus `compaction.prune = false` |
| Crush | `providers.<id>.base_url` in `~/.config/crush/crush.json` |
| OpenClaw | `models.providers.{openai,anthropic,google}.baseUrl` in `~/.openclaw/openclaw.json` |
| Kimi Code | `providers.<id>.base_url` in `~/.kimi-code/config.toml` |
| Goose | `OPENAI_HOST` / `ANTHROPIC_HOST` / `GOOGLE_HOST` in `~/.config/goose/config.yaml` |
| Hermes Agent | `model.base_url` in `~/.hermes/config.yaml`, plus the protocol Hermes would have picked from the original host (`api_mode: codex_responses` for api.openai.com; Gemini's OpenAI-compatible endpoint for Google) |
| nanobot | `providers.{openai,gemini,anthropic}.apiBase` in `~/.nanobot/config.json`, plus `apiType: responses` for a provider that was talking to api.openai.com |
| DeepSeek Harness | `llm-deepseek` `baseURL` and existing `llm-pi-ai` routes in `~/.dsh/cordis.patch.yml` |
| WorkBuddy / CodeBuddy Code | `url` of each custom model in `~/.workbuddy/models.json` / `~/.codebuddy/models.json` |
| mini-swe-agent | `OPENAI_BASE_URL` / `GEMINI_API_BASE` / `ANTHROPIC_BASE_URL` in its global `.env` (read by litellm) |

`relay run <harness> -- <args>` runs Codex or Claude Code through a private, temporary
Relay instead. `relay install` also turns the harness's own auto-compaction off, so Relay's
strategy is the one that runs (see [Harness auto-compaction](#harness-auto-compaction));
`relay uninstall` restores it.

## Architecture

```
 harness request ─► codec (protocol) ─► harness profile ─► View ─► strategy ─► Rewrite
                    items, legal cuts    injected context,   the conversation   cut + head
                                         identity, state     only
 upstream ◄─ codec (legal placement) ◄─ harness profile (state placed into the head) ◄─┘
```

Three layers keep strategies free of harness and protocol details:

- the **codec** knows the wire protocol: items, legal cuts, how a message is written, and
  where the protocol allows system messages;
- the **harness profile** knows what the harness writes itself: which items are injected
  context (`refine`), what makes two requests the same conversation for the prefix store
  (`identity`, robust to instructions re-rendered in place and resumed histories), the
  harness's current state (`state`: its instructions, environment, modes at their latest
  version, and the history items that state supersedes), where that state goes in a
  rewritten history (`place`), and which requests are the harness compacting itself
  (`compacting`: they get the stored compaction but never a new one);
- the **strategy** sees only the conversation (user, assistant, tool items and summaries)
  and decides what to keep and what to write. Every rewrite gets the harness's current
  state placed into it, so a new strategy inherits that, and cache hits, for free.

| Module | Responsibility |
| --- | --- |
| `relay/core/ir.py` | The protocol-neutral `Item` / `View` / `Rewrite` a strategy works with. Items keep a reference to their wire item, so whatever a strategy keeps is forwarded byte-for-byte. |
| `relay/protocols/` | One codec per wire protocol (OpenAI Responses, OpenAI Chat Completions, Anthropic Messages, Gemini): item kinds and text, legal cut points (never between a tool call and its result), prefix canonicalization, synthetic messages, summary requests, usage and overflow errors. |
| `relay/harnesses/` | Harness profiles: injected context, conversation identity, current state and its placement, the harness's own compaction requests, detection from request headers, and the settings `relay install` writes. |
| `relay/install.py` | `relay install` / `uninstall`: reversible edits of JSON, TOML, YAML and `.env` config files, and the endpoint mounts Relay forwards. |
| `relay/strategies/` | Strategies. `compaction.py` ports Codex's local compaction. |
| `relay/core/engine.py` | Per-request orchestration: restore the stored rewrite, estimate tokens, plan on the conversation, place the harness's state, remember. It records each request's cache decision (matched depth) and warns when a request no longer reaches its conversation's last compaction, with the item where it diverged. Strategy failures never reach the harness; the request is forwarded with the last good rewrite. |
| `relay/core/store.py` | Exact-prefix store (a trie over item identities): a rewrite computed for one request is found again by every later request that extends the same history, including other conversations forked from it (sub-agents). Relay keeps no other session state. |
| `relay/providers.py` | Upstream routing (installed mounts, else by path) and model context windows. |
| `relay/transport/proxy.py` | The HTTP proxy. Responses, including streams, are relayed unchanged. |

Relay never translates between protocols: a request is forwarded in the protocol the
harness spoke. Models are reached through any provider that serves that protocol.

Token counts come from the usage the upstream reported for the previous request of the
same conversation plus an estimate of the new items, so no extra token-counting calls are
made. The estimate (all of it when nothing was reported yet: a conversation's first
request, or after Relay joined late, restarted, or the harness compacted itself) counts
four bytes of text per token, as Codex does (OpenAI and Gemini tokenizers measured about
4.6, so it errs high), or 2.5 for Claude (measured 2.7), and counts opaque content the
model still reads (encrypted reasoning and compactions, redacted thinking, thought
signatures) at its decoded size. If the upstream still rejects a
request as too long, Relay compacts and retries it once.

## Compaction

Behavior follows `codex-rs/core/src/compact.rs`; prompts are vendored verbatim in
`relay/prompts/`.

- Compaction triggers at 90% of the model's context window (`RELAY_COMPACT_THRESHOLD` or
  `RELAY_COMPACT_GROWTH` override this), and always at 95% of the window.
- The summary request is the harness's own request with the history so far and
  Codex's compaction prompt as the last user message, sent to the same upstream (with
  tools disabled). It shares the previous request's prefix, so provider prompt caching
  applies. Like Codex's `drain_to_completed`, a summary counts only if its response
  completed: a stream cut short or a response stopped by a token limit is retried (up to
  5 times) and never used. A history too long for one summary request (Relay joined a
  long session late, dropped out for a while, or restarted and lost its state, while the
  harness kept and resends everything) is summarized in order, a piece at a time, each
  piece after the summary so far: the skipped compactions, caught up. The same happens if
  the upstream reports an overflow.
- The new context follows Codex's replacement history: the newest real user messages up
  to `RELAY_RETAIN_USER_TOKENS` (the oldest one kept is truncated in the middle; media is
  reduced to text; earlier summaries are dropped) and the summary as a user message
  starting with Codex's summary prefix. Mid-turn (the request ends with tool results)
  everything is summarized, the initial context sits just above the last real user
  message and the summary comes last. At the start of a turn the summary covers the
  history before it, the initial context is re-injected after the summary, and the new
  turn follows verbatim.
- The initial context is what the harness injected before the model's first action:
  system messages and context messages (environment, instructions, `<system-reminder>`
  blocks). Like Codex, which rebuilds it from session state, Relay takes it from the
  harness's own history, which is resent with every request, so it survives any number of
  compactions. System messages (Codex's base instructions) stay first, except on the
  Anthropic API, where a `system` message may only precede the model's turn: there it
  follows a mid-turn summary and waits for the next mid-turn compaction after a turn start.

### Harness state

Harnesses tell the model about their state (instruction files, environment, date, modes,
memory, MCP servers) in three ways, and each needs a different treatment for the compacted
context to describe the session as it is now:

| Where the state is | Harnesses | What Relay does |
| --- | --- | --- |
| Fields sent with every request (top-level system prompt, instructions, tools) | Hermes Agent, nanobot, mini-swe-agent; Claude Code's and Gemini CLI's system prompts | Nothing to do: these are forwarded as sent. |
| A first message re-rendered in place | OpenCode, Kilo, Crush, OpenClaw, Goose (system message with AGENTS.md / SOUL.md / `.goosehints`); Gemini CLI (`<session_context>` in the first user message); WorkBuddy (rules and memory in the first user message) | Prefix matching compares those items by position and user messages without the injected blocks (`Harness.identity`), so the compaction survives the change; the current version is what gets forwarded. |
| Updates appended to the history | Codex (world-state sections); pi (a new system prompt on resume); Kimi Code (date and mode reminders); DeepSeek Harness (runtime snapshots, changed instruction files); Claude Code (MCP instructions as system messages) | The profile re-renders the initial context at compaction: Codex merges its section updates like Codex does; the others name each piece of state (`Harness.state_key`) and the latest version of each is re-injected. |

Claude Code's other updates (instruction files re-read on resume, reminders) arrive inside
user messages, which compaction keeps. Notifications (background tasks finishing) are
events, not state, and are left to the summary.

Sub-agents and multi-agent sessions need no special case: every sub-agent seen (Codex,
Gemini CLI, OpenCode, Kilo, Crush, OpenClaw, Goose, Kimi Code, DeepSeek Harness, WorkBuddy,
Hermes) runs its own conversation through the same Relay, and the prefix store keeps one
compaction per conversation, matched by its own history. A sub-agent forked from its
parent's history (Codex `fork_turns`) extends the parent's prefix and so starts from the
parent's compaction.

Deliberate differences from Codex: a compaction that would free less than
`RELAY_COMPACT_MIN_GAIN` of the threshold is skipped, so a threshold set close to the
fixed prompt overhead (system prompt and tools) does not trigger a summary on every
request; the summary request keeps the tool definitions with tool calls disabled
(Anthropic rejects tool history without them) where Codex sends none; a history too long
for one summary request is summarized piece by piece instead of dropping its oldest items;
and an empty summary is a failed compaction rather than "(no summary available)".

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `RELAY_COMPACT_THRESHOLD` | `RELAY_COMPACT_RATIO` × context window | Prompt tokens that trigger compaction. |
| `RELAY_COMPACT_RATIO` | `0.9` | Share of the model's context window used when no threshold is set. |
| `RELAY_COMPACT_GROWTH` | unset | Instead of a threshold, compact after this many tokens were added since the context window began (Codex's `BodyAfterPrefix` scope); independent of each harness's fixed prompt overhead. |
| `RELAY_CONTEXT_WINDOW` | from the model name | Overrides the context window table in `relay/providers.py`. |
| `RELAY_RETAIN_USER_TOKENS` | `20000` | Budget for recent user messages kept verbatim. |
| `RELAY_COMPACT_MIN_GAIN` | `0.1` | Skip compactions that free less than this share of the threshold (or growth). |
| `RELAY_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Upstream for the Responses API. |
| `RELAY_ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Upstream for the Messages API. |
| `RELAY_OPENAI_API_KEY`, `RELAY_ANTHROPIC_API_KEY` | unset | Replace the harness's credentials for that upstream. |
| `RELAY_HARNESS` | detected | Force a harness profile by name, e.g. `codex`, `hermes`, `generic`. |
| `RELAY_EVENT_LOG` | unset | Append one JSON line per compaction (sizes, timing, the new head). |
| `RELAY_HOST`, `RELAY_PORT` | `127.0.0.1`, `8787` | Server address for `relay serve`. |
| `RELAY_CACHE_MAX_ENTRIES`, `RELAY_CACHE_MAX_BYTES`, `RELAY_CACHE_TTL_SECONDS`, `RELAY_CACHE_SECRET` | `4096`, 256 MiB, 6 h, random | Limits of the in-memory prefix store. |

Set `RELAY_CONTEXT_WINDOW` to the window the harness itself uses (Codex's catalog gives the
gpt-5.6 and gpt-6 families 272k by default, shown as 258k usable): the table covers older
models only, an unknown model falls back to 128k, and the 95% limit uses the same window, so
setting only a threshold is not enough. Relay's threshold must also stay below the
harness's own trigger (see [Harness auto-compaction](#harness-auto-compaction)).

Other providers are configured by endpoint, for example
`RELAY_OPENAI_BASE_URL=https://api.x.ai/v1` for Grok, or
`RELAY_OPENAI_BASE_URL=https://openrouter.ai/api/v1` /
`RELAY_ANTHROPIC_BASE_URL=https://openrouter.ai/api` with the matching `*_API_KEY` for
OpenRouter.

## Support

Checked in Docker against real models with `tests/docker/check.py`: a two-turn session
compacting every 5k new tokens, so each run compacts at least twice mid-turn and once at
a turn start, with every compaction checked (summary content, Codex layout, initial
context kept, latest instructions forwarded; see [Development](#development)). Between
the turns the project's instruction files change, and turn 2 hands work to a sub-agent
where the harness has one:

| Harness | Protocol | GPT | Gemini | Claude |
| --- | --- | --- | --- | --- |
| Codex 0.160 | Responses | ✅ ChatGPT login | | |
| Claude Code 2.1.283 | Messages | ✅ via LiteLLM² | | ✅ subscription login |
| Gemini CLI 0.62 | Gemini | ✅ via LiteLLM² | ✅ | |
| pi 1.0 | Responses / Gemini | ✅ | ✅ | |
| OpenCode 1.18 | Responses / Gemini | ✅ | ⚠️¹ | |
| Kilo CLI 7.8 | Responses / Gemini | ✅ | ✅ | |
| Crush 0.97 | Chat Completions / Gemini | ✅ | ✅ | |
| OpenClaw 2026.9 | Responses / Gemini | ✅ | ✅ | |
| Goose 1.52 | Responses / Gemini | ✅ | ✅ | |
| Kimi Code 2.1 | Responses / Gemini | ✅ | ✅ (gemini-2.5-flash-lite) | |
| Hermes Agent 0.19 | Responses / Chat Completions | ✅ | ✅ | |
| nanobot 0.3.5 | Responses / Chat Completions | ✅ | ✅ | |
| DeepSeek Harness 0.2 | Responses / Gemini | ✅ | ✅ | |
| WorkBuddy (CodeBuddy Code 2.161) | Chat Completions | ✅³ | ✅ | |
| mini-swe-agent 2.4 | Responses / Gemini | ✅⁴ | ✅⁴ | |

Models: `gpt-6-luna`, `gemini-3.8-flash`, `claude-sonnet-5-5` unless noted. Some
harness/model pairs fail without Relay too and were tested with the closest model the
harness supports: Kimi Code requests a thinking level gemini-3.8-flash rejects.
¹ Every compaction passes the checks, but when Gemini answers with nothing but a thought,
OpenCode sends a request that ends with that model turn, which Gemini rejects; the same
task fails the same way without Relay, so the run does not finish.
² The harness was already configured to use a LiteLLM gateway that translates its protocol to
OpenAI; `relay install` wrapped that gateway (harness → Relay → LiteLLM → OpenAI). Relay does
not translate protocols itself.
³ WorkBuddy is a desktop app; its engine ships as the CodeBuddy Code CLI, which was run on
`~/.workbuddy`. Built-in models go through Tencent's service and cannot be managed; custom
models can. gpt-6 accepts tools on Chat Completions only with `reasoning_effort: none`, so
thinking was turned off; nanobot likewise needs a reasoning effort set to stop sending
`temperature` to gpt-6.

⁴ One turn: mini cannot continue a session, so the turn-start path is not exercised.

Resumed sessions keep their compaction: several harnesses rewrite their history when a
session is resumed without changing what the model reads (Codex writes absent reasoning
content as null, Gemini CLI repeats tool results and replaces thought signatures, OpenClaw
drops reasoning summaries, Hermes drops tool names, CodeBuddy adds its own metadata), and
the codecs compare items by what reaches the model, so the stored compaction still
applies. Claude Code can leave earlier tool calls out of a resumed history; that is a
different history, so Relay summarizes it again.

Hermes Agent and nanobot choose their protocol from the endpoint's host name, which is why
`relay install` pins it for them. Hermes ignores Anthropic endpoints on hosts it does not
recognize, so its native Anthropic provider is not supported. DeepSeek models on DeepSeek
Harness were not checked (no key), but use the same Messages codec as Claude Code.

Cursor, Windsurf, Amp and similar products route model requests through their own
servers, so a local proxy cannot manage their context. (Cursor Agent's `--endpoint`
exchanges the API key with Cursor's backend first, and the desktop app's OpenAI base-URL
override did not reach a local endpoint.)

Beyond the matrix: `--subagent` (a sub-agent reads three files, so its own conversation
compacts) passed for Codex, Claude Code, OpenCode, Kilo, Crush, Goose, Kimi Code, DeepSeek
Harness and WorkBuddy; `--window` (Relay's own 90% trigger on a small window) for pi, Codex
and Claude Code; `--native-compact` (Claude Code's own `/compact` between the turns, whose
history then starts with Claude Code's summary) for Claude Code. Hermes Agent's background
sub-agents do not run under `hermes chat -q`, which exits with the turn. Kimi Code and Hermes
Agent keep the instructions they loaded when a session is resumed (the request itself still
carries the old version), so they answer with the old codename;
`gemini-2.5-flash-lite` on Kimi Code sometimes re-reads files and writes summaries that miss
facts, with or without earlier summaries in view.

Realistic sessions (`tests/docker/session.py`): a real repository (more-itertools with a
planted bug) and four turns over the harness's own resume: a tour of the code, a bug fix
with a regression test, a feature after a changelog rule is added to the instructions, and
a summary of the session; hidden checks then test the fix (beyond the repository's own
tests), the feature, its stub, tests and changelog entry, the full suite, and that the
summary recalls both changes. Every harness that resumes a session passed with compaction
on (WorkBuddy with plan mode off: run headless, it waits for a plan approval that cannot
come, with or without Relay), as did Codex and
Claude Code with Relay joining late, dropping out for a turn, or restarting before every
turn (histories up to five times the window were caught up in pieces); with the tickets
coming from an MCP server, a mockup image to look at and a release to look up; with the
session forked after turn 3 and, in Claude Code, rewound to the end of turn 2 (each branch
found the compaction of the history it shares); and with the harness's own compaction
firing first (Codex's server-side `compaction_trigger`, Claude Code's `/compact`).

Remote Control: Codex's (`codex app-server --remote-control`, driven from the ChatGPT app)
reaches its service at `chatgpt_base_url` and the model at `openai_base_url`, so it works
through Relay. `check.py codex:remote` runs the session through `tests/docker/codex_remote.py`,
a stand-in for the service and the app (nothing is enrolled with the account; the rest of
the ChatGPT backend is passed through to chatgpt.com); it passed, and the app shows the full
conversation, its context meter the compacted size. Claude Code's (`claude remote-control`)
refuses to start while `ANTHROPIC_BASE_URL` points anywhere but api.anthropic.com, so it
does not run through Relay as installed; sessions that an already-running one spawns read
`settings.json` afresh, so `relay install` sends them through Relay.

Harness-side context management changes history Relay has stored: OpenCode prunes old
tool outputs, which `relay install` turns off, and Gemini CLI masks them, which no setting
stops, so its profile compares tool results by their call and the prefix still matches.

### Harness auto-compaction

Left on, a harness's own compaction runs first wherever its trigger is at or below Relay's
(Codex compacts before the request that would cross a tie), or where it measures its own
history, which keeps growing behind Relay (Hermes Agent); the session still goes on, but
with the harness's summaries instead of Relay's. `relay install` turns it off, through a
switch where there is one and otherwise by moving the trigger out of reach (10⁹ tokens):

| Harness | Its own trigger | `relay install` sets |
| --- | --- | --- |
| Codex | `model_auto_compact_token_limit`, capped at 90% of `model_context_window` (server-side `compaction_trigger`) | both to 10⁹; Codex caps the window at the model's maximum, so its trigger moves to 90% of that: 784.8k for the gpt-6 family, but 244.8k for gpt-5.5, whose maximum is its default (keep Relay's threshold below it) |
| Claude Code | near the window | `env.DISABLE_AUTO_COMPACT=1` in `settings.json` (`/compact` still works) |
| WorkBuddy, CodeBuddy | forced at 92% of a model's `contextWindow` or `maxInputTokens` (this build runs no other) | `autoCompactEnabled: false`, `env.CODEBUDDY_AUTOCOMPACT_PCT_OVERRIDE=100` |
| Gemini CLI | 50% of the window | `model.compressionThreshold: 10⁹` |
| pi | the window minus a reserve | `compaction.enabled: false` |
| OpenCode, Kilo | the input limit minus a reserve | `compaction.auto: false` (and `compaction.prune: false`) |
| Crush | near the window | `options.disable_auto_summarize: true` |
| OpenClaw | the window minus a reserve | `agents.defaults.compaction.enabled: false` (overflow recovery and `/compact` stay) |
| Goose | 80% of the context limit | `GOOSE_AUTO_COMPACT_THRESHOLD: 0.99` (no switch; values outside (0, 1) may mean the default) |
| Kimi Code | 85% of a model's `max_context_size`, or within 50k of it | every model's `max_context_size: 10⁹` (it must be positive) |
| Hermes Agent | half its window (at least 75% below 512k), measured on its own history | `compression.enabled: false` |
| nanobot | `contextWindowTokens` | `agents.defaults.contextWindowTokens: 10⁹` (0 would starve its memory archive) |
| DeepSeek Harness | `compaction-basic` at 80%, with tool results trimmed first | `compaction-basic` `auto: false`, `tool-result-pruner` `thresholdChars: 10⁹` |
| mini-swe-agent | none | nothing |

Each was checked in Docker (`check.py --harness-compaction`): with the user's configuration
set to make the harness compact itself at about 20k tokens, it did; after `relay install`
it did not, and the task still finished. Hermes Agent, which refuses windows under 64k, was
checked in the realistic session, where it had compacted itself before.

Codex measures its context as the usage reported for its last request plus what it has added
since; that request is the one Relay compacted, so its trigger stays out of reach even when
its own history passes it. Checked with Codex's trigger at 25k and Relay's threshold at 16k:
Codex's history grew to 34k, the usage it was told stayed under 22k, and it never compacted
itself (in `codex exec` and under Remote Control); with Relay not compacting, it did.

Known limitations: state lives in memory, so a restart costs a catch-up summary per
conversation; WebSocket transports are refused (Codex falls back to HTTP); an overflow
reported inside an already-started stream is not retried; token-counting endpoints are
passed through unchanged; Gemini CLI's Google-login mode (Code Assist API) is not
covered yet.

## Extending Relay

- **Harness**: subclass `Harness` in `relay/harnesses/`: `matches` (request headers),
  `refine` and `injected` (what it writes into the conversation), and, where the defaults do
  not fit, `identity`, `state_key` or `state`, `place`, `compacting`; then `settings` for
  `relay install`. Add it to `HARNESSES`. Run `tests/docker/check.py --probe` to see where it
  keeps state, then a two-turn check, and turn the run into a replay fixture
  (`python -m tests.replay.build NAME HOME`): `tests/test_replay.py` is the contract every
  harness must meet (every request finds its stored compaction, the latest state reaches
  the model, the request stays well-formed).
- **Protocol**: implement the `Codec` interface in `relay/protocols/base.py` and register
  it in `CODECS`; add an upstream for it in `relay/providers.py`.
- **Strategy**: implement `Strategy` (`fingerprint` and `plan(view, summarizer)`), returning
  a `Rewrite` that replaces `view.items[:cut]` with `head`. The view holds the conversation
  only; never put harness context into `head`, the profile places the current state.

The earlier strategies (checkpoint, sliding window, rolling memory, RLM, context
folding, AgentFold, AutoCompact, selective discard, PRO-LONG, multi-granularity
compaction) live unchanged in `relay/experimental/` until they are ported to this
interface.

## Development

```bash
pip install -e '.[dev]'
pytest                                  # unit, fake-upstream and replay tests (real sessions of every harness)
```

Real harnesses are checked in Docker against real models, with a low compaction
threshold so a short task compacts several times. The image holds the harness CLIs;
each run gets a throwaway HOME, so the host's own configurations are never touched.

```bash
docker build -t relay-harness-test tests/docker
RELAY_TEST_KEYS=keys.env python tests/docker/check.py pi:gpt          # one harness × model
RELAY_TEST_KEYS=keys.env python tests/docker/check.py --matrix        # all of them
```

`keys.env` holds `OPENAI_API_KEY`, `GEMINI_API_KEY` and `ANTHROPIC_API_KEY`; Codex uses
a copy of the host's ChatGPT login with its refresh token voided, so a container can never
rotate the token the host holds. Each run is two turns of one session (the second through the
harness's own continue/resume), so Relay compacts mid-turn and at the start of a turn.
A run passes only if, for every compaction, the summary contains every code read so far,
the new context has Codex's layout and keeps the initial context (at its latest state) and
user messages, and every later request of that conversation that Relay forwarded carries
just its latest summary and none of the summarized tool output; the harness's latest
instructions reach the model; no conversation read a file twice; both answers are right;
no compaction failed; and the upstream accepted every model call. `--probe` records a
session without compacting, to see where a harness keeps its state.

```bash
RELAY_TEST_KEYS=keys.env python tests/docker/session.py codex:gpt-api --repo PATH --window 32000
    # --baseline | --transition late-join|drop-out|restart | --tools | --branch | --harness-compact N
```

runs the realistic session described under [Support](#support) and
prints its hidden checks with Relay's record of it (compactions per turn, summary requests
per compaction, cache hits, divergences, the harness's own compactions).
