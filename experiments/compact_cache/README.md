# Compact + Cache compatibility

The current active experiment uses a short conversation and four checks:
the Harness request reaches Relay, Relay compacts the forwarded task-model
input, the following request restores the exact saved checkpoint without
summarizing its old prefix again, and the Harness consumes the response.

Run a local scripted smoke case (no paid API request):

```bash
./.venv/bin/python -m experiments.compact_cache.run \
  --harness codex --model gpt-6-luna --mode smoke
```

The model name labels a candidate in smoke mode; the upstream is scripted.
`--mode live` now dispatches the six non-Cursor Harnesses through one entry
and one OpenAI model route. Live diagnostic results are recorded under `runs/`.
Gemini CLI and Claude Code use experimental protocol bridges for GPT.
The `local-files` live case uses two actual user messages in one restored
Harness session. Its isolated workspace contains `config/default.json` (200 ms)
and `config/production.json` (750 ms). The first message asks for the production
value and source file; the second asks for three waits (2250 ms). Harness tools
are enabled. The Relay request trace records item types, tool names and call IDs,
forwarded summary presence, cache match depth and item digests without storing
full request bodies or API credentials. Run it with:

```bash
RELAY_LIVE_TESTS=1 ./.venv/bin/python -m experiments.compact_cache.run \
  --harness pi --model gpt-6-luna --mode live --case local-files
```

The same command accepts `codex`, `opencode`, `mini-swe`, `gemini-cli`, and
`claude-code` as the Harness. The trigger threshold is calibrated between the
first model request and the first request carrying a completed tool result.
Codex and OpenCode use the committed Compact implementation. Pi, mini-SWE,
Gemini CLI and Claude Code use the experiment-only independent summary-input
budget noted below. This case allows up to eight Relay/task-model requests,
24,000 task input tokens per request, 100,000 task input tokens overall, six
summary calls, and 90,000 summary input tokens overall. It stops after the
second user turn if the Harness completes normally.
The isolated Codex configuration sets `model_reasoning_effort = "none"` for
this GPT-6 Luna case. With its previous fallback reasoning setting, Codex
rewrote prior reasoning items on `exec resume --last`, so exact-prefix Cache
missed across the two user turns. The `none` configuration produced a stable
prefix and passed; this does not validate Codex resume with reasoning enabled.
To diagnose that resume mismatch once, run the Codex `local-files` live case
with `--codex-reasoning-diagnostic`. This omits the `none` setting and saves
the complete incoming reasoning items in `runs/*/codex_reasoning_raw.json`
with mode 600. The file may contain model-private fields and stays in the
Git-ignored run directory; the printed result contains only item digests.

Gemini CLI and Claude Code now use experimental GPT tool-call bridges for this
case. The bridges map the Harness tool schemas and model tool calls; Gemini
duplicate identical tool results are deduplicated on ingress, and Claude Code's
empty optional `Read.pages` argument is omitted. These are protocol adapter
changes, not changes to `relay/strategies/compact.py`.
The planned first model is `gpt-6-luna`; the runner requires
`RELAY_LIVE_TESTS=1` and a mode-600 `/private/tmp/relay-openai-api-key` file.
Most live runs are capped at four task-model requests, 12,000 input tokens per
task request, 24,000 cumulative task input tokens, and 512 output tokens per
task request. Summary requests have a separate cap of four calls, 12,000
input tokens per call, 24,000 cumulative input tokens, and 256 output tokens
per call. The runner writes full evidence to `runs/*/result.json` and prints a
short status line. `turn_traces` records Relay ingress, task-model calls,
summary calls and cache hits per turn. `threshold_calibration` records the
largest atomic summary segment and the selected threshold. The calibration
uses the strategy's existing token-count and safe-boundary helpers without
editing `relay/strategies/compact.py`; a threshold above one safe segment does
not guarantee that a later segment with a prior summary will fit. The test
threshold is not the product's default threshold.
For Pi, mini-SWE, Gemini CLI, and Claude Code, current follow-up live runs use
an experiment-only diagnostic variant: the trigger remains between the first
two turns, while each summary request has a separate 12,000-token input cap.
These runs diagnose other Harness/Relay behavior; passing them does not prove
the committed Compact threshold policy works. See `NOTES.md` for the policy
question to discuss.
The mini-SWE diagnostic allows up to six task requests so its tool loop can
submit a final answer; Gemini CLI allows up to 36,000 cumulative task input
tokens because its own prompt uses over 8,000 tokens per turn. Per-request
limits and summary budgets remain unchanged.
Skips and unsupported protocols are not passes. The original `tests/test_codex_e2e.py` remains a repository
regression test, separate from this experiment.

Current local smoke support: Gemini CLI, Codex, OpenCode, Pi, mini-SWE, and
Claude Code. Cursor Agent is installed locally, but its `--endpoint` calls
`/auth/exchange_user_api_key` on a Cursor backend rather than Relay's managed
`/v1/responses` route. A desktop `Override OpenAI Base URL` probe used a
loopback-only fake endpoint and key. The available Grok 4.6 request completed
without reaching that endpoint; a selected GPT request was rejected by the
current Cursor account before model routing (`Free plans can only use Auto`).
The temporary desktop settings were turned off and the fake key cleared.
Cursor requires a verified model-provider route before it can enter this
smoke contract.
