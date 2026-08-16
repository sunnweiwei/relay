<p align="center">
  <img src="assets/relay.gif" alt="Relay" width="1000">
</p>

# Relay

Relay is transparent context management for append-only OpenAI Responses API
agent loops. Use it as a Python client wrapper or an OpenAI-compatible proxy.

## Install

```bash
pip install -e .
```

RLM support uses the authors' official package:

```bash
pip install -e '.[rlm]'
```

## Python

Wrap a synchronous OpenAI client and keep the rest of the agent loop unchanged:

```python
from openai import OpenAI
from relay import Checkpoint, wrap

client = wrap(
    OpenAI(),
    Checkpoint(),
    checkpoint_mode="cache",
)

trajectory = [{"role": "user", "content": "Build the project"}]
response = client.responses.create(model="your-model", input=trajectory)
trajectory.extend(response.output)
```

`wrap()` returns a client-compatible view and does not modify the original
client. Only `client.responses` is context-managed.

## Proxy

For agents that cannot accept a wrapped client, run Relay as a server:

```bash
export RELAY_STRATEGY=checkpoint
export RELAY_CHECKPOINT_MODE=cache
relay
```

Point the agent at Relay without changing its loop:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
your-agent
```

## Configuration

| Strategy | Python | `RELAY_STRATEGY` | Behavior | Configuration |
| --- | --- | --- | --- | --- |
| Compact | `Compact()` | `compact` | Replace active context with a compacted checkpoint. | `RELAY_COMPACT_THRESHOLD=120000` |
| Checkpoint | `Checkpoint()` | `checkpoint` | Create chunk checkpoints, then replace old chunks as context grows. | `RELAY_CHECKPOINT_THRESHOLD=30000`<br>`RELAY_CONTEXT_THRESHOLD=120000` |
| Sliding window | `SlidingWindow()` | `sliding_window` | Keep the longest tool-safe suffix. | `RELAY_SLIDING_WINDOW_TOKENS=120000` |
| Rolling memory | `RollingMemory()` | `rolling_memory` | Recursively update working memory while keeping the newest tool-safe segment verbatim. | `RELAY_MEMORY_MODEL`<br>`RELAY_MEMORY_MAX_OUTPUT_TOKENS=4000`<br>`RELAY_MEMORY_UPDATE_INPUT_TOKENS=120000` |
| RLM | `RLM()` | `rlm` | Run the official Recursive Language Model over the full request, then render its result as one Responses turn. | `RELAY_RLM_MODEL`<br>`RELAY_RLM_MAX_DEPTH=1`<br>`RELAY_RLM_MAX_ITERATIONS=30`<br>`RELAY_RLM_ENVIRONMENT=local`<br>`RELAY_RLM_MAX_TIMEOUT`<br>`RELAY_RLM_MAX_TOKENS` |
| Context Folding | `ContextFolding()` | `context_folding` | Hide branch control, then replace a completed branch with its return report. | `RELAY_CONTEXT_FOLDING_MODEL`<br>`RELAY_CONTEXT_FOLDING_MAX_OUTPUT_TOKENS=2000`<br>`RELAY_CONTEXT_FOLDING_MAX_BRANCH_STEPS=200`<br>`RELAY_CONTEXT_FOLDING_MAX_BRANCH_TOKENS=32768`<br>`RELAY_CONTEXT_FOLDING_MAX_BRANCHES=10` |
| AgentFold | `AgentFold()` | `agent_fold` | Maintain official-style multi-scale summaries plus one raw latest interaction. | `RELAY_AGENT_FOLD_MODEL`<br>`RELAY_AGENT_FOLD_MAX_OUTPUT_TOKENS=4000` |
| AutoCompact | `AutoCompact()` | `auto_compact` | Let a manager choose task-aware compaction points; keep the initial task and recent interactions verbatim. | `RELAY_AUTO_COMPACT_MODEL`<br>`RELAY_AUTO_COMPACT_FALLBACK_THRESHOLD=120000`<br>`RELAY_AUTO_COMPACT_KEEP_RECENT=2`<br>`RELAY_AUTO_COMPACT_MIN_INTERACTIONS=1`<br>`RELAY_AUTO_COMPACT_MAX_OUTPUT_TOKENS=4000` |
| Multi-gran Compact | `MultiGranCompact()` | `multi_gran_compact` | Fold raw context on a fixed token cadence into accumulating memory notes; a dedicated compactor picks trajectory/transition/state granularity per span. | `RELAY_MULTI_GRAN_THRESHOLD=30000`<br>`RELAY_MULTI_GRAN_MAX_COMPACTION=10`<br>`RELAY_MULTI_GRAN_MODEL`<br>`RELAY_MULTI_GRAN_BASE_URL`<br>`RELAY_MULTI_GRAN_API_KEY`<br>`RELAY_MULTI_GRAN_REASONING_EFFORT`<br>`RELAY_MULTI_GRAN_TOKENIZER=Qwen/Qwen3.5-9B`<br>`RELAY_MULTI_GRAN_TASK_PROFILE=general` |
| PRO-LONG | `ProLong()` | `prolong` | Keep a lossless structured log; a private resumable model searches it with Read/Grep/Python equivalents and supplies context to the passive task model. | `RELAY_PROLONG_MODEL`<br>`RELAY_PROLONG_CONTEXT_THRESHOLD=120000`<br>`RELAY_PROLONG_MANAGER_COMPACT_THRESHOLD=120000`<br>`RELAY_PROLONG_MAX_OUTPUT_TOKENS=4000`<br>`RELAY_PROLONG_MAX_STEPS=6`<br>`RELAY_PROLONG_ENABLE_PYTHON=true` |

| Checkpoint mode | Python | Environment | Behavior |
| --- | --- | --- | --- |
| Cache | `checkpoint_mode="cache"` | `RELAY_CHECKPOINT_MODE=cache` | Store artifacts in Relay's exact-prefix cache without changing agent responses. |
| Inline | `checkpoint_mode="inline"` | `RELAY_CHECKPOINT_MODE=inline` | Return Relay checkpoint items for the agent to append to its trajectory. |

Cache mode is recommended for transparent integration. Inline checkpoints are
Relay-specific and require Relay to remain in the request path when replayed.

RLM follows the official fresh-query behavior: every request processes the full
trajectory with `persistent=False` and `compaction=False`. It does not create or
reuse Relay checkpoints. Its manager model uses Chat Completions; the original
request model remains the Responses renderer. The default `local` RLM environment
executes model-generated Python in the Relay process; use an isolated official
RLM environment for untrusted workloads.

Context Folding follows FoldAgent's branch/return state transition, but Relay's
branch decision is hidden from the task trajectory. AgentFold follows the
official multi-scale summary update and applies each fold to the next turn.
AutoCompact follows its published inference behavior; its project currently
does not publish inference code or model weights, so Relay uses a hidden manager
for the learned compact/keep decision.

Multi-gran Compact is a port of FoldAgent's prompt-based (`pg_compaction`)
scheme. Compaction fires on a fixed token cadence: from the last sealed note,
raw interactions accumulate until a tool-safe boundary first crosses
`RELAY_MULTI_GRAN_THRESHOLD`, and that span is folded into one memory note (a
user turn written in the agent's own voice). Notes accumulate and are never
re-summarized, so the active context is `[protected prefix, note_1, ..., note_k,
raw tail]`; `RELAY_MULTI_GRAN_MAX_COMPACTION` caps the fold count and the final
note carries a commit-now footer. Unlike the other manager strategies, the
compactor is a genuinely separate endpoint speaking Chat Completions —
configured with `RELAY_MULTI_GRAN_BASE_URL` / `RELAY_MULTI_GRAN_API_KEY` /
`RELAY_MULTI_GRAN_MODEL` (mirroring FoldAgent's `--compact_base_url` /
`--compact_model_name`). When `RELAY_MULTI_GRAN_BASE_URL` is unset it inherits
the task upstream, so a model-name-only override still works. Cadence tokens are
counted with the compactor's own tokenizer (`RELAY_MULTI_GRAN_TOKENIZER`,
default Qwen3.5-9B), falling back to a length estimate when it cannot be loaded.
The compaction prompt is domain-agnostic by default; `RELAY_MULTI_GRAN_TASK_PROFILE`
selects a full built-in prompt bundle (`general` or `browsecomp`, the original
search/docid-tuned prompt) — each bundle carries its own system prompt, memory
header and commit-now footer.

PRO-LONG follows the official lossless-log design, adapted so the task model
never performs context-management actions. A private model inherits the task
context, keeps its own replayable Responses trajectory, and searches the full
structured log programmatically. Relay preserves native compaction items for
both model trajectories and stores the external log in the exact-prefix cache.
Set `RELAY_PROLONG_MODEL` to a smaller Responses model; when unset, Relay uses
the task model. `log_python` executes manager-generated local Python and should
be disabled for untrusted workloads with `RELAY_PROLONG_ENABLE_PYTHON=false`.

Sources: [PRO-LONG](https://github.com/alexisfox7/PRO-LONG),
[FoldAgent](https://github.com/sunnweiwei/FoldAgent),
[AgentFold](https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent/AgentFold),
and [AutoCompact](https://autocompact.github.io/).

## Codex

Start Relay with the API key Codex already uses:

```bash
export OPENAI_API_KEY=...
export RELAY_CHECKPOINT_MODE=cache
relay
```

Add a provider to `~/.codex/config.toml`:

```toml
model_provider = "relay"
model_auto_compact_token_limit = 1000000000

[model_providers.relay]
name = "Relay"
base_url = "http://127.0.0.1:8787/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
supports_websockets = false
```

Run Codex normally. Relay uses the Responses SSE transport and keeps its
checkpoints in the local exact-prefix cache. The high Codex compaction limit
keeps Codex's own compactor from replacing the append-only trajectory first.