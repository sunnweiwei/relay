# Relay

Relay is transparent context management for append-only OpenAI Responses API
agent loops. Use it as a Python client wrapper or an OpenAI-compatible proxy.

## Install

```bash
pip install -e .
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

- `Compact()` / `RELAY_STRATEGY=compact`: replace the active context with one
  compacted checkpoint at `RELAY_COMPACT_THRESHOLD` (default `120000`).
- `Checkpoint()` / `RELAY_STRATEGY=checkpoint`: create chunk checkpoints at
  `RELAY_CHECKPOINT_THRESHOLD` (default `30000`) and replace old chunks after
  `RELAY_CONTEXT_THRESHOLD` (default `120000`).
- `SlidingWindow()` / `RELAY_STRATEGY=sliding_window`: keep the longest
  tool-safe suffix within `RELAY_SLIDING_WINDOW_TOKENS` (default `120000`).
- `RollingMemory()` / `RELAY_STRATEGY=rolling_memory`: recursively update a
  compact working memory and keep the newest tool-safe segment verbatim.
  Configure its updater with `RELAY_MEMORY_MODEL`,
  `RELAY_MEMORY_MAX_OUTPUT_TOKENS` (default `4000`), and
  `RELAY_MEMORY_UPDATE_INPUT_TOKENS` (default `120000`).
- `checkpoint_mode="cache"` / `RELAY_CHECKPOINT_MODE=cache`: keep artifacts in
  Relay's exact-prefix cache and leave agent responses unchanged.
- `checkpoint_mode="inline"` / `RELAY_CHECKPOINT_MODE=inline`: return Relay
  checkpoint items for the agent to append to its trajectory.

Cache mode is recommended for transparent integration. Inline checkpoints are
Relay-specific and require Relay to remain in the request path when replayed.

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
