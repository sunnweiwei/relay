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
- `checkpoint_mode="cache"` / `RELAY_CHECKPOINT_MODE=cache`: keep artifacts in
  Relay's exact-prefix cache and leave agent responses unchanged.
- `checkpoint_mode="inline"` / `RELAY_CHECKPOINT_MODE=inline`: return Relay
  checkpoint items for the agent to append to its trajectory.

Cache mode is recommended for transparent integration. Inline checkpoints are
Relay-specific and require Relay to remain in the request path when replayed.
