# Relay

Relay is a transparent middleware for LLM context management. It intercepts model requests and responses to optimize context while remaining invisible to both the agent and the model provider.

Agents keep their normal append-only loop while Relay applies Codex-compatible context compaction behind an OpenAI-compatible `/v1/responses` endpoint.

```bash
pip install -e .
relay
```

Point the agent's OpenAI base URL to `http://127.0.0.1:8787/v1`.

Relay checkpoints use the Responses compaction-item shape but are Relay-specific; keep Relay in the request path when replaying them.

Set `RELAY_CHECKPOINT_MODE=cache` to keep checkpoints in Relay's process-local,
tenant-partitioned prefix cache instead of returning compaction items. Cache mode
requires a distinct Bearer credential for each tenant; cache misses rebuild from
the full append-only trajectory.

Set `RELAY_STRATEGY=checkpoint` for delayed hierarchical compaction. It creates
chunk checkpoints at `RELAY_CHECKPOINT_THRESHOLD` without changing the active
context, replaces oldest chunks after `RELAY_CONTEXT_THRESHOLD`, and recursively
merges checkpoint chunks when they reach the checkpoint threshold.
