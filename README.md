# Relay

Relay is a transparent middleware for LLM context management. It intercepts model requests and responses to optimize context while remaining invisible to both the agent and the model provider.

Agents keep their normal append-only loop while Relay applies Codex-compatible context compaction behind an OpenAI-compatible `/v1/responses` endpoint.

```bash
pip install -e .
relay
```

Point the agent's OpenAI base URL to `http://127.0.0.1:8787/v1`.

Relay checkpoints use the Responses compaction-item shape but are Relay-specific; keep Relay in the request path when replaying them.
