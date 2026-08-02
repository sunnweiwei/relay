# relay

An OpenAI-compatible reverse proxy for context management in append-only Responses API agent loops.

It intercepts `/v1/responses`, applies a configurable context policy, and forwards the transformed request to OpenAI while keeping the agent loop unchanged.

```bash
pip install -e .
relay
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```
