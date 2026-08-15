from __future__ import annotations

import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from relay import MultiGranCompact, PrefixCheckpointCache
from relay.proxy import ProxyConfig, create_app
from relay.strategies.multi_gran_compact import MEMORY_HEADER
from tests.test_codex_e2e import _FakeResponsesUpstream, _serve

# A live compactor endpoint (vLLM/OpenAI chat.completions) must be configured for this
# end-to-end test; without it there is nothing real to exercise, so the test skips.
COMPACT_BASE_URL = os.getenv("RELAY_MULTI_GRAN_BASE_URL")


@unittest.skipUnless(find_spec("minisweagent"), "mini-swe-agent is not installed")
@unittest.skipUnless(
    COMPACT_BASE_URL,
    "set RELAY_MULTI_GRAN_BASE_URL (+ _MODEL/_API_KEY) to a live compactor endpoint",
)
class MultiGranCompactMiniSWEEndToEndTests(unittest.TestCase):
    """Drive a real mini-swe-agent loop through Relay with a REAL compactor.

    The task model is the fake Responses upstream (scripted 3-turn bash-and-submit run),
    but MultiGranCompact's compactor points at a genuinely separate endpoint —
    ``RELAY_MULTI_GRAN_BASE_URL`` / ``_MODEL`` / ``_API_KEY`` (e.g. a local vLLM Qwen) —
    speaking Chat Completions. Cadence tokens are counted with the real compactor
    tokenizer (``RELAY_MULTI_GRAN_TOKENIZER``, default Qwen3.5-9B), so this exercises the
    true fold path: real note generation, real token counting, real prompt.

    Run it, for example, with::

        export RELAY_MULTI_GRAN_BASE_URL=http://localhost:8018/v1
        export RELAY_MULTI_GRAN_MODEL=Qwen/Qwen3.5-9B
        export RELAY_MULTI_GRAN_API_KEY=dummy
        python -m unittest tests.test_multi_gran_compact_e2e -v

    The fold threshold defaults to a tiny 20 tokens so a fold is guaranteed within the
    3-turn run; override with ``RELAY_MULTI_GRAN_THRESHOLD`` to test a realistic cadence.
    """

    def test_multi_gran_compact_runs_with_a_real_compactor(self) -> None:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_response_model import LitellmResponseModel

        upstream = _FakeResponsesUpstream(mini_mode=True)
        cache = PrefixCheckpointCache(secret=b"mini-multi-gran-real-test")
        strategy = MultiGranCompact(
            compact_threshold=int(os.getenv("RELAY_MULTI_GRAN_THRESHOLD", "20")),
            max_compaction=int(os.getenv("RELAY_MULTI_GRAN_MAX_COMPACTION", "10")),
            compact_base_url=COMPACT_BASE_URL,
            compact_model=os.getenv("RELAY_MULTI_GRAN_MODEL"),
            compact_api_key=os.getenv("RELAY_MULTI_GRAN_API_KEY") or "dummy",
            reasoning_effort=os.getenv("RELAY_MULTI_GRAN_REASONING_EFFORT") or None,
            tokenizer_name=os.getenv("RELAY_MULTI_GRAN_TOKENIZER", "Qwen/Qwen3.5-9B"),
            verbose=True,
        )

        with _serve(upstream.app) as upstream_url:
            relay = create_app(
                strategy,
                ProxyConfig(
                    upstream_base_url=f"{upstream_url}/v1",
                    upstream_api_key="upstream-test-key",
                    checkpoint_mode="cache",
                ),
                checkpoint_cache=cache,
            )
            with _serve(relay) as relay_url, tempfile.TemporaryDirectory() as root:
                model = LitellmResponseModel(
                    model_name="openai/relay-test-model",
                    model_kwargs={
                        "api_base": f"{relay_url}/v1",
                        "api_key": "tenant-a",
                    },
                    cost_tracking="ignore_errors",
                )
                agent = DefaultAgent(
                    model,
                    LocalEnvironment(cwd=str(Path(root))),
                    system_template="You are a coding agent. Always call bash.",
                    instance_template="Solve: {{task}}",
                    step_limit=5,
                    cost_limit=0,
                )
                result = agent.run("Relay multi-gran compact real-compactor smoke test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertIn("relay-mini-ok", result["submission"])
        self.assertEqual(len(upstream.main_requests), 3)
        # A folded memory note (real compactor output, wrapped in the memory header)
        # replaced raw context in what was forwarded to the task upstream.
        note_items = [
            item
            for body in upstream.main_requests
            for item in body.get("input", [])
            if isinstance(item, dict)
            and str(item.get("content", "")).startswith(MEMORY_HEADER)
        ]
        self.assertTrue(note_items, "expected at least one folded memory note upstream")
        # The note carries real content beyond the header (the compactor actually wrote something).
        self.assertTrue(
            any(len(item["content"]) > len(MEMORY_HEADER) + 1 for item in note_items)
        )
        # Transparent cache mode: no compaction items leak into the agent-visible stream.
        for body in upstream.main_requests:
            self.assertFalse(
                any(
                    item.get("type") == "compaction"
                    for item in body.get("input", [])
                )
            )
        self.assertGreaterEqual(cache.stats().hits, 1)


if __name__ == "__main__":
    unittest.main()
