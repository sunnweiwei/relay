from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from relay import AgentFold, AutoCompact, ContextFolding, PrefixCheckpointCache
from relay.proxy import ProxyConfig, create_app
from tests.test_codex_e2e import _FakeResponsesUpstream, _serve


@unittest.skipUnless(find_spec("minisweagent"), "mini-swe-agent is not installed")
class MiniSWEAgentEndToEndTests(unittest.TestCase):
    def test_all_adaptive_strategies_run_through_the_responses_harness(self) -> None:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_response_model import LitellmResponseModel

        strategies = [
            ContextFolding(),
            AgentFold(),
            AutoCompact(fallback_threshold=1_000_000),
        ]
        for strategy in strategies:
            with self.subTest(strategy=strategy.name):
                upstream = _FakeResponsesUpstream(mini_mode=True)
                cache = PrefixCheckpointCache(
                    secret=f"mini-{strategy.name}-test".encode()
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
                        result = agent.run("Relay compatibility smoke test")

                self.assertEqual(result["exit_status"], "Submitted")
                self.assertIn("relay-mini-ok", result["submission"])
                self.assertEqual(len(upstream.main_requests), 3)
                self.assertEqual(len(upstream.manager_requests), 2)
                self.assertGreaterEqual(cache.stats().hits, 1)
                for body in upstream.main_requests:
                    self.assertFalse(
                        any(
                            item.get("type") == "compaction"
                            for item in body.get("input", [])
                        )
                    )


if __name__ == "__main__":
    unittest.main()
