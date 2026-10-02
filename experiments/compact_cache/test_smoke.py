"""One two-turn Compact + Cache contract for each available Harness."""
from __future__ import annotations

import pytest

from experiments.compact_cache.checks import HARNESSES, assert_contract
from experiments.compact_cache.harnesses import RUNNERS


@pytest.mark.parametrize("harness", HARNESSES)
def test_smoke(harness: str) -> None:
    runner = RUNNERS.get(harness)
    if runner is None:
        pytest.skip(f"{harness}: no verified Relay protocol adapter")
    cache, summaries, raw, forwarded, marker = runner()
    assert_contract(cache, summaries, raw, forwarded, latest_marker=marker)
