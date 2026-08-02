from __future__ import annotations

import os
from typing import Callable

from .agent_fold import AgentFold
from .base import ContextStrategy
from .full_history import FullHistory
from .native_compaction import NativeCompaction
from .openai_truncation import OpenAITruncation
from .rollback_folding import RollbackFolding
from .rolling_memory import RollingMemory
from .sliding_window import SlidingWindow
from .standalone_compaction import StandaloneCompaction
from .threshold_compaction import ThresholdCompaction


def strategy_from_env() -> ContextStrategy:
    name = os.getenv("RELAY_STRATEGY", "threshold").lower()
    threshold = int(os.getenv("RELAY_COMPACT_THRESHOLD", "120000"))
    manager = os.getenv("RELAY_MANAGER_MODEL") or None
    strategies: dict[str, Callable[[], ContextStrategy]] = {
        "threshold": lambda: ThresholdCompaction(threshold, manager),
        "sliding": lambda: SlidingWindow(
            max_items=int(os.getenv("RELAY_SLIDING_ITEMS", "64")),
            compact_threshold=threshold,
        ),
        "folding": lambda: RollbackFolding(threshold, manager),
        "agent_fold": lambda: AgentFold(threshold, manager),
        "rolling_memory": lambda: RollingMemory(manager),
        "native": lambda: NativeCompaction(threshold),
        "standalone": lambda: StandaloneCompaction(threshold),
        "truncation": OpenAITruncation,
        "full_history": FullHistory,
    }
    try:
        return strategies[name]()
    except KeyError as exc:
        raise ValueError(f"unknown RELAY_STRATEGY: {name}") from exc
