import os

from .base import ContextStrategy as ContextStrategy
from .base import GeneratedCheckpoint as GeneratedCheckpoint
from .base import PreparedInput as PreparedInput
from .checkpoint import Checkpoint
from .compact import CODEX_COMPACTION_PROMPT as CODEX_COMPACTION_PROMPT
from .compact import CODEX_SUMMARY_PREFIX as CODEX_SUMMARY_PREFIX
from .compact import Compact
from .rolling_memory import RollingMemory
from .sliding_window import SlidingWindow


def strategy_from_env() -> ContextStrategy:
    name = os.getenv("RELAY_STRATEGY", "compact")
    if name == "compact":
        return Compact.from_env()
    if name == "checkpoint":
        return Checkpoint.from_env()
    if name == "sliding_window":
        return SlidingWindow.from_env()
    if name == "rolling_memory":
        return RollingMemory.from_env()
    raise ValueError(
        "RELAY_STRATEGY must be 'compact', 'checkpoint', 'sliding_window', "
        "or 'rolling_memory'"
    )


__all__ = ["Checkpoint", "Compact", "RollingMemory", "SlidingWindow"]
