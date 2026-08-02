import os

from .base import ContextStrategy as ContextStrategy
from .base import GeneratedCheckpoint as GeneratedCheckpoint
from .base import PreparedInput as PreparedInput
from .checkpoint import Checkpoint
from .compact import CODEX_COMPACTION_PROMPT as CODEX_COMPACTION_PROMPT
from .compact import CODEX_SUMMARY_PREFIX as CODEX_SUMMARY_PREFIX
from .compact import Compact


def strategy_from_env() -> ContextStrategy:
    name = os.getenv("RELAY_STRATEGY", "compact")
    if name == "compact":
        return Compact.from_env()
    if name == "checkpoint":
        return Checkpoint.from_env()
    raise ValueError("RELAY_STRATEGY must be 'compact' or 'checkpoint'")


__all__ = ["Checkpoint", "Compact"]
