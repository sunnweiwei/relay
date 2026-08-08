import os

from .agent_fold import AgentFold
from .auto_compact import AutoCompact
from .base import ContextStrategy as ContextStrategy
from .base import GeneratedCheckpoint as GeneratedCheckpoint
from .base import PreparedInput as PreparedInput
from .checkpoint import Checkpoint
from .compact import CODEX_COMPACTION_PROMPT as CODEX_COMPACTION_PROMPT
from .compact import CODEX_SUMMARY_PREFIX as CODEX_SUMMARY_PREFIX
from .compact import Compact
from .context_folding import ContextFolding
from .prolong import ProLong
from .rlm import RLM
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
    if name == "rlm":
        return RLM.from_env()
    if name == "context_folding":
        return ContextFolding.from_env()
    if name == "agent_fold":
        return AgentFold.from_env()
    if name == "auto_compact":
        return AutoCompact.from_env()
    if name == "prolong":
        return ProLong.from_env()
    raise ValueError(
        "RELAY_STRATEGY must be 'compact', 'checkpoint', 'sliding_window', "
        "'rolling_memory', 'rlm', 'context_folding', 'agent_fold', or "
        "'auto_compact' or 'prolong'"
    )


__all__ = [
    "RLM",
    "AgentFold",
    "AutoCompact",
    "Checkpoint",
    "Compact",
    "ContextFolding",
    "ProLong",
    "RollingMemory",
    "SlidingWindow",
]
