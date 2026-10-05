from .base import Strategy, Summarizer
from .clm import ContextLanguageModel
from .compaction import Compaction

STRATEGIES = {"compaction": Compaction, "clm": ContextLanguageModel}  # RELAY_STRATEGY

__all__ = ["STRATEGIES", "Compaction", "ContextLanguageModel", "Strategy", "Summarizer"]
