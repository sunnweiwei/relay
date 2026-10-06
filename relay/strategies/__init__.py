from .acm import ACM
from .autocompact import AutoCompact
from .base import Strategy, Summarizer
from .clm import ContextLanguageModel
from .compaction import Compaction
from .folding import ContextFolding
from .mem1 import MEM1
from .prolong import ProLong
from .rlm import RLM, PersistentRLM
from .selfcompact import SelfCompact

STRATEGIES = {"compaction": Compaction, "clm": ContextLanguageModel, "folding": ContextFolding, "prolong": ProLong,
              "selfcompact": SelfCompact, "autocompact": AutoCompact, "acm": ACM, "mem1": MEM1, "rlm": RLM,
              "rlm_persistent": PersistentRLM}  # RELAY_STRATEGY

__all__ = ["ACM", "MEM1", "RLM", "STRATEGIES", "AutoCompact", "Compaction", "ContextFolding", "ContextLanguageModel",
           "PersistentRLM", "ProLong", "SelfCompact", "Strategy", "Summarizer"]
