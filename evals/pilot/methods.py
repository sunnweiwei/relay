"""The context-management methods this evaluation compared.

Relay's strategy registry can grow after the evaluation (later strategies are not part of
these results), so planned cells, report rows and port assignments use this fixed list.
"""

EVALUATED_METHODS = ("compaction", "clm", "folding", "prolong", "selfcompact",
                     "autocompact", "acm", "ace", "mem1", "rlm", "rlm_persistent")
