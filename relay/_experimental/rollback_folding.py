"""Experimental rollback-folding operator."""

from .folding_base import ModelFold


class RollbackFolding(ModelFold):
    def __init__(
        self, compact_threshold: int = 120_000, manager_model: str | None = None
    ) -> None:
        super().__init__("rollback", compact_threshold, manager_model)
