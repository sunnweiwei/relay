from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..strategies.base import BaseStrategy, PreparedInput


class OpenAITruncation(BaseStrategy):
    """Use the Responses API's built-in automatic truncation."""

    name = "openai_truncation"

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        return PreparedInput(deepcopy(active), {"truncation": "auto"})
