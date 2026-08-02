from __future__ import annotations

from typing import Any


class OfficialRLMAdapter:
    """Direct adapter for the authors' `rlms` package.

    RLM replaces a completion call and returns a final text answer. It is not a
    transparent Responses tool-loop middleware: an external coding agent cannot
    receive ordinary Responses function-call items from it. Use this adapter for
    text completions or provide the RLM itself with callable tools.

    The current official persistent+compaction mode is enabled by default. In
    that mode each call must supply only the newly appended context segment;
    the official runtime keeps prior REPL contexts itself. Set both flags to
    False and supply the full context to reproduce a fresh paper-style query.
    """

    SOURCE_REPOSITORY = "https://github.com/alexzhang13/rlm"
    SOURCE_COMMIT = "72d6940142ddfb84ee6be573dc999a37e633e671"

    def __init__(
        self,
        model: str,
        *,
        max_depth: int = 1,
        max_iterations: int = 30,
        environment: str = "local",
        persistent: bool = True,
        compaction: bool = True,
        custom_tools: dict[str, Any] | None = None,
    ) -> None:
        try:
            from rlm import RLM
        except ImportError as exc:
            raise RuntimeError(
                "install the experimental dependency: pip install rlms==0.1.3"
            ) from exc
        self._runtime = RLM(
            backend="openai",
            backend_kwargs={"model_name": model},
            environment=environment,
            max_depth=max_depth,
            max_iterations=max_iterations,
            persistent=persistent,
            compaction=compaction,
            custom_tools=custom_tools,
        )

    def completion(self, request: dict[str, Any], root_prompt: str | None = None) -> Any:
        items = request.get("input")
        if not isinstance(items, list):
            raise TypeError("official RLM adapter expects Responses-style list input")
        return self._runtime.completion(items, root_prompt=root_prompt)

    def close(self) -> None:
        cleanup = getattr(self._runtime, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def __enter__(self) -> OfficialRLMAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
