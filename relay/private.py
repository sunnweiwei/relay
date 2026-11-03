"""Files only their owner reads: Relay's settings hold API keys, its logs what conversations said."""

from __future__ import annotations

import os
from pathlib import Path


def write(path: Path, text: str) -> None:
    """`path` replaced by `text` at once (a reader never sees half a file), owner-only."""

    temporary = path.with_suffix(".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as stream:
        stream.write(text)
    os.chmod(temporary, 0o600)  # (a file left from before keeps the mode it had)
    temporary.replace(path)


def append(path: str | Path, text: str) -> None:
    """`text` at the end of `path`, which is owner-only if Relay creates it."""

    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8") as stream:
        stream.write(text)
