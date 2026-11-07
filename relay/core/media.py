"""Media where only text goes, the same for every strategy: an item's images, files and audio as lines
a model can read, keep or delete, as marks in a transcript, and counted in a note. A strategy decides
what stays; what kinds there are is not its business.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace

from .ir import Item, Media
from .tokens import media_tokens

LINE = "[{type} {n} of this turn, ~{tokens:,} tokens]"
LINES = re.compile(r"^\[\S+ (\d+) of this turn, ~[\d,]+ tokens\]$", re.M)


def lines(item: Item) -> str:
    """A line for each of the item's media."""

    return "\n".join(LINE.format(type=m.type, n=n, tokens=media_tokens(m)) for n, m in enumerate(item.media, start=1))


def kept(item: Item, body: str) -> Item:
    """`item` with `body` as its text, and of its media those whose lines `body` kept."""

    numbers = {int(n) for n in LINES.findall(body)}
    media = tuple(m for n, m in enumerate(item.media, start=1) if n in numbers)
    return replace(item, text=LINES.sub("", body).strip(), media=media)


def marks(media: Iterable[Media], start: int = 0) -> list[str]:
    """Each one's mark where it stood in a transcript, numbered on from `start`."""

    return [f"[{m.type} {n}]" for n, m in enumerate(media, start=start + 1)]


def counted(media: Iterable[Media]) -> str:
    """"1 image and 2 files"."""

    return " and ".join(f"{n} {kind}{'s' if n > 1 else ''}" for kind, n in Counter(m.type for m in media).items())


def removed(media: Iterable[Media]) -> str:
    """What stands in an item for media taken out of it, where nothing else says so."""

    return f"[{counted(media)} removed]"
