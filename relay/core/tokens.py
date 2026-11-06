"""Token approximations shared by the engine and strategies (same rules as Codex)."""

from __future__ import annotations

import base64
import math
import struct
from functools import lru_cache

from .ir import Item, Media

PIXELS_PER_TOKEN = 750  # Anthropic's documented rule; gpt-5's patches measure ~10% fewer
IMAGE_TOKENS = 1600  # an image whose size cannot be read (a URL, a file id)
BYTES_PER_TOKEN = 4  # Codex's rule; OpenAI and Gemini tokenizers measured about 4.6, so it errs high
DENSER = (("claude", 2.5),)  # Claude's tokenizer measured about 2.7 bytes a token


def bytes_per_token(model: object) -> float:
    name = model.rsplit("/", 1)[-1].lower() if isinstance(model, str) else ""
    return next((size for prefix, size in DENSER if name.startswith(prefix)), BYTES_PER_TOKEN)


def approx_tokens(text: str, per_token: float = BYTES_PER_TOKEN) -> int:
    return math.ceil(len(text.encode("utf-8")) / per_token)


def item_tokens(item: Item, per_token: float = BYTES_PER_TOKEN) -> int:
    """An item's text and opaque content (encrypted reasoning or compaction, signatures), and its images."""

    return math.ceil((len(item.text.encode("utf-8")) + item.opaque) / per_token) + sum(map(media_tokens, item.media))


def media_tokens(media: Media) -> int:
    """An image's tokens, from its size in pixels (its base64 is not text the model reads)."""

    if media.type != "image":
        return 0
    size = _image_size(media.data) if media.data else None
    return math.ceil(size[0] * size[1] / PIXELS_PER_TOKEN) if size else IMAGE_TOKENS


@lru_cache(maxsize=4096)
def _image_size(data: str) -> tuple[int, int] | None:
    """Width and height of a base64 PNG, JPEG, GIF or WebP, read from its header."""

    head = base64.b64decode(data[:64] + "=" * (-len(data[:64]) % 4))
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", head[6:10])
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP" and head[12:16] == b"VP8 ":
        width, height = struct.unpack("<HH", head[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if head[:2] == b"\xff\xd8":  # JPEG: the first start-of-frame marker
        raw, n = base64.b64decode(data + "=" * (-len(data) % 4)), 2
        while n + 9 < len(raw):
            marker, length = raw[n + 1], struct.unpack(">H", raw[n + 2:n + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", raw[n + 5:n + 9])
                return width, height
            n += 2 + length
    return None


def truncate_middle(text: str, max_tokens: int) -> str:
    """Keep the head and tail of `text` within `max_tokens`, marking the cut."""

    budget = max(0, max_tokens) * BYTES_PER_TOKEN
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    left = encoded[: budget // 2].decode("utf-8", errors="ignore")
    right = encoded[len(encoded) - (budget - budget // 2) :].decode("utf-8", errors="ignore")
    removed = approx_tokens(text) - max(0, max_tokens)
    return f"{left}…{removed} tokens truncated…{right}"
