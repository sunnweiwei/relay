"""One explicit, local input-budget policy for planning and dispatch.

This counter is a reproducible evaluation convention, not a claim to reproduce
the provider's private tokenizer/template. Provider usage stays a separate cost
measurement and can additionally detect a disagreement after a response.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import math
import struct
from typing import Any, ClassVar

from ..protocols.base import Body, Codec
from .tokens import BYTES_PER_TOKEN, counted_item_tokens, structured_tokens
from .ir import Item


class InputAccountingError(ValueError):
    """The declared counter cannot account for this request reliably."""

    def __init__(self, message: str, *, category: str = "input_accounting_error") -> None:
        self.category = category
        super().__init__(message)


MEDIA_INPUT_BUDGET_COUNTER = "o200k_fields_reasoning_usage_framing8_gpt6_luna_media_empirical_v1"
PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER = "o200k_fields_reasoning_usage_framing8_gpt6_luna_png_jpeg_empirical_v1"
_MEDIA_COUNTERS = {MEDIA_INPUT_BUDGET_COUNTER, PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER}
_MEDIA_MAX_BASE64_CHARS = 16 * 1024 * 1024
_MEDIA_MAX_DIMENSION = 2048
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read JPEG dimensions by walking bounded marker segments to a SOF marker."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise InputAccountingError("Inline JPEG is missing its SOI marker", category="invalid_image_header")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            raise InputAccountingError("Invalid JPEG marker boundary", category="invalid_image_header")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0x00:
            raise InputAccountingError("Unexpected stuffed JPEG byte before scan data",
                                       category="invalid_image_header")
        if marker in {0x01, 0xD8, *range(0xD0, 0xD8)}:
            continue
        if marker in {0xD9, 0xDA}:
            break
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[offset:offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            raise InputAccountingError("Truncated JPEG marker segment", category="invalid_image_header")
        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 8:
                raise InputAccountingError("JPEG SOF segment is too short", category="invalid_image_header")
            height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
            return width, height
        offset += segment_length
    raise InputAccountingError("JPEG dimensions were not found before image scan data",
                               category="invalid_image_header")


def _image_dimensions(data: bytes) -> tuple[str, int, int]:
    """Detect a bounded inline PNG/JPEG signature and read its encoded dimensions."""
    if len(data) >= 24 and data[:8] == _PNG_SIGNATURE and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", width, height
    if data.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(data)
        return "image/jpeg", width, height
    raise InputAccountingError("Inline image signature is not PNG or JPEG", category="invalid_image_header")


class BudgetExceeded(RuntimeError):
    def __init__(self, tokens: int, limit: int, scope: str, detail: str = "") -> None:
        self.tokens, self.limit, self.scope, self.detail = tokens, limit, scope, detail
        super().__init__(f"{scope} input requires {tokens} budget tokens; limit is {limit}. {detail}".strip())

    def payload(self) -> Body:
        return {"error": {"code": "context_length_exceeded", "message": str(self),
                          "source": "relay_input_budget", "scope": self.scope,
                          "input_tokens": self.tokens, "input_limit": self.limit}}


@dataclass(frozen=True)
class InputBudget:
    """Decoded tool fields + o200k text + observed reasoning + framing, version 3.

    `limit` is an INPUT limit. An optional total context limit can also reserve
    output space. It never silently changes the caller's generation settings.
    """

    limit: int
    context_limit: int | None = None
    output_reserve: int = 0
    name: str = "o200k_fields_reasoning_usage_framing8_v3"
    reasoning_usage: dict[str, tuple[int, int]] = field(default_factory=dict, compare=False, repr=False)
    dispatch_slack: int = 0
    overflow_fallback: str = "error"
    # Encrypted reasoning whose usage this process never saw (a resumed conversation, a restarted Relay):
    # "error" refuses the request; "estimate" counts its ciphertext as Relay estimates elsewhere.
    unknown_opaque: str = "error"
    # An image the media convention does not cover: "error" refuses the request; "estimate" counts an upper bound.
    unknown_media: str = "error"
    MEDIA_COUNTER_NAME: ClassVar[str] = MEDIA_INPUT_BUDGET_COUNTER
    MEDIA_COUNTER_NAMES: ClassVar[frozenset[str]] = frozenset(_MEDIA_COUNTERS)


    def __post_init__(self) -> None:
        if self.overflow_fallback not in {"error", "truncate"}:
            raise ValueError("overflow_fallback must be error or truncate")
        if self.unknown_opaque not in {"error", "estimate"}:
            raise ValueError("unknown_opaque must be error or estimate")
        if self.unknown_media not in {"error", "estimate"}:
            raise ValueError("unknown_media must be error or estimate")
        if self.limit <= 0 or self.output_reserve < 0 or self.dispatch_slack < 0:
            raise ValueError("Input budget must be positive; output reserve cannot be negative")
        if self.context_limit is not None and self.context_limit <= self.output_reserve:
            raise ValueError("Output reserve leaves no room for input")
        if self.output_reserve and self.context_limit is None:
            raise ValueError("A total context limit is required for an output reserve")
        # Fail at setup, not halfway through a trajectory, if the optional
        # evaluation tokenizer is unavailable. No byte-count fallback.
        self.encoding()

    @staticmethod
    def encoding() -> Any:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")

    def text_tokens(self, text: str) -> int:
        return len(self.encoding().encode(text, disallowed_special=()))

    def observe_response(self, response: Body) -> list[dict[str, Any]]:
        """Remember only digests and usage, never decrypt opaque reasoning.

        Usage is aggregate per response. For multiple reasoning items, charge
        the full aggregate to each item as an explicit upper bound. This keeps
        partial histories and strategy item arithmetic conservative and uses
        exactly the same convention in planning and dispatch.
        """
        usage = response.get("usage") or {}
        tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        if not isinstance(tokens, int) or tokens < 0:
            return []
        keys = {hashlib.sha256(item["encrypted_content"].encode()).hexdigest()
                for item in response.get("output") or [] if item.get("type") == "reasoning"
                and isinstance(item.get("encrypted_content"), str) and item["encrypted_content"]}
        records = []
        for key in sorted(keys):
            value = (tokens, len(keys))
            if key in self.reasoning_usage and self.reasoning_usage[key] != value:
                raise InputAccountingError("Conflicting provider reasoning usage for the same opaque item")
            self.reasoning_usage[key] = value
            records.append({"digest": key, "tokens": tokens, "response_items": len(keys)})
        return records

    def opaque_tokens(self, item: Item) -> int:
        if not item.opaque:
            return 0
        if item.opaque_key not in self.reasoning_usage:
            if self.unknown_opaque == "estimate":
                return math.ceil(item.opaque / BYTES_PER_TOKEN)
            raise InputAccountingError("Opaque token usage is unknown; ciphertext size is not a token count")
        return self.reasoning_usage[item.opaque_key][0]

    def item_tokens(self, item: Item) -> int:
        media_counter = self.media_item_tokens if self.name in _MEDIA_COUNTERS else None
        if item.media and media_counter is None:
            raise InputAccountingError("Media token accounting is not supported by this input-budget policy",
                                       category="unsupported_policy")
        return counted_item_tokens(item, self.text_tokens, self.opaque_tokens, media_counter)

    def preamble_tokens(self, codec: Codec, body: Body) -> int:
        instructions = body.get("instructions") or ""
        return (self.text_tokens(instructions) if isinstance(instructions, str)
                else structured_tokens(instructions, self.text_tokens)) + (
                    structured_tokens(body["tools"], self.text_tokens) if body.get("tools") else 0) + 3

    @property
    def input_limit(self) -> int:
        return min(self.limit, self.context_limit - self.output_reserve) if self.context_limit else self.limit

    def fingerprint(self) -> dict[str, Any]:
        result = {"counter": self.name, "input_limit": self.limit,
                  "context_limit": self.context_limit, "output_reserve": self.output_reserve,
                  "dispatch_slack": self.dispatch_slack, "dispatch_input_limit": self.limit_for({}),
                  **({"overflow_fallback": "truncate_oldest_groups_v1"} if self.overflow_fallback == "truncate" else {})}
        if self.name in _MEDIA_COUNTERS:
            mime = ["image/png"]
            if self.name == PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER:
                mime.append("image/jpeg")
            result["media_policy"] = {
                "kind": "empirical_image_patch_upper_bound_v1",
                "model": "gpt-6-luna",
                "detail": ["high", "auto_or_omitted"],
                "mime": mime,
                "max_dimension": _MEDIA_MAX_DIMENSION,
                "formula": "128 + ceil(1.25 * ceil(width/32) * ceil(height/32)) per image, using original dimensions",
                "resize_handling": "No resize reduction is credited; raw dimensions are charged conservatively through 2048px per side.",
                "largest_empirically_observed_dimension": 768,
                "interpretation": "Conservative empirical harness upper-bound convention, not exact provider image-token accounting.",
            }
        return result

    def media_item_tokens(self, item: Item) -> int:
        """Apply the selected, explicitly versioned Luna image convention."""
        try:
            return self._media_item_tokens(item)
        except InputAccountingError as exc:
            exc.media_diagnostics = [self._media_diagnostic(image, exc.category) for image in item.media]
            raise

    def _media_item_tokens(self, item: Item) -> int:
        if self.name not in _MEDIA_COUNTERS:
            raise InputAccountingError("Media token accounting is not supported by this input-budget policy",
                                       category="unsupported_policy")
        total = 0
        for image in item.media:
            try:
                total += self._image_tokens(image)
            except InputAccountingError:
                if self.unknown_media != "estimate":
                    raise
                total += self._image_estimate(image)
        return total

    def _image_estimate(self, image: Any) -> int:
        """An image the convention does not cover (another format, past 2048 px, by URL, low detail), counted as an
        upper bound: its pixels scaled to fit 2048 px a side when they can be read, else a 2048 x 2048 image."""

        width = height = _MEDIA_MAX_DIMENSION
        try:
            if image.data and not image.url:
                _, w, h = _image_dimensions(base64.b64decode(image.data, validate=True))
                if w > 0 and h > 0:
                    scale = min(1.0, _MEDIA_MAX_DIMENSION / max(w, h))
                    width, height = max(1, math.ceil(w * scale)), max(1, math.ceil(h * scale))
        except (InputAccountingError, ValueError, base64.binascii.Error):
            pass
        return 128 + math.ceil(1.25 * ((width + 31) // 32) * ((height + 31) // 32))

    def _image_tokens(self, image: Any) -> int:
        if image.type != "image":
            raise InputAccountingError("Empirical image accounting supports image media only",
                                       category="unsupported_media_kind")
        if image.detail not in (None, "auto", "high"):
            raise InputAccountingError("Empirical image accounting supports high or automatic detail only",
                                       category="unsupported_detail")
        if not image.data or image.url:
            raise InputAccountingError("Empirical image accounting requires inline base64 image data",
                                       category="non_inline_media")
        if len(image.data) > _MEDIA_MAX_BASE64_CHARS:
            raise InputAccountingError("Inline image exceeds the media accounting size limit",
                                       category="encoded_image_too_large")
        try:
            data = base64.b64decode(image.data, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise InputAccountingError("Inline image base64 is malformed",
                                       category="malformed_base64") from exc
        detected_mime, width, height = _image_dimensions(data)
        declared_mime = image.mime.lower()
        allowed_mimes = {"image/png"}
        if self.name == PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER:
            allowed_mimes.add("image/jpeg")
        if self.name == MEDIA_INPUT_BUDGET_COUNTER and declared_mime != "image/png":
            raise InputAccountingError("PNG images only are supported by this accounting policy",
                                       category="unsupported_mime")
        if detected_mime != declared_mime:
            raise InputAccountingError("Declared image MIME does not match its encoded bytes",
                                       category="mime_signature_mismatch")
        if declared_mime not in allowed_mimes:
            if self.name == MEDIA_INPUT_BUDGET_COUNTER:
                raise InputAccountingError("PNG images only are supported by this accounting policy",
                                           category="unsupported_mime")
            raise InputAccountingError("Image MIME is unsupported by the selected accounting policy",
                                       category="unsupported_mime")
        if width <= 0 or height <= 0 or max(width, height) > _MEDIA_MAX_DIMENSION:
            raise InputAccountingError(
                f"Empirical image policy supports dimensions only through {_MEDIA_MAX_DIMENSION}px per side",
                category="invalid_or_oversized_dimensions")
        patches = ((width + 31) // 32) * ((height + 31) // 32)
        return 128 + math.ceil(1.25 * patches)

    @staticmethod
    def _safe_media_mime(value: str) -> str:
        normalized = value.lower()
        return normalized if normalized in {"image/png", "image/jpeg"} else "other"

    @staticmethod
    def _safe_media_detail(value: str | None) -> str | None:
        return value if value in {None, "auto", "high", "low", "original"} else "other"

    def _media_diagnostic(self, image: Any, error_category: str) -> dict[str, Any]:
        encoded = image.data or ""
        byte_count = None
        width = height = None
        detected_mime = None
        parse_category = None
        if encoded and len(encoded) <= _MEDIA_MAX_BASE64_CHARS:
            try:
                data = base64.b64decode(encoded, validate=True)
                byte_count = len(data)
                detected_mime, width, height = _image_dimensions(data)
            except InputAccountingError as exc:
                parse_category = exc.category
            except (ValueError, base64.binascii.Error):
                parse_category = "malformed_base64"
        elif len(encoded) > _MEDIA_MAX_BASE64_CHARS:
            parse_category = "encoded_image_too_large"
        return {
            "media_kind": "image" if image.type == "image" else "other",
            "declared_mime": self._safe_media_mime(image.mime),
            "detected_mime": detected_mime,
            "detail": self._safe_media_detail(image.detail),
            "inline": bool(encoded) and not bool(image.url),
            "encoded_char_count": len(encoded),
            "decoded_byte_count": byte_count,
            "width": width,
            "height": height,
            "error_category": parse_category or error_category,
        }

    def media_diagnostics(self, codec: Codec, body: Body,
                          error_category: str = "input_accounting_error") -> list[dict[str, Any]]:
        """Return metadata only. Never include image bytes, URLs, or associated text."""
        records: list[dict[str, Any]] = []
        for wire in codec.items(body):
            item = codec.classify(wire)
            for image in item.media:
                records.append(self._media_diagnostic(image, error_category))
        return records

    def count(self, codec: Codec, body: Body) -> int:
        if codec.name not in ("openai_responses", "openai_chat") or not codec.managed(body):
            raise InputAccountingError("The strict input counter supports stateless Responses and Chat Completions "
                                       "requests only", category="unsupported_request_shape")
        items = [codec.classify(wire) for wire in codec.items(body)]
        has_media = any(item.media for item in items)
        try:
            if has_media and self.name not in _MEDIA_COUNTERS:
                raise InputAccountingError("Media token accounting is not supported by this input-budget policy",
                                           category="unsupported_policy")
            total = self.preamble_tokens(codec, body)
            if has_media and body.get("model") != "gpt-6-luna":
                raise InputAccountingError("Empirical image accounting is calibrated only for gpt-6-luna",
                                           category="unsupported_model")
            for item in items:
                total += self.item_tokens(item)
            # Response schemas and program definitions are outside input[] and
            # codec.preamble(), but can contribute model-facing instructions.
            extra = {key: body[key] for key in ("access_programs",) if body.get(key)}
            if isinstance(body.get("text"), dict) and body["text"].get("format"):
                extra["response_format"] = body["text"]["format"]
            if extra:
                total += structured_tokens(extra, self.text_tokens)
            return total
        except InputAccountingError as exc:
            if has_media:
                exc.media_diagnostics = self.media_diagnostics(codec, body, exc.category)
            raise

    def limit_for(self, body: Body) -> int:
        # Methods keep the nominal window. Only dispatch gets the explicitly
        # declared tolerance, bounded by any real context/output constraint.
        limit = self.limit + self.dispatch_slack
        if self.context_limit is not None:
            requested_output = body.get("max_output_tokens", self.output_reserve)
            if not isinstance(requested_output, int) or requested_output < 0:
                raise ValueError("max_output_tokens must be a nonnegative integer")
            limit = min(limit, self.context_limit - max(requested_output, self.output_reserve))
        return limit

    def check(self, codec: Codec, body: Body, scope: str) -> int:
        tokens = self.count(codec, body)
        limit = self.limit_for(body)
        if tokens > limit:
            raise BudgetExceeded(tokens, limit, scope)
        return tokens
