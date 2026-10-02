"""Exact-prefix state store.

Harnesses resend their whole append-only history on every request, so the state a
strategy computed for one request is found again by walking the canonical item
prefix of the next one. A keyed hash trie gives each prefix one node; only nodes
carrying a value count toward the limits. Values are soft state: a miss is always
recoverable by recomputing from the request.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass(eq=False)
class _Node:
    digest: bytes
    depth: int
    parent: _Node | None = None
    children: dict[bytes, _Node] = field(default_factory=dict)
    value: dict[str, Any] | None = None
    size: int = 0
    expires_at: float | None = None


class PrefixStore:
    def __init__(
        self,
        *,
        max_entries: int = 4_096,
        max_bytes: int = 256 * 1024 * 1024,
        max_nodes: int = 200_000,
        ttl_seconds: float | None = 6 * 60 * 60,
        secret: bytes | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(max_entries, max_bytes, max_nodes) <= 0:
            raise ValueError("store limits must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_nodes = max_nodes
        self.ttl_seconds = ttl_seconds
        self._secret = secret or secrets.token_bytes(32)
        self._clock = clock
        self._roots: dict[bytes, _Node] = {}
        self._lru: OrderedDict[bytes, _Node] = OrderedDict()
        self._lock = RLock()
        self._nodes = 0
        self._bytes = 0

    @classmethod
    def from_env(cls) -> PrefixStore:
        secret = os.getenv("RELAY_CACHE_SECRET")
        ttl = float(os.getenv("RELAY_CACHE_TTL_SECONDS", str(6 * 60 * 60)))
        return cls(
            max_entries=int(os.getenv("RELAY_CACHE_MAX_ENTRIES", "4096")),
            max_bytes=int(os.getenv("RELAY_CACHE_MAX_BYTES", str(256 * 1024 * 1024))),
            max_nodes=int(os.getenv("RELAY_CACHE_MAX_NODES", "200000")),
            ttl_seconds=ttl if ttl > 0 else None,
            secret=secret.encode() if secret else None,
        )

    def partition(self, *parts: str) -> bytes:
        """An opaque partition id; the parts (e.g. credentials) are not retained."""

        return self._digest(b"partition", *(part.encode() for part in parts))

    def match(self, partition: bytes, keys: Sequence[bytes]) -> tuple[int, dict[str, Any]] | None:
        """Return (depth, value) for the deepest stored prefix of `keys`."""

        now = self._clock()
        with self._lock:
            node = self._roots.get(partition)
            best = node if node is not None and self._live(node, now) else None
            for key in keys:
                if node is None:
                    break
                node = node.children.get(self._digest(b"item", node.digest, key))
                if node is not None and self._live(node, now):
                    best = node
            if best is None:
                return None
            self._lru.move_to_end(best.digest)
            return best.depth, deepcopy(best.value)  # type: ignore[arg-type]

    def put(self, partition: bytes, keys: Sequence[bytes], value: dict[str, Any]) -> None:
        """Attach `value` to the node for exactly `keys`."""

        stored = deepcopy(value)
        size = len(json.dumps(stored, separators=(",", ":")))
        if size > self.max_bytes:
            return
        now = self._clock()
        with self._lock:
            node = self._roots.get(partition)
            if node is None:
                node = self._roots[partition] = _Node(partition, 0)
                self._nodes += 1
            for key in keys:
                digest = self._digest(b"item", node.digest, key)
                child = node.children.get(digest)
                if child is None:
                    child = node.children[digest] = _Node(digest, node.depth + 1, node)
                    self._nodes += 1
                node = child
            if node.value is not None:
                self._bytes -= node.size
            node.value, node.size = stored, size
            node.expires_at = None if self.ttl_seconds is None else now + self.ttl_seconds
            self._bytes += size
            self._lru[node.digest] = node
            self._lru.move_to_end(node.digest)
            while self._lru and (
                len(self._lru) > self.max_entries
                or self._bytes > self.max_bytes
                or self._nodes > self.max_nodes
            ):
                self._drop(next(iter(self._lru.values())))

    def __len__(self) -> int:
        return len(self._lru)

    def _digest(self, *parts: bytes) -> bytes:
        mac = hmac.new(self._secret, digestmod=hashlib.sha256)
        for part in parts:
            mac.update(len(part).to_bytes(8, "big"))
            mac.update(part)
        return mac.digest()

    def _live(self, node: _Node, now: float) -> bool:
        if node.value is None:
            return False
        if node.expires_at is not None and node.expires_at <= now:
            self._drop(node)
            return False
        return True

    def _drop(self, node: _Node) -> None:
        self._lru.pop(node.digest, None)
        self._bytes -= node.size
        node.value, node.size, node.expires_at = None, 0, None
        while not node.children and node.value is None:
            if node.parent is None:
                self._roots.pop(node.digest, None)
                self._nodes -= 1
                return
            node.parent.children.pop(node.digest, None)
            self._nodes -= 1
            node = node.parent
