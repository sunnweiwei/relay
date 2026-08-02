from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


def _canonical(value: Any) -> bytes:
    """Encode JSON data so semantically identical objects have one identity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


@dataclass(frozen=True)
class PrefixMatch:
    artifact: dict[str, Any]
    matched_items: int


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    puts: int
    evictions: int
    entries: int
    nodes: int
    bytes: int


@dataclass(eq=False)
class _Node:
    digest: bytes
    depth: int
    parent: _Node | None = None
    children: dict[bytes, _Node] = field(default_factory=dict)
    artifact: dict[str, Any] | None = None
    checkpoint_bytes: int = 0
    expires_at: float | None = None


class PrefixCheckpointCache:
    """Tenant-partitioned exact-prefix checkpoint cache.

    A keyed hash trie gives each canonical trajectory prefix one node. Only
    checkpointed prefixes carry an opaque strategy artifact; sibling branches
    share their ancestor nodes. Artifacts are soft state: a miss must always be
    recoverable by replaying the supplied trajectory.
    """

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
        if max_entries <= 0 or max_bytes <= 0 or max_nodes <= 0:
            raise ValueError("cache limits must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("cache TTL must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_nodes = max_nodes
        self.ttl_seconds = ttl_seconds
        self._secret = secret or secrets.token_bytes(32)
        self._clock = clock
        self._roots: dict[bytes, _Node] = {}
        self._lru: OrderedDict[bytes, _Node] = OrderedDict()
        self._lock = RLock()
        self._entries = 0
        self._nodes = 0
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

    @classmethod
    def from_env(cls) -> PrefixCheckpointCache:
        secret = os.getenv("RELAY_CACHE_SECRET")
        ttl = float(os.getenv("RELAY_CACHE_TTL_SECONDS", str(6 * 60 * 60)))
        return cls(
            max_entries=int(os.getenv("RELAY_CACHE_MAX_ENTRIES", "4096")),
            max_bytes=int(os.getenv("RELAY_CACHE_MAX_BYTES", str(256 * 1024 * 1024))),
            max_nodes=int(os.getenv("RELAY_CACHE_MAX_NODES", "200000")),
            ttl_seconds=ttl,
            secret=secret.encode() if secret else None,
        )

    def partition(self, namespace: str, scope: Mapping[str, Any]) -> bytes:
        """Return an opaque partition ID without retaining tenant credentials."""

        return self._digest(
            b"partition\0",
            _canonical({"namespace": namespace, "scope": dict(scope)}),
        )

    def match(
        self,
        partition: bytes,
        trajectory: Sequence[Mapping[str, Any]],
    ) -> PrefixMatch | None:
        """Return the checkpoint on the longest exact cached prefix."""

        now = self._clock()
        with self._lock:
            node = self._roots.get(partition)
            if node is None:
                self._misses += 1
                return None

            best: _Node | None = self._valid_checkpoint(node, now)
            for item in trajectory:
                digest = self._child_digest(node.digest, item)
                child = node.children.get(digest)
                if child is None:
                    break
                node = child
                candidate = self._valid_checkpoint(node, now)
                if candidate is not None:
                    best = candidate

            if best is None or best.artifact is None:
                self._misses += 1
                return None
            self._hits += 1
            self._lru.move_to_end(best.digest)
            return PrefixMatch(
                deepcopy(best.artifact),
                best.depth,
            )

    def put(
        self,
        partition: bytes,
        trajectory: Sequence[Mapping[str, Any]],
        artifact: Mapping[str, Any],
    ) -> None:
        """Attach a checkpoint to the node for an exact trajectory prefix."""

        self.put_prefixes(
            partition,
            trajectory,
            [(len(trajectory), artifact)],
        )

    def put_prefixes(
        self,
        partition: bytes,
        trajectory: Sequence[Mapping[str, Any]],
        checkpoints: Sequence[
            tuple[int, Mapping[str, Any]]
        ],
    ) -> None:
        """Attach several checkpoints while traversing one trajectory once."""

        prepared: dict[int, tuple[dict[str, Any], int]] = {}
        for depth, artifact in checkpoints:
            if depth < 0 or depth > len(trajectory):
                raise ValueError("checkpoint prefix depth is outside the trajectory")
            stored = deepcopy(dict(artifact))
            stored_bytes = len(_canonical(stored))
            if stored_bytes <= self.max_bytes:
                prepared[depth] = (stored, stored_bytes)
        if not prepared:
            return

        now = self._clock()
        with self._lock:
            self._expire(now)
            node = self._roots.get(partition)
            if node is None:
                node = _Node(partition, depth=0)
                self._roots[partition] = node
                self._nodes += 1

            if 0 in prepared:
                self._attach_checkpoint(node, *prepared[0], now)
            last_depth = max(prepared)
            if last_depth > 0:
                for depth, item in enumerate(trajectory, start=1):
                    digest = self._child_digest(node.digest, item)
                    child = node.children.get(digest)
                    if child is None:
                        child = _Node(digest, depth=node.depth + 1, parent=node)
                        node.children[digest] = child
                        self._nodes += 1
                    node = child
                    if depth in prepared:
                        self._attach_checkpoint(node, *prepared[depth], now)
                    if depth == last_depth:
                        break

            self._puts += len(prepared)
            self._evict_to_limits()

    def _attach_checkpoint(
        self,
        node: _Node,
        stored: dict[str, Any],
        stored_bytes: int,
        now: float,
    ) -> None:
        if node.artifact is None:
            self._entries += 1
        else:
            self._bytes -= node.checkpoint_bytes
            self._lru.pop(node.digest, None)
        node.artifact = stored
        node.checkpoint_bytes = stored_bytes
        node.expires_at = None if self.ttl_seconds is None else now + self.ttl_seconds
        self._bytes += stored_bytes
        self._lru[node.digest] = node

    def clear(self) -> None:
        with self._lock:
            self._roots.clear()
            self._lru.clear()
            self._entries = 0
            self._nodes = 0
            self._bytes = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                puts=self._puts,
                evictions=self._evictions,
                entries=self._entries,
                nodes=self._nodes,
                bytes=self._bytes,
            )

    def _digest(self, *parts: bytes) -> bytes:
        value = hmac.new(self._secret, digestmod=hashlib.sha256)
        for part in parts:
            value.update(part)
        return value.digest()

    def _child_digest(self, parent: bytes, item: Mapping[str, Any]) -> bytes:
        return self._digest(b"item\0", parent, b"\0", _canonical(dict(item)))

    def _valid_checkpoint(self, node: _Node, now: float) -> _Node | None:
        if node.artifact is None:
            return None
        if node.expires_at is not None and node.expires_at <= now:
            self._drop_checkpoint(node)
            return None
        return node

    def _expire(self, now: float) -> None:
        for node in list(self._lru.values()):
            if node.expires_at is not None and node.expires_at <= now:
                self._drop_checkpoint(node)

    def _evict_to_limits(self) -> None:
        while self._lru and (
            self._entries > self.max_entries
            or self._bytes > self.max_bytes
            or self._nodes > self.max_nodes
        ):
            _, node = next(iter(self._lru.items()))
            self._drop_checkpoint(node)

    def _drop_checkpoint(self, node: _Node) -> None:
        if node.artifact is None:
            return
        self._lru.pop(node.digest, None)
        self._entries -= 1
        self._bytes -= node.checkpoint_bytes
        node.artifact = None
        node.checkpoint_bytes = 0
        node.expires_at = None
        self._evictions += 1
        self._prune(node)

    def _prune(self, node: _Node) -> None:
        while node.parent is not None and not node.children and node.artifact is None:
            parent = node.parent
            parent.children.pop(node.digest, None)
            self._nodes -= 1
            node = parent
        if node.parent is None and not node.children and node.artifact is None:
            self._roots.pop(node.digest, None)
            self._nodes -= 1
