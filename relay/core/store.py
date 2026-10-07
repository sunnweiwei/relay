"""Exact-prefix state store.

Harnesses resend their whole append-only history on every request, so the state a
strategy computed for one request is found again by walking the canonical item
prefix of the next one. A keyed hash trie gives each prefix one node; only nodes
carrying a value count toward the limits. Values are soft state: a miss is always
recoverable by recomputing from the request.

The store can be kept on disk (`path`, with a fixed `secret`), so that a restarted Relay
finds the contexts it had. Relay does this by default (`from_env`: ~/.relay/store-<port>.json,
one per Relay process, with a secret it generates once and keeps beside it; RELAY_CACHE_PATH=off
keeps it in memory): a strategy's result is then not recomputed, and a result that cannot be
recomputed (the edits a model made under CLM) is not lost. Only the keyed digests of the
prefixes are written, never the items; values are written as they are stored (summaries and
strategy state, conversation text). A writer thread, off the request path, writes what changed
every `save_interval` seconds (0: on every change), as does `flush()` (Relay calls it on
shutdown), atomically and readable by its owner only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

log = logging.getLogger("relay")
FORMAT = 2  # 2: every node written once, with its parent (1 repeated every entry's whole path)


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
        path: str | os.PathLike[str] | None = None,
        save_interval: float = 1.0,
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
        if path is not None and secret is None:  # a random secret makes every saved key unreachable
            log.warning("the prefix store is kept in memory only: a store on disk needs a fixed secret")
            path = None
        self._path = None if path is None else os.fspath(path)
        self._save_interval = save_interval
        self._save_lock = RLock()
        self._dirty = False
        if self._path is not None:
            self._load()
            if save_interval > 0:
                threading.Thread(target=_write_every, args=(weakref.ref(self), save_interval), daemon=True,
                                 name="relay-store-writer").start()

    @classmethod
    def from_env(cls) -> PrefixStore:
        secret: str | bytes | None = os.getenv("RELAY_CACHE_SECRET")
        ttl = float(os.getenv("RELAY_CACHE_TTL_SECONDS", str(6 * 60 * 60)))
        default = os.path.join(os.path.expanduser("~"), ".relay", f"store-{os.getenv('RELAY_PORT', '8787')}.json")
        path: str | None = os.getenv("RELAY_CACHE_PATH", default)  # (one file per Relay process: they do not share it)
        if path.strip().lower() in ("", "off", "none", "memory"):
            path = None
        if path is not None and not secret:
            secret = _kept_secret(f"{path}.secret")
            path = path if secret else None
        return cls(
            max_entries=int(os.getenv("RELAY_CACHE_MAX_ENTRIES", "4096")),
            max_bytes=int(os.getenv("RELAY_CACHE_MAX_BYTES", str(256 * 1024 * 1024))),
            max_nodes=int(os.getenv("RELAY_CACHE_MAX_NODES", "200000")),
            ttl_seconds=ttl if ttl > 0 else None,
            secret=secret.encode() if isinstance(secret, str) else secret or None,
            path=path,
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
            self._evict()
            self._dirty = True
        if self._path is not None and self._save_interval <= 0:
            self.flush()

    def flush(self) -> None:
        """Write the store to its file, if it has one and changed since it was last written."""

        if self._path is None:
            return
        with self._save_lock:
            with self._lock:
                if not self._dirty:
                    return
                now, wall = self._clock(), time.time()
                # Each node once, a parent before its children: the prefixes of one conversation share
                # their nodes, so the file grows with the trie rather than with entries times depth.
                index: dict[int, int] = {}
                nodes: list[list[Any]] = []
                entries = []
                for node in self._lru.values():  # least recently used first, as they are reloaded
                    if node.value is None or (node.expires_at is not None and node.expires_at <= now):
                        continue
                    trail, step = [], node
                    while step is not None and id(step) not in index:
                        trail.append(step)
                        step = step.parent
                    for missing in reversed(trail):
                        parent = -1 if missing.parent is None else index[id(missing.parent)]
                        index[id(missing)] = len(nodes)
                        nodes.append([missing.digest.hex(), parent])
                    expires = None if node.expires_at is None else wall + node.expires_at - now
                    entries.append([index[id(node)], node.value, expires])
                self._dirty = False
            # Values are replaced, never changed in place, so they are written outside the lock.
            data = json.dumps({"format": FORMAT, "check": self._digest(b"check").hex(), "nodes": nodes,
                               "entries": entries}, separators=(",", ":"))
            temporary = f"{self._path}.{os.getpid()}.tmp"
            try:
                os.makedirs(os.path.dirname(self._path) or ".", mode=0o700, exist_ok=True)
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    file.write(data)
                os.replace(temporary, self._path)
            except OSError:
                log.warning("could not write the prefix store to %s", self._path, exc_info=True)
                with self._lock:
                    self._dirty = True

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as file:  # type: ignore[arg-type]
                data = json.load(file)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            log.warning("could not read the prefix store from %s; starting empty", self._path, exc_info=True)
            return
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            log.warning("the prefix store in %s has an unknown format; starting empty", self._path)
            return
        if data.get("check") != self._digest(b"check").hex():
            log.warning("the prefix store in %s was written with another secret; starting empty", self._path)
            return
        now, wall = self._clock(), time.time()
        try:
            written = [(bytes.fromhex(digest), int(parent)) for digest, parent in data["nodes"]]
            entries = [(int(at), value, expires) for at, value, expires in data["entries"]]
        except (KeyError, TypeError, ValueError):
            log.warning("the prefix store in %s is damaged; starting empty", self._path)
            return
        with self._lock:
            built: list[_Node | None] = []
            for digest, parent in written:  # a parent always comes before its children
                if parent == -1:
                    node = self._roots.get(digest)
                    if node is None:
                        node = self._roots[digest] = _Node(digest, 0)
                        self._nodes += 1
                elif 0 <= parent < len(built) and (up := built[parent]) is not None:
                    node = up.children.get(digest)
                    if node is None:
                        node = up.children[digest] = _Node(digest, up.depth + 1, up)
                        self._nodes += 1
                else:
                    node = None
                built.append(node)
            for at, value, expires in entries:
                node = built[at] if 0 <= at < len(built) else None
                if node is None or not isinstance(value, dict) or (expires is not None and expires <= wall):
                    continue
                if node.value is not None:
                    self._bytes -= node.size
                node.value, node.size = value, len(json.dumps(value, separators=(",", ":")))
                node.expires_at = None if expires is None else now + expires - wall
                self._bytes += node.size
                self._lru[node.digest] = node
                self._lru.move_to_end(node.digest)
            self._evict()

    def _evict(self) -> None:
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


def _write_every(store: weakref.ref[PrefixStore], interval: float) -> None:
    """A store's writer: what changed, every `interval` seconds, for as long as the store lives."""

    while True:
        time.sleep(interval)
        if (live := store()) is None:
            return
        live.flush()
        del live


def _kept_secret(path: str) -> bytes | None:
    """The store's secret, generated on first use and kept in `path` (readable by its owner only),
    so that a restarted Relay computes the same keys; None if it cannot be kept."""

    try:
        os.makedirs(os.path.dirname(path) or ".", mode=0o700, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            with open(path, "rb") as file:
                secret = file.read()
            if len(secret) >= 16:
                return secret
            raise ValueError(f"{path} holds no usable secret") from None
        secret = secrets.token_bytes(32)
        with os.fdopen(descriptor, "wb") as file:
            file.write(secret)
        return secret
    except (OSError, ValueError):
        log.warning("the prefix store is kept in memory only: no secret could be kept in %s", path, exc_info=True)
        return None
