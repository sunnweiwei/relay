from __future__ import annotations

import unittest

from relay.core.store import PrefixStore


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


KEYS = [b"a", b"b", b"c", b"d"]


class PrefixStoreTests(unittest.TestCase):
    def test_returns_the_deepest_stored_prefix(self) -> None:
        store = PrefixStore()
        partition = store.partition("tenant")
        store.put(partition, KEYS[:1], {"v": 1})
        store.put(partition, KEYS[:3], {"v": 3})
        self.assertEqual(store.match(partition, KEYS), (3, {"v": 3}))
        self.assertEqual(store.match(partition, [b"a", b"x", b"c"]), (1, {"v": 1}))
        self.assertIsNone(store.match(partition, [b"x"]))

    def test_partitions_are_isolated(self) -> None:
        store = PrefixStore()
        store.put(store.partition("a"), KEYS, {"v": 1})
        self.assertIsNone(store.match(store.partition("b"), KEYS))

    def test_values_expire(self) -> None:
        clock = Clock()
        store = PrefixStore(ttl_seconds=10, clock=clock)
        partition = store.partition("t")
        store.put(partition, KEYS, {"v": 1})
        clock.now = 11
        self.assertIsNone(store.match(partition, KEYS))
        self.assertEqual(len(store), 0)

    def test_least_recently_used_values_are_evicted(self) -> None:
        store = PrefixStore(max_entries=2)
        partition = store.partition("t")
        store.put(partition, KEYS[:1], {"v": 1})
        store.put(partition, KEYS[:2], {"v": 2})
        store.match(partition, KEYS[:1])  # refresh the first entry
        store.put(partition, KEYS[:3], {"v": 3})
        self.assertEqual(store.match(partition, KEYS[:2]), (1, {"v": 1}))
        self.assertEqual(len(store), 2)

    def test_stored_values_are_copies(self) -> None:
        store = PrefixStore()
        partition = store.partition("t")
        value = {"head": [1]}
        store.put(partition, KEYS, value)
        value["head"].append(2)
        found = store.match(partition, KEYS)[1]
        found["head"].append(3)
        self.assertEqual(store.match(partition, KEYS)[1], {"head": [1]})


if __name__ == "__main__":
    unittest.main()
