"""The prefix store kept on disk (RELAY_CACHE_PATH with RELAY_CACHE_SECRET): a restarted Relay
finds the contexts it had. Off by default; a store without a fixed secret stays in memory."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.core.store import PrefixStore
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from relay.strategies import Compaction, ContextLanguageModel
from tests.test_engine import Upstream, body, msg, step

SECRET = b"s" * 32
CODEC, HARNESS = OpenAIResponses(), Harness()


class StoreOnDiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(tempfile.mkdtemp()) / "store.json"

    def store(self, **options: Any) -> PrefixStore:
        options.setdefault("secret", SECRET)
        return PrefixStore(path=self.path, **options)

    def test_a_restarted_store_finds_what_was_put(self) -> None:
        store = self.store()
        partition = store.partition("tenant")
        store.put(partition, [b"a", b"b"], {"covered": 2})
        store.put(partition, [b"a", b"b", b"c"], {"covered": 3})
        store.flush()
        again = self.store()
        self.assertEqual(again.match(partition, [b"a", b"b", b"c", b"d"]), (3, {"covered": 3}))
        self.assertEqual(again.match(partition, [b"a", b"b", b"x"]), (2, {"covered": 2}))
        self.assertEqual(len(again), 2)

    def test_only_digests_are_written_never_the_items(self) -> None:
        store = self.store()
        store.put(store.partition("sk-secret-key"), [b"the user's prompt"], {"v": 1})
        store.flush()
        text = self.path.read_text()
        self.assertNotIn("the user's prompt", text)
        self.assertNotIn("sk-secret-key", text)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_another_secret_starts_empty(self) -> None:
        store = self.store()
        store.put(store.partition("t"), [b"a"], {"v": 1})
        store.flush()
        with self.assertLogs("relay", "WARNING"):
            other = self.store(secret=b"o" * 32)
        self.assertEqual(len(other), 0)

    def test_without_a_fixed_secret_nothing_is_written(self) -> None:
        with self.assertLogs("relay", "WARNING"):
            store = PrefixStore(path=self.path)
        store.put(store.partition("t"), [b"a"], {"v": 1})
        store.flush()
        self.assertFalse(self.path.exists())

    def test_without_a_path_nothing_is_written(self) -> None:
        directory = self.path.parent
        store = PrefixStore(secret=SECRET)
        store.put(store.partition("t"), [b"a"], {"v": 1})
        store.flush()
        self.assertEqual(list(directory.iterdir()), [])

    def test_expiry_holds_across_a_restart(self) -> None:
        store = self.store(ttl_seconds=0.2)
        partition = store.partition("t")
        store.put(partition, [b"a"], {"v": 1})
        store.flush()
        self.assertIsNotNone(self.store(ttl_seconds=0.2).match(partition, [b"a"]))
        time.sleep(0.3)
        self.assertIsNone(self.store(ttl_seconds=0.2).match(partition, [b"a"]))

    def test_limits_hold_when_loading(self) -> None:
        store = self.store()
        partition = store.partition("t")
        for n in range(5):
            store.put(partition, [bytes([n])], {"v": n})
        store.flush()
        small = self.store(max_entries=2)
        self.assertEqual(len(small), 2)
        self.assertIsNotNone(small.match(partition, [bytes([4])]))  # the most recent are kept
        self.assertIsNone(small.match(partition, [bytes([0])]))

    def test_a_damaged_file_starts_empty(self) -> None:
        self.path.write_text("{not json")
        with self.assertLogs("relay", "WARNING"):
            store = self.store()
        self.assertEqual(len(store), 0)
        store.put(store.partition("t"), [b"a"], {"v": 1})
        store.flush()
        self.assertEqual(json.loads(self.path.read_text())["format"], 1)

    def test_writes_are_throttled_and_flush_writes_the_rest(self) -> None:
        store = self.store(save_interval=3600)
        store.put(store.partition("t"), [b"a"], {"v": 1})
        self.assertFalse(self.path.exists())
        store.flush()
        self.assertTrue(self.path.exists())
        eager = PrefixStore(path=self.path.with_name("eager.json"), secret=SECRET, save_interval=0)
        eager.put(eager.partition("t"), [b"a"], {"v": 1})
        self.assertTrue(self.path.with_name("eager.json").exists())

    def test_by_default_relay_keeps_its_store_and_secret_in_the_home_folder(self) -> None:
        home = self.path.parent
        with unittest.mock.patch.dict(os.environ, {"HOME": str(home)}):
            os.environ.pop("RELAY_CACHE_PATH", None), os.environ.pop("RELAY_CACHE_SECRET", None)
            store = PrefixStore.from_env()
            partition = store.partition("t")
            store.put(partition, [b"a"], {"v": 1})
            store.flush()
            secret = home / ".relay" / "store.json.secret"
            self.assertEqual(secret.stat().st_mode & 0o777, 0o600)
            self.assertEqual((home / ".relay").stat().st_mode & 0o777, 0o700)
            self.assertEqual(PrefixStore.from_env().match(partition, [b"a"]), (1, {"v": 1}))  # restarted

    def test_off_keeps_the_store_in_memory(self) -> None:
        home = self.path.parent
        with unittest.mock.patch.dict(os.environ, {"HOME": str(home), "RELAY_CACHE_PATH": "off"}):
            store = PrefixStore.from_env()
        store.put(store.partition("t"), [b"a"], {"v": 1})
        store.flush()
        self.assertEqual(list(home.iterdir()), [])

    def test_from_env(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"RELAY_CACHE_PATH": str(self.path), "RELAY_CACHE_SECRET": "k"}):
            store = PrefixStore.from_env()
        store.put(store.partition("t"), [b"a"], {"v": 1})
        store.flush()
        self.assertTrue(self.path.exists())


class RestartTests(unittest.TestCase):
    """A restarted Relay is a new Engine with a new store loaded from the same file."""

    def setUp(self) -> None:
        self.path = Path(tempfile.mkdtemp()) / "store.json"

    def store(self) -> PrefixStore:
        return PrefixStore(path=self.path, secret=SECRET, save_interval=0)

    def test_an_empty_store_given_to_the_engine_is_the_one_used(self) -> None:
        store = self.store()
        self.assertIs(Engine(Compaction(), store).store, store)

    def test_compaction_is_not_recomputed_after_a_restart(self) -> None:
        upstream = Upstream()
        history = [msg("developer", "rules"), msg("user", "task"), *step(1), *step(2)]
        first = Engine(Compaction(threshold=300, min_gain=0), self.store()).prepare(
            CODEC, HARNESS, body(*history), tenant="t", post=upstream)
        self.assertTrue(first.compacted)
        restarted = Engine(Compaction(threshold=300, min_gain=0), self.store())
        later = restarted.prepare(CODEC, HARNESS, body(*history, *step(3, size=10)), tenant="t", post=upstream)
        self.assertEqual(len(upstream.requests), 1)  # no second summary
        self.assertEqual(later.body["input"][:3], first.body["input"])

    def test_clm_edits_survive_a_restart(self) -> None:
        directory = tempfile.mkdtemp()

        def engine() -> Engine:
            return Engine(ContextLanguageModel(budget=200_000, directory=directory), self.store())

        def unreachable(request: dict[str, Any]) -> tuple[int, Any]:
            raise AssertionError("CLM makes no requests of its own")

        def send(relay: Engine, *items: dict[str, Any]) -> str:
            tools = [{"type": "function", "name": "sh", "parameters": {}}]
            exchange = relay.prepare(CODEC, HARNESS, {"model": "m", "input": list(items), "tools": tools},
                                     tenant="t", post=unreachable)
            return json.dumps(exchange.body)

        history = [msg("user", "read the files"), *step(1, 2_000), *step(2, 2_000)]
        relay = engine()
        send(relay, *history)
        mirror = next(Path(directory).glob("*.md"))
        mirror.write_text(re.sub(r"x{2000}", "nothing useful", mirror.read_text(), count=1))
        history += step(3, 10)
        self.assertIn("nothing useful", send(relay, *history))
        sent = send(engine(), *history, *step(4, 10))  # Relay restarted
        self.assertIn("nothing useful", sent)
        self.assertEqual(sent.count("x" * 2_000), 1)  # the removed output stays removed
        self.assertEqual(len(list(Path(directory).glob("*.md"))), 1)  # the same conversation, the same file


if __name__ == "__main__":
    unittest.main()
