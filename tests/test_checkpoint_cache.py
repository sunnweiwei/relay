from __future__ import annotations

import unittest

from relay import PrefixCheckpointCache


def message(text: str) -> dict:
    return {"type": "message", "role": "user", "content": text}


def artifact(text: str) -> dict:
    return {"kind": "test", "value": text}


class PrefixCheckpointCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = PrefixCheckpointCache(
            max_entries=16,
            max_bytes=1_000_000,
            max_nodes=1_000,
            ttl_seconds=None,
            secret=b"test-secret",
        )
        self.partition = self.cache.partition("tenant-a", {"model": "test"})

    def test_longest_exact_prefix_survives_branching(self) -> None:
        root = message("root")
        branch_a = message("branch-a")
        self.cache.put(self.partition, [root], artifact("root checkpoint"))
        self.cache.put(
            self.partition,
            [root, branch_a],
            artifact("branch-a checkpoint"),
        )

        exact_branch = self.cache.match(
            self.partition, [root, branch_a, message("tail")]
        )
        sibling_branch = self.cache.match(self.partition, [root, message("branch-b")])
        changed_root = self.cache.match(self.partition, [message("changed"), branch_a])

        assert exact_branch is not None and sibling_branch is not None
        self.assertEqual(exact_branch.matched_items, 2)
        self.assertEqual(exact_branch.artifact, artifact("branch-a checkpoint"))
        self.assertEqual(sibling_branch.matched_items, 1)
        self.assertEqual(sibling_branch.artifact, artifact("root checkpoint"))
        self.assertIsNone(changed_root)

    def test_object_key_order_is_canonical_but_content_is_exact(self) -> None:
        reordered = {"content": "root", "role": "user", "type": "message"}
        self.cache.put(self.partition, [message("root")], artifact("checkpoint"))

        self.assertIsNotNone(self.cache.match(self.partition, [reordered]))
        self.assertIsNone(self.cache.match(self.partition, [message("Root")]))

    def test_reasoning_absent_content_matches_null_after_resume(self) -> None:
        reasoning = {
            "type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "checked config"}],
            "encrypted_content": "opaque-reasoning",
        }
        original = [message("inspect config"), reasoning, message("750 ms")]
        self.cache.put(self.partition, original, artifact("summary checkpoint"))

        resumed_reasoning = {**reasoning, "content": None}
        resumed = [original[0], resumed_reasoning, original[2], message("continue")]
        match = self.cache.match(self.partition, resumed)

        assert match is not None
        self.assertEqual(match.matched_items, 3)
        self.assertEqual(match.artifact, artifact("summary checkpoint"))
        self.assertNotIn("content", reasoning)
        self.assertIsNone(resumed_reasoning["content"])

        changed_reasoning = {**resumed_reasoning, "encrypted_content": "different"}
        self.assertIsNone(self.cache.match(
            self.partition, [original[0], changed_reasoning, original[2]]
        ))
        visible_content = {**reasoning, "content": [{"type": "reasoning_text", "text": "new"}]}
        self.assertIsNone(self.cache.match(
            self.partition, [original[0], visible_content, original[2]]
        ))

    def test_null_content_is_not_ignored_for_other_item_types(self) -> None:
        assistant = {"type": "message", "role": "assistant"}
        self.cache.put(self.partition, [assistant], artifact("checkpoint"))

        self.assertIsNone(self.cache.match(
            self.partition, [{**assistant, "content": None}]
        ))

    def test_tenants_and_request_scopes_cannot_cross_hit(self) -> None:
        tenant_b = self.cache.partition("tenant-b", {"model": "test"})
        other_model = self.cache.partition("tenant-a", {"model": "other"})
        trajectory = [message("private")]
        self.cache.put(self.partition, trajectory, artifact("secret checkpoint"))

        self.assertIsNone(self.cache.match(tenant_b, trajectory))
        self.assertIsNone(self.cache.match(other_model, trajectory))

    def test_lru_evicts_the_least_recently_used_checkpoint(self) -> None:
        cache = PrefixCheckpointCache(
            max_entries=2,
            max_bytes=1_000_000,
            max_nodes=100,
            ttl_seconds=None,
            secret=b"test-secret",
        )
        partition = cache.partition("tenant", {"model": "test"})
        cache.put(partition, [message("a")], artifact("checkpoint-a"))
        cache.put(partition, [message("b")], artifact("checkpoint-b"))
        self.assertIsNotNone(cache.match(partition, [message("a")]))
        cache.put(partition, [message("c")], artifact("checkpoint-c"))

        self.assertIsNotNone(cache.match(partition, [message("a")]))
        self.assertIsNone(cache.match(partition, [message("b")]))
        self.assertIsNotNone(cache.match(partition, [message("c")]))
        self.assertEqual(cache.stats().entries, 2)
        self.assertEqual(cache.stats().evictions, 1)

    def test_ttl_expiry_turns_a_hit_into_a_normal_miss(self) -> None:
        now = [0.0]
        cache = PrefixCheckpointCache(
            max_entries=4,
            max_bytes=1_000_000,
            max_nodes=100,
            ttl_seconds=10,
            secret=b"test-secret",
            clock=lambda: now[0],
        )
        partition = cache.partition("tenant", {"model": "test"})
        cache.put(partition, [message("a")], artifact("checkpoint"))
        self.assertIsNotNone(cache.match(partition, [message("a")]))

        now[0] = 11
        self.assertIsNone(cache.match(partition, [message("a")]))
        self.assertEqual(cache.stats().entries, 0)


if __name__ == "__main__":
    unittest.main()
