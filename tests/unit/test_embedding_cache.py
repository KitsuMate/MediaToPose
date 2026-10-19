"""Prompt embedding cache keyed by encoder, variant, template and prompt."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from media_to_pose.core.cache import embeddings

KEY = embeddings.EmbeddingKey("a" * 40, "INT4", 1, "A person walks.")


def _vector(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(4096).astype(np.float32)


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.cache = embeddings.EmbeddingCache(self.directory)

    def test_key_is_the_sha256_of_the_four_fields(self) -> None:
        expected = hashlib.sha256(
            f"{'a' * 40}|INT4|1|A person walks.".encode()
        ).hexdigest()
        self.assertEqual(KEY.digest, expected)
        for changed in (
            replace(KEY, encoder_revision="b" * 40),
            replace(KEY, variant="BF16"),
            replace(KEY, template_version=2),
            replace(KEY, prompt="A person runs."),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(changed.digest, KEY.digest)

    def test_round_trip_keeps_the_vector_and_the_encoded_text(self) -> None:
        self.assertIsNone(self.cache.load(KEY))
        vector = _vector()
        self.cache.store(KEY, vector, text="A person.", truncated=True)
        cached = self.cache.load(KEY)
        assert cached is not None
        np.testing.assert_array_equal(cached.vector, vector)
        self.assertEqual((cached.text, cached.truncated), ("A person.", True))
        self.assertEqual(cached.vector.dtype, np.float32)
        with np.load(self.directory / f"{KEY.digest}.npz", allow_pickle=False) as data:
            self.assertEqual(data["embedding"].shape, (4096,))

    def test_entry_whose_manifest_disagrees_is_ignored(self) -> None:
        self.cache.store(KEY, _vector())
        other = replace(KEY, prompt="Something else.")
        manifest = self.directory / f"{KEY.digest}.json"
        manifest.replace(self.directory / f"{other.digest}.json")
        (self.directory / f"{KEY.digest}.npz").replace(
            self.directory / f"{other.digest}.npz"
        )
        self.assertIsNone(self.cache.load(other))

    def test_damaged_or_invalid_entries_are_misses(self) -> None:
        self.cache.store(KEY, _vector())
        (self.directory / f"{KEY.digest}.npz").write_bytes(b"not an npz")
        self.assertIsNone(self.cache.load(KEY))
        with self.assertRaises(ValueError):
            self.cache.store(KEY, np.full(4096, np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.cache.store(KEY, np.zeros(16, dtype=np.float32))

    def test_least_recently_used_entries_are_pruned_beyond_the_limit(self) -> None:
        keys = [replace(KEY, prompt=f"Prompt {index}.") for index in range(3)]
        for index, key in enumerate(keys):
            self.cache.store(key, _vector(index))
            stamp = 1_000_000 + index
            os.utime(self.directory / f"{key.digest}.json", (stamp, stamp))
        self.assertIsNotNone(self.cache.load(keys[0]))  # Now the most recent.
        limit = self.cache.usage_bytes() - 1  # One entry too many.
        limited = embeddings.EmbeddingCache(self.directory, limit_bytes=limit)
        limited.prune()
        self.assertIsNotNone(limited.load(keys[0]))
        self.assertIsNone(limited.load(keys[1]))
        self.assertIsNotNone(limited.load(keys[2]))
        limited.clear()
        self.assertEqual(limited.usage_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
