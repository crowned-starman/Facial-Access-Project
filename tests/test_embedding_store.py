"""Unit tests without webcam or InsightFace weights."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
from core.embedding_store import EmbeddingStore


def vector(index=0):
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1.0
    return result


class TestEmbeddingStore(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = patch.object(config, "EMBEDDINGS_DIR", self.folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = EmbeddingStore()

    def test_empty(self):
        self.assertEqual(self.store.find_match(vector()), (None, 0.0))

    def test_save_reload_match_and_delete(self):
        self.store.save_user("user_1", np.stack([vector(), vector()]))
        loaded = EmbeddingStore()
        self.assertEqual(loaded.find_match(vector())[0], "user_1")
        self.assertIsNone(loaded.find_match(vector(1))[0])
        self.assertTrue(loaded.delete_user("user_1"))
        self.assertFalse(loaded.delete_user("user_1"))
        self.assertFalse((Path(self.folder.name) / "user_1.npy").exists())

    def test_invalid_names(self):
        for name in ("../outside", "a/b", "", ".hidden", "a" * 65, "á"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.save_user(name, np.stack([vector()]))

    def test_invalid_vectors(self):
        for bad in (
            np.zeros((0, 512)),
            np.zeros((1, 511)),
            np.zeros((1, 512)),
            np.full((1, 512), np.nan),
            np.full((1, 512), np.inf),
            np.ones((101, 512)),
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.store.save_user("invalid", bad)
        self.assertFalse((Path(self.folder.name) / "invalid.npy").exists())

    def test_invalid_query_and_threshold(self):
        self.store.save_user("user", np.stack([vector()]))
        for bad in (np.zeros(512), np.full(512, np.nan), np.zeros(511)):
            with self.assertRaises(ValueError):
                self.store.find_match(bad)
        for threshold in (-1, 0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                self.store.find_match(vector(), threshold=threshold)

    def test_corrupt_file_ignored(self):
        (Path(self.folder.name) / "corrupt.npy").write_bytes(b"invalid")
        self.assertEqual(EmbeddingStore().users, [])

    def test_normalizes(self):
        self.store.save_user("user", np.stack([vector() * 4]))
        who, score = self.store.find_match(vector() * 9)
        self.assertEqual(who, "user")
        self.assertAlmostEqual(score, 1.0, places=5)

    @unittest.skipUnless(os.name == "posix", "POSIX only")
    def test_file_permissions(self):
        self.store.save_user("user", np.stack([vector()]))
        mode = (Path(self.folder.name) / "user.npy").stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
