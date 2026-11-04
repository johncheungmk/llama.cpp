import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flash_next_calibration import (QsaBucketIndex, calibrate, check_profile,
                                    fit_projection, load_capture, qsa_scores, top_ids)

PROFILE = json.loads((Path(__file__).parent / "flash-next-profile.json").read_text())


class FlashNextTests(unittest.TestCase):
    def test_qsa_rectifies_each_head_before_sum(self):
        keys = np.zeros((2, 128), np.float32)
        keys[:, 0] = [1, -1]
        query = np.zeros((4, 128), np.float32)
        query[:, 0] = [2, -3, 0, 0]
        np.testing.assert_allclose(qsa_scores(keys, query), np.array([2, 3]) / np.sqrt(128), atol=1e-6)
        # Summing query heads before ReLU would produce [0, 1], which is wrong.

    def test_pca_recovers_rotated_low_rank_subspace(self):
        rng = np.random.default_rng(123)
        basis, _ = np.linalg.qr(rng.normal(size=(128, 8)))
        keys = (rng.normal(size=(256, 8)) @ basis.T).astype(np.float32)
        queries = (rng.normal(size=(32, 4, 8)) @ basis.T).astype(np.float32)
        projection = fit_projection(keys, queries, 8)
        np.testing.assert_allclose(projection @ projection.T, np.eye(8), atol=1e-5)
        np.testing.assert_allclose(keys @ projection.T @ projection, keys, atol=2e-6)
        np.testing.assert_allclose((queries @ projection.T) @ (keys @ projection.T).T,
                                   queries @ keys.T, atol=1e-5)

    def test_exhaustive_budget_matches_qsa_oracle_and_ties(self):
        rng = np.random.default_rng(11)
        keys = rng.normal(size=(997, 128)).astype(np.float32)
        query = rng.normal(size=(4, 128)).astype(np.float32)
        index = QsaBucketIndex(keys, np.eye(128, dtype=np.float32)[:16])
        ids, stats = index.select(query, 31, len(keys))
        expected = sorted(range(len(keys)), key=lambda i: (-float(qsa_scores(keys, query)[i]), i))[:31]
        np.testing.assert_array_equal(ids, sorted(expected))
        self.assertEqual(stats["keys_rescored"], len(keys))
        np.testing.assert_array_equal(top_ids(np.ones(100), 7), np.arange(7))

    def test_future_keys_cannot_change_prefix_selection(self):
        rng = np.random.default_rng(6)
        keys = rng.normal(size=(300, 128)).astype(np.float32)
        query = rng.normal(size=(4, 128)).astype(np.float32)
        projection = np.eye(128, dtype=np.float32)[:8]
        first, stats = QsaBucketIndex(keys, projection).select(query, 8, 64, visible=133)
        keys[133:] = rng.normal(10000, 1000, size=keys[133:].shape)
        second, _ = QsaBucketIndex(keys, projection).select(query, 8, 64, visible=133)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first < 133))
        self.assertLessEqual(stats["keys_rescored"], 64)
        self.assertTrue(stats["prefix_rebuilt"])

    def test_invalid_profile_projection_and_inputs(self):
        with self.assertRaises(ValueError):
            check_profile({**PROFILE, "indexer_heads": 1})
        with self.assertRaises(ValueError):
            QsaBucketIndex(np.zeros((1, 128)), np.ones((8, 128)))
        with self.assertRaises(ValueError):
            fit_projection(np.zeros((16, 128)), np.full((2, 4, 128), np.nan), 4)
        index = QsaBucketIndex(np.zeros((16, 128)), np.eye(128)[:8])
        for k, budget, visible in ((5, 4, 16), (1, 0, 16), (1, 4, 0), (1, 4, 17)):
            with self.assertRaises(ValueError):
                index.select(np.zeros((4, 128)), k, budget, visible)

    def test_capture_provenance_disjointness_and_no_pickle(self):
        rng = np.random.default_rng(55)
        with tempfile.TemporaryDirectory() as directory:
            def capture(name, document):
                path = Path(directory) / name
                manifest = {"schema": 1, "source_kind": "synthetic_fixture", "stage": "post_norm_post_rope_pooled_indexer",
                            "model_id": PROFILE["model_id"], "layer": 3, "document_ids": [document]}
                np.savez(path, keys=rng.normal(size=(1024, 128)).astype(np.float32),
                         queries=rng.normal(size=(8, 4, 128)).astype(np.float32),
                         visible=np.full(8, 1024, dtype=np.int64), manifest=json.dumps(manifest))
                return path
            train, heldout = capture("train.npz", "train"), capture("heldout.npz", "heldout")
            projection, report = calibrate(train, heldout, PROFILE, 8, 512)
            self.assertEqual(projection.shape, (8, 128))
            self.assertFalse(report["serving_ready"])
            self.assertEqual(report["source_kind"], "synthetic_fixture")
            self.assertEqual(report["queries_requiring_pruning"], 8)
            with self.assertRaises(ValueError):
                calibrate(train, train, PROFILE, 8, 512)
            overlap = capture("overlap.npz", "train")
            with self.assertRaises(ValueError):
                calibrate(train, overlap, PROFILE, 8, 512)
            malicious = Path(directory) / "object.npz"
            np.savez(malicious, keys=np.array([{}], dtype=object), queries=[1], visible=[1], manifest="{}")
            with self.assertRaises(ValueError):
                load_capture(malicious, PROFILE)


if __name__ == "__main__":
    unittest.main()
