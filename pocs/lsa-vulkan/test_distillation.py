import unittest

import numpy as np

from flash_next_calibration import fit_projection
from train_qsa_projection import fit, objective


class DistillationTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(223)
        keys = rng.normal(size=(17, 128))
        queries = rng.normal(size=(2, 4, 128))
        projection = rng.normal(0, 0.1, size=(4, 128))
        visible = np.array([9, 17])
        _, gradient = objective(keys, queries, visible, projection)
        for row, col in ((0, 0), (0, 17), (1, 4), (2, 55), (3, 127)):
            plus, minus = projection.copy(), projection.copy()
            plus[row, col] += 1e-5
            minus[row, col] -= 1e-5
            numeric = (objective(keys, queries, visible, plus)[0] - objective(keys, queries, visible, minus)[0]) / 2e-5
            self.assertAlmostEqual(float(gradient[row, col]), numeric, places=6)

    def test_training_reduces_objective_and_preserves_geometry(self):
        rng = np.random.default_rng(441)
        keys = rng.normal(size=(128, 128)).astype(np.float32)
        queries = rng.normal(size=(16, 4, 128)).astype(np.float32)
        initial = fit_projection(keys, queries, 16)
        projection, report = fit(keys, queries, np.full(16, 128), initial, steps=8)
        np.testing.assert_allclose(projection @ projection.T, np.eye(16), atol=1e-5)
        self.assertGreater(report["accepted_steps"], 0)
        self.assertLess(report["training_kl_trace"][-1], report["training_kl_trace"][0])
        self.assertFalse(report["heldout_used_for_training"])

    def test_masked_future_keys_do_not_change_objective(self):
        rng = np.random.default_rng(822)
        keys = rng.normal(size=(17, 128))
        queries = rng.normal(size=(1, 4, 128))
        projection = rng.normal(0, 0.1, size=(8, 128))
        first, grad = objective(keys, queries, np.array([9]), projection)
        keys[9:] *= 10000
        second, other = objective(keys, queries, np.array([9]), projection)
        self.assertAlmostEqual(first, second, places=12)
        np.testing.assert_allclose(grad, other, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
