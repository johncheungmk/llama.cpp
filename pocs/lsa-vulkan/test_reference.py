import math
import unittest

from lsa_reference import LatentTree, attention, dot, project


class ReferenceTests(unittest.TestCase):
    def test_exhaustive_candidates_match_brute_force(self):
        keys = [[math.sin(i), math.cos(i), i / 100] for i in range(113)]
        query = [0.1, 0.9, -0.5]
        tree = LatentTree(keys, 7)
        selected, stats = tree.select(query, 11, 113)
        expected = sorted(sorted(range(113), key=lambda i: (-dot(query, keys[i]), i))[:11])
        self.assertEqual(selected, expected)
        self.assertEqual(stats["keys_scored"], 113)

    def test_future_needle_cannot_enter_causal_prefix(self):
        keys = [[float(i), 1.0] for i in range(128)]
        keys[127] = [100000.0, 1.0]
        selected, stats = LatentTree(keys, 8).select([1, 0], 4, 16, visible=73, sink=2, window=3)
        self.assertTrue(set([0, 1, 70, 71, 72]).issubset(selected))
        self.assertTrue(all(i < 73 for i in selected))
        self.assertLessEqual(stats["keys_scored"], 16)

    def test_sparse_scores_and_full_softmax_agree_on_needle(self):
        keys = [[0.0, 0.0] for _ in range(128)]
        keys[35] = [20.0, 0.0]
        values = [[float(i), -float(i)] for i in range(128)]
        query = [20.0, 0.0]
        ids, _ = LatentTree(keys, 8).select(query, 1, 8)
        self.assertEqual(ids, [35])
        for a, b in zip(attention(keys, values, query, ids), attention(keys, values, query, list(range(128)))):
            self.assertAlmostEqual(a, b, places=10)

    def test_full_selection_and_projection_geometry(self):
        keys = [[1, 2, 3], [-1, 4, 8], [2, -2, 0]]
        latent = project(keys, [[1, 0, 0], [0, 0, 2]])
        self.assertEqual(latent, [[1, 6], [-1, 16], [2, 0]])
        selected, _ = LatentTree(latent, 1).select([1, 1], 3, 3)
        self.assertEqual(selected, [0, 1, 2])

    def test_bounded_search_finds_separated_cluster(self):
        keys = [[float(i % 16), float(i // 16)] for i in range(4096)]
        query = [0.0, 1.0]
        ids, stats = LatentTree(keys, 16).select(query, 16, 256)
        exact = sorted(sorted(range(4096), key=lambda i: (-dot(query, keys[i]), i))[:16])
        self.assertEqual(ids, exact)
        self.assertLessEqual(stats["keys_scored"], 256)

    def test_invalid_inputs_are_rejected(self):
        for keys in ([], [[1], [1, 2]], [[float("nan")]], [[float("inf")]]):
            with self.assertRaises(ValueError):
                LatentTree(keys)
        tree = LatentTree([[1, 2], [3, 4]])
        for args in (([1, 2], 3, 2), ([1], 1, 2), ([1, 2], 1, 0)):
            with self.assertRaises(ValueError):
                tree.select(*args)
        for ids in ([], [0, 0], [-1], [2], [0.5]):
            with self.assertRaises(ValueError):
                attention([[1, 2], [3, 4]], [[1], [2]], [1, 1], ids)


if __name__ == "__main__":
    unittest.main()
