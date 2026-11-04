import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flash_next_calibration import load_capture
from native_score_reference import scores, verify, verify_projection
from capture_flash_next_cpu import capture_tensor_names


class NativeReferenceTests(unittest.TestCase):
    def test_cooperative_operand_oracle_requires_f32_accumulation_accuracy(self):
        rng = np.random.default_rng(909)
        q, k = rng.normal(size=(8, 4, 16)), rng.normal(size=(17, 16))
        rounded_q, rounded_k = q.astype(np.float16).astype(np.float64), k.astype(np.float16).astype(np.float64)
        native = scores(rounded_q, rounded_k).astype(np.float32)
        mask = np.ones(native.shape, bool)
        report = verify(q, k, native, mask, mixed_operands=True)
        self.assertEqual(report["operand_oracle"], "F16_both")
        with self.assertRaises(RuntimeError):
            verify(q, k, native + 0.001, mask, mixed_operands=True)
        x = rng.normal(size=(7, 128))
        p = np.linalg.qr(rng.normal(size=(128, 16)))[0].T
        projected = (x.astype(np.float16).astype(np.float64) @ p.astype(np.float16).astype(np.float64).T).astype(np.float32)
        self.assertEqual(verify_projection(x, p, projected, True)["operand_oracle"], "F16_both")
        with self.assertRaises(RuntimeError):
            verify_projection(x, p, projected + 0.01, True)

    def test_native_projected_pooled_key_uses_final_callback_name(self):
        layers = list(range(3, 48, 4))
        original = capture_tensor_names(layers, False)
        projected = capture_tensor_names(layers, True)
        self.assertEqual(len(original), 36)
        self.assertEqual(len(projected), 72)
        self.assertEqual(len(set(projected.values())), 72)
        for layer in layers:
            self.assertEqual(original[f"indexer_k-{layer}"], (layer, "keys"))
            self.assertNotIn(f"indexer_k-{layer}", projected)
            self.assertEqual(projected[f"experimental_qsa_pca_k-{layer}"], (layer, "keys"))

    def test_half_dot4_arithmetic_is_inside_derived_bound(self):
        rng = np.random.default_rng(631)
        q, k = rng.normal(size=(2, 4, 128)).astype(np.float32), rng.normal(size=(3, 128)).astype(np.float32)
        native = np.zeros((2, 3), np.float32)
        for token in range(2):
            for key in range(3):
                for head in range(4):
                    products = q[token, head].astype(np.float16) * k[key].astype(np.float16)
                    groups = products.reshape(32, 4).sum(axis=1, dtype=np.float16)
                    dot = groups.astype(np.float32).sum(dtype=np.float32)
                    native[token, key] += max(dot, 0) / np.sqrt(128)
        result = verify(q, k, native, np.ones((2, 3), dtype=bool), half_tile=True)
        self.assertLessEqual(result["max_error_to_bound_ratio"], 1)
        with self.assertRaises(RuntimeError):
            verify(q, k, native + 1000, np.ones((2, 3), dtype=bool), half_tile=True)
        verify(q, k, scores(q, k), np.ones((2, 3), dtype=bool), half_tile=False)
        tiny_q, tiny_k = np.full_like(q, 1e-5), np.full_like(k, 1e-5)
        verify(tiny_q, tiny_k, np.zeros((2, 3)), np.ones((2, 3), dtype=bool), half_tile=True)

    def test_optional_native_teacher_archive_is_bounded_and_validated(self):
        root = Path(__file__).parent
        profile = json.loads((root / "flash-next-profile.json").read_text())
        manifest = {"schema": 1, "source_kind": "synthetic_fixture", "stage": "post_norm_post_rope_pooled_indexer",
                    "model_id": profile["model_id"], "layer": 3, "document_ids": ["test"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npz"
            def write(teacher):
                np.savez(path, keys=np.zeros((64, 128), np.float32), queries=np.zeros((8, 4, 128), np.float32),
                         visible=np.full(8, 64, np.int64), native_scores=teacher, manifest=json.dumps(manifest))
            write(np.zeros((8, 64), np.float32))
            result = load_capture(path, profile)
            self.assertEqual(result[4].shape, (8, 64))
            write(np.full((8, 64), -1, np.float32))
            with self.assertRaises(ValueError):
                load_capture(path, profile)


if __name__ == "__main__":
    unittest.main()
