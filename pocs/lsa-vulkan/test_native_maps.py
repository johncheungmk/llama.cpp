import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import numpy as np

from compare_model_quality import compare
from export_native_maps import export


class NativeMapTests(unittest.TestCase):
    def test_export_rejects_oversized_expanded_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "layer-3-rank16.npz", "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("projection.npy", b"\0" * (1024**2 + 1))
                archive.writestr("manifest.npy", b"")
            with self.assertRaisesRegex(ValueError, "expanded map archive"):
                export(root / "map.qpc", root, rank=16)
            self.assertFalse((root / "map.qpc").exists())

    def test_identity_export_layout_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.qpc"
            report = export(path, rank=128, identity=True)
            data = path.read_bytes()
            self.assertEqual(data[:8], b"QPCAMAP1")
            self.assertEqual(struct.unpack_from("<III", data, 8), (128, 128, 12))
            self.assertEqual(len(data), 20 + 12 * (4 + 128 * 128 * 4))
            self.assertTrue(report["identity_reference"])
            self.assertFalse(report["serving_ready"])
            self.assertEqual(struct.unpack_from("<I", data, 20)[0], 3)
            np.testing.assert_array_equal(np.frombuffer(data, dtype="<f4", count=128 * 128, offset=24).reshape(128, 128), np.eye(128))
            with self.assertRaises(ValueError):
                export(path, rank=128, identity=True)
            with self.assertRaises(ValueError):
                export(Path(directory) / "bad.qpc", rank=16, identity=True)

    def test_model_quality_identity_and_changed_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits = np.array([[0.3, 0.4, 0.5], [0, 2, 1], [1, 0, 3], [0, 1, 2]], np.float32)
            manifest = {"model_id": "test", "prompt_sha256": "test-prompt", "tokens": 4096, "gpu_layers": 4,
                        "context_tokens": 8192, "batch_tokens": 256, "cpu_threads": 4, "attention_kv_types": "Q8_0/Q8_0",
                        "vocabulary_size": 3, "runtime_dll_sha256": {}, "source_sha256": "test-source", "support_source_sha256": {},
                        "prefill_seconds": 1.0, "decode_tokens_per_second": 1.0, "generated_token_ids": [1, 2]}
            def write(name, values, meta=manifest):
                path = root / name
                np.savez(path, logits=values, targets=np.array([1, 1, 2]), manifest=json.dumps(meta))
                return path
            baseline = write("baseline.npz", logits)
            identical = write("same.npz", logits)
            result = compare(baseline, identical)
            self.assertEqual(result["max_abs_logit_difference"], 0)
            self.assertEqual(result["mean_nll_delta"], 0)
            self.assertEqual(result["mean_kl_baseline_to_candidate"], 0)
            self.assertEqual(result["logit_top1_agreement"], 1)
            changed = logits.copy()
            changed[0, 0] += 0.5
            altered = compare(baseline, write("changed.npz", changed))
            self.assertGreater(altered["mean_kl_baseline_to_candidate"], 0)
            self.assertGreater(altered["mean_nll_delta"], 0)

    def test_quality_comparison_rejects_unmatched_input(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.npz", Path(directory) / "second.npz"
            for path in (first, second):
                np.savez(path, logits=np.zeros((4, 3), np.float32), targets=np.zeros(3, np.int64), manifest=json.dumps({}))
            with self.assertRaises(ValueError):
                compare(first, second)


if __name__ == "__main__":
    unittest.main()
