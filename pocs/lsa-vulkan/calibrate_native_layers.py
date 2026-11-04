"""Fit all twelve experimental maps, keeping held-out captures out of training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from export_native_maps import export
from flash_next_calibration import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True, help="all-layer capture JSON basename")
    parser.add_argument("--heldout", type=Path, required=True, help="all-layer capture JSON basename")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(16, 32, 64), default=64)
    args = parser.parse_args()
    for path in (args.train, args.heldout):
        if path.stat().st_size > 256 * 1024:
            parser.error("capture summary exceeds 256 KiB")
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(summary, list) or len(summary) != 12 or any(
                entry.get("tokens") != 4096 or entry.get("source_kind") != "real_model_capture" or
                entry.get("layer") != 3 + 4 * index for index, entry in enumerate(summary)):
            parser.error("this batch experiment requires all 12 ordered real 4096-token layers")
    args.output_directory.mkdir(exist_ok=False)
    profile = json.loads(args.profile.read_text(encoding="utf-8-sig"))
    reports = {}
    for method in ("pca", "distill"):
        directory = args.output_directory / method
        directory.mkdir()
        layers = []
        for layer in range(3, 48, 4):
            train = args.train.with_name(f"{args.train.stem}.layer-{layer}.npz")
            heldout = args.heldout.with_name(f"{args.heldout.stem}.layer-{layer}.npz")
            projection, report = calibrate(train, heldout, profile, args.rank, 1024, method=method, steps=50)
            path = directory / f"layer-{layer}-rank{args.rank}.npz"
            with path.open("xb") as stream:
                np.savez(stream, projection=projection, manifest=json.dumps(report))
            report["map_archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            layers.append(report)
            print("LAYER_COMPLETE", method, layer, report["native_flat_projected_top512_nontrivial_recall_mean"], flush=True)
        reports[method] = {"layers": layers, "export": export(args.output_directory / f"{method}-rank{args.rank}.qpc", directory, args.rank)}
    result = {"schema": 1, "kind": "real_checkpoint_4K_calibration", "rank": args.rank,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in (Path(__file__).name, "flash_next_calibration.py", "train_qsa_projection.py", "export_native_maps.py")},
              "methods": reports, "serving_ready": False,
              "limitations": "Two synthetic public-example documents, 4096 model tokens each; 32 sampled late-position queries per layer. Native flat recall is the diagnostic relevant to the native implementation. Bucket reranking recall with 1024 candidates is trivially exact at 1024 captured pools and does not validate search. No 256K generalization or model quality approval."}
    with (args.output_directory / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
