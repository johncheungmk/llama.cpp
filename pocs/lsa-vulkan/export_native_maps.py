"""Export experimental Flash Next PCA maps to the bounded native QPCAMAP1 format."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np


def export(output, directory=None, rank=16, identity=False):
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise ValueError("output or sidecar already exists")
    if rank not in (16, 32, 64, 128) or (identity and rank != 128):
        raise ValueError("supported ranks are 16/32/64/128; identity requires rank 128")
    data = bytearray(b"QPCAMAP1" + struct.pack("<III", 128, rank, 12))
    sources = {}
    for layer in range(3, 48, 4):
        if identity:
            projection = np.eye(128, dtype=np.float32)
        else:
            path = Path(directory) / f"layer-{layer}-rank{rank}.npz"
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("map archive exceeds 1 MiB")
            with zipfile.ZipFile(path) as archive:
                if len(archive.infolist()) != 2 or sum(x.file_size for x in archive.infolist()) > 1024**2:
                    raise ValueError("expanded map archive exceeds bounds")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"projection", "manifest"}:
                    raise ValueError("unexpected map members")
                projection = np.asarray(archive["projection"], dtype=np.float32)
                manifest = json.loads(str(archive["manifest"]))
            if manifest.get("source_kind") != "real_model_capture" or manifest.get("model_id") != "Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL" or manifest.get("layer") != layer or manifest.get("rank") != rank:
                raise ValueError("map provenance/model/layer/rank differs")
            sources[str(layer)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if projection.shape != (rank, 128) or not np.isfinite(projection).all() or not np.allclose(projection @ projection.T, np.eye(rank), atol=1e-4):
            raise ValueError("projection shape/values/orthogonality invalid")
        data.extend(struct.pack("<I", layer))
        data.extend(np.asarray(projection, dtype="<f4").tobytes())
    report = {"schema": 1, "format": "QPCAMAP1", "rank": rank, "identity_reference": identity,
              "model_id": "Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL", "serving_ready": False,
              "source_map_sha256": sources, "sha256": hashlib.sha256(data).hexdigest(),
              "scope": "Experimental flat PCA indexer scoring; no hierarchical search or quality approval. Native geometry checks do not authenticate checkpoint weights."}
    with output.open("xb") as stream:
        stream.write(data)
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--map-directory", type=Path)
    group.add_argument("--identity", action="store_true")
    parser.add_argument("--rank", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.map_directory, args.rank, args.identity), indent=2))


if __name__ == "__main__":
    main()
