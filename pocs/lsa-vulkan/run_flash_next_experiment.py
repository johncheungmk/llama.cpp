"""Measured offline QSA search and integrated synthetic Vulkan latency; no model loaded."""
import os

# Pin BLAS before importing NumPy; also applies to the small PCA fit.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import array
import ctypes as C
import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from flash_next_calibration import QsaBucketIndex, fit_projection, qsa_scores, top_ids
from ggml_vulkan import Vulkan
from lsa_reference import LatentTree, attention
from run_experiment import fixture


def qsa_fixture(pools, seed, basis):
    rng = np.random.default_rng(seed)
    clusters = np.arange(pools) % 16
    latent = rng.normal(0, 0.12, size=(pools, 16))
    latent[np.arange(pools), clusters] += 6
    keys = latent @ basis.T + rng.normal(0, 0.01, size=(pools, 128))
    queries = []
    for i in range(8):
        heads = rng.normal(0, 0.03, size=(4, 16))
        heads[:, (i + 3) % 16] += 1
        queries.append(heads @ basis.T)
    return np.asarray(keys, np.float32), np.asarray(queries, np.float32)


def timed_search(projection, pools, basis):
    keys, queries = qsa_fixture(pools, 93 + pools, basis)
    start = time.perf_counter()
    index = QsaBucketIndex(keys, projection)
    build = time.perf_counter() - start
    exact_times, sparse_times, recalls, stats = [], [], [], []
    budget = min(pools, 4096)
    for query in queries[:2]:
        index.select(query, 512, budget)
        top_ids(qsa_scores(keys, query), min(512, pools))
    for i, query in enumerate(queries):
        def exact():
            start = time.perf_counter()
            result = top_ids(qsa_scores(keys, query), min(512, pools))
            exact_times.append(time.perf_counter() - start)
            return result

        def sparse():
            start = time.perf_counter()
            result, info = index.select(query, 512, budget)
            sparse_times.append(time.perf_counter() - start)
            stats.append(info)
            return result

        expected, actual = (exact(), sparse()) if i % 2 == 0 else (None, None)
        if i % 2:
            actual, expected = sparse(), exact()
        recalls.append(len(np.intersect1d(expected, actual)) / len(expected))
    return {"pools": pools, "represented_tokens": pools * 4, "queries": len(queries),
            "selected_pools": min(512, pools), "candidate_budget": budget,
            "index_build_seconds": build, "exact_qsa_seconds": exact_times,
            "bucket_qsa_seconds": sparse_times, "recall": recalls,
            "exact_median_seconds": statistics.median(exact_times),
            "bucket_median_seconds": statistics.median(sparse_times), "search_stats": stats}


def update(graph, index, rows, integer=False):
    tensor, original = graph.inputs[index]
    data = array.array("i" if integer else "f", (x for row in rows for x in row)).tobytes()
    if len(data) != len(original):
        raise ValueError("fixed-shape graph input size changed")
    graph.engine.set_tensor(tensor, C.create_string_buffer(data), 0, len(data))


def integrated(engine, tokens, repeats=11):
    # Same fixture, map, candidate budget, and attention for old/new selection.
    keys, values, query, projection = fixture(tokens, width=128)
    latent = [key[:8] for key in keys]
    old = LatentTree(latent, 32)
    new = QsaBucketIndex(keys, projection)
    heads = np.tile(np.asarray(query, dtype=np.float32) / 4, (4, 1))
    oracle = top_ids(qsa_scores(np.asarray(keys, np.float32), heads), 32)

    def select(method):
        if method == "tree":
            ids, _ = old.select(query[:8], 32, 256)
        else:
            ids, _ = new.select(heads, 32, 256)
        return list(sorted(set(int(x) for x in ids) | {0, 1} | set(range(tokens - 8, tokens))))

    selections = {method: select(method) for method in ("tree", "bucket")}
    graphs = {"dense": engine.attention_graph(keys, values, query)}
    try:
        for method, ids in selections.items():
            graphs[method] = engine.attention_graph(keys, values, query, ids)
        for graph, output in graphs.values():
            graph.run(output)
        timings = {method: [] for method in graphs}
        outputs = {}
        for i in range(repeats):
            for method in (("dense", "tree", "bucket") if i % 2 == 0 else ("bucket", "tree", "dense")):
                graph, output = graphs[method]
                start = time.perf_counter()
                if method != "dense":
                    ids = select(method)
                    update(graph, 3, [ids], integer=True)
                update(graph, 2, [query])
                _, outputs[method] = graph.run(output)
                timings[method].append(time.perf_counter() - start)
        reference_errors = {}
        for method in graphs:
            ids = list(range(tokens)) if method == "dense" else selections[method]
            expected = attention(keys, values, query, ids)
            difference = max(abs(a-b) for a,b in zip(expected, outputs[method]))
            if difference > 0.0003:
                raise AssertionError(f"{method} numerical reference check failed: {difference}")
            reference_errors[method] = difference
        return {"tokens": tokens, "fixture_key_value_dimension": 128,
                "attention_heads": 1, "search_rank": 8, "search_top_k": 32, "candidate_budget": 256,
                "method": "Sequential CPU selection, fresh index/query serialization and upload, graph compute/sync, and output readback under one wall clock. Prebuilt index/graph. Fixed repeated query. This is not model inference.",
                "samples_seconds": timings,
                "medians_seconds": {method: statistics.median(samples) for method, samples in timings.items()},
                "cpu_reference_max_abs_error": reference_errors,
                "top32_recall": {method: len(set(ids) & set(oracle)) / len(oracle) for method, ids in selections.items()},
                "sparse_dense_max_abs_error": {method: max(abs(a-b) for a,b in zip(outputs[method], outputs["dense"])) for method in selections}}
    finally:
        for graph, _ in graphs.values():
            graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    root = Path(__file__).parent
    rng = np.random.default_rng(777)
    basis, _ = np.linalg.qr(rng.normal(size=(128, 16)))
    train_keys, train_queries = qsa_fixture(4096, 91, basis)
    start = time.perf_counter()
    projection = fit_projection(train_keys, train_queries, 16)
    fit_seconds = time.perf_counter() - start
    result = {"schema": 1, "kind": "synthetic_flash_next_geometry_offline_experiment",
              "numpy": np.__version__, "platform": platform.platform(), "python": platform.python_version(),
              "source_kind": "synthetic_fixture", "real_model_capture": False,
              "serving_integration": False, "model_quality_validated": False,
              "blas_threads": 1, "projection_fit_seconds": fit_seconds,
              "profile": json.loads((root / "flash-next-profile.json").read_text()),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                                ("flash_next_calibration.py", "run_flash_next_experiment.py", "test_flash_next.py", "ggml_vulkan.py", "lsa_reference.py", "run_experiment.py")},
              "qsa_cpu_search": [], "integrated_synthetic_vulkan": []}
    for pools in (4096, 16384, 65536):
        row = timed_search(projection, pools, basis)
        result["qsa_cpu_search"].append(row)
        print(json.dumps(row), flush=True)
    engine = Vulkan(args.runtime)
    try:
        result["device"], result["dll_sha256"] = engine.device, engine.fingerprints
        for tokens in (512, 4096, 16384):
            row = integrated(engine, tokens)
            result["integrated_synthetic_vulkan"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        engine.close()
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print("SYNTHETIC_FLASH_NEXT_GEOMETRY_PASS", flush=True)


if __name__ == "__main__":
    main()
