"""Synthetic Vulkan validation and graph timings; never loads a language model."""
import argparse
import array
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from ggml_vulkan import Vulkan
from lsa_reference import LatentTree, attention, dot, project

BASE_COMMIT = "5e03a1ff450cae37a774684de55d14fb48b1f90b"


def f32_rows(rows):
    return [list(array.array("f", row)) for row in rows]


def error(a, b):
    if len(a) != len(b):
        raise AssertionError("output dimensions differ")
    return max(abs(x - y) for x, y in zip(a, b))


def fixture(tokens, width=256, rank=8):
    rng = random.Random(4361)
    keys, values = [], []
    for i in range(tokens):
        cluster = (i // 32) % rank
        key = [rng.gauss(0, 0.15) for _ in range(width)]
        key[cluster] += 6
        value = [rng.gauss(0, 0.05) for _ in range(width)]
        value[cluster] += 1
        keys.append(key)
        values.append(value)
    query = [0.0] * width
    query[3] = 0.75 * math.sqrt(width)
    projection = [[float(i == j) for j in range(width)] for i in range(rank)]
    return f32_rows(keys), f32_rows(values), query, projection


def checked_attention(engine, keys, values, query, ids=None):
    graph, output = engine.attention_graph(keys, values, query, ids)
    try:
        _, result = graph.run(output)
        expected = attention(keys, values, query, list(range(len(keys))) if ids is None else ids)
        difference = error(result, expected)
        if difference > 0.0003:
            raise AssertionError(f"Vulkan attention differs from CPU reference: {difference}")
        return difference
    finally:
        graph.close()


def validate(engine):
    checks = {}
    keys, values, query, projection = fixture(97, width=64)
    latent, _ = engine.project(keys, projection)
    expected = project(keys, projection)
    difference = max(error(a, b) for a, b in zip(latent, expected))
    if difference > 0.0003:
        raise AssertionError(f"Vulkan projection differs: {difference}")
    checks["projection_max_abs_error"] = difference
    checks["dense_cpu_max_abs_error"] = checked_attention(engine, keys, values, query)
    checks["full_gather_cpu_max_abs_error"] = checked_attention(engine, keys, values, query, list(range(97)))
    ids, _ = LatentTree(latent, 8).select(project([query], projection)[0], 8, 24, visible=73, sink=2, window=4)
    if any(i >= 73 for i in ids):
        raise AssertionError("future token selected")
    checks["causal_sparse_cpu_max_abs_error"] = checked_attention(engine, keys, values, query, ids)
    # A much larger future logit must not affect the gathered causal result.
    keys[96][3] = 10000
    checks["future_needle_cpu_max_abs_error"] = checked_attention(engine, keys, values, query, ids)
    keys = [[0.0] * 64 for _ in range(128)]
    keys[35][0] = 20
    values = [[float(i) / 128, -float(i) / 128] for i in range(128)]
    query = [20.0] + [0.0] * 63
    ids, _ = LatentTree([row[:8] for row in keys], 8).select(query[:8], 1, 8)
    if ids != [35]:
        raise AssertionError("synthetic needle not selected")
    graph, output = engine.attention_graph(keys, values, query, ids)
    try:
        _, sparse = graph.run(output)
        checks["needle_sparse_vs_dense_max_abs_error"] = error(sparse, attention(keys, values, query, list(range(128))))
        if checks["needle_sparse_vs_dense_max_abs_error"] > 0.0003:
            raise AssertionError("needle sparse attention differs from dense")
    finally:
        graph.close()
    return checks


def benchmark(engine, tokens, repeats):
    keys, values, query, projection = fixture(tokens)
    start = time.perf_counter()
    latent, projection_seconds = engine.project(keys, projection)
    projection_wall = time.perf_counter() - start
    start = time.perf_counter()
    tree = LatentTree(latent, 32)
    build_seconds = time.perf_counter() - start
    latent_query = project([query], projection)[0]
    ids, stats = tree.select(latent_query, 32, 256, sink=2, window=8)
    oracle = set(sorted(range(tokens), key=lambda i: (-dot(latent_query, latent[i]), i))[:32])
    recall = len(oracle & set(ids)) / len(oracle)
    dense, dense_output = engine.attention_graph(keys, values, query)
    sparse = None
    try:
        sparse, sparse_output = engine.attention_graph(keys, values, query, ids)
        if dense.bytes + sparse.bytes > 128 * 1024 * 1024:
            raise MemoryError("Combined replay buffers exceed 128 MiB")
        dense.run(dense_output)
        sparse.run(sparse_output)
        dense_times, sparse_times, search_times = [], [], []
        for i in range(repeats):
            start = time.perf_counter()
            selected, _ = tree.select(latent_query, 32, 256, sink=2, window=8)
            search_times.append(time.perf_counter() - start)
            if selected != ids:
                raise AssertionError("fixed-query selection changed")
            for label in (("dense", "sparse") if i % 2 == 0 else ("sparse", "dense")):
                if label == "dense":
                    elapsed, dense_result = dense.run(dense_output)
                    dense_times.append(elapsed)
                else:
                    elapsed, sparse_result = sparse.run(sparse_output)
                    sparse_times.append(elapsed)
        sparse_reference = attention(keys, values, query, ids)
        sparse_error = error(sparse_result, sparse_reference)
        dense_error = error(dense_result, attention(keys, values, query, list(range(tokens))))
        if max(sparse_error, dense_error) > 0.0003:
            raise AssertionError("benchmark reference check failed")
        return {
            "tokens": tokens, "key_dimension": 256, "value_dimension": 256,
            "latent_dimension": 8, "queries": 1, "leaf_size": 32,
            "top_k": 32, "candidate_budget": 256, "sink": 2, "window": 8,
            "selection": stats, "latent_top32_recall": recall,
            "projection_gpu_seconds": projection_seconds,
            "projection_with_setup_and_transfers_seconds": projection_wall,
            "cpu_tree_build_seconds": build_seconds,
            "cpu_search_median_seconds": statistics.median(search_times),
            "dense_graph_median_seconds": statistics.median(dense_times),
            "sparse_graph_median_seconds": statistics.median(sparse_times),
            "dense_graph_samples_seconds": dense_times,
            "sparse_graph_samples_seconds": sparse_times,
            "cpu_search_samples_seconds": search_times,
            "graph_only_ratio_dense_over_sparse": statistics.median(dense_times) / statistics.median(sparse_times),
            "dense_cpu_max_abs_error": dense_error,
            "sparse_cpu_max_abs_error": sparse_error,
            "sparse_vs_dense_max_abs_error": error(sparse_result, dense_result),
            "combined_tensor_buffer_bytes": dense.bytes + sparse.bytes,
        }
    finally:
        dense.close()
        if sparse:
            sparse.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, help="Trusted b11681 Windows DLL directory")
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 4096, 16384])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 50 or not 1 <= len(args.tokens) <= 8 or any(not 256 <= n <= 16384 for n in args.tokens):
        parser.error("repeats must be 1..50; supply 1..8 token lengths in 256..16384")
    if args.output.exists():
        parser.error("output exists; use a new filename")
    engine = Vulkan(args.runtime)
    try:
        result = {
            "schema": 1, "kind": "synthetic_vulkan_primitive_poc",
            "upstream_base": BASE_COMMIT, "device": engine.device,
            "platform": platform.platform(), "python": platform.python_version(),
            "dll_sha256": engine.fingerprints,
            "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("ggml_vulkan.py", "lsa_reference.py", "run_experiment.py", "test_reference.py")},
            "private_process_environment": {"GGML_VK_DISABLE_F16": "1", "GGML_VK_DISABLE_COOPMAT": "1"},
            "model_integration": False, "learned_projection_map": False,
            "production_ready": False, "fixture_seed": 4361,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "method": "F32 Vulkan projection, GPU gather and attention; CPU median-split tree and best-first search. Synthetic untrained identity-prefix projection. Alternating preallocated fixed-query graph replays after warmup; graph timings include submit and synchronization, exclude setup, uploads, CPU search, output readback and index construction. Neither tokens/sec nor model serving performance.",
            "correctness": validate(engine), "benchmarks": [],
        }
        for tokens in args.tokens:
            row = benchmark(engine, tokens, args.repeats)
            result["benchmarks"].append(row)
            print(json.dumps(row), flush=True)
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("SYNTHETIC_VULKAN_POC_PASS", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
