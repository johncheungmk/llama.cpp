"""Offline Flash Next QSA calibration/search. Not a serving integration or D-LSA port."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

MAX_POOLS = 65536
MAX_QUERIES = 256


def finite(value, shape_tail, name, max_rows):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != tuple(shape_tail):
        raise ValueError(f"{name}: invalid shape")
    if not 1 <= value.shape[0] <= max_rows or not np.isfinite(value).all():
        raise ValueError(f"{name}: invalid count or non-finite values")
    return np.ascontiguousarray(value)


def check_profile(profile):
    if profile.get("architecture") != "qwen4exp" or profile.get("indexer_dimension") != 128 or profile.get("indexer_heads") != 4:
        raise ValueError("Only the inspected Flash Next 4-head, 128-wide QSA geometry is supported")
    if profile.get("pool_size") != 4 or profile.get("token_budget") != 2048:
        raise ValueError("Unsupported QSA pooling/budget")
    if profile.get("qsa_layers") != list(range(3, 48, 4)):
        raise ValueError("Unsupported QSA layer schedule")
    return profile


def fit_projection(keys, queries, rank):
    """Uncentered, equal key/query second-moment PCA; no distortion guarantee."""
    keys = finite(keys, (128,), "keys", MAX_POOLS)
    queries = finite(queries, (4, 128), "queries", MAX_QUERIES)
    if type(rank) is not int or not 1 <= rank <= 128 or rank > min(len(keys), queries.shape[0] * 4):
        raise ValueError("rank exceeds the supported dimension or sample count")
    k = keys.astype(np.float64)
    q = queries.reshape(-1, 128).astype(np.float64)
    moment = (k.T @ k / len(k) + q.T @ q / len(q)) * 0.5
    _, vectors = np.linalg.eigh(moment)
    projection = vectors[:, ::-1][:, :rank].T.copy()
    # Canonical signs improve reproducibility; repeated eigenspaces may still rotate.
    for row in projection:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return np.asarray(projection, dtype=np.float32)


def qsa_scores(keys, query):
    """Match qwen4exp's summed rectified head dots, scaled by sqrt(128)."""
    return np.maximum(keys @ query.T, 0).sum(axis=1) / math.sqrt(128)


def top_ids(scores, k):
    """O(N) partition with deterministic lowest-ID tie handling."""
    if not 1 <= k <= len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid top-k input")
    if k == len(scores):
        return np.arange(len(scores), dtype=np.int64)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k - len(above)]
    return np.sort(np.concatenate((above, tied)))


class QsaBucketIndex:
    """Vectorized flat buckets with residual-aware bounds and full-space rescoring.

    This still scans every bucket. Budget-limited selection is approximate. It is
    deliberately separate from the first prototype's single-query latent tree.
    """

    def __init__(self, keys, projection, bucket_size=32):
        self.keys = finite(keys, (128,), "keys", MAX_POOLS)
        projection = np.asarray(projection, dtype=np.float32)
        if projection.ndim != 2 or not 1 <= projection.shape[0] <= 128 or projection.shape[1] != 128 or not np.isfinite(projection).all():
            raise ValueError("invalid projection")
        if not np.allclose(projection @ projection.T, np.eye(len(projection)), atol=1e-4):
            raise ValueError("projection rows must be orthonormal")
        if type(bucket_size) is not int or not 1 <= bucket_size <= 256:
            raise ValueError("invalid bucket size")
        self.projection, self.bucket_size = projection.copy(), bucket_size
        latent = self.keys @ projection.T

        def partition(ids):
            if len(ids) <= bucket_size:
                return [ids]
            axis = int(np.argmax(np.var(latent[ids], axis=0)))
            ordered = ids[np.argsort(latent[ids, axis], kind="stable")]
            middle = len(ids) // 2
            return partition(ordered[:middle]) + partition(ordered[middle:])

        buckets = partition(np.arange(len(keys)))
        self.order = np.concatenate(buckets)
        latent = latent[self.order]
        self.starts = np.cumsum([0] + [len(ids) for ids in buckets[:-1]])
        self.ends = np.cumsum([len(ids) for ids in buckets])
        self.lower = np.minimum.reduceat(latent, self.starts)
        self.upper = np.maximum.reduceat(latent, self.starts)
        residual = self.keys - (self.keys @ projection.T) @ projection
        norms = np.linalg.norm(residual[self.order], axis=1)
        self.residual_max = np.maximum.reduceat(norms, self.starts)

    def select(self, query, k, candidate_budget, visible=None):
        query = finite(np.asarray(query)[None, ...], (4, 128), "query", 1)[0]
        visible = len(self.keys) if visible is None else visible
        if type(visible) is not int or not 1 <= visible <= len(self.keys):
            raise ValueError("visible must identify a nonempty causal pool prefix")
        if type(k) is not int or type(candidate_budget) is not int or not 1 <= k <= candidate_budget <= MAX_POOLS:
            raise ValueError("invalid selection budget")
        if visible < len(self.keys):
            # Future keys must not change bucket geometry or candidate ranking.
            # A serving implementation needs incrementally maintained prefix indices.
            prefix = QsaBucketIndex(self.keys[:visible], self.projection, self.bucket_size)
            ids, stats = prefix.select(query, k, candidate_budget)
            stats["prefix_rebuilt"] = True
            return ids, stats
        k = min(k, visible)
        if candidate_budget >= visible:
            candidates = np.arange(visible)
            bounds_scored = 0
        else:
            q_latent = query @ self.projection.T
            q_residual = query - q_latent @ self.projection
            dots = np.maximum(q_latent, 0) @ self.upper.T + np.minimum(q_latent, 0) @ self.lower.T
            dots += np.linalg.norm(q_residual, axis=1)[:, None] * self.residual_max[None, :]
            bounds = np.maximum(dots, 0).sum(axis=0) / math.sqrt(128)
            ordered_buckets = np.argsort(-bounds, kind="stable")
            # Gather whole buckets in order, filter future pools before rescoring.
            candidate_parts, count = [], 0
            for bucket in ordered_buckets:
                ids = self.order[self.starts[bucket]:self.ends[bucket]]
                ids = ids[ids < visible]
                if len(ids):
                    ids = ids[:candidate_budget - count]
                    candidate_parts.append(ids)
                    count += len(ids)
                if count >= min(visible, candidate_budget):
                    break
            candidates = np.sort(np.concatenate(candidate_parts))
            bounds_scored = len(bounds)
        scores = qsa_scores(self.keys[candidates], query)
        ids = candidates[top_ids(scores, min(k, len(candidates)))]
        return ids, {"buckets_scored": bounds_scored, "keys_rescored": len(candidates)}


def load_capture(path, profile):
    """Bounded numeric NPZ with an explicit provenance manifest; no pickle."""
    check_profile(profile)
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("capture archive too large")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != 4 or sum(x.file_size for x in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("invalid capture size or member count")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"keys", "queries", "visible", "manifest"}:
            raise ValueError("unexpected capture members")
        manifest_array = archive["manifest"]
        if manifest_array.ndim != 0 or manifest_array.dtype.kind != "U" or manifest_array.nbytes > 65536:
            raise ValueError("invalid manifest storage")
        manifest = json.loads(str(manifest_array))
        keys = finite(archive["keys"], (128,), "keys", MAX_POOLS)
        queries = finite(archive["queries"], (4, 128), "queries", MAX_QUERIES)
        visible = np.asarray(archive["visible"])
    if visible.dtype.kind not in "iu" or visible.shape != (len(queries),) or np.any(visible < 1) or np.any(visible > len(keys)):
        raise ValueError("invalid causal visibility")
    if manifest.get("schema") != 1 or manifest.get("source_kind") not in ("real_model_capture", "synthetic_fixture"):
        raise ValueError("unrecognized capture provenance")
    if manifest.get("stage") != "post_norm_post_rope_pooled_indexer" or manifest.get("layer") not in profile["qsa_layers"]:
        raise ValueError("wrong tensor stage or layer")
    if manifest.get("model_id") != profile["model_id"]:
        raise ValueError("model identity differs")
    documents = manifest.get("document_ids")
    if not isinstance(documents, list) or not documents or any(not isinstance(x, str) or not x for x in documents) or len(set(documents)) != len(documents):
        raise ValueError("document provenance is required")
    return keys, queries, visible, manifest


def calibrate(train_path, heldout_path, profile, rank, candidate_budget):
    train = load_capture(train_path, profile)
    heldout = load_capture(heldout_path, profile)
    if train[3]["layer"] != heldout[3]["layer"] or train[3]["source_kind"] != heldout[3]["source_kind"]:
        raise ValueError("training and held-out capture kinds/layers differ")
    if set(train[3]["document_ids"]) & set(heldout[3]["document_ids"]):
        raise ValueError("training and held-out document IDs overlap")
    if hashlib.sha256(Path(train_path).read_bytes()).digest() == hashlib.sha256(Path(heldout_path).read_bytes()).digest():
        raise ValueError("training and held-out archives are identical")
    projection = fit_projection(train[0], train[1], rank)
    index = QsaBucketIndex(heldout[0], projection)
    recalls, diagnostic_recalls, nontrivial_diagnostic_recalls = [], [], []
    squared_error, squared_teacher = 0.0, 0.0
    pool_budget = profile["token_budget"] // profile["pool_size"]
    if type(candidate_budget) is not int or not pool_budget <= candidate_budget <= MAX_POOLS:
        raise ValueError("candidate budget is outside QSA limits")
    for query, visible in zip(heldout[1], heldout[2]):
        visible = int(visible)
        expected = top_ids(qsa_scores(heldout[0][:visible], query), min(pool_budget, visible))
        actual, _ = index.select(query, pool_budget, candidate_budget, visible)
        recalls.append(len(np.intersect1d(expected, actual)) / len(expected))
        teacher = qsa_scores(heldout[0][:visible], query)
        latent_keys, latent_query = heldout[0][:visible] @ projection.T, query @ projection.T
        approximate = np.maximum(latent_keys @ latent_query.T, 0).sum(axis=1) / math.sqrt(128)
        squared_error += float(np.sum((approximate.astype(np.float64) - teacher) ** 2))
        squared_teacher += float(np.sum(teacher.astype(np.float64) ** 2))
        diagnostic_k = min(32, visible)
        diagnostic_recall = len(np.intersect1d(top_ids(teacher, diagnostic_k), top_ids(approximate, diagnostic_k))) / diagnostic_k
        diagnostic_recalls.append(diagnostic_recall)
        if visible > 32:
            nontrivial_diagnostic_recalls.append(diagnostic_recall)
    report = {
        "schema": 1, "method": "uncentered_equal_key_query_second_moment_PCA_with_full_QSA_rescoring",
        "model_id": profile["model_id"], "layer": train[3]["layer"], "rank": rank,
        "source_kind": train[3]["source_kind"], "serving_ready": False,
        "train_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "heldout_sha256": hashlib.sha256(Path(heldout_path).read_bytes()).hexdigest(),
        "heldout_recall_mean": float(np.mean(recalls)), "heldout_recall_min": min(recalls),
        "heldout_queries": len(recalls), "candidate_budget": candidate_budget,
        "queries_requiring_pruning": int(np.count_nonzero(heldout[2] > candidate_budget)),
        "all_queries_below_checkpoint_selection_budget": bool(np.all(heldout[2] <= pool_budget)),
        "projected_qsa_scores_relative_l2_error": math.sqrt(squared_error / max(squared_teacher, 1e-30)),
        "diagnostic_top32_projected_score_recall_mean": float(np.mean(diagnostic_recalls)),
        "diagnostic_top32_nontrivial_recall_mean": float(np.mean(nontrivial_diagnostic_recalls)) if nontrivial_diagnostic_recalls else None,
        "diagnostic_top32_nontrivial_queries": len(nontrivial_diagnostic_recalls),
        "diagnostic_top32_recall_per_query": diagnostic_recalls,
        "train_pools": len(train[0]), "heldout_pools": len(heldout[0]),
        "teacher": "checkpoint QSA pooled-block indexer, not dense attention or final model outputs",
        "limitations": "Offline single-layer map. No serving loader, KV lifecycle, model quality or end-to-end performance validation. Real capture provenance is supplied by the caller, not independently authenticated.",
    }
    return projection, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--train-capture", type=Path, required=True)
    parser.add_argument("--heldout-capture", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--candidate-budget", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation prevents silently replacing a previous calibration.
    if args.output.exists():
        parser.error("output exists; select a new filename")
    profile = json.loads(args.profile.read_text(encoding="utf-8-sig"))
    projection, report = calibrate(args.train_capture, args.heldout_capture, profile, args.rank, args.candidate_budget)
    with args.output.open("xb") as output:
        np.savez(output, projection=projection, manifest=json.dumps(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
