"""Compare matched teacher-forced native model logits; not a task-accuracy suite."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def load(path):
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("quality archive exceeds 128 MiB")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != 3 or sum(x.file_size for x in archive.infolist()) > 128 * 1024**2:
            raise ValueError("quality archive storage invalid")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"logits", "targets", "manifest"}:
            raise ValueError("unexpected quality archive members")
        logits = np.asarray(archive["logits"])
        targets = np.asarray(archive["targets"])
        manifest = json.loads(str(archive["manifest"]))
    if logits.dtype.kind != "f" or logits.ndim != 2 or not 2 <= logits.shape[0] <= 33 or not 1 <= logits.shape[1] <= 1000000 or not np.isfinite(logits).all():
        raise ValueError("invalid logits")
    if targets.dtype.kind not in "iu" or targets.shape != (len(logits) - 1,) or np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("invalid target IDs")
    return logits, targets, manifest


def log_softmax(logits):
    values = logits.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def compare(baseline, candidate):
    original, targets, before = load(baseline)
    modified, other_targets, after = load(candidate)
    fields = ("model_id", "prompt_sha256", "tokens", "gpu_layers", "context_tokens", "batch_tokens", "cpu_threads", "attention_kv_types", "vocabulary_size", "runtime_dll_sha256", "source_sha256", "support_source_sha256")
    if any(key not in before or key not in after or before[key] != after[key] for key in fields):
        raise ValueError("baseline and candidate settings/runtime/input differ")
    if original.shape != modified.shape or not np.array_equal(targets, other_targets):
        raise ValueError("quality rows or targets differ")
    a, b = log_softmax(original), log_softmax(modified)
    indices = np.arange(len(targets))
    nll_a, nll_b = float(-a[indices, targets].mean()), float(-b[indices, targets].mean())
    kl = float(np.mean(np.sum(np.exp(a) * (a - b), axis=1)))
    return {"schema": 1, "kind": "matched_native_model_logit_comparison", "settings": {key: before[key] for key in fields},
            "baseline_archive_sha256": hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
            "candidate_archive_sha256": hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
            "baseline_map_sha256": before.get("native_map_sha256"), "candidate_map_sha256": after.get("native_map_sha256"),
            "teacher_forced_positions": len(targets), "baseline_mean_nll": nll_a, "candidate_mean_nll": nll_b,
            "mean_nll_delta": nll_b - nll_a, "baseline_perplexity": math.exp(min(nll_a, 700)),
            "candidate_perplexity": math.exp(min(nll_b, 700)), "mean_kl_baseline_to_candidate": max(0.0, kl),
            "logit_top1_agreement": float(np.mean(np.argmax(original, axis=1) == np.argmax(modified, axis=1))),
            "max_abs_logit_difference": float(np.max(np.abs(original - modified))),
            "greedy_sequence_equal": before.get("generated_token_ids") == after.get("generated_token_ids"),
            "baseline_prefill_seconds": before["prefill_seconds"], "candidate_prefill_seconds": after["prefill_seconds"],
            "baseline_decode_tokens_per_second": before.get("decode_tokens_per_second"),
            "candidate_decode_tokens_per_second": after.get("decode_tokens_per_second"),
            "candidate_serialized_state_refused": after.get("serialized_state_refused", False),
            "production_ready": False, "limitations": "One synthetic public-example document; up to 32 teacher-forced next-token positions and 8/16 greedy steps. These are matched C-API model timings, not HTTP server throughput, task accuracy, RULER, or a 256K/C5 benchmark. Prefill includes final-layer activation callbacks and extra logit rows; model loading is excluded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
