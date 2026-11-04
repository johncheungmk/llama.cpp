"""Bounded b11681 Windows CPU-only capture of one Flash Next QSA layer.

Uses a separate model/context; never attaches to a serving process. The scheduler
callback deliberately stops evaluation once the requested indexer tensors exist.
This does not produce logits or measure model inference performance.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np

P, SZ, I = C.c_void_p, C.c_size_t, C.c_int32


class ModelParams(C.Structure):
    _fields_ = [("devices", P), ("tensor_buft_overrides", P), ("n_gpu_layers", I), ("split_mode", I),
                ("load_mode", I), ("lazy_mode", I), ("main_gpu", I), ("tensor_split", P),
                ("progress_callback", P), ("progress_callback_user_data", P), ("kv_overrides", P),
                *[(name, C.c_bool) for name in ("vocab_only", "check_tensors", "use_extra_bufts", "no_host", "no_alloc", "load_mtp")]]


class ContextParams(C.Structure):
    _fields_ = [*[(name, C.c_uint32) for name in ("n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_rs_seq", "n_outputs_max", "n_outputs_max_per_seq")],
                ("n_threads", I), ("n_threads_batch", I),
                *[(name, I) for name in ("ctx_type", "rope_scaling_type", "pooling_type", "attention_type", "flash_attn_type")],
                *[(name, C.c_float) for name in ("rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow")],
                ("yarn_orig_ctx", C.c_uint32), ("defrag_thold", C.c_float), ("cb_eval", P), ("cb_eval_user_data", P),
                ("type_k", I), ("type_v", I), ("moe_cache_size", SZ), ("abort_callback", P), ("abort_callback_data", P),
                *[(name, C.c_bool) for name in ("embeddings", "offload_kqv", "no_perf", "op_offload", "swa_full", "kv_unified")],
                ("samplers", P), ("n_samplers", SZ), ("ctx_other", P)]


class Batch(C.Structure):
    _fields_ = [("n_tokens", I), ("token", P), ("embd", P), ("pos", P), ("n_seq_id", P), ("seq_id", P), ("logits", P)]


class TensorPrefix(C.Structure):
    _fields_ = [("type", I), ("buffer", P), ("ne", C.c_int64 * 4), ("nb", SZ * 4)]


class JobBasic(C.Structure):
    _fields_ = [("process_time", C.c_int64), ("job_time", C.c_int64), ("flags", C.c_uint32),
                ("min_ws", SZ), ("max_ws", SZ), ("active_process_limit", C.c_uint32),
                ("affinity", SZ), ("priority", C.c_uint32), ("scheduling", C.c_uint32)]


class JobExtended(C.Structure):
    _fields_ = [("basic", JobBasic), ("io", C.c_uint64 * 6), ("process_memory", SZ),
                ("job_memory", SZ), ("peak_process_memory", SZ), ("peak_job_memory", SZ)]


def enforce_limits():
    if os.name != "nt" or C.sizeof(P) != 8:
        raise RuntimeError("Windows x64 is required")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    def api(name, ret, args):
        func = getattr(kernel, name)
        func.restype, func.argtypes = ret, args
        return func
    process = api("GetCurrentProcess", P, [])()
    job = api("CreateJobObjectW", P, [P, C.c_wchar_p])(None, None)
    info = JobExtended()
    info.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY (private commit)
    info.process_memory = 6 * 1024**3
    if not job or not api("SetInformationJobObject", I, [P, I, P, C.c_uint32])(job, 9, C.byref(info), C.sizeof(info)):
        raise C.WinError(C.get_last_error())
    if not api("AssignProcessToJobObject", I, [P, P])(job, process):
        raise C.WinError(C.get_last_error())
    if not api("SetProcessWorkingSetSizeEx", I, [P, SZ, SZ, C.c_uint32])(process, 16 * 1024**2, 8 * 1024**3, 0x6):
        raise C.WinError(C.get_last_error())
    if not api("SetPriorityClass", I, [P, C.c_uint32])(process, 0x4000):
        raise C.WinError(C.get_last_error())
    return kernel, job  # retain the handle until process exit


def capture(args):
    limits = enforce_limits()
    if (C.sizeof(ModelParams), C.sizeof(ContextParams), C.sizeof(Batch)) != (80, 168, 56):
        raise RuntimeError("Unexpected b11681 ABI layout")
    runtime = args.runtime.resolve(strict=True)
    dll_directory = os.add_dll_directory(str(runtime))
    base = C.CDLL(str(runtime / "ggml-base.dll"))
    ggml = C.CDLL(str(runtime / "ggml.dll"))
    llama = C.CDLL(str(runtime / "llama.dll"))
    def bind(lib, name, ret, arguments):
        func = getattr(lib, name)
        func.restype, func.argtypes = ret, arguments
        return func
    load_backend = bind(ggml, "ggml_backend_load", P, [C.c_char_p])
    if not load_backend(os.fsencode(runtime / "ggml-cpu-zen4.dll")):
        raise RuntimeError("Cannot load the explicit CPU backend")
    bind(llama, "llama_backend_init", None, [])()
    model_params = bind(llama, "llama_model_default_params", ModelParams, [])()
    model_params.n_gpu_layers = 0
    model_params.load_mode, model_params.lazy_mode = 1, 2  # mmap and lazy PLE
    model_params.use_extra_bufts = False  # avoid weight repacking/copies
    model_params.vocab_only = args.vocab_only
    model_params.load_mtp = False
    print("CAPTURE_LIMITS_ACTIVE_CPU_ONLY_LOADING", flush=True)
    model = bind(llama, "llama_model_load_from_file", P, [C.c_char_p, ModelParams])(os.fsencode(args.model), model_params)
    if not model:
        raise RuntimeError("Model load failed within the capture resource limits")
    context = None
    try:
        print("CAPTURE_MODEL_LOADED", flush=True)
        meta = bind(llama, "llama_model_meta_val_str", I, [P, C.c_char_p, P, SZ])
        def metadata(key):
            value = C.create_string_buffer(512)
            if meta(model, key.encode(), value, len(value)) < 0:
                raise RuntimeError(f"Missing model metadata: {key}")
            return value.value.decode()
        for key, expected in {"general.architecture": "qwen4exp", "qwen4exp.attention.indexer.key_length": "128", "qwen4exp.attention.indexer.head_count": "4"}.items():
            if metadata(key) != expected:
                raise RuntimeError(f"Unsupported model geometry: {key}")
        if args.vocab_only:
            print("CAPTURE_VOCAB_ABI_PASS", flush=True)
            return
        vocab = bind(llama, "llama_model_get_vocab", P, [P])(model)
        text = args.prompt.read_bytes()
        if not 1 <= len(text) <= 65536:
            raise ValueError("prompt must contain 1..65536 UTF-8 bytes")
        tokenize = bind(llama, "llama_tokenize", I, [P, C.c_char_p, I, P, I, C.c_bool, C.c_bool])
        count = tokenize(vocab, text, len(text), None, 0, True, False)
        capacity = -count if count < 0 else count
        if not 1 <= capacity <= 16384:
            raise ValueError("prompt token count outside experiment limits")
        tokens = (I * capacity)()
        count = tokenize(vocab, text, len(text), tokens, capacity, True, False)
        count = min(count, args.tokens)
        count -= count % 4
        if count < 32:
            raise ValueError("At least 32 tokens are needed")
        get_name = bind(base, "ggml_get_name", C.c_char_p, [P])
        nbytes = bind(base, "ggml_nbytes", SZ, [P])
        get_tensor = bind(base, "ggml_backend_tensor_get", None, [P, P, SZ, SZ])
        collected, errors = {}, []
        layers = list(range(3, 48, 4)) if args.all_layers else [args.layer]
        names = {f"indexer_{kind}-{layer}": (layer, label)
                 for layer in layers for kind, label in (("k", "keys"), ("q", "queries"), ("score", "scores"))}
        callback_type = C.CFUNCTYPE(C.c_bool, P, C.c_bool, P)
        @callback_type
        def callback(tensor, ask, unused):
            try:
                name = get_name(tensor).decode()
                if ask:
                    return name in names
                if name not in names:
                    return True
                header = C.cast(tensor, C.POINTER(TensorPrefix)).contents
                size = nbytes(tensor)
                if header.type != 0 or not 1 <= size <= 16 * 1024**2 or any(not 1 <= n <= 65536 for n in header.ne):
                    raise ValueError("Unexpected capture tensor type/shape")
                storage = C.create_string_buffer(size)
                get_tensor(tensor, storage, 0, size)
                data = np.ndarray(tuple(reversed(header.ne)), dtype="float32", buffer=storage,
                                  strides=tuple(reversed(header.nb))).copy()
                collected[names[name]] = data
                print("CAPTURE_TENSOR", name, data.shape, flush=True)
                return len(collected) < len(names)  # intentionally abort after requested tensors
            except BaseException as exc:
                errors.append(str(exc))
                return False
        params = bind(llama, "llama_context_default_params", ContextParams, [])()
        params.n_ctx, params.n_batch, params.n_ubatch = 512, 256, 256
        params.n_seq_max, params.n_threads, params.n_threads_batch = 1, 4, 4
        params.type_k, params.type_v, params.flash_attn_type = 8, 8, 1  # Q8_0, as in the serving configuration
        params.offload_kqv, params.op_offload = False, False
        params.cb_eval = C.cast(callback, P).value
        context = bind(llama, "llama_init_from_model", P, [P, ContextParams])(model, params)
        if not context:
            raise RuntimeError("Context initialization failed within capture limits")
        batch = bind(llama, "llama_batch_get_one", Batch, [P, I])(tokens, count)
        status = bind(llama, "llama_decode", I, [P, Batch])(context, batch)
        if errors or set(collected) != set(names.values()):
            raise RuntimeError(f"Incomplete capture, decode status {status}, errors {errors}")
        summaries = []
        for layer in layers:
            keys = collected[(layer, "keys")].reshape(-1, 128)[:count // 4]
            queries = collected[(layer, "queries")].reshape(-1, 4, 128)
            if len(queries) != count or not np.isfinite(keys).all() or not np.isfinite(queries).all():
                raise RuntimeError("Invalid capture geometry or values")
            native_scores = collected[(layer, "scores")].reshape(count, -1)[:, :count // 4]
            expected_visible = np.arange(count // 4)[None, :] < ((np.arange(count) + 1) // 4)[:, None]
            if not np.array_equal(np.isfinite(native_scores), expected_visible):
                raise RuntimeError("Native pool ordering/mask differs from the contiguous-prefix assumption")
            estimates = np.maximum(np.einsum("thd,pd->thp", queries, keys), 0).sum(axis=1) / np.sqrt(128)
            score_error = float(np.max(np.abs(estimates[expected_visible] - native_scores[expected_visible])))
            if not np.allclose(estimates[expected_visible], native_scores[expected_visible], rtol=1e-4, atol=1e-3):
                raise RuntimeError(f"Native QSA score reference differs: {score_error}")
            positions = np.arange(31, count, 32)
            manifest = {"schema": 1, "source_kind": "real_model_capture", "stage": "post_norm_post_rope_pooled_indexer",
                        "model_id": "Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL", "layer": layer,
                        "document_ids": [args.document_id], "prompt_sha256": hashlib.sha256(text).hexdigest(),
                        "tokens": count, "decode_status": status, "intentional_early_abort": True,
                        "attention_kv_types": "Q8_0/Q8_0", "flash_attention": True,
                        "device": "CPU", "checkpoint_identity": "caller label and verified metadata, not full weight hash",
                        "scope": "One independent contiguous text sequence; no prefix reuse, eviction or mixed sequences",
                        "native_qsa_score_max_abs_error": score_error, "native_causal_pool_mask_verified": True,
                        "query_positions": positions.tolist()}
            path = args.output.with_name(f"{args.output.stem}.layer-{layer}.npz") if args.all_layers else args.output
            with path.open("xb") as output:
                np.savez(output, keys=keys, queries=queries[positions], visible=(positions + 1) // 4, manifest=json.dumps(manifest))
            summaries.append(manifest)
        if args.all_layers:
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(summaries, output, indent=2)
        print("REAL_CAPTURE_PASS", json.dumps(summaries), flush=True)
    finally:
        if context:
            bind(llama, "llama_free", None, [P])(context)
        bind(llama, "llama_model_free", None, [P])(model)
        dll_directory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--layer", type=int, default=3, choices=range(3, 48, 4))
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--tokens", type=int, default=256, choices=(32, 64, 128, 256))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    if args.all_layers and any(args.output.with_name(f"{args.output.stem}.layer-{layer}.npz").exists() for layer in range(3, 48, 4)):
        parser.error("a per-layer output already exists")
    timer = threading.Timer(180, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        capture(args)
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
