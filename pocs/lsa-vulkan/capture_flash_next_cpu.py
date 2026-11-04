"""Bounded b11681 Windows Flash Next QSA capture and optional logit validation.

Uses a separate model/context; never attaches to a serving process. The scheduler
callback stops evaluation after requested indexer tensors by default. Explicit
options enable hybrid Vulkan offload, complete logits and bounded generation.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import struct

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from native_score_reference import verify as verify_scores, verify_projection

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


def enforce_limits(commit_gib=6):
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
    info.basic.flags = 0x100 | 0x2000  # private commit limit and kill descendants when this process closes its job
    info.process_memory = commit_gib * 1024**3
    if not job or not api("SetInformationJobObject", I, [P, I, P, C.c_uint32])(job, 9, C.byref(info), C.sizeof(info)):
        raise C.WinError(C.get_last_error())
    if not api("AssignProcessToJobObject", I, [P, P])(job, process):
        raise C.WinError(C.get_last_error())
    if not api("SetProcessWorkingSetSizeEx", I, [P, SZ, SZ, C.c_uint32])(process, 16 * 1024**2, 8 * 1024**3, 0x6):
        raise C.WinError(C.get_last_error())
    if not api("SetPriorityClass", I, [P, C.c_uint32])(process, 0x4000):
        raise C.WinError(C.get_last_error())
    return kernel, job  # retain the handle until process exit


def capture_tensor_names(layers, projected):
    names = {f"indexer_{kind}-{layer}": (layer, label)
             for layer in layers for kind, label in (("k", "keys"), ("q", "queries"), ("score", "scores"))}
    if projected:
        for layer in layers:
            # Native cb() names the same pooled tensor twice; its final name wins.
            del names[f"indexer_k-{layer}"]
            names[f"experimental_qsa_pca_k-{layer}"] = (layer, "keys")
            names[f"experimental_qsa_pca_q-{layer}"] = (layer, "projected_queries")
            names[f"indexer_pool_k_new-{layer}"] = (layer, "new_keys")
            names[f"experimental_qsa_pca_k_new-{layer}"] = (layer, "projected_new_keys")
    return names


def capture(args):
    capture_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    support_source_sha256 = {"native_score_reference.py": hashlib.sha256((Path(__file__).parent / "native_score_reference.py").read_bytes()).hexdigest()}
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    limits = enforce_limits(args.commit_limit)
    if (C.sizeof(ModelParams), C.sizeof(ContextParams), C.sizeof(Batch)) != (80, 168, 56):
        raise RuntimeError("Unexpected b11681 ABI layout")
    runtime = args.runtime.resolve(strict=True)
    dll_directory = os.add_dll_directory(str(runtime))
    base = C.CDLL(str(runtime / "ggml-base.dll"))
    ggml = C.CDLL(str(runtime / "ggml.dll"))
    llama = C.CDLL(str(runtime / "llama.dll"))
    # Native LoadLibrary backend loading does not consistently use Python's
    # AddDllDirectory search path. Load this trusted release dependency explicitly.
    openmp = C.CDLL(str(runtime / "libomp.dll")) if (runtime / "libomp.dll").exists() else None
    def bind(lib, name, ret, arguments):
        func = getattr(lib, name)
        func.restype, func.argtypes = ret, arguments
        return func
    load_backend = bind(ggml, "ggml_backend_load", P, [C.c_char_p])
    if not load_backend(os.fsencode(runtime / "ggml-cpu-zen4.dll")):
        raise RuntimeError("Cannot load the explicit CPU backend")
    if args.gpu_layers:
        if not load_backend(os.fsencode(runtime / "ggml-vulkan.dll")):
            raise RuntimeError("Cannot load the explicit Vulkan backend")
        device = bind(ggml, "ggml_backend_dev_by_name", P, [C.c_char_p])(b"Vulkan0")
        if not device:
            raise RuntimeError("Vulkan0 is absent")
        free, total = SZ(), SZ()
        bind(base, "ggml_backend_dev_memory", None, [P, P, P])(device, C.byref(free), C.byref(total))
        if free.value < 48 * 1024**3:
            raise RuntimeError("Hybrid capture requires at least 48 GiB of reported free Vulkan budget")
        print("HYBRID_VULKAN_FREE_BYTES", free.value, "GPU_LAYERS", args.gpu_layers, flush=True)
    expected_maps = {}
    active_map = os.environ.get("LLAMA_QSA_PCA_MAP")
    if active_map:
        if args.expected_native_map is None or Path(active_map).resolve() != args.expected_native_map.resolve():
            raise ValueError("Projected-cache validation requires the exact --expected-native-map")
        data = args.expected_native_map.read_bytes()
        if not 20 <= len(data) <= 1024**2 or data[:8] != b"QPCAMAP1":
            raise ValueError("invalid expected native map")
        dimension, rank, layers = struct.unpack_from("<III", data, 8)
        if dimension != 128 or rank not in (16, 32, 64, 128) or layers != 12 or len(data) != 20 + 12 * (4 + rank * 128 * 4):
            raise ValueError("invalid expected native map dimensions")
        offset = 20
        for layer in range(3, 48, 4):
            if struct.unpack_from("<I", data, offset)[0] != layer:
                raise ValueError("invalid expected native map layer order")
            offset += 4
            expected_maps[layer] = np.frombuffer(data, dtype="<f4", count=rank * 128, offset=offset).reshape(rank, 128).copy()
            offset += rank * 128 * 4
    elif args.expected_native_map:
        raise ValueError("--expected-native-map supplied without LLAMA_QSA_PCA_MAP")
    bind(llama, "llama_backend_init", None, [])()
    model_params = bind(llama, "llama_model_default_params", ModelParams, [])()
    model_params.n_gpu_layers = args.gpu_layers
    model_params.load_mode, model_params.lazy_mode = 1, 2  # mmap and lazy PLE
    model_params.use_extra_bufts = False  # avoid weight repacking/copies
    model_params.vocab_only = args.vocab_only
    model_params.load_mtp = False
    print("CAPTURE_LIMITS_ACTIVE_LOADING", flush=True)
    model = bind(llama, "llama_model_load_from_file", P, [C.c_char_p, ModelParams])(os.fsencode(args.model), model_params)
    if not model:
        info = JobExtended()
        kernel, job = limits
        query = kernel.QueryInformationJobObject
        query.restype, query.argtypes = I, [P, I, P, C.c_uint32, P]
        if query(job, 9, C.byref(info), C.sizeof(info), None):
            print("CAPTURE_FAILED_PEAK_COMMIT_BYTES", info.peak_process_memory, "LIMIT_BYTES", info.process_memory, flush=True)
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
        collected, collected_host, errors = {}, {}, []
        capture_final = False
        layers = list(range(3, 48, 4)) if args.all_layers else [args.layer]
        names = capture_tensor_names(layers, bool(expected_maps))
        buffer_is_host = bind(base, "ggml_backend_buffer_is_host", C.c_bool, [P])
        callback_type = C.CFUNCTYPE(C.c_bool, P, C.c_bool, P)
        @callback_type
        def callback(tensor, ask, unused):
            try:
                name = get_name(tensor).decode()
                if ask:
                    return capture_final and name in names
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
                collected_host[names[name]] = bool(buffer_is_host(header.buffer))
                print("CAPTURE_TENSOR", name, data.shape, flush=True)
                return args.complete_eval or len(collected) < len(names)
            except BaseException as exc:
                errors.append(str(exc))
                return False
        params = bind(llama, "llama_context_default_params", ContextParams, [])()
        params.n_ctx, params.n_batch, params.n_ubatch = max(512, 1 << (count + args.generate - 1).bit_length()), 256, 256
        params.n_seq_max, params.n_threads, params.n_threads_batch = 1, args.threads, args.threads
        params.type_k, params.type_v, params.flash_attn_type = 8, 8, 1  # Q8_0, as in the serving configuration
        params.offload_kqv, params.op_offload = bool(args.gpu_layers), bool(args.gpu_layers)
        params.cb_eval = C.cast(callback, P).value
        context = bind(llama, "llama_init_from_model", P, [P, ContextParams])(model, params)
        if not context:
            raise RuntimeError("Context initialization failed within capture limits")
        batch_one = bind(llama, "llama_batch_get_one", Batch, [P, I])
        decode = bind(llama, "llama_decode", I, [P, Batch])
        prefill_start = time.perf_counter()
        final_offset, final_count = 0, 0
        for offset in range(0, count, 256):
            chunk = min(256, count - offset)
            capture_final = offset + chunk == count
            batch = batch_one(C.cast(C.byref(tokens, offset * 4), P), chunk)
            # Keep only the final 33 logits when completing evaluation; all earlier
            # batches need no output rows. Decode/generation later uses one row.
            flags = (C.c_int8 * chunk)()
            if capture_final and args.complete_eval:
                for index in range(max(0, chunk - 33), chunk):
                    flags[index] = 1
            elif capture_final:
                flags[-1] = 1
            batch.logits = C.cast(flags, P).value
            status = decode(context, batch)
            if status != 0 or errors:
                raise RuntimeError(f"chunk at {offset} failed: status={status}, errors={errors}")
            final_offset, final_count = offset, chunk
            print("CAPTURE_PREFILL_PROGRESS", offset + chunk, count, flush=True)
        bind(llama, "llama_synchronize", None, [P])(context)
        prefill_seconds = time.perf_counter() - prefill_start
        if errors or set(collected) != set(names.values()):
            raise RuntimeError(f"Incomplete capture, decode status {status}, errors {errors}")
        summaries = []
        for layer in layers:
            key_dimension = expected_maps[layer].shape[0] if expected_maps else 128
            keys = collected[(layer, "keys")].reshape(-1, key_dimension)[:count // 4]
            queries = collected[(layer, "queries")].reshape(-1, 4, 128)
            if len(queries) != final_count or not np.isfinite(keys).all() or not np.isfinite(queries).all():
                raise RuntimeError("Invalid capture geometry or values")
            native_scores = collected[(layer, "scores")].reshape(final_count, -1)[:, :count // 4]
            expected_visible = np.arange(count // 4)[None, :] < ((np.arange(final_count) + final_offset + 1) // 4)[:, None]
            if not np.array_equal(np.isfinite(native_scores), expected_visible):
                raise RuntimeError("Native pool ordering/mask differs from the contiguous-prefix assumption")
            projection_errors = None
            if expected_maps:
                score_queries = collected[(layer, "projected_queries")].reshape(final_count, 4, key_dimension)
                new_keys = collected[(layer, "new_keys")].reshape(-1, 128)
                projected_new = collected[(layer, "projected_new_keys")].reshape(-1, key_dimension)
                projection_errors = {"queries": verify_projection(queries, expected_maps[layer], score_queries, bool(args.gpu_layers)),
                                     "new_keys": verify_projection(new_keys, expected_maps[layer], projected_new, bool(args.gpu_layers))}
            else:
                score_queries = queries
            half_tile = not collected_host[(layer, "scores")] and (not expected_maps or key_dimension == 128)
            score_check = verify_scores(score_queries, keys, native_scores, expected_visible, half_tile,
                                        mixed_operands=bool(expected_maps and args.gpu_layers and key_dimension != 128))
            if expected_maps and args.gpu_layers and collected_host[(layer, "scores")]:
                raise RuntimeError("experimental hybrid validation requires Vulkan-resident scores")
            score_error = score_check["max_abs_error"]
            positions = np.arange(args.query_step - 1, final_count, args.query_step)
            global_positions = positions + final_offset
            manifest = {"schema": 1, "source_kind": "real_model_capture", "stage": "projected_native_indexer" if expected_maps else "post_norm_post_rope_pooled_indexer",
                        "model_id": "Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL", "layer": layer,
                        "document_ids": [args.document_id], "prompt_sha256": hashlib.sha256(text).hexdigest(),
                        "tokens": count, "decode_status": status, "intentional_early_abort": not args.complete_eval,
                        "attention_kv_types": "Q8_0/Q8_0", "flash_attention": True,
                        "device": "CPU" if not args.gpu_layers else f"CPU/Vulkan hybrid, n_gpu_layers={args.gpu_layers}, KQV offload enabled",
                        "checkpoint_identity": "caller label and verified metadata, not full weight hash",
                        "scope": "One independent contiguous text sequence; no prefix reuse, eviction or mixed sequences",
                        "native_qsa_score_max_abs_error": score_error, "native_causal_pool_mask_verified": True,
                        "native_score_buffer_is_host": collected_host[(layer, "scores")],
                        "source_sha256": capture_source_sha256, "support_source_sha256": support_source_sha256,
                        "score_arithmetic_check": score_check, "projection_reference": projection_errors,
                        "query_positions": global_positions.tolist(), "prefill_seconds": prefill_seconds,
                        "gpu_layers": args.gpu_layers, "private_commit_limit_gib": args.commit_limit,
                        "cpu_threads": args.threads,
                        "working_set_limit_gib": 8, "watchdog_seconds": args.timeout,
                        "native_map_sha256": hashlib.sha256(args.expected_native_map.read_bytes()).hexdigest() if expected_maps else None}
            path = args.output.with_name(f"{args.output.stem}.layer-{layer}.npz") if args.all_layers else args.output
            with path.open("xb") as output:
                np.savez(output, keys=keys, queries=queries[positions], visible=(global_positions + 1) // 4,
                         native_scores=np.where(np.isfinite(native_scores[positions]), native_scores[positions], 0).astype(np.float32), manifest=json.dumps(manifest))
            summaries.append(manifest)
        if args.complete_eval:
            vocab_size = bind(llama, "llama_vocab_n_tokens", I, [P])(vocab)
            if not 1 <= vocab_size <= 1000000:
                raise RuntimeError("vocabulary exceeds bounded logit capture")
            get_logits = bind(llama, "llama_get_logits_ith", P, [P, I])
            rows = min(33, final_count)
            logits = []
            for index in range(-rows, 0):
                pointer = get_logits(context, index)
                if not pointer:
                    raise RuntimeError("requested logit row absent")
                logits.append(np.ctypeslib.as_array(C.cast(pointer, C.POINTER(C.c_float)), shape=(vocab_size,)).copy())
            logits = np.asarray(logits)
            if not np.isfinite(logits).all():
                raise RuntimeError("non-finite model logits")
            targets = np.asarray(list(tokens)[count - rows + 1:count], dtype=np.int64)
            capture_final = False
            step_seconds, generated = [], []
            current = logits[-1]
            for _ in range(args.generate):
                token = int(np.argmax(current))
                generated.append(token)
                item = (I * 1)(token)
                start = time.perf_counter()
                if decode(context, batch_one(item, 1)) != 0:
                    raise RuntimeError("generation decode failed")
                pointer = get_logits(context, -1)
                if not pointer:
                    raise RuntimeError("generation logits absent")
                current = np.ctypeslib.as_array(C.cast(pointer, C.POINTER(C.c_float)), shape=(vocab_size,)).copy()
                step_seconds.append(time.perf_counter() - start)
            quality = {"schema": 1, "tokens": count, "gpu_layers": args.gpu_layers,
                       "model_id": "Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL", "prompt_sha256": hashlib.sha256(text).hexdigest(),
                       "context_tokens": params.n_ctx, "batch_tokens": 256, "cpu_threads": args.threads,
                       "offload_kqv": bool(args.gpu_layers), "op_offload": bool(args.gpu_layers),
                       "attention_kv_types": "Q8_0/Q8_0", "vocabulary_size": vocab_size,
                       "source_sha256": capture_source_sha256,
                       "support_source_sha256": support_source_sha256,
                       "runtime_dll_sha256": {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu-zen4.dll", "ggml-vulkan.dll", "libomp.dll")},
                       "prefill_seconds": prefill_seconds, "decode_step_seconds": step_seconds,
                       "generated_token_ids": generated, "decode_tokens_per_second": len(step_seconds) / sum(step_seconds) if step_seconds else None,
                       "native_map_sha256": summaries[0]["native_map_sha256"], "model_quality_approved": False}
            if expected_maps:
                size = bind(llama, "llama_state_get_size", SZ, [P])(context)
                seq_size = bind(llama, "llama_state_seq_get_size", SZ, [P, I])(context, 0)
                invalid = C.create_string_buffer(1)
                read = bind(llama, "llama_state_set_data", SZ, [P, P, SZ])(context, invalid, 1)
                if size != 0 or seq_size != 0 or read != 0:
                    raise RuntimeError("experimental serialized state was not refused")
                quality["serialized_state_refused"] = True
            with args.output.with_suffix(args.output.suffix + ".quality.npz").open("xb") as output:
                np.savez(output, logits=logits, targets=targets, manifest=json.dumps(quality))
            print("MODEL_QUALITY_CAPTURE", json.dumps(quality), flush=True)
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
    parser.add_argument("--tokens", type=int, default=256, choices=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192))
    parser.add_argument("--query-step", type=int, default=32, choices=(8, 16, 32))
    parser.add_argument("--gpu-layers", type=int, default=0, choices=(0, 4, 8, 16))
    parser.add_argument("--commit-limit", type=int, default=6, choices=(6, 40))
    parser.add_argument("--threads", type=int, default=4, choices=(4, 8, 12))
    parser.add_argument("--timeout", type=int, default=180, choices=(180, 600, 1200, 1800))
    parser.add_argument("--complete-eval", action="store_true")
    parser.add_argument("--generate", type=int, default=0, choices=(0, 8, 16))
    parser.add_argument("--expected-native-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    if args.generate and not args.complete_eval:
        parser.error("generation requires --complete-eval")
    if args.commit_limit != 6 and not args.gpu_layers:
        parser.error("the larger commit ceiling is only for GPU mappings; CPU capture retains 6 GiB")
    if args.complete_eval and args.output.with_suffix(args.output.suffix + ".quality.npz").exists():
        parser.error("quality output exists")
    if args.all_layers and any(args.output.with_name(f"{args.output.stem}.layer-{layer}.npz").exists() for layer in range(3, 48, 4)):
        parser.error("a per-layer output already exists")
    timer = threading.Timer(args.timeout, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        capture(args)
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
