# Experimental native Flash Next indexer

This branch adds an opt-in low-rank QSA scoring path to the pinned b11681 native model and `llama-server`. It is an independent experiment. It is not Prefrontal's hierarchical LSA implementation, and a map is not approved for production by loading successfully. The earlier synthetic search and short-capture results remain in [FLASH_NEXT.md](FLASH_NEXT.md).

## Native graph and cache

The loader accepts only the inspected Flash Next geometry: 48 layers, QSA layers 3,7,...,47, four indexer query heads, 128 indexer dimensions, four tokens per pool, and a 2,048-token selection budget. Checkpoint labels and geometry do not authenticate the weight contents. MTP and more than one physical GPU are rejected in this experimental mode.

Each layer has an orthonormal projection with rank 16, 32, 64, or 128; rank is a dimension count, not a quantization bit width. Query normalization and RoPE precede projection. Raw indexer keys remain in their original cache lanes. Each newly completed or regrouped pool is averaged, normalized and rotated by the native implementation, then projected. Its projected vector is padded to 128 lanes and written to the existing pooled slot. Previously completed pools reuse those values. Native cache invalidation and re-pooling remain responsible for edits. This does not reduce the allocated KV/indexer cache size. Prefix reads from quantized storage round up to a whole quantization block, dequantize, then trim when necessary. This supports rank16 with the 32-value Q8_0 block size.

The score is `sum_h(ReLU(dot(P*q_h, P*k))) / sqrt(128)`. The scale remains the original scale. Ranks below 128 use four ggml matmuls, per-head ReLUs, additions, scaling, and the native causal pool mask. Projection and scoring explicitly permit FP16-or-better operands and require F32 accumulation. Tensor storage remains F32 at these operations. The pinned Vulkan vector path uses F32 arithmetic; batched cooperative kernels round operands internally to FP16. They scan every visible pool, then use the existing native top-512-pool selection and attention. This is linear scanning in a smaller space; there is no GPU tree traversal or candidate reranking. Rank 128 uses the stock fused indexer and provides an identity-map control.

GPU attention offload can be enabled even when most weight layers are on the CPU. Projection constants therefore have independent CPU and single-GPU copies, selected by the context's KQV-offload flag. No public C API parameter structure changes are required. The tested Windows build reuses the trusted b11681 Vulkan plugin because the graph uses existing operations; it does not need a new Vulkan shader or SDK.

Serialized full and sequence state reads/writes are refused while maps are enabled. This prevents incompatible original/projected pooled slots from entering prompt caches or state files. For an isolated serving test, disable RAM prompt caching and context checkpoints. Fresh contiguous prefill and generation are the validation scope; prefix reuse, arbitrary edits, eviction, multiple slots and long-running cache reuse still need dedicated tests.

## Build and test

Use the pinned upstream source, a C++17 compiler, CMake and an existing compatible Vulkan backend, or build that backend using the upstream instructions. The Windows test build used official Microsoft Build Tools, MSVC 14.44 and Windows SDK 26100. Its CPU-only core was configured with dynamic backends and loaded the existing official Vulkan and Zen 4 CPU plugins. Do not mix this with an unrelated llama.cpp ABI.

```powershell
cmake --build build --target llama-server test-qwen4exp-pca
build/bin/test-qwen4exp-pca.exe
build/bin/test-qwen4exp-pca.exe --vulkan
python -m unittest discover -s pocs/lsa-vulkan -p 'test_*.py'
```

The C++ tests check bounded map parsing and malformed input rejection, CPU/GPU constant placement, independent CPU-copy retention, and numerical scores/causal masks at all four supported ranks with both three and 32 queries. The second shape exercises the batched cooperative kernel. A single consistent F32 or rounded-operand oracle must match the whole test within 1e-4; the full unrounded difference is also reported. Each Vulkan graph operation must be supported by that backend; the test refuses a CPU fallback. The Python tests check capture validation, search references, map export, matched logit comparisons, FP16 error bounds and strict rounded-operand oracles, and the distillation gradient/objective.

## Capture, calibration and quality

`capture_flash_next_cpu.py` retains its historical filename and CPU-only defaults. It now supports an explicit bounded hybrid configuration. It never attaches to the production process. All new model tests use separate model/context instances, mmap/lazy PLE, disabled repacking/MTP, Q8_0 K/V, 256-token chunks, and a watchdog. A Windows job limits private commit and kills descendants when the capture process exits; the process has an 8 GiB hard working-set limit and below-normal priority. A larger 40 GiB private-commit ceiling is available only with explicit GPU offload. GPU mappings, file mappings and driver allocations have different accounting; these limits are not global memory isolation.

Two independently written public-example documents are truncated to 4,096 model tokens. The final chunk supplies 32 sampled queries per layer at positions 3847,3855,...,4095 and up to 1,024 completed pools. All these queries exceed the native 512-pool selection budget. The collector verifies exact causal visibility, pooled order, and native scores. Native score rows are included in new captures and used as the training/evaluation teacher. The stock Vulkan fused indexer rounds operands and intermediate dot products to FP16, so its numerical check uses an independently derived conservative error bound. The projected path must additionally match one consistent F32 or FP16-operand oracle within a dimension-dependent F32 accumulator/reduction bound. Projection outputs receive the same independent operand-oracle check. Unrounded error is reported separately; the conservative conversion bound alone cannot pass a projected result. GPU tests also require Vulkan-resident score buffers. Source/helper fingerprints are frozen at capture start. The projected pooled-key collector uses the graph's final callback name, `experimental_qsa_pca_k`, because its earlier `indexer_k` alias is overwritten.

```powershell
python pocs/lsa-vulkan/capture_flash_next_cpu.py --runtime C:\trusted\native-bin --model C:\models\checkpoint-00001-of-00004.gguf --prompt train.txt --document-id public-train --all-layers --tokens 4096 --query-step 8 --gpu-layers 4 --commit-limit 40 --threads 12 --timeout 1800 --output train.json
python pocs/lsa-vulkan/capture_flash_next_cpu.py --runtime C:\trusted\native-bin --model C:\models\checkpoint-00001-of-00004.gguf --prompt heldout.txt --document-id public-heldout --all-layers --tokens 4096 --query-step 8 --gpu-layers 4 --commit-limit 40 --threads 12 --timeout 1800 --complete-eval --generate 8 --output baseline.json
python pocs/lsa-vulkan/calibrate_native_layers.py --profile pocs/lsa-vulkan/flash-next-profile.json --train train.json --heldout baseline.json --rank 64 --output-directory calibration-new
```

The batch calibrator produces separate PCA and independently distilled maps. Distillation starts from uncentered PCA and minimizes the training teacher's causal-pool softmax KL at temperature 2, using an analytic gradient, a tangent update and QR retraction on orthonormal rows. It performs at most 50 accepted updates with backtracking. Held-out data is used only for reporting. This small independent training method is not Prefrontal's D-LSA recipe. Native flat top-512 recall is reported separately from the older bucket/reranking diagnostic; with 1,024 pools and 1,024 bucket candidates, the latter is trivially exact.

Exported `QPCAMAP1` files contain an 8-byte magic/version, little-endian dimension/rank/layer-count u32 fields, then the ordered layer IDs and rank-by-128 F32 row-major projections. The native parser requires finite orthonormal rows, bounded file size, exact layer order and no trailing data. Sidecars record source-map hashes. Neither format nor `serving_ready: false` is a security signature or quality certification.

```powershell
$env:LLAMA_QSA_PCA_MAP='C:\experiments\calibration-new\distill-rank64.qpc'
$env:LLAMA_QSA_PCA_EXPERIMENTAL='1'
python pocs/lsa-vulkan/capture_flash_next_cpu.py --runtime C:\trusted\native-bin --model C:\models\checkpoint-00001-of-00004.gguf --prompt heldout.txt --document-id public-heldout --all-layers --tokens 4096 --query-step 8 --gpu-layers 4 --commit-limit 40 --threads 12 --timeout 1800 --complete-eval --generate 8 --expected-native-map $env:LLAMA_QSA_PCA_MAP --output candidate.json
python pocs/lsa-vulkan/compare_model_quality.py --baseline baseline.json.quality.npz --candidate candidate.json.quality.npz --output comparison.json
Remove-Item Env:LLAMA_QSA_PCA_MAP,Env:LLAMA_QSA_PCA_EXPERIMENTAL
```

The comparator requires matching prompt, runtime DLL and capture/helper source hashes, context/batch size, threads, offload layers, KV types, vocabulary and target IDs. It reports 32-position teacher-forced NLL/perplexity, KL, logit top-1 agreement, greedy output agreement and C API timings. Prefill includes final-chunk activation callbacks and extra logit rows; decoding disables activation callbacks. These timings exclude model loading and are not an HTTP benchmark. One document and eight greedy steps do not establish task accuracy, 256K quality or C1-C5 throughput.

`run_native_server_probe.py` starts the experimental server on **127.0.0.1:18080 only**, disables state caching/checkpoints, sends one eight-token chat request and terminates its own server in `finally`. It refuses an occupied port and existing outputs. Its child inherits the job limit and receives its own working-set limit. This is a serving smoke test, not a speed or cache-lifecycle qualification. Run model tests sequentially on an idle GPU and provide an external memory/production-activity guard when sharing a machine.

## Resource boundary

The AMD machine has 192 GiB unified physical memory with a 160 GiB GPU carveout and approximately 31.7 GiB visible to Windows. The unchanged production process uses about 94 GiB of dedicated GPU memory with five 262,144-token slots. A second fully offloaded weight copy cannot fit alongside it. In addition, Windows free commit was only about 11.8 GiB before isolated model loading. A separate 16-layer load failed allocating 1,030,953,856 bytes with `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`, even after raising only the test process's private-commit ceiling. The Vulkan device's reported free budget did not predict that failure.

Four GPU layers load successfully alongside production, with most weight computation on CPU and KQV offload on Vulkan. This permits graph, output and serving validation without replacing the live server. Matched full-GPU performance testing requires a maintenance window to stop and restore production, or another sufficiently large machine. No pagefile, carveout, security setting or production launcher change is part of this experiment.
