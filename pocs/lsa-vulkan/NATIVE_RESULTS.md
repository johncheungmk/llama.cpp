# Native Flash Next experiment

The experimental map path runs in the native model and an isolated localhost `llama-server`. It is flat low-rank QSA scanning, not hierarchical LSA. Production remains unchanged. Full-GPU model speed, 256K quality and five concurrent occupied 256K contexts are not validated by these results.

## Build and numerical checks

The pinned b11681 source plus this branch builds with official Microsoft Build Tools, MSVC 14.44.35207 and Windows SDK 26100. The shared core uses dynamic backends, with the existing official b11681 Zen 4 CPU and Vulkan plugins. No new shader, Vulkan SDK, ROCm/SGLang installation or execution of Prefrontal code was required. Official portable Python 3.14.10 and the verified NumPy 2.5.4 wheel remain isolated in the development directory.

The native loader test accepts a valid map and rejects ten malformed map cases. CPU/GPU constant placement and CPU-copy retention pass. Four-head projected scoring and causal masks pass on both CPU and Vulkan for ranks 16,32,64,128 with both three and 32 queries. Three-query Vulkan vector fixtures use F32 operands; 32-query cooperative fixtures round both operands to FP16 and accumulate in F32. The entire fixture must match one consistent independent operand oracle within 1e-4. The largest observed oracle error is 6.324e-6; the largest difference from unrounded scores is 0.000736. Vulkan tests refuse unsupported operations and CPU fallback. All **23 Python test groups pass locally and on the AMD machine**. [Raw numerical cases](results/native-unit-results.json) and [final build/source/DLL fingerprints](results/native-build-final.json) are provided.

## Real 4K calibration

Training and held-out documents are the separate [public fixtures](fixtures/README.md). Training used the stock b11681 runtime; the held-out baseline uses the new native build with maps disabled. Each capture has 4,096 model tokens and all twelve QSA layers. Each layer supplies 1,024 pooled keys and 32 late-position four-head queries at positions 3847,3855,...,4095. Every sampled query requires pruning at the native 512-pool budget.

The collector validates pooled order, causal masks and native scores. Stock Vulkan scores use FP16 dot4 intermediates; maximum training/reference absolute difference is 0.004843, inside the conservative arithmetic bound. This is distinct from the F32 projected-operation numerical test. Attention K/V and indexer caches use the serving configuration's Q8_0 types.

| Rank-64 map | Mean top-512 pool recall | Range of layer means | Worst sampled query | Score relative L2 error across layers |
| --- | ---: | ---: | ---: | ---: |
| Uncentered PCA | 91.3142% | 89.0076%-93.2373% | 80.6641% | 5.0615%-12.6894% |
| Teacher distillation | 91.4312% | 89.2517%-93.4570% | 80.6641% | 5.1144%-12.4873% |

The independent distillation method starts from PCA and performs 50 training-only tangent updates with QR retraction and backtracking. Normalizing the tangent displacement reduces training KL by approximately half; held-out recall improves only 0.117 percentage points over PCA. No held-out gradient or map selection is used. The fixed revised optimizer's map is used for the model-output test. The small gain does **not** resolve the ranking loss, and two repetitive documents do not establish generalization.

The raw [final calibration](results/native-calibration-rank64.json) includes all layer/query diagnostics, training loss traces, source/capture/map hashes, and separate native-flat versus bucket-reranking recall. The older bucket diagnostic covers all 1,024 pools in this fixture and is trivially exact; it is not the native path's quality result. [Initial optimizer diagnostics](results/native-calibration-initial.json) are retained separately. Their historical source hash refers to the earlier update rule `learning_rate/max(1, gradient_norm)`; the revised rule uses a bounded normalized tangent displacement with training-only backtracking.

The first rank64 native 4K prefill completed, but its collector failed with `Incomplete capture, decode status 0, errors []`: the graph renames the same pooled-key tensor to `experimental_qsa_pca_k`, while the collector expected its earlier `indexer_k` name. No final quality archive or generation timing from that run is claimed. The corrected collector uses the final name, freezes fingerprints at capture start, verifies Vulkan score residency and checks projection/scoring against strict rounded-operand oracles. Native operations explicitly permit FP16-or-better operands with F32 accumulation. A separate fix reads whole quantization blocks before trimming the rank16 prefix, resolving the initial rank16/Q8_0 alignment case. These fixes have passed the real-model control below. Fresh matched 4K baseline/projected runs are in progress; no completed pruning-quality model comparison is claimed yet.

## All-visible native model control

The final build was tested on the same 32-token held-out prefix with maps disabled and with the experimental rank16 map. Each run used one 512-token context, twelve CPU threads, four GPU weight layers, Q8_0 K/V, 256-token batch/ubatch, and eight greedy generation steps. All visible pools fit within the native selection budget, so this checks that the integrated path preserves model outputs when no pruning is necessary.

Across 31 teacher-forced positions, the maximum full-vocabulary logit difference, NLL delta and KL divergence are **exactly zero**; logit top-1 agreement is 100%. The eight generated token IDs also match exactly. All twelve projected score buffers are Vulkan-resident; their largest error against the independent operand oracle is 3.087e-6. Projection checks, causal masks and rejection of full/sequence saved-state operations pass. This validates the quantized rank16 cache read and the all-visible output control, not long-context pruning quality. [Matched output comparison](results/native-control-quality.json), [baseline capture checks](results/native-control-baseline-capture.json), and [projected capture checks](results/native-control-rank16-capture.json) include the exact runtime and source fingerprints.

## Vulkan indexer overhead

This is a single-query synthetic subgraph test with four query heads, 128 dimensions, rank64 and top512 selection. The prebuilt graphs include pooled-row gathering, query projection for the low-rank path, scores and selection. They exclude setup, uploads, cache construction/new-pool projection, output readback, attention and model execution. Each graph has three warmups and fifteen alternating synchronized measurements. The cache fixture uses F32 raw128|pooled128 rows; the actual model's cache is quantized.

| Represented tokens | Pooled vectors | Stock fused indexer (ms) | Rank64 path (ms) |
| ---: | ---: | ---: | ---: |
| 4,096 | 1,024 | 0.0620 | 0.1050 |
| 16,384 | 4,096 | 0.0871 | 0.1265 |
| 65,536 | 16,384 | 0.1806 | 0.2165 |
| 262,144 | 65,536 | 0.7113 | 0.6253 |

The final projected path is about 12.1% faster on the largest indexer fixture and slower on the other three. Existing operation/submission overhead limits short-context gains. The fixture lies entirely in the rank64 subspace; synthetic selection recall is 100%,100%,99.8047%,100%. The stock shader's FP16 dot4 arithmetic differs from the projected vector path's F32 arithmetic and can change near-tied selections. These recall values do not replace the real-checkpoint calibration table. [Final raw samples and exact timing scope](results/native-indexer-benchmark-final.json) are provided; [initial measurements](results/native-indexer-benchmark.json) remain as historical evidence for the initial build. No model tokens/sec speedup follows from these subgraph timings.

## Native server smoke

The isolated rank64 server bound only to `127.0.0.1:18080`, with one 512-token slot, four GPU weight layers, twelve CPU threads, Q8_0 K/V, no MTP, RAM prompt cache or context checkpoints. The remaining weight layers ran on CPU. It loaded the map, became healthy and returned eight tokens for a 24-token public-example prompt: `The number after two is **3**.` The request took 60.57 seconds. The server's own reported decode rate was 0.1912 tokens/sec; this is a bounded hybrid smoke result, not a production or full-GPU benchmark. The server was terminated and its localhost port verified closed. [Raw response and settings](results/native-server-smoke.json).

## Resource and scope limits

The machine's 192 GiB unified memory has a 160 GiB GPU carveout and approximately 31.7 GiB visible to Windows. Production uses approximately 94 GiB of dedicated GPU memory. A separate sixteen-layer GPU load failed allocating 1,030,953,856 bytes with `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`, including with a larger test-only private-commit ceiling. The device's reported free Vulkan budget did not predict usable system headroom. Four GPU layers fit alongside production; a second fully GPU-resident copy does not fit the current dedicated allocation.

Model tests therefore use a CPU/Vulkan hybrid, an 8 GiB hard working-set limit and a 40 GiB private-commit ceiling, below-normal priority, and a watchdog. An external guard aborts the test if Windows free commit/physical RAM falls below 3 GiB, production starts processing a request, or production availability cannot be verified. Windows free commit was only about 11.8 GiB before model loading. File mappings and driver allocations have different accounting; the guard is not global isolation.

The live launcher, scheduled task and service are preserved. Full-GPU C1-C5 testing requires a separately authorized maintenance window to stop and restore production, or another large machine. Broader quality data, 256K validation, prefix/edit/eviction and concurrent-slot tests, and efficient hierarchical GPU search remain outstanding. The flat low-rank experiment does not provide production LSA readiness or a model throughput recommendation.
