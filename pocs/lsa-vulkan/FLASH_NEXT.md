# Flash Next calibration and search: phase 2

This phase implements a real-checkpoint calibration path and reduces experimental search overhead. It does **not** enable LSA in llama-server, produce a validated long-context model, or measure Flash Next model tokens/sec.

## Exact geometry and scoring

The installed Qwen3.8-Flash-Next Q4_K_XL-IQ4NL GGUF reports architecture `qwen4exp`: 48 layers, 24 attention query heads, two attention KV heads, and 256-dimensional attention heads. Its 12 QSA layers are 3, 7, ..., 47 (zero-based). Each QSA indexer has four 128-dimensional query heads and a shared pooled-key vector. Four tokens form one pool; the checkpoint selects up to 512 pools (2,048 tokens), plus its native incomplete tail.

The teacher pool score is `sum_h(max(0, dot(query_h, pooled_key))) / sqrt(128)`, matching [the pinned native implementation](../../src/models/qwen4exp.cpp). Rectification happens separately per head, before summation. Summing heads before rectification is incorrect. The first prototype's single-vector search is not a general substitute for this objective. [flash-next-profile.json](flash-next-profile.json) records the observed geometry; its checkpoint label and metadata are not a full weight digest.

## Implemented path

- `capture_flash_next_cpu.py` loads the existing GGUF through b11681's public C API in a **separate CPU-only process**. It explicitly loads the Zen 4 CPU backend, disables weight repacking and MTP, uses mmap/lazy PLE, four CPU threads, and the serving configuration's Q8_0 K/V types with flash attention enabled. No serving-process attachment or GPU model allocation occurs.
- The capture process requires successful Windows job/private-commit and hard working-set limits: 6 GiB private commit, 8 GiB working set, a 180-second watchdog, and below-normal priority. These are limits on this process only. File mappings, OS cache and driver memory have different accounting; this is not a system-wide isolation guarantee. Limits failing to initialize stop the capture before loading weights.
- Scheduler callbacks read `indexer_q`, `indexer_k`, and native `indexer_score` at requested layers. They verify pool order, causal masks, and scores against an independent NumPy calculation, then intentionally end graph evaluation after the requested data. Capture is limited to one fresh contiguous text sequence of at most 256 tokens, with no prefix reuse, eviction, or mixed sequences.
- `flash_next_calibration.py` fits an uncentered equal-key/query second-moment PCA projection. It preserves the score origin rather than subtracting a mean before the per-head ReLU. This is an independently implemented experimental PCA method, **not a distilled D-LSA map or a reproduction of Prefrontal's training-free algorithm**.
- Held-out evaluation rejects overlapping document IDs or identical archives, wrong model/layer/stage labels, object/pickle arrays, oversized archives, invalid shapes, and non-finite tensors. Provenance labels are caller supplied, not cryptographically authenticated model identities. Outputs always carry `serving_ready: false`.
- `QsaBucketIndex` partitions projected keys into median-split buckets, scores every bucket using per-head bounding boxes plus residual-norm terms, rescores a capped candidate set in the full 128-dimensional space, and returns top pools with deterministic ties. This is a vectorized flat bucket scan, with linear bucket work; it is not a GPU hierarchy or subquadratic worst-case implementation. For earlier causal prefixes it rebuilds from the prefix so future keys cannot influence bucket construction or ranking. A serving implementation needs incremental maintenance.

## Reproduce calibration

Use an isolated Python environment and NumPy. AMD tests used the official NumPy 2.5.4 Windows CPython 3.14 wheel, checked against PyPI's SHA-256 `2ca144f15135b6212a5c47b1e2aeca6e412f102f95a2d5d88d8aec77eb255de3`. Local tests used verified NumPy 2.3.5 with Python 3.11. No NumPy binaries are committed. The original stdlib-only experiment remains available separately.

Use trusted official b11681 Windows DLLs, including `llama.dll`, `ggml-base.dll`, `ggml.dll`, `ggml-cpu-zen4.dll`, and their release dependencies. CPU capture is specific to the tested x64 ABI/CPU backend and refuses other ABI layouts. On this AMD PC it completed within its process limits; it is not intended as an unattended general model loader.

From the repository root in PowerShell, with two different UTF-8 text files and new output filenames:

```powershell
python -m unittest discover -s pocs/lsa-vulkan -p 'test_*.py' -v
python pocs/lsa-vulkan/capture_flash_next_cpu.py --runtime C:\trusted\b11681 --model C:\models\checkpoint-00001-of-00004.gguf --prompt train.txt --document-id train-doc-1 --all-layers --output train.json
python pocs/lsa-vulkan/capture_flash_next_cpu.py --runtime C:\trusted\b11681 --model C:\models\checkpoint-00001-of-00004.gguf --prompt heldout.txt --document-id heldout-doc-1 --all-layers --output heldout.json
foreach ($layer in 3,7,11,15,19,23,27,31,35,39,43,47) {
    python pocs/lsa-vulkan/flash_next_calibration.py --profile pocs/lsa-vulkan/flash-next-profile.json --train-capture "train.layer-$layer.npz" --heldout-capture "heldout.layer-$layer.npz" --rank 16 --output "map-layer-$layer.npz"
    if ($LASTEXITCODE -ne 0) { throw 'Calibration failed' }
}
```

The NPZ capture has numeric `keys` `[pools,128]`, `queries` `[queries,4,128]`, integer `visible` `[queries]` naming complete causal pool counts, and a scalar Unicode JSON `manifest`. The required manifest fields are `schema:1`, `source_kind`, `model_id`, `layer`, `stage:post_norm_post_rope_pooled_indexer`, and disjoint nonempty `document_ids`. The collector supplies additional prompt hashes, native-score checks, query positions and cache settings. Loading uses `allow_pickle=False`.

All 12 rank-16 maps were created on the AMD PC in the isolated development directory's `calibration-v2` folder. They are retained there as experimental NPZ files; map hashes and diagnostics are in [the public calibration report](results/flash-next-real-calibration.json). No map is installed into the server. The captures and map binaries are not published with this branch.

## Real-checkpoint validation limits

Both training and held-out captures used separate, independently written public-example prompts, truncated to 256 tokens each. Per layer, training provided 64 pooled keys and eight four-head query snapshots. Held-out visibility ranges from eight to 64 complete pools. Native teacher scores agree with the independent calculation to a maximum absolute error of **3.12e-6** across all 24 layer/document captures; causal pool masks agree exactly.

The projected scores' held-out relative L2 errors range from **5.80% to 16.23%** across layers. This measures score approximation, not model answer accuracy. On the four query snapshots with more than 32 visible pools, mean diagnostic top-32 recall ranges from **75.78% to 89.84%** across layers. This indicates material ranking loss in the low-rank scores. Shorter prefixes select every key and cannot test ranking loss; their trivial recall is separated in the raw report.

At the checkpoint's actual 512-pool budget, **every held-out query selects all visible pools**. Its resulting selection recall is therefore trivially 100%; it provides no evidence for long-context pruning quality. No long-context perplexity, RULER, task accuracy, output equivalence, or generation-speed result is claimed. Calibration on two short examples cannot establish generalization to 256K positions or a different quantization/backend.

## Performance method and scope

Run the separate performance experiment with trusted b11681 Vulkan DLLs:

```powershell
python pocs/lsa-vulkan/run_flash_next_experiment.py --runtime C:\trusted\b11681-vulkan --output new-flash-results.json
```

This produces two **synthetic** comparisons, independent of the real model captures/maps. OpenBLAS/OMP are pinned to one thread in this experiment process.

1. QSA CPU search uses the real four-head/128-dimensional geometry, a rank-16 PCA fit on separate synthetic training keys/queries, 32-key buckets, 512 selected pools, and 4,096 candidates. Eight held-out synthetic queries are alternated against an exact NumPy QSA scan. Pooled counts of 4,096 / 16,384 / 65,536 represent 16K / 64K / 256K tokens. These are synthetic index sizes, not actual model contexts.
2. An integrated wall-clock comparison uses the same fixed synthetic fixture and rank-8 identity map for the old tree and new bucket search. It includes CPU selection, fresh query/index serialization and upload, GPU compute/synchronization, and output readback under one timer, with prebuilt graphs/indices. Eleven alternating repeats follow warmup. This is one attention head with 128-dimensional keys/values and top-32 plus sink/window rows; it is **not** the full Flash Next geometry or inference pipeline. A changing query/index lifecycle and index construction/update cost remain outside this comparison. Each GPU result is checked against its selected CPU attention oracle.

The dense comparator is an ordinary ggml operation graph, not tuned fused production attention. CPU search and Vulkan comparisons use different fixtures/objectives; their ratios must not be multiplied together. The measured rows and all samples are in [flash-next-v5.json](results/flash-next-v5.json).

| Synthetic represented tokens | Exact CPU QSA scan (ms) | Bucket selection (ms) | Mean / minimum pool recall |
| ---: | ---: | ---: | ---: |
| 16,384 | 0.1272 | 0.4938 | 100.0000% / 100.0000% |
| 65,536 | 0.4333 | 0.7260 | 100.0000% / 100.0000% |
| 262,144 | 4.5309 | 0.9471 | 99.9512% / 99.6094% |

Recall uses the top-512 native-geometry QSA teacher on these synthetic fixtures. At 4,096 pools the candidate budget covers all keys, so recall is exact and no search pruning occurs. These CPU scan timings are not Vulkan indexer timings.

| Synthetic attention tokens | Dense integrated (ms) | Original tree integrated (ms) | Bucket integrated (ms) |
| ---: | ---: | ---: | ---: |
| 512 | 0.2297 | 0.4089 | 0.3302 |
| 4,096 | 0.1823 | 0.4761 | 0.3346 |
| 16,384 | 0.2923 | 0.9194 | 0.5683 |

The new selection path reduces overhead relative to the original tree, but the dense graph remains faster at all three tested attention sizes. On this separate fixture, original/new top-32 recalls are identical: 100%, 62.5%, and 43.75%. Sparse-versus-dense output error remains nonzero; the change does not resolve the first prototype's approximation-quality problem.

## Remaining blocker to a model throughput claim

The maps are offline experiments. There is no validated native adapter that loads them into `llama-server`, manages projected pooled keys across append/eviction/prefix-sharing, and performs the intended selection on Vulkan. Publishing a model tokens/sec comparison before that integration and long-context quality validation would label the existing native QSA server as LSA incorrectly.

A second fully GPU-resident copy of the approximately 95.4 GiB GGUF weights would also exceed the configured 160 GiB GPU allocation alongside the running copy, before KV/workspace. The CPU-only collector avoids this duplication, but is not a substitute for matched GPU model serving measurements. The current server is preserved; this phase does not include a service switch or an upstream PR.

All 12 CPU test groups passed locally and on the AMD PC. [The post-run service audit](results/service-preservation-phase2.json) confirms the same serving PID and launcher hash, healthy running task, five idle 262,144-token slots, and no remaining capture processes. Source/DLL fingerprints accompany the calibration and synthetic results. Historical phase-1 source hashes refer to that earlier implementation and remain unchanged.
