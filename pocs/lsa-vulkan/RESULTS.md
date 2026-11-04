# AMD Vulkan proof-of-concept results

These are synthetic primitive measurements, **not Qwen3.8-Flash-Next inference results**. No language model or learned LSA map is loaded. There is no demonstrated end-to-end speedup.

Tested on 2026-11-04 at 12:17 UTC: AMD Ryzen AI Max+ PRO 495 / Radeon 8065S (`gfx1151`), Windows 11 build 26200, AMD driver 32.0.32015.7029, Python 3.14.10. The official llama.cpp b11681 Vulkan DLLs were copied into an isolated development directory. Strict F32 flags disable F16 and cooperative-matrix paths in the experiment process. Production model settings were unchanged.

## Timing

One query, 256-dimensional keys and values, latent dimension 8, top-k 32, candidate budget 256, leaf size 32, two sink and eight recent tokens. Each fixture selected 42 rows. Seed 4361; 11 alternating dense/sparse graph replays after warmup. All values below are milliseconds, except the GPU-only ratio.

| Synthetic tokens | Dense graph median | Sparse graph median | GPU-only dense/sparse ratio | CPU search median | Search + sparse medians* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.1171 | 0.1097 | 1.07x | 0.1874 | 0.2971 |
| 4,096 | 0.1325 | 0.1080 | 1.23x | 0.2298 | 0.3378 |
| 16,384 | 0.5952 | 0.1453 | 4.10x | 0.5810 | 0.7263 |

*A sum of separately measured medians, not an integrated timing. It omits query projection, fresh-index transfer, index updates, and other overhead. The sparse replay uses fixed indices. Even this partial estimate is slower than the dense graph at every tested length. A faster sparse GPU operation alone is insufficient to establish an inference speedup. The dense graph is not a fused production attention baseline.

One-time projection including graph setup, input conversion/transfers, and result readback took 13.7 / 100.9 / 824.9 ms respectively; the GPU projection-only timer was 0.269 / 0.399 / 0.837 ms. CPU tree construction took 4.28 / 66.62 / 575.48 ms. These costs are excluded from the replay table. Combined dense and sparse tensor buffers peaked at 84,216,480 bytes for the 16K fixture, excluding backend workspace and Python memory. Full K/V are retained in both graphs, so this is not a KV memory-saving demonstration.

## Correctness and approximation quality

All six CPU test groups passed locally and on the AMD PC. GPU projection, dense attention, full gather, causal sparse attention, a future-needle exclusion case, and a controlled dominant-needle case passed the 0.0003 maximum absolute error threshold against the appropriate CPU reference. Across the benchmark fixtures, the largest GPU-versus-CPU attention error was below 1.2e-7. Source and DLL SHA-256 fingerprints are included in the raw JSON.

Passing these checks means the GPU operations compute their selected attention accurately. It does **not** establish that selection preserves dense attention quality:

| Synthetic tokens | Latent top-32 recall among selected rows | Sparse versus dense max absolute output error |
| ---: | ---: | ---: |
| 512 | 100% | 0.07283 |
| 4,096 | 56.25% | 0.07101 |
| 16,384 | 43.75% | 0.05644 |

Recall compares the selected set (including sink/window rows) with exhaustive top-32 in the synthetic latent space. The untrained map and fixed search budget have not been validated on model activations. These output errors are reported observations, not an acceptable model-quality threshold.

## Evidence and serving preservation

- [Raw measurements and numerical checks](results/gfx1151-windows-f32.json), including all 11 timing samples per fixture, method, source hashes, and DLL hashes.
- [Service preservation checks](results/service-preservation.json): unchanged serving PID and launcher hash, scheduled task running, health `ok`, five idle slots, 262,144 tokens allocated per slot. No server restart was performed for this experiment. Allocation is not proof of five fully occupied 256K sequences.

Only an official portable Python runtime was added in the separate development directory; existing official DLLs were copied there. No SGLang/ROCm installation, security changes, model modifications, or Prefrontal third-party code execution occurred. The portable Python archive was verified against the official release manifest (SHA-256 `b553e7dac884469b40e0740bb527271c3378f700b427e0485835f1bb6424361c`); the Python executable's signature was valid and identified the Python Software Foundation.

The next useful engineering work is a trained Flash Next map and model/KV integration, followed by model accuracy testing and a complete matched benchmark. The experiment does not justify changing the production server or predicting an LSA tokens-per-second result.
