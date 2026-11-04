# Experimental latent sparse attention primitives on AMD Vulkan

This is a standalone, synthetic proof of concept in a llama.cpp fork. It runs latent projection, selected-row gathering, and attention through ggml's existing Vulkan backend on an AMD Radeon 8065S (`gfx1151`). It does **not** add LSA to llama-server or Qwen3.8-Flash-Next, and it does not demonstrate an end-to-end inference speedup.

Base: llama.cpp b11681, commit `5e03a1ff450cae37a774684de55d14fb48b1f90b`. The experiment uses the public ggml C API; no llama.cpp core code, public headers, or dependent libraries were modified. No separate library fork is needed for this experiment.

## What is implemented

- `lsa_reference.py`: CPU projection/attention oracle and a median-split latent-key tree. Best-first traversal prioritizes axis-aligned bounding-box dot-product upper bounds. It scores a bounded number of candidate keys, ranks those candidates, and includes optional sink and recent-window tokens. Visibility is a causal prefix.
- `ggml_vulkan.py`: a small 64-bit Windows ctypes harness for the b11681 C API. Vulkan performs F32 projection and gathered attention using existing `mul_mat`, `get_rows`, scale, softmax, transpose, and contiguous operations. Unsupported operations fail; there is no CPU graph fallback.
- `run_experiment.py`: deterministic synthetic numerical checks and fixed-query graph timings. The projection is an **untrained identity-prefix map**, not a released or learned model map.
- `test_reference.py`: six CPU test groups covering exhaustive selection, causal visibility, a dominant needle, projection geometry, bounded cluster search, and invalid inputs.

The implementation is independently written experimental code under this repository's MIT license. It does not copy or execute Prefrontal-AI code. It is inspired by the [LSA description](https://prefrontal.ai/news/lsa/), rather than a faithful port or reproduction of that implementation.

## Reproduce

Use Python 3.12 or later (tested with official portable Python 3.14.10) and trusted, matching Windows x64 Vulkan binaries from [llama.cpp b11681](https://github.com/ggml-org/llama.cpp/releases/tag/b11681). Keep all DLLs required by that release together, including `ggml-base.dll`, `ggml.dll`, `ggml-vulkan.dll`, and `libomp.dll` where supplied. DLL loading executes native code: only pass a directory you trust. Other C API versions are not supported by this harness.

From the repository root in PowerShell:

```powershell
python -m unittest discover -s pocs/lsa-vulkan -p test_reference.py -v
python pocs/lsa-vulkan/run_experiment.py --runtime C:\trusted\llama-b11681-vulkan --tokens 512 4096 16384 --repeats 11 --output new-results.json
```

No pip packages, model downloads, SGLang, ROCm, compiler, administrator privileges, network access, or server configuration changes are needed to run the experiment. The runtime directory is supplied explicitly; the code does not download dependencies. On a shared GPU, wait until the serving workload is idle before running.

The harness sets `GGML_VK_DISABLE_F16=1` and `GGML_VK_DISABLE_COOPMAT=1` in its own process to require the tested F32 path. These settings do not change other processes or persistent system settings. This is a numerical validation configuration, not a tuned production baseline. CLI sizes are capped at 16,384 tokens, with 64 MiB per tensor buffer and a 128 MiB combined replay-buffer check. These limits exclude driver/backend workspace, Python objects, and transient transfers; this is not a general-purpose sandbox. Existing output files are refused.

## Measurements and limits

See [RESULTS.md](RESULTS.md) and [raw measurements](results/gfx1151-windows-f32.json). GPU primitives agree with the CPU oracle, but sparse attention is only approximate relative to dense attention. The measured search recall and approximation errors are published separately from numerical correctness.

The graph timer includes submission, GPU computation, and synchronization. It excludes setup, uploads, CPU selection, index construction, and result readback. Sparse graphs replay a fixed selected index set; the timed search runs separately and does not upload fresh indices into each replay. Adding the search and sparse-graph medians is only a partial cost estimate, not a measured integrated latency. It is already slower than dense attention at all measured lengths.

This synthetic experiment has one query and one head, with 256-wide keys/values and an eight-dimensional latent map. It does not cover multi-head/GQA behavior, RoPE, batch scheduling, quantized KV, paging, KV mutation/eviction, prefix reuse, or model execution. Full keys and values remain allocated on the GPU; this implementation does **not** reduce KV memory. The dense comparator uses ordinary ggml operations, not a tuned fused attention kernel. Repeated sub-millisecond measurements on a shared desktop are preliminary and are not representative serving benchmarks.

The tree's candidate cap bounds rescoring, not total internal-node traversal or tree construction; no subquadratic worst-case guarantee is claimed. In the raw schema, `centroids_scored` is a legacy field name for bounding-box node scores, not centroid distance calculations. Candidate-budget search is approximate and is not guaranteed to recover exact top-k, even though node bounds are valid.

## What is still required for Flash Next

The [published LSA map release](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1) inspected for this work supplies maps for Qwen3.8-27B and GLM5.3-Flash. No compatible trained Flash Next map was available in that release. A map trained and validated for the exact Flash Next layers and attention geometry is a prerequisite; the synthetic map here cannot substitute for it.

A serving implementation would then need model-specific graph integration, correct head/position handling, a persistent incrementally maintained index tied to the KV lifecycle, efficient selection/index transfer or GPU search, and proper synchronization across concurrent slots. It must pass dense equivalence tests when selecting all tokens, causal/prefix/eviction tests, and long-context quality evaluations with the trained map before matched model throughput/TTFT benchmarks. The current 16K synthetic run does not validate five 256K contexts or any tokens-per-second improvement.

The existing five-slot server was preserved during this experiment; [service-preservation.json](results/service-preservation.json) records the post-run checks. No change to serving behavior is included in this branch.
