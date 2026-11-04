# Proposed full-GPU comparison

This procedure is prepared but has not been executed. It requires separate authorization for a maintenance window because the current instruction preserves the live service. The native implementation, model-output controls and hybrid 4K validation are separate from this proposed full-GPU benchmark.

## Window and restoration

Reserve approximately 30 minutes for two server loads, warmup, matched C1-C5 runs and restoration; actual duration can vary. Before stopping anything, record the production PID, exact scheduled-task definition, launcher SHA-256, model path and arguments, health and all five slot sizes. Require them to match the known service. Keep the production launcher, task definition, model and security settings unchanged.

After explicit window approval, stop the production task and verify its owned server has exited before loading the separate test server. Run one experimental server at a time on `127.0.0.1:18080`; refuse an occupied port. Never keep a baseline and candidate model resident together. Terminate only the owned test process in `finally`, then restart the existing production task and verify health, the original launcher hash and five 262,144-token slots. Failure or memory pressure aborts the experiment and prioritizes restoration. Do not change the pagefile, GPU carveout, firewall or SSH configuration.

The current production process reports approximately 98.4 GiB of private commit, so the hybrid collector's 40 GiB ceiling cannot be reused for a full-GPU model. A proposed test-only ceiling is 120 GiB, subject to a fresh commit/budget check after production stops. This is a process limit, not reserved memory or a verified fit guarantee. Monitor real Windows free commit and physical RAM, retaining at least 3 GiB; the Vulkan device's free-budget number alone is insufficient. Keep a bounded working set and record it identically for both arms. Abort if production reappears during the test.

## Same binary and settings

Use the final experimental DLLs for both arms, the exact existing GGUF, the same official Vulkan plugin and resolved CPU-thread defaults. Baseline unsets both map environment variables. Candidate sets `LLAMA_QSA_PCA_EXPERIMENTAL=1` and the fixed rank64 distilled map whose SHA-256 is recorded in the calibration report. Keep backend precision flags unchanged. This compares the map path with its disabled baseline, rather than conflating a runtime build change with map performance.

The intended separate-server arguments are:

```text
llama-server -m MODEL --host 127.0.0.1 --port 18080 --alias qwen-native-compare
  --device Vulkan0 -ngl 99 -c 1310720 -np 5 -ctk q8_0 -ctv q8_0 -fa on
  -b 2048 -ub 1024 -lm mmap --lazy-mode on --no-repack --fit off
  --cache-ram 0 --ctx-checkpoints 0 --spec-type none --jinja --slots
```

Disable saved-state checkpoints and RAM prompt caches for both arms because the experimental map cache deliberately refuses serialization. Check that the server actually exposes five 262,144-token slots before sending requests. These values allocate 256K per slot; they do not populate five contexts with 256K tokens.

## Request and measurement protocol

[`benchmark_native_concurrency.py`](benchmark_native_concurrency.py) reuses the earlier bounded streaming benchmark method, with a fixed localhost endpoint and strict model/slot/context checks. It never stops or starts services and refuses existing output files. It is prepared and syntax/CLI checked; it has not yet been run against the proposed full-GPU servers.

```powershell
python pocs/lsa-vulkan/benchmark_native_concurrency.py --output NEW_RESULTS.json
```

Use one separate warmup before each arm, then three synchronized bursts per concurrency in this order: `1,2,3,4,5,5,4,3,2,1,1,3,5,2,4`. Each request uses the same public home-library prompt, temperature zero, seed42, thinking disabled, no prompt caching and 256 forced output tokens. Stop on a busy slot or request error. Record API usage, cached tokens, prompt size, time to first content, client stream intervals, server decode rate, end-to-end request rate, burst aggregate rate, actual overlapping streams and errors. Burst aggregate is total completion tokens divided by synchronized release-to-last-completion wall time; dividing it by concurrency is an effective per-user rate, not the individual decode rate.

Report every sample and the matched settings/source/DLL/map hashes. Keep server-reported decode, client streaming and end-to-end throughput separate. Do not label a synthetic indexer reduction as a model speedup. Representative long-context quality, 256K occupancy and prefix/eviction/concurrent-cache validation remain separate requirements even if short-prompt full-GPU throughput improves.
