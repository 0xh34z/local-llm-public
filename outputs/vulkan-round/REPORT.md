# Vulkan comparison and MTP draft timing (November 19, 2026)

This follows `outputs/ceiling-round/REPORT.md`. It asks whether the same llama.cpp build is faster on Vulkan, and where HIP's MTP draft overhead comes from.

## Summary

- **Keep HIP for Pi.** Vulkan, using the official b11888 release (same commit `48c9f08b7`, AMD proprietary driver, KHR_coopmat), was faster at short context but much slower at the contexts that matter here. Sampled MTP 2 rates in tokens/s: short 27.88 vs 25.85, 103K 12.37 vs 17.36, 160K 9.84 vs 15.01. Production files and Pi are unchanged (`restored-service.json`: production DLL `51230C04…B4B5`, health ok, `READY`, 3861 tokens restored).
- **Vulkan's matrix kernels show that HIP has headroom.** In Vulkan's own timer, routed-expert and dense matrix work in a 3-token verify step run at about 94–96% of the byte floor. HIP's are slower: 3-token verify at depth 0 takes 68.1 ms on HIP versus 61.7 ms on Vulkan, and 1-token decode 45.8–46.1 versus 42.2 ms.
- **Vulkan's attention is the reason it loses.** Its 3-token flash attention at 103K took 85 ms per verify step, against about 39 ms on HIP (diagnostic timers).
- **HIP's MTP draft overhead is real.** Measured inside the server, `common_speculative_draft()` took 23.8 ms per cycle on HIP and 17.1 ms on Vulkan at short context. The bandwidth floor is 15.6 ms for the two draft passes plus the catch-up pass.
- **HIP graph replay is active but buys nothing.** Turning it off (`GGML_CUDA_DISABLE_GRAPHS=1`) changed depth-0 decode by 0.1–0.5 ms, within noise. Kernel launch overhead is not the bottleneck; kernel efficiency is.

## Conditions

These match the ceiling round: F16 K/V, flash attention, batch 2048, microbatch 512, eight threads, one slot, context 163840, Balanced plan, on AC. The HIP runs use `work/hip-kv-split-bin` (eight-block split). The Vulkan runs use `work/vulkan-bin`, the user-downloaded official Windows Vulkan x64 release b11888; `ggml-vulkan.dll` SHA-256 is `49C8F389…D247`. Same model file. The scripts take a backend switch: `run_depth_bench.ps1 -Runtime vulkan`, and `run_ceiling_server.py --order` with entries like `hip2`, `vk2`, `vk0` and a `t` suffix for trace logging. The launcher is `Start-CeilingServer.ps1 -Device Vulkan0`, and `Stop-CeilingServer.ps1` stops either device only while idle.

## Measurements

### `llama-bench` per step, ms (5 repetitions; HIP values from the ceiling round)

| Depth | tg HIP | tg Vulkan | pp2 HIP | pp2 Vulkan | pp3 HIP | pp3 Vulkan | pp4 HIP | pp4 Vulkan |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 46.1 | 42.2 | 59.1 | 52.7 | 68.4 | 61.7 | 77.5 | 70.2 |
| 32768 | 54.8 | 53.9 | 70.6 | 69.7 | 82.2 | 87.5 | | 100.0 |
| 103194 | 73.9 | 77.8 | 89.9 | 108.0 | 100.2 | 147.1 | 128.0 | 167.6 |
| 160511 | 87.8 | 97.2 | 120.4 | 141.4 | 123.8 | 193.1 | 159.5 | 222.2 |

Vulkan source: `depth-bench/`; its smoke run is in `depth-bench-smoke/`. A HIP graphs-on control at depth 0 measured 45.79 (tg), 58.40 (pp2) and 67.29 (pp3) ms. With graphs off it measured 45.91, 58.73 and 67.81 ms (`hip-graphs-on-d0/`, `hip-graphs-off-d0/`).

### Server decode, tokens/s (ms per cycle)

Two runs: `server-cycles/` with order HIP 2, Vulkan 2 (it then stopped because the production stop helper ignores Vulkan servers; fixed), and `server-cycles-2/` with order Vulkan 2, Vulkan 0, HIP 2, HIP 2 trace, Vulkan 2 trace. The workloads match the ceiling round. Pooled:

| Setting | Short sampled | Short repetition | 103K sampled | 103K repetition | 160K sampled | 160K repetition |
|---|---:|---:|---:|---:|---:|---:|
| HIP MTP 2 | 25.85 (94.9) | 30.97 | 17.36 (139.1) | 21.15 | 15.01 (163.2) | 17.94 |
| Vulkan MTP 2 | 27.88 (87.2) | 33.58 | 12.37 (189.0) | 15.60 | 9.84 (245.4) | 12.05 |
| Vulkan MTP 0 | 22.82 (43.8) | 22.83 | 12.71 (78.7) | 12.75 | 10.20 (98.0) | 10.23 |
| HIP MTP 0 (ceiling round) | 21.65 (46.2) | 21.62 | 13.80 (72.5) | 13.82 | 11.47 (87.2) | 11.49 |

Trace logging (`-lv 4`) did not change speed: HIP trace and normal runs agree within 1%, and so do the two Vulkan runs. Vulkan server overhead without speculation is 0.7–1.7 ms per token over `llama-bench`.

### MTP `draft()` time per verify cycle, ms

This comes from llama.cpp's own `dur(b,g,a)` counters, which are printed at trace level and parsed by `outputs/analyze_draft_trace.py` (`server-cycles-2/draft-trace.json`). It covers the two single-token MTP passes, their GPU top-k sampling, and the wait for the catch-up pass queued just before.

| Context | HIP | Vulkan | Floor (2 drafts + catch-up) |
|---|---:|---:|---:|
| Short | 23.8 | 17.1 | 15.6 |
| 103K | 32.7 | 32.1 | 23.4 |
| 160K | 37.7 | 40.5 | 27.7 |

Server cycle minus `llama-bench` verify left 26.4 / 39.3 / 39.6 ms on HIP (ceiling round). `draft()` accounts for 23.8 / 32.7 / 37.7 ms of that. The remaining 2–7 ms is the catch-up launch, sampling of the verify step and server bookkeeping, plus differences between `llama-bench` and server verify.

### Diagnostic op timings at the restored 103K prefix

`outputs/run_op_profile.py` ran one 48-token MTP 2 continuation per backend, and `outputs/analyze_op_profile.py` summarized it (`op-profile-103k/op-summary.json`). The HIP profiler synchronizes each fused op group and disables graph replay: it ran at 4.3 tokens/s and inflated a verify step to 145.6 ms, against about 100 ms real. It also dropped 164 negative samples. The Vulkan perf logger uses timestamp queries: it ran at 12.0 tokens/s (unprofiled 12.4), and its 3-token verify total of 151 ms matches the 147 ms `llama-bench` step. Times are ms per graph.

| Graph | Backend | Attention | Routed experts | Output head | Dense matmul | DeltaNet | Other | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Verify, 3 tokens | HIP | 39.0 | 35.3 | 7.0 | 36.8 | 6.2 | 21.2 | 145.6 |
| Verify, 3 tokens | Vulkan | 85.1 | 23.3 | 6.4 | 20.3 | 4.3 | 11.8 | 151.2 |
| MTP draft, 1 token | HIP | 2.7 | 0.3 | 7.2 | 0.9 | | 0.4 | 11.4 |
| MTP draft, 1 token | Vulkan | 3.5 | 0.2 | 6.4 | 0.7 | | 0.3 | 11.1 |
| MTP catch-up, 3 tokens | HIP | 3.6 | 0.7 | | 0.9 | | 0.4 | 5.6 |
| MTP catch-up, 3 tokens | Vulkan | 8.5 | 0.6 | | 0.7 | | 0.3 | 10.1 |

## Calculations

- Routed experts for a 3-token step read about 23.3 × 40 layers' worth of expert slices, 1.79 GB, a floor of 21.8 ms at 81.98 GB/s. Vulkan's 23.3 ms is about 94% of the floor. HIP's profiled 35.3 ms includes per-group sync gaps. If those gaps are about 45 µs per group, as the 45 ms gap between total profiled and real time suggests, its real time is roughly 30 ms (about 72%).
- Non-routed layer weights are 1.597 GB, a floor of 19.5 ms. Vulkan's dense class (20.3 ms) is about 96% of that.
- **Hypothetical HIP gain.** If HIP's verify matmuls matched Vulkan's depth-0 speed (−6.7 ms per verify) and its `draft()` matched Vulkan's short-context cost (−6.7 ms), keeping HIP's attention, an MTP 2 cycle would fall from 139.1 to about 125.7 ms at 103K and from 163.2 to about 149.8 ms at 160K. That is about 19.3 and 16.4 tokens/s at measured acceptance, roughly +11% and +9%. This is an estimate; no such kernel exists.

## Hypotheses (not tested)

- **Routed-expert kernel.** The HIP gap is mainly in the multi-token routed-expert matrix-vector path (`mul_mat_vec_q` with ids, one warp per token on RDNA3) and in the many small Q8_0 dense projections. Vulkan's `MUL_MAT_ID_VEC` and `MUL_MAT_VEC` shaders on this GPU are the existence proof.
- **Draft passes.** HIP's extra `draft()` time at short context is about 6.7 ms per cycle, but the two MTP draft graphs differ by only about 0.3 ms each in the diagnostic op timers. So most of the gap is probably outside the kernels: host work, synchronization, or graph handling of the small MTP graphs. Host-side timestamps around `llama_process` in an isolated build would test this.
- **Mixing backends.** Running attention on HIP and matrix work on Vulkan in one process isn't practical. llama.cpp schedules by tensor and layer, not by op type, and would copy activations between two GPU APIs for every layer.

## Linux

This round strengthens the earlier conclusion. On Windows both backends' server overhead is small, and the gaps are in specific kernels. The Linux Mesa RADV Vulkan driver may compile the same shaders differently from AMD's Windows driver, and Linux ROCm runs the same HIP kernel sources. That makes Linux a possible way to test RADV attention speed, not an expected general gain.

## Files

- `depth-bench-smoke/`, `depth-bench/`: Vulkan `llama-bench`.
- `hip-graphs-on-d0/`, `hip-graphs-off-d0/`: HIP graph-replay check.
- `server-cycles/`, `server-cycles-2/`: server cases, `samples.json`, `summary.json`; `server-cycles-2/draft-trace.json`.
- `op-profile-103k/`: per-backend logs, profile summaries and `op-summary.json` (diagnostic only).
- `analysis.json`: `outputs/analyze_ceiling.py` over the HIP ceiling-round benches, the HIP graphs-on control, the Vulkan bench and both server runs.
- `restored-service.json`: Pi restore evidence (`work/vulkan-round/restore-pi.ps1`).
