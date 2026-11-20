# HIP matrix-vector kernel round (November 19, 2026)

Goal: close part of the gap to Vulkan's matrix kernels (see `../vulkan-round/REPORT.md`) with a faster HIP quantized matrix-vector (MMVQ) path for the 1–4 token steps that MTP decoding uses.

## Result

**No production change.** The best variant `c1` was clearly faster for isolated operations: 3-token q5_K expert down −10.7%, dense Q8_0 −2–3%, 1-token q5_K expert down −7%. End-to-end it was indistinguishable from production. Bracketed server runs (sampled, tokens/s) gave 17.57 vs 17.58 at 103K, 15.19 vs 15.21 at 160K and 25.95 vs 26.03 at short context. Pi was restored with the production DLL `51230C04…B4B5`, health ok, `READY`, 3861 tokens restored (`restored-service.json`).

Two related hypotheses were also tested and rejected:

- **Kernel dispatch overhead.** A back-to-back tiny HIP kernel costs 1.2 µs of CPU time and about 0.07 µs of GPU time, and HIP graph replay is no faster.
- **HIP graph capture causing the MTP draft overhead.** Turning graphs off made the server about 1% slower, and `draft()` per cycle went from 23.3 to 24.0 ms at short context.

## What was built

All of this is isolated; production files are unchanged.

- `outputs/build_mmvq_variants.py` copies `mmvq.cu` and adds compile-time overrides to the RDNA2 MMVQ table, which is the one gfx1152 (RDNA 3.5) uses. That table has no tuned entries upstream: every launch uses one warp and one row per block. The knobs are:
  - `NW1`/`RPB1`: 1-token warps and rows;
  - `NW`/`RPB`: multi-token dense warps and rows;
  - `MOE`: rows per block in the multi-token expert kernel (upstream 2).

  The script compiles with the build's exact command, swaps the object into the eight-block build's link, and writes `work/hip-mmvq/<name>/bin` plus `build.json` with hashes. `--async-profiler` also patches `ggml-cuda.cu` with `GGML_HIP_OP_PROFILE=2`, which records unsynchronized event pairs.
- `outputs/build_matvec_probe.py` builds `test-backend-ops.cpp` with a new env-gated case set, `LLAMA_HIP_MATVEC_PROBE`, in the private source tree. It covers the model's exact shapes at 1–4 tokens: dense Q8_0 8192×2048, 2048×4096, 4096×2048, 512×2048 and 2048×512; routed experts (256 experts, 8 used) for q4_K gate/up (k=2048), q5_K/q6_K down (k=512) and the fused q4_K gate+up+SwiGLU.
- `outputs/run_matvec_probe.py` checks each runtime against the CPU backend, then interleaves perf repeats across runtimes.
- `outputs/launch_overhead.hip` is the dispatch micro-benchmark (`launch-overhead.jsonl`).

## Measurements

### Correctness

Every variant passed all 36 probe cases against the CPU backend (normalized MSE ≤ 5e-4), including the fused gate+up path. Logs are in `probe-baseline/`, `probe-screen-1/` and `probe-screen-2/`.

### Isolated operation time, µs (median of 3 repeats, `probe-screen-2`)

| Case | Control | c1 (RPB1=2, RPB=2, MOE=4) | Change |
|---|---:|---:|---:|
| Q8_0 8192×2048, 1 token | 226.2 | 220.4 | −2.6% |
| Q8_0 8192×2048, 3 tokens | 233.1 | 227.8 | −2.3% |
| Q8_0 4096×2048, 3 tokens | 119.2 | 116.5 | −2.3% |
| q4_K expert gate, 3 tokens | 184.2 | 183.9 | 0% |
| q5_K expert down, 1 token | 86.7 | 80.4 | −7.3% |
| q5_K expert down, 3 tokens | 264.6 | 236.2 | −10.7% |
| q6_K expert down, 3 tokens | 285.6 | 267.1 | −6.5% |

The first screen (`probe-screen-1`) tested single knobs. More warps per block was slower everywhere; 4 warps for 1-token k=512 cases tripled their time. Two rows per block helped, and so did four expert rows. In the second screen, 8 expert rows (`c2`) slowed the q4_K gate by 3–4%, and 4 rows for 1-token steps (`c3`) slowed the big 1-token matrix. Summed over the 3-token verify shapes, `c1` should save about 1.7 ms per verify step (calculation). Perf mode times only a test's final node, so the fused gate+up speed could not be measured in isolation.

### In the model

- `llama-bench` at depth 0, alternating control, c1, control, c1 (5 repetitions each): 3-token steps took 67.99, 68.00, 68.34 and 70.25 ms; 1-token steps 45.91, 46.00, 46.52 and 46.76 ms. No gain, and drift upward over time.
- Server, MTP 2, trace logging, order production, c1, production graphs-off, c1, production (`server-cycles/`, `server-cycles/draft-trace.json`):

| Setting | Short sampled | 103K sampled | 160K sampled | 103K repetition | 160K repetition | `draft()`/cycle short |
|---|---:|---:|---:|---:|---:|---:|
| Production (cases 1, 5) | 26.03 | 17.58 | 15.21 | 21.31 | 18.10 | 23.3 ms |
| c1 (cases 2, 4) | 25.95 | 17.57 | 15.19 | 21.42 | 18.21 | 23.2 ms |
| Production, graphs off (case 3) | 25.49 | 17.36 | 15.03 | 21.13 | 18.04 | 24.0 ms |

From case 1 to case 5, production itself drifted down about 1–2%. The c1 cases sit inside that drift.

### Dispatch and profiler overhead (`launch-overhead.jsonl`, 3 runs)

| Case | CPU µs per kernel | GPU µs per kernel |
|---|---:|---:|
| Tiny kernel, back to back | 1.21–1.27 | 0.067–0.068 |
| Tiny kernel, replayed from a 1000-launch graph | 1.19–1.22 | 0.066–0.071 |
| Tiny kernel with an event pair around it | 49–68 | 49–68 |

Event records cost about 25 µs each on this platform. That is why both HIP op profilers inflate small ops: the synchronized one reported 146 ms per 3-token verify step and the unsynchronized one 175 ms, against about 100 ms real (`op-profile-async/`). HIP per-op timings from either profiler are therefore not usable for small kernels.

The same program's 1.6 MB streaming case reported about 115 GB/s. That is above the 89.6 GB/s nominal peak of the two DDR5-5600 64-bit channels, so it is an artifact of that small test and is **discarded**. The validated 1 GiB measurement (81.98 GB/s, `../stream-bandwidth-gfx1152/`) remains the bandwidth reference.

## Interpretation

These are hypotheses.

- The isolated kernels are already close to bandwidth: 66–78 GB/s against an 82 GB/s measured stream. The remaining per-op gains are small, and inside the full graph they are within run-to-run drift. A ~1.7 ms predicted saving per 139 ms cycle would be about 1.2%, too small to resolve against 1–2% drift without many more repeats.
- Vulkan's lead at short context therefore probably does not come from large matvec kernels alone. More likely sources are the many small operations (912 non-matmul op groups per verify graph on HIP), differences in which ops are fused, or graph-level scheduling. Proper attribution needs a hardware-timestamp profiler, which the Windows ROCm SDK used here doesn't ship.
- The 6.7 ms gap in HIP's `draft()` cost against Vulkan at short context remains unexplained. It is not graph capture, and the draft kernels nearly match in op timers.

## Files

- `probe-baseline/`, `probe-screen-1/`, `probe-screen-2/`: probe logs and `summary.json`.
- `bench-d0-1-ctl/` … `bench-d0-4-c1/`: depth-0 `llama-bench`.
- `op-profile-async/`: unsynchronized-event profile logs (diagnostic, inflated).
- `server-cycles/`: server samples, summary, draft trace.
- `launch-overhead.jsonl`: dispatch micro-benchmark.
- `restored-service.json`: Pi restore evidence (`work/hip-mmvq/restore-pi.ps1`).
- Variant DLLs: `work/hip-mmvq/<name>/bin` (ctl, d12, d21, d22, d41, d14, s2, s4, s12, m1, m4, c1, c2, c3, m8, pctl, pc1), each with `build.json`.
