# Attention workgroup and query-storage round (November 19, 2026)

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

Two private changes to the long-context Q3/D256/F16 attention kernel on gfx1152 were tested against the current eight-block configuration. Both were numerically acceptable and both were slower. The production eight-block launcher and `work/hip-kv-split-bin/ggml-hip.dll` (SHA-256 `51230c04…b4b5`) are unchanged. [Protocol](Protocol.md) · [Build](Build.md) · [Compiler resource notes](Source-notes.md)

## Measurements

### 1. 128 versus 256 threads per workgroup

The private DLL is `work/hip-workgroup/bin/ggml-hip.dll`, SHA-256 `fe56b184…886f`. Both settings ran with 32 columns, eight stream-K blocks, F16 K/V and the same reduction. Logs confirmed `warps=4` (128 threads, 34,368 shared bytes) and `warps=8` (256 threads, 34,816 shared bytes).

**Numerical check.** All 8 contiguous/strided fixtures at 103K and 160K passed the independent float64 reference (threshold NMSE 0.0005). Candidate GPU NMSE was 1.34e-4 to 2.05e-4, versus 6.5e-5 to 1.11e-4 for the control. The upstream CPU comparison failed for both, as in earlier rounds. [Screen summary](../workgroup-screen/summary.json)

**Isolated operator timing** (strided fixtures, nine positions interleaved 256/128, medians):

| Run | 103K, 256 | 103K, 128 | 160K, 256 | 160K, 128 |
|---|---:|---:|---:|---:|
| First screen | 3,544 µs | 5,290 µs (+49.3%) | 5,536 µs | 8,243 µs (+48.9%) |
| Clean repeat after generation | 3,759 µs | 5,550 µs (+47.6%) | 5,736 µs | 8,435 µs (+47.1%) |

[First screen](../workgroup-screen/summary.json) · [Clean repeat](../workgroup-perf-clean/summary.json)

**Normal inference.** Restored identical synthetic prefixes (103,194 and 160,511 cached tokens), with operation profiling off and the usual flags. Order: 256, 128, 256. Rates are pooled generation tokens/s.

| Context, workload | 256 (both controls) | Slowest 256 | 128 | Slowest 128 | Change |
|---|---:|---:|---:|---:|---:|
| 103K sampled (2 requests/case) | 16.96 | 16.49 | 14.86 | 14.18 | −12.4% |
| 160K sampled (2 requests/case) | 14.59 | 14.00 | 11.73 | 11.50 | −19.6% |
| 103K repetition (128 tokens) | 20.48 | 20.41 | 17.40 | 17.40 | −15.0% |
| 160K repetition (128 tokens) | 17.34 | 17.27 | 14.44 | 14.44 | −16.8% |

Sampled outputs diverged after a short shared prefix, so draft work differs between cases. In the matched-token prefixes alone (7–72 tokens) the candidate was also slower in each comparison. [Comparison](../workgroup-generation/comparison.json) · [Samples](../workgroup-generation/samples.json) · [Dispatch markers](../workgroup-generation/case-02-128/dispatch.json)

**GPU sensors during generation.** Median GPU clock was 2,590–2,806 MHz for the candidate and 2,487–2,848 MHz for the controls, with power 24–29 W and temperature 68–74 °C in all cases. The candidate's clocks were not lower than the controls'. [Aligned sensor summary](generation-sensors.json) · [Raw ADLX samples](gpu-sensors.csv)

### 2. Query in shared memory (LDS) versus registers, both at 256 threads

The private DLL is `work/hip-query-lds/bin/ggml-hip.dll`, SHA-256 `9bcb045e…d140`. The LDS path uses `Q_in_reg=false`, keeping 256 threads, 32 columns and eight blocks. Logs confirmed `shared=51264` for LDS and `shared=34816` for registers.

**Numerical check.** All 8 fixtures passed the float64 reference. LDS GPU NMSE was 6.9e-5 to 1.11e-4, the same range as the register control.

**Isolated operator timing** (medians of interleaved runs):

| Fixture | Registers | LDS | Change |
|---|---:|---:|---:|
| Strided 103K | 3,732 µs | 4,298 µs | +15.2% |
| Strided 160K | 5,755 µs | 6,607 µs | +14.8% |

Every LDS sample was slower than every register sample at the same length. The pre-set rule ran generation only if LDS was within 5% at either length, so this candidate stopped at the operator screen. No end-to-end generation was measured for it. [Screen summary](../query-storage-screen/summary.json)

## Calculations

- The previous round's slowest 160K sampled code continuation needed about 4.1 ms/token of improvement to reach 15 tokens/s. Neither candidate provides it: both add attention time per call.
- The 128-thread kernel's ~47–49% longer kernel time is consistent with the end-to-end loss of 12–20%, because attention is only part of each decode step.

## Interpretation (hypotheses, not measured causes)

- Compiler descriptors show 478 VGPR spills and 1,312 private bytes per lane at 128 threads, versus 169 spills and 416 bytes at 256. The resulting scratch traffic, plus halving the waves per column, plausibly explains the 128-thread loss. Hardware counters were not collected.
- LDS query storage cut reported spills from 169 to 13, but was 15% slower. Here the extra shared-memory loads and the larger per-block allocation appear to cost more than the register spills they avoid. Occupancy was not measured directly.

## Decision and state

- Keep 256 threads, register-held Q, 32 columns and the eight-block launcher (`outputs/Start-KVSplitServer.ps1`). Neither experiment is a production candidate.
- The experimental DLLs remain isolated in `work/hip-workgroup/bin` and `work/hip-query-lds/bin`. No production file was modified.
- After the round, Pi was restored on the production DLL. Health was ok, the Pi CLI smoke test replied `READY`, and the 3,861-token snapshot `before-workgroup-20261119-0016` was restored. [Restored-service record](restored-service.json)
- The Windows power plan was Balanced at the start and end, on AC with the battery charging.
