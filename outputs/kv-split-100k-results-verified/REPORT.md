# KV work splitting: a useful eight-block candidate at 103K

Keeping the normal 32-column MMA tile but splitting its KV work across eight blocks improved this synthetic long-context workload. The candidate averaged **16.698 generation TPS**, versus **14.722 TPS** for its neighboring automatic control blocks: a calculated **13.42% gain**. Sixteen blocks did not help. The original production backend remains unchanged; the eight-block backend is available through an opt-in launcher.

This is evidence for a useful configuration on this hardware and workload, not a global optimum, a sustained 15 TPS floor, or a coding/reasoning/tool-quality guarantee.

## Normal-mode generation measurements

Ryzen AI 7 350 / Radeon 860M, gfx1152, Windows, isolated ROCm 10.0. llama.cpp b11888, commit `48c9f08b7b25766324282b7605177e2f2be755bb`. Model: `C:\Users\kasper\Desktop\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`.

All requests allocated 163840 context and had **103220 actual input tokens**, including 103194 restored cached tokens and 26 newly evaluated tokens. Each block discarded a 64-token warmup, then completed three 256-token stochastic samples and a separate 256-token greedy sample. All 20 measured requests completed without truncation. Normal graph replay was enabled; synchronized operation profiling was disabled.

| Block | KV work blocks | Stochastic mean TPS | Range TPS | Greedy TPS |
|---|---:|---:|---:|---:|
| 1 | automatic: 4 | 14.894 | 14.328-15.880 | 15.615 |
| 2 | double: 8 | **16.698** | **16.056-17.541** | **17.650** |
| 3 | automatic: 4 | 14.550 | 14.193-15.151 | 15.633 |
| 4 | quad: 16 | 13.996 | 13.705-14.477 | 14.863 |
| 5 | automatic: 4 | 14.614 | 14.190-15.176 | 15.655 |

The eight-block result exceeds the mean of blocks 1 and 3 by 13.42%. Sixteen blocks is 4.02% slower than the mean of blocks 3 and 5. Automatic first-to-last block drift was approximately -1.88%. Corresponding automatic trials reproduced their output hashes and MTP proposal/acceptance counts.

Both candidates changed their generated token streams and MTP work, including in greedy mode. These are practical end-to-end measurements, not fixed-token-stream speedups attributable entirely to the attention kernel. Three candidate request averages above 16 TPS do not establish a sustained floor across time, reasoning, code or tool workloads.

Launch configuration: ROCm0, F16 K/V, MTP 2, one slot, GPU layers 99, Flash Attention, Jinja, batch 2048, microbatch 512, eight main/draft/generation/batch threads, auto load mode, log verbosity 3, port 9931. Every block uses the same private experimental DLL, with the override disabled for automatic controls. This comparison does not substitute a differently compiled stock DLL into the control blocks.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, ignore_eos enabled, completed empty thinking block. Greedy uses temperature zero. Primary request SHA256: `a84687703f85aa775ba0a36d59c4fb6072110775967518481bc5f43e87f11283`.

## What changed

The existing MMA tile remains 4 query columns x 8 grouped heads. A private host-side launch override in `fattn-common.cuh` and `fattn.cu` adjusts the existing stream-K grid, including its existing partial-result reduction. Kernel arithmetic source and weight/KV precision were not changed.

The audit records the automatic configuration as four blocks, a resident-block estimate of four, two output tiles, 1616 KV tiles at padded length 103424, eight warps per block, 34816 shared-memory bytes and a KV batch of 64. The experiments choose 2, 8 or 16 blocks by scaling the automatic count, bounded by available KV tiles and rounded to whole output-tile multiples. These are launch values, not measured occupancy or memory-bandwidth counters.

The override applies only on gfx1152 for 32-column MMA with F32 Q `[256,3,16,1]`, F16 K/V with head size 256 and two KV heads, a padded KV length of at least 100000, one sequence, a mask, no sinks, ALiBi or softcap. Other configurations retain automatic launch settings. The affected Q3 shape occurs in target verification and draft-state mirroring; it is not restricted to the target context. Change settings by restarting the server, not by attempting to change a live graph's captured configuration.

## Isolated operation screening

Three perf-mode runs per configuration, with reversed/rotated orders, measured the regular-logit Q3 synthetic fixture. The harness repeats an operation graph for at least one second. These are isolated operation timings, not synchronized profiler rankings or full-model wall-time shares.

| Blocks | Median Q3 time, microseconds |
|---|---:|
| automatic: 4 | 5201.24 |
| half: 2 | 9890.95 |
| double: 8 | 3859.14 |
| quad: 16 | 3825.96 |

Half was excluded from full-model testing after its clear screening regression. Double and quad were tested normally. Quad illustrates why a faster isolated operation does not establish faster overall inference: it did not improve the generation workload.

Perf fixtures are randomly initialized in separate processes, and the long-context Q1 and large-logit Q3 cases are also retained in each raw log. There were no hardware performance counters or exhaustive launch-configuration searches.

## Numerical evidence and limitations

Initial independent float64 NMSE values for regular Q3 at padded KV length 103424 were 0.000130191 (automatic), 0.000266363 (half), 0.000069866 (double), and 0.000034508 (quad). All were below the unchanged 0.0005 numerical gate.

Additional captured fixtures cover contiguous and non-contiguous tensor layouts and 160000 KV length:

| Fixture | Automatic GPU NMSE | Eight blocks GPU NMSE | Sixteen blocks GPU NMSE |
|---|---:|---:|---:|
| Strided 103424 KV | 0.000142004 | 0.000068994 | 0.000035181 |
| Strided 160000 KV | 0.000222403 | 0.000104017 | 0.000052344 |
| Contiguous 160000 KV | 0.000211911 | 0.000106869 | 0.000057805 |

All nine additional cases passed the independent 0.0005 gate, had equal captured GPU/CPU input values, finite outputs, and confirmed the intended split in their audit logs. Each cell uses its own random fixture. These numbers are not paired accuracy rankings, and lower synthetic NMSE does not demonstrate better model quality.

The upstream CPU comparisons still fail for the regular-logit cases; the original backend had the same discrepancy in the earlier experiment. The CPU result is substantially farther from float64 on these fixtures (roughly 0.0022-0.0036 NMSE). Failure logs and exit codes were preserved. The independent reference uses float64 reference energy in its NMSE denominator; the upstream GPU-versus-CPU comparison uses GPU energy. Large-logit Q3 upstream checks passed in the initial three-case probe. The Q1 reference discrepancy remains unresolved, and Q1 is outside this override.

The numerical tests are limited synthetic cases, not exhaustive actual-model-layout coverage. No coding correctness, reasoning, retrieval or tool-call quality benchmark was performed in this comparison. The saved prefix was built using the baseline, and the outputs are forced lengths. There were no randomized end-to-end block orders, controlled power/temperature conditions or confidence intervals. Cold prefill and actual populated 160K inference are separate experiments.

## Use and reproduce

Optional launcher, defaulting to the eight-block candidate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\outputs\Start-KVSplitServer.ps1
```

Use `-KvSplit auto` for automatic control with the same private DLL. This is separate from the original `Start-ROCmServer.ps1`, whose backend and defaults are unchanged. The new launcher retains F16 K/V, MTP 2, context 163840 and the normal batch/thread defaults. No driver or global environment change is needed. The local experimental runtime is ignored by Git.

Use `Stop-KVSplitServer.ps1` to stop this private server while idle; the original stop helper needs an explicit backend directory for it. Follow-up workload, prompt-processing, 128K and 160K evidence is consolidated in [Tuning-round-3-results.md](../Tuning-round-3-results.md). Later larger-context samples include request averages below 15 TPS.

- [Raw samples](samples.json), [summary](summary.json), [protocol and DLL hash](manifest.json), [table](RESULTS.md). Per-case directories contain launch flags, props, restore responses, requests, events, results and logs.
- [Kernel screening](../kv-split-kernel-tests/summary.json), [extended numerical checks](../kv-split-extended-numerics/summary.json), with individual logs and float64 results alongside them. Raw binary fixtures are ignored under `work/kv-split-*`.
- [Patch](../hip-kv-split-build/experiment.patch), [artifact hashes](../hip-kv-split-build/artifact-manifest.json), [sweep runner](../run_kv_split_sweep.py), [numerical screen](../screen_kv_split.py), [extended numerical runner](../check_extended_attention.py), [launcher](../Start-KVSplitServer.ps1).

Run `work/rocm10-runtime/Scripts/python.exe outputs/run_kv_split_sweep.py --out-dir outputs/<fresh-directory>` to repeat the throughput protocol. It requires the local experimental backend, numerical results and saved synthetic slot. An initial attempt in `../kv-split-100k-results/` failed before any inference request because the trimmed runtime lacked application dependencies (Windows 0xc0000135). Those files were restored and this complete comparison rerun separately. The failed startup is excluded.

All 20 measured requests, audit records, graph reuse and final backend hash were checked. The original packaged HIP DLL and prior profile-only overlay were hash-verified unchanged. No commit or publication was performed.
