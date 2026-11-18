# First private HIP diagnostic build and 100K-context profile

The diagnostic build works on this Radeon 860M. It loads and restores the existing 103K-token slot, and both normal and timed modes reproduce the known-good 64-token continuation exactly. The working inference backend was not replaced.

This makes a focused kernel experiment technically practical. Flash Attention is the largest individual operation shape in the target-like context. The next useful experiment is an isolated, correctness-checked comparison of the existing attention implementations for the measured shape, followed by an uninstrumented end-to-end comparison. A new kernel implementation has not been written or installed, and no production speedup is claimed.

## Build

- Source: llama.cpp b11888, exact commit `48c9f08b7b25766324282b7605177e2f2be755bb`.
- Local source checkout: `work/llama.cpp`; an 88-line private diagnostic diff changes only `ggml/src/ggml-cuda/ggml-cuda.cu`.
- Separate build environment: `work/rocm10-build-tools`, official AMD ROCm 10.0.0 core, libraries, devel and gfx1152 packages. `rocm-sdk init` expanded its local development tree.
- Compiler: AMD clang 23.0.0git, LLVM revision `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, Windows MSVC target; Visual Studio 2022 Build Tools and Ninja.
- Release build, HIP graphs compiled in, GPU target `gfx1152`, dynamic backend loading, CPU backend compilation disabled. The application and CPU backend DLLs in the isolated server copy are the original b11888 binaries.
- Output: `work/hip-diagnostic-build/bin/ggml-hip.dll`, 72023552 bytes. It was copied into `work/hip-diagnostic-bin`, alongside copies of the normal application/runtime DLLs. The original backend directory remains unchanged.

The first compile needed explicit HIP/device-library paths for the AMD wheel layout. The successful build used `--hip-path` and `--rocm-device-lib-path` pointing inside the development environment. Compiler warnings and the failed first attempt are retained in `../hip-diagnostic-build/`. The successful build completed all 154 steps and linked the HIP DLL.

The [configuration and DLL hashes](../hip-diagnostic-build/artifact-manifest.json), [build arguments](../hip-diagnostic-build/configuration.json), [build script](../Build-HIPDiagnostic.ps1) and [source patch](../hip-op-profile.patch) are retained. Upstream references: [release workflow](https://github.com/ggml-org/llama.cpp/blob/48c9f08b7/.github/workflows/release.yml), [Windows ROCm setup](https://github.com/ggml-org/llama.cpp/blob/48c9f08b7/.github/actions/windows-setup-rocm/action.yml), [HIP CMake configuration](https://github.com/ggml-org/llama.cpp/blob/48c9f08b7/ggml/src/ggml-hip/CMakeLists.txt).

## Runtime checks

The isolated backend enumerated ROCm0, Radeon 860M, gfx1152. The application uses the normal 163840 context allocation, one slot, GPU layers 99, Flash Attention, F16 K/V, MTP 2, batch 2048, microbatch 512, eight main/draft/batch CPU workers and auto load mode.

| Check | Normal diagnostic backend | Operation-timing mode |
|---|---:|---:|
| Input tokens | 103220 | 103220 |
| Reused tokens | 103194 | 103194 |
| New prompt tokens processed | 26 | 26 |
| Generated tokens | 64 | 64 |
| Draft tokens proposed / accepted | 56 / 34 | 56 / 34 |
| Output matches known-good continuation | yes | yes |
| Truncation | false | false |
| Prompt evaluation seconds | 0.919 | 0.910 |
| Generation tokens/s | 14.140 | 4.060 |
| Client request wall seconds | 5.483 | 16.489 |

The normal mode emitted zero operation-profile records. Timed mode is selected only with `GGML_HIP_OP_PROFILE=1`. It disables graph replay and concurrent-region scheduling, records HIP events around each ordinary fused operation group, and synchronizes after each group. This intentionally disrupts execution and causes substantial overhead. The normal-mode number is a single short validation run, not evidence that the custom build improves performance.

Requests match exactly, use the same saved prefix and sampling as the cache-restoration probe, and bypass thinking with a completed empty thinking block. Matching 64 output token IDs is a limited state-integrity check, not a coding/reasoning/tool-use quality evaluation.

## Measured operation candidates

43,161 operation-group records were collected. Only graphs containing full attention with KV length at least 100000 and no more than three query tokens were sampled. Their padded KV length was 103424. There were 29 graphs in a GDN-present backend context and 85 in a separate GDN-absent context. These are target-like and draft-like families by inference. Draft-like graphs include state-mirroring work as well as draft-token proposals; graph counts are not token counts.

After excluding invalid negative event durations, the provisional target-like GPU totals are:

| Category | Share of selected target-like GPU event time |
|---|---:|
| Dense/vocabulary matrix operations | 29.0% |
| Attention | 28.5% |
| Other groups | 21.1% |
| Expert/shared groups | 18.2% |
| Recurrent groups | 3.2% |

The single largest target-like shape was Flash Attention with Q `[256,3,16,1]` and K `[256,103424,2,1]`: 280 calls averaged 4.991 ms of event time and 5.065 ms of synchronized host wall time per call, with no negative samples. Source inspection points to the MMA/WMMA family for this main verification shape under the current dispatch conditions; the profile does not itself record individual GPU kernel symbols. That dispatch inference should be explicitly checked in an isolated kernel experiment.

The draft-like context's largest shape was a Q8_0 vocabulary projection with weights `[2048,248320,1,1]` and input `[2048,1,1,1]`: 56 calls averaged 6.828 ms of event time, with no negative samples. It accounted for about 42.7% of that context's nonnegative event time. Rewriting it cannot be assumed to improve bandwidth or compute utilization without an isolated benchmark or hardware counters.

Detailed shapes, valid-sample counts, host wall times, categories and per-context totals are in [OPERATION-TIMINGS.md](OPERATION-TIMINGS.md) and [timing-summary.json](timing-summary.json).

## Timer quality and interpretation

2,908 of 43,161 event samples (6.74%) were invalid negative durations, with a minimum of -0.429170 ms. Their absolute sum was 33.648 ms. The cause has not been established. The raw signed values remain in the local log; the analyzer excludes them from GPU sums and means and reports invalid counts for each shape. Their absolute sum does not bound the error in other samples. The largest attention and vocabulary-projection groups contained no negative samples, but that alone does not prove precise absolute timing.

These data support provisional candidate selection. They are not a production wall-time decomposition or hardware-counter trace. Synchronization, disabled graphs, disabled concurrency, logging, clock behavior, event-timer anomalies and omitted graphs/CPU work limit interpretation. Do not use the table as an Amdahl speed prediction. GPU percentages from separate contexts must not be added as if each were a fraction of the whole request.

There was one normal-mode validation and one timed request, without temperature/power controls or repeated profiling trials. No fresh 100K prefill, populated 160K run, register-pressure report, occupancy measurement, memory-bandwidth measurement or isolated attention-kernel correctness/performance test was performed. The earlier Vulkan timing proportions must not be substituted for these HIP results.

## Reproduction

The isolated development environment can be recreated without changing the inference runtime:

```powershell
.\work\rocm10-runtime\Scripts\python.exe -m venv .\work\rocm10-build-tools
.\work\rocm10-build-tools\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1152]==10.0.0"
.\work\rocm10-build-tools\Scripts\rocm-sdk.exe init
```

Use the exact source commit and apply `outputs/hip-op-profile.patch`. `outputs/Build-HIPDiagnostic.ps1` configures and builds the backend. `outputs/Start-HIPDiagnostic.ps1` runs the existing isolated overlay in normal mode, or with `-Profile` for diagnostic timing. It restores its inherited profile flag when it exits. This wrapper expects the normal application DLL copies and newly built HIP DLL in `work/hip-diagnostic-bin`; the normal production launcher does not select that directory by default.

`probe_slot_cache.py restore` performs the saved-prefix continuation, and `analyze_hip_op_profile.py` produces the operation summary. Every new test needs a fresh output directory. The raw request/result files, launch command and validation are saved with this report. The private patch is for local diagnostics; it is not an upstream contribution.

The diagnostic inference server was shut down after collection. The production DLL hash was checked after the tests, and no driver, system settings, model weights or Pi conversation were changed.
