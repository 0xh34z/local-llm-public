# MMA tile-column experiment: keep automatic selection

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

Two further MMA configurations were built and tested on the Radeon 860M (gfx1152). Neither improved practical generation at the restored 103K prefix. Keep the normal ROCm backend, F16 K/V, MTP 2 and automatic attention dispatch. No experimental backend was installed as the production default.

## Measured results

Each block discarded a 64-token warmup, then completed three 256-token stochastic continuations and one separate 256-token greedy continuation. Order was automatic, 16 columns, automatic, 64 columns, automatic. All 20 measured requests completed.

| Block | MMA columns | Stochastic mean TPS | Stochastic range TPS | Greedy TPS |
|---|---|---:|---:|---:|
| 1 | automatic (32) | 14.625 | 14.103-15.304 | 15.654 |
| 2 | 16 | 10.732 | 10.517-10.848 | 11.879 |
| 3 | automatic (32) | 14.525 | 14.194-14.972 | 15.606 |
| 4 | 64 | 13.585 | 12.792-15.028 | 13.982 |
| 5 | automatic (32) | 14.545 | 14.094-15.148 | 15.602 |

The 16-column candidate was 26.37% slower than the mean of its two neighboring automatic blocks. The 64-column candidate was 6.54% slower than its neighboring controls. These percentages are calculations from the measured block means, not isolated kernel speed measurements.

The automatic controls averaged 14.565 TPS over nine stochastic requests. Their first-to-last mean drift was -0.545%; the spread between their highest and lowest block means was approximately 0.69%. Corresponding automatic trials reproduced their token hashes and MTP draft/acceptance counts. Both candidate configurations changed generated tokens and draft work, including in the greedy tests. Thus the candidate comparisons include changed model trajectories and are not comparisons of identical token streams. They provide no evidence for a practical gain; they do not establish a fixed-work kernel slowdown of those exact percentages.

The 15 TPS sustained floor remains unproven. These results do not establish that all possible kernel tuning is exhausted.

## Exact workload and implementation

Base source: llama.cpp b11888, commit `48c9f08b7b25766324282b7605177e2f2be755bb`. ROCm 10 runtime, private gfx1152-only build, graph replay enabled, synchronized operation profiling disabled.

Model: `$env:LLM_MODEL_PATH (e.g. ~\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf)`. Every request had 103220 input tokens, restored 103194 cached tokens, and processed 26 additional tokens. Allocation was 163840; output budget was 256. Configuration: ROCm0, F16 K/V, MTP 2, one slot, GPU layers 99, Flash Attention, Jinja, batch 2048, microbatch 512, eight main/draft/generation/batch threads, auto load mode, port 9931. Launch arguments and server properties are retained in each case directory.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, ignore_eos enabled, completed empty thinking block. Greedy controls use temperature zero. Primary request SHA256 is `a84687703f85aa775ba0a36d59c4fb6072110775967518481bc5f43e87f11283`.

The change selects existing compiled MMA template configurations in `ggml/src/ggml-cuda/fattn.cu`: query columns x grouped heads are 2x8 for 16 columns, 4x8 for the normal 32 columns, and 8x8 for 64 columns. This changes tiling within the MMA family; it does not introduce a new attention algorithm or change the kernel arithmetic source.

Override scope: gfx1152, F32 Q `[256,3,16,1]`, F16 K/V with head size 256 and two KV heads, padded KV length at least 100000, one sequence, mask present, no sinks, ALiBi or softcap, and the same stride-alignment requirements as normal grouped-head dispatch. Other query counts, shapes and prompt processing retain their normal settings. The affected shape can occur in target verification and draft-state mirroring. Logs confirm MMA family 400 and actual column selection for every block. Graph-reuse records are present and operation-profile records are absent.

## Numerical checks and limits

Both candidates were tested with the upstream attention probe: regular-logit Q1 and Q3 cases plus a large-logit Q3 case. The final binary passed the large-logit case but failed both regular-logit comparisons against the upstream CPU path, as did the original backend in the prior experiment. Failures and nonzero exits were preserved, not suppressed or addressed by increasing the tolerance.

Independent NumPy float64 checks of captured regular-logit Q3 fixtures gave:

| Final binary candidate | GPU NMSE versus float64 | CPU NMSE versus float64 | GPU <= 0.0005 |
|---|---:|---:|---|
| 16 columns | 0.000173770 | 0.002281116 | yes |
| 64 columns | 0.000259529 | 0.002257868 | yes |

Each candidate has its own random fixture. The rows are not paired accuracy rankings. Captured GPU and CPU Q/K/V/mask inputs were equal. The independent implementation uses float64 reference energy in its NMSE denominator; the upstream GPU-versus-CPU comparison uses GPU energy. The CPU-reference discrepancy and the Q1 comparison remain unresolved. These limited fixtures do not establish coding, reasoning, retrieval or tool-call quality. Probe Q is contiguous; actual model layouts are not exhaustively covered.

No fresh cold-prefill, populated 160K, real Pi tool cycle, power/temperature control, randomized order, confidence interval or isolated kernel microbenchmark was added. The saved prefix was created with the baseline backend. Candidate runs therefore evaluate continuations from that state, not a conversation populated entirely with the candidate.

## Evidence and reproduction

- [Raw measured samples](samples.json), [summary](summary.json), [protocol and final DLL hash](manifest.json), [generated table](RESULTS.md). Case directories contain requests, streamed events, results, restore responses, launch flags, properties and server logs.
- [Numerical logs and float64 results](../mma-tile-kernel-tests/), including `16-final-float64.json` and `64-final-float64.json`. Raw fixtures remain ignored under `work/mma-tile-*-float64-final`.
- [Final dispatch patch](../hip-mma-tile-build/dispatch-final.patch), [final artifact hashes](../hip-mma-tile-build/artifact-manifest-final.json), and final compile log in `../hip-mma-tile-build-audit/`.
- [Sweep runner](../run_mma_tile_sweep.py), [analyzer](../analyze_mma_tile_sweep.py), [experiment launcher](../Start-AttentionExperiment.ps1), [numerical wrapper](../Invoke-AttentionProbe.ps1).

The first automatic-only attempt is preserved separately in `../mma-tile-100k-results/`. Its audit message used the filtered GGML info logger and was absent from the server log. The controller stopped before any candidate run. The final build writes that one-time marker directly to stderr, and the entire five-block comparison above was rerun in this new directory. The earlier attempt is excluded from all reported means.

To reproduce the final throughput protocol, use `work/rocm10-runtime/Scripts/python.exe outputs/run_mma_tile_sweep.py --out-dir outputs/<fresh-directory>`. It requires the local experimental runtime, saved synthetic slot and final numerical results; it refuses an existing manifest or an already running server on the test port. The numerical wrapper retains the known upstream comparison failure exit, so interpret it with the independent float64 evidence.

The original packaged HIP DLL and prior profiling overlay were hash-verified unchanged. The experiment used only `work/hip-mma-tile-bin`. The inference server was stopped after the final idle request. No driver, global environment, Pi configuration, model weights, commits or published artifacts were changed.
