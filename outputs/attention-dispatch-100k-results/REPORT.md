# Attention dispatch experiment: keep automatic MMA selection

The isolated gfx1152 experiment built and ran successfully, but no alternative qualified as an improvement. Keep the working backend and automatic attention selection. No experimental choice was installed as the production default.

- Automatic selection chose MMA for the measured Q3 shape and averaged **15.073 TPS** across six normal-mode stochastic continuations in this experiment.
- Forcing the vector implementation for that shape passed the independent numerical check but measured **1.925 TPS** on its first complete continuation. Further vector repetitions were stopped because of the severe regression.
- Forcing the tile implementation exceeded the independent numerical tolerance and was excluded from full-model benchmarking.

This is a dispatch experiment using existing implementations, not a newly written attention algorithm. The next viable kernel research target would be the existing MMA implementation's tiling, reduction or work distribution on gfx1152, using an isolated benchmark and an independent correctness reference. This experiment provides no evidence that such work would produce a particular speedup.

## Narrow scope of the change

Source: llama.cpp b11888, commit `48c9f08b7b25766324282b7605177e2f2be755bb`. The private change is in `ggml/src/ggml-cuda/fattn.cu`, with separate opt-in test cases and tensor export in `tests/test-backend-ops.cpp`.

`GGML_HIP_FA_EXPERIMENT=auto|tile|vec|mma` changes dispatch only on gfx1152 when Q is F32 with shape `[256,3,16,1]`, K/V are F16 with head size 256 and two KV heads, padded KV length is at least 100000, batch is one, and there is a mask with no sinks, ALiBi or softcap. Other query counts, short contexts, prompt batches and other shapes retain normal dispatch. The same shape can occur in both target verification and draft-state mirroring, so the override is not restricted to only the target context.

The experiment uses `work/hip-attention-experiment-bin`; the original backend and previous diagnostic overlay were hash-checked and remain unchanged. Graph replay is enabled and `GGML_HIP_OP_PROFILE=0`. Logs confirmed automatic mode selected kernel 400 (MMA) and vector mode selected kernel 100. No synchronized operation-profile records appeared in the full-model benchmark.

## Numerical checks and the reference discrepancy

The upstream `test-backend-ops.cpp` harness was compiled with opt-in cases for the measured head/GQA/KV shape. It checks query counts one and three with regular random logits, plus query count three with larger Q values (`q_range=20`). K/V are F16 views of buffers with twice the logical KV length. The query tensors in these tests are contiguous; this is not exhaustive coverage of actual model layouts.

The unchanged packaged backend, the new automatic mode, and the tile/vector/forced-MMA modes all passed the large-logit case but failed the regular-logit CPU comparisons at the existing NMSE threshold of 0.0005. Because the unchanged packaged backend showed the same pattern, this was investigated rather than attributed to the new selector.

Captured Q, K, V and mask tensors were confirmed equal between the GPU and CPU comparison paths. An independent NumPy float64 calculation then evaluated scaled QK, additive mask, stable softmax and the V reduction. The GPU's automatic result was much closer to float64 than the CPU comparison result on the captured Q3 case.

| Mode | GPU NMSE versus float64 | CPU comparison NMSE versus float64 | GPU within 0.0005 |
|---|---:|---:|---|
| Automatic MMA | 0.000132896 | 0.002119481 | yes |
| Forced tile | 0.001113490 | 0.002217462 | **no** |
| Forced vector | 0.000032818 | 0.002182676 | yes |
| Forced MMA | 0.000139498 | 0.002279803 | yes |

Each row is a separate randomly initialized fixture; numerical rankings between rows are not a controlled paired accuracy comparison. The tolerance was not increased. The float64 test uses float64 reference energy in the NMSE denominator, while the upstream GPU-versus-CPU calculation uses GPU output energy. These are explicitly different reference comparisons with the same threshold.

The CPU comparison path is not accurate enough to serve as the gold reference for these captured regular-logit Q3 fixtures. Its underlying cause was not established. The Q1 CPU discrepancy was not independently resolved in float64. This does not demonstrate a coding, reasoning or retrieval failure in the model, and no general quality guarantee follows from these few synthetic tensors.

The tile mode was rejected on its independent numerical error. Vector and automatic MMA qualified for a bounded throughput experiment. Large-logit upstream comparisons passed for both, but the float64 captures cover only the regular-logit Q3 case per mode. No full-model quality benchmark was performed.

Numerical evidence is in `../attention-kernel-tests/`, including packaged-backend comparisons, alternative logs and `*-float64.json`. Raw exported fixtures remain local under `work/attention-float64-*` and are ignored by Git. The independent implementation is [check_attention_float64.py](../check_attention_float64.py).

## Normal-mode generation at a restored 103K prefix

All requests allocated 163840 context, used the same restored 103194-token prefix, processed 26 new tokens, and generated 256 tokens without truncation. Configuration: ROCm0, Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL, F16 K/V, MTP 2, one slot, GPU layers 99, Flash Attention, batch 2048, microbatch 512, eight main/draft/batch threads, auto load mode and log verbosity 3.

Primary sampling is temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, completed empty thinking blocks and forced output length. All primary request JSON hashes match: `a84687703f85aa775ba0a36d59c4fb6072110775967518481bc5f43e87f11283`.

| Block | Completed primary samples | Mean TPS | Range TPS |
|---|---:|---:|---:|
| Automatic MMA, first | 3 | 15.329 | 14.919-15.991 |
| Forced vector | 1 | 1.925 | one sample |
| Automatic MMA, final | 3 | 14.818 | 14.363-15.522 |
| Automatic MMA, pooled | 6 | 15.073 | 14.363-15.991 |

The vector request spent 132.455 seconds in generation. It proposed 234 draft tokens and accepted 138; the first automatic control proposed 212 and accepted 149. The vector output tokens differ from the automatic output, so the end-to-end comparison includes changed generated content and draft work. It is not a fixed-token-stream kernel microbenchmark. The observed task performance is nevertheless unsuitable for the user's 15 TPS target.

The first and final automatic blocks reproduced corresponding output sequences and draft counts. Their mean speed drifted -3.335%, which also shows why these controls should not be compared directly with older 14 TPS runs. Primary automatic samples still dipped below 15 TPS; a dependable 15 TPS floor remains unproven.

Both automatic blocks also ran one 256-token greedy control. These measured 15.924 and 15.798 TPS with matching output token IDs and draft counts (206 proposed / 152 accepted). No vector greedy control was completed after its regression was established.

## Adaptive stopping and limits

The original plan called for automatic/vector/automatic, with a 64-token warmup, three stochastic samples and one greedy sample per block. After the vector warmup and first complete sample showed severe regression, remaining vector repeats were cancelled. The benchmark controller was stopped, the isolated server shut down, and the final automatic block resumed explicitly. Any unfinished requests are excluded. `early-stop.json` records this deviation; the reproducible runner now skips remaining candidate trials automatically after a severe first-sample regression.

There were no randomized block orders, power/temperature controls, confidence intervals, actual Pi tool cycles, cold-prefill measurements or populated 160K tests. The vector result has only one complete measured primary sample and is not a precise estimate of its average performance. It was sufficient to reject the candidate. The original saved prefix was built using the baseline; this does not evaluate a whole conversation built from scratch with the alternative kernel.

The basic existing-kernel substitution did not improve this workload. A more substantial MMA kernel optimization would need new implementation work, controlled isolated timings and numerical validation, followed by ordinary end-to-end tests. No speedup is promised from that possible future work.

## Reproduction and retained artifacts

- [Summary](summary.json), [measured samples](samples.json), [planned protocol](manifest.json), [early stopping](early-stop.json).
- Case directories contain actual launch flags, server properties, logs, cache restores, request JSON, stream events and results.
- [Dispatch patch](../hip-attention-dispatch.patch), [test and tensor-export patch](../hip-attention-tests.patch), [experiment build manifest](../hip-attention-build/artifact-manifest.json).
- [Build helper](../Build-HIPDiagnostic.ps1), [numerical probe wrapper](../Invoke-AttentionProbe.ps1), [normal-mode experiment launcher](../Start-AttentionExperiment.ps1), [benchmark controller](../run_attention_sweep.py).

The build helper's `-BuildAttentionTests` option compiles the upstream test harness using the same local toolchain. Normal server experiments use `Start-AttentionExperiment.ps1`; it defaults to automatic mode. The numerical probes intentionally retain and report their upstream CPU-comparison failures. Use the float64 checks to interpret those failures; do not suppress them or relax the tolerance. These tools are research artifacts, not the recommended Pi launcher.

The ordinary `Start-ROCmServer.ps1` remains the recommended launcher. The inference server is shut down. Production weights, caches, drivers, global settings and Pi conversations were not changed; saved test-cache reads were used for this experiment.
