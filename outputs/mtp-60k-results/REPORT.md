# Measured 60K versus 100K MTP benchmark

Correction: the 60K test uses exactly 61440 populated input tokens (60 x 1024), with a 64000-token allocation and ONE MTP draft token. The saved server-props.json confirms the 64000 allocation. The user identified the draft count, and the current process command line was independently checked afterward and confirms --spec-draft-n-max 1. The previous report incorrectly assumed the earlier 100K server settings were unchanged. File names containing mtp2 are historical labels, not verified draft-count metadata.

Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. Build b11888/48c9f08b7, Windows Vulkan on the user's Radeon 860M. At 60K: MTP draft-mtp with maximum ONE draft token, context allocation 64000. At 100K: maximum TWO draft tokens, context allocation 163840. Both: Q8 K/V, GPU layers 99, Flash Attention on, batch 2048, microbatch 512, CPU threads 8.

| Measurement | 61440-token input | Previous 102400-token input |
|---|---:|---:|
| Fresh prompt processing time | 420.017 s | 929.807 s |
| Fresh prompt processing rate | 146.280 tokens/s | 110.130 tokens/s |
| Fresh request first generated token | 420.062 s | 929.850 s |
| Generation rate, 512-token response | 15.054 tokens/s | 9.781 tokens/s |
| Fresh request total wall time | 454.012 s | 982.104 s |
| Follow-up input tokens | 61979 | 102939 |
| Follow-up cached tokens | 61951 | 102911 |
| Follow-up newly processed tokens | 28 | 28 |
| Follow-up prompt processing time | 1.068 s | 1.402 s |
| Follow-up first generated token | 1.108 s | 1.466 s |
| Follow-up generation rate, 256 tokens | 15.886 tokens/s | 9.906 tokens/s |
| Follow-up total wall time | 17.166 s | 27.219 s |
| Fresh request draft tokens accepted/proposed | 234/277 | 308/405 |
| Follow-up draft tokens accepted/proposed | 124/131 | 153/202 |
| Truncated | false on both requests | false on both requests |

The fresh 60K request reused zero cached tokens. Generated token counts and cache/new-prompt accounting were validated against the returned timing fields. MTP activity is confirmed by nonzero draft counts and returned speculative.types containing draft-mtp.

Method: same deterministic synthetic Python corpus, wrapper/question, sampling settings and forced output lengths as the previous 100K run, but DIFFERENT draft counts and context allocations. Temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0. The input uses fewer corpus tokens while retaining the same final question and completed empty thinking block. Output is non-thinking, fixed-length generation using ignore_eos=true. The follow-up appends sampled token IDs and the same follow-up question used at 100K.

These are single-run synthetic throughput measurements, not coding, reasoning, retrieval, or tool-call quality scores. Depth, draft count, allocation, response content and MTP acceptance differ between the runs. The observed generation improvement cannot be attributed solely to context depth or to draft count. No controlled MTP-off/1/2/3 comparison was performed. Background workload, power, thermals, and run-to-run variability were not controlled. The server remains running with the benchmark conversation in its slot.

Reproduce from the workspace with:

```powershell
python outputs/run_mtp_benchmark.py --prompt-tokens 61440 --mtp-draft 1 --expected-context 64000 --out-dir outputs/mtp-60k-repeat
```

The exact requests, corpus, manifest, server properties, raw stream events, and response summaries are saved alongside this report. Use the verified one-draft-token, 64000-context launch command to reproduce the server configuration. Re-running with the same output directory overwrites those files.
