# Radeon 860M: ROCm draft depth and KV precision at 100K

Keep **ROCm + F16 K/V + MTP 2** as the current generation baseline. MTP 3 produced no generation gain in this sample, and Q8 K/V was substantially slower. This is the best generation configuration measured so far, not proof of a global optimum or a sustained 15 tokens/s minimum.

## Measurements

Hardware: user's Ryzen AI 7 350 / Radeon 860M (gfx1152), 64 GB DDR5-5600, Windows. llama.cpp b11888 / 48c9f08b7; isolated ROCm 10.0 runtime; Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. GPU explicitly selected as ROCm0. All runs allocate 163840 tokens, one slot, 99 GPU layers, Flash Attention, batch 2048, microbatch 512, eight threads, auto load mode, and the same model weights.

| Measurement | F16 KV, MTP 2 (earlier baseline) | F16 KV, MTP 3 (new) | Q8 KV, MTP 2 (new) |
|---|---:|---:|---:|
| Cold input tokens | 102400 | 102400 | 102400 |
| Prompt evaluation seconds | 807.994 | 758.600 | 776.234 |
| Prompt tokens/s | 126.734 | 134.986 | 131.919 |
| Cold client TTFT seconds | 808.047 | 758.684 | 776.294 |
| Initial generation tokens/s | **15.744** | 15.684 | 11.675 |
| Initial output tokens | 512 | 512 | 512 |
| Cold client wall seconds | 840.517 | 791.281 | 820.083 |
| Initial accepted / proposed drafts | 314 / 392 | 339 / 513 | 303 / 415 |
| Follow-up reused tokens | 102911 | 102911 | 102911 |
| Follow-up newly processed tokens | 28 | 28 | 28 |
| Follow-up prompt evaluation seconds | 0.615 | 0.645 | 0.699 |
| Follow-up client TTFT seconds | **0.689** | 0.709 | 0.768 |
| Follow-up generation tokens/s | **14.975** | 14.295 | 11.268 |
| Follow-up output tokens | 256 | 256 | 256 |
| Follow-up client wall seconds | **17.731** | 18.561 | 23.413 |
| Follow-up accepted / proposed drafts | 152 / 205 | 161 / 279 | 149 / 211 |
| Truncated requests | none | none | none |

MTP 3's initial generation is essentially tied with MTP 2; its follow-up was approximately 4.5% slower. Q8 K/V generation was approximately 26% slower initially and 25% slower on the follow-up than the earlier F16 MTP-2 baseline. There is no throughput reason in these samples to accept KV quantization for this ROCm setup.

Cold request wall time was lowest with MTP 3 because its measured prefill was faster. The two new prefills were approximately 12m39s and 12m56s; the earlier F16 MTP-2 prefill was 13m28s. Runs occurred at different times without controlled temperatures, power limits or background workloads, so do not attribute every prefill difference to the changed flag. F16 MTP 2 is the recommendation for generation and cached continuations, not the winner of every cold-wall metric.

The 15 tokens/s requirement remains marginal at populated 100K context: the best follow-up averaged 14.975, and these measurements do not establish a minimum rate. Real Pi calls previously observed at much smaller context ran faster; those do not predict 100K–160K behavior. No populated 160K run was made.

## Reproduction and limitations

Every cold request file is byte-for-byte identical: SHA256 `572a0e822a86213d60e076b486c7f72a22d8040a322d69605b487130c63c31cd`. Live process arguments, K/V precision, draft count, server-reported context and idle state were checked before each benchmark. Each cold request processed all 102400 tokens with zero cached input. Each continuation reused 102911 tokens and processed 28 new tokens.

The synthetic Python corpus has repeated structure with unique module names and numbers. Temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1 and presence penalty 0. Output uses a completed empty thinking block and forced lengths (ignore_eos). Follow-ups include each run's generated tokens and therefore are not identical inputs across runs. There was one cold-plus-follow-up pair per configuration, without randomized order or confidence intervals. No coding correctness, reasoning, retrieval, tool-call quality or quantized-cache quality evaluation was performed. These are normal inference runs, without the Vulkan performance logger.

Raw evidence:

- [Earlier F16 MTP-2 baseline](rocm-100k-mtp2-f16-results/REPORT.md)
- [New F16 MTP-3 report](rocm-100k-mtp3-f16-results/REPORT.md), with server log, requests, results and SSE events alongside it
- [Q8 MTP-2 cold result](rocm-100k-mtp2-q8-results/cold-102400-mtp2-result.json) and [cached continuation](rocm-100k-mtp2-q8-results/append-102400-mtp2-result.json), with verified launch and server log alongside them
- [Benchmark script](run_mtp_benchmark.py)

## Restored configuration

The Q8 test server was stopped while idle. F16 MTP 2 was restored on port 9931. The server restart clears the in-memory conversation cache. The earlier Pi log was preserved as `rocm-server-before-cache-tests.log`; the restored server logs to `rocm-server.log`.

The launcher now defaults to MTP 2 and F16, and accepts explicit `-KvCache f16` or `-KvCache q8_0` plus an optional `-LogPath`. Its PowerShell syntax was checked. Future launch, with only one server on port 9931:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kasper\Documents\GitHub\local-llm\outputs\Start-ROCmServer.ps1" -DraftTokens 2 -KvCache f16
```

Remaining distinct work would be a batch/microbatch sweep for cold prefill, repeated trials if small differences matter, and inspection of real Pi prefix reuse. None of those has been measured in this comparison. A custom engine is not required to reproduce the results above.
