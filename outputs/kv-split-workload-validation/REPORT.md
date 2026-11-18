# Eight-block throughput gains on two additional 103K prompts

The eight-block KV split improved both additional synthetic throughput prompts. This supports the earlier gain extending beyond one continuation, while remaining a small performance sample rather than a speed floor or quality evaluation.

| Prompt | Automatic first TPS | Eight blocks TPS | Automatic final TPS | Gain versus mean control |
|---|---:|---:|---:|---:|
| Python LRU cache implementation and tests | 16.605 | **18.527** | 16.607 | **11.57%** |
| Open-thinking concurrency/interleaving analysis | 17.124 | **18.720** | 17.169 | **9.17%** |

Each cell is one 512-token forced generation. There is one candidate sample per prompt, bracketed by two automatic controls. Corresponding automatic controls reproduced their entire output token sequences and draft proposal/acceptance counts; candidate token sequences differ. As in the original comparison, these are end-to-end request results that include altered MTP work, not fixed-token-stream kernel percentages. Percentages are calculations from measured request averages.

## Workload and checks

Both prompts reuse the same 103194-token saved baseline prefix. Code input length is 103256, with 62 new evaluated tokens; reasoning input length is 103278, with 84 new tokens. Every measured request generated 512 tokens without truncation. Each fresh server first performed a discarded 64-token code warmup. The original prefix was restored separately before every request.

The code prompt requests a bounded thread-safe LRU cache and tests, with the empty thinking block already completed in the prompt. The reasoning prompt requests an analysis of non-atomic read/modify/write balance updates and leaves `<think>` open. Its measured 512-token output did not contain a closing thinking tag, so that row measures a reasoning segment, not time to a completed final answer. Forced output lengths and this synthetic prefix do not model a complete SWE or Pi task. No generated code was executed or scored, and no tool-call protocol or long-context retrieval accuracy was evaluated.

Settings: Ryzen AI 7 350 / Radeon 860M gfx1152, Windows, llama.cpp b11888 / `48c9f08b7b25766324282b7605177e2f2be755bb`, isolated ROCm 10 libraries and the same private HIP DLL for all controls/candidates. Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf at the existing Desktop path. F16 K/V, MTP 2, allocation 163840, one slot, GPU layers 99, Flash Attention, Jinja, batch 2048, microbatch 512, eight main/draft/generation/batch threads, auto load mode, normal graph replay, operation profiling disabled. The only selected backend setting is automatic four blocks or double/eight blocks for the narrow long-context Q3 shape.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, ignore_eos enabled. Actual process flags and server properties are retained per block. All six measured requests were verified for populated/cache/new token counts, output length and no truncation.

Code candidate TTFT was 1.045 seconds and client wall time 28.634 seconds. Reasoning candidate TTFT was 1.321 seconds and client wall time 28.626 seconds. These exclude slot restoration and server startup. The controls' higher rates than the original continuation also show that MTP acceptance and generated content materially affect throughput. No universal 15 TPS floor follows from these samples.

## Evidence and reproduction

- [Manifest and exact prompts](manifest.json), [samples](samples.json), [summary](summary.json).
- Case directories contain actual launches, server properties/logs, restore responses, requests, events and complete results.
- [Runner](../run_workload_validation.py), [opt-in launcher](../Start-KVSplitServer.ps1), [primary KV split report](../kv-split-100k-results-verified/REPORT.md).

The runner uses the local runtime and saved synthetic prefix and refuses an existing output directory or an occupied server port. Choose a fresh directory in the script before repeating. It stops only its owned idle server. No production backend/default was replaced. No controlled temperature/power measurements, randomized block order, confidence interval or model-quality test was added.
