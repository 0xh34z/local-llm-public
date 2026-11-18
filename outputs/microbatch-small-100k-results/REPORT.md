# Smaller microbatches: a modest incremental-prefill option

Microbatch 256 processed this fixed long-context suffix in **49.015 seconds**, versus **50.353 seconds** for pooled microbatch-512 controls: **2.66% shorter prompt time**. Microbatch 128 was slower. The two 512 control means differed by only -0.139%, so the 256 result is more encouraging than the drift-confounded logical-batch/thread sweep. It remains a small, narrowly measured benefit.

Keep 512 as the general default until cold-prefill and combined-workload behavior are established. Microbatch 256 is available as an explicit incremental-prefill experiment; do not infer that it accelerates a fresh cold 100K prompt or improves generation.

| Block | Microbatch | Measured requests | Mean prompt seconds | Range seconds |
|---|---:|---:|---:|---:|
| 1 | 512 | 2 | 50.388 | 50.262-50.514 |
| 2 | 256 | 2 | 49.015 | 48.951-49.078 |
| 3 | 128 | 2 | 52.381 | 52.290-52.471 |
| 4 | 512 | 2 | 50.318 | 50.183-50.453 |

The 128 result was 4.03% longer than pooled controls. These percentages are calculations from measured means. There were two measured samples per setting and a fixed order, without randomized repeats, power/temperature control or confidence intervals.

## Actual workload

Every measured request had 107291 input tokens, restored the same 103194-token prefix, evaluated 4097 tokens, and generated one identical output token (ID 27775), without truncation. The new suffix itself is 4096 tokens; the extra evaluated token comes from the saved history's final unevaluated output token. All eight measured requests share SHA256 `c27057ee54067b77eae29d809b505fcbf47fa33617fbc602638ef5bec5a2e6ca`.

Each server first ran a discarded 1024-token suffix warmup (1025 newly evaluated tokens). The original baseline slot was restored before every warmup and measured request. This holds prefix state constant while changing how new input is processed; it does not compare conversations populated entirely using each microbatch. Restore and server startup are excluded from request TTFT. A one-token completion's generation TPS is not meaningful and is not used.

Hardware: Ryzen AI 7 350 / Radeon 860M gfx1152, Windows, 64 GB DDR5-5600. Original ROCm b11888 backend and isolated ROCm 10 libraries. Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf at the existing Desktop model path. F16 K/V, MTP 2, context allocation 163840, one slot, GPU layers 99, Flash Attention, Jinja, logical batch 2048, eight main/draft generation and batch threads, auto load mode, normal graph replay and no operation profiling. Temperature zero, forced one-token output. Only microbatch size changes.

This tests incremental prefill at an already populated 103K prefix, not a complete cold 100K/160K input, a Pi tool cycle, a generation-speed change, or model quality. The eight-block KV optimization was not used here; these samples use the original packaged backend. The combined effect of that optimization and microbatch 256 was not established in this sweep.

## Evidence and use

- [Manifest](manifest.json), [summary](summary.json), [table](RESULTS.md). Case directories retain launch arguments, server properties, restore responses, complete requests, streamed events, results and logs.
- [Runner](../run_prefill_settings_sweep.py): repeat with `--microbatch-sweep --out-dir outputs/<fresh-directory>` using the local runtime and saved synthetic prefix.
- The existing launchers accept `-MicroBatchSize 256` for an explicit trial. Their default remains 512.

All eight measured requests were checked for correct input/cache/new-token counts, identical one-token output and no truncation. No new full cold-prefill claim is made. Original backend files and launcher defaults remain unchanged. The test server was stopped while idle before subsequent experiments.
