# Decode ceiling round (November 19, 2026)

Question: how far is current generation speed from what this laptop's memory bandwidth allows, and where does the rest of the time go?

## Summary

- **No change to production.** Pi was restored with the production DLL (`51230C04…B4B5`), MTP 2 and the 3861-token slot `pi-before-ceiling-20261119`; health and the Pi `READY` smoke test passed (`restored-service.json`).
- **Server overhead is negligible.** Without speculation the server's per-token time matches `llama-bench` decode within 1.5 ms at 1K, 103K and 160K (it was slightly *faster* at depth).
- **Single-token attention already runs at bandwidth.** Decode time grows by 0.26 µs per populated token; reading that token's F16 K/V at 81.98 GB/s takes 0.25 µs.
- **MTP 2 stays the best setting.** MTP 3 was slower than bracketed MTP 2 for sampled output at every length (103K 15.59 vs 17.19 tokens/s; 160K 13.20 vs 14.91). MTP 1 was slower too. MTP 3 won only on the short repetition workload.
- **Bandwidth ceiling (calculation).** With measured acceptance, reading every required byte once at 81.98 GB/s would allow about 24.5 tokens/s at 103K and 20.9 at 160K, against measured 17.3 and 15.0. The cycle runs at about 70–72% of that floor. The remaining time is spread over three places of similar size (table below), none of which a single change removes.
- **Linux:** the measured software overhead outside GPU kernels is close to zero, so a reinstall would only help if the Linux driver/compiler stack runs the same HIP kernels faster or holds higher clocks. That is not tested here; see "Linux" below.

## Conditions

ROCm HIP gfx1152, llama.cpp b11888 with the eight-block KV split DLL (`work/hip-kv-split-bin`), F16 K/V, flash attention, batch 2048, microbatch 512, eight threads, one slot, context 163840. Windows Balanced plan, on AC (battery status 2). During `llama-bench`, median GPU clock was 2653–2684 MHz and memory clock 2754–2770 MHz while the GPU was busy; maximum temperature 81 °C.

## Measurements

### Main model alone (`llama-bench`, 5 repetitions per point)

`outputs/run_depth_bench.ps1`; depth = populated KV positions (random tokens, prefilled once per depth). `tg` is per generated token from `-n 32`; `ppN` is one N-token batch, the closest stand-in for an MTP verify step. Both runs are listed where they overlap.

| Depth | tg (ms/token) | pp2 (ms) | pp3 (ms) | pp4 (ms) |
|---:|---:|---:|---:|---:|
| 0 | 46.1 | 59.4 / 58.8 | 67.2 / 69.7 | 77.5 |
| 8192 | 47.7 | 58.6 | 67.6 | |
| 32768 | 54.8 | 70.6 | 82.2 | |
| 65536 | 64.6 | 80.0 | 96.3 | |
| 103194 | 73.9 | 90.1 / 89.8 | 99.1 / 101.3 | 128.0 |
| 131072 | 79.9 | 117.3 | 117.8 | |
| 160511 | 87.8 | 125.3 / 115.4 | 123.5 / 124.1 | 159.5 |

pp2 at 160K varied by 8% between runs (standard deviations 5.8 and 1.3 ms). Sources: `depth-bench/bench.jsonl`, `depth-bench-pp234/bench.jsonl`.

### Server decode (`outputs/run_ceiling_server.py`)

Each case starts a fresh server with `Start-CeilingServer.ps1` (production flags, varying only `--spec-draft-n-max`, or no speculation for MTP 0). "Short" is the last 1024 tokens of the 103K prompt; 103K/160K are the restored snapshots used in earlier rounds. Sampled = 256 tokens, temperature 0.6, seed 42; repetition = 128 tokens, temperature 0. A cycle is one main-model step: generated tokens minus accepted drafts.

Run 1 order MTP 2, 0, 1, 2 (`server-cycles/`); run 2 order MTP 2, 3, 0, 3, 2 (`server-cycles-mtp3/`). MTP 2 brackets agreed within about 1–3% in both runs. Run 1's MTP 0 case skipped the restored contexts because of a script bug (fixed in `run_fine_split.py`, which now treats a missing draft count as zero); run 2 covers them.

Pooled over both runs, in tokens/s (ms per cycle, tokens per cycle):

| Setting | Short sampled | Short repetition | 103K sampled | 103K repetition | 160K sampled | 160K repetition |
|---|---:|---:|---:|---:|---:|---:|
| MTP 0 | 21.65 (46.2, 1.00) | 21.62 | 13.80 (72.5) | 13.82 | 11.47 (87.2) | 11.49 |
| MTP 1 | 24.66 (74.7, 1.84) | 26.57 | 15.13 (119.1, 1.80) | 16.94 | 12.40 (141.5, 1.75) | 14.04 |
| MTP 2 | 25.87 (94.9, 2.45) | 30.99 | 17.31 (139.5, 2.42) | 20.90 | 14.99 (163.4, 2.45) | 18.07 |
| MTP 3 | 24.60 (115.2, 2.83) | 34.09 | 15.59 (179.5, 2.80) | 22.03 | 13.20 (215.5, 2.84) | 14.12 |

MTP 3 against its bracketing MTP 2 controls in run 2 (tokens/s): short sampled 24.60 vs 25.89, 103K sampled 15.59 vs 17.19, 160K sampled 13.20 vs 14.91, 103K repetition 22.03 vs 20.57, 160K repetition 14.12 vs 18.09. At 160K repetition, MTP 3 accepted only 85 of 123 drafts.

## Calculations

### Byte floors

`outputs/byte_budget.py` reads the GGUF tensor table (`byte-budget.json`). Per main step: 1.597 GB non-routed weights in layers 0–39, plus a 0.540 GB output head, plus 0.614 GB of routed experts per token, plus 20480 B of K/V per populated token, plus 0.132 GB of DeltaNet state read and written. The layer-40 MTP head reads 0.042 GB non-routed plus 0.015 GB of experts plus the output head, plus 2048 B of K/V per token. For N-token batches, routed experts are counted as independent uniform routing (3 tokens → 23.3 of 24 possible experts per layer, so cross-token expert reuse is worth under 3%).

`common_speculative_impl_draft_mtp` runs three MTP-layer decodes per MTP 2 cycle. One is a catch-up pass over the verified rows, without the output head; the other two are single-token draft passes with the head (`work/llama.cpp/common/speculative.cpp`, `process()` and `draft()`). `outputs/analyze_ceiling.py` includes all three; `analysis.json` holds every row.

| Main step | Depth 0 floor | Measured | Ratio | Depth 160511 floor | Measured | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 token (tg) | 35.2 ms | 46.1 | 1.31 | 75.3 ms | 87.8 | 1.17 |
| 3 tokens (pp3) | 49.4 ms | 68.4 | 1.38 | 89.5 ms | 123.8 | 1.38 |
| 4 tokens (pp4) | 56.2 ms | 77.5 | 1.38 | 96.3 ms | 159.5 | 1.66 |

### Cost per populated token (slope from depth 0 to 160511)

Measured with `llama-bench`; floor = 20480 B / 81.98 GB/s = 0.25 µs.

| Batch | Kernel (code reading, `ggml_cuda_get_best_fattn_kernel`) | Slope | vs floor |
|---|---|---:|---:|
| 1 token | Tile (Q×GQA 8 = 8 ≤ 16, so no WMMA) | 0.26 µs | 1.04× |
| 2 tokens | Tile (16 ≤ 16) | 0.35–0.41 µs | 1.4–1.6× |
| 3 tokens | MMA/WMMA, eight-block split (Q=3, ≥100K only) | 0.35 µs | 1.4× |
| 4 tokens | MMA/WMMA, default four blocks (split not enabled for Q=4) | 0.51 µs | 2.0× |

### Where the MTP 2 cycle goes at 160K (sampled, 163.4 ms measured)

| Part | Measured | Floor | Excess |
|---|---:|---:|---:|
| Verify step, weights and state (pp3 at depth 0) | 68.4 ms | 49.4 ms | 19.0 ms |
| Verify step, depth-dependent attention (pp3 160K − pp3 0) | 55.4 ms | 40.1 ms | 15.3 ms |
| MTP passes and server (server cycle − pp3 at 160K) | 39.6 ms | 27.7 ms | 11.9 ms |
| **Cycle** | **163.4 ms** | **117.2 ms** | **46.2 ms** |

At 103K the same split is 68.4 / 31.8 / 39.3 ms measured against 49.4 / 25.8 / 23.4 ms floors (139.5 vs 98.6 ms).

### What-if ceilings (tokens/s, measured acceptance kept)

| Scenario | 103K sampled | 160K sampled |
|---|---:|---:|
| Measured MTP 2 | 17.3 | 15.0 |
| 3-token attention as efficient as 1-token (depth part = tg's depth part) | 17.8 | 16.4 |
| Every part at the best observed efficiency, 1.17× floor | 21.0 | 17.9 |
| Every byte read once at 81.98 GB/s | 24.5 | 20.9 |
| MTP 3 with an eight-block split for 4-token verify (pp4 depth part = pp3's) | 17.4 | 15.0 |

The last row is why no Q=4 split build was made. Even if it worked fully, MTP 3 would only tie MTP 2 for sampled output.

## Hypotheses (not tested in this round)

- **MTP-pass overhead.** The three small MTP-layer decodes per cycle take about 12 ms more than their bytes (≈4 ms each, even at short context). Likely causes are fixed per-call host cost (graph build, scheduling, synchronization) or poor efficiency of small kernels. Instrumented timings of `llama_process` on the draft context would separate the two.
- **Extra verify tokens.** Each token past the first costs 10–11 ms at depth 0 against a 7.1 ms byte floor, below the single-token path's 76% efficiency. The RDNA3 multi-token MMVQ MoE path assigns one warp per token and dense fusion only runs for one column; that may explain part of it. Because routing rarely overlaps, a weight-reuse kernel would have little to gain. Per-expert efficiency is the lever.
- **Q=3 attention.** The MMA path reaches about 71% of bandwidth on K/V; the tile path at one token reaches about 96%. At most 13.7 ms per cycle at 160K is available here (16.4 tokens/s). Earlier rounds already rejected forced tile/vector, 16/64-column tiles, 128-thread workgroups and LDS-resident Q.

## Linux

Software outside GPU kernels costs almost nothing measurable (MTP 0 server = `llama-bench` within 1.5 ms). The remaining gaps are in kernel efficiency and in the MTP pass cost; Linux would run the same HIP kernel sources. It could still differ through clocks and power management (the Windows Best performance plan already gave +3.4–3.9% at 103K/160K in the hardware round), a different ROCm compiler, or a lower per-call host cost for the small MTP decodes. None of that is measured. A cheaper first comparison on this Windows install would be the llama.cpp Vulkan backend at the same depths and flags.

## Files

- `byte-budget.json`: floors from `outputs/byte_budget.py` (calculation).
- `depth-bench/`, `depth-bench-pp234/`, `depth-bench-smoke/`: `llama-bench` JSONL, stderr, manifests, GPU sensors.
- `server-cycles/`, `server-cycles-mtp3/`: per-case server logs, requests, results, `samples.json`, `summary.json`.
- `analysis.json`: `outputs/analyze_ceiling.py --bench depth-bench/bench.jsonl depth-bench-pp234/bench.jsonl --server server-cycles/samples.json server-cycles-mtp3/samples.json`.
- `restored-service.json`: Pi restore evidence. The restore script used is `work/ceiling-round/restore-pi.ps1`.
