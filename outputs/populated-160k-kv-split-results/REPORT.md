# Populated 160K: useful optimization, no dependable 15 TPS floor

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

The eight-block backend completed actual 160000-token input without truncation, but generation varied substantially by request. The initial 512-token generation reached **17.610 TPS**; a subsequent 256-token cached follow-up measured **13.631 TPS**, below the user's practical target. A separate restored-prefix comparison measured **15.150 TPS** for eight blocks versus **12.850 TPS** for neighboring automatic controls, an observed **17.90% gain**.

Thus the optimization remains useful at this context, while 160K actual input is not established as consistently usable under a 15 TPS floor. Allocation remained 163840 throughout; these are populated-context measurements, not allocation-only tests.

## Extending the saved prefix to 160000 input tokens

The test restored the existing 103194-token synthetic prefix and evaluated **56806 new tokens** to reach **160000 total input tokens**. The input includes a new repeated-structure synthetic code section and a validation/test-design request. It is not a completely cold 160K prompt, and the timing below must not be described as one.

| Measurement | Observed |
|---|---:|
| Reused input tokens | 103194 |
| Newly evaluated input tokens | 56806 |
| Final total input tokens | 160000 |
| New-input prompt evaluation | 820.722 s |
| Prompt evaluation throughput, new tokens only | 69.215 tokens/s |
| Client TTFT | 820.803 s |
| Generated output tokens | 512 |
| Generation throughput | 17.610 TPS |
| Client request wall time | 849.835 s |
| MTP proposals / accepted | 348 / 337 |

Server startup and the preceding slot restore are excluded. The large new-input wait remains significant despite the generation improvement. No controlled cold 160K benchmark or causal comparison against the older cold 100K results was performed.

The completed state was saved separately as `work/slot-cache/synthetic-160k-kv-split.bin`: **160511 cached tokens**, **4085390040 bytes**, client save time **7.646 seconds**. The final generated token is not yet evaluated, which accounts for the difference between the 160512-token completed history and 160511 cached tokens. The original 103K snapshot was preserved. This is synthetic benchmark state, not a Pi conversation. The save followed recent inference and is not a cold-storage or reboot test.

## Live cached follow-up

The next request had **160537 actual input tokens**, reused **160511**, evaluated **26**, and generated **256** without truncation.

- Prompt evaluation: 0.655 s; client TTFT: 0.744 s.
- Generation: **13.631 TPS**; client wall time: **19.466 s**.
- MTP proposals / accepted: **215 / 147**.

This differs from the initial high-acceptance generation and shows why one 17.6 TPS sample cannot establish a floor. The comparison also changes request content, generation budget and sampling state; it does not isolate any one of those causes.

## Matched restored-prefix comparison

After restarting each server, the saved 160511-token prefix was restored before a discarded 64-token warmup and again before a measured **512-token** request. Every measured request had 160537 input tokens, reused 160511 and evaluated 26. The three measured requests share SHA256 `4b109329f5cc2b626a48dd1da782a9a46559911a9c1e897ad45f88b501394265`.

| Block | Configuration | Generation TPS | Draft proposals / accepted |
|---|---|---:|---:|
| 1 | Automatic four blocks | 12.907 | 410 / 305 |
| 2 | Eight blocks | **15.150** | 404 / 309 |
| 3 | Automatic four blocks | 12.792 | 410 / 305 |

The controls averaged 12.850 TPS and drifted -0.892% from first to last. Their complete generated token sequences and draft counts match. The candidate output differs. The calculated gain is 17.90% over the mean control; this is a practical end-to-end comparison with changed generated content and MTP work, not a fixed-token-stream kernel percentage. There is only one measured candidate request in this bracket.

Do not pool the 256-token live follow-up with these 512-token restored trials as repetitions of one protocol. Their warmup/sampling histories and output budgets differ.

## Settings, scope and evidence

Hardware: Ryzen AI 7 350 / Radeon 860M gfx1152, Windows, 64 GB DDR5-5600. llama.cpp b11888, commit `48c9f08b7b25766324282b7605177e2f2be755bb`, private HIP backend with isolated ROCm 10. Same Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf at the existing Desktop model path. F16 K/V, MTP 2, context allocation 163840, one slot, GPU layers 99, Flash Attention, Jinja, batch 2048, microbatch 512, eight main/draft/generation/batch threads, auto load mode, normal graph replay, operation profiling disabled. Automatic and candidate blocks use the same private DLL; only the narrow KV work-split mode changes.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, ignore_eos enabled, completed empty thinking blocks. The raw first and follow-up requests and results are retained at the report root; case directories retain the restored comparison. Launch flags and server properties are saved for every process.

At 160000 input tokens, the 163840 allocation leaves only 3840 tokens for subsequent output/history growth; these tests stayed within it. The actual populated prefix was verified before every measurement. This does not establish retrieval, reasoning, coding or tool-call quality through the context. Independent numerical attention checks at contiguous/strided 160K KV length passed earlier, but are limited synthetic fixtures, not a model-quality evaluation.

No power/temperature controls, randomized block order, confidence intervals, real Pi/SWE task or full cold 160K input was added. The saved comparison prefix was generated using the eight-block candidate; holding that same state fixed across modes evaluates continuations from it, not entire conversations populated independently with each configuration.

- [Manifest](manifest.json), [summary](summary.json), [bracket samples](bracket-samples.json), [saved-prefix metadata](saved-prefix.json).
- Raw `populate-160000-*` and `cached-160k-followup-*` files, plus `server/` and per-case launches, logs, props, restore responses and results.
- [Runner](../run_populated_160k.py), [opt-in launcher](../Start-KVSplitServer.ps1), [primary optimization report](../kv-split-100k-results-verified/REPORT.md).

The runner now also supports `--input-tokens 128000` for a separate middle-point experiment. To repeat this 160K protocol, pass a fresh `--out-dir` and fresh `--slot-filename`. Historical files above were preserved. All bracket requests were checked for expected populated/cache/new-token counts, output length and no truncation. Original production files and defaults remain unchanged.
