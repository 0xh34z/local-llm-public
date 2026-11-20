# Populated 128K: useful gain, still borderline against 15 TPS

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

The eight-block candidate completed actual 128000-token input without truncation. Initial generation measured **16.604 TPS**, the live 256-token cached follow-up **15.154 TPS**, and a restored-prefix 512-token trial **14.779 TPS**. Thus 128K was not reliably above the user's 15 TPS threshold in this small sample.

The restored trial still improved on matched automatic controls: **14.779 versus 13.840 TPS**, a calculated **6.79% gain**. This experiment was added after the 160K follow-up fell below the target, rather than inferring an unmeasured middle-point rate.

## Actual input extension

The original 103194-token snapshot was restored. The request evaluated **24806 new tokens**, reaching **128000 total input tokens**, then generated 512. The repeated-structure synthetic code and question follow the same construction as the 160K test, but the context and resulting generated content differ. This is not a completely cold 128K input.

| Measurement | Observed |
|---|---:|
| Reused input tokens | 103194 |
| Newly evaluated input tokens | 24806 |
| Total input tokens | 128000 |
| New-input prompt evaluation | 306.675 s |
| Prompt throughput, new tokens only | 80.887 tokens/s |
| Client TTFT | 306.723 s |
| Initial generated tokens | 512 |
| Initial generation throughput | 16.604 TPS |
| Initial request wall time | 337.507 s |
| MTP proposals / accepted | 408 / 306 |

The completed state was saved separately as `work/slot-cache/synthetic-128k-kv-split.bin`, with **128511 cached tokens**, **3362830040 bytes** and a client save time of **7.803 seconds**. The completed history is 128512 tokens; its final generated token has not yet been evaluated. This is synthetic benchmark state, not a Pi conversation. Save/restore followed recent use and does not establish cold-storage or reboot latency.

The live follow-up contained **128537 input tokens**, reused **128511**, evaluated **26** and generated **256** without truncation. Prompt evaluation was 0.619 seconds, client TTFT **0.668 seconds**, generation **15.154 TPS**, and request wall time **17.505 seconds**. It proposed 221 MTP tokens and accepted 143.

## Restored-prefix comparison

Fresh servers used automatic, eight-block, then automatic mode. Before each discarded 64-token warmup and each measured 512-token request, the same 128511-token snapshot was restored. All three measured requests had 128537 input tokens, reused 128511 and processed 26.

| Block | KV split | Generation TPS | Draft proposals / accepted |
|---|---|---:|---:|
| 1 | Automatic four blocks | 13.848 | 425 / 298 |
| 2 | Eight blocks | **14.779** | 453 / 284 |
| 3 | Automatic four blocks | 13.831 | 425 / 298 |

Controls averaged 13.840 TPS and drifted -0.129%. Their complete output token sequences and MTP counts match. Candidate output differs and its acceptance was lower, so the calculated 6.79% gain is an end-to-end result, not a fixed-token-stream kernel percentage. There is one measured candidate request in this bracket.

All three bracket request files share SHA256 `fc263f4d505b465a09a9374918f535628668d5e5e600ca27911fd9dcaff46afd`. Do not pool the live 256-token follow-up and these 512-token restored trials as repetitions of one protocol: their budgets and warmup/sampling histories differ.

## Configuration and limits

Ryzen AI 7 350 / Radeon 860M gfx1152, Windows, 64 GB DDR5-5600. llama.cpp b11888 / `48c9f08b7b25766324282b7605177e2f2be755bb`, isolated ROCm 10 libraries, private HIP backend. Existing Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf at the Desktop model path. F16 K/V, MTP 2, allocation **163840**, one slot, GPU layers 99, Flash Attention, Jinja, batch 2048, microbatch 512, eight main/draft/generation/batch threads, auto load mode, normal graph replay, operation profiling disabled. Controls and candidate use the same DLL with only the narrow KV split mode changed.

Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, ignore_eos enabled, completed empty thinking blocks. At 128000 initial input tokens, allocation leaves 35840 tokens for output and further history; rate while that space fills was not established.

The prefix was created using the candidate, then held fixed across modes. This compares continuations, not entire conversations independently populated with each setting. There were no controlled power/temperature measurements, randomized order, confidence intervals, real Pi/SWE tasks or quality evaluations. Independent numerical attention fixtures previously covered sampled contiguous/strided 103K and 160K KV lengths; no separate float64 fixture was added exactly at 128K. No input/output quality or retrieval conclusion follows from these timings. Differences between 103K, 128K and 160K request rates are confounded by generated content, acceptance and timing; they are not a calibrated context-length performance curve.

## Evidence

- [Manifest](manifest.json), [summary](summary.json), [bracket samples](bracket-samples.json), [saved-prefix metadata](saved-prefix.json).
- Raw `populate-128000-*` and `cached-128k-followup-*` files, plus server/case directories with flags, props, restores, logs, requests and results.
- [Runner](../run_populated_160k.py): use `--input-tokens 128000`, a fresh `--out-dir`, and a fresh `--slot-filename` for repetition.
- [Primary KV split report](../kv-split-100k-results-verified/REPORT.md), [160K comparison](../populated-160k-kv-split-results/REPORT.md).

All bracket request lengths, cache/new-token counts, output counts and lack of truncation were checked. The original 103K snapshot, production backend files and launcher defaults remain unchanged. The final server was stopped while idle.
