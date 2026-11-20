# Logical batch and prompt threads: no convincing gain

Keep logical batch 2048 and eight prompt/batch threads. The tested alternatives did not establish a useful improvement when processing the same new input at a restored 103K prefix. The final baseline was 3.38% slower than the initial baseline, exceeding the apparent improvement of any candidate relative to pooled controls.

This is a matched long-context suffix test, not a fresh cold 100K prompt and not a generation-speed benchmark.

| Block | Logical batch | Main and draft prompt threads | Measured runs | Mean prompt seconds | Range seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 2048 | 8 | 2 | 48.606 | 48.532-48.680 |
| 2 | 512 | 8 | 2 | 48.399 | 48.125-48.674 |
| 3 | 2048 | 4 | 2 | 48.670 | 48.596-48.744 |
| 4 | 4096 | 8 | 2 | 49.903 | 49.735-50.071 |
| 5 | 2048 | 16 | 2 | 50.115 | 49.704-50.527 |
| 6 | 2048 | 8 | 2 | 50.249 | 50.114-50.384 |

The pooled baseline mean was 49.427 seconds. The fastest candidate, batch 512, was 2.08% shorter than that pooled baseline but only 0.43% shorter than the initial control. Since the controls drifted 3.38% and the candidate order was not randomized, that is insufficient evidence to change the default. The later batch-4096 and thread-16 samples should likewise not be interpreted as isolated setting-induced regressions; their difference from early controls is confounded by drift.

## Actual workload and settings

Every measured request contained 107291 input tokens: the saved synthetic history plus a fixed 4096-token new suffix. The native cache restored 103194 tokens, so 4097 tokens were evaluated, including the previous history's last unevaluated output token. Each request generated one token. All 12 requests completed without truncation, shared one request SHA256 (`c27057ee54067b77eae29d809b505fcbf47fa33617fbc602638ef5bec5a2e6ca`) and returned the same output token ID, 27775.

Each fresh server first ran a discarded 1024-token suffix warmup, then restored the original slot independently before each measured request. The warmup evaluated 1025 tokens and generated one token. Restore/startup time is excluded from prompt timings and request TTFT; the raw restore responses remain recorded. TTFT closely follows prompt evaluation because only one output token is requested. The reported generation TPS for a one-token completion is not meaningful and is not used.

Hardware: Ryzen AI 7 350 / Radeon 860M gfx1152, Windows, 64 GB DDR5-5600. Original packaged ROCm b11888 backend with isolated ROCm 10 libraries; no custom attention override. Model: `$env:LLM_MODEL_PATH (e.g. ~\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf)`. F16 K/V, MTP 2, allocated context 163840, one slot, GPU layers 99, Flash Attention, Jinja, microbatch 512, eight main and draft generation threads, auto load mode, port 9931, normal graph replay and no synchronized operation profiling. Main and draft batch thread counts change together. Sampling uses temperature zero and forced one-token output.

Model/build/context fingerprints and actual launch flags were verified. Slot state was held constant by restoring the same baseline snapshot through the native endpoint. This varies the processing settings for new tokens while retaining the baseline-created prefix; it does not test a conversation populated from scratch with each setting. The user-facing save/restore helper's stricter batch fingerprint was not changed.

## Limits and reproduction

Only two measured samples per setting and a single fixed order were used. No power/temperature controls, confidence intervals, full cold-prefill comparisons or task-quality evaluations were added. A 4K suffix is representative of this bounded synthetic new-input workload, not all real tool outputs or full repository ingestion. No candidate qualified for a claim about faster cold 100K processing, and no extra expensive cold pair was justified by this screen.

- [Manifest](manifest.json), [summary with raw timing records](summary.json), [table](RESULTS.md).
- Each case directory retains launch flags, props, server log, restores, request JSON, events and results.
- [Runner](../run_prefill_settings_sweep.py) and [server lifecycle helper](../benchmark_server.py). Use a fresh `--out-dir` to repeat the default batch/thread protocol. The runner also supports a separate `--microbatch-sweep`; those results are stored independently.

The runner was subsequently extended to support the separate microbatch sweep; the default batch/thread choices and request workload remain the same. Historical manifest/order format and records above were preserved. The original backend and production launcher defaults were not changed. The server was stopped while idle at the end of this sweep before the next experiment began.
