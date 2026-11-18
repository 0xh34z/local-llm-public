# Qwen3.6 35B-A3B: completed 100K drafting comparison

User hardware: Radeon 860M on Windows, Vulkan. llama.cpp b11888/48c9f08b7. Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. All runs used context allocation 163840, one slot, GPU layers 99, Flash Attention on, Q8 K/V, batch 2048, microbatch 512 and eight CPU threads. Load mode was auto (explicit on the new runs, default on the original MTP-2 run).

The disabled-drafting run's process command line was independently checked and saved in server-launch.json. It explicitly uses --spec-type none. Both response summaries contain no nonzero draft-token statistics. Their generation settings serialize speculative.types as "none,none" (only disabled types, with no drafting method). Both requests completed without truncation.

| Metric | MTP off, new | MTP 1 | MTP 2 | MTP 3 |
|---|---:|---:|---:|---:|
| Fresh input tokens | 102400 | 102400 | 102400 | 102400 |
| Fresh cached tokens | 0 | 0 | 0 | 0 |
| Prompt processing seconds | 952.462 | 916.927 | 929.807 | 947.643 |
| Prompt processing tokens/s | 107.511 | 111.677 | 110.130 | 108.058 |
| Client first-token seconds | 952.508 | 916.973 | 929.850 | 947.700 |
| Initial generated tokens | 512 | 512 | 512 | 512 |
| Initial generation tokens/s | 10.324 | 11.632 | 9.781 | 10.334 |
| Initial generation seconds | 49.497 | 43.929 | 52.242 | 49.450 |
| Cold total wall seconds | 1002.017 | 960.914 | 982.104 | 997.162 |
| Follow-up reused tokens | 102911 | 102911 | 102911 | 102911 |
| Follow-up new prompt tokens | 28 | 28 | 28 | 28 |
| Follow-up first-token seconds | 1.486 | 1.416 | 1.466 | 1.483 |
| Follow-up generated tokens | 256 | 256 | 256 | 256 |
| Follow-up generation tokens/s | 10.293 | 11.021 | 9.906 | 9.935 |
| Follow-up total wall seconds | 26.274 | 24.562 | 27.219 | 27.157 |

The complete cold API request payloads match across all four configurations, including every prompt token, sampling field, seed and output limit. Follow-up input lengths match, but follow-up content includes the configuration's own generated answer and therefore differs. MTP 1 has the highest observed initial and follow-up generation rates. Its measured gain over disabled drafting is about 13% initially and 7% on the follow-up. Counts 2 and 3 do not outperform the disabled baseline consistently in these samples. None of these configurations meets the user's 15 tokens/s requirement at populated 100K context.

Method: deterministic synthetic Python corpus, 102400 total input tokens including fixed question and manually supplied Qwen delimiters, non-thinking output, temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0. Output lengths are forced using ignore_eos=true. A fresh request is followed by an exact append of generated token IDs and a fixed new question. Streamed prefill progress placeholder token IDs are excluded from output counts and first-token timing.

Each configuration was measured once at a different time. Power, thermals and background activity were not controlled. Stochastic outputs, expert routing and MTP acceptance vary. Rankings and percentage changes describe these measurements; they are not confidence-bounded causal estimates or evidence about quality. Prefill differences in particular do not establish that disabling MTP makes prefill slower. No coding, reasoning, retrieval or tool-call quality score was measured.

The user's separate earlier llama-bench pp102400 result (115.58 tokens/s) used a different synthetic workload and execution path and is not a matched API baseline.

The server remains active with drafting disabled and the synthetic benchmark conversation in its slot.

Reproduce using a fresh output directory:

```powershell
python outputs/run_mtp_benchmark.py --prompt-tokens 102400 --mtp-draft 0 --expected-context 163840 --out-dir outputs/mtp-100k-off-repeat
```

Saved alongside this report: launch command, server properties, corpus, manifest, exact requests, raw events and response summaries.
