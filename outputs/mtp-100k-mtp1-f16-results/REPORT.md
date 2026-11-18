# Qwen 100K: MTP 1 with F16 versus Q8 K/V caches

Measured on the user's Windows Vulkan llama-server, Radeon 860M, llama.cpp b11888/48c9f08b7. Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. Both runs used context allocation 163840, one slot, GPU layers 99, Flash Attention on, batch 2048, microbatch 512, eight CPU threads, automatic load mode and draft-mtp with maximum one draft token.

The F16 process command line was independently inspected before submitting the benchmark and saved in server-launch.json. The harness verified explicit -ctk f16 and -ctv f16, draft count 1 and context allocation 163840. The complete cold API request payload matches the previous Q8/MTP-1 request exactly, including every prompt token and sampling setting.

| Metric | F16 K/V, new | Q8 K/V, earlier |
|---|---:|---:|
| Fresh input tokens | 102400 | 102400 |
| Fresh cached tokens | 0 | 0 |
| Prefill time | 1109.878 s | 916.927 s |
| Prefill rate | 92.262 tokens/s | 111.677 tokens/s |
| Client first generated token | 1109.918 s | 916.973 s |
| Initial generated tokens | 512 | 512 |
| Initial generation time | 41.290 s | 43.929 s |
| Initial generation rate | 12.376 tokens/s | 11.632 tokens/s |
| Cold request total wall | 1151.220 s | 960.914 s |
| Initial accepted/proposed draft tokens | 238/273 | 232/278 |
| Follow-up input tokens | 102939 | 102939 |
| Follow-up cached tokens | 102911 | 102911 |
| Follow-up newly processed tokens | 28 | 28 |
| Follow-up prefill time | 1.194 s | 1.360 s |
| Follow-up client first generated token | 1.258 s | 1.416 s |
| Follow-up generated tokens | 256 | 256 |
| Follow-up generation time | 21.168 s | 23.138 s |
| Follow-up generation rate | 12.046 tokens/s | 11.021 tokens/s |
| Follow-up total wall | 22.437 s | 24.562 s |
| Follow-up accepted/proposed draft tokens | 116/138 | 108/146 |
| Truncated | false on both requests | false on both requests |

F16 has the highest initial and follow-up generation rates measured so far at 100K in this workload: about 6% and 9% above the Q8/MTP-1 samples respectively. It also took about 21% longer for prefill and about 20% longer for the complete fresh request. Fresh ingestion is roughly 18 minutes 30 seconds, compared with 15 minutes 17 seconds for Q8. Both cache configurations remain below the user's 15 tokens/s threshold at populated 100K context.

This does not establish that F16 is unconditionally faster or a better overall configuration. The two configurations were measured once at different times. Output text, expert routing and draft acceptance differ; follow-up prompts append each run's own generated answer. Power, thermals and background activity were not controlled. No confidence interval or quality score was measured. F16's higher numerical precision is not itself an observed coding/tool reliability improvement.

Method: same deterministic synthetic Python corpus and fixed coding question as prior runs. Non-thinking prompting with an empty completed thinking block. Temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0. Output lengths are forced with ignore_eos=true. A fresh request is followed by an append-only continuation. Stream progress placeholder tokens are excluded from output token counts and first-token latency. Timings include the active MTP execution path. The measurements are throughput probes, not reasoning, retrieval, coding or tool-call correctness evaluations.

The server remains active with F16 K/V and MTP 1; its slot contains the synthetic benchmark conversation.

Reproduce using a fresh output directory:

```powershell
python outputs/run_mtp_benchmark.py --prompt-tokens 102400 --mtp-draft 1 --expected-context 163840 --expected-kv f16 --out-dir outputs/mtp-100k-mtp1-f16-repeat
```

Saved alongside this report: verified launch command, server properties, corpus, manifest, exact requests, raw stream events and response summaries.
