# Qwen 100K: MTP 1 versus previous MTP 2

Measured on the user's running Windows Vulkan llama-server, Radeon 860M, b11888/48c9f08b7. Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf.

The live process launch arguments were checked before the MTP-1 run and saved in server-launch.json. The harness verifies the draft count and context allocation before sending the benchmark. Both configurations use context allocation 163840, one slot, GPU layers 99, Flash Attention on, Q8 K/V, batch 2048, microbatch 512, eight CPU threads, and automatic load mode (explicit on MTP 1, default on the earlier MTP 2).

| Metric | MTP 1, new run | MTP 2, earlier run |
|---|---:|---:|
| Fresh input tokens | 102400 | 102400 |
| Fresh cached tokens | 0 | 0 |
| Prefill time | 916.927 s | 929.807 s |
| Prefill rate | 111.677 tokens/s | 110.130 tokens/s |
| Client first generated token | 916.973 s | 929.850 s |
| Generated tokens | 512 | 512 |
| Generation time | 43.929 s | 52.242 s |
| Generation rate | 11.632 tokens/s | 9.781 tokens/s |
| Total client wall time | 960.914 s | 982.104 s |
| Draft tokens accepted/proposed | 232/278 | 308/405 |
| Cached follow-up input tokens | 102939 | 102939 |
| Follow-up reused tokens | 102911 | 102911 |
| Follow-up newly processed tokens | 28 | 28 |
| Follow-up prefill time | 1.360 s | 1.402 s |
| Follow-up first generated token | 1.416 s | 1.466 s |
| Follow-up generated tokens | 256 | 256 |
| Follow-up generation rate | 11.021 tokens/s | 9.906 tokens/s |
| Follow-up total client wall time | 24.562 s | 27.219 s |
| Follow-up draft accepted/proposed | 108/146 | 153/202 |
| Truncated | false on both requests | false on both requests |

The complete cold API payload is identical to the earlier MTP-2 payload, including all token IDs, sampling settings, seed and output limit. Both runs use the same deterministic synthetic Python corpus and question with a completed empty thinking block. Temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0. Output lengths are forced using ignore_eos=true. Streaming prefill progress placeholder tokens are excluded from generated tokens and first-token time.

Each configuration was measured once. This is a matched-input, matched-settings comparison, with no control for background workload, power or thermal conditions, and no confidence interval. Speculative decoding can produce different sampled responses even with the same seed; follow-up prompts contain each run's generated response and are therefore different token sequences. These are non-thinking synthetic throughput measurements, not reasoning, coding, retrieval or tool-call quality scores.

MTP 1 was about 19% faster for the initial generation and about 11% faster for follow-up generation in these samples. Neither passes the user's 15 tokens/s requirement at 100K. Fresh ingestion remains the dominant cost: reducing the 512-token generation time by about eight seconds makes only a small difference to a roughly sixteen-minute fresh request. Prefix reuse avoids almost all repeated ingestion but does not remove the long-context generation cost.

The earlier 60K measurement used MTP 1 and context allocation 64000. It is not an MTP-2 measurement and cannot isolate context-depth effects when compared directly with the original 100K MTP-2 run.

The server remains active with its MTP-1 configuration and the synthetic benchmark conversation in its slot.

Reproduce with a fresh output directory:

```powershell
python outputs/run_mtp_benchmark.py --prompt-tokens 102400 --mtp-draft 1 --expected-context 163840 --out-dir outputs/mtp-100k-mtp1-repeat
```

Saved alongside this report: verified server launch, server properties, corpus, manifest, exact requests, stream events and result summaries.
