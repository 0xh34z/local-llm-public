# Qwen 100K: MTP draft counts 1, 2 and 3

All measurements below used the user's Radeon 860M, Windows Vulkan, llama.cpp b11888/48c9f08b7, and Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. The populated cold input was exactly 102400 tokens. Context allocation was 163840, one slot, GPU layers 99, Flash Attention enabled, Q8 K/V, batch 2048, microbatch 512 and eight CPU threads. Load mode was auto (explicit on counts 1 and 3, default on the earlier count-2 run).

The count-3 process command line was verified before submitting the request and saved in server-launch.json. The complete cold request payload is identical to the count-1 and count-2 payloads, including every prompt token, seed, sampling setting and output limit.

| Metric | MTP 1 | MTP 2 | MTP 3, new |
|---|---:|---:|---:|
| Cold input tokens | 102400 | 102400 | 102400 |
| Cold cache reuse | 0 | 0 | 0 |
| Prefill time | 916.927 s | 929.807 s | 947.643 s |
| Prefill rate | 111.677 tokens/s | 110.130 tokens/s | 108.058 tokens/s |
| First generated token, client wall | 916.973 s | 929.850 s | 947.700 s |
| Initial generation tokens | 512 | 512 | 512 |
| Initial generation rate | 11.632 tokens/s | 9.781 tokens/s | 10.334 tokens/s |
| Initial generation time | 43.929 s | 52.242 s | 49.450 s |
| Cold request total wall | 960.914 s | 982.104 s | 997.162 s |
| Cold accepted/proposed draft tokens | 232/278 | 308/405 | 340/510 |
| Follow-up input tokens | 102939 | 102939 | 102939 |
| Follow-up cached tokens | 102911 | 102911 | 102911 |
| Follow-up newly processed tokens | 28 | 28 | 28 |
| Follow-up prefill time | 1.360 s | 1.402 s | 1.434 s |
| Follow-up first generated token | 1.416 s | 1.466 s | 1.483 s |
| Follow-up generation tokens | 256 | 256 | 256 |
| Follow-up generation rate | 11.021 tokens/s | 9.906 tokens/s | 9.935 tokens/s |
| Follow-up total wall | 24.562 s | 27.219 s | 27.157 s |
| Follow-up accepted/proposed draft tokens | 108/146 | 153/202 | 167/264 |
| Truncated | false on both requests | false on both requests | false on both requests |

Count 1 has the highest measured generation rate in these samples. None of counts 1, 2 or 3 meets the user's 15 tokens/s requirement at 100K. More drafting does not reliably improve speed on this setup and workload. MTP-disabled generation at populated 100K has not yet been measured; the earlier llama-bench prefill run does not provide that missing control.

Method and limits: one fresh request followed by an append-only cached continuation per configuration, measured once each. Input is a deterministic synthetic Python corpus with repeated structure and unique names/numeric values, plus a fixed coding question. Responses use non-thinking prompting, temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repeat penalty 1 and presence penalty 0. Lengths are forced with ignore_eos=true. This is throughput testing, not coding, reasoning, retrieval or tool-call quality evaluation. Generated answers, expert routing and follow-up token sequences may differ across configurations despite identical initial seed and input. Background workload, power and thermal state were not controlled. Small differences do not establish statistical superiority.

The server remains active with MTP 3 and the synthetic benchmark conversation in its slot.

Reproduce with a fresh output directory:

```powershell
python outputs/run_mtp_benchmark.py --prompt-tokens 102400 --mtp-draft 3 --expected-context 163840 --out-dir outputs/mtp-100k-mtp3-repeat
```

The exact launch command, server properties, requests, corpus, manifest, raw timing stream and response summaries are saved alongside this report.
