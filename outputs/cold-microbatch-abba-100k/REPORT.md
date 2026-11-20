# Cold prefill at 102,400 tokens on Windows

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

Microbatch 1024 cut mean time to first output by **31.07 seconds** on the Radeon 860M, from 706.33 to 675.27 seconds. Both 1024 trials beat both 512 controls. Use 1024 for large, uncached prompts on the tested eight-block backend. Generation averaged 17.73 tokens/s at 1024 and 17.78 at 512. [Per-run timings](samples.json)

| Run order | Microbatch | Prompt seconds | Prompt tokens/s | First token seconds | Generation tokens/s | Total seconds |
|---|---:|---:|---:|---:|---:|---:|
| [1](case-1-512/cold-result.json) | 512 | 708.253 | 144.581 | 708.321 | 17.801 | 737.040 |
| [2](case-2-1024/cold-result.json) | 1024 | 675.266 | 151.644 | 675.311 | 17.710 | 704.177 |
| [3](case-3-1024/cold-result.json) | 1024 | 675.176 | 151.664 | 675.221 | 17.748 | 704.025 |
| [4](case-4-512/cold-result.json) | 512 | 704.304 | 145.392 | 704.347 | 17.767 | 733.119 |

Mean server prompt evaluation fell from 706.278 to 675.221 seconds (4.40%). Mean client request time, including 512 generated tokens, fell from 735.080 to 704.101 seconds (4.21%). All four responses contained the same 512 token IDs, and MTP accepted 305 of 410 draft tokens in every run. The 512 controls differed by 3.949 seconds of prefill, and the 1024 repeats differed by 0.090 seconds. [Request and backend hashes](manifest.json)

## Conditions and scope

Each run started a fresh server, waited for readiness, and submitted the same tokenized synthetic code request with seed 42 and prompt caching disabled. Each final response reported 102,400 prompt tokens, zero cache hits and no truncation. Client timing began at request submission and excludes model loading. [Original request](../rocm-100k-mtp2-f16-results/cold-102400-mtp2-request.json) · [Runner](../run_cold_microbatch_bracket.py)

Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL ran on ROCm0 with 163,840 context allocated, F16 K/V, one slot, Flash Attention, GPU layers 99, logical batch 2048, eight main/draft generation and batch threads, and MTP depth 2. The process arguments differed only in `-ub`. Each server log confirmed the eight-block dispatch. [512 process arguments](case-1-512/server-launch.json) · [1024 process arguments](case-2-1024/server-launch.json)

The experiment used two trials per setting and one prompt. Temperature, clock speed and background load were uncontrolled. An early spot-check recorded AC online, 100% charge and the Windows Balanced plan. The observed speedup applies to full cold prefill. Cached Pi follow-ups have separate latency measurements, and performance at other populated contexts requires direct measurement. [Power sample](power-sample-first.json) · [Cached Pi measurement](../Windows-speed-round.md)

## GPU readings during prefill

AMD's driver reported 99.4–99.7% mean GPU utilization and 24–25 W of GPU power across sampled portions of the four runs. Each table cell shows the arithmetic mean followed by the sampled minimum and maximum. Sampling was every 15 seconds. The driver supplied no board-power reading. [Raw sensor series](../laptop-hardware-round/gpu-sensors.csv)

| Run | Samples | GPU °C | GPU clock MHz | Utilization % | GPU power W |
|---|---:|---:|---:|---:|---:|
| 1 · 512 | 30 | 70.1 (67–74) | 2697 (2479–2844) | 99.4 (96–100) | 24.50 (24–25) |
| 2 · 1024 | 45 | 71.3 (68–74) | 2706 (2437–2877) | 99.5 (94–100) | 24.67 (24–25) |
| 3 · 1024 | 45 | 71.0 (65–73) | 2692 (2437–2854) | 99.5 (97–100) | 24.67 (24–25) |
| 4 · 512 | 47 | 71.9 (70–75) | 2713 (2490–2864) | 99.7 (97–100) | 24.62 (24–25) |

The observer started **272.4 seconds into run 1**. In runs 2–4 the first and last samples fell within 13 seconds of the prefill boundaries. To align the clocks, the final stream event's elapsed time was subtracted from its log's UTC last-write time. This request-start estimate agreed with the request file's timestamp within 11 milliseconds in all four runs. Each prefill window ends at the first generated token and begins one server-reported prefill duration earlier. Temperature, utilization and clock readings describe sampled conditions. They do not identify the hardware bottleneck. [UTC windows, coverage and statistics](prefill-gpu-sensors.json)
