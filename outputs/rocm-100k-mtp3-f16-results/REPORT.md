# ROCm MTP 3 versus MTP 2 at populated 100K context

Measured on the user's Radeon 860M / Ryzen AI 7 350, Windows, llama.cpp b11888 (48c9f08b7), Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL. Both configurations use ROCm0, 163840 allocated context, one slot, 99 offloaded layers, Flash Attention, F16 K/V, batch 2048, microbatch 512, eight threads and auto load mode.

| Metric | MTP 3, new | MTP 2, earlier |
|---|---:|---:|
| Cold input tokens | 102400 | 102400 |
| Prompt evaluation seconds | 758.600 | 807.994 |
| Prompt tokens/s | 134.986 | 126.734 |
| Cold client TTFT seconds | 758.684 | 808.047 |
| Initial generation tokens/s | 15.684 | 15.744 |
| Initial generated tokens | 512 | 512 |
| Cold client wall seconds | 791.281 | 840.517 |
| Initial accepted / proposed draft tokens | 339 / 513 | 314 / 392 |
| Follow-up reused input tokens | 102911 | 102911 |
| Follow-up newly processed tokens | 28 | 28 |
| Follow-up TTFT seconds | 0.709 | 0.689 |
| Follow-up generation tokens/s | 14.295 | 14.975 |
| Follow-up generated tokens | 256 | 256 |
| Follow-up client wall seconds | 18.561 | 17.731 |
| Follow-up accepted / proposed draft tokens | 161 / 279 | 152 / 205 |
| Truncation | false on both requests | false on both requests |

The cold request JSON hashes match exactly (SHA256 572a0e822a86213d60e076b486c7f72a22d8040a322d69605b487130c63c31cd). Initial generation is essentially tied; MTP 3's follow-up is about 4.5% slower. MTP 3's cold prompt evaluation was about 6.1% faster in this run, shortening the cold request, but runs were taken at different times without power, temperature or background-workload controls. This does not establish that the draft setting caused the prefill difference.

Keep MTP 2 as the generation baseline. Neither configuration demonstrates a sustained 15 tokens/s floor. Single runs, stochastic output, repetitive synthetic input, forced output lengths and completed empty thinking blocks limit generalization. Follow-up input includes each run's own generated output, so those continuations are not identical token sequences. No coding, reasoning, retrieval or tool-call quality was evaluated.

The server launch and reported context were verified before submission. Raw requests, results, SSE timing events and server log are saved in this directory. The benchmark script is ../run_mtp_benchmark.py.
