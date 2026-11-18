# Qwen3.6 35B-A3B MTP: measured 100K performance

Measured on the user's running localhost llama-server; not an extrapolation from another GPU.

## Configuration

- Build: b11888, 48c9f08b7; Windows Vulkan, Radeon 860M.
- Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf.
- Context allocation: 163840, one slot; GPU layers 99; Flash Attention on.
- K/V cache: q8_0 / q8_0; batch 2048; microbatch 512; CPU threads 8.
- MTP: draft-mtp, maximum two draft tokens. Verified in process command line, active slot, and returned draft-token statistics.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42; repetition penalty 1, presence penalty 0.

## Results

| Metric | Fresh prompt | Appended follow-up |
|---|---:|---:|
| Input tokens | 102400 | 102939 |
| Reused tokens | 0 | 102911 |
| Newly processed tokens | 102400 | 28 |
| Prompt evaluation | 929.807 s | 1.402 s |
| Prompt evaluation rate | 110.130 tokens/s | 19.970 tokens/s |
| First generated token, client wall clock | 929.850 s | 1.466 s |
| Generated tokens | 512 | 256 |
| Generation evaluation | 52.242 s | 25.742 s |
| Generation evaluation rate | 9.781 tokens/s | 9.906 tokens/s |
| Total client wall time | 982.104 s | 27.219 s |
| Draft tokens accepted / proposed | 308 / 405 | 153 / 202 |
| Truncated | false | false |

Prompt tokens + cached tokens are accounted for independently of output. The follow-up appends all 512 generated token IDs and 27 new wrapper/question tokens; the server reprocesses one previously generated token, hence 28 newly processed tokens.

## Method and interpretation

One cold request followed by one append-only continuation, sent sequentially to /completion. The script generates deterministic synthetic Python modules with unique class/function names and numeric values. It uses the server tokenizer and truncates the corpus to make an exactly 102400-token complete input including manually supplied Qwen chat delimiters. No user repository or personal files are included. Requests and corpus are saved beside this report.

The response prompt requests direct code output using an empty completed thinking block. Output lengths are forced with ignore_eos=true. This is a synthetic throughput probe, not a coding, reasoning, retrieval, or tool-call quality evaluation. Repeated corpus structure and selected expert routing may affect performance. Each request was measured once, with no confidence interval or control for background workload and power/thermal state.

All server timing fields include the active MTP execution path. The generation rate is the server's reported predicted_per_second, using its internal timing convention. Client first-token and total wall times are reported separately. During prefill, progress notifications include a placeholder token ID; these notifications are excluded from generated-token counts and client first-token time. The original cold stream was retained and its summary normalized to exclude those placeholders; the saved runnable script contains the corrected filter.

This demonstrates that this configuration runs with 100K populated tokens and retains its prefix on an append-only follow-up. Fresh ingestion takes about 15.5 minutes, whereas the cached follow-up begins output in about 1.5 seconds. Generation remains about 10 tokens/s in this workload. Context quality at 100K, populated 128K/160K speed, and the causal benefit of MTP versus no MTP remain unmeasured.

The user's earlier llama-bench result (115.58 tokens/s, about 886 s for pp102400 without MTP) used different synthetic inputs and a different execution path. Comparing it with this run does not isolate MTP overhead. The earlier short-context MTP server result of 25.76 tokens/s also does not represent populated-100K speed.

The running server was left active with its original configuration. Its slot now contains the synthetic benchmark conversation.

## Files

- ../run_mtp_benchmark.py: runnable Python standard-library harness.
- manifest.json: corpus hash and measurement metadata.
- server-props.json: server properties captured before the test.
- synthetic-corpus.py.txt: generated source corpus.
- *-request.json: exact native API request payloads, including token IDs.
- *-events.jsonl: raw server events with client elapsed timestamps.
- *-result.json: complete response, generated token IDs, final timings, and client wall times.
