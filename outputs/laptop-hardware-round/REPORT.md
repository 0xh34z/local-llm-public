# Windows inference speed on the Framework Laptop 13

Increasing the microbatch from 512 to 1024 saved **31 seconds before the first token** when the server had to read a fresh 102,400-token prompt. For generation from a populated context, the existing eight-block attention kernel was fastest in every comparison. The model was Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL on the integrated Radeon 860M, with F16 K/V and MTP 2 throughout.

## Reading a cold prompt

Four fresh server processes read the same synthetic code prompt in 512, 1024, 1024, 512 microbatch order. Average time from request submission to first output fell from **706.33 to 675.27 seconds**. All four completions contained identical 512 token IDs and accepted 305 of 410 proposed draft tokens. Generation averaged 17.78 and 17.73 tokens/s respectively. The timing starts after the weights have loaded. [Raw results and sensor observations](../cold-microbatch-abba-100k/REPORT.md)

For an anticipated fresh 100K prompt, start the server with the existing launcher and `-MicroBatchSize 1024`. For day-to-day Pi use, 512 remains the general setting. An earlier restored-prefix test with 8,193 new tokens measured 99.39 seconds at both settings. A real Pi follow-up that reused 103,397 tokens and evaluated 53 new ones reached its first output in 1.03 seconds at 512. [Launcher](../Start-KVSplitServer.ps1) · [Prefix reuse](../Cold-prefill-and-reuse.md) · [Pi tool-cycle measurement](../Windows-speed-round.md)

## Generating with the context populated

Seven fresh processes ran the split counts in the order 8, 6, 8, 10, 8, 12, 8. The changes below compare each candidate with its two neighboring controls. The eight-block reference rates pool the time for all eight sampled continuations at each length. The 163,840-token allocation, batch 2048, microbatch 512, eight CPU threads, model and cache precision stayed fixed.

| Actual input | Eight blocks, pooled tokens/s | Range across eight requests | Six blocks | Ten blocks | Twelve blocks |
|---|---:|---:|---:|---:|---:|
| 103,220 | 17.14 | 16.69–17.72 | −12.1% | −10.6% | −10.7% |
| 160,537 | 14.78 | 14.13–15.42 | −8.7% | −15.5% | −17.9% |

Each sampled request restored its slot, evaluated 26 new prompt tokens and returned 256 tokens. Sampled completions can exercise different expert and draft work. On a separate repetitive 160K prompt, all modes produced identical 128-token sequences and draft work. Six, ten and twelve blocks were respectively 5.0%, 13.2% and 15.7% slower on that comparison. The sixteen numerical fixtures passed the independent float64 reference on the scoped Q3 shape. The existing disagreement with the upstream CPU reference persists. [Full-model report](../fine-split-generation/REPORT.md) · [Sampled requests](../fine-split-generation/samples.json) · [Numerical and operator evidence](../fine-split-screen/summary.json)

## Plugged-in Windows power mode

On the established eight-block runtime with microbatch 1024, the Windows **Best performance** AC setting improved generation by 3–4% on both tested continuations. Four fresh servers ran in Balanced, Best performance, Best performance, Balanced order. At each length, all four generated identical 256-token outputs with identical draft counts. The table pools generation time for two requests per cell. [Timings and token hashes](../ac-mode-100k-160k/samples.json) · [Matched comparison](../ac-mode-100k-160k/comparison.json)

| Populated input | Balanced | Best performance | Change |
|---|---:|---:|---:|
| 103,220 tokens | 17.43 tokens/s | 18.03 tokens/s | +3.4% |
| 160,537 tokens | 14.26 tokens/s | 14.81 tokens/s | +3.9% |

Across four sensor samples per mode during these short generations, AMD reported 24–25 W and 64–72 °C on Balanced, versus 34–36 W and 77–83 °C on Best performance. Sampling every 15 seconds gives a coarse view. Wall-socket power and sustained thermal behavior remain unmeasured. Both 160K Best performance trials fell below 15 tokens/s. The benchmark restored the original Balanced AC setting. [Sensor measurements and rates](../ac-mode-100k-160k/comparison.json) · [Restored setting](../ac-mode-100k-160k/restored.json) · [Windows AC power modes](https://learn.microsoft.com/en-us/windows/win32/api/powrprof/nf-powrprof-powersetuserconfiguredacpowermode)

## How close is this to the hardware limit?

The best of four large GPU memory-read configurations sustained **81.98 GB/s** by device timing and **81.7 GB/s** by host timing. Each timed batch validated its checksum. An earlier benchmark of the model's single-token vocabulary projection processed **78.63 GB/s** of logical weight bytes, about **96%** of that 81.98 GB/s reference. This makes that particular operation a poor target for large gains through memory-read tuning alone. The comparison uses tensor sizes and elapsed times and has no DRAM transaction counters. [Bandwidth probe](../stream-bandwidth-gfx1152/REPORT.md) · [Projection measurement](../Windows-speed-round.md)

The eight-block attention operator reached 3.561 ms for 103,424 cached positions and 5.691 ms for 160,000 in isolated strided fixtures. Counting each unique F16 key and value byte once gives 59.48 and 57.58 GB/s. Attention also does arithmetic, reads masks and reduces partial results. Those rates measure a different workload and cannot be converted into a percentage speedup available to Pi. [Operator samples](../fine-split-screen/summary.json)

At 100K, the existing generation path cleared the 15 tokens/s target in these sampled continuations. At 160K, several requests fell below it. These measurements bound expectations for this model, workload and runtime. They do not yield a maximum tokens/s for the laptop. [System configuration and telemetry](Telemetry.md)
