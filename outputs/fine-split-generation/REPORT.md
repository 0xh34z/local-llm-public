# Attention work splitting at 103K and 160K

Keep the existing eight-block attention setting on the Radeon 860M. It led all three alternatives at both prompt lengths. Across four independent control launches it averaged **17.14 generation tokens/s at 103,220 input tokens** and **14.78 at 160,537** on sampled code. Individual control runs ranged from 16.69 to 17.72 and 14.13 to 15.42 respectively. The 160K measurements crossed the user's 15 tokens/s target in both directions.

## Restored code continuations

Seven fresh servers ran in the order 8, 6, 8, 10, 8, 12, 8 blocks. Each candidate generated two 256-token continuations per context. The adjacent control servers produced four comparison samples at each context. The table pools the two candidate times and four control times separately. Rates use the server's convention, which attributes the first output token to prompt evaluation. Every timed request restored 103,194 or 160,511 tokens and evaluated 26 additional input tokens.

| Blocks | 103K candidate | 103K control | Change | 160K candidate | 160K control | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 15.18 | 17.28 | −12.15% | 13.57 | 14.86 | −8.69% |
| 10 | 15.42 | 17.24 | −10.56% | 12.57 | 14.88 | −15.52% |
| 12 | 15.19 | 17.01 | −10.65% | 12.07 | 14.70 | −17.92% |

The same trial position reproduced its token IDs and aggregate draft/accept counts across all four eight-block servers. Sampled code diverged between block counts, changing speculative work. Elapsed time across complete sampled continuations measures the speed seen by a user on these prompts. [Raw generation samples](samples.json) · [Bracketed and prefix comparisons](comparison.json)

## Matched-work continuation

Every server generated the same 128-token greedy four-color sequence at both lengths. At 160,743 input tokens, each candidate and both adjacent controls reproduced the full sequence and the cumulative MTP draft/accept schedule at every token, finishing with 85 drafted and 84 accepted. The corresponding eight-block controls beat six blocks by the smallest margin. This directly tests the effect of block count with identical visible output and observed speculative work.

| Blocks | 160K candidate, tok/s | 160K adjacent controls, tok/s | Change |
|---:|---:|---:|---:|
| 6 | 16.76 | 17.63 | −4.97% |
| 10 | 15.39 | 17.73 | −13.23% |
| 12 | 15.00 | 17.80 | −15.73% |

At 103,426 input tokens the greedy outputs also matched, with some differences in the timing of accepted drafts across the 128 tokens. The full-sequence rates were 5.72%, 11.00% and 11.95% below the respective brackets for 6, 10 and 12 blocks. With eight blocks the deliberately predictable sequence averaged 20.80 tok/s at 103K and 17.71 tok/s at 160K. These rates show how much throughput depends on what the model is writing.

## Fixed-work attention

At both tested sequence lengths, eight GPU work blocks gave the shortest median time for one attention operation. The table shows median milliseconds over three separate process launches. Each launch timed repeated operations on the same tensor shape. The 103K test used 103,424 keys per head and the 160K test used 160,000. Both used the strided layout, three queries, 16 query heads, two K/V heads, dimension 256 and F16 K/V.

| GPU work blocks | 103K, ms | 160K, ms | Largest GPU NMSE across four fixtures |
|---:|---:|---:|---:|
| 6 | 4.017 | 6.425 | 0.0001570 |
| 8 | 3.561 | 5.691 | 0.0001081 |
| 10 | 5.144 | 8.090 | 0.0000881 |
| 12 | 5.396 | 8.636 | 0.0000729 |

Each mode passed the independent float64 reference on regular-logit contiguous and strided fixtures at both sizes. Inputs matched between GPU and CPU within each fixture. The threshold was 0.0005 normalized mean squared error. The upstream CPU comparison failed on all 16 launches. CPU NMSE against float64 ranged from 0.002175 to 0.003600. The comparison covers the listed Q3 synthetic inputs. Q1 remains outside the independent reference used here. [Numerical results and individual timing samples](../fine-split-screen/summary.json)

A calculation from the eight-block medians gives 59.48 GB/s and 57.58 GB/s of unique K/V tensor payload. Each key position contains two heads × 256 values × two F16 bytes × two tensors. The calculation counts each value once and excludes masks, intermediate reductions and rereads. Measuring physical DRAM traffic and attributing it to this operation requires hardware counters.

## Method and reproduction

All four modes used the same isolated private DLL, SHA-256 `b3f79a5fa02a5dc65e95b8c36b91867fd213c41125af731d98268b7fcf0ae9d5`. The experiment kept 32-column MMA dispatch and changed the number of stream-K work blocks for the targeted attention shape. The eight-block control uses environment value `double`. Each probe and benchmark server recorded its dispatch. [Build script](../build_fine_kv.py)

Each generation server ran one slot on ROCm0, F16 K/V, MTP depth two, 163,840 allocated context, logical batch 2,048, microbatch 512 and eight threads for generation and prompt processing. The runner checked the live server command line and model, verified the restored and evaluated token counts, and collected token IDs, MTP counters and per-token times. Each server ran a 48-token warmup before the timed requests. Sampling used temperature 0.6, top-k 20, top-p 0.95 and seed 42. A separate greedy continuation appended a repeated four-color sequence to encourage identical tokens and speculative work.

From the repository root, choose new output directory names for a rerun.

```powershell
& .\work\rocm10-runtime\Scripts\python.exe outputs/run_fine_split.py screen outputs/fine-split-screen-new
& .\work\rocm10-runtime\Scripts\python.exe outputs/run_fine_split.py generate outputs/fine-split-generation-new --screen outputs/fine-split-screen-new
& .\work\rocm10-runtime\Scripts\python.exe outputs/analyze_fine_split.py outputs/fine-split-generation-new
```
