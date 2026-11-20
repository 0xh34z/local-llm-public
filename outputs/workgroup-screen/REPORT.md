# Numerical screen for the attention workgroup experiment

Both GPU workgroup sizes passed the independent float64 comparison on each sampled layout and context. The threshold was normalized mean squared error (NMSE) of 0.0005. All four GPU/CPU input arrays matched within every probe process. Separate probe processes generated different inputs, including different query hashes for 128 and 256 threads. Each GPU output was therefore compared to its own reference. [All checks and dispatch logs](summary.json)

| Fixture | GPU NMSE, 256 threads | GPU NMSE, 128 threads |
|---|---:|---:|
| 103,424 keys, contiguous | 0.00007391 | 0.00013408 |
| 103,424 keys, strided | 0.00006496 | 0.00015142 |
| 160,000 keys, contiguous | 0.00010787 | 0.00020257 |
| 160,000 keys, strided | 0.00011114 | 0.00020514 |

The checked operation used three queries, dimension 256, 16 query heads, two K/V heads, F32 queries and F16 K/V. The reference reads exported data, computes the dot products, softmax and reduction in float64, and compares both GPU and upstream CPU outputs. The upstream test returned exit code 1 for all eight launches. CPU NMSE against float64 ranged from 0.002142 to 0.003600 on those fixtures. This evidence covers the regular-logit Q3 inputs. Q1 and model-level output quality require their own checks. [Reference implementation](../check_attention_float64.py)

Runtime markers recorded eight stream-K blocks, two output tiles, 64 KV rows per iteration and 32 columns for the scoped kernel. The 128-thread branch used four waves and 34,368 bytes of dynamic shared memory. The 256-thread branch used eight waves and 34,816 bytes. A short concurrent CPU compile overlapped six of the initial 18 operator timings (November 19 at 00:18:09.998 through 00:18:20.771 local). Initial timing data are retained in this directory. The model trial and the repeat operator pass use separate results directories. [Initial samples](summary.json) · [Build evidence](../attention-workgroup-round/Build.md)
