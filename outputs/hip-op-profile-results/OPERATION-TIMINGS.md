# HIP operation-timing diagnostic

These are synchronized operation-group GPU timings from a private diagnostic build. They are not individual-kernel hardware counters or fractions of production request wall time.

The diagnostic disables graph replay and concurrent-region scheduling but retains ordinary operation fusions. A group containing multiple operations is assigned as a whole to its category. Only graphs containing Flash Attention at >=100K KV length and <=3 query tokens are sampled; other graphs and CPU work are outside this summary.

Timer-quality warning: 2908 of 43161 records have invalid negative GPU durations (6.74%). The minimum is -0.429170 ms. Negative records are retained in the raw log and excluded from GPU totals/means below. Their absolute sum does not bound all event-timer error. Treat these as provisional rankings.

## GDN absent (draft-like, inferred)

Context `0000017932AE6700`: 85 graphs, 2914 timed groups, query lengths [1, 3], KV lengths [103424].

| Category | Recorded GPU ms | Share of this context |
|---|---:|---:|
| dense/vocabulary matmul | 469.134 | 52.43% |
| attention | 295.507 | 33.03% |
| other groups | 89.910 | 10.05% |
| expert/shared groups | 40.223 | 4.50% |

| Top operation group | Source 0 type / shape | Source 1 shape | Calls | Invalid timings | Mean valid GPU ms | Mean synchronized wall ms | Share |
|---|---|---|---:|---:|---:|---:|---:|
| MUL_MAT | q8_0 / 2048,248320,1,1 | 2048,1,1,1 | 56 | 0 | 6.828 | 6.897 | 42.73% |
| FLASH_ATTN_EXT | f32 / 256,1,16,1 | 256,103424,2,1 | 57 | 0 | 2.746 | 2.808 | 17.50% |
| FLASH_ATTN_EXT | f32 / 256,3,16,1 | 256,103424,2,1 | 28 | 0 | 4.963 | 5.027 | 15.53% |
| MUL_MAT | q8_0 / 2048,8192,1,1 | 2048,1,1,1 | 57 | 0 | 0.290 | 0.353 | 1.85% |
| RMS_NORM,MUL | f32 / 2048,3,1,1 | 0,0,0,0 | 140 | 14 | 0.113 | 0.127 | 1.60% |
| MUL_MAT_ID,MUL_MAT_ID,GLU | q4_K / 2048,512,256,1 | 2048,1,3,1 | 28 | 0 | 0.449 | 0.516 | 1.41% |
| RMS_NORM,MUL | f32 / 2048,1,1,1 | 0,0,0,0 | 285 | 41 | 0.048 | 0.126 | 1.31% |
| MUL_MAT_ID,MUL_MAT_ID,GLU | q4_K / 2048,512,256,1 | 2048,1,1,1 | 57 | 0 | 0.182 | 0.255 | 1.16% |
| MUL_MAT | q8_0 / 4096,2048,1,1 | 4096,1,1,1 | 57 | 0 | 0.178 | 0.247 | 1.13% |
| MUL_MAT | q8_0 / 4096,2048,1,1 | 4096,3,1,1 | 56 | 0 | 0.176 | 0.251 | 1.10% |

## GDN present (target-like, inferred)

Context `0000017360E94020`: 29 graphs, 40247 timed groups, query lengths [1, 3], KV lengths [103424].

| Category | Recorded GPU ms | Share of this context |
|---|---:|---:|
| dense/vocabulary matmul | 1452.445 | 29.04% |
| attention | 1424.643 | 28.48% |
| other groups | 1054.391 | 21.08% |
| expert/shared groups | 908.365 | 18.16% |
| recurrent groups | 162.067 | 3.24% |

| Top operation group | Source 0 type / shape | Source 1 shape | Calls | Invalid timings | Mean valid GPU ms | Mean synchronized wall ms | Share |
|---|---|---|---:|---:|---:|---:|---:|
| FLASH_ATTN_EXT | f32 / 256,3,16,1 | 256,103424,2,1 | 280 | 0 | 4.991 | 5.065 | 27.94% |
| MUL_MAT_ID,MUL_MAT_ID,GLU | q4_K / 2048,512,256,1 | 2048,1,3,1 | 1092 | 0 | 0.457 | 0.529 | 9.99% |
| MUL_MAT | q8_0 / 2048,8192,1,1 | 2048,3,1,1 | 1120 | 0 | 0.341 | 0.421 | 7.64% |
| MUL_MAT_ID | q5_K / 512,2048,256,1 | 512,8,3,1 | 1036 | 0 | 0.341 | 0.422 | 7.07% |
| MUL_MAT | q8_0 / 4096,2048,1,1 | 4096,3,1,1 | 1120 | 0 | 0.187 | 0.263 | 4.18% |
| MUL_MAT | q8_0 / 2048,248320,1,1 | 2048,3,1,1 | 28 | 0 | 7.066 | 7.136 | 3.96% |
| MUL_MAT | q8_0 / 2048,4096,1,1 | 2048,3,1,1 | 840 | 1 | 0.217 | 0.304 | 3.64% |
| MUL_MAT | q8_0 / 2048,512,1,1 | 2048,3,1,1 | 2800 | 10 | 0.062 | 0.130 | 3.48% |
| GATED_DELTA_NET,VIEW,VIEW,CPY | f32 / 128,16,3,1 | 128,16,3,1 | 840 | 1 | 0.189 | 0.281 | 3.18% |
| CONCAT | f32 / 3,8192,1,1 | 3,8192,1,1 | 840 | 1 | 0.162 | 0.246 | 2.72% |

## Interpretation limits

Device-event waits and logging change scheduling, clock behavior and CPU/GPU overlap. Kernel launch/host overhead is not isolated by an operation timer. Small operations can be distorted by event overhead. Context labels above are inferences, and their percentages must not be summed as if each were a fraction of the whole request. This identifies candidates for further profiling and isolated benchmarks; it does not establish bandwidth saturation, occupancy limits, register spills or a speedup achievable by rewriting a kernel.
