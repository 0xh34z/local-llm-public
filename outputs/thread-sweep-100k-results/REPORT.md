# CPU thread sweep at a restored 100K prefix

Keep the existing eight-thread default. No convincing generation-speed gain was found.

The highest grouped mean exceeds the pooled eight-thread baseline by 0.80%. The eight-thread control changed from 14.265 to 13.904 TPS (-2.53%) between the first and final blocks.

| Main / draft threads | Measured runs | Mean generation TPS | SD | Range TPS | Mean TTFT s |
|---|---:|---:|---:|---:|---:|
| 1 / 1 | 3 | 14.143 | 0.459 | 13.764–14.653 | 0.596 |
| 2 / 2 | 3 | 14.197 | 0.640 | 13.705–14.921 | 0.620 |
| 4 / 4 | 3 | 14.033 | 0.491 | 13.693–14.595 | 0.633 |
| 6 / 6 | 3 | 14.151 | 0.688 | 13.607–14.925 | 0.619 |
| 8 / 8 | 6 | 14.085 | 0.521 | 13.422–14.811 | 0.633 |

Main and MTP-draft generation thread counts change together. Main and draft batch threads remain fixed at 8. All runs use ROCm0, F16 K/V, MTP 2, context 163840, one slot, batch 2048, microbatch 512, eight batch threads and log verbosity 3.

Each server receives a discarded 64-token warmup, then three measured 256-token continuations. The same original saved slot is restored before every request. Requests use identical input tokens, temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repeat penalty 1, presence penalty 0, completed empty thinking blocks, and ignore_eos=true. Order: 8, 2, 6, 1, 4, 8. The second 8-thread block brackets time drift. Warmups and reload times are excluded from this table.

No affinity pinning, power-limit or temperature control was applied. Repeats within a server are not independent sessions or diverse coding tasks. Reported range is a range of request averages, not an instantaneous minimum. This is a synthetic generation test at a populated prefix, not a cold-prefill or coding/tool-use quality benchmark. Batch-thread optimization is not tested.

Each case directory contains verified live launch arguments, properties, the server log, restore responses, raw requests, streamed events and complete results. See summary.json for per-case numbers and output token hashes. The snapshot is benchmark state, not a user Pi conversation.

## Matched-work verification

All measured requests share one SHA256 hash: a84687703f85aa775ba0a36d59c4fb6072110775967518481bc5f43e87f11283. Corresponding repetition numbers across all six server runs produced identical output token sequences: True. Corresponding draft proposal/acceptance counts also matched: True. The three repetitions produce three different output streams; the SD in the table therefore includes content/acceptance differences, not just timing noise. Comparisons at the same repetition number are paired by output stream.

Every measured request has 103220 input tokens, reuses 103194 tokens, processes 26 tokens, and generates 256 tokens without truncation. 0 of the 18 measured request averages reached 15 TPS. These are a different continuation from earlier 15–16 TPS samples.

Prompt and draft batch threads remained at eight, so this does not evaluate batch-thread counts or full cold-prefill speed. Main and draft generation workers were swept together; their independent contributions and CPU affinity were not tested. CPU utilization and performance with concurrent tools were not measured. The server was shut down after this run.
