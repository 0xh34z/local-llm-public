# Autonomous tuning round: 2026-10-28

The user authorized proceeding through useful tests without further intervention. No production kernel replacement, driver/global setting change, commit or publication is implied by a promising screening result.

Planned sequence:

1. Keep 32-column MMA; test half/double/quad stream-K block counts for the narrow gfx1152 Q3/D256/F16 shape. Numerical checks against independent float64, then repeated isolated operation timing. Check the two best qualifying candidates in bracketed normal-mode generation at a restored 103K prefix.
2. Screen logical batch size and prompt thread count with matched 4096-token suffixes at the same restored 103K prefix. Hold generation threads, microbatch, MTP, precision and allocated context fixed. These tests are not full cold-prefill results. Confirm a material candidate before changing defaults; only claim a cold-prefill gain after a matched full cold test.
3. If practical to continue, measure actual populated 160K inference with the baseline or a qualified candidate. This is a stress measurement, not merely allocation of a 160K window. Record how much prefix was reused and the remaining generation budget.

The existing 100K numerical fixture CPU-reference discrepancy remains unresolved; preserve failure logs. Any numerical gate is limited to the captured test cases, not a model quality guarantee. Keep original benchmarks and original backend hashes.

Completed: KV split screening, bracketed 103K generation, extra numerical layouts, logical-batch/prompt-thread sweep, microbatches 128/256 versus 512, additional code/open-thinking throughput prompts, and populated 160K inference. A populated-128K comparison was added after a 160K candidate follow-up fell below 15 TPS. Eight blocks is useful but neither larger context established the desired floor. See `Tuning-round-3-results.md` for consolidated findings and limits. No new full cold-prefill claim or production default replacement was made. All benchmark servers were stopped.
