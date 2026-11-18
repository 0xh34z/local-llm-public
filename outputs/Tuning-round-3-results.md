# Tuning results: October 28-29, 2026

The useful improvement is an opt-in private HIP backend that splits the existing 32-column MMA attention tile's KV work across **eight blocks instead of four** for the narrow long-context Q3 shape. It improved every tested matched generation workload in this round without lowering weight or KV precision. This is a targeted launch/work-distribution change, not a new inference engine.

Use **F16 K/V, MTP 2, allocation 163840, one slot, batch 2048, microbatch 512 and eight threads** with the eight-block launcher as the candidate configuration. The original packaged backend and normal launcher are preserved. Approximately 100K actual input is the more comfortable tested working region; neither 128K nor 160K established the user's desired 15 TPS floor. No tested region establishes a universal floor across all tasks.

## Matched normal-mode generation

All rows use the same Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf, unchanged F16 caches and MTP 2. Controls and candidates use the same private DLL. The control column averages the neighboring automatic blocks. Percent gains are calculations from request averages, not fixed-token-stream kernel percentages.

| Actual input tokens | Workload | Automatic control TPS | Eight-block TPS | Observed gain |
|---:|---|---:|---:|---:|
| 103220 | Original continuation, three candidate samples | 14.722 | **16.698** | 13.42% |
| 103256 | Code, one candidate sample | 16.606 | **18.527** | 11.57% |
| 103278 | Open-thinking segment, one candidate sample | 17.147 | **18.720** | 9.17% |
| 128537 | Restored follow-up, one candidate sample | 13.840 | **14.779** | 6.79% |
| 160537 | Restored follow-up, one candidate sample | 12.850 | **15.150** | 17.90% |

The original 103K candidate samples ranged 16.056-17.541 TPS. The code and open-thinking outputs were each 512 tokens; the original continuation used three 256-token samples plus a separate greedy control. Candidate generated tokens and MTP work differ from automatic controls. Corresponding automatic controls reproduced their token sequences and draft counts in all these comparisons.

These rows are not a calibrated TPS-versus-context curve: prompts, generated content, acceptance and timing differ between workloads. In particular, the live 256-token follow-ups measured **15.154 TPS at 128537 input tokens** and **13.631 TPS at 160537**. The initial 512-token generations measured 16.604 and 17.610 TPS respectively. Do not hide these lower follow-ups behind a favorable longer-generation average or pool them with the restored 512-token protocol.

The open-thinking segment remained within reasoning for its measured 512 tokens; it does not measure time to a complete final answer. Forced output lengths, few samples and uncontrolled power/temperature conditions limit all throughput conclusions. No SWE, actual Pi tool cycle, tool-call accuracy or long-context retrieval quality was evaluated.

## Prompt processing and populated context

The logical-batch/prompt-thread sweep found no convincing improvement. The 2048/eight-thread controls drifted from 48.606 to 50.249 seconds for the same 4097 newly evaluated tokens at a restored 103K prefix. That 3.38% drift exceeds any candidate's apparent benefit; retain the defaults.

A separate bracketed microbatch sweep was more stable. Microbatch 256 processed the same new-token work in **49.015 seconds**, versus **50.353 seconds** for pooled 512 controls, a **2.66% reduction**. Microbatch 128 was slower at 52.381 seconds. Microbatch 256 is a modest incremental-prefill option, not an established cold-prefill or combined-generation winner. Keep 512 as the general default; `-MicroBatchSize 256` remains available for an explicit trial.

Extending the saved 103194-token prefix to 128000 actual input tokens required evaluating 24806 new tokens and took **306.675 seconds** of prompt evaluation. Extending it to 160000 required 56806 new tokens and took **820.722 seconds**. These are measurements of extensions from a saved prefix, not completely cold 128K/160K prompts. Large new input remains expensive even with faster generation. No new full cold-100K improvement is claimed.

Allocation stayed 163840. Initial input of 128000 leaves 35840 tokens of space; 160000 leaves only 3840. These tests do not establish the rate or model quality while all remaining space fills. Separate synthetic 128K and 160K snapshots are retained under ignored `work/slot-cache/`; the original 103K snapshot is preserved.

## Numerical and implementation evidence

The eight-block mode passed the sampled independent float64 attention checks for contiguous 103K, strided 103K, strided 160K and contiguous 160K fixtures at the unchanged NMSE threshold of 0.0005. The upstream CPU comparisons still fail on regular-logit fixtures, as the original backend did; their underlying discrepancy and Q1 reference remain unresolved. Those failures were retained, not suppressed or addressed by loosening the gate. Limited operator checks are not model-quality certification.

The private change scales the existing stream-K launch and uses the existing reduction. It applies only on gfx1152 to the F32 Q `[256,3,16,1]`, F16 K/V, D256/two-KV-head, 32-column MMA configuration at padded KV length at least 100000, with a mask and no sinks/ALiBi/softcap. Other shapes retain automatic behavior. Both target verification and draft-state mirroring can use the shape. The tile size and precision remain unchanged. Sixteen blocks passed numerical screening but was slower in full-model generation; two blocks was much slower in isolated screening.

Base is llama.cpp b11888 / `48c9f08b7b25766324282b7605177e2f2be755bb`, ROCm 10 runtime and a private gfx1152 build. Normal graph replay remains enabled; operation profiling is disabled. The original packaged backend and prior diagnostic overlay were hash-verified unchanged. Runtime paths are repository-relative; model weights remain at the existing Desktop path. No driver/global environment/Pi configuration change, commit or publication was performed.

## Start and stop

From this repository, use the optional candidate launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\outputs\Start-KVSplitServer.ps1
```

For an automatic control using the same private DLL, add `-KvSplit auto`. To stop that server while idle:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\outputs\Stop-KVSplitServer.ps1
```

The stop helper refuses to interrupt an active request. The original `Start-ROCmServer.ps1` remains available as the packaged-backend baseline. The candidate launcher keeps the same localhost port and model path and does not automatically restore a synthetic benchmark slot. It uses the local ignored `work/hip-kv-split-bin` runtime and is pinned to this tested build; an unrelated future release needs its own rebuild/validation.

## Reports and raw evidence

- [KV work-splitting screen and 103K comparison](kv-split-100k-results-verified/REPORT.md).
- [Additional code/open-thinking throughput](kv-split-workload-validation/REPORT.md).
- [Logical batch and prompt threads](prefill-settings-100k-results/REPORT.md).
- [Microbatches 128/256/512](microbatch-small-100k-results/REPORT.md).
- [Populated 128K](populated-128k-kv-split-results/REPORT.md).
- [Populated 160K](populated-160k-kv-split-results/REPORT.md).

Each report links manifests, actual process flags, server properties, requests, streamed events, restore responses, results and logs. Failed startup evidence is kept separately from successful comparisons. This completes this bounded tuning round; it does not establish a global hardware ceiling or rule out further kernel, model or real-agent optimization.
