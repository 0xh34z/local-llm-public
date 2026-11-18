# Windows inference and Pi file reads

On the Ryzen AI 7 350 / Radeon 860M, keep the existing eight-block attention launcher for long-context generation. For questions about a known part of a file, Pi can save substantial time by asking its read tool for the relevant lines. The project now includes a prompt template for that workflow.

## Live Pi tool cycles

Pi 1.5.2 ran its installed CLI and built-in read tool against a local b11888 server. Each scenario started a new server and an ephemeral Pi session. The fixture's `delay(attempt)` returns `min(37, 3 ** attempt)`. The larger file adds 600 unrelated constants after the function. Every completed short scenario answered that the cap is 37 and the first attempt reaching it is 4.

| Pi request after the tool returned | Cached input | Newly evaluated | First output | Request finished |
|---|---:|---:|---:|---:|
| Read the two-line file | 936 | 36 | 0.348 s | 1.833 s |
| Read the entire 603-line file | 955 | 7,152 | 20.105 s | 22.083 s |
| Read lines 1–2 of the same file | 964 | 53 | 0.391 s | 1.889 s |
| Use the project command on lines 1–2 | 972 | 53 | 0.394 s | 2.000 s |

The two-line request saved 19.71 seconds to first output after the tool returned in the matched-file comparison. Adding both model requests in each cycle gives 26.43 seconds for the full file and 6.26 seconds for the range. Each row is one observed tool cycle. Full and ranged reads used different instructions, and the model chose its own tool arguments and answer length. The measurement begins at Pi's provider request hook and counts the first nonempty content, reasoning, or tool-call delta. The table excludes server startup, the initial request that decided to call the tool, and filesystem read time. The synthetic task tests lookup of an already identified function.

From Pi in this project, use `/read-range <file> <first-line> <number-of-lines> <question>`. For example, `/read-range outputs/Start-KVSplitServer.ps1 1 12 Which batch sizes does the launcher default to?`. A running Pi session can discover the project template after `/reload` when project resources are enabled. The model can ask for further context when the snippet is insufficient. [Project template](../.pi/prompts/read-range.md)

The harness used an isolated Pi profile, local-only provider, the read tool, thinking off, seed 42, temperature zero and a 384-token output limit. The evidence includes request timings, server logs, CLI events, tool arguments and responses. [Small file](pi-prefix-windows-run/requests.jsonl) · [Full file](pi-full-read-windows/requests.jsonl) · [Range](pi-range-read-windows/requests.jsonl) · [Template and verified answer](pi-template-windows/tool-result.json) · [Pi runner](run_pi_prefix.py)

### The same command with a populated 103K context

A fresh server evaluated **103,349 input tokens** for the initial Pi request, including 102,400 tokens of synthetic project background. The first output arrived after 731.816 seconds (12 minutes 12 seconds). Pi then read lines 1–2 of the 603-line fixture. Its follow-up reused **103,397 tokens**, evaluated **53 new tokens**, produced its first output in **1.034 seconds** and completed the correct answer in **3.597 seconds**. Generation averaged 19.17 tokens/s for those 50 output tokens. Both requests fit without truncation. [Per-request timings](pi-template-103k-windows/requests.jsonl) · [Tool arguments and answer](pi-template-103k-windows/tool-result.json)

This was one live local Pi tool cycle with a synthetic long system prefix and a simple question. The 103K prefix stayed in the running server across the read. The isolated profile used a longer HTTP idle timeout, disabled retries and requested stream progress for the cold pass. The whole-file comparison above used the shorter context. [Server and Pi invocation](pi-template-103k-windows/launch.json) · [Prefix fingerprint](pi-template-103k-windows/context.json)

## Vocabulary projection

The experiment specializes the dense Q8_0 projection from 2,048 hidden features to 248,320 logits for a single input token on gfx1152. Four or eight independent 32-lane waves in a workgroup each compute one row using the existing dot-product and warp reduction. All other shapes fall through to the original dispatch. The head reads 540,344,320 weight bytes per invocation (248,320 rows × 64 blocks × 34 bytes). At the measured control time this corresponds to roughly 78.6 GB/s of weight data. This calculation uses the tensor size and elapsed time. Hardware memory-traffic counters were unavailable in this experiment.

| Projection dispatch | Mean of per-process medians | Change in operation time |
|---|---:|---:|
| Original | 6.872 ms | Reference |
| Four rows per workgroup | 6.856 ms | −0.24% |
| Eight rows per workgroup | 6.827 ms | −0.66% |

The screen alternated seven processes in the order original, four, eight, original, eight, four, original. Each process warmed up for 24 graphs, then timed seven sets of 24 synchronized graph executions. The full-sized tensor repeated 256 seeded weight rows with mixed signed quants and exactly representable scales. Every one of the 248,320 output values matched its independent arithmetic reference in each process. This checks the measured shape and fixture. [Kernel](vocab-projection.cuh) · [Build script](build_vocab_probe.py) · [Screen](screen_vocab_projection.py) · [All timings and checks](vocab-projection-screen-2/samples.json)

## Full-model generation

| Populated input | Original projection | Eight-row projection | Candidate difference |
|---|---:|---:|---:|
| 103,220 tokens | 17.467 tokens/s | 17.291 tokens/s | −1.01% |
| 160,537 tokens | 15.061 tokens/s | 14.902 tokens/s | −1.05% |

These are arithmetic means of request rates. The candidate server was bracketed by two control servers using the same experimental DLL in its original dispatch mode. All three used the already tuned eight-block attention path, F16 K/V, MTP 2, 163,840 allocated tokens, one slot, batch 2,048, microbatch 512 and eight threads. Logs confirm both the attention and projection dispatch. Every measured request restored a synthetic snapshot and generated 256 tokens. The 103K requests reused 103,194 tokens and the 160K requests reused 160,511. Both evaluated 26 new input tokens. Each server first completed a discarded 48-token warmup, then ran 103K, 160K, 160K, 103K in that order.

Generated token hashes and draft acceptance counts matched across all three servers at corresponding trial positions. The two repetitions at each length produced different outputs. The first control was faster than the final control, and the candidate means fell between them at both lengths. With four control samples and two candidate samples per length, the data support keeping the existing configuration. They do not resolve a one-percent effect or establish a sustained 15 tokens/s minimum. Power, clocks and temperature were uncontrolled.

The existing eight-block runtime retained HIP DLL SHA256 `51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5`. The experimental build lives in `work/hip-vocab-probe/bin`. An initial screen failed to discover the GPU. Explicitly loading the private HIP DLL resolved discovery before timing started. [Launch and DLL details](vocab-long-context-windows/manifest.json) · [All twelve generation samples](vocab-long-context-windows/samples.json) · [Generation runner](run_vocab_continuations.py) · [Working launcher](Start-KVSplitServer.ps1)

## Repeating the measurements

The subsequent project extension also supports ordinary questions without a line number. It previews 40 lines by default for the local model and suggests searching if the excerpt ends early. [Automatic lookup measurements](pi-search-after-preview/REPORT.md)

Run from the repository root with fresh results-directory names and the benchmark ports free. The projection build uses the existing private HIP build objects, compiler and Visual Studio environment. The Pi command works with the eight-block server independently of the experimental build.

```powershell
$benchPython = '.\work\rocm10-runtime\Scripts\python.exe'
& $benchPython outputs/run_pi_prefix.py outputs/pi-range-repeat --read template
& $benchPython outputs/run_pi_prefix.py outputs/pi-long-range-repeat --read template --long-prefix
& $benchPython outputs/build_vocab_probe.py
& $benchPython outputs/screen_vocab_projection.py outputs/vocab-screen-repeat
& $benchPython outputs/run_vocab_continuations.py outputs/vocab-generation-repeat outputs/vocab-screen-repeat
```
