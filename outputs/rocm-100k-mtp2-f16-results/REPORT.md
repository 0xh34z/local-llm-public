# ROCm/HIP MTP 2 at populated 100K context

Radeon 860M (gfx1152), Windows, llama.cpp b11888/48c9f08b7, Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. Configuration: explicitly selected ROCm0, context allocation 163840, one slot, GPU layers 99, Flash Attention on, F16 K/V, batch 2048, microbatch 512, eight CPU threads, auto load mode, draft-mtp with maximum two draft tokens. ROCm 10.0.0 runtime libraries and gfx1152 device package were installed in an isolated workspace virtual environment. No display driver or system PATH changes were made.

The live launch arguments and context were verified before submission. Cold API payloads match the prior HIP MTP-1 and Vulkan F16 MTP-1 requests exactly, including all prompt token IDs, sampling settings and output lengths. All configurations below use F16 K/V; the original Q8 measurements are not mixed into this backend comparison.

| Metric | HIP MTP 2, new | HIP MTP 1 | Vulkan MTP 1 |
|---|---:|---:|---:|
| Cold input tokens | 102400 | 102400 | 102400 |
| Cold cached tokens | 0 | 0 | 0 |
| Prompt evaluation seconds | 807.994 | 835.951 | 1109.878 |
| Prompt evaluation tokens/s | 126.734 | 122.495 | 92.262 |
| Client first-token seconds | 808.047 | 836.070 | 1109.918 |
| Initial output tokens | 512 | 512 | 512 |
| Initial generation seconds | 32.456 | 36.852 | 41.290 |
| Initial generation tokens/s | 15.744 | 13.866 | 12.376 |
| Total cold client wall seconds | 840.517 | 872.937 | 1151.220 |
| Initial accepted/proposed draft tokens | 314/392 | 241/269 | 238/273 |
| Follow-up input tokens | 102939 | 102939 | 102939 |
| Follow-up reused tokens | 102911 | 102911 | 102911 |
| Follow-up new prompt tokens | 28 | 28 | 28 |
| Follow-up prompt evaluation seconds | 0.615 | 0.738 | 1.194 |
| Follow-up first-token seconds | 0.689 | 0.817 | 1.258 |
| Follow-up output tokens | 256 | 256 | 256 |
| Follow-up generation tokens/s | 14.975 | 13.618 | 12.046 |
| Total follow-up client wall seconds | 17.731 | 19.550 | 22.437 |
| Follow-up accepted/proposed draft tokens | 152/205 | 120/134 | 116/138 |
| Truncated | false on both requests | false on both requests | false on both requests |

HIP MTP 2 is the fastest configuration measured in these samples. Compared with Vulkan F16 MTP 1, generation is about 27% faster initially and 24% faster on the follow-up, and the total cold request is about 27% shorter. Compared with HIP MTP 1, generation is about 14% faster initially and 10% faster on the follow-up.

The strict 15 tokens/s threshold is NOT established as consistently satisfied: initial generation is above it, but the 14.975 follow-up average is slightly below. Calling that approximately 15 is fair as a rounded description, not evidence of a sustained minimum or adequate headroom. Real reasoning and Pi/SWE tool cycles remain unmeasured.

Method: one cold synthetic Python request followed by one append-only continuation. Temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0. Output is prompted without thinking and forced to the stated lengths using ignore_eos=true. The input is generated, repetitive-structure Python with unique names and numeric values. Stochastic output, expert routing and draft acceptance vary; follow-up token sequences differ across configurations. Each configuration was measured once at a different time without thermal, power or background-workload controls. No confidence interval, coding, reasoning, retrieval or tool-call correctness score was measured. These runs do not use the Vulkan profiler.

The server is left running in the background with HIP MTP 2, F16 K/V and port 9931. Its slot contains the synthetic benchmark conversation. Start-ROCmServer.ps1 supplies the required runtime DLL search paths; launching the downloaded executable in a new shell without those paths does not reproduce this repaired setup.

Future launch (run one server at a time):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kasper\Documents\GitHub\local-llm\outputs\Start-ROCmServer.ps1" -DraftTokens 2
```

An idle matching server can be stopped using outputs/Stop-ROCmServer.ps1. It refuses to interrupt a busy request. The runtime setup is documented in ../ROCm-runtime-notes.md. Exact requests, server launch, server properties, manifests and raw timing events are saved alongside this report.
