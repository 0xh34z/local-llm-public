# Matched 100K backend comparison: ROCm/HIP versus Vulkan

User's Radeon 860M on Windows. Same llama.cpp b11888/48c9f08b7 and Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. Both: context allocation 163840, one slot, GPU layers 99, Flash Attention on, F16 K/V, batch 2048, microbatch 512, eight CPU threads, auto load mode, MTP maximum one draft token. Neither measurement used Vulkan profiling instrumentation.

The complete cold API payload is identical across backends, including every input token and all sampling settings. ROCm explicitly selects --device ROCm0 and uses AMD's isolated ROCm 10.0.0 core, libraries and gfx1152 runtime packages. The live process command line and server properties were saved before the test.

| Metric | ROCm/HIP | Vulkan |
|---|---:|---:|
| Fresh input tokens | 102400 | 102400 |
| Fresh cache reuse | 0 | 0 |
| Prompt evaluation seconds | 835.951 | 1109.878 |
| Prompt evaluation tokens/s | 122.495 | 92.262 |
| Client first-token seconds | 836.070 | 1109.918 |
| Initial output tokens | 512 | 512 |
| Initial generation seconds | 36.852 | 41.290 |
| Initial generation tokens/s | 13.866 | 12.376 |
| Total fresh client wall seconds | 872.937 | 1151.220 |
| Initial draft accepted/proposed | 241/269 | 238/273 |
| Follow-up input tokens | 102939 | 102939 |
| Follow-up cached tokens | 102911 | 102911 |
| Follow-up newly processed tokens | 28 | 28 |
| Follow-up prompt evaluation seconds | 0.738 | 1.194 |
| Follow-up client first-token seconds | 0.817 | 1.258 |
| Follow-up output tokens | 256 | 256 |
| Follow-up generation seconds | 18.726 | 21.168 |
| Follow-up generation tokens/s | 13.618 | 12.046 |
| Follow-up total client wall seconds | 19.550 | 22.437 |
| Follow-up draft accepted/proposed | 120/134 | 116/138 |
| Truncated | false on both requests | false on both requests |

In these samples HIP improves generation by about 12-13% and reduces the cold request wall time by about 24%. It still falls below the user's 15 tokens/s threshold at populated 100K context.

Method: deterministic synthetic Python corpus, non-thinking prompting, temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repeat penalty 1, presence penalty 0. Output lengths forced with ignore_eos=true. The follow-up appends each run's generated token IDs and the same new question. Output text, expert routing and draft acceptance can differ across backends; follow-up token sequences are not identical. Each configuration was measured once at a different time without power, thermal or background-load controls. No confidence interval or coding/tool reliability score was measured. A successful smoke test and these requests confirm computation, not broad correctness parity.

ROCm's initial "Available devices: (none)" was repaired by supplying the missing HIP math-library dependencies through a temporary process PATH. No display driver or system PATH was changed. See ../ROCm-runtime-notes.md and ../Start-ROCmServer.ps1 for the setup.

After this run, its log was preserved as server-log.txt and the agent-owned idle background server was restarted for a separate MTP-2 HIP test. This report describes MTP 1 only; consult the separate MTP-2 results for that configuration.
