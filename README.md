# Local LLM Inference on Integrated Graphics

**Running Qwen3.6-35B-A3B (MoE) on a Radeon 860M at 15–19 tok/s with 100K+ token context — using custom ROCm kernel tuning on Windows.**

This repo documents the methodology, benchmarks, and tools for running frontier-class LLMs on consumer integrated GPU hardware. It's an ongoing research project — not finished, but well-documented.

---

## TL;DR

| Metric | Result |
|--------|--------|
| Hardware | Ryzen AI 7 350 / Radeon 860M (gfx1152), 64 GB DDR5-5600 |
| Model | Qwen3.6-35B-A3B-MTP UD-Q4_K_XL (~21 GB) |
| Generation speed | 15–19 tok/s at 100K+ context |
| Context hit rate | 99.8% at 185K tokens |
| First token latency | 0.62s (cached) |
| Platform | Windows 11 + ROCm 10.0 (isolated runtime) |

---

## What's Here

This repository contains:

- **Benchmark reports** — Controlled experiments with matched workloads, float64 correctness checks, and documented results
- **Scripts** — Runnable benchmarks, slot cache helpers, Pi extensions
- **Kernel patches** — HIP attention kernel modifications (for reference)
- **Tuning notes** — ROCm runtime setup, Pi integration guides

## Hardware Used

- **CPU:** AMD Ryzen AI 7 350 (Zen 5, 8C/16T, 2.0–5.0 GHz)
- **GPU:** Radeon 860M (gfx1152, RDNA 3) — integrated
- **RAM:** 64 GB DDR5-5600 (2×32 GB, dual-channel)
- **OS:** Windows 11
- **Inference engine:** llama.cpp b11888

## Key Techniques

### ROCm on Windows
AMD's ROCm is Linux-first. This project ships an isolated ROCm 10.0 runtime that works on Windows without driver conflicts. See [ROCm runtime notes](outputs/ROCm-runtime-notes.md) for setup details.

### Custom KV Split Attention Kernel
A targeted HIP backend modification splits the 32-column MMA attention tile's KV work across **eight blocks instead of four** for the long-context shape on gfx1152. Results:

| Workload | Control TPS | 8-Block TPS | Gain |
|----------|------------:|------------:|-----:|
| 103K original continuation | 14.72 | **16.70** | +13.4% |
| 103K code generation | 16.61 | **18.53** | +11.6% |
| 103K open-thinking | 17.15 | **18.72** | +9.2% |
| 128K restored follow-up | 13.84 | **14.78** | +6.8% |
| 160K restored follow-up | 12.85 | **15.15** | +17.9% |

See [Tuning Round 3](outputs/Tuning-round-3-results.md) for full details.

### Multi-Token Prediction (MTP)
Uses 2 draft tokens per step with ~75% acceptance rate, effectively doubling throughput on accepted tokens. F16 KV cache preferred over Q8 (Q8 was ~25% slower).

### Slot Cache Save/Restore
Saves 100K+ token conversation state to disk (~2.5 GB snapshot) and reloads in ~2 seconds. Near-instant TTFT after reload because the prefix is cached. See [Cold prefill and reuse](outputs/Cold-prefill-and-reuse.md).

## Current Baseline Configuration

```
Model:     Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
Backend:   ROCm0 (llama.cpp b11888, isolated ROCm 10.0)
Context:   163840 tokens
GPU layers: 99 (maximum)
K/V cache: F16
MTP:       2 (draft tokens)
Flash Attn: on
Batch size: 2048
Microbatch: 512
Threads:   8
```

## Benchmark Reports

| Report | Description |
|--------|-------------|
| [Tuning Round 3](outputs/Tuning-round-3-results.md) | KV split, microbatch, prefill experiments |
| [ROCm Tuning Results](outputs/ROCm-tuning-results.md) | MTP depth and KV precision comparison |
| [Cold Prefill & Reuse](outputs/Cold-prefill-and-reuse.md) | Cache save/restore workflow |
| [Thread Sweep](outputs/thread-sweep-100k-results/REPORT.md) | Generation worker count (1–8) |
| [MMA Tile Sweep](outputs/mma-tile-100k-results-verified/REPORT.md) | Tile column size comparison |
| [Attention Dispatch](outputs/attention-dispatch-100k-results/REPORT.md) | Forced tile vs automatic MMA |
| [HIP Op Profile](outputs/hip-op-profile-results/REPORT.md) | Custom HIP diagnostic backend |
| [Vulkan Profile](outputs/vulkan-profile-100k-diagnostic/REPORT.md) | Vulkan long-context diagnostics |
| [Populated 128K](outputs/populated-128k-kv-split-results/REPORT.md) | Long-context at 128K tokens |
| [Populated 160K](outputs/populated-160k-kv-split-results/REPORT.md) | Long-context at 160K tokens |

## What's Not Included

The following are excluded from this repo (private work-in-progress):

- Custom HIP backend binaries (built locally, not shared)
- ROCm runtime installation (requires ROCm SDK)
- Model weights (Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf, ~21 GB)
- Slot cache files
- Server logs

## How to Run

The launcher expects:

1. llama.cpp b11888 ROCm backend in `work/backends/`
2. Isolated ROCm 10.0 runtime in `work/rocm10-runtime/`
3. Model weights at `C:\Users\kasper\Desktop\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`

Start the server:

```powershell
.\llm.ps1
```

Or with explicit options:

```powershell
.\outputs\Start-ROCmServer.ps1 -DraftTokens 2 -KvCache f16 -Threads 8 -BatchSize 2048 -MicroBatchSize 512
```

## Pi Extensions

This repo also includes Pi (coding agent) extensions:

- **Performance monitor** — TTFT, TPS, output count, elapsed time per request
- **Web search** — Tavily search and page reading integration

See their respective READMEs in `outputs/pi-extensions/`.

## Methodology Notes

- All benchmarks use controlled, matched workloads with byte-identical inputs where possible
- Float64 correctness checks verify numerical equivalence against CPU reference
- Generated TPS measurements are synthetic throughput — not quality scores
- No sustained TPS floor has been established; results vary by workload
- See individual reports for full methodology and limitations

## License

Benchmark reports and scripts are shared under Apache 2.0. The model (Qwen3.6) is under its respective license.

---

*This is an ongoing project. Results are experimental and subject to change. No production guarantee is provided.*
