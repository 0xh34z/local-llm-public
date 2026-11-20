# Local LLM on integrated graphics

Notes, scripts and benchmark evidence from tuning Qwen3.6-35B-A3B (a MoE model with a native MTP draft head) on a Framework Laptop 13: Ryzen AI 7 350, Radeon 860M (gfx1152), 64 GB DDR5-5600, Windows 11. The goal is a usable local coding agent at 100K+ populated context, with a practical target of 15 generation tokens/s.

Blog post: "Running LLM to the silicon limit" on [kaspervandekimmenade.nl](https://kaspervandekimmenade.nl/)

This is a personal research repo, not a polished project. Numbers are measurements from one machine. Where something is a calculation or a hypothesis, the report says so.

## Headline results

| Measurement | Result |
|---|---|
| Baseline | ROCm (HIP), F16 K/V, MTP 2, context 163840, one slot, batch 2048, microbatch 512, 8 threads |
| Generation at ~103K populated context | about 17 tokens/s sampled; 16.70 with the 8-block KV split vs 14.72 for bracketed controls on the original synthetic benchmark |
| Generation at ~160K populated context | about 15 tokens/s sampled, not a guaranteed floor |
| Restored 103K session, short follow-up | 53 new tokens evaluated, first output in about 1 s |
| Fully uncached 102400-token prompt | about 11 to 12 minutes to first output, which is why slot caching matters |
| Measured memory read bandwidth | 81.98 GB/s (theoretical 89.6 GB/s) |

MTP 2 runs at roughly 70 to 72 percent of its byte-bandwidth floor at 103K and 160K. See the ceiling report below for the breakdown.

## Key findings

- **Single-token attention is already at the memory floor** (0.26 µs vs 0.25 µs per populated token), so the remaining headroom is in 3-token verification and the MTP draft passes. See [ceiling round](outputs/ceiling-round/REPORT.md).
- **MTP 2 beats MTP 1 and MTP 3** on sampled code output at 103K and 160K. See [ROCm tuning results](outputs/ROCm-tuning-results.md) and the ceiling round.
- **Eight stream-K blocks instead of four** for the Q3/D256/F16 attention shape at 100K+ tokens gave about +13% on the original benchmark. Six, ten and twelve blocks lost. The change is [outputs/hip-kv-split-build/experiment.patch](outputs/hip-kv-split-build/experiment.patch); see [KV split results](outputs/kv-split-100k-results-verified/REPORT.md) and [Tuning round 3](outputs/Tuning-round-3-results.md).
- **HIP beats Vulkan at long context** (17.36 vs 12.37 tokens/s at 103K) even though Vulkan is faster at short context. See [Vulkan round](outputs/vulkan-round/REPORT.md).
- **Several ideas were tested and rejected**: forced tile or vector attention, other MMA tile widths, 128-thread workgroups, Q in LDS, MMVQ launch shapes, thread sweeps. They are documented so nobody repeats them. See the [attention workgroup round](outputs/attention-workgroup-round/REPORT.md) and the [MMVQ round](outputs/mmvq-round/REPORT.md).
- **Per-op profilers on Windows lie.** `hipEventRecord` costs about 25 µs per event, which inflates small ops. See [HIP op profile](outputs/hip-op-profile-results/REPORT.md).
- **Slot caching makes it usable.** A saved 103K-token prefix restores in about 2 seconds. See [Cold prefill and reuse](outputs/Cold-prefill-and-reuse.md) and [Windows speed round](outputs/Windows-speed-round.md).

## Reports

| Report | What it covers |
|---|---|
| [Ceiling round](outputs/ceiling-round/REPORT.md) | Measured decode vs GGUF byte floors |
| [Stream bandwidth](outputs/stream-bandwidth-gfx1152/REPORT.md) | Sustained memory read bandwidth on gfx1152 |
| [Vulkan round](outputs/vulkan-round/REPORT.md) | HIP vs Vulkan across context depths |
| [MMVQ round](outputs/mmvq-round/REPORT.md) | Matrix-vector launch shape experiments |
| [Attention workgroup round](outputs/attention-workgroup-round/REPORT.md) | 128 threads and Q-in-LDS experiments |
| [Laptop hardware round](outputs/laptop-hardware-round/REPORT.md) | AC power mode, block counts, cold microbatch |
| [Cold microbatch A/B](outputs/cold-microbatch-abba-100k/REPORT.md) | Microbatch 512 vs 1024 on a cold 102400-token prompt |
| [Tuning round 3](outputs/Tuning-round-3-results.md) | KV split follow-ups, 128K and 160K, microbatch |
| [KV split](outputs/kv-split-100k-results-verified/REPORT.md) | The 8-block experiment |
| [MMA tile](outputs/mma-tile-100k-results-verified/REPORT.md) | Tile column width comparison |
| [Attention dispatch](outputs/attention-dispatch-100k-results/REPORT.md) | Forced tile vs forced vector vs automatic |
| [Thread sweep](outputs/thread-sweep-100k-results/REPORT.md) | Generation thread count |
| [HIP op profile](outputs/hip-op-profile-results/REPORT.md) | Diagnostic backend timings (provisional) |
| [Cold prefill and reuse](outputs/Cold-prefill-and-reuse.md) | Slot save and restore workflow |
| [ROCm tuning results](outputs/ROCm-tuning-results.md) | MTP depth and K/V precision |

Small summary files (summaries, samples, manifests, comparisons) are included next to the reports, with local paths replaced by placeholders such as `<repo>`, `<models>` and `%USERPROFILE%`. The full raw data is not public: request and response dumps, generated text, server and build logs, test fixtures and saved slots. Some report links therefore do not resolve, and those reports have a note at the top.

## Layout

- `outputs/`: launchers, benchmark and analysis scripts, patches, reports, and Pi extensions.
- `work/`: not in this repo. Runtimes, backends, private HIP builds and slot caches go here locally.

Useful scripts: `Start-ROCmServer.ps1` (launcher), `Manage-LLMSlot.ps1` (slot save and restore), `byte_budget.py` (GGUF byte floors), `stream_bandwidth.hip` (bandwidth probe), `launch_overhead.hip` (dispatch and event overhead probe), `run_depth_bench.ps1` (depth benchmark, `-Runtime vulkan` for Vulkan).

## Running it

Not included, you have to provide them:

1. The llama.cpp b11888 ROCm backend at `work/backends/llama-b11888-bin-win-rocm-10.0-x64`.
2. An isolated ROCm runtime at `work/rocm10-runtime` (see [ROCm runtime notes](outputs/ROCm-runtime-notes.md) and `outputs/rocm-runtime-requirements.txt`).
3. The model: [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), `UD-Q4_K_XL`. Set `LLM_MODEL_PATH` to the file, or put it at `~\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`.

```powershell
$env:LLM_MODEL_PATH = 'D:\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf'   # optional
.\llm.ps1
```

With explicit options:

```powershell
.\outputs\Start-ROCmServer.ps1 -DraftTokens 2 -KvCache f16 -Threads 8 -BatchSize 2048 -MicroBatchSize 512
```

Experimental HIP builds (8-block KV split and the other rounds) are private builds that have to be compiled from the patches against the matching llama.cpp commit (`48c9f08b7b25766324282b7605177e2f2be755bb`). There are no build instructions yet. Some scripts have other environment overrides, for example `NUMPY_PYTHON` for `screen_kv_split.py`.

## Pi extensions

In `outputs/pi-extensions/` for the [Pi](https://github.com/earendil-works/pi) coding agent:

- `performance`: shows TTFT and tokens/s in the footer
- `websearch`: Tavily search and page reading
- `read-range`: `/read-range <file> <first-line> <line-count> <question>` prompt plus a local read tool, so the model only pulls the lines it needs

## Methodology notes

- Comparisons are bracketed (control, candidate, candidate, control) because laptop thermals and clock drift can fake a few percent.
- Tokens/s figures are synthetic throughput on specific prompts, not quality scores. Candidate runs often generate different tokens and different MTP work, so these are end-to-end results, not fixed-work kernel percentages.
- Kernel changes are checked against an independent float64 reference. The upstream CPU comparison was inaccurate on the captured long-context fixtures, and the Q1 discrepancy is unresolved.
- No sustained generation floor has been established. Allocated context is not the same as populated context, so reports state both.
- Nothing here is a production recommendation or a quality guarantee.

## License

[MIT](LICENSE): use it for anything, no warranty. This covers the scripts, patches and reports in this repo. The model weights and llama.cpp are not included and have their own licenses.
