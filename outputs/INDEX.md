# Artifact index

This page separates what is public from what the reports refer to but does not
ship. It is checked by `outputs/check_links.py` together with the README and
BUILD guide.

## Start here

- [README](../README.md) - headline results, methodology and caveats.
- [BUILD.md](../BUILD.md) - runtime setup, baseline release and patched DLL build.
- [run_benchmarks.ps1](run_benchmarks.ps1) - three-test summary benchmark.
- [run_depth_bench.ps1](run_depth_bench.ps1) - main-model decode versus depth.
- [byte_budget.py](byte_budget.py) - GGUF byte-floor calculation.
- [Check-Repo.ps1](Check-Repo.ps1) - runs every local check in one command.
- [check_json.py](check_json.py) - validates strict UTF-8 JSON.
- [check_hashes.py](check_hashes.py) - validates published SHA-256 values.
- [check_links.py](check_links.py) - validates Markdown links; `--all` audits reports too.
- [private-links.txt](private-links.txt) - known-private link targets skipped by `--all`.
- [test_byte_budget.py](test_byte_budget.py) - synthetic-GGUF unit tests.
- [requirements.txt](requirements.txt) - Python dependencies for `byte_budget.py` and its tests.

## Main reports

| Report | What it covers |
|---|---|
| [Ceiling round](ceiling-round/REPORT.md) | Measured decode vs GGUF byte floors |
| [Stream bandwidth](stream-bandwidth-gfx1152/REPORT.md) | Sustained memory read bandwidth on gfx1152 |
| [Vulkan round](vulkan-round/REPORT.md) | HIP vs Vulkan across context depths |
| [MMVQ round](mmvq-round/REPORT.md) | Matrix-vector launch shape experiments |
| [Attention workgroup round](attention-workgroup-round/REPORT.md) | 128 threads and Q-in-LDS experiments |
| [Laptop hardware round](laptop-hardware-round/REPORT.md) | AC mode, block counts, cold microbatch |
| [Cold microbatch A/B](cold-microbatch-abba-100k/REPORT.md) | Microbatch 512 vs 1024 on a cold prompt |
| [Tuning round 3](Tuning-round-3-results.md) | KV split follow-ups, 128K and 160K |
| [KV split](kv-split-100k-results-verified/REPORT.md) | The 8-block experiment |
| [MMA tile](mma-tile-100k-results-verified/REPORT.md) | Tile column width comparison |
| [Attention dispatch](attention-dispatch-100k-results/REPORT.md) | Forced tile vs vector vs automatic |
| [Thread sweep](thread-sweep-100k-results/REPORT.md) | Generation thread count |
| [HIP op profile](hip-op-profile-results/REPORT.md) | Diagnostic backend timings (provisional) |
| [Cold prefill and reuse](Cold-prefill-and-reuse.md) | Slot save and restore workflow |
| [ROCm tuning results](ROCm-tuning-results.md) | MTP depth and K/V precision |

## Public supporting data

Each report directory contains the summaries, manifests and sample tables needed
to read its claims. A few JSON files have a UTF-8 BOM; `check_links.py`, CI and
the report-analysis scripts read them with `utf-8-sig`.

The raw file is not always included. Comparisons, request/response dumps,
generated text and server logs are omitted for size and privacy. Some report
links therefore point at files that are not present. Those exact targets are
listed in [private-links.txt](private-links.txt); `python outputs/check_links.py --all`
skips them and CI runs that full audit. This is expected and is documented at the
top of the README.

## Intentionally private or local

These live under `work/` or in per-machine directories and are never committed:

- ROCm runtime and build tool virtual environments.
- Official and private backend binaries, plus built `ggml-hip.dll` variants.
- The model GGUF (about 21 GB).
- Saved session slots, including the 103K and 160K prefixes used by cached tests.
- ADLX telemetry CSVs, build logs and full request dumps.
- Pi agent state under `~\.pi\agent`, needed by the websearch extension tests.

## Local checks

Run everything with one command after installing `outputs/requirements.txt`:

```powershell
.\outputs\Check-Repo.ps1
```

Individual checks are also available:

```powershell
python -m compileall -q outputs
python outputs/check_json.py
python outputs/check_hashes.py
python outputs/check_links.py --all
python outputs/test_byte_budget.py
node --test outputs/pi-extensions/performance/performance.test.mjs
node --test outputs/pi-extensions/websearch/websearch.test.mjs
```

The websearch unit tests use `@sinclair/typebox` from their own `package.json`.
The one integration test that loads an installed Pi agent skips automatically when
Pi is not present.

PowerShell scripts can be parsed without running them:

```powershell
$tokens = $null; $errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path 'outputs\run_benchmarks.ps1'), [ref]$tokens, [ref]$errors) | Out-Null
$errors
```
