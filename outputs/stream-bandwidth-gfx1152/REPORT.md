# Radeon 860M streaming read bandwidth

The fastest tested configuration read a 1 GiB buffer at **81.98 GB/s** by GPU event time and **81.73 GB/s** by host time. Both are medians of seven batches. All 28 batches across the four configurations passed checksum validation. [Raw timings](samples.jsonl)

| Input | Workgroups | Median GPU GB/s | GPU range GB/s | Median wall GB/s |
|---|---:|---:|---:|---:|
| 512 MiB | 128 | 78.29 | 77.28–79.74 | 78.10 |
| 1 GiB | 512 | 81.98 | 81.70–82.70 | 81.73 |
| 1 GiB | 128 | 79.62 | 75.77–80.04 | 79.39 |
| 512 MiB | 512 | 79.56 | 79.00–80.13 | 79.23 |

The original single-token Q8 vocabulary projection processed 540,344,320 logical weight bytes in 6.872 ms in its earlier isolated test. Its effective weight-read rate of 78.6 GB/s is about **96% of the fastest streaming median**. That comparison suggests limited room to speed up the projection solely by streaming the same weight bytes faster. Other operations, reducing repeated work and changing memory traffic can affect inference differently. No end-to-end token rate follows from either byte rate. [Projection measurements](../Windows-speed-round.md)

## Measurement

The HIP probe reads an initialized, device-allocated buffer sequentially across adjacent threads. Each 256-thread workgroup uses grid strides and four independent 16-byte loads per loop iteration. Inspection of the compiled gfx1152 code showed four `global_load_b128` instructions and a final `global_store_b128` in the read kernel. Each thread writes a 16-byte checksum. The host independently checks all four checksum lanes for every thread against the arithmetic sum of the expected input.

Each case warms up for three sweeps, then measures seven batches of six complete reads. The host poisons the output before each batch and validates it afterward. HIP event time covers the six launches and host elapsed time covers event submission through completion. Both rates count input bytes only, in decimal GB/s. Buffer initialization, output poisoning, checksum copies and validation happen outside both timed intervals.

The CPU is a Ryzen AI 7 350. Windows reports two 32 GiB memory devices, a Hynix HMCG88AGBSA095N and a Micron CT32G56C46S5.M16D1, each with a 64-bit data width and configured at DDR5-5600. Assuming both 64-bit channels operate in parallel, their nominal combined transfer rate is 89.6 GB/s. Actual read rate depends on access pattern, cache behavior, concurrent memory traffic, clock and power conditions. The experiment has no DRAM transaction counters.

The four configurations ran once in the table's order on the same Windows installation, immediately after the fine-split inference server exited and before the power comparison started. Different kernel schedules, memory allocation patterns, and operating conditions can yield different rates. Runs used `work/hip-stream-bandwidth/stream_bandwidth.exe` with SHA256 `746ef0b45a383d61af35e6f508ce4069b8a3ce5d9a0c3d7fb95d31057b9e62e1`. [HIP kernel](../stream_bandwidth.hip) · [Build and measurement script](../run_stream_bandwidth.py)

## Reproduce

Use the project Python environment with the local ROCm build SDK and Visual Studio Build Tools. Run the measurement with the inference server idle and GPU benchmarks stopped. Choose an unused result path for a fresh run.

```powershell
& .\work\rocm10-runtime\Scripts\python.exe outputs/run_stream_bandwidth.py build
& .\work\rocm10-runtime\Scripts\python.exe outputs/run_stream_bandwidth.py measure outputs/stream-bandwidth-gfx1152/repeat.jsonl
```
