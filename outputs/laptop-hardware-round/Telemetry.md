# GPU telemetry on the Framework Laptop 13

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

The benchmark machine has a Ryzen AI 7 350, Radeon 860M and one 32 GiB DDR5-5600 module on each of channels A and B. The Windows power scheme was Balanced. The AC power mode can vary independently within that scheme. Windows hardware and driver information is recorded in [system.json](system.json). AMD lists DDR5-5600 support for this processor. [AMD processor specifications](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html)

The probe selected the GPU whose reported name contains `860M` and called `GetCurrentGPUMetrics` every 15 seconds. It recorded wall-clock Unix milliseconds after each driver call. The driver supplied core and memory clocks, GPU utilization, temperature and GPU power. An empty field means the getter did not return a value. GPU power is the driver's reading in watts. The total-board-power getter returned no values. The CSV includes idle periods between experiments. Use request timestamps when interpreting the samples. [Raw observations](gpu-sensors.csv) · [Probe source](observe_gpu.cpp) · [AMD metrics interface](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX/blob/32b5a740d42295c5dfe9026b9f52683da0f3af91/SDK/Include/IPerformanceMonitoring.h)

## Reproduction

The build used Visual Studio 2022 x64 Build Tools, the installed AMD driver and the AMD ADLX SDK at commit `32b5a740d42295c5dfe9026b9f52683da0f3af91`, checked out under `work/adlx-sdk`. In an x64 Native Tools Command Prompt, run from the repository root. Create `work/adlx-telemetry` if needed.

```bat
pushd work\adlx-telemetry
cl /nologo /EHsc /std:c++17 /O2 /DUNICODE /D_UNICODE /I..\adlx-sdk ..\..\outputs\laptop-hardware-round\observe_gpu.cpp ..\adlx-sdk\SDK\ADLXHelper\Windows\Cpp\ADLXHelper.cpp ..\adlx-sdk\SDK\Platform\Windows\WinAPIs.cpp /Fe:observe_gpu.exe
popd
```

For one reading, run `work\adlx-telemetry\observe_gpu.exe 0 unused`. For periodic readings, pass a positive interval in seconds and an initially absent stop-file path. Redirect stdout to a fresh CSV. Creating the stop file ends monitoring after the current sleep.
