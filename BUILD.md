# Build and run the backends

These instructions reproduce the two runtimes used by the reports:

- the official llama.cpp b11888 Windows ROCm release for the unpatched baseline;
- a locally built `ggml-hip.dll` containing the optional eight-block KV split patch.

The patched build is a **replacement DLL only**. It is copied next to the official
`llama-server.exe`, `llama-bench.exe` and runtime DLLs, so the application and CPU
backend stay identical to the release.

Every measurement in the reports names the backend DLL and its SHA-256. Match those
hashes before comparing numbers.

---

## Hardware and software

- **GPU:** AMD Radeon 860M (gfx1152, RDNA 3.5). The ROCm wheel targets `gfx1152`.
- **OS:** Windows 11. Commands are PowerShell.
- **CPU:** 8 threads or more for the default batch settings.
- **RAM:** 64 GB unified memory. The model plus a 163840-token F16 KV cache is about
  24 GB at 103K populated context.
- **Visual Studio 2022 Build Tools** with the C++ workload and Windows 11 SDK.
- **CMake 3.28+** and **Ninja** on `PATH` (`winget install Kitware.CMake NinjaBuild.Ninja`).
- **Python 3.11+**.
- About 30 GB free for the model, source, build tree and downloaded releases.

Run the build steps from the repository root. The repository uses `work/` for all
local artifacts; it is ignored by Git.

---

## 1. Runtime environment for inference

The official ROCm package can load the HIP backend but does not put `hipblas.dll`
on `PATH`. The launchers solve that by pointing at an isolated runtime under
`work\rocm10-runtime`.

```powershell
python -m venv work\rocm10-runtime
& .\work\rocm10-runtime\Scripts\python.exe -m pip install --upgrade pip
& .\work\rocm10-runtime\Scripts\python.exe -m pip install `
    --index-url https://stable.repo.amd.com/rocm/whl-next/ `
    rocm==10.0.0 rocm-sdk-core==10.0.0 rocm-sdk-libraries==10.0.0 rocm-sdk-device-gfx1152==10.0.0
```

The runtime venv does not need the compiler packages. `Start-ROCmServer.ps1`,
`Start-KVSplitServer.ps1` and `run_depth_bench.ps1` prepend
`_rocm_sdk_core\bin` and `_rocm_sdk_libraries\bin` to the process `PATH`
automatically.

---

## 2. Baseline backend: official b11888 ROCm release

Download the official Windows ROCm x64 asset. Do not rebuild the baseline if you
want to compare hashes with the reports: the published DLL is a multi-target
release of about 1.03 GB, while a local `gfx1152` build is about 72 MB.

```powershell
New-Item -ItemType Directory -Path work\backends -Force | Out-Null
Invoke-WebRequest `
    -Uri 'https://github.com/ggml-org/llama.cpp/releases/download/b11888/llama-b11888-bin-win-rocm-10.0-x64.zip' `
    -OutFile work\llama-b11888-rocm.zip
Expand-Archive -Path work\llama-b11888-rocm.zip -DestinationPath work\backends -Force
```

The extracted folder must end up as:

```text
work\backends\llama-b11888-bin-win-rocm-10.0-x64\
    llama-server.exe
    llama-bench.exe
    ggml-hip.dll
    ...other runtime DLLs
```

If the archive expands flat or under a different folder name, rename or move it so
the path above contains `llama-server.exe`.

Verify the backend:

```powershell
.\llm.ps1 -ListDevices
```

Expected:

```text
ROCm0: AMD Radeon (TM) 860M Graphics
gfx1152, VMM: no, Wave Size: 32
```

If you see `Available devices: (none)`, the isolated runtime paths are missing.
Use the launcher rather than running `llama-server.exe` directly.

Reference hash:

```powershell
(Get-FileHash work\backends\llama-b11888-bin-win-rocm-10.0-x64\ggml-hip.dll -Algorithm SHA256).Hash
```

| File | SHA-256 |
|---|---|
| official b11888 ROCm `ggml-hip.dll` | `96bb239b4ae8f343e33512287dfcc89fc18e5d5ebf98632622085cd88825a357` |

This is the unpatched DLL used for the baseline and control runs. If your download
does not match, stop and check the release tag before comparing performance.

---

## 3. Optional: build the eight-block patched backend

This builds only `ggml-hip.dll` from the same commit. The patch changes the
Stream-K launch distribution in `fattn.cu` / `fattn-common.cuh`; it does not change
kernel arithmetic or K/V precision.

### 3.1 Build environment

Create a separate venv with the ROCm development tree. This matches the build
recorded in `outputs/hip-op-profile-results/REPORT.md`.

```powershell
& .\work\rocm10-runtime\Scripts\python.exe -m venv work\rocm10-build-tools
& .\work\rocm10-build-tools\Scripts\python.exe -m pip install --upgrade pip
& .\work\rocm10-build-tools\Scripts\python.exe -m pip install `
    --index-url https://stable.repo.amd.com/rocm/whl-next/ `
    'rocm[libraries,devel,device-gfx1152]==10.0.0'
& .\work\rocm10-build-tools\Scripts\rocm-sdk.exe init
```

Initialize the MSVC environment in the same PowerShell window:

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
```

Then set the ROCm compiler paths:

```powershell
$sdk = (& .\work\rocm10-build-tools\Scripts\rocm-sdk.exe path --root | Select-Object -Last 1).Trim()
if (-not $sdk) { throw 'rocm-sdk path --root failed' }
$sdkForward = $sdk.Replace('\', '/')
$env:HIP_PATH = $sdk
$env:ROCM_PATH = $sdk
$env:HIPCXX = "$sdk\lib\llvm\bin\clang++.exe"
$env:PATH = "$sdk\bin;$env:PATH"
```

`$sdk` is the expanded development tree (normally
`work\rocm10-build-tools\Lib\site-packages\_rocm_sdk_devel`). The compiler is
`$sdk\lib\llvm\bin\clang.exe` / `clang++.exe`, not the Python interpreter.

### 3.2 Source and patch

```powershell
git clone https://github.com/ggml-org/llama.cpp.git work\llama.cpp-b11888
Set-Location work\llama.cpp-b11888
git checkout 48c9f08b7b25766324282b7605177e2f2be755bb
git apply --check ..\..\outputs\hip-kv-split-build\experiment.patch
git apply --whitespace=fix ..\..\outputs\hip-kv-split-build\experiment.patch
```

### 3.3 Configure and build only the HIP backend

```powershell
cmake -S . -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="$sdk" `
    -DHIP_PATH="$sdk" `
    -DCMAKE_C_COMPILER="$sdk\lib\llvm\bin\clang.exe" `
    -DCMAKE_CXX_COMPILER="$sdk\lib\llvm\bin\clang++.exe" `
    "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types" `
    "-DCMAKE_CXX_FLAGS=--hip-path=$sdkForward --rocm-device-lib-path=$sdkForward/lib/llvm/amdgcn/bitcode" `
    -DGGML_HIP=ON `
    -DGGML_HIP_GRAPHS=ON `
    -DGGML_BACKEND_DL=ON `
    -DGGML_NATIVE=OFF `
    -DGGML_CPU=OFF `
    -DAMDGPU_TARGETS=gfx1152 `
    -DLLAMA_BUILD_COMMON=OFF `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_TOOLS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_SERVER=OFF `
    -DLLAMA_BUILD_APP=OFF

cmake --build build --target ggml-hip --parallel $env:NUMBER_OF_PROCESSORS
```

The DLL is normally `build\bin\ggml-hip.dll`. If it is elsewhere:

```powershell
Get-ChildItem build -Recurse -Filter ggml-hip.dll | Select-Object FullName
```

### 3.4 Put the patched DLL next to the release application

Copy the baseline directory first. The patched build contains only `ggml-hip.dll`;
`llama-server.exe`, `llama-bench.exe`, the CPU backend and the runtime DLLs must
come from the release.

```powershell
Set-Location ..\..
$splitDir = 'work\hip-kv-split-bin'
New-Item -ItemType Directory -Path $splitDir -Force | Out-Null
Copy-Item -Path 'work\backends\llama-b11888-bin-win-rocm-10.0-x64\*' -Destination $splitDir -Recurse -Force
$builtDll = 'work\llama.cpp-b11888\build\bin\ggml-hip.dll'
if (-not (Test-Path $builtDll)) {
    $found = Get-ChildItem 'work\llama.cpp-b11888\build' -Recurse -Filter ggml-hip.dll |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $found) { throw 'Built ggml-hip.dll not found; see step 3.3.' }
    $builtDll = $found
}
Copy-Item $builtDll (Join-Path $splitDir 'ggml-hip.dll') -Force
```

Reference hash from this machine:

```powershell
(Get-FileHash work\hip-kv-split-bin\ggml-hip.dll -Algorithm SHA256).Hash
```

| File | SHA-256 |
|---|---|
| eight-block patched `ggml-hip.dll` | `51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5` |

A bit-identical rebuild is not guaranteed across compiler updates or different
absolute paths. What must match is the source commit, the patch and the numerical
checks in `outputs/kv-split-100k-results-verified/REPORT.md`. If the hash differs,
record your own hash and re-run the numerical fixtures before comparing speed.

The same value is used as the control hash in `outputs/build_fine_kv.py`,
`outputs/build_attention_workgroup.py`, `outputs/build_mmvq_variants.py` and
`outputs/build_vocab_probe.py`.

---

## 4. Optional: official Vulkan release

```powershell
Invoke-WebRequest `
    -Uri 'https://github.com/ggml-org/llama.cpp/releases/download/b11888/llama-b11888-bin-win-vulkan-x64.zip' `
    -OutFile work\llama-b11888-vulkan.zip
Expand-Archive -Path work\llama-b11888-vulkan.zip -DestinationPath work\vulkan-bin -Force
```

`work\vulkan-bin` must contain `llama-server.exe`, `llama-bench.exe` and
`ggml-vulkan.dll`. Rename or move the extracted folder if necessary.

```powershell
(Get-FileHash work\vulkan-bin\ggml-vulkan.dll -Algorithm SHA256).Hash
```

The reports publish only a truncated comparison value: the hash starts with
`49C8F389` and ends with `D247`. The full value is not stored in this repository;
record the full hash of your own download in your results.

---

## 5. Model

Download `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` from
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
(about 21 GB).

Place it at:

```text
%USERPROFILE%\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
```

Or set:

```powershell
$env:LLM_MODEL_PATH = 'D:\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf'
```

---

## 6. Verify and run

Baseline device enumeration:

```powershell
.\llm.ps1 -ListDevices
```

Patched backend:

```powershell
# Window 1
.\outputs\Start-KVSplitServer.ps1

# Window 2
Invoke-RestMethod http://127.0.0.1:9931/health
```

Stop it while idle with `.\outputs\Stop-KVSplitServer.ps1`.

Quick benchmark:

```powershell
.\outputs\run_benchmarks.ps1
```

This runs the 103K populated decode test with `llama-bench`. The two cached-prefix
tests need a saved slot; pass `-Slot103k` and `-Slot160k` if you have one. Without
those arguments the script records them as skipped. See the header of
`outputs/run_benchmarks.ps1`.

---

## Directory layout after build

```text
<repo root>
├── llm.ps1                         # short launcher entry point
├── work/
│   ├── rocm10-runtime/             # isolated ROCm 10 runtime (step 1)
│   ├── rocm10-build-tools/         # compiler/devel tree (step 3)
│   ├── llama.cpp-b11888/           # patched source (step 3)
│   ├── backends/
│   │   └── llama-b11888-bin-win-rocm-10.0-x64/   # official release (step 2)
│   ├── hip-kv-split-bin/           # release files + patched DLL (step 3)
│   ├── vulkan-bin/                 # official Vulkan release (step 4)
│   └── slot-cache/                 # saved session slots (not in Git)
└── outputs/
    ├── Start-ROCmServer.ps1
    ├── Start-KVSplitServer.ps1
    ├── run_depth_bench.ps1
    ├── run_benchmarks.ps1
    └── hip-kv-split-build/
        └── experiment.patch
```

---

## Reproduction notes

- Every report names the backend DLL, its SHA-256 and the llama.cpp commit
  (`48c9f08b7`).
- `run_depth_bench.ps1` records the backend and llama-bench hashes in `manifest.json`.
- Comparisons are bracketed (control, candidate, candidate, control) to reduce the
  effect of laptop thermals and clock drift.
- Token/s figures are synthetic throughput on specific prompts, not quality scores.
- The official ROCm release and the local gfx1152 build are not byte-identical.
  Compare each against its own control, not against the other.
