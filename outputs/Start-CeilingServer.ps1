# Experiment-only launcher: the production eight-block server flags, with MTP draft tokens 0 (off), 1, 2 or 3,
# optionally on another backend directory/device (e.g. the b11888 Vulkan release on Vulkan0).
# Production use stays with Start-KVSplitServer.ps1.
param(
    [ValidateRange(0,3)][int]$DraftTokens = 2,
    [string]$BackendDirectory,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [ValidateSet('ROCm0', 'Vulkan0')][string]$Device = 'ROCm0',
    [ValidateRange(3,4)][int]$LogVerbosity = 3,  # 4 = trace: prints cumulative MTP draft() time after each request
    [switch]$OpTiming,  # diagnostic only: synchronized HIP op profile / Vulkan perf logger, distorts speed
    [switch]$NoGraphs,  # GGML_CUDA_DISABLE_GRAPHS=1: no HIP graph capture/replay
    [ValidateSet('1','2')][string]$OpProfileMode = '1',  # 2 = unsynchronized HIP event pairs (build_mmvq_variants.py --async-profiler)
    [ValidateRange(1, 65535)][int]$Port = 9931,
    [string]$ModelPath,
    [ValidateRange(1024, 1048576)][int]$Context = 163840
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root 'work\hip-kv-split-bin'
if ($BackendDirectory) { $backend = [IO.Path]::GetFullPath($BackendDirectory) }
$settings = @{ GGML_HIP_FA_KV_SPLIT = 'double'; GGML_HIP_FA_MMA_TILE = 'auto'; GGML_HIP_FA_EXPERIMENT = 'auto'; GGML_HIP_OP_PROFILE = '0' }
if ($OpTiming) { $settings.GGML_HIP_OP_PROFILE = $OpProfileMode; $settings.GGML_VK_PERF_LOGGER = '1' }
if ($NoGraphs) { $settings.GGML_CUDA_DISABLE_GRAPHS = '1' } else { [Environment]::SetEnvironmentVariable('GGML_CUDA_DISABLE_GRAPHS', $null, 'Process') }
foreach ($k in $settings.Keys) { [Environment]::SetEnvironmentVariable($k, $settings[$k], 'Process') }
$env:PATH = "$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_core\bin;$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_libraries\bin;$env:PATH"
$modelPath = $ModelPath
if (-not $modelPath) { $modelPath = $env:LLM_MODEL_PATH }
if (-not $modelPath) { $modelPath = Join-Path $HOME 'Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf' }
$arguments = @('-m', $modelPath, '--device', $Device,
    '-c', "$Context", '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja', '-ctk', 'f16', '-ctv', 'f16',
    '-b', '2048', '-ub', '512', '-t', '8', '-tb', '8', '-td', '8', '-tbd', '8')
if ($DraftTokens -gt 0) { $arguments += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', "$DraftTokens") }
$arguments += @('--port', "$Port", '-lm', 'auto', '-lv', "$LogVerbosity", '--slot-save-path', (Join-Path $root 'work\slot-cache'))
$log = [IO.Path]::GetFullPath($LogPath)
$ErrorActionPreference = 'Continue'
& (Join-Path $backend 'llama-server.exe') @arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE; see $log" }
