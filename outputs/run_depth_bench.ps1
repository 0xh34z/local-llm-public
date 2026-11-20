# Main-model decode cost versus populated depth, without the server or speculation.
# pp2/pp3 approximate the 2- and 3-token verify batches; tg32 is single-token decode.
# llama-bench prefills each depth once with random tokens and restores that state for each repetition.
param(
    [Parameter(Mandatory = $true)][string]$OutDirectory,
    [string]$Depths = '0,8192,32768,65536,103194,131072,160511',
    [string]$Prompts = '2,3',
    [int]$Generate = 32,
    [int]$Repetitions = 5,
    [int]$MicroBatchSize = 512,
    [ValidateSet('hip', 'vulkan')][string]$Runtime = 'hip',
    [string]$BackendDirectory,  # optional HIP runtime override, e.g. work\hip-mmvq\c1\bin
    [string]$ModelPath,
    [string]$Device,
    [ValidateRange(1, 65535)][int]$Port = 9931  # only used to avoid clashing with a running server
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function ConvertTo-ArgumentString {
    param([string[]]$Items)
    $quoted = foreach ($item in $Items) {
        if ($item -match '[\s"]') { '"' + ($item -replace '"', '\"') + '"' } else { $item }
    }
    return ($quoted -join ' ')
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port has a server; stop it while idle first." }
if (Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue) { throw 'Another llama process is running.' }
$out = [IO.Path]::GetFullPath($OutDirectory)
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
$backend, $defaultDevice, $dll = if ($Runtime -eq 'vulkan') { (Join-Path $root 'work\vulkan-bin'), 'Vulkan0', 'ggml-vulkan.dll' }
    else { (Join-Path $root 'work\hip-kv-split-bin'), 'ROCm0', 'ggml-hip.dll' }
$device = if ($Device) { $Device } else { $defaultDevice }
if ($BackendDirectory) {
    if ($Runtime -ne 'hip') { throw '-BackendDirectory is only for HIP runtimes' }
    $backend = if ([IO.Path]::IsPathRooted($BackendDirectory)) { $BackendDirectory } else { Join-Path $root $BackendDirectory }
    $backend = [IO.Path]::GetFullPath($backend)
}
foreach ($required in @((Join-Path $backend 'llama-bench.exe'), (Join-Path $backend $dll))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required backend file missing: $required" }
}
$model = $ModelPath
if (-not $model) { $model = $env:LLM_MODEL_PATH }
if (-not $model) { $model = Join-Path $HOME 'Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf' }
if (-not (Test-Path -LiteralPath $model)) { throw "Model not found: $model. Set -ModelPath, LLM_MODEL_PATH, or place it under ~\Models." }
$settings = [ordered]@{}
$previousSettings = @{}
$previousPath = $env:PATH
if ($Runtime -eq 'hip') {
    $settings = [ordered]@{ GGML_HIP_FA_KV_SPLIT = 'double'; GGML_HIP_FA_MMA_TILE = 'auto'; GGML_HIP_FA_EXPERIMENT = 'auto'; GGML_HIP_OP_PROFILE = '0' }
    foreach ($k in $settings.Keys) {
        $previousSettings[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
        [Environment]::SetEnvironmentVariable($k, $settings[$k], 'Process')
    }
    $env:PATH = "$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_core\bin;$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_libraries\bin;$env:PATH"
}
$arguments = @('-m', $model, '-dev', $device, '-ngl', '99', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16',
    '-b', '2048', '-ub', "$MicroBatchSize", '-t', '8', '-p', $Prompts, '-n', "$Generate", '-d', $Depths,
    '-r', "$Repetitions", '-o', 'jsonl', '--progress')
[ordered]@{
    started_utc = [DateTime]::UtcNow.ToString('o'); arguments = $arguments; environment = $settings
    backend = $backend; backend_dll = $dll; backend_sha256 = (Get-FileHash (Join-Path $backend $dll)).Hash
    bench_sha256 = (Get-FileHash (Join-Path $backend 'llama-bench.exe')).Hash
    power_scheme = (powercfg /getactivescheme) -join ' '
    battery_status = (Get-CimInstance Win32_Battery).BatteryStatus
} | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $out 'manifest.json') -Encoding utf8
$stop = Join-Path $out 'telemetry.stop'
$telemetry = $null
$telemetryExe = Join-Path $root 'work\adlx-telemetry\observe_gpu.exe'
if (Test-Path -LiteralPath $telemetryExe) {
    $telemetry = Start-Process -FilePath $telemetryExe -ArgumentList @('2', "`"$stop`"") `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out 'gpu-sensors.csv') -RedirectStandardError (Join-Path $out 'telemetry-stderr.log')
}
try {
    $bench = Start-Process -FilePath (Join-Path $backend 'llama-bench.exe') -ArgumentList (ConvertTo-ArgumentString $arguments) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $out 'bench.jsonl') -RedirectStandardError (Join-Path $out 'bench-stderr.log')
    if ($bench.ExitCode -ne 0) { throw "llama-bench exited $($bench.ExitCode)" }
}
finally {
    New-Item -ItemType File -Path $stop -Force | Out-Null
    if ($telemetry) { $telemetry.WaitForExit(10000) | Out-Null }
    foreach ($k in $previousSettings.Keys) { [Environment]::SetEnvironmentVariable($k, $previousSettings[$k], 'Process') }
    $env:PATH = $previousPath
    [ordered]@{ ended_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content (Join-Path $out 'ended.json') -Encoding utf8
}
