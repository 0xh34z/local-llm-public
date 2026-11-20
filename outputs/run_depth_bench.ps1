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
    [string]$BackendDirectory  # optional HIP runtime override, e.g. work\hip-mmvq\c1\bin
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
if (Get-NetTCPConnection -LocalPort 9931 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 9931 has a server; stop it while idle first.' }
if (Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue) { throw 'Another llama process is running.' }
$out = [IO.Path]::GetFullPath($OutDirectory)
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
$backend, $device, $dll = if ($Runtime -eq 'vulkan') { (Join-Path $root 'work\vulkan-bin'), 'Vulkan0', 'ggml-vulkan.dll' }
    else { (Join-Path $root 'work\hip-kv-split-bin'), 'ROCm0', 'ggml-hip.dll' }
if ($BackendDirectory) {
    if ($Runtime -ne 'hip') { throw '-BackendDirectory is only for HIP runtimes' }
    $backend = [IO.Path]::GetFullPath((Join-Path $root $BackendDirectory))
}
$model = $(if ($env:LLM_MODEL_PATH) { $env:LLM_MODEL_PATH } else { Join-Path $HOME 'Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf' })
$settings = [ordered]@{ GGML_HIP_FA_KV_SPLIT = 'double'; GGML_HIP_FA_MMA_TILE = 'auto'; GGML_HIP_FA_EXPERIMENT = 'auto'; GGML_HIP_OP_PROFILE = '0' }
foreach ($k in $settings.Keys) { [Environment]::SetEnvironmentVariable($k, $settings[$k], 'Process') }
$env:PATH = "$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_core\bin;$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_libraries\bin;$env:PATH"
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
$telemetry = Start-Process -FilePath (Join-Path $root 'work\adlx-telemetry\observe_gpu.exe') -ArgumentList @('2', "`"$stop`"") `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out 'gpu-sensors.csv') -RedirectStandardError (Join-Path $out 'telemetry-stderr.log')
try {
    $bench = Start-Process -FilePath (Join-Path $backend 'llama-bench.exe') -ArgumentList $arguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $out 'bench.jsonl') -RedirectStandardError (Join-Path $out 'bench-stderr.log')
    if ($bench.ExitCode -ne 0) { throw "llama-bench exited $($bench.ExitCode)" }
}
finally {
    New-Item -ItemType File -Path $stop -Force | Out-Null
    $telemetry.WaitForExit(10000) | Out-Null
    [ordered]@{ ended_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content (Join-Path $out 'ended.json') -Encoding utf8
}
