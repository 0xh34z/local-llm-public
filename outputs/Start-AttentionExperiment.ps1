param(
    [ValidateSet('auto','tile','vec','mma')][string]$AttentionKernel = 'auto',
    [ValidateSet('auto','16','64')][string]$MmaColumns = 'auto',
    [ValidateSet('auto','half','double','quad')][string]$KvSplit = 'auto',
    [string]$BackendDirectory,
    [switch]$ListDevices,
    [string]$LogPath
)
$ErrorActionPreference = 'Stop'
$attentionRoot = Split-Path -Parent $PSScriptRoot
$attentionBackend = Join-Path $attentionRoot 'work\hip-attention-experiment-bin'
if ($BackendDirectory) { $attentionBackend = [IO.Path]::GetFullPath($BackendDirectory) }
$attentionPreviousTile = [Environment]::GetEnvironmentVariable('GGML_HIP_FA_MMA_TILE','Process')
$attentionPreviousSplit = [Environment]::GetEnvironmentVariable('GGML_HIP_FA_KV_SPLIT','Process')
$attentionPreviousMode = [Environment]::GetEnvironmentVariable('GGML_HIP_FA_EXPERIMENT','Process')
$attentionPreviousProfile = [Environment]::GetEnvironmentVariable('GGML_HIP_OP_PROFILE','Process')
if (-not $LogPath) { $LogPath = Join-Path $PSScriptRoot "hip-attention-$AttentionKernel.log" }
try {
    $env:GGML_HIP_FA_EXPERIMENT = $AttentionKernel
    $env:GGML_HIP_FA_MMA_TILE = $MmaColumns
    $env:GGML_HIP_FA_KV_SPLIT = $KvSplit
    $env:GGML_HIP_OP_PROFILE = '0'
    & (Join-Path $PSScriptRoot 'Start-ROCmServer.ps1') -BackendDirectory $attentionBackend `
        -DraftTokens 2 -KvCache f16 -Threads 8 -BatchThreads 8 -MicroBatchSize 512 `
        -LogPath $LogPath -ListDevices:$ListDevices
}
finally {
    [Environment]::SetEnvironmentVariable('GGML_HIP_FA_KV_SPLIT',$attentionPreviousSplit,'Process')
    [Environment]::SetEnvironmentVariable('GGML_HIP_FA_MMA_TILE',$attentionPreviousTile,'Process')
    [Environment]::SetEnvironmentVariable('GGML_HIP_FA_EXPERIMENT',$attentionPreviousMode,'Process')
    [Environment]::SetEnvironmentVariable('GGML_HIP_OP_PROFILE',$attentionPreviousProfile,'Process')
}
