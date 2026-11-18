param(
    [ValidateSet('auto','double')][string]$KvSplit = 'double',
    [ValidateRange(32,8192)][int]$BatchSize = 2048,
    [ValidateRange(32,8192)][int]$MicroBatchSize = 512,
    [ValidateRange(1,64)][int]$BatchThreads = 8,
    [string]$BackendDirectory,
    [string]$LogPath
)
$ErrorActionPreference = 'Stop'
$splitRoot = Split-Path -Parent $PSScriptRoot
$splitBackend = Join-Path $splitRoot 'work\hip-kv-split-bin'
if ($BackendDirectory) { $splitBackend = [IO.Path]::GetFullPath($BackendDirectory) }
if (-not $LogPath) { $LogPath = Join-Path $PSScriptRoot "kv-split-$KvSplit-server.log" }
$splitSettings = @{
    GGML_HIP_FA_KV_SPLIT = $KvSplit
    GGML_HIP_FA_MMA_TILE = 'auto'
    GGML_HIP_FA_EXPERIMENT = 'auto'
    GGML_HIP_OP_PROFILE = '0'
}
$splitSaved = @{}
try {
    foreach ($splitKey in $splitSettings.Keys) {
        $splitSaved[$splitKey] = [Environment]::GetEnvironmentVariable($splitKey,'Process')
        [Environment]::SetEnvironmentVariable($splitKey,$splitSettings[$splitKey],'Process')
    }
    & (Join-Path $PSScriptRoot 'Start-ROCmServer.ps1') -BackendDirectory $splitBackend `
        -DraftTokens 2 -KvCache f16 -Threads 8 -DraftThreads 8 -BatchThreads $BatchThreads `
        -BatchSize $BatchSize -MicroBatchSize $MicroBatchSize -LogPath $LogPath
}
finally {
    foreach ($splitKey in $splitSaved.Keys) {
        [Environment]::SetEnvironmentVariable($splitKey,$splitSaved[$splitKey],'Process')
    }
}
