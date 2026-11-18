param(
    [switch]$Profile,
    [switch]$ListDevices,
    [string]$LogPath
)
$ErrorActionPreference = 'Stop'
$hipDiagnosticRoot = Split-Path -Parent $PSScriptRoot
$hipDiagnosticBackend = Join-Path $hipDiagnosticRoot 'work\hip-diagnostic-bin'
$hipDiagnosticPrevious = [Environment]::GetEnvironmentVariable('GGML_HIP_OP_PROFILE','Process')
if (-not $LogPath) {
    $hipDiagnosticName = if ($Profile) { 'server-profile.log' } else { 'server-normal.log' }
    $LogPath = Join-Path (Join-Path $PSScriptRoot 'hip-diagnostic-build') $hipDiagnosticName
}
try {
    $env:GGML_HIP_OP_PROFILE = if ($Profile) { '1' } else { '0' }
    & (Join-Path $PSScriptRoot 'Start-ROCmServer.ps1') -BackendDirectory $hipDiagnosticBackend `
        -DraftTokens 2 -KvCache f16 -Threads 8 -BatchThreads 8 -MicroBatchSize 512 `
        -LogPath $LogPath -ListDevices:$ListDevices
}
finally {
    [Environment]::SetEnvironmentVariable('GGML_HIP_OP_PROFILE',$hipDiagnosticPrevious,'Process')
}
