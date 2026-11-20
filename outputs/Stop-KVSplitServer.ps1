param(
    [ValidateRange(1, 65535)][int]$Port = 9931
)
$ErrorActionPreference = 'Stop'
$splitStopRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'Stop-ROCmServer.ps1') `
    -BackendDirectory (Join-Path $splitStopRoot 'work\hip-kv-split-bin') `
    -Port $Port -Device ROCm0
