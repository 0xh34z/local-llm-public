$ErrorActionPreference = 'Stop'
$splitStopRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'Stop-ROCmServer.ps1') `
    -BackendDirectory (Join-Path $splitStopRoot 'work\hip-kv-split-bin')
