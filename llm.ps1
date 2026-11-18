# Short entry point for the standard local LLM server.
param(
    [switch]$ListDevices
)

& (Join-Path $PSScriptRoot 'outputs\Start-ROCmServer.ps1') @PSBoundParameters
