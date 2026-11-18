param([string]$BackendDirectory)
$ErrorActionPreference = 'Stop'
$rocmBackendDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'work\backends\llama-b11888-bin-win-rocm-10.0-x64'
if ($BackendDirectory) { $rocmBackendDirectory = [IO.Path]::GetFullPath($BackendDirectory) }
$rocmExpectedExe = Join-Path $rocmBackendDirectory 'llama-server.exe'
$rocmMatches = @(Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" | Where-Object {
    $_.ExecutablePath -eq $rocmExpectedExe -and
    $_.CommandLine -match '--port\s+9931(?:\s|$)' -and
    $_.CommandLine -match '--device\s+ROCm0(?:\s|$)'
})
if ($rocmMatches.Count -eq 0) { Write-Output 'No matching ROCm server is running.'; exit 0 }
if ($rocmMatches.Count -ne 1) { throw 'More than one matching process; refusing to stop any.' }
$rocmSlots = Invoke-RestMethod -Uri 'http://127.0.0.1:9931/slots' -TimeoutSec 20
if (@($rocmSlots | Where-Object { $_.is_processing }).Count -gt 0) {
    throw 'The ROCm server is processing a request. Wait for it to finish before stopping.'
}
Stop-Process -Id $rocmMatches[0].ProcessId -ErrorAction Stop
Write-Output "Stopped ROCm server PID $($rocmMatches[0].ProcessId)."
