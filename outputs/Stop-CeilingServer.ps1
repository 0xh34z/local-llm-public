# Stops the experiment server from Start-CeilingServer.ps1 (ROCm0 or Vulkan0) only while it is idle.
param(
    [Parameter(Mandatory = $true)][string]$BackendDirectory,
    [ValidateRange(1, 65535)][int]$Port = 9931
)
$ErrorActionPreference = 'Stop'
$expected = Join-Path ([IO.Path]::GetFullPath($BackendDirectory)) 'llama-server.exe'
$found = @(Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" | Where-Object {
    $_.ExecutablePath -eq $expected -and $_.CommandLine -match "--port\s+$Port(?:\s|$)" -and
    $_.CommandLine -match '--device\s+(ROCm0|Vulkan0)(?:\s|$)'
})
if ($found.Count -eq 0) { Write-Output 'No matching experiment server is running.'; exit 0 }
if ($found.Count -ne 1) { throw 'More than one matching process; refusing to stop any.' }
$slots = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/slots" -TimeoutSec 20
if (@($slots | Where-Object { $_.is_processing }).Count -gt 0) {
    throw 'The server is processing a request. Wait for it to finish before stopping.'
}
Stop-Process -Id $found[0].ProcessId -ErrorAction Stop
Write-Output "Stopped experiment server PID $($found[0].ProcessId)."
