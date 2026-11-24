param(
    [string]$BackendDirectory,
    [ValidateRange(1, 65535)][int]$Port = 9931,
    [ValidateSet('ROCm0', 'Vulkan0')][string]$Device = 'ROCm0'
)
$ErrorActionPreference = 'Stop'
$rocmWorkspace = Split-Path -Parent $PSScriptRoot
$rocmBackendDirectory = Join-Path $rocmWorkspace 'work\backends\llama-b11888-bin-win-rocm-10.0-x64'
$rocmCurrentFile = Join-Path $rocmWorkspace 'work\backends\current.txt'
if (Test-Path -LiteralPath $rocmCurrentFile) {
    $rocmCurrent = (Get-Content -LiteralPath $rocmCurrentFile -Raw).Trim()
    if ($rocmCurrent) { $rocmBackendDirectory = Join-Path $rocmWorkspace "work\backends\$rocmCurrent" }
}
if ($BackendDirectory) { $rocmBackendDirectory = [IO.Path]::GetFullPath($BackendDirectory) }
$rocmExpectedExe = Join-Path $rocmBackendDirectory 'llama-server.exe'
$rocmMatches = @(Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" | Where-Object {
    $_.ExecutablePath -eq $rocmExpectedExe -and
    $_.CommandLine -match "--port\s+$Port(?:\s|$)" -and
    $_.CommandLine -match "--device\s+$Device(?:\s|$)"
})
if ($rocmMatches.Count -eq 0) { Write-Output 'No matching ROCm server is running.'; exit 0 }
if ($rocmMatches.Count -ne 1) { throw 'More than one matching process; refusing to stop any.' }
$rocmSlots = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/slots" -TimeoutSec 20
if (@($rocmSlots | Where-Object { $_.is_processing }).Count -gt 0) {
    throw 'The ROCm server is processing a request. Wait for it to finish before stopping.'
}
Stop-Process -Id $rocmMatches[0].ProcessId -ErrorAction Stop
Write-Output "Stopped ROCm server PID $($rocmMatches[0].ProcessId)."
