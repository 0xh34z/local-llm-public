param(
    [switch]$ListDevices,
    [ValidateSet(1,2,3)][int]$DraftTokens = 2,
    [ValidateSet('f16','q8_0')][string]$KvCache = 'f16',
    [string]$BackendDirectory,
    [string]$LogPath,
    [ValidateRange(32,8192)][int]$BatchSize = 2048,
    [ValidateRange(32,8192)][int]$MicroBatchSize = 512,
    [ValidateRange(1,64)][int]$Threads = 8,
    [ValidateRange(1,64)][int]$BatchThreads = 8,
    [ValidateRange(0,64)][int]$DraftThreads = 0,
    [string]$SlotSavePath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'work\slot-cache'),
    [ValidateRange(0,5)][int]$LogVerbosity = 3
)
$ErrorActionPreference = 'Stop'
$rocmWorkspace = Split-Path -Parent $PSScriptRoot
$rocmBackendDirectory = Join-Path $rocmWorkspace 'work\backends\llama-b11888-bin-win-rocm-10.0-x64'
if ($BackendDirectory) { $rocmBackendDirectory = [IO.Path]::GetFullPath($BackendDirectory) }
$rocmExecutable = Join-Path $rocmBackendDirectory 'llama-server.exe'
$rocmCoreBin = Join-Path $rocmWorkspace 'work\rocm10-runtime\Lib\site-packages\_rocm_sdk_core\bin'
$rocmLibrariesBin = Join-Path $rocmWorkspace 'work\rocm10-runtime\Lib\site-packages\_rocm_sdk_libraries\bin'
$rocmModel = 'C:\Users\kasper\Desktop\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf'
$rocmLog = Join-Path $PSScriptRoot 'rocm-server.log'
if ($LogPath) { $rocmLog = [System.IO.Path]::GetFullPath($LogPath) }
foreach ($rocmRequired in @($rocmExecutable, $rocmCoreBin, $rocmLibrariesBin)) {
    if (-not (Test-Path -LiteralPath $rocmRequired)) { throw "Required path missing: $rocmRequired" }
}
$rocmPreviousPath = $env:PATH
$rocmPreviousPreference = $ErrorActionPreference
$rocmDraftThreads = $DraftThreads
if ($rocmDraftThreads -eq 0) { $rocmDraftThreads = $Threads }
try {
    $env:PATH = "$rocmCoreBin;$rocmLibrariesBin;$rocmPreviousPath"
    if ($ListDevices) {
        $rocmLog = Join-Path $PSScriptRoot 'rocm-devices.log'
        $rocmArguments = @('-lv', '4', '--list-devices')
    } else {
        $rocmArguments = @(
            '-m', $rocmModel, '--device', 'ROCm0',
            '-c', '163840', '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja',
            '-ctk', $KvCache, '-ctv', $KvCache, '-b', "$BatchSize", '-ub', "$MicroBatchSize",
            '-t', "$Threads", '-tb', "$BatchThreads", '-td', "$rocmDraftThreads", '-tbd', "$BatchThreads",
            '--spec-type', 'draft-mtp', '--spec-draft-n-max', "$DraftTokens",
            '--port', '9931', '-lm', 'auto', '-lv', "$LogVerbosity"
        )
        if ($SlotSavePath) {
            $rocmSlotDirectory = [System.IO.Path]::GetFullPath($SlotSavePath)
            if (-not (Test-Path -LiteralPath $rocmSlotDirectory)) {
                New-Item -ItemType Directory -Path $rocmSlotDirectory -ErrorAction Stop | Out-Null
            }
            $rocmArguments += @('--slot-save-path', $rocmSlotDirectory)
        }
        Write-Host "ROCm0 server with isolated ROCm 10 libraries. Log: $rocmLog"
    }
    $ErrorActionPreference = 'Continue'
    & $rocmExecutable @rocmArguments 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $rocmLog
    $rocmExitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $rocmPreviousPath
    $ErrorActionPreference = $rocmPreviousPreference
}
if ($rocmExitCode -ne 0) { throw "ROCm server exited with code $rocmExitCode; see $rocmLog" }
