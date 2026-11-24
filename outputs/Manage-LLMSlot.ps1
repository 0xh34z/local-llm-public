param(
    [Parameter(Mandatory=$true)][ValidateSet('Save','Restore')][string]$Action,
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name = 'pi-session'
)
$ErrorActionPreference = 'Stop'
$slotCacheDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'work\slot-cache'
$slotCacheFile = "$Name.bin"
$slotMetadataFile = Join-Path $slotCacheDirectory "$Name.json"
$slotBaseUrl = 'http://127.0.0.1:9931'
$slotStatus = Invoke-RestMethod -Uri "$slotBaseUrl/slots" -TimeoutSec 20
if (@($slotStatus).Count -ne 1 -or @($slotStatus | Where-Object { $_.is_processing }).Count -gt 0) {
    throw 'This helper requires one idle slot. Pause Pi before using it.'
}
$slotProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" | Where-Object {
    $_.CommandLine -match '--port\s+9931(?:\s|$)'
})
if ($slotProcesses.Count -ne 1) { throw 'Cannot uniquely identify the local server on port 9931.' }
$slotCommandLine = $slotProcesses[0].CommandLine
$slotPathMatch = [regex]::Match($slotCommandLine, '--slot-save-path\s+(?:"([^"]+)"|(\S+))')
if (-not $slotPathMatch.Success) { throw 'Launch the server with -SlotSavePath pointing to work\slot-cache first.' }
$slotActualDirectory = $slotPathMatch.Groups[1].Value
if (-not $slotActualDirectory) { $slotActualDirectory = $slotPathMatch.Groups[2].Value }
if ([IO.Path]::GetFullPath($slotActualDirectory).TrimEnd('\') -ne [IO.Path]::GetFullPath($slotCacheDirectory).TrimEnd('\')) {
    throw 'The running server uses a different slot-save directory.'
}
$slotProps = Invoke-RestMethod -Uri "$slotBaseUrl/props" -TimeoutSec 20
$slotWeights = Get-Item -LiteralPath $slotProps.model_path -ErrorAction Stop
$slotFingerprint = [ordered]@{
    ModelPath = $slotProps.model_path
    ModelBytes = $slotWeights.Length
    ModelModifiedUtc = $slotWeights.LastWriteTimeUtc.ToString('o')
    Build = $slotProps.build_info
    Context = $slotProps.default_generation_settings.n_ctx
}
foreach ($slotFlag in @('-ctk','-ctv','-b','-ub')) {
    $slotFlagMatch = [regex]::Match($slotCommandLine, '(?:^|\s)' + [regex]::Escape($slotFlag) + '\s+(\S+)')
    if (-not $slotFlagMatch.Success) { throw "Cannot verify server flag $slotFlag." }
    $slotFingerprint[$slotFlag] = $slotFlagMatch.Groups[1].Value
}
foreach ($slotOptionalFlag in @('--spec-type','--spec-draft-n-max')) {
    $slotFlagMatch = [regex]::Match($slotCommandLine, '(?:^|\s)' + [regex]::Escape($slotOptionalFlag) + '\s+(\S+)')
    $slotFingerprint[$slotOptionalFlag] = if ($slotFlagMatch.Success) { $slotFlagMatch.Groups[1].Value } else { $null }
}
$slotFingerprintJson = ConvertTo-Json -InputObject $slotFingerprint -Compress
if ($Action -eq 'Restore') {
    if (-not (Test-Path -LiteralPath $slotMetadataFile)) { throw "Snapshot metadata missing: $slotMetadataFile" }
    $slotSavedMetadata = Get-Content -LiteralPath $slotMetadataFile -Raw | ConvertFrom-Json
    if ($slotSavedMetadata.Fingerprint -ne $slotFingerprintJson) {
        throw 'Model/build/context/cache/draft/batch settings differ from the saved slot. Start the matching configuration.'
    }
}
$slotWatch = [Diagnostics.Stopwatch]::StartNew()
$slotResponse = Invoke-RestMethod -Method Post -Uri "$slotBaseUrl/slots/0?action=$($Action.ToLowerInvariant())" `
    -ContentType 'application/json' -Body (@{ filename = $slotCacheFile } | ConvertTo-Json) -TimeoutSec 600
$slotWatch.Stop()
if ($Action -eq 'Save') {
    [ordered]@{
        Fingerprint = $slotFingerprintJson
        SavedAtUtc = [DateTime]::UtcNow.ToString('o')
        Filename = $slotCacheFile
        ServerResponse = $slotResponse
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $slotMetadataFile -Encoding UTF8
}
[pscustomobject]@{
    Action = $Action
    Name = $Name
    Seconds = [Math]::Round($slotWatch.Elapsed.TotalSeconds,3)
    File = Join-Path $slotCacheDirectory $slotCacheFile
    Response = $slotResponse
} | ConvertTo-Json -Depth 5
