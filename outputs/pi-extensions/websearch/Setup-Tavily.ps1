$ErrorActionPreference = 'Stop'
$destination = Join-Path $env:USERPROFILE '.pi\agent\websearch.json'
$secret = Read-Host 'Tavily API key (hidden; saved locally in your Pi profile)' -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
    if ($plainKey -notmatch '^tvly-\S+$') { throw 'Expected a Tavily key starting with tvly-. No file written.' }
    $json = @{ tavilyApiKey = $plainKey } | ConvertTo-Json
    [IO.File]::WriteAllText($destination, $json, (New-Object Text.UTF8Encoding($false)))
    Write-Host 'Key saved in your private Pi profile (plaintext JSON, outside Git). Use /reload in Pi, then /websearch-status.'
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $plainKey = $null
    $json = $null
    $secret.Dispose()
}
