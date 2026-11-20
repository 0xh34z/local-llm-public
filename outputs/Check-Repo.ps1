# Run every local repository check from the repository root.
#
# Requires: Python 3.11+, Node.js 22+, npm, and the Python packages in
# outputs/requirements.txt plus "gguf" for the byte_budget unit test.
# The websearch loader test skips itself when Pi is not installed.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$failures = @()

function Invoke-Check {
    param([string]$Name, [string]$File, [string[]]$Arguments)
    Write-Host ''
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $File @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) { $failures += "$Name (exit $code)" }
}

# Parse every PowerShell script without executing it.
Write-Host ''
Write-Host '=== PowerShell parse ===' -ForegroundColor Cyan
$parseErrors = @()
Get-ChildItem -Path . -Recurse -Filter *.ps1 | Where-Object {
    'work' -notin $_.FullName.Split('\')
} | ForEach-Object {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count -gt 0) {
        $parseErrors += $_.FullName
        $errors | ForEach-Object { Write-Host "  line $($_.Extent.StartLineNumber): $($_.Message)" }
    }
}
if ($parseErrors.Count -gt 0) { $failures += "PowerShell parse ($($parseErrors.Count) files)" }

Invoke-Check 'Python compile' 'python' @('-m', 'compileall', '-q', 'outputs')
Invoke-Check 'JSON validation' 'python' @('outputs\check_json.py')
Invoke-Check 'SHA-256 validation' 'python' @('outputs\check_hashes.py')
Invoke-Check 'Markdown links' 'python' @('outputs\check_links.py', '--all')
Invoke-Check 'byte_budget tests' 'python' @('outputs\test_byte_budget.py')
Invoke-Check 'Performance extension tests' 'node' @('--test', 'outputs\pi-extensions\performance\performance.test.mjs')
Invoke-Check 'Websearch dependencies' 'npm' @('ci', '--prefix', 'outputs\pi-extensions\websearch')
Invoke-Check 'Websearch extension tests' 'node' @('--test', 'outputs\pi-extensions\websearch\websearch.test.mjs')
Invoke-Check 'Working tree whitespace' 'git' @('diff', '--check')

Write-Host ''
if ($failures.Count -gt 0) {
    Write-Host 'FAILED:' -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host 'All repository checks passed.' -ForegroundColor Green
