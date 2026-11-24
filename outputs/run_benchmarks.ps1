# Run the standard benchmark suite and print a summary table.
#
# Tests:
#   1. 103K populated decode (llama-bench, pp2/pp3 and tg32)
#   2. 103K cached prefix (server slot restore + short continuation), optional
#   3. 160K cached prefix (server slot restore + short continuation), optional
#
# Tests 2 and 3 need a saved slot. Slots are large and contain generated
# conversation state, so this repo does not ship them. Create one with
# Manage-LLMSlot.ps1 or probe_slot_cache.py and pass the file name:
#   .\outputs\run_benchmarks.ps1 -Slot103k my-103k.bin -Slot160k my-160k.bin
# Without those arguments the cached tests are skipped and recorded as skipped.
#
# Usage:
#   .\outputs\run_benchmarks.ps1
#   .\outputs\run_benchmarks.ps1 -Runtime Vulkan -Slot103k my-103k.bin
#   .\outputs\run_benchmarks.ps1 -Port 19931 -Context 163840 -ModelPath D:\Models\model.gguf
#   .\outputs\run_benchmarks.ps1 -OutDir C:\bench-results
#
# Prerequisites (see BUILD.md):
#   - work/hip-kv-split-bin (default HIP) or work/vulkan-bin
#   - work/rocm10-runtime (ROCm wheels) for HIP
#   - Model at $env:LLM_MODEL_PATH or ~\Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
#
# Expected ranges on a Ryzen AI 7 350 / Radeon 860M, 8-block HIP backend:
#   Test 1 tg32 at depth 103194: about 15-17 tokens/s
#   Test 2 cached 103K: about 14-15 tokens/s, prompt cache about 103K tokens
#   Test 3 cached 160K: about 13-15 tokens/s, prompt cache about 160K tokens

param(
    [ValidateSet('hip', 'vulkan')][string]$Runtime = 'hip',
    [ValidateRange(1, 65535)][int]$Port = 9931,
    [ValidateRange(1024, 1048576)][int]$Context = 163840,
    [string]$ModelPath,
    [string]$Device,
    [string]$OutDir,
    [string]$BackendDirectory,
    [string]$Slot103k,
    [string]$Slot160k,
    [ValidateRange(1, 512)][int]$Generate = 28,
    [switch]$SkipDecode
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$baseUrl = "http://127.0.0.1:$Port"
$slotDir = Join-Path $root 'work\slot-cache'
$hipSettings = [ordered]@{
    GGML_HIP_FA_KV_SPLIT = 'double'
    GGML_HIP_FA_MMA_TILE = 'auto'
    GGML_HIP_FA_EXPERIMENT = 'auto'
    GGML_HIP_OP_PROFILE = '0'
}

function ConvertTo-ArgumentString {
    param([string[]]$Items)
    $quoted = foreach ($item in $Items) {
        if ($item -match '[\s"]') { '"' + ($item -replace '"', '\"') + '"' } else { $item }
    }
    return ($quoted -join ' ')
}

function Invoke-JsonPost {
    param([string]$Uri, $Body, [int]$TimeoutSec = 600)
    $json = $Body | ConvertTo-Json -Depth 8 -Compress
    return Invoke-RestMethod -Uri $Uri -Method Post -ContentType 'application/json' -Body $json -TimeoutSec $TimeoutSec
}

function Wait-ServerReady {
    param([string]$Uri, $Process, [int]$TimeoutSeconds = 300)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process -and $Process.HasExited) { return $false }
        try {
            $response = Invoke-WebRequest -Uri "$Uri/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($response.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-BenchRow {
    param($Rows, [int]$Depth, [string]$Kind)
    foreach ($row in $Rows) {
        if ([int]$row.n_depth -ne $Depth) { continue }
        $nPrompt = [int]$row.n_prompt
        $nGen = [int]$row.n_gen
        if ($Kind -eq 'tg' -and $nPrompt -eq 0 -and $nGen -gt 0) { return $row }
        if ($Kind -eq 'pp2' -and $nPrompt -eq 2 -and $nGen -eq 0) { return $row }
        if ($Kind -eq 'pp3' -and $nPrompt -eq 3 -and $nGen -eq 0) { return $row }
    }
    return $null
}

function Resolve-SlotPath {
    param([string]$SlotArgument, [string]$Directory)
    $slotPath = if ([IO.Path]::IsPathRooted($SlotArgument)) { $SlotArgument } else { Join-Path $Directory $SlotArgument }
    $full = [IO.Path]::GetFullPath($slotPath)
    $directoryFull = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
    if ([IO.Path]::GetDirectoryName($full).TrimEnd('\') -ne $directoryFull) {
        throw "Slot files must be inside $Directory because llama-server only reads that directory."
    }
    return $full
}

function Invoke-CachedTest {
    param([string]$SlotArgument, [int]$MinRestored, [string]$OutDirectory)
    $slotFull = Resolve-SlotPath -SlotArgument $SlotArgument -Directory $slotDir
    if (-not (Test-Path -LiteralPath $slotFull)) {
        return [ordered]@{ status = 'skipped'; reason = "Slot file not found: $slotFull" }
    }
    New-Item -ItemType Directory -Path $OutDirectory -Force | Out-Null
    $slotName = [IO.Path]::GetFileName($slotFull)
    $restore = Invoke-JsonPost -Uri "$baseUrl/slots/0?action=restore" -Body @{ filename = $slotName } -TimeoutSec 600
    $restored = 0
    if ($null -ne $restore.n_restored) { $restored = [int]$restore.n_restored }
    if ($restored -lt $MinRestored) {
        throw "Restored $restored tokens from $slotName; expected at least $MinRestored. The slot must match this model, build, context and cache type."
    }
    $body = @{
        prompt = $suffixTokens
        id_slot = 0
        cache_prompt = $true
        n_predict = $Generate
        temperature = 0.6
        top_k = 20
        top_p = 0.95
        min_p = 0.0
        seed = 42
        repeat_penalty = 1.0
        presence_penalty = 0.0
        ignore_eos = $true
        stream = $false
        return_tokens = $true
        timings_per_token = $true
    }
    $response = Invoke-JsonPost -Uri "$baseUrl/completion" -Body $body -TimeoutSec 1800
    if ($null -eq $response.timings) { throw 'Server response did not include timings.' }
    $timings = $response.timings
    $perSecond = $null
    if ($null -ne $timings.predicted_per_second) {
        $perSecond = [double]$timings.predicted_per_second
    } elseif ($null -ne $timings.predicted_ms -and [double]$timings.predicted_ms -gt 0) {
        $perSecond = [double]$timings.predicted_n / ([double]$timings.predicted_ms / 1000.0)
    }
    $predictedRounded = $null
    if ($null -ne $perSecond) { $predictedRounded = [math]::Round($perSecond, 2) }
    $record = [ordered]@{
        status = 'ok'
        slot = $slotName
        restored_tokens = $restored
        prompt_n = $timings.prompt_n
        cache_n = $timings.cache_n
        predicted_n = $timings.predicted_n
        prompt_ms = [math]::Round([double]$timings.prompt_ms, 2)
        predicted_ms = [math]::Round([double]$timings.predicted_ms, 2)
        predicted_per_second = $predictedRounded
        draft_n = $timings.draft_n
        draft_n_accepted = $timings.draft_n_accepted
        truncated = [bool]$response.truncated
    }
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutDirectory 'result.json') -Encoding utf8
    return $record
}

# Resolve the backend and device.
if ($Runtime -eq 'vulkan') {
    $backend = if ($BackendDirectory) { $BackendDirectory } else { Join-Path $root 'work\vulkan-bin' }
    $defaultDevice = 'Vulkan0'
    $dll = 'ggml-vulkan.dll'
} else {
    $kvSplitBackend = Join-Path $root 'work\hip-kv-split-bin'
    if ($BackendDirectory) {
        $backend = $BackendDirectory
    } elseif (Test-Path -LiteralPath $kvSplitBackend) {
        $backend = $kvSplitBackend
    } else {
        $currentFile = Join-Path $root 'work\backends\current.txt'
        if (Test-Path -LiteralPath $currentFile) {
            $currentRel = (Get-Content -LiteralPath $currentFile -Raw).Trim()
            if ($currentRel) { $backend = Join-Path $root "work\backends\$currentRel" }
        }
        if (-not $backend) { $backend = $kvSplitBackend }
    }
    $defaultDevice = 'ROCm0'
    $dll = 'ggml-hip.dll'
}
$device = if ($Device) { $Device } else { $defaultDevice }
$backend = [IO.Path]::GetFullPath($backend)
foreach ($required in @((Join-Path $backend 'llama-server.exe'), (Join-Path $backend 'llama-bench.exe'), (Join-Path $backend $dll))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required backend file missing: $required. See BUILD.md." }
}

$model = $ModelPath
if (-not $model) { $model = $env:LLM_MODEL_PATH }
if (-not $model) {
    $repoModels = @(Get-ChildItem -LiteralPath (Join-Path $root 'models') -Filter *.gguf -ErrorAction SilentlyContinue)
    if ($repoModels.Count -eq 1) { $model = $repoModels[0].FullName }
}
if (-not $model) { $model = Join-Path $HOME 'Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf' }
$model = [IO.Path]::GetFullPath($model)
if (-not (Test-Path -LiteralPath $model)) { throw "Model not found: $model. Set -ModelPath, LLM_MODEL_PATH, or place it under ~\Models." }

if ($OutDir) { $outBase = [IO.Path]::GetFullPath($OutDir) }
else { $outBase = Join-Path $root ("outputs\benchmark-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Path $outBase -Force | Out-Null

$test1 = [ordered]@{ status = 'skipped'; reason = 'SkipDecode was set.' }
$test2 = [ordered]@{ status = 'skipped'; reason = 'No -Slot103k supplied.' }
$test3 = [ordered]@{ status = 'skipped'; reason = 'No -Slot160k supplied.' }

# Validate supplied slots before starting a 24 GB server. A typo should not cost
# several minutes of model load time.
if ($Slot103k) {
    $slotPath = Resolve-SlotPath -SlotArgument $Slot103k -Directory $slotDir
    if (-not (Test-Path -LiteralPath $slotPath)) {
        $test2 = [ordered]@{ status = 'skipped'; reason = "Slot file not found: $slotPath" }
        $Slot103k = $null
    }
}
if ($Slot160k) {
    $slotPath = Resolve-SlotPath -SlotArgument $Slot160k -Directory $slotDir
    if (-not (Test-Path -LiteralPath $slotPath)) {
        $test3 = [ordered]@{ status = 'skipped'; reason = "Slot file not found: $slotPath" }
        $Slot160k = $null
    }
}

if (-not $SkipDecode) {
    Write-Host ''
    Write-Host '=== Test 1: 103K populated decode (llama-bench) ===' -ForegroundColor Cyan
    $test1Dir = Join-Path $outBase 'test1-decode-103k'
    $depthArgs = @{
        OutDirectory = $test1Dir
        Depths = '103194'
        Prompts = '2,3'
        Generate = 32
        Repetitions = 5
        Runtime = $Runtime
        Port = $Port
        ModelPath = $model
        Device = $device
    }
    if ($Runtime -eq 'hip') { $depthArgs['BackendDirectory'] = $backend }
    & (Join-Path $root 'outputs\run_depth_bench.ps1') @depthArgs
    $benchRows = @()
    foreach ($line in Get-Content -LiteralPath (Join-Path $test1Dir 'bench.jsonl')) {
        if (-not $line.Trim()) { continue }
        try { $benchRows += ($line | ConvertFrom-Json) } catch {}
    }
    $tg = Get-BenchRow $benchRows 103194 'tg'
    $pp2 = Get-BenchRow $benchRows 103194 'pp2'
    $pp3 = Get-BenchRow $benchRows 103194 'pp3'
    if ($null -eq $tg) { throw 'No tg row for depth 103194 in bench.jsonl.' }
    $pp2Total = $null
    $pp3Total = $null
    if ($null -ne $pp2) { $pp2Total = [math]::Round([double]$pp2.avg_ns / 1e6, 2) }
    if ($null -ne $pp3) { $pp3Total = [math]::Round([double]$pp3.avg_ns / 1e6, 2) }
    $test1 = [ordered]@{
        status = 'ok'
        depth = 103194
        tg_tok_s = [math]::Round([double]$tg.avg_ts, 2)
        tg_stddev_tok_s = [math]::Round([double]$tg.stddev_ts, 2)
        tg_ms_per_token = [math]::Round(([double]$tg.avg_ns / 1e6) / [int]$tg.n_gen, 2)
        pp2_total_ms = $pp2Total
        pp3_total_ms = $pp3Total
    }
    Write-Host "  tg32: $($test1.tg_tok_s) tokens/s, $($test1.tg_ms_per_token) ms/token"
}

# The cached tests need a server. Start one only if a slot was supplied.
if ($Slot103k -or $Slot160k) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already in use. Stop the existing server first or pass -Port."
    }
    if (-not (Test-Path -LiteralPath $slotDir)) { New-Item -ItemType Directory -Path $slotDir -Force | Out-Null }
    New-Item -ItemType Directory -Path $outBase -Force | Out-Null
    $suffix = "`n<|im_end|>`n<|im_start|>user`nContinue the implementation.`n<|im_end|>`n<|im_start|>assistant`n"

    $serverArgs = @(
        '-m', $model, '--device', $device, '-c', "$Context", '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja',
        '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '512', '-t', '8', '-tb', '8', '-td', '0', '-tbd', '8',
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '--port', "$Port", '-lm', 'auto', '-lv', '2',
        '--slot-save-path', $slotDir
    )
    $savedPath = $env:PATH
    $savedHip = @{}
    $serverProc = $null
    try {
        if ($Runtime -eq 'hip') {
            $env:PATH = "$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_core\bin;$root\work\rocm10-runtime\Lib\site-packages\_rocm_sdk_libraries\bin;$env:PATH"
            foreach ($key in $hipSettings.Keys) {
                $savedHip[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
                [Environment]::SetEnvironmentVariable($key, $hipSettings[$key], 'Process')
            }
        }
        $serverLog = Join-Path $outBase 'server.log'
        $serverErrorLog = Join-Path $outBase 'server-stderr.log'
        $serverProc = Start-Process -FilePath (Join-Path $backend 'llama-server.exe') `
            -ArgumentList (ConvertTo-ArgumentString $serverArgs) -PassThru -NoNewWindow `
            -RedirectStandardOutput $serverLog -RedirectStandardError $serverErrorLog
        if (-not (Wait-ServerReady -Uri $baseUrl -Process $serverProc -TimeoutSeconds 300)) {
            throw "Server did not become ready. See $serverLog and $serverErrorLog."
        }
        $suffixTokens = @((Invoke-JsonPost -Uri "$baseUrl/tokenize" -Body @{ content = $suffix; add_special = $false; parse_special = $true } -TimeoutSec 30).tokens)
        if (-not $suffixTokens -or $suffixTokens.Count -eq 0) {
            throw 'The server tokenizer returned no tokens for the continuation suffix.'
        }
        if ($Slot103k) {
            Write-Host ''
            Write-Host '=== Test 2: 103K cached prefix (server) ===' -ForegroundColor Cyan
            $test2 = Invoke-CachedTest -SlotArgument $Slot103k -MinRestored 100000 -OutDirectory (Join-Path $outBase 'test2-cached-103k')
            Write-Host "  cache_n: $($test2.cache_n), $($test2.predicted_per_second) tokens/s"
        }
        if ($Slot160k) {
            Write-Host ''
            Write-Host '=== Test 3: 160K cached prefix (server) ===' -ForegroundColor Cyan
            $test3 = Invoke-CachedTest -SlotArgument $Slot160k -MinRestored 150000 -OutDirectory (Join-Path $outBase 'test3-cached-160k')
            Write-Host "  cache_n: $($test3.cache_n), $($test3.predicted_per_second) tokens/s"
        }
    } finally {
        if ($serverProc -and -not $serverProc.HasExited) { Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue }
        foreach ($key in $savedHip.Keys) { [Environment]::SetEnvironmentVariable($key, $savedHip[$key], 'Process') }
        $env:PATH = $savedPath
    }
}

$dllHash = (Get-FileHash -LiteralPath (Join-Path $backend $dll) -Algorithm SHA256).Hash
$environment = if ($Runtime -eq 'hip') { $hipSettings } else { [ordered]@{} }
$summary = [ordered]@{
    timestamp = (Get-Date -Format 'o')
    runtime = $Runtime
    port = $Port
    context = $Context
    backend = $backend
    device = $device
    backend_dll = $dll
    backend_sha256 = $dllHash
    model = $model
    model_bytes = (Get-Item -LiteralPath $model).Length
    environment = $environment
    test1_decode_103k = $test1
    test2_cached_103k = $test2
    test3_cached_160k = $test3
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outBase 'summary.json') -Encoding utf8

Write-Host ''
Write-Host '========================================='
Write-Host '  BENCHMARK SUMMARY'
Write-Host '========================================='
Write-Host "  Backend: $Runtime ($device)"
Write-Host "  Model:   $(Split-Path -Leaf $model)"
Write-Host "  DLL:     $dllHash"
Write-Host ''
Write-Host '  Test 1 - 103K populated decode:'
if ($test1.status -eq 'ok') {
    Write-Host "    tg32: $($test1.tg_tok_s) tokens/s ($($test1.tg_ms_per_token) ms/token)"
    Write-Host "    pp2:  $($test1.pp2_total_ms) ms    pp3: $($test1.pp3_total_ms) ms"
} else {
    Write-Host "    $($test1.status): $($test1.reason)"
}
foreach ($pair in @(@('Test 2 - 103K cached', $test2), @('Test 3 - 160K cached', $test3))) {
    Write-Host ''
    Write-Host "  $($pair[0]):"
    if ($pair[1].status -eq 'ok') {
        Write-Host "    cache: $($pair[1].cache_n) tokens    new: $($pair[1].prompt_n)    out: $($pair[1].predicted_n)"
        Write-Host "    speed: $($pair[1].predicted_per_second) tokens/s    TTFT: $($pair[1].prompt_ms) ms"
    } else {
        Write-Host "    $($pair[1].status): $($pair[1].reason)"
    }
}
Write-Host ''
Write-Host "Full results saved to: $outBase"
