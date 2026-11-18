param(
    [ValidateSet('auto','tile','vec','mma')][string]$AttentionKernel = 'auto',
    [ValidateSet('auto','16','64')][string]$MmaColumns = 'auto',
    [ValidateSet('auto','half','double','quad')][string]$KvSplit = 'auto',
    [ValidateSet('test','perf')][string]$Mode = 'test',
    [string]$BackendDirectory,
    [string]$DumpDirectory,
    [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
$probeRoot = Split-Path -Parent $PSScriptRoot
$probeBackend = Join-Path $probeRoot 'work\hip-attention-experiment-bin'
if ($BackendDirectory) { $probeBackend = [IO.Path]::GetFullPath($BackendDirectory) }
$probeCoreBin = Join-Path $probeRoot 'work\rocm10-runtime\Lib\site-packages\_rocm_sdk_core\bin'
$probeLibrariesBin = Join-Path $probeRoot 'work\rocm10-runtime\Lib\site-packages\_rocm_sdk_libraries\bin'
$probeSettings = @{
    PATH = "$probeBackend;$probeCoreBin;$probeLibrariesBin;$env:PATH"
    GGML_BACKEND_PATH = $null
    LLAMA_HIP_ATTENTION_PROBE = '1'
    GGML_HIP_FA_EXPERIMENT = $AttentionKernel
    GGML_HIP_FA_MMA_TILE = $MmaColumns
    GGML_HIP_FA_KV_SPLIT = $KvSplit
    GGML_HIP_OP_PROFILE = '0'
    LLAMA_HIP_ATTENTION_DUMP = $null
}
if ($DumpDirectory) {
    $probeDumpDirectory = [IO.Path]::GetFullPath($DumpDirectory)
    New-Item -ItemType Directory -Path $probeDumpDirectory -Force | Out-Null
    $probeSettings['LLAMA_HIP_ATTENTION_DUMP'] = $probeDumpDirectory
}
$probeSaved = @{}
try {
    foreach ($probeKey in $probeSettings.Keys) {
        $probeSaved[$probeKey] = [Environment]::GetEnvironmentVariable($probeKey,'Process')
        [Environment]::SetEnvironmentVariable($probeKey,$probeSettings[$probeKey],'Process')
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent ([IO.Path]::GetFullPath($LogPath))) -Force | Out-Null
    $ErrorActionPreference = 'Continue'
    & (Join-Path $probeBackend 'test-attention-probe.exe') $Mode -b ROCm0 -o FLASH_ATTN_EXT 2>&1 |
        ForEach-Object { $_.ToString() } | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "Attention $Mode failed for $AttentionKernel (exit $LASTEXITCODE)" }
}
finally {
    foreach ($probeKey in $probeSaved.Keys) {
        [Environment]::SetEnvironmentVariable($probeKey,$probeSaved[$probeKey],'Process')
    }
}
