# Short entry point for the standard local LLM server.
#   .\llm.ps1                      start the server
#   .\llm.ps1 -DownloadModel       download the recommended model into .\models
#   .\llm.ps1 -Update              install the latest official llama.cpp release, smoke-test it, make it active
#   .\llm.ps1 -Update -Tag b12000  install a specific release
param(
    [switch]$ListDevices,
    [switch]$Update,
    [switch]$DownloadModel,
    [string]$Tag
)

# File name of the model in .\models. Leave empty to use the only .gguf in that folder.
$ModelName = ''

if ($DownloadModel) {
    & python (Join-Path $PSScriptRoot 'outputs\download_model.py')
    return
}

$modelsDir = Join-Path $PSScriptRoot 'models'
if (-not $ModelName) {
    $found = @(Get-ChildItem -LiteralPath $modelsDir -Filter *.gguf -ErrorAction SilentlyContinue)
    if ($found.Count -eq 1) { $ModelName = $found[0].Name }
    elseif ($found.Count -gt 1) { throw "Several .gguf files in $modelsDir; set `$ModelName in llm.ps1 to choose one." }
}
$modelPath = Join-Path $modelsDir $ModelName
$haveModel = $ModelName -and (Test-Path -LiteralPath $modelPath)

if ($Update) {
    $updateArgs = @((Join-Path $PSScriptRoot 'outputs\update_backend.py'))
    if ($Tag) { $updateArgs += @('--tag', $Tag) }
    if ($haveModel) { $updateArgs += @('--model', $modelPath) } else { $updateArgs += '--skip-test' }
    & python @updateArgs
    return
}

if (-not $ListDevices -and -not $haveModel) {
    throw "No model found in $modelsDir.`nRun .\llm.ps1 -DownloadModel, or place a .gguf there (and set `$ModelName if there are several)."
}

& (Join-Path $PSScriptRoot 'outputs\Start-ROCmServer.ps1') -ListDevices:$ListDevices -ModelPath $modelPath
