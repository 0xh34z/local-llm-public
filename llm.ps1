# Short entry point for the standard local LLM server.
#   .\llm.ps1                 start the server
#   .\llm.ps1 -Update         install the latest official llama.cpp release, smoke-test it, make it active
#   .\llm.ps1 -Update -Tag b12000   install a specific release
param(
    [switch]$ListDevices,
    [switch]$Update,
    [string]$Tag
)

# Put any .gguf in .\models and set its file name here.
$ModelName = 'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'

$modelPath = Join-Path $PSScriptRoot "models\$ModelName"

if ($Update) {
    $updateArgs = @((Join-Path $PSScriptRoot 'outputs\update_backend.py'))
    if ($Tag) { $updateArgs += @('--tag', $Tag) }
    if (Test-Path -LiteralPath $modelPath) { $updateArgs += @('--model', $modelPath) } else { $updateArgs += '--skip-test' }
    & python @updateArgs
    return
}

if (-not $ListDevices -and -not (Test-Path -LiteralPath $modelPath)) {
    throw "Model not found: $modelPath`nPlace the .gguf in $PSScriptRoot\models or edit `$ModelName in llm.ps1."
}

& (Join-Path $PSScriptRoot 'outputs\Start-ROCmServer.ps1') -ListDevices:$ListDevices -ModelPath $modelPath
