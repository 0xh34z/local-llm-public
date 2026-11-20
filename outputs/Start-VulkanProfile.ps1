$ErrorActionPreference = 'Stop'
$profileExe = Join-Path (Split-Path -Parent $PSScriptRoot) 'work\backends\llama-b11888-bin-win-vulkan-x64\llama-server.exe'
$profileModel = $(if ($env:LLM_MODEL_PATH) { $env:LLM_MODEL_PATH } else { Join-Path $HOME 'Models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf' })
$profileLog = Join-Path $PSScriptRoot 'vulkan-profile.log'
$profileVariables = @{
    GGML_VK_PERF_LOGGER = '1'
    GGML_VK_PERF_LOGGER_CONCURRENT = '1'
    GGML_VK_PERF_LOGGER_FREQUENCY = '32'
}
$profilePrevious = @{}
foreach ($profileName in $profileVariables.Keys) {
    $profilePrevious[$profileName] = [Environment]::GetEnvironmentVariable($profileName, 'Process')
}
try {
    foreach ($profileName in $profileVariables.Keys) {
        [Environment]::SetEnvironmentVariable($profileName, $profileVariables[$profileName], 'Process')
    }
    Write-Host "Diagnostic Vulkan profiling; timing overhead is expected. Log: $profileLog"
    $profileArguments = @(
        '-m', $profileModel,
        '-c', '163840', '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja',
        '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '512', '-t', '8',
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '1',
        '--port', '9931', '-lm', 'auto'
    )
    # Keep the server in the foreground and retain both its output streams.
    # Windows PowerShell 5.1 wraps native stderr lines as ErrorRecords.
    # llama.cpp logs normally to stderr, so Stop would abort on the first log.
    $profileSavedErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $profileExe @profileArguments 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $profileLog
        $profileExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $profileSavedErrorAction
    }
    if ($profileExitCode -ne 0) {
        throw "llama-server exited with code $profileExitCode; see $profileLog"
    }
}
finally {
    foreach ($profileName in $profileVariables.Keys) {
        [Environment]::SetEnvironmentVariable($profileName, $profilePrevious[$profileName], 'Process')
    }
}
