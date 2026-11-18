param(
    [ValidateSet('Configure','Build','All')][string]$Action = 'All',
    [ValidateRange(1,8)][int]$Jobs = 4,
    [string]$LogDirectory,
    [switch]$BuildAttentionTests
)
$ErrorActionPreference = 'Stop'
$diagnosticRoot = Split-Path -Parent $PSScriptRoot
$diagnosticSource = Join-Path $diagnosticRoot 'work\llama.cpp'
$diagnosticBuild = Join-Path $diagnosticRoot 'work\hip-diagnostic-build'
$diagnosticSdkTool = Join-Path $diagnosticRoot 'work\rocm10-build-tools\Scripts\rocm-sdk.exe'
$diagnosticLogs = Join-Path $PSScriptRoot 'hip-diagnostic-build'
if ($LogDirectory) { $diagnosticLogs = [IO.Path]::GetFullPath($LogDirectory) }
$diagnosticCompilerCore = Join-Path $diagnosticRoot 'work\rocm10-build-tools\Lib\site-packages\_rocm_sdk_core\bin'
$diagnosticCompilerLibraries = Join-Path $diagnosticRoot 'work\rocm10-build-tools\Lib\site-packages\_rocm_sdk_libraries\bin'
$diagnosticVs = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
$diagnosticVcVars = Join-Path $diagnosticVs 'VC\Auxiliary\Build\vcvars64.bat'
$diagnosticNinja = Join-Path $diagnosticVs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
New-Item -ItemType Directory -Path $diagnosticLogs -Force | Out-Null
$diagnosticCommit = (& git -C $diagnosticSource rev-parse HEAD).Trim()
if ($diagnosticCommit -ne '48c9f08b7b25766324282b7605177e2f2be755bb') { throw 'Unexpected llama.cpp source commit' }
$diagnosticSavedEnvironment = @{}
function Set-DiagnosticEnvironment([string]$Key, [string]$Value) {
    if (-not $diagnosticSavedEnvironment.ContainsKey($Key)) {
        $diagnosticSavedEnvironment[$Key] = [Environment]::GetEnvironmentVariable($Key,'Process')
    }
    [Environment]::SetEnvironmentVariable($Key,$Value,'Process')
}
try {
    $diagnosticVcCommand = 'call "' + $diagnosticVcVars + '" >nul && set'
    $diagnosticVcLines = & $env:COMSPEC /d /c $diagnosticVcCommand
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment initialization failed' }
    foreach ($diagnosticLine in $diagnosticVcLines) {
        if ($diagnosticLine -match '^([^=]+)=(.*)$') {
            $diagnosticKey = $Matches[1]
            $diagnosticValue = $Matches[2]
            if ([Environment]::GetEnvironmentVariable($diagnosticKey,'Process') -ne $diagnosticValue) {
                Set-DiagnosticEnvironment $diagnosticKey $diagnosticValue
            }
        }
    }
    $diagnosticSdk = (& $diagnosticSdkTool path --root).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $diagnosticSdk) { throw 'ROCm development tree is unavailable' }
    Set-DiagnosticEnvironment 'HIP_PATH' $diagnosticSdk
    Set-DiagnosticEnvironment 'ROCM_PATH' $diagnosticSdk
    Set-DiagnosticEnvironment 'PATH' "$diagnosticCompilerCore;$diagnosticCompilerLibraries;$diagnosticSdk\bin;$env:PATH"
    $diagnosticSdkForward = $diagnosticSdk.Replace('\','/')
    $diagnosticArguments = @(
        '-S', $diagnosticSource, '-B', $diagnosticBuild, '-G', 'Ninja',
        "-DCMAKE_MAKE_PROGRAM=$diagnosticNinja", '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_PREFIX_PATH=$diagnosticSdk", "-DHIP_PATH=$diagnosticSdk",
        "-DCMAKE_C_COMPILER=$diagnosticSdk\lib\llvm\bin\clang.exe",
        "-DCMAKE_CXX_COMPILER=$diagnosticSdk\lib\llvm\bin\clang++.exe",
        '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types',
        "-DCMAKE_CXX_FLAGS=--hip-path=$diagnosticSdkForward --rocm-device-lib-path=$diagnosticSdkForward/lib/llvm/amdgcn/bitcode",
        '-DGGML_BACKEND_DL=ON', '-DGGML_NATIVE=OFF', '-DGGML_CPU=OFF',
        '-DGGML_HIP=ON', '-DGGML_HIP_GRAPHS=ON', '-DAMDGPU_TARGETS=gfx1152',
        '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_TOOLS=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF'
    )
    [ordered]@{ Commit=$diagnosticCommit; Target='gfx1152'; Jobs=$Jobs; Arguments=$diagnosticArguments } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $diagnosticLogs 'configuration.json') -Encoding utf8
    $ErrorActionPreference = 'Continue'
    if ($Action -in @('Configure','All')) {
        & cmake @diagnosticArguments 2>&1 | ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath (Join-Path $diagnosticLogs 'configure.log')
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }
    }
    if ($Action -in @('Build','All')) {
        & cmake --build $diagnosticBuild --target ggml-hip --parallel $Jobs 2>&1 |
            ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $diagnosticLogs 'build.log') |
            Where-Object { $_ -match '^\[[0-9]+/|^FAILED:|error:|^ninja:' }
        if ($LASTEXITCODE -ne 0) { throw "HIP backend build failed: $LASTEXITCODE" }
        if ($BuildAttentionTests) {
            & cmake --build $diagnosticBuild --target ggml --parallel $Jobs 2>&1 |
                ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $diagnosticLogs 'ggml-build.log')
            if ($LASTEXITCODE -ne 0) { throw 'ggml library build failed' }
            $diagnosticTest = Join-Path $diagnosticBuild 'bin\test-attention-probe.exe'
            $diagnosticTestArguments = @(
                '-std=c++17', '-O2', '-DNDEBUG', '-DGGML_SHARED', '-DGGML_BACKEND_DL',
                '-I', (Join-Path $diagnosticSource 'ggml\include'),
                (Join-Path $diagnosticSource 'tests\test-backend-ops.cpp'),
                (Join-Path $diagnosticBuild 'ggml\src\ggml.lib'),
                (Join-Path $diagnosticBuild 'ggml\src\ggml-base.lib'), '-o', $diagnosticTest
            )
            & "$diagnosticSdk\lib\llvm\bin\clang++.exe" @diagnosticTestArguments 2>&1 |
                ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $diagnosticLogs 'attention-test-build.log')
            if ($LASTEXITCODE -ne 0) { throw 'Attention test executable build failed' }
        }
    }
}
finally {
    foreach ($diagnosticKey in $diagnosticSavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($diagnosticKey,$diagnosticSavedEnvironment[$diagnosticKey],'Process')
    }
}
