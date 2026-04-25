# build_windows.ps1 — Configure and build Biona Axon on Windows with MSVC
#
# Prerequisites:
#   - Visual Studio 2022 Community (MSVC)
#   - ONNX Runtime extracted to C:\onnxruntime  (or set $OnnxRuntimeRoot)
#   - OpenSSL from Anaconda (or set $OpenSSLRoot)
#
# Usage:  .\build_windows.ps1

param(
    [string]$OnnxRuntimeRoot = "C:\onnxruntime",
    [string]$OpenSSLRoot     = "C:\Users\Home\Anaconda-latest\Library",
    [string]$BuildType       = "Release",
    [string]$BuildDir        = "build_msvc"
)

$ProjectRoot = "$PSScriptRoot\biona\axon"

Write-Host "=== Biona Axon Windows Build ===" -ForegroundColor Cyan
Write-Host "Project:    $ProjectRoot"
Write-Host "Build dir:  $BuildDir"
Write-Host "ONNX root:  $OnnxRuntimeRoot"
Write-Host "OpenSSL:    $OpenSSLRoot"

# Verify ONNX Runtime root
if (-not (Test-Path "$OnnxRuntimeRoot\include\onnxruntime_cxx_api.h")) {
    Write-Error "ONNX Runtime headers not found at $OnnxRuntimeRoot\include\"
    exit 1
}

# Verify OpenSSL
if (-not (Test-Path "$OpenSSLRoot\include\openssl\ssl.h")) {
    Write-Error "OpenSSL headers not found at $OpenSSLRoot\include\openssl\"
    exit 1
}

# Configure
$cmakeArgs = @(
    "-S", $ProjectRoot,
    "-B", "$PSScriptRoot\$BuildDir",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DONNXRUNTIME_ROOT_DIR=$OnnxRuntimeRoot",
    "-DOPENSSL_ROOT_DIR=$OpenSSLRoot",
    "-DOPENSSL_INCLUDE_DIR=$OpenSSLRoot\include",
    "-DOPENSSL_LIBRARIES=$OpenSSLRoot\lib",
    "-DBUILD_TESTING=ON"
)

Write-Host "`n--- CMake configure ---" -ForegroundColor Yellow
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }

# Build
Write-Host "`n--- CMake build ---" -ForegroundColor Yellow
cmake --build "$PSScriptRoot\$BuildDir" --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) { Write-Error "CMake build failed"; exit 1 }

Write-Host "`n=== Build complete ===" -ForegroundColor Green
Write-Host "To run tests: ctest --test-dir $PSScriptRoot\$BuildDir -C $BuildType -V"
