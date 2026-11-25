@echo off
rem Build the windows-release preset (LichtFeld-Studio.exe).
rem Usage: scripts\build-windows-release.cmd [clean]
rem
rem With Chinese MSVC, cl reports its /showIncludes notes in UTF-8 and the
rem compiler cache garbles them. Unless CMake generates under code page 65001
rem with the cache off, ninja records no header dependencies and header edits
rem never recompile their users.
setlocal

chcp 65001 >nul
for %%i in ("%~dp0..") do set "ROOT=%%~fi"
set "BUILD=%ROOT%\build-windows-release"

if defined VCINSTALLDIR goto :have_msvc
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
    echo Visual Studio with the C++ x64 tools was not found.
    exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
chcp 65001 >nul
:have_msvc

rem Clean before configuring: clean also deletes the version .rc files that configure generates.
if /i not "%~1"=="clean" goto :check_rules
cmake --build "%BUILD%" --target clean
if errorlevel 1 exit /b 1
goto :configure

:check_rules
rem If rules.ninja was previously generated under a non-UTF-8 code page (e.g. CP936),
rem force CMake to regenerate it under 65001 and reset .ninja_deps.
if exist "%BUILD%\CMakeFiles\rules.ninja" (
    powershell -NoProfile -Command "$p = '%BUILD%\CMakeFiles\rules.ninja'; $utf8 = [System.Text.UTF8Encoding]::new($false, $true); $bytes = [System.IO.File]::ReadAllBytes($p); try { $text = $utf8.GetString($bytes) } catch { Remove-Item -Force $p, '%BUILD%\.ninja_deps' -ErrorAction SilentlyContinue; Write-Host 'Regenerating rules.ninja under UTF-8 (65001)...' }"
    if errorlevel 1 exit /b 1
)

:configure
pushd "%ROOT%"
cmake --preset windows-release -DENABLE_COMPILER_CACHE=OFF >nul
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" exit /b %RC%

rem Purge any first-party .obj whose .ninja_deps record has #deps 0 (e.g. from an
rem earlier IDE/CP936/cached build) so header edits never leave stale objects behind.
if exist "%BUILD%\.ninja_deps" if exist "%BUILD%\CMakeCache.txt" (
    powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; $build = '%BUILD%'; $line = Select-String -Path (Join-Path $build 'CMakeCache.txt') -Pattern '^CMAKE_MAKE_PROGRAM:FILEPATH=(.+)$' | Select-Object -First 1; if (-not $line) { exit 0 }; $ninja = $line.Matches[0].Groups[1].Value.Trim(); if (-not (Test-Path $ninja)) { exit 0 }; $depsOut = & $ninja -C $build -t deps; if ($LASTEXITCODE -ne 0) { throw 'ninja -t deps failed' }; $stale = @(); foreach ($row in $depsOut) { if ($row -match '^((?:CMakeFiles|src)/.+\.(?:c|cc|cpp|cxx|cu)\.obj):\s+#deps 0,') { $obj = Join-Path $build $Matches[1]; if (Test-Path $obj) { $stale += $obj } } }; if ($stale.Count -gt 0) { Write-Host ('Purging ' + $stale.Count + ' zero-dep object file(s) to restore header dependency tracking...'); Remove-Item -Force -LiteralPath $stale }"
    if errorlevel 1 exit /b 1
)

cmake --build "%BUILD%" --target LichtFeld-Studio
if errorlevel 1 exit /b 1

"%BUILD%\LichtFeld-Studio.exe" --version
if errorlevel 1 (
    echo Smoke test failed: LichtFeld-Studio.exe --version returned errorlevel %ERRORLEVEL%.
    exit /b 1
)

echo.
echo Built %BUILD%\LichtFeld-Studio.exe
