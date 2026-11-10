@echo off
rem Build the headless trainer (PTX-only, sm80+) and package it in one step.
rem Usage: scripts\package-windows-trainer.cmd [config.json] [geosunpre-revision] [output-dir]
rem   config.json          default: %USERPROFILE%\Desktop\config.json
rem   geosunpre-revision   default: HEAD of the sibling ..\GeosunPre checkout
rem   output-dir           default: build-delivery\LFS-trainer-sm80-<timestamp>
rem See build-windows-release.cmd for why the code page and compiler cache matter.
setlocal

chcp 65001 >nul
for %%i in ("%~dp0..") do set "ROOT=%%~fi"
set "BUILD=%ROOT%\build-windows-trainer-ptx80"

set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=%USERPROFILE%\Desktop\config.json"
if not exist "%CONFIG%" (
    echo Training config not found: %CONFIG%
    exit /b 1
)

set "GSREV=%~2"
if not defined GSREV (
    for /f "usebackq delims=" %%i in (`git -C "%ROOT%\..\GeosunPre" rev-parse --short^=7 HEAD 2^>nul`) do set "GSREV=%%i"
)
if not defined GSREV (
    echo Pass the GeosunPre revision as the second argument; ..\GeosunPre was not readable.
    exit /b 1
)

set "OUT=%~3"
if defined OUT goto :have_out
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"`) do set "STAMP=%%i"
set "OUT=%ROOT%\build-delivery\LFS-trainer-sm80-%STAMP%"
:have_out

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
:have_msvc

pushd "%ROOT%"
cmake --preset windows-trainer -B "%BUILD%" -DBUILD_CUDA_PTX_ONLY=ON -DBUILD_CUDA_MIN_SM=80 -DENABLE_COMPILER_CACHE=OFF >nul
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" exit /b %RC%

cmake --build "%BUILD%" --target lfs_train
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0package-windows-trainer.ps1" -BuildDir "%BUILD%" -Config "%CONFIG%" -Output "%OUT%" -GeosunPreRevision "%GSREV%"
exit /b %ERRORLEVEL%
