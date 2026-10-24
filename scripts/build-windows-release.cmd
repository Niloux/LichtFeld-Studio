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
:have_msvc

rem Clean before configuring: clean also deletes the version .rc files that configure generates.
if /i not "%~1"=="clean" goto :configure
cmake --build "%BUILD%" --target clean
if errorlevel 1 exit /b 1

:configure
pushd "%ROOT%"
cmake --preset windows-release -DENABLE_COMPILER_CACHE=OFF >nul
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" exit /b %RC%

cmake --build "%BUILD%" --target LichtFeld-Studio
if errorlevel 1 exit /b 1

echo.
echo Built %BUILD%\LichtFeld-Studio.exe
