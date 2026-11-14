@echo off
rem Face blur launcher for Windows.
rem First run: finds Python (installs it with winget if missing), creates a
rem local .venv next to this file and installs the requirements into it.
rem Later runs reuse the .venv.
rem
rem Usage:
rem   run.bat INPUT_DIR [OUTPUT_DIR] [--method pixelate] [--strength 2] ...
rem   or drag a dataset folder onto run.bat (output goes to FOLDER_blurred)

setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "VPY=%VENV%\Scripts\python.exe"

if exist "%VENV%\.installed" goto run

echo [setup] Preparing environment (first run only)...

rem --- find a working Python (skips the Microsoft Store stub) ---
set "PY="
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>&1 && set "PY=python"

if not defined PY (
    echo [setup] Python not found. Installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
    ) else (
        echo [error] Could not install Python automatically.
        echo         Install it from https://www.python.org/downloads/ and run this again.
        goto fail
    )
)

rem --- create venv and install requirements ---
if exist "%VENV%" rmdir /s /q "%VENV%"
echo [setup] Creating virtual environment in %VENV%
%PY% -m venv "%VENV%" || goto setup_fail
echo [setup] Installing requirements...
"%VPY%" -m pip install --disable-pip-version-check -q -r "%HERE%requirements.txt" || goto setup_fail
echo ok> "%VENV%\.installed"
echo [setup] Done.

:run
if "%~1"=="" (
    echo Usage: run.bat INPUT_DIR [OUTPUT_DIR] [--method gaussian^|pixelate] [--strength 1-3]
    echo    or: drag a dataset folder onto run.bat
    goto fail
)

set "PAUSE_AT_END="
if "%~2"=="" set "PAUSE_AT_END=1"
set "IN=%~1"
set "OUT=%~2"
rem No output dir given (or 2nd arg is an option) -> use INPUT_blurred
if "%OUT%"=="" goto default_out
if "%OUT:~0,2%"=="--" goto default_out
shift
shift
goto collect

:default_out
set "OUT=%~f1_blurred"
shift

:collect
rem Remaining args (options) are passed through.
set "REST="
:loop
if "%~1"=="" goto go
set "REST=%REST% %1"
shift
goto loop

:go
"%VPY%" "%HERE%blur_faces.py" "%IN%" "%OUT%" %REST%
set "RC=%ERRORLEVEL%"
if defined PAUSE_AT_END pause
exit /b %RC%

:setup_fail
echo [error] Setup failed. Fix the error above and run again.
if exist "%VENV%" rmdir /s /q "%VENV%"
:fail
pause
exit /b 1
