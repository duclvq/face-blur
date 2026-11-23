@echo off
rem Face blur launcher for Windows.
rem First run: finds Python (installs it with winget if missing), creates a
rem local .venv next to this file and installs the requirements into it.
rem Later runs reuse it.
rem
rem Usage:
rem   run.bat INPUT_FOLDER [OUTPUT_FOLDER] [--method pixelate] [--strength 2] [--device cpu] ...
rem   or drag a folder onto run.bat
rem
rem INPUT_FOLDER is a folder of images. Without OUTPUT_FOLDER, results go to
rem INPUT_FOLDER_blurred next to it. For videos, see tools\blur_video.py.

setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "VPY=%VENV%\Scripts\python.exe"
rem Drag and drop or no OUTPUT: keep the window open at the end.
set "PAUSE_AT_END="
if "%~2"=="" set "PAUSE_AT_END=1"

if exist "%VENV%\.installed-v3" goto args

echo [setup] Preparing environment (first run only, takes a few minutes)...

rem --- find a working Python 3.11-3.14 (the versions onnxruntime supports;
rem     skips the Microsoft Store stub) ---
set "PY="
for %%v in (3.12 3.13 3.11 3.14) do if not defined PY py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
if not defined PY python -c "import sys; sys.exit(not (3,11) <= sys.version_info[:2] <= (3,14))" >nul 2>&1 && set "PY=python"

if not defined PY (
    echo [setup] Python 3.11-3.14 not found. Installing Python 3.12 with winget...
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
"%VPY%" -m pip install --disable-pip-version-check --prefer-binary --timeout 60 --retries 10 -r "%HERE%requirements.txt" || goto setup_fail
echo ok> "%VENV%\.installed-v3"
echo [setup] Done.

:args
if "%~1"=="" (
    echo Usage: run.bat INPUT_FOLDER [OUTPUT_FOLDER] [--method gaussian^|pixelate] [--strength 1-3] [--device auto^|cpu^|gpu^|npu]
    echo    INPUT_FOLDER: a folder of images
    echo    or: drag a folder onto run.bat
    goto fail
)
if not exist "%~1" (
    echo [error] Input not found: %~1
    goto fail
)

set "IN=%~f1"
set "OUT=%~2"
if "%OUT%"=="" goto default_out
if "%OUT:~0,2%"=="--" goto default_out
set "OUT=%~f2"
shift
shift
goto collect

:default_out
set "OUT="
shift

:collect
rem Remaining args (options) are passed through.
set "REST="
:loop
if "%~1"=="" goto pick
set "REST=%REST% %1"
shift
goto loop

:pick
if not exist "%IN%\" (
    echo [error] INPUT must be a folder of images.
    echo         For a video, see tools\blur_video.py.
    goto fail
)
if "%OUT%"=="" set "OUT=%IN%_blurred"

:go
"%VPY%" "%HERE%tools\blur_images.py" "%IN%" "%OUT%" %REST%
set "RC=%ERRORLEVEL%"
if defined PAUSE_AT_END pause
exit /b %RC%

:setup_fail
echo [error] Setup failed. Fix the error above and run again.
if exist "%VENV%" rmdir /s /q "%VENV%"
:fail
if defined PAUSE_AT_END pause
exit /b 1
