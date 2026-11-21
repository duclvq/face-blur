@echo off
rem Face blur launcher for Windows.
rem First run: finds Python (installs it with winget if missing), creates a
rem local .venv next to this file and installs the requirements into it (plus
rem the NVIDIA GPU packages when nvidia-smi finds a GPU). Later runs reuse it.
rem
rem Usage:
rem   run.bat INPUT [OUTPUT] [--method pixelate] [--strength 2] [--cpu] ...
rem   or drag a folder or a video onto run.bat
rem
rem INPUT can be a folder of images, a folder of videos, or one video file.
rem Without OUTPUT, results go next to the input: FOLDER_blurred, or
rem NAME_blurred.EXT for a single video.

setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "VPY=%VENV%\Scripts\python.exe"
rem Drag and drop or no OUTPUT: keep the window open at the end.
set "PAUSE_AT_END="
if "%~2"=="" set "PAUSE_AT_END=1"

if exist "%VENV%\.installed-v2" goto args

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
echo ok> "%VENV%\.installed-v2"
echo [setup] Done.

:args
if "%~1"=="" (
    echo Usage: run.bat INPUT [OUTPUT] [--method gaussian^|pixelate] [--strength 1-3] [--cpu]
    echo    INPUT: a folder of images, a folder of videos, or one video file
    echo    or: drag a folder or a video onto run.bat
    goto fail
)
if not exist "%~1" (
    echo [error] Input not found: %~1
    goto fail
)

set "IN=%~f1"
set "INDIR=%~dp1"
set "INNAME=%~n1"
set "INEXT=%~x1"
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
rem Remaining args (options) are passed through; note --cpu.
set "REST="
set "GPU=1"
:loop
if "%~1"=="" goto pick
if /i "%~1"=="--cpu" set "GPU=0"
set "REST=%REST% %1"
shift
goto loop

:pick
rem --- videos -> blur_video.py, images -> blur_faces.py ---
set "MODEFILE=%TEMP%\faceblur_mode_%RANDOM%.txt"
"%VPY%" -c "import sys,pathlib;p=pathlib.Path(sys.argv[1]);V={'.mp4','.mov','.avi','.mkv','.webm','.m4v'};I={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'};fs=[f.suffix.lower() for f in p.rglob('*') if f.is_file() and not f.name.startswith('.')] if p.is_dir() else [];nv=sum(s in V for s in fs);ni=sum(s in I for s in fs);print(('video' if p.suffix.lower() in V else 'notvideo') if p.is_file() else 'video' if nv and not ni else 'mixed' if nv else 'image')" "%IN%" > "%MODEFILE%"
set /p MODE=<"%MODEFILE%"
del "%MODEFILE%" >nul 2>&1

set "SCRIPT=blur_faces.py"
if "%MODE%"=="video" set "SCRIPT=blur_video.py"
if "%MODE%"=="notvideo" (
    echo [error] A single file must be a video ^(mp4, mov, avi, mkv, webm, m4v^).
    echo         For images, give the folder that contains them.
    goto fail
)
if "%MODE%"=="mixed" (
    echo [info] The folder has images and videos; blurring the images.
    echo        Run: .venv\Scripts\python blur_video.py on it to blur the videos too.
)

if not "%OUT%"=="" goto gpu
if "%MODE%"=="video" if exist "%IN%\" goto out_folder
if "%MODE%"=="video" (
    set "OUT=%INDIR%%INNAME%_blurred%INEXT%"
    goto gpu
)
:out_folder
set "OUT=%IN%_blurred"

:gpu
rem --- NVIDIA GPU: onnxruntime-gpu replaces the CPU onnxruntime (once) ---
if "%GPU%"=="0" goto ffmpeg_check
if exist "%VENV%\.installed-gpu-v2" goto ffmpeg_check
nvidia-smi -L >nul 2>&1 || goto ffmpeg_check
echo [setup] NVIDIA GPU found. Installing GPU packages (first time only)...
"%VPY%" -m pip uninstall -y -q onnxruntime >nul 2>&1
"%VPY%" -m pip install --disable-pip-version-check --prefer-binary --timeout 60 --retries 10 -r "%HERE%requirements-gpu.txt"
if errorlevel 1 (
    echo [warn] Could not install the NVIDIA GPU packages; running on the CPU.
    "%VPY%" -m pip install --disable-pip-version-check --timeout 60 --retries 10 onnxruntime
) else (
    echo ok> "%VENV%\.installed-gpu-v2"
    echo [setup] Done.
)

:ffmpeg_check
if not "%SCRIPT%"=="blur_video.py" goto go
where ffmpeg >nul 2>&1 && goto go
echo [warn] ffmpeg not found: output videos will have no sound.
echo        Install it (winget install ffmpeg) to keep audio and the original format.

:go
"%VPY%" "%HERE%%SCRIPT%" "%IN%" "%OUT%" %REST%
set "RC=%ERRORLEVEL%"
if defined PAUSE_AT_END pause
exit /b %RC%

:setup_fail
echo [error] Setup failed. Fix the error above and run again.
if exist "%VENV%" rmdir /s /q "%VENV%"
:fail
if defined PAUSE_AT_END pause
exit /b 1
