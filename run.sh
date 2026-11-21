#!/usr/bin/env bash
# Face blur launcher for Linux (and macOS).
# First run: creates a local .venv next to this file and installs the
# requirements into it (GPU extras too when --gpu is given). Later runs reuse it.
#
# Usage:
#   ./run.sh INPUT [OUTPUT] [--method pixelate] [--strength 2] [--gpu] ...
#
# INPUT can be a folder of images, a folder of videos, or one video file.
# Without OUTPUT, results go next to the input: FOLDER_blurred, or
# NAME_blurred.EXT for a single video.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
VPY="$VENV/bin/python"
VIDEO_RE='\.(mp4|mov|avi|mkv|webm|m4v)$'
IMAGE_RE='\.(jpg|jpeg|png|bmp|webp|tif|tiff)$'

usage() {
    echo "Usage: $0 INPUT [OUTPUT] [--method gaussian|pixelate] [--strength 1-3] [--gpu] ..."
    echo "  INPUT: a folder of images, a folder of videos, or one video file"
    exit 1
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac

# --- arguments: INPUT, optional OUTPUT, then options passed through ---
IN="$1"; shift
OUT=""
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then OUT="$1"; shift; fi
GPU=0
for a in "$@"; do [ "$a" = "--gpu" ] && GPU=1; done

[ -e "$IN" ] || { echo "[error] Input not found: $IN"; exit 1; }
IN="$(cd "$(dirname "$IN")" && pwd)/$(basename "$IN")"

# --- pick the script: videos -> blur_video.py, images -> blur_faces.py ---
SCRIPT=blur_faces.py
if [ -f "$IN" ]; then
    if echo "$IN" | grep -qiE "$VIDEO_RE"; then
        SCRIPT=blur_video.py
    else
        echo "[error] A single file must be a video (mp4, mov, avi, mkv, webm, m4v)."
        echo "        For images, give the folder that contains them."
        exit 1
    fi
else
    n_img=$(find "$IN" -type f ! -name '._*' | grep -ciE "$IMAGE_RE" || true)
    n_vid=$(find "$IN" -type f ! -name '._*' | grep -ciE "$VIDEO_RE" || true)
    if [ "$n_vid" -gt 0 ] && [ "$n_img" -eq 0 ]; then
        SCRIPT=blur_video.py
    elif [ "$n_vid" -gt 0 ]; then
        echo "[info] $IN has $n_img image(s) and $n_vid video(s); blurring the images."
        echo "       Run blur_video.py on it to blur the videos too."
    fi
fi

if [ -z "$OUT" ]; then
    if [ -f "$IN" ]; then
        base="$(basename "$IN")"
        OUT="$(dirname "$IN")/${base%.*}_blurred.${base##*.}"
    else
        OUT="${IN%/}_blurred"
    fi
fi

# --- first run: create the venv and install requirements ---
# Returns non-zero if the install fails; the marker file is written only on success.
setup() {
    local req="$1" marker="$2"
    [ -f "$VENV/$marker" ] && return 0
    echo "[setup] Installing $(basename "$req") into $VENV (first run only)..."
    if command -v uv >/dev/null 2>&1; then
        if [ ! -x "$VPY" ]; then uv venv -q "$VENV" || return 1; fi
        uv pip install -q --python "$VPY" -r "$req" || return 1
    else
        local py
        py="$(command -v python3 || command -v python || true)"
        if [ -z "$py" ]; then
            echo "[error] Python 3.8+ not found. Install it (e.g. sudo apt install python3 python3-venv)."
            return 1
        fi
        if [ ! -x "$VPY" ]; then
            "$py" -m venv "$VENV" || {
                echo "[error] Could not create a virtual environment."
                echo "        On Debian/Ubuntu: sudo apt install python3-venv"
                rm -rf "$VENV"; return 1; }
        fi
        "$VPY" -m pip install --disable-pip-version-check -q -r "$req" || return 1
    fi
    echo ok > "$VENV/$marker"
    echo "[setup] Done."
}

setup "$HERE/requirements.txt" .installed || {
    echo "[error] Setup failed. Fix the error above and run again."; exit 1; }
if [ "$GPU" -eq 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        echo "[warn] --gpu: no NVIDIA GPU/driver found (nvidia-smi); running on the CPU."
    elif ! setup "$HERE/requirements-gpu.txt" .installed-gpu; then
        echo "[warn] --gpu: could not install the GPU packages; running on the CPU."
    fi
fi

if [ "$SCRIPT" = blur_video.py ] && ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[warn] ffmpeg not found: output videos will have no sound."
    echo "       Install it (e.g. sudo apt install ffmpeg) to keep audio and the original format."
fi

exec "$VPY" "$HERE/$SCRIPT" "$IN" "$OUT" "$@"
