#!/usr/bin/env bash
# Face blur launcher for Linux (and macOS).
# First run: creates a local .venv next to this file and installs the
# requirements into it. Later runs reuse it. --cpu turns off hardware
# acceleration (Qualcomm NPU).
#
# Usage:
#   ./run.sh INPUT_FOLDER [OUTPUT_FOLDER] [--method pixelate] [--strength 2] [--cpu] ...
#
# INPUT_FOLDER is a folder of images. Without OUTPUT_FOLDER, results go to
# INPUT_FOLDER_blurred next to it. For videos, see tools/blur_video.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
VPY="$VENV/bin/python"

usage() {
    echo "Usage: $0 INPUT_FOLDER [OUTPUT_FOLDER] [--method gaussian|pixelate] [--strength 1-3] [--cpu] ..."
    echo "  INPUT_FOLDER: a folder of images"
    exit 1
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac

# --- arguments: INPUT, optional OUTPUT, then options passed through ---
IN="$1"; shift
OUT=""
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then OUT="$1"; shift; fi

[ -e "$IN" ] || { echo "[error] Input not found: $IN"; exit 1; }
IN="$(cd "$(dirname "$IN")" && pwd)/$(basename "$IN")"

[ -d "$IN" ] || { echo "[error] INPUT must be a folder of images."; echo "        For a video, see tools/blur_video.py."; exit 1; }
[ -n "$OUT" ] || OUT="${IN%/}_blurred"

# --- first run: create the venv and install requirements ---
# Returns non-zero if the install fails; the marker file is written only on success.
setup() {
    local req="$1" marker="$2"
    [ -f "$VENV/$marker" ] && return 0
    echo "[setup] Installing $(basename "$req") into $VENV (first run only)..."
    if command -v uv >/dev/null 2>&1; then
        if [ ! -x "$VPY" ]; then uv venv -q --python ">=3.11,<3.15" "$VENV" || return 1; fi
        uv pip install -q --python "$VPY" -r "$req" || return 1
    else
        local py
        # onnxruntime has wheels for Python 3.11-3.14 only.
        py=""
        for c in python3.12 python3.13 python3.11 python3.14 python3 python; do
            if command -v "$c" >/dev/null 2>&1 && "$c" -c \
                'import sys; sys.exit(not (3,11) <= sys.version_info[:2] <= (3,14))' 2>/dev/null; then
                py="$(command -v "$c")"; break
            fi
        done
        if [ -z "$py" ]; then
            echo "[error] Python 3.11-3.14 not found. Install it (e.g. sudo apt install python3 python3-venv)."
            return 1
        fi
        if [ ! -x "$VPY" ]; then
            "$py" -m venv "$VENV" || {
                echo "[error] Could not create a virtual environment."
                echo "        On Debian/Ubuntu: sudo apt install python3-venv"
                rm -rf "$VENV"; return 1; }
        fi
        "$VPY" -m pip install --disable-pip-version-check --prefer-binary -q -r "$req" || return 1
    fi
    echo ok > "$VENV/$marker"
    echo "[setup] Done."
}

setup "$HERE/requirements.txt" .installed-v3 || {
    echo "[error] Setup failed. Fix the error above and run again."; exit 1; }

exec "$VPY" "$HERE/tools/blur_images.py" "$IN" "$OUT" "$@"
