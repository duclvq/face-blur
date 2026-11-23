#!/usr/bin/env python3
"""Blur the faces in a video file and save it as an MP4.

This is the only file in the repo that encodes video. The library itself
(faceblur.py) works on numpy frames; this shows one way to read a video,
run the frames through FaceBlurrer.stream() and write the result.

    python tools/blur_video.py IN.mp4 OUT.mp4 [--model yunet]

Writes H.264 with ffmpeg when it is installed, otherwise MPEG-4 Part 2 with
OpenCV. Sound, subtitles and the source codec are not kept.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the library is in the repo root
from faceblur import DEFAULT_MODEL, MODELS, FaceBlurrer  # noqa: E402


class Mp4Writer:
    """BGR frames -> MP4: H.264 through an ffmpeg pipe, or OpenCV's mp4v."""

    def __init__(self, path, size, fps):
        w, h = size
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6g}",
                 "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(path)],
                stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def main():
    ap = argparse.ArgumentParser(description="Blur the faces in a video file.")
    ap.add_argument("input")
    ap.add_argument("output", help="output .mp4")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    a = ap.parse_args()
    if Path(a.input).resolve() == Path(a.output).resolve():
        sys.exit("output must not be the input file")

    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fb = FaceBlurrer(model=a.model)
    print(fb.backend)
    live = fb.stream(fps=fps)
    out = Mp4Writer(a.output, size, fps)
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            live.put(frame)
            while (res := live.get()) is not None:
                out.write(res.blurred)
                n += 1
        for res in live.flush():            # the frames still held at the end
            out.write(res.blurred)
            n += 1
    finally:
        cap.release()
        out.close()
    print(f"{n} frames -> {a.output}")


if __name__ == "__main__":
    main()
