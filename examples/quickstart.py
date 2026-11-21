#!/usr/bin/env python3
"""Quick start: blur faces from your own Python code.

Run from the repository root (after `pip install -r requirements.txt`):

    python examples/quickstart.py                    # image + video demos
    python examples/quickstart.py --webcam           # live webcam, q to quit

Each demo is a short function: copy the one you need into your app.
Outputs go to examples/quickstart_out/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # in your app: put face-blur on the path, or copy faceblur.py + blur_*.py + models/

import cv2

from faceblur import FaceBlurrer

OUT = ROOT / "examples" / "quickstart_out"


def demo_numpy_image(fb):
    """1. You already have an image in memory (BGR numpy array)."""
    img = cv2.imread(str(ROOT / "examples/input/crowd/film_set.jpg"))
    faces = fb.detect(img)                 # [(x, y, w, h), ...]
    print(f"[image] {len(faces)} faces, first boxes: {faces[:3]}")

    blurred = fb.blur(img)                 # returns a blurred copy; img is unchanged
    cv2.imwrite(str(OUT / "film_set_blurred.jpg"), blurred)

    # Or decide yourself which boxes to blur, e.g. only the larger faces:
    big = [b for b in faces if b[2] >= 60]
    cv2.imwrite(str(OUT / "film_set_big_only.jpg"), fb.blur_boxes(img, big))


def demo_image_file(fb):
    """2. File in, file out."""
    n = fb.blur_file(ROOT / "examples/input/sports/football.jpg", OUT / "football_blurred.jpg")
    print(f"[file] football.jpg: {n} faces blurred")


def demo_video_file(fb):
    """3. A video file. Keeps format and sound when ffmpeg is installed."""
    src = ROOT / "examples/video/street_before_after.mp4"
    info = fb.blur_video(src, OUT / "street_blurred.mp4")
    print(f"[video] {info['frames']} frames, faces in {info['frames_blurred']}, "
          f"audio {info['audio']}")


def demo_webcam(fb, camera=0):
    """4. Live frames from a webcam (or an RTSP URL instead of 0)."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {camera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    live = fb.stream(fps=fps, delay=0.5)   # frames come out 0.5 s late, faces covered from their first frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for out in live.push(frame):
                cv2.imshow("blurred (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Create once and reuse: loading the detector is the slow part.
    fb = FaceBlurrer(model="yunet", method="gaussian", strength=3)
    print(fb.device)

    if "--webcam" in sys.argv:
        demo_webcam(fb)
        return
    demo_numpy_image(fb)
    demo_image_file(fb)
    demo_video_file(fb)
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
