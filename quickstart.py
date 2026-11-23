#!/usr/bin/env python3
"""Quick start: face-blur as the analysis step of a live pipeline.

    RTSP -> decode -> [ faceblur: frame in -> original / blurred / face boxes ] -> encode -> live view

Decoding and encoding (hardware when available) are in videoio.py; this file
only shows the analysis step.

    python quickstart.py --input rtsp://camera/stream                                # view in a window (no delay)
    python quickstart.py --input video.mp4                                           # a video file
    python quickstart.py --input video.mp4 --model yolo11n_1280                      # small faces, slower
    python quickstart.py --input video.mp4 --model face_det_lite                     # Qualcomm's detector
    python quickstart.py --input video.mp4 --tiles 2                                 # small faces too
    python quickstart.py --input video.mp4 --tiles 2 --detect-every 3                # detect every 3rd frame
    python quickstart.py --input video.mp4 --threshold 0.6                           # fewer false faces
    python quickstart.py --input video.mp4 --delay 12                                # 12 frames later (0.5 s at 25 fps), about half the misses
    python quickstart.py --input video.mp4 --method pixelate --shape rect            # pixel blocks over the whole box
    python quickstart.py --input rtsp://camera/stream --output rtsp://server/blurred # re-stream
    python quickstart.py                                                             # the sample clip
"""
import argparse
import threading
import time
from pathlib import Path

import cv2

from faceblur import DEFAULT_MODEL, MODELS, FaceBlurrer
from videoio import Stats, VideoSink, VideoSource, draw_faces

SAMPLE = str(Path(__file__).resolve().parent / "examples/video/street_before_after.mp4")


def main():
    ap = argparse.ArgumentParser(description="Blur faces in a live feed or a video.")
    ap.add_argument("--input", default=SAMPLE, help="rtsp:// URL or video file (default: the sample clip)")
    ap.add_argument("--output", help="rtsp://, srt://, udp:// URL or .mp4 file (default: show in a window)")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help=f"face detector (default: {DEFAULT_MODEL})")
    ap.add_argument("--detect-every", type=int, default=1, metavar="N",
                    help="run the detector every N frames; tracking (ByteTrack + optical flow) "
                         "moves the faces in between (default 1 = every frame)")
    ap.add_argument("--delay", type=int, default=0, metavar="FRAMES",
                    help="hold the output this many frames so later detections fix earlier "
                         "frames: about 0.5 s worth (e.g. 12 at 25 fps) leaves half as many faces "
                         "unblurred (default 0 = no delay)")
    ap.add_argument("--threshold", type=float, default=None, metavar="T",
                    help="confidence a detection needs to count as a face, 0..1 (default: the "
                         "model's own, yolo11n 0.3, face_det_lite 0.55, yunet 0.7). Lower finds more faces "
                         "and more false ones")
    ap.add_argument("--min-size", type=int, default=None, metavar="PX",
                    help="skip faces smaller than PX pixels (default: 24 for yunet, no limit otherwise)")
    ap.add_argument("--cpu", action="store_true", help="face detection on the CPU only (no Qualcomm NPU)")
    # how the blur looks
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian",
                    help="smooth blur or pixel blocks (default gaussian)")
    ap.add_argument("--strength", type=int, choices=[1, 2, 3], default=3, help="1 (light) .. 3 (heavy, default)")
    ap.add_argument("--shape", choices=["oval", "rect"], default="oval",
                    help="blur an oval that follows the face, or the whole box (safer) (default oval)")
    ap.add_argument("--pad", type=float, default=0.15, metavar="F",
                    help="margin around each face, as a fraction of its size, 0..1 (default 0.15)")
    ap.add_argument("--no-mask", action="store_true",
                    help="do not blur: output the original frames (use with --show-scores to see the "
                         "boxes); for apps that mask the faces themselves from res.boxes")
    ap.add_argument("--show-scores", action="store_true", help="draw each face box with its confidence")
    ap.add_argument("--tiles", type=int, default=1, metavar="N",
                    help="also detect on an N x N grid of tiles, for small faces (default 1 = off; "
                         "costs 1 + N*N detector runs per frame)")
    args = ap.parse_args()

    video = VideoSource(args.input)                   # decoded BGR frames
    sink = VideoSink(args.output, video.size, video.fps) if args.output else None

    fb = FaceBlurrer(model=args.model, tiles=args.tiles, threshold=args.threshold,  # load once, reuse
                     min_size=args.min_size, gpu=not args.cpu,
                     method=args.method, strength=args.strength, shape=args.shape, pad=args.pad,
                     mask=not args.no_mask)
    live = fb.stream(fps=video.fps, detect_every=args.detect_every, delay=args.delay)

    stats = Stats()                                   # fps and speed, drawn on the frame
    analysis = [0.0]                                  # seconds the last put() took

    def show(res):                                    # one analysed frame -> output; False to stop
        # res.original  the input frame, unchanged
        # res.blurred   a blurred copy (None with --no-mask: mask res.boxes yourself)
        # res.boxes     the faces, [(x, y, w, h), ...]
        # res.scores    the confidence of each face
        out = res.original.copy() if res.blurred is None else res.blurred.copy()
        if args.show_scores:
            draw_faces(out, res.boxes, res.scores)
        stats.draw(out, stats.start() - analysis[0], faces=len(res.boxes), dropped=video.dropped)
        if sink:
            sink.write(out)
            return True
        cv2.imshow("face-blur (q to quit)", out)
        return cv2.waitKey(1) & 0xFF != ord("q")

    # Input thread: decode -> put (analyse, buffer). Output (this thread): one frame per tick of
    # its own clock at the source frame rate, so the output frame rate stays steady even when
    # frames arrive unevenly or analysis time varies.
    done, stop = threading.Event(), threading.Event()

    def read():
        try:
            for frame in video:
                if stop.is_set():
                    break
                t = time.perf_counter()
                live.put(frame)
                analysis[0] = time.perf_counter() - t
        finally:
            done.set()

    threading.Thread(target=read, daemon=True).start()
    tick, last = 1.0 / video.fps, None
    next_t = time.perf_counter()
    while True:
        while live.ready > 2:                         # output fell behind: skip to keep the delay
            live.get()
        res = live.get(force=done.is_set())           # at the end, empty the buffer
        if res is None and done.is_set():
            break
        res = res or last                             # input late: show the last frame again
        if res is not None:
            last = res
            if not show(res):
                stop.set()
                break
        next_t += tick
        time.sleep(max(0.0, next_t - time.perf_counter()))
        if time.perf_counter() - next_t > tick:       # too far behind (slow display): resync the clock
            next_t = time.perf_counter()

    if sink:
        sink.close()
    else:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
