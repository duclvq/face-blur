#!/usr/bin/env python3
"""Blur faces in a live feed and preview it (tracker.py, no output delay).

Usage:
    python tools/blur_live.py SOURCE [--show] [--model face_det_lite|yunet|...]
                              [--detect-every 3] ...

SOURCE is a webcam index (0), an RTSP/RTMP/HTTP/UDP URL, or a video file.
--show previews the blurred frames; without it the run only prints timings.
Nothing is written to disk: see tools/blur_video.py to save a video, or use
faceblur.StreamBlurrer from your own code.

Live sources: when the machine is slower than the stream, input frames are
dropped (never shown unprocessed), and a lost stream is reconnected.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the library is in the repo root
import detector  # noqa: E402
from detector import DEFAULT_MODEL, MODELS, blur_region, use_model, use_npu  # noqa: E402
from tracker import Analyzer  # noqa: E402


# --------------------------------------------------------------------------- input / output

def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    return cap if cap.isOpened() else None


def is_live(src):
    return src.isdigit() or "://" in src


def reader(src, q, stop, live, realtime, fps, stats):
    """Read frames into q. Live: drop the oldest frame when behind; reconnect."""
    cap, i, period, next_t = open_source(src), 0, 1.0 / fps, time.time()
    retries = 0
    while not stop.is_set():
        ok, frame = cap.read() if cap is not None else (False, None)
        if not ok:
            if not live:
                break
            retries += 1
            if retries > stats["max_retries"]:
                print("\n  ERROR: stream lost, giving up")
                break
            print(f"\n  stream lost, reconnecting ({retries})...")
            if cap is not None:
                cap.release()
            time.sleep(1.0)
            cap = open_source(src)
            continue
        retries = 0
        if realtime:  # a file played at its own speed, like a live source
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
        item = (i, frame, time.time())
        i += 1
        if live or realtime:
            while True:
                try:
                    q.put_nowait(item)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                        stats["dropped"] += 1
                    except queue.Empty:
                        pass
        else:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    pass
    if cap is not None:
        cap.release()
    q.put(None)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Track and blur faces in a live video stream "
                                             "(preview only; nothing is saved).")
    ap.add_argument("source", help="webcam index (0), stream URL, or video file")
    ap.add_argument("--show", action="store_true", help="show the blurred video in a window")
    ap.add_argument("--detect-every", type=int, default=1, metavar="N",
                    help="run the detector every N frames (default 1: every frame); optical "
                         "flow moves faces in between. Raise it on a slow machine")
    ap.add_argument("--low-score", type=float, default=0.3, metavar="S",
                    help="ByteTrack second stage: detections scoring S up to the normal "
                         "threshold keep an existing face tracked (never start a new one). "
                         "0 turns it off")
    ap.add_argument("--fps", type=float, default=None, help="frame rate if the source does not report it")
    ap.add_argument("--realtime", action="store_true",
                    help="play a video file at its own speed, like a live source (for testing)")
    ap.add_argument("--max-retries", type=int, default=30, help="reconnect attempts for a lost stream")
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                    help="face detector: yolo11n_640 (default; YOLO11n-face, AGPL-3.0 weights), "
                         "yolo11n_1280 (finds small faces, about 4x slower), face_det_lite "
                         "(Qualcomm Lightweight-Face-Detection, misses small faces) or yunet "
                         "(simple scenes with large faces). All but yunet run on the Qualcomm "
                         "NPU with onnxruntime-qnn")
    ap.add_argument("--min-size", type=int, default=None,
                    help="smallest face size in px (default: 8 for yunet, no limit otherwise)")
    ap.add_argument("--pad", type=float, default=0.2, help="margin around each face, 0..1")
    ap.add_argument("--shape", choices=["rect", "oval"], default="oval")
    ap.add_argument("--device", choices=["cpu", "gpu", "npu"], default="cpu",
                    help="where the detector runs: cpu (default), gpu (NVIDIA, needs onnxruntime-gpu) or npu (Qualcomm, needs onnxruntime-qnn)")
    ap.add_argument("--npu", nargs="?", const="face_det_lite", metavar="MODEL",
                    help="face detection on a Qualcomm Hexagon NPU (onnxruntime-qnn): "
                         "face_det_lite (default; Qualcomm Lightweight-Face-Detection, float) or "
                         "yunet[:WxH] (quantized YuNet, 640x384, 960x544 (default) or 1280x736). "
                         "--model already runs on the NPU when onnxruntime-qnn is installed. "
                         "See docs/hexagon.md")
    ap.add_argument("--npu-precision", choices=["a8w8", "a16w8", "fp32"], default="a16w8",
                    help="--npu yunet only: a16w8 (default), a8w8 (older Hexagon) or fp32 (run as fp16)")
    ap.add_argument("--npu-allow-cpu", action="store_true", help=argparse.SUPPRESS)  # test w/o QNN
    ap.add_argument("--boxes-json", help=argparse.SUPPRESS)  # per-frame boxes, for evaluation
    o = ap.parse_args()
    npu_model, npu_size = None, None
    if o.npu:
        npu_model, _, size = o.npu.partition(":")
        if npu_model not in ("face_det_lite", "yunet"):
            ap.error("--npu takes face_det_lite or yunet[:WxH]")
        if npu_model == "yunet":
            npu_size = tuple(int(v) for v in (size or "960x544").lower().split("x"))
        if o.model != npu_model:
            print(f"--npu uses {npu_model} (not {o.model})")
        o.model = npu_model
        o.device = "cpu" if o.npu_allow_cpu else "npu"
    if o.detect_every < 1 or not 0 <= o.pad <= 1:
        ap.error("--detect-every >= 1, --pad 0..1")
    if o.min_size is None:
        o.min_size = 8 if o.model == "yunet" else 0

    cap = open_source(o.source)
    if cap is None:
        sys.exit(f"Cannot open source: {o.source}")
    ok, first = cap.read()
    fps = o.fps or cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ok:
        sys.exit(f"No frames from source: {o.source}")
    if not fps or fps <= 0 or fps > 240:
        fps = 25.0
    H, W = first.shape[:2]
    live = is_live(o.source)
    print(use_model(o.model, o.device))
    npu_msg = None
    if npu_model == "yunet":
        npu_msg = use_npu("yunet", npu_size, o.npu_precision, device=o.device)
    elif npu_model:  # face_det_lite: use_model loaded it, on the NPU when QNN is there
        if not detector._npu.on_npu and not o.npu_allow_cpu:
            npu_msg = "WARNING: onnxruntime has no QNN support (pip install onnxruntime-qnn); using CPU"
    if npu_msg:
        print(npu_msg)
    print(f"Source: {W}x{H} @ {fps:.3f} fps ({'live' if live else 'realtime file' if o.realtime else 'file'}), "
          f"detector every {o.detect_every} frame(s)")

    stats = dict(dropped=0, max_retries=o.max_retries)
    stop = threading.Event()
    q_in = queue.Queue(2 if (live or o.realtime) else 16)
    t_read = threading.Thread(target=reader, args=(o.source, q_in, stop, live, o.realtime, fps, stats),
                              daemon=True)
    t_read.start()

    an = Analyzer(o, fps)
    dump = {} if o.boxes_json else None
    n, t0, last, lat = 0, time.time(), 0.0, []
    tty = sys.stdout.isatty()
    try:
        while not stop.is_set():
            item = q_in.get()
            if item is None:
                break
            i, frame, t_in = item
            boxes = an.push(frame)
            if boxes is None:  # analysis failed: blur the whole frame
                blur_region(frame, (0, 0, W, H), "pixelate", 3, 0, "rect")
                boxes = []
            for b in boxes:
                blur_region(frame, b[:4], o.method, o.strength, o.pad, o.shape)
            if dump is not None:
                dump[i] = boxes
            if o.show:
                cv2.imshow("face-blur (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            lat.append(time.time() - t_in)
            n += 1
            if tty and time.time() - last > 0.5:
                last = time.time()
                print(f"\r  frame {i}  {n / (last - t0):.1f} fps  faces {len(boxes)}  "
                      f"latency {lat[-1] * 1000:.0f} ms  dropped {stats['dropped']}   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopping...")
    finally:
        stop.set()
        if o.show:
            cv2.destroyAllWindows()

    secs = time.time() - t0
    lat = sorted(lat) or [0]
    s = an.stats
    nf = max(s["frames"], 1)
    print(f"\r  done: {n} frames, {stats['dropped']} dropped, {n / max(secs, 1e-9):.1f} fps; "
          f"latency median {lat[len(lat) // 2] * 1000:.0f} ms, p95 {lat[int(len(lat) * 0.95)] * 1000:.0f} ms")
    print(f"  per frame: detector {s['t_detect'] / nf * 1000:.1f} ms, "
          f"tracking {s['t_track'] / nf * 1000:.1f} ms")
    print(f"  detector runs {s['detections']}, low-score matches {s['low_matches']}, "
          f"scene cuts {s['cuts']}, failed frames {s['failed']}")
    if dump is not None:
        Path(o.boxes_json).write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
