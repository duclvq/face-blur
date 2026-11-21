#!/usr/bin/env python3
"""Blur faces in a live video stream with YuNet.

Usage:
    python blur_stream.py SOURCE [OUTPUT] [--show] [--delay 0.5] [--detect-every 2] ...

SOURCE is a webcam index (0), an RTSP/RTMP/HTTP/UDP URL, or a video file.
OUTPUT is a video file, or a stream URL (rtmp://, rtsp://, srt://, udp://).

This adapts the ideas of InsightFace PrivateFrame (which needs the whole video)
to a live, causal setting with a short, fixed output delay:

  * Detector anchors: YuNet runs every N frames (and after a scene cut).
  * Tracking in between: sparse Lucas-Kanade optical flow on a small region
    around each face (with a forward-backward check), fed into a
    constant-velocity alpha-beta filter (a steady-state Kalman filter), so a
    face stays covered while the detector is not run or misses it.
  * Local re-detection: a tracked face the detector missed is searched for
    again in an upscaled crop around where it should be (helps small faces).
  * Look-back within the delay: frames wait --delay seconds before output.
    A newly found face is tracked backwards through those frames, so its
    first appearance is blurred too; the frames between two detections of a
    face are re-filled from both ends; boxes are smoothed with past and
    future frames (no lag).
  * Scene cuts reset tracking, so a box never jumps across a cut.
  * Fail closed: if analysing a frame fails, that frame is blurred whole.
  * Live sources: when the machine is slower than the stream, input frames are
    dropped (never sent out unprocessed), and a lost stream is reconnected.

Audio is not passed through.
"""
import argparse
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from blur_faces import (SCORE_THRESHOLD, YUNET_MODEL, blur_region, detect_faces,
                        gpu_message, use_gpu)


# --------------------------------------------------------------------------- geometry
# Boxes are float (x0, y0, x1, y1). Filter state is (cx, cy, log w, log h).

def to_state(b):
    w, h = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, math.log(w), math.log(h)])


def to_box(s):
    w, h = math.exp(s[2]), math.exp(s[3])
    return np.array([s[0] - w / 2, s[1] - h / 2, s[0] + w / 2, s[1] + h / 2])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_face(a, b):
    """Association gate, after PrivateFrame's: overlap or close centres, similar size."""
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    if min(area_a, area_b) <= 0 or max(area_a, area_b) / min(area_a, area_b) > 5:
        return 0.0
    o = iou(a, b)
    dist = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    near = dist / math.sqrt(min(area_a, area_b))
    if o >= 0.05 or near <= 1.0:
        return o + max(0.0, 1.0 - near) * 0.5  # score for greedy matching
    return 0.0


def union_box(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])


def clip_box(b, W, H):
    return np.array([min(max(b[0], 0), W), min(max(b[1], 0), H),
                     min(max(b[2], 0), W), min(max(b[3], 0), H)])


def visible(b, W, H):
    c = clip_box(b, W, H)
    return c[2] - c[0] >= 2 and c[3] - c[1] >= 2


# --------------------------------------------------------------------------- optical flow

ROI_SIDE = 128
LK = dict(winSize=(15, 15), maxLevel=2,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))


def flow_box(g0, g1, box):
    """Move box from gray frame g0 to g1 with sparse LK flow, or None.

    Points are tracked on a region around the face only; a forward-backward
    check drops bad points; the median shift moves the box and the median
    change of point spacing scales it.
    """
    H, W = g0.shape
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return None
    m = max(w, h) * 0.5 + 16
    rx0, ry0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    rx1, ry1 = int(min(W, x1 + m)), int(min(H, y1 + m))
    if rx1 - rx0 < 16 or ry1 - ry0 < 16:
        return None
    p, c = g0[ry0:ry1, rx0:rx1], g1[ry0:ry1, rx0:rx1]
    # Large faces: track on a copy scaled to <= ROI_SIDE px (same motion, much cheaper).
    k = max(1, math.ceil(max(p.shape) / ROI_SIDE))
    f = 1.0 / k
    if k > 1:  # every k-th pixel: no resize cost, and LK works on a pyramid anyway
        p, c = np.ascontiguousarray(p[::k, ::k]), np.ascontiguousarray(c[::k, ::k])
    # Features on the face and a little around it (head and shoulders move with it).
    mask = np.zeros_like(p)
    ex, ey = w * 0.15, h * 0.15
    cv2.rectangle(mask, (int((x0 - ex - rx0) * f), int((y0 - ey - ry0) * f)),
                  (int((x1 + ex - rx0) * f), int((y1 + ey - ry0) * f)), 255, -1)
    pts = cv2.goodFeaturesToTrack(p, maxCorners=24, qualityLevel=0.01,
                                  minDistance=max(2.0, min(w, h) * f / 8), mask=mask, blockSize=5)
    if pts is None or len(pts) < 4:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(p, c, pts, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(c, p, nxt, None, **LK)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < max(1.0, 0.05 * min(w, h) * f))
    if good.sum() < 4:
        return None
    a, b = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
    d = np.median(b - a, axis=0) / f
    i, j = np.triu_indices(len(a), 1)
    da = np.linalg.norm(a[i] - a[j], axis=1)
    db = np.linalg.norm(b[i] - b[j], axis=1)
    ok = da > 1
    scale = float(np.clip(np.median(db[ok] / da[ok]), 0.8, 1.25)) if ok.any() else 1.0
    cx, cy = (x0 + x1) / 2 + d[0], (y0 + y1) / 2 + d[1]
    return np.array([cx - w * scale / 2, cy - h * scale / 2, cx + w * scale / 2, cy + h * scale / 2])


# --------------------------------------------------------------------------- tracks

class Track:
    """One face: an alpha-beta filter plus its boxes in the buffered frames."""
    _next = 0

    def __init__(self, box, frame):
        Track._next += 1
        self.id = Track._next
        self.s = to_state(box)
        self.v = np.zeros(4)
        self.last_seen = frame          # last frame confirmed by the detector
        self.anchors = [(frame, np.array(box, float))]
        self.boxes = {frame: np.array(box, float)}

    @property
    def box(self):
        return to_box(self.s)

    def predict(self):
        self.s = self.s + self.v
        self.v *= 0.9  # without measurements, slow down instead of drifting away

    def correct(self, box, alpha, beta=0.3):
        r = to_state(box) - self.s
        self.s = self.s + alpha * r
        self.v = self.v + beta * r


def associate(tracks, dets):
    """Greedy matching by same_face score. Returns (pairs, unmatched_dets)."""
    cand = sorted(((same_face(t.box, d), ti, di) for ti, t in enumerate(tracks)
                   for di, d in enumerate(dets)), reverse=True)
    used_t, used_d, pairs = set(), set(), []
    for score, ti, di in cand:
        if score <= 0:
            break
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        pairs.append((tracks[ti], dets[di]))
    return pairs, [d for i, d in enumerate(dets) if i not in used_d]


_local = None


def local_redetect(frame, box, min_size):
    """Search for a face again in an upscaled crop around box (or None)."""
    global _local
    H, W = frame.shape[:2]
    w, h = box[2] - box[0], box[3] - box[1]
    m = max(w, h) * 1.0 + 8
    x0, y0 = int(max(0, box[0] - m)), int(max(0, box[1] - m))
    x1, y1 = int(min(W, box[2] + m)), int(min(H, box[3] + m))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = frame[y0:y1, x0:x1]
    s = max(1.0, min(4.0, 160 / max(crop.shape[:2])))  # small faces get bigger
    if s > 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    ch, cw = crop.shape[:2]
    if _local is None:
        _local = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (cw, ch), 0.5, 0.3, 50)
    _local.setInputSize((cw, ch))
    _, faces = _local.detect(crop)
    if faces is None:
        return None
    best, best_score = None, 0.0
    for f in faces:
        fb = np.array([f[0] / s + x0, f[1] / s + y0, (f[0] + f[2]) / s + x0, (f[1] + f[3]) / s + y0])
        if min(fb[2] - fb[0], fb[3] - fb[1]) < min_size * 0.5:
            continue
        sc = same_face(box, fb)
        if sc > best_score:
            best, best_score = fb, sc
    return best


# --------------------------------------------------------------------------- scene cuts

class SceneCut:
    """Content jump much larger than the recent frame-to-frame change."""

    def __init__(self):
        self.prev = None
        self.hist = deque(maxlen=30)

    def __call__(self, gray):
        h, w = gray.shape
        sub = gray[::max(1, h // 128), ::max(1, w // 128)]  # cheap subsample
        small = cv2.resize(sub, (64, 64), interpolation=cv2.INTER_AREA)
        cut = False
        if self.prev is not None:
            diff = float(cv2.absdiff(small, self.prev).mean())
            base = float(np.median(self.hist)) if self.hist else 0.0
            cut = diff > 25 and diff > 3 * base + 10
            if not cut:
                self.hist.append(diff)
        self.prev = small
        return cut


# --------------------------------------------------------------------------- analysis

class Analyzer:
    """Causal tracking with a look-back window of `delay` frames."""

    def __init__(self, opts, fps):
        self.o = opts
        self.delay = max(0, round(opts.delay * fps))
        self.hold = max(1, round(opts.hold * fps))
        self.tracks = []
        self.retired = []         # dropped tracks that still own boxes in the buffer
        self.cur_gray = None      # gray of the frame being analysed
        self.buf = deque()        # entries: dict(i, frame, gray, cut, fail, t_in)
        self.cut = SceneCut()
        self.prev_gray = None
        self.stats = dict(detections=0, redetections=0, backfilled=0, refilled=0, cuts=0, failed=0)

    # -- one new frame
    def push(self, i, frame, t_in):
        entry = dict(i=i, frame=frame, gray=None, cut=False, fail=False, t_in=t_in)
        try:
            self._analyse(entry)
        except Exception as e:  # fail closed: this frame is blurred whole
            entry["fail"] = True
            self.stats["failed"] += 1
            print(f"\n  WARNING: frame {i} analysis failed ({e}); blurring the whole frame")
        self.buf.append(entry)
        out = []
        while len(self.buf) > self.delay:
            out.append(self._emit(self.buf.popleft()))
        return out

    def flush(self):
        out = []
        while self.buf:
            out.append(self._emit(self.buf.popleft()))
        return out

    def _analyse(self, e):
        o, i, frame = self.o, e["i"], e["frame"]
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        e["gray"] = self.cur_gray = gray
        if self.cut(gray):
            e["cut"] = True
            self.stats["cuts"] += 1
            self.retired += self.tracks
            self.tracks = []  # never carry a box across a cut

        # 1. Move every track to this frame: optical flow, else prediction.
        for t in self.tracks:
            t.predict()
            if self.prev_gray is not None:
                fb = flow_box(self.prev_gray, gray, t.boxes.get(i - 1, t.box))
                if fb is not None:
                    t.correct(fb, alpha=0.7)

        # 2. Detector anchors every N frames, after a cut, and while a face is lost.
        if i % o.detect_every == 0 or e["cut"]:
            dets = [np.array([x, y, x + w, y + h], float)
                    for x, y, w, h in detect_faces(frame, o.min_size)]
            self.stats["detections"] += 1
            pairs, new = associate(self.tracks, dets)
            matched = set()
            for t, d in pairs:
                self._confirm(t, d, i)
                matched.add(t.id)
            # Local re-detection for tracked faces the detector missed.
            budget = 4
            for t in self.tracks:
                # only for a face seen at the previous detector run (a fresh miss)
                if t.id in matched or budget == 0 or i - t.last_seen > 2 * o.detect_every:
                    continue
                budget -= 1
                d = local_redetect(frame, t.box, o.min_size)
                if d is not None:
                    self.stats["redetections"] += 1
                    self._confirm(t, d, i)
                    matched.add(t.id)
            for d in new:
                t = Track(d, i)
                self.tracks.append(t)
                self._backfill(t, i, d)

        # 3. Drop faces not confirmed for --hold seconds or out of the picture.
        keep = [t for t in self.tracks if i - t.last_seen <= self.hold and visible(t.box, W, H)]
        self.retired += [t for t in self.tracks if t not in keep]
        self.tracks = keep
        for t in self.tracks:
            t.boxes[i] = t.box
        self.prev_gray = gray

    def _confirm(self, t, d, i):
        """A detection of track t at frame i: correct it and re-fill its gap."""
        prev_frame, prev_box = t.anchors[-1]
        t.correct(d, alpha=0.9)
        t.last_seen = i
        t.anchors.append((i, d))
        if len(t.anchors) > 8:
            t.anchors.pop(0)
        # Frames between the two detections that are still in the buffer:
        # blend the flow-tracked box with interpolation between the anchors
        # (both ends). Where they disagree, cover both.
        gap = i - prev_frame
        if gap > 1:
            a, b = to_state(prev_box), to_state(d)
            for e in self.buf:
                k = e["i"]
                if prev_frame < k < i and k in t.boxes:
                    lin = to_box(a + (b - a) * (k - prev_frame) / gap)
                    fl = t.boxes[k]
                    t.boxes[k] = (lin + fl) / 2 if iou(lin, fl) >= 0.5 else union_box(lin, fl)
                    self.stats["refilled"] += 1

    def _backfill(self, t, i, box):
        """Track a new face backwards through the buffered frames."""
        b, g, misses = np.array(box, float), self.cur_gray, 0
        for e in reversed(self.buf):
            nb = flow_box(g, e["gray"], b)
            if nb is None:
                misses += 1
                if misses > 2:
                    break
                nb = b  # keep the box still for a frame or two rather than stop at once
            t.boxes[e["i"]] = nb
            self.stats["backfilled"] += 1
            b, g = nb, e["gray"]
            if e["cut"]:
                break  # the face cannot be in frames before a cut

    def _emit(self, e):
        """Final boxes for a frame leaving the buffer, smoothed with its neighbours."""
        k = e["i"]
        boxes = []
        r = self.o.smooth
        for t in self.tracks + self.retired:
            if k not in t.boxes:
                continue
            raw = t.boxes[k]
            near = [to_state(t.boxes[j]) for j in range(k - r, k + r + 1) if j in t.boxes]
            sm = to_box(np.median(near, axis=0)) if len(near) >= 3 else raw
            boxes.append(sm if iou(sm, raw) >= 0.6 else union_box(sm, raw))
        for t in self.tracks + self.retired:  # forget frames that have left the buffer
            for j in [j for j in t.boxes if j <= k - r - 1]:
                del t.boxes[j]
        self.retired = [t for t in self.retired if any(j > k for j in t.boxes)]
        H, W = e["frame"].shape[:2]
        out = []
        for b in boxes:
            c = clip_box(b, W, H)
            if c[2] - c[0] >= 2 and c[3] - c[1] >= 2:
                out.append((int(c[0]), int(c[1]), int(round(c[2] - c[0])), int(round(c[3] - c[1]))))
        return e, out


# --------------------------------------------------------------------------- input / output

def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    return cap if cap.isOpened() else None


def is_live(src):
    return src.isdigit() or "://" in src


def ffmpeg_writer(out, size, fps, gpu):
    """ffmpeg process that takes raw BGR frames on stdin."""
    w, h = size
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"]
    if gpu:
        from blur_video import ffmpeg_encoders, nvenc_works
        nv = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll"]
        if "h264_nvenc" in ffmpeg_encoders() and nvenc_works(tuple(nv)):
            enc = nv
    fmt = []
    if out.startswith("rtmp://"):
        fmt = ["-f", "flv"]
    elif out.startswith("rtsp://"):
        fmt = ["-f", "rtsp", "-rtsp_transport", "tcp"]
    elif out.startswith(("srt://", "udp://")):
        fmt = ["-f", "mpegts"]
    quality = ["-b:v", "4M", "-maxrate", "4M", "-bufsize", "2M"] if "://" in out else ["-crf", "20"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6g}", "-i", "-",
           *enc, "-bf", "0", "-g", str(max(1, round(fps * 2))), "-pix_fmt", "yuv420p",
           *quality, *fmt, out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


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


def writer_loop(q, opts, size, fps, stats, stop):
    """Blur and write/show frames from q."""
    proc = ffmpeg_writer(opts.output, size, fps, not opts.cpu) if opts.output else None
    W, H = size
    last_i = None
    try:
        while True:
            item = q.get()
            if item is None:
                break
            e, boxes = item
            frame = e["frame"]
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            if e["fail"]:
                blur_region(frame, (0, 0, W, H), "pixelate", 3, 0, "rect")
            for b in boxes:
                blur_region(frame, b, opts.method, opts.strength, opts.pad, opts.shape)
            # Input frames dropped (machine slower than the stream): repeat this
            # blurred frame in their place so the output keeps the stream's timing.
            copies = 1 if last_i is None else max(1, e["i"] - last_i)
            last_i = e["i"]
            if proc is not None:
                try:
                    data = frame.tobytes()
                    for _ in range(copies):
                        proc.stdin.write(data)
                except BrokenPipeError:
                    print("\n  ERROR: ffmpeg output closed")
                    stop.set()
                    break
            if opts.show:
                cv2.imshow("face-blur (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
            stats["latency"].append(time.time() - e["t_in"])
            stats["written"] += 1
    finally:
        if proc is not None:
            proc.stdin.close()
            proc.wait()
        if opts.show:
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Blur faces in a live video stream (YuNet).")
    ap.add_argument("source", help="webcam index (0), stream URL, or video file")
    ap.add_argument("output", nargs="?", help="video file or stream URL (rtmp://, rtsp://, srt://, udp://)")
    ap.add_argument("--show", action="store_true", help="show the blurred video in a window")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="output delay in seconds, used to blur a face's first frames "
                         "and to re-fill gaps (0 = no delay, less protection)")
    ap.add_argument("--detect-every", type=int, default=3, metavar="N",
                    help="run the detector every N frames; optical flow tracks faces in between")
    ap.add_argument("--hold", type=float, default=1.0,
                    help="seconds a face keeps being tracked and blurred after the detector loses it")
    ap.add_argument("--smooth", type=int, default=2, metavar="K",
                    help="smooth boxes over K frames before and after (needs --delay)")
    ap.add_argument("--fps", type=float, default=None, help="frame rate if the source does not report it")
    ap.add_argument("--realtime", action="store_true",
                    help="play a video file at its own speed, like a live source (for testing)")
    ap.add_argument("--max-retries", type=int, default=30, help="reconnect attempts for a lost stream")
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--min-size", type=int, default=8, help="smallest face size in px")
    ap.add_argument("--pad", type=float, default=0.2, help="margin around each face, 0..1")
    ap.add_argument("--shape", choices=["rect", "oval"], default="oval")
    ap.add_argument("--cpu", action="store_true", help="do not use an NVIDIA GPU")
    ap.add_argument("--boxes-json", help=argparse.SUPPRESS)  # per-frame boxes, for evaluation
    o = ap.parse_args()
    if not o.output and not o.show:
        ap.error("give an OUTPUT file/URL, or --show")
    if o.detect_every < 1 or o.delay < 0 or o.hold <= 0 or not 0 <= o.pad <= 1:
        ap.error("--detect-every >= 1, --delay >= 0, --hold > 0, --pad 0..1")
    if o.output and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is needed for OUTPUT (install it, or use --show only)")
    if o.output and "://" not in o.output and Path(o.output).resolve() == Path(o.source).resolve():
        sys.exit("OUTPUT must not be the input file")
    if not o.smooth or not o.delay:
        o.smooth = 0

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
    print("Face detector: YuNet (OpenCV)")
    if not o.cpu:
        print(gpu_message(use_gpu()))
    print(f"Source: {W}x{H} @ {fps:.3f} fps ({'live' if live else 'realtime file' if o.realtime else 'file'}), "
          f"delay {o.delay:g} s, detector every {o.detect_every} frame(s)")

    stats = dict(dropped=0, written=0, latency=[], max_retries=o.max_retries)
    stop = threading.Event()
    q_in, q_out = queue.Queue(4 if (live or o.realtime) else 16), queue.Queue(64)
    t_read = threading.Thread(target=reader, args=(o.source, q_in, stop, live, o.realtime, fps, stats),
                              daemon=True)
    t_write = threading.Thread(target=writer_loop, args=(q_out, o, (W, H), fps, stats, stop), daemon=True)
    t_read.start()
    t_write.start()

    an = Analyzer(o, fps)
    dump = {} if o.boxes_json else None
    n, t0, last = 0, time.time(), 0.0
    tty = sys.stdout.isatty()

    def send(items):
        for e, boxes in items:
            if dump is not None:
                dump[e["i"]] = boxes
            e["gray"] = None
            q_out.put((e, boxes))

    try:
        while not stop.is_set():
            item = q_in.get()
            if item is None:
                break
            i, frame, t_in = item
            send(an.push(i, frame, t_in))
            n += 1
            if tty and time.time() - last > 0.5:
                last = time.time()
                lat = stats["latency"][-1] if stats["latency"] else 0
                print(f"\r  frame {i}  {n / (last - t0):.1f} fps  faces {len(an.tracks)}  "
                      f"latency {lat:.2f} s  dropped {stats['dropped']}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopping...")
    finally:
        stop.set()
        send(an.flush())
        q_out.put(None)
        t_write.join()

    secs = time.time() - t0
    lat = sorted(stats["latency"]) or [0]
    s = an.stats
    print(f"\r  done: {n} frames in, {stats['written']} out, {stats['dropped']} dropped, "
          f"{n / max(secs, 1e-9):.1f} fps; latency median {lat[len(lat) // 2]:.2f} s, "
          f"p95 {lat[int(len(lat) * 0.95)]:.2f} s")
    print(f"  detector runs {s['detections']}, local re-detections {s['redetections']}, "
          f"look-back boxes {s['backfilled']}, re-filled boxes {s['refilled']}, "
          f"scene cuts {s['cuts']}, failed frames {s['failed']}")
    if dump is not None:
        Path(o.boxes_json).write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
