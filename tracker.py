"""Face tracking for live video.

Analyzer(opts, fps).push(frame) returns the faces of that frame as
(x, y, w, h, score, track_id), or None if analysing it failed (then blur the
whole frame). score is the detector confidence at the face's last detection.
Lookahead(analyzer, delay) adds an optional delay (a number of frames) that
uses the frames after a detection to fix the frames before it.
faceblur.StreamBlurrer wraps it; tools/blur_live.py is a preview CLI.

How it works:

  * Detector: YOLO11n-face 640 by default, or YOLO11n-face 1280,
    face_det_lite (Qualcomm Lightweight-Face-Detection) or YuNet, on every
    frame (options are the Analyzer's opts, named like the CLI flags).
  * Tracker: ByteTrack (bytetrack/, MIT): Kalman filter + Hungarian matching,
    high-score detections first, then the tracks left over against low-score
    ones (--low-score), so a face that turns or blurs keeps its track.
    Low-score detections never start a track. Every detection is blurred,
    also before ByteTrack confirms the track.
  * A face is blurred in the frames the detector finds it in (no hold); a
    delay (Lookahead) fills the frames it was missed in when it is found again.
  * Scene cuts reset tracking, so a box never jumps across a cut.
  * Fail closed: if analysing a frame fails, that frame is blurred whole.
"""
import math
import time
import threading
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

from detector import detect_faces_scored, face_threshold
from bytetrack import BYTETracker, TrackState


# --------------------------------------------------------------------------- geometry
# Boxes are float (x0, y0, x1, y1).

def clip_box(b, W, H):
    return np.array([min(max(b[0], 0), W), min(max(b[1], 0), H),
                     min(max(b[2], 0), W), min(max(b[3], 0), H)])


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

REID_SECONDS = 1.0  # ByteTrack keeps a lost track this long, to give it its old id back
BACKFILL_SCORE = 0.1  # detections kept down to this score, to follow a new face back
BACKFILL_IOU = 0.3    # overlap a box in an earlier frame needs to be the same face
TRACK_BYTES = 1600  # memory of one ByteTrack track with its Kalman state (tracemalloc)
FACE_BYTES = 500    # memory of one Face


class Face:
    """The box of one ByteTrack track in the current frame."""

    def __init__(self, box, frame, score):
        self.box = np.array(box, float)
        self.last_seen = frame   # last frame the detector found it
        self.score = score       # detector confidence at that frame


class Analyzer:
    """Detector + ByteTrack, one frame in, its face boxes out (no delay)."""

    def __init__(self, opts, fps):
        self.o = opts
        self.fps = fps
        self.low = getattr(opts, "low_score", 0.3)  # 0 = one-stage association only
        self.cut = SceneCut()
        self.i = 0
        self.cut_at = -1         # frame of the last scene cut
        self.dets = None         # detections of the last frame, see _analyse
        self.faces = {}          # ByteTrack track_id -> Face
        self.tracker = self._new_tracker()
        self.max_state = getattr(opts, "max_state_mb", 100) * 1e6
        self.reset_requested = False
        self.stats = dict(detections=0, low_matches=0, cuts=0, resets=0, failed=0, frames=0,
                          t_detect=0.0, t_track=0.0)

    def reset(self):
        """Forget every tracked face; the next frame is analysed from scratch,
        detector included. Safe to call from another thread: push() applies it."""
        self.reset_requested = True

    @property
    def state_bytes(self):
        """Memory held by the tracking state (estimate)."""
        t = self.tracker
        tracks = len(t.tracked_stracks) + len(t.lost_stracks) + len(t.removed_stracks)
        return tracks * TRACK_BYTES + len(self.faces) * FACE_BYTES

    def _new_tracker(self):
        hi = face_threshold()
        low = self.low if 0 < self.low < hi else hi
        # track_buffer is counted in tracker updates (one per frame).
        updates = max(1, math.ceil(REID_SECONDS * self.fps))
        t = BYTETracker(SimpleNamespace(track_thresh=hi, track_low_thresh=low, match_thresh=0.8,
                                        track_buffer=updates), frame_rate=30)
        t.det_thresh = hi  # ByteTrack starts tracks at thresh + 0.1; start at every confident face
        return t

    def push(self, frame):
        """Face boxes (x, y, w, h) in this frame, or None if the analysis failed
        (then blur the whole frame)."""
        i = self.i
        self.i += 1
        self.stats["frames"] += 1
        try:
            return self._analyse(i, frame)
        except Exception as e:  # fail closed
            self.stats["failed"] += 1
            print(f"\n  WARNING: frame {i} analysis failed ({e}); blurring the whole frame")
            return None

    def _analyse(self, i, frame):
        o = self.o
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cut = self.cut(gray)
        reset = self.reset_requested or self.state_bytes > self.max_state
        if cut:
            self.cut_at = i
            self.stats["cuts"] += 1
        elif reset:
            self.stats["resets"] += 1
        if cut or reset:
            self.reset_requested = False
            self.faces = {}  # never carry a box across a cut
            self.tracker = self._new_tracker()

        t = time.perf_counter()
        low = self.tracker.args.track_low_thresh
        found = detect_faces_scored(frame, o.min_size, min(low, BACKFILL_SCORE))
        t1 = time.perf_counter()
        self.stats["t_detect"] += t1 - t
        self.stats["detections"] += 1
        self.dets = np.array([[x, y, x + w, y + h, s] for x, y, w, h, s in found], float).reshape(-1, 5)
        self.tracker.update(self.dets[self.dets[:, 4] >= low])
        # ByteTrack keeps every track it ever removed and checks its lost tracks
        # against all of them on each update, so memory and time per frame grow for
        # as long as the stream runs. Only a removed track still among the tracked
        # or lost ones can change that check; drop the others.
        alive = {s.track_id for s in self.tracker.tracked_stracks + self.tracker.lost_stracks}
        self.tracker.removed_stracks = [s for s in self.tracker.removed_stracks if s.track_id in alive]
        hi = self.tracker.args.track_thresh
        fid = self.tracker.frame_id
        faces = {}
        for st in self.tracker.tracked_stracks + self.tracker.lost_stracks:
            f = self.faces.get(st.track_id)
            if st.state == TrackState.Tracked and st.frame_id == fid:  # found in this frame
                if f is not None and st.score < hi:
                    self.stats["low_matches"] += 1
                f = Face(st.tlbr, i, st.score) if f is None else f
                f.box, f.last_seen, f.score = st.tlbr, i, float(st.score)
            if f is not None:
                faces[st.track_id] = f
        self.faces = faces  # tracks ByteTrack removed are dropped
        t = t1

        out = []
        for k, f in self.faces.items():
            # Found in this frame (lost faces are kept for ByteTrack only).
            if f.last_seen == i:
                c = clip_box(f.box, W, H)
                if c[2] - c[0] >= 2 and c[3] - c[1] >= 2:
                    out.append((int(c[0]), int(c[1]), int(round(c[2] - c[0])), int(round(c[3] - c[1])),
                                f.score, k))
        self.stats["t_track"] += time.perf_counter() - t
        return out

    @property
    def tracks(self):
        return list(self.faces.values())


# --------------------------------------------------------------------------- optional delay

def best_match(dets, box):
    """The detection box (x0, y0, x1, y1) overlapping `box` most, if its IoU
    is at least BACKFILL_IOU, else None."""
    if len(dets) == 0:
        return None
    d = dets[:, :4]
    ix = np.maximum(0, np.minimum(d[:, 2], box[2]) - np.maximum(d[:, 0], box[0]))
    iy = np.maximum(0, np.minimum(d[:, 3], box[3]) - np.maximum(d[:, 1], box[1]))
    inter = ix * iy
    union = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    iou = inter / np.maximum(union, 1e-9)
    k = int(iou.argmax())
    return d[k].copy() if iou[k] >= BACKFILL_IOU else None


def covered(faces, box):
    """True if a face (x, y, w, h, ...) in `faces` is the same face as `box`
    (x0, y0, x1, y1): they overlap by 60% of the smaller one."""
    for g in faces:
        ix = max(0, min(box[2], g[0] + g[2]) - max(box[0], g[0]))
        iy = max(0, min(box[3], g[1] + g[3]) - max(box[1], g[1]))
        if ix * iy >= 0.6 * min((box[2] - box[0]) * (box[3] - box[1]), g[2] * g[3]):
            return True
    return False


class Lookahead:
    """A buffer of analysed frames, so what is found later fixes them:

      * a face seen for the first time is followed back through the held
        frames by matching it to the detector's boxes there, low scores
        included (the face was there but scored below the threshold);
      * a face found again after it was lost (same ByteTrack id) is filled in
        the frames between, by interpolating its box.

    put(frame) analyses a frame and adds it; pop() takes the oldest out as
    (frame, faces); len() is the number held. The caller decides when to pop,
    i.e. how many frames to keep. put() and pop() may run in two threads.
    push(frame) = put, then pop while more than `delay` are held.
    """

    def __init__(self, analyzer, delay=0):
        self.an, self.delay = analyzer, int(delay)
        self.buf = deque()       # [i, frame, faces or None, detections or None (failed frame)]
        self.seen = {}           # track id -> (frame index, box) where last present
        self.i = 0
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.buf)

    def put(self, frame):
        faces = self.an.push(frame)          # the slow part, outside the lock
        dets = self.an.dets if faces is not None else None
        with self.lock:
            i = self.i
            self.i += 1
            if faces is not None:
                faces = list(faces)
                for f in faces:
                    tid, box = f[5], np.array([f[0], f[1], f[0] + f[2], f[1] + f[3]], float)
                    last = self.seen.get(tid)
                    if last is None:
                        self._back(f, box)
                    elif i - last[0] > 1:
                        self._gap(f, last, i, box)
                    self.seen[tid] = (i, box)
            self.buf.append([i, frame, faces, dets])
            if i % 100 == 0:  # forget faces last seen before the buffer
                first = self.buf[0][0]
                self.seen = {k: v for k, v in self.seen.items() if v[0] >= first - 1}

    def pop(self):
        """The oldest (frame, faces), or None if empty."""
        with self.lock:
            return self._emit(self.buf.popleft()) if self.buf else None

    def push(self, frame):
        self.put(frame)
        out = []
        while len(self.buf) > self.delay:
            out.append(self.pop())
        return out

    def flush(self):
        with self.lock:
            out = [self._emit(e) for e in self.buf]
            self.buf.clear()
        return out

    @staticmethod
    def _emit(e):
        return e[1], e[2]

    @staticmethod
    def _add(entry, box, f):
        if entry[2] is None or any(g[5] == f[5] for g in entry[2]):
            return
        H, W = entry[1].shape[:2]
        c = clip_box(box, W, H)
        if c[2] - c[0] < 2 or c[3] - c[1] < 2:
            return
        new = (int(c[0]), int(c[1]), int(round(c[2] - c[0])), int(round(c[3] - c[1])), f[4], f[5])
        # The same face under another track id (ByteTrack started a new one):
        # keep one box, the larger.
        for k, g in enumerate(entry[2]):
            ix = max(0, min(new[0] + new[2], g[0] + g[2]) - max(new[0], g[0]))
            iy = max(0, min(new[1] + new[3], g[1] + g[3]) - max(new[1], g[1]))
            a, b = new[2] * new[3], g[2] * g[3]
            if ix * iy >= 0.6 * min(a, b):
                if a > b:
                    entry[2][k] = new
                return
        entry[2].append(new)

    def _back(self, f, box):
        """A new face: follow it back through the held frames, from detector box
        to detector box (best overlap, low scores included). Stops where no box
        matches (the face was not there yet, or hidden), where the face is
        already blurred (an earlier track covers it from there on), at a scene
        cut and at a failed frame."""
        b = box
        for e in reversed(self.buf):
            if e[2] is None or e[0] < self.an.cut_at or covered(e[2], b):
                break
            m = best_match(e[3], b)
            if m is None:
                break
            self._add(e, m, f)
            b = m

    def _gap(self, f, last, i, box):
        """A face found again: fill the held frames since it was last present."""
        j, prev = last
        for e in self.buf:
            if j < e[0] < i:
                k = (e[0] - j) / (i - j)
                self._add(e, prev + (box - prev) * k, f)
