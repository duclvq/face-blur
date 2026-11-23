"""24/7 live-feed tests: memory and time per frame of a stream that never stops.

    python -m pytest tests/test_24_7.py -s                  # all but the real-time soak (~8 min)
    FACEBLUR_SOAK_MINUTES=30 python -m pytest tests/test_24_7.py -s -k soak

Input: a synthetic crowd (textured faces that walk in, stay, walk out) with
a scripted detector, so hours of feed run in minutes; the soak loops the real
street clip of test_live_feed.py at its real frame rate.

The scripted detector replaces tracker.detect_faces_scored only; ByteTrack,
the delay buffer and FrameResult are the real code.
"""
import os
import statistics
import threading
import time
import tracemalloc

import cv2
import numpy as np
import psutil
import pytest

from test_live_feed import clip_fps, frames as clip_frames  # also puts the repo on sys.path

import tracker  # noqa: E402
from faceblur import FaceBlurrer  # noqa: E402

FPS = 30


# --------------------------------------------------------------------------- synthetic crowd

class Crowd:
    """A textured background and textured face patches that walk through it.
    detect() reports the faces visible in the current frame, like a detector
    that misses one face in ten and sometimes gives a low score."""

    def __init__(self, W=640, H=360, spawn_per_s=2.0, seed=0):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.bg = cv2.GaussianBlur(self.rng.integers(0, 256, (H, W, 3), np.uint8), (0, 0), 1.2)
        self.spawn = spawn_per_s / FPS
        self.people = []
        self.spawned = 0
        self.calls = 0

    def step(self):
        r = self.rng
        if r.random() < self.spawn:
            s = int(r.integers(24, 72))
            tex = cv2.GaussianBlur(r.integers(0, 256, (s, s, 3), np.uint8), (0, 0), 1.0)
            side = r.integers(0, 2)
            self.people.append(dict(
                x=float(-s + 1 if side == 0 else self.W - 1), y=float(r.uniform(0, self.H - s)),
                vx=float(r.uniform(0.5, 3.0)) * (1 if side == 0 else -1), vy=float(r.uniform(-0.3, 0.3)),
                s=s, tex=tex, life=int(r.integers(FPS, 8 * FPS))))
            self.spawned += 1
        for p in self.people:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.people = [p for p in self.people
                       if p["life"] > 0 and -p["s"] < p["x"] < self.W and -p["s"] < p["y"] < self.H]

    def frame(self):
        img = self.bg.copy()
        for p in self.people:
            x0, y0, s = int(p["x"]), int(p["y"]), p["s"]
            ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + s, self.W), min(y0 + s, self.H)
            if ax1 > ax0 and ay1 > ay0:
                img[ay0:ay1, ax0:ax1] = p["tex"][ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        return img

    def next(self):
        self.step()
        return self.frame()

    def detect(self, img, min_size=0, thresh=None):
        self.calls += 1
        out = []
        for p in self.people:
            if self.rng.random() < 0.1:
                continue
            x0, y0 = max(p["x"], 0), max(p["y"], 0)
            x1, y1 = min(p["x"] + p["s"], self.W), min(p["y"] + p["s"], self.H)
            if x1 - x0 > 4 and y1 - y0 > 4:
                score = 0.8 if self.rng.random() < 0.85 else 0.2
                if thresh is None or score >= thresh:
                    out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), score))
        return out


@pytest.fixture
def scripted(monkeypatch):
    """Use a Crowd as the detector; returns a factory for crowds."""
    holder = {}

    def make(**kw):
        holder["crowd"] = c = Crowd(**kw)
        return c

    monkeypatch.setattr(tracker, "detect_faces_scored",
                        lambda img, min_size=0, thresh=None: holder["crowd"].detect(img, min_size, thresh))
    return make


def ms(t0):
    return (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- long run

HOURS = float(os.environ.get("FACEBLUR_SIM_HOURS", "0.17"))


def test_memory_and_frame_time_stay_flat(scripted):
    """A fixed camera with people coming and going, no scene cut, for
    FACEBLUR_SIM_HOURS of feed (default 10 min at 30 fps, about 2 per second
    walking through). Memory and the time per frame must not grow with it.
    Memory is compared between two points after the buffer is full, so the
    frames it holds cancel out."""
    crowd = scripted(spawn_per_s=2.0)
    live = FaceBlurrer(mask=False).stream(fps=FPS, delay=12)
    n = int(HOURS * 3600 * FPS)
    warm, window = n // 6, min(3000, n // 6)
    early, late = [], []
    mem_start = None
    for k in range(n):
        f = crowd.next()
        t = time.perf_counter()
        live.put(f)
        while live.get() is not None:
            pass
        dt = ms(t)
        if warm <= k < warm + window:
            early.append(dt)
        if k >= n - window:
            late.append(dt)
        if k == warm - 200:
            tracemalloc.start()
        if k == warm:
            mem_start = tracemalloc.take_snapshot()
    end = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grown = sum(s.size_diff for s in end.compare_to(mem_start, "filename"))
    top = end.compare_to(mem_start, "lineno")[:3]
    bt = live.analyzer.tracker
    print(f"\n  {n} frames ({n / FPS / 60:.0f} min of feed), {crowd.spawned} people walked through")
    print(f"  per frame: median {statistics.median(early):.2f} ms early, {statistics.median(late):.2f} ms late")
    print(f"  Python memory grown since frame {warm}: {grown / 1e6:.2f} MB; top: "
          + "; ".join(f"{s.traceback[0].filename.split(os.sep)[-1]}:{s.traceback[0].lineno} "
                      f"{s.size_diff / 1e6:+.2f} MB" for s in top))
    print(f"  ByteTrack lists: tracked {len(bt.tracked_stracks)}, lost {len(bt.lost_stracks)}, "
          f"removed {len(bt.removed_stracks)}; faces {len(live.analyzer.faces)}, "
          f"seen {len(live.buffer.seen)}, buffered {live.buffered}")
    assert grown < 1e6 * HOURS * 3, f"memory grows: {grown / 1e6:.1f} MB in {HOURS * 5 / 6:.2f} h of feed"
    assert statistics.median(late) < 1.5 * statistics.median(early) + 0.5, "frames get slower over time"


def test_frame_time_after_a_week_without_scene_cut(scripted):
    """ByteTrack keeps every track it ever removed (removed_stracks) and scans
    them on each update. A week on a busy fixed camera, one new face every 2 s:
    about 300k tracks. The time per frame must stay within 2 ms of a fresh stream."""
    crowd = scripted(spawn_per_s=2.0)
    live = FaceBlurrer(mask=False).stream(fps=FPS)

    def median_put(k=600):
        ts = []
        for _ in range(k):
            f = crowd.next()
            t = time.perf_counter()
            live.put(f)
            live.get()
            ts.append(ms(t))
        return statistics.median(ts[k // 3:])

    median_put(300)
    fresh = median_put()
    bt = live.analyzer.tracker
    template = bt.removed_stracks[:1] or bt.tracked_stracks[:1]
    assert template, "the scripted crowd produced no track"
    import copy
    week = 7 * 24 * 3600 // 2
    for k in range(week - len(bt.removed_stracks)):
        t = copy.copy(template[0])
        t.track_id = 10_000_000 + k
        bt.removed_stracks.append(t)
    aged = median_put()
    print(f"\n  per frame: {fresh:.2f} ms fresh, {aged:.2f} ms with {len(bt.removed_stracks)} "
          f"removed tracks (a week without a scene cut)")
    assert aged < fresh + 2.0, f"a week-old stream costs {aged - fresh:.1f} ms more per frame"


def test_reset_forgets_faces_keeps_frames(scripted):
    """live.reset() from the output thread: no frame lost, the faces found
    after it get new track ids, and the faces on screen stay blurred."""
    crowd = scripted(spawn_per_s=3.0)
    live = FaceBlurrer(mask=False).stream(fps=FPS, delay=6)
    tag, order, k = {}, [], 0
    for step in range(600):
        f = crowd.next()
        tag[id(f)] = step
        if step == 301:
            before = set(live.analyzer.faces)
            live.reset()
        live.put(f)
        if step == 301:
            after = set(live.analyzer.faces)
            visible = sum(1 for p in crowd.people if p["x"] > 0 and p["x"] + p["s"] < crowd.W)
        order += [tag.pop(id(r.original)) for r in iter(live.get, None)]
    order += [tag.pop(id(r.original)) for r in live.flush()]
    assert order == list(range(600))
    assert live.resets == 1
    assert before and after and not (before & after), "track ids carried across reset()"
    assert len(after) >= 0.5 * visible, "faces left unblurred right after reset()"


def test_state_limit_resets_the_stream(scripted):
    """With a tiny max_state_mb the stream resets itself whenever the tracking
    state passes it, and never holds more than the limit plus one frame of
    new faces (the check runs before each frame's update)."""
    crowd = scripted(spawn_per_s=3.0)
    live = FaceBlurrer(mask=False).stream(fps=FPS, max_state_mb=0.02)
    peak, max_faces = 0.0, 0
    for _ in range(3000):
        live.put(crowd.next())
        live.get()
        peak = max(peak, live.state_mb)
        max_faces = max(max_faces, len(crowd.people))
    print(f"\n  {live.resets} automatic resets, peak state {peak * 1000:.0f} KB")
    assert live.resets > 0
    one_frame = max_faces * (tracker.TRACK_BYTES + tracker.FACE_BYTES) / 1e6
    assert peak < 0.02 + one_frame, f"peak {peak * 1000:.0f} KB, limit 20 KB + one frame {one_frame * 1000:.0f} KB"
    with pytest.raises(ValueError):
        FaceBlurrer(mask=False).stream(fps=FPS, max_state_mb=0)


def test_default_state_stays_small(scripted):
    """At the default limit a busy camera never needs a reset: the tracking
    state stays far below 100 MB (removed tracks are not kept)."""
    crowd = scripted(spawn_per_s=3.0)
    live = FaceBlurrer(mask=False).stream(fps=FPS)
    peak = 0.0
    for _ in range(6000):
        live.put(crowd.next())
        live.get()
        peak = max(peak, live.state_mb)
    print(f"\n  peak tracking state {peak * 1000:.0f} KB over 6000 frames, {crowd.spawned} people")
    assert live.resets == 0 and peak < 1.0


# --------------------------------------------------------------------------- real-time soak (opt-in)

def real_frames(n=None):
    out = []
    for f in clip_frames():
        out.append(f)
        if n and len(out) >= n:
            break
    return out


SOAK_MIN = float(os.environ.get("FACEBLUR_SOAK_MINUTES", "0"))


@pytest.mark.skipif(SOAK_MIN <= 0, reason="set FACEBLUR_SOAK_MINUTES to run the real-time soak")
def test_soak_real_time():
    """The real clip looped (mirrored every other pass, so tracks restart) at
    its real frame rate for FACEBLUR_SOAK_MINUTES, input and output in two
    threads. Process memory (RSS) and the slow frames (p95) must stay flat."""
    clip = real_frames()
    fps = clip_fps()
    live = FaceBlurrer(mask=True).stream(fps=fps, delay=12)
    proc = psutil.Process()
    stop = threading.Event()
    put_ms, errors, outs, ticks = [], [], [0], [0]

    def producer():
        t0, k = time.perf_counter(), 0
        try:
            while not stop.is_set():
                f = clip[k % len(clip)]
                if (k // len(clip)) % 2:
                    f = np.ascontiguousarray(f[:, ::-1])
                target = t0 + k / fps
                while time.perf_counter() < target:
                    time.sleep(0.001)
                t = time.perf_counter()
                live.put(f)
                put_ms.append((t, ms(t)))
                k += 1
        except Exception as e:
            errors.append(e)

    def consumer():
        t0 = time.perf_counter()
        try:
            while not stop.is_set():
                ticks[0] += 1
                r = live.get(force=live.buffered > 12 + 30)
                if r is not None:
                    r.blurred
                    outs[0] += 1
                while time.perf_counter() < t0 + ticks[0] / fps:
                    time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    th = [threading.Thread(target=producer, daemon=True), threading.Thread(target=consumer, daemon=True)]
    for t in th:
        t.start()
    start = time.perf_counter()
    rss = []
    while time.perf_counter() - start < SOAK_MIN * 60 and not errors:
        time.sleep(30)
        rss.append((time.perf_counter() - start, proc.memory_info().rss / 1e6))
        print(f"  {rss[-1][0] / 60:5.1f} min  RSS {rss[-1][1]:7.1f} MB  out {outs[0]}  "
              f"buffered {live.buffered}", flush=True)
    stop.set()
    for t in th:
        t.join(timeout=10)
    assert not errors, errors
    base = statistics.median(m for _, m in rss[2:6]) if len(rss) > 6 else rss[0][1]
    assert rss[-1][1] < base + 50, f"RSS grew {rss[-1][1] - base:.0f} MB"
    t_end = put_ms[-1][0]
    first = [m for t, m in put_ms if t - put_ms[0][0] < 120]
    last = [m for t, m in put_ms if t_end - t < 120]
    p95a, p95b = np.percentile(first, 95), np.percentile(last, 95)
    print(f"  put() p95: {p95a:.1f} ms first 2 min, {p95b:.1f} ms last 2 min; "
          f"{outs[0]} frames out in {ticks[0]} ticks")
    assert p95b < 1.5 * p95a + 2, "slow frames got slower over the soak"
