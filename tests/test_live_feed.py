"""Real-world tests for the live-feed API (FaceBlurrer.stream: put / get / delay).

Footage: a real street crowd (Times Square). By default the original half of
examples/video/street_before_after.mp4 (640x360, 29.97 fps); point
FACEBLUR_TEST_VIDEO at any other clip (FACEBLUR_TEST_CROP=0 to use the whole
frame, FACEBLUR_TEST_FRAMES to limit the length).

    python -m pytest tests/test_live_feed.py -s        # assertions + printed numbers
    python tests/test_live_feed.py                     # the report only

What is checked:
  * contract: get() returns None until `delay` newer frames are behind the
    oldest one, every frame comes out exactly once, in order, unchanged,
    with buffered / ready matching the buffer;
  * live.delay changed while running: no frame lost, repeated or reordered;
  * two threads (README pattern): input paced at the clip's frame rate with
    arrival jitter, output on its own fixed clock, last result reused on
    None, get(force=True) when the buffer grows too far;
  * accuracy: faces left unblurred against a stronger reference
    (yolo11n_1280 on every frame, score >= 0.5), with and without a delay.
    This is a proxy ground truth, not hand labels.
"""
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import detector  # noqa: E402
from faceblur import FaceBlurrer  # noqa: E402

DEFAULT_CLIP = ROOT / "examples/video/street_before_after.mp4"
VIDEO = Path(os.environ.get("FACEBLUR_TEST_VIDEO", DEFAULT_CLIP))
CROP = os.environ.get("FACEBLUR_TEST_CROP", "1" if VIDEO == DEFAULT_CLIP else "0") == "1"
MAX_FRAMES = int(os.environ.get("FACEBLUR_TEST_FRAMES", "0")) or None

REF_MODEL, REF_SCORE = "yolo11n_1280", 0.5   # reference "ground truth"
MIN_REF_SIZE = 16                            # ignore reference faces smaller than this (px)
COVERED = 0.5                                # a face counts as blurred if this much of it is boxed


# --------------------------------------------------------------------------- footage

def clip_fps():
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return fps


def frames():
    """Decoded BGR frames of the clip, one at a time (as a live source would)."""
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        pytest.skip(f"cannot open {VIDEO}")
    n = 0
    while MAX_FRAMES is None or n < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        if CROP:  # side-by-side before/after: keep the original half
            f = np.ascontiguousarray(f[:, : f.shape[1] // 2])
        n += 1
        yield f
    cap.release()


_reference = None


def reference_faces():
    """Per frame, the faces a stronger detector finds: [(x, y, w, h), ...]."""
    global _reference
    if _reference is None:
        detector.use_model(REF_MODEL)
        detector.use_threshold(None)
        _reference = [[f[:4] for f in detector.detect_faces_scored(fr, 0, REF_SCORE)
                       if min(f[2], f[3]) >= MIN_REF_SIZE] for fr in frames()]
    return _reference


def coverage(face, boxes):
    """Share of `face` covered by the union of `boxes`."""
    x, y, w, h = face
    m = np.zeros((h, w), bool)
    for bx, by, bw, bh in boxes:
        x0, y0 = max(bx - x, 0), max(by - y, 0)
        x1, y1 = min(bx + bw - x, w), min(by + bh - y, h)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m.mean() if m.size else 1.0


# --------------------------------------------------------------------------- runs

def run_single_thread(delay, model="yolo11n_640"):
    """The quickstart loop: put() each frame, get() while frames are ready, flush at the end.
    Returns (boxes per output frame, put() times in ms, max frames held)."""
    fb = FaceBlurrer(model=model, mask=False)
    live = fb.stream(fps=clip_fps(), delay=delay)
    out, put_ms, held = [], [], 0
    for f in frames():
        t = time.perf_counter()
        live.put(f)
        put_ms.append((time.perf_counter() - t) * 1000)
        held = max(held, live.buffered)
        while (res := live.get()) is not None:
            out.append(res.boxes)
    out += [r.boxes for r in live.flush()]
    return out, put_ms, held


def miss_rate(boxes_per_frame, ref):
    assert len(boxes_per_frame) == len(ref)
    total = missed = frames_with_miss = 0
    for boxes, faces in zip(boxes_per_frame, ref):
        m = sum(coverage(face, boxes) < COVERED for face in faces)
        total += len(faces)
        missed += m
        frames_with_miss += m > 0
    return missed, total, frames_with_miss


def run_two_threads(delay, max_extra, jitter, seed=0, model="yolo11n_640"):
    """Input thread paced at the clip fps with random arrival jitter; output on a fixed
    clock in this thread (README pattern). Returns a dict of what happened."""
    fps = clip_fps()
    period = 1.0 / fps
    fb = FaceBlurrer(model=model, mask=True)
    live = fb.stream(fps=fps, delay=delay)
    rng = random.Random(seed)
    tag, arrival = {}, {}            # id(frame) -> input index, input index -> arrival time
    done, errors = threading.Event(), []
    t0 = time.perf_counter() + 0.2

    def producer():
        try:
            for k, f in enumerate(frames()):
                target = t0 + k * period + rng.uniform(0, jitter)
                while (now := time.perf_counter()) < target:
                    time.sleep(min(0.002, target - now))
                tag[id(f)] = k
                arrival[k] = time.perf_counter()
                live.put(f)
        except Exception as e:  # pragma: no cover - reported by the assertion below
            errors.append(e)
        finally:
            done.set()

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    order, latency, get_ms, ticks, repeats, forced = [], [], [], 0, 0, 0
    last = None
    tick = t0 + period / 2
    while True:
        while (now := time.perf_counter()) < tick:
            time.sleep(min(0.002, tick - now))
        tick += period
        ticks += 1
        finished = done.is_set()
        if finished and live.buffered == 0:
            break
        tg = time.perf_counter()
        if live.buffered > delay + max_extra or (finished and live.buffered):
            res = live.get(force=True)
            forced += not finished
        else:
            res = live.get()
        get_ms.append((time.perf_counter() - tg) * 1000)  # time the output clock waits on put()
        if res is None:
            repeats += last is not None   # output clock: show the last frame again
            continue
        k = tag.pop(id(res.original))
        order.append(k)
        latency.append(time.perf_counter() - arrival[k])
        res.blurred                       # make the blurred copy, as a real output would
        last = res
    th.join()
    return dict(errors=errors, order=order, latency_ms=[x * 1000 for x in latency], get_ms=get_ms,
                ticks=ticks, repeats=repeats, forced=forced)


# --------------------------------------------------------------------------- tests

def test_put_get_contract():
    delay = 12
    fb = FaceBlurrer(mask=True)
    live = fb.stream(fps=clip_fps(), delay=delay)
    src, outs, sums = [], [], []
    for k, f in enumerate(frames()):
        src.append(f)
        sums.append(int(f.sum()))
        live.put(f)
        assert live.buffered == min(k + 1, delay + 1)
        assert live.ready == (1 if k >= delay else 0)
        res = live.get()
        if k < delay:
            assert res is None, f"frame came out after {k + 1} puts with delay={delay}"
        else:
            assert res is not None
            outs.append(res)
        assert live.get() is None  # nothing more ready until the next put
    assert live.buffered == delay
    outs += live.flush()
    assert live.buffered == 0 and live.get() is None and live.get(force=True) is None

    assert len(outs) == len(src)
    H, W = src[0].shape[:2]
    blurred_frames = 0
    for k, (f, res) in enumerate(zip(src, outs)):
        assert res.original is f, f"output {k} is not input {k}"
        assert int(res.original.sum()) == sums[k], "input frame was modified"
        assert not res.failed
        assert len(res.boxes) == len(res.scores)
        for x, y, w, h in res.boxes:
            assert 0 <= x and 0 <= y and w >= 2 and h >= 2 and x + w <= W and y + h <= H
        if res.boxes:
            assert res.blurred.shape == f.shape and not np.array_equal(res.blurred, f)
            blurred_frames += 1
    assert blurred_frames > 0.8 * len(src), "a crowd clip should have faces in most frames"


def test_delay_changed_while_running():
    fb = FaceBlurrer(mask=False)
    live = fb.stream(fps=clip_fps(), delay=0)
    schedule = {0: 0, 40: 12, 120: 4, 200: 25, 300: 0}
    tag, order = {}, []
    for k, f in enumerate(frames()):
        if k in schedule:
            live.delay = schedule[k]
        tag[id(f)] = k
        live.put(f)
        while (res := live.get()) is not None:
            order.append(tag.pop(id(res.original)))
        assert live.buffered == min(live.delay, k + 1 - len(order))
    order += [tag.pop(id(r.original)) for r in live.flush()]
    assert order == list(range(len(order))), "frames lost, repeated or reordered"
    with pytest.raises(ValueError):
        live.delay = -1
    with pytest.raises(ValueError):
        live.delay = 2.5


@pytest.mark.parametrize("delay", [0, 12])
def test_two_threads_fixed_output_clock(delay):
    r = run_two_threads(delay=delay, max_extra=6, jitter=0.5 / clip_fps())
    print(f"\n  two threads, delay={delay}: {len(r['order'])} frames out in {r['ticks']} ticks, "
          f"{r['repeats']} repeats, {r['forced']} forced, latency median "
          f"{statistics.median(r['latency_ms']):.0f} ms, p95 "
          f"{np.percentile(r['latency_ms'], 95):.0f} ms")
    assert not r["errors"], r["errors"]
    assert r["order"] == list(range(len(r["order"]))), "frames lost, repeated or reordered"
    assert len(r["order"]) == sum(1 for _ in frames())


def test_delay_leaves_fewer_faces_unblurred():
    ref = reference_faces()
    total = sum(map(len, ref))
    assert total > 0
    results = {d: miss_rate(run_single_thread(d)[0], ref)[0] for d in (0, 12)}
    print(f"\n  unblurred reference faces (of {total}): "
          + ", ".join(f"delay={d}: {m}" for d, m in results.items()))
    assert results[12] <= results[0], "delay left more faces unblurred"


# --------------------------------------------------------------------------- report

def report():
    fps = clip_fps()
    first = next(frames())
    n = sum(1 for _ in frames())
    print(f"Clip: {VIDEO.name}{' (original half)' if CROP else ''}, "
          f"{first.shape[1]}x{first.shape[0]}, {fps:.2f} fps, {n} frames ({n / fps:.1f} s)")
    t = time.perf_counter()
    ref = reference_faces()
    total = sum(map(len, ref))
    print(f"Reference: {REF_MODEL} >= {REF_SCORE}, faces >= {MIN_REF_SIZE} px: {total} faces "
          f"in {sum(1 for r in ref if r)} frames ({time.perf_counter() - t:.0f} s)\n")

    print("| delay | unblurred faces | frames with a miss | put() median / p95 ms | max held |")
    print("|---|---|---|---|---|")
    for delay in (0, 6, 12, 25):
        boxes, put_ms, held = run_single_thread(delay)
        m, tot, fm = miss_rate(boxes, ref)
        print(f"| {delay} | {m} ({100 * m / tot:.1f}%) | {fm} | "
              f"{statistics.median(put_ms):.1f} / {np.percentile(put_ms, 95):.1f} | {held} |")

    print("\nTwo threads, input paced at the clip rate with jitter, output on a fixed clock:")
    print("| delay | jitter | frames out | ticks | repeats | forced | latency median / p95 ms | get() max ms | in order |")
    print("|---|---|---|---|---|---|---|---|---|")
    for delay, jit in ((0, 0.5), (12, 0.5), (12, 2.0)):
        r = run_two_threads(delay=delay, max_extra=6, jitter=jit / fps)
        ok = not r["errors"] and r["order"] == list(range(n))
        print(f"| {delay} | {jit:g} frame | {len(r['order'])} | {r['ticks']} | {r['repeats']} | "
              f"{r['forced']} | {statistics.median(r['latency_ms']):.0f} / "
              f"{np.percentile(r['latency_ms'], 95):.0f} | {max(r['get_ms']):.1f} | {'yes' if ok else 'NO ' + str(r['errors'])} |")


if __name__ == "__main__":
    report()
