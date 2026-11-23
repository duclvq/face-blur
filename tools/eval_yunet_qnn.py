#!/usr/bin/env python3
"""Compare the fixed-size / quantized YuNet models against the float dynamic
YuNet at full resolution (the reference).

For each model: faces found, recall / precision against the reference boxes
(IoU >= 0.5, both at the same score threshold), mean |score change| of the
matched faces, and ms per frame (letterbox + inference + decode).

Data: examples/input images at their own size, and the street clip (or any
clip given with --video, e.g. a 1080p copy) without the calibration frames
(every 10th frame).

    python tools/eval_yunet_qnn.py [--video street1080.mp4] [--qnn]

--qnn runs the models with the QNN EP (on a Snapdragon device) instead of the
CPU EP; the reference always runs on the CPU.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import detector  # noqa: E402
from hexagon import PRECISIONS, SIZES, QnnYuNet, model_path  # noqa: E402
from make_yunet_qnn import CALIB_EVERY, IMAGE_EXTS, STREET, street_frames  # noqa: E402


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match(ref, det):
    """Greedy IoU >= 0.5 matching. Returns [(ref_i, det_j)]."""
    pairs = sorted(((iou(r, d), i, j) for i, r in enumerate(ref) for j, d in enumerate(det)), reverse=True)
    used_r, used_d, out = set(), set(), []
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in used_r and j not in used_d:
            used_r.add(i), used_d.add(j), out.append((i, j))
    return out


def clip(faces, shape):
    H, W = shape[:2]
    out = []
    for x, y, w, h, s in faces:
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1 - x0, y1 - y0, s))
    return out


class Stats:
    def __init__(self):
        self.n_ref = self.n_det = self.tp = self.n_ref_big = self.tp_big = 0
        self.drift, self.ms = [], []

    def add(self, ref, det, ms, big=32):
        m = match(ref, det)
        self.n_ref += len(ref)
        self.n_det += len(det)
        self.tp += len(m)
        big_ref = {i for i, r in enumerate(ref) if min(r[2], r[3]) >= big}
        self.n_ref_big += len(big_ref)
        self.tp_big += sum(i in big_ref for i, _ in m)
        self.drift += [abs(ref[i][4] - det[j][4]) for i, j in m]
        self.ms.append(ms)

    def row(self):
        r = self.tp / max(self.n_ref, 1)
        rb = self.tp_big / max(self.n_ref_big, 1)
        p = self.tp / max(self.n_det, 1)
        d = np.mean(self.drift) if self.drift else 0.0
        return (f"{self.n_det:6d} {r:7.3f} {rb:8.3f} {p:7.3f} {d:7.4f} "
                f"{np.median(self.ms):8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=STREET)
    ap.add_argument("--thresh", type=float, default=detector.SCORE_THRESHOLD)
    ap.add_argument("--sizes", nargs="+", default=[f"{w}x{h}" for w, h in SIZES])
    ap.add_argument("--precisions", nargs="+", default=list(PRECISIONS))
    ap.add_argument("--qnn", action="store_true", help="run the models on the QNN EP")
    ap.add_argument("--max-frames", type=int, default=0)
    o = ap.parse_args()

    sets = {"images": [cv2.imread(str(p)) for p in sorted((ROOT / "examples" / "input").rglob("*"))
                       if p.suffix.lower() in IMAGE_EXTS]}
    frames = [f for i, f in street_frames(o.video) if i % CALIB_EVERY]
    sets["video"] = frames[: o.max_frames] if o.max_frames else frames
    print(f"images: {len(sets['images'])}, video frames: {len(sets['video'])} "
          f"({sets['video'][0].shape[1]}x{sets['video'][0].shape[0]}), threshold {o.thresh}")

    ref_det = detector.OrtYuNet(["CPUExecutionProvider"])
    ref_det.detect(sets["video"][0], 0, o.thresh)
    refs, ref_ms = {}, {}
    for name, imgs in sets.items():
        refs[name], ref_ms[name] = [], []
        for im in imgs:
            t = time.perf_counter()
            refs[name].append(clip(ref_det.detect(im, 0, o.thresh), im.shape))
            ref_ms[name].append((time.perf_counter() - t) * 1e3)
    print("reference = float dynamic YuNet at full resolution, CPU EP")
    print(f"{'set':7s} {'model':16s} {'faces':>6s} {'recall':>7s} {'rec>=32':>8s} {'prec':>7s} "
          f"{'|dS|':>7s} {'ms/frame':>8s}")
    for name in sets:
        print(f"{name:7s} {'reference':16s} {sum(map(len, refs[name])):6d} {1:7.3f} {1:8.3f} {1:7.3f} "
              f"{0:7.4f} {np.median(ref_ms[name]):8.1f}")
    providers = None if o.qnn else ["CPUExecutionProvider"]
    for sz in o.sizes:
        size = tuple(int(v) for v in sz.split("x"))
        for prec in o.precisions:
            if not model_path(size, prec).exists():
                continue
            det = QnnYuNet(size, prec, providers=providers)
            det.detect(sets["video"][0], 0, o.thresh)  # warm-up
            for name, imgs in sets.items():
                st = Stats()
                for im, ref in zip(imgs, refs[name]):
                    t = time.perf_counter()
                    faces = det.detect(im, 0, o.thresh)
                    st.add(ref, faces, (time.perf_counter() - t) * 1e3)
                print(f"{name:7s} {sz + '_' + prec:16s} {st.row()}   [{det.provider}]")


if __name__ == "__main__":
    main()
