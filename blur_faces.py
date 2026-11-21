#!/usr/bin/env python3
"""Detect faces in every image of a dataset folder and blur them.

Usage:
    python blur_faces.py INPUT_DIR OUTPUT_DIR [--method gaussian|pixelate] [--strength N]

The folder structure of INPUT_DIR is mirrored into OUTPUT_DIR.
Images with no detected faces are copied unchanged.
"""
import argparse
import csv
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# YuNet: small (230 KB) neural face detector bundled in models/ (MIT license).
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
# The same network with a free input size, for onnxruntime (GPU path).
YUNET_DYNAMIC = YUNET_MODEL.with_name("face_detection_yunet_2023mar_dynamic.onnx")
SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K = 0.6, 0.3, 5000
# Images of one size needed before that size moves to the GPU (see use_gpu).
# Roughly where the one-off GPU set-up for a size (a few seconds) is paid back.
GPU_AFTER_IMAGES = 20


_yunet = None


def detect_faces_yunet(img, min_size=24, score=SCORE_THRESHOLD):
    """Return a list of (x, y, w, h) face boxes using YuNet."""
    global _yunet
    h, w = img.shape[:2]
    if _yunet is None:
        _yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), score,
                                           NMS_THRESHOLD, TOP_K)
    _yunet.setInputSize((w, h))
    _, faces = _yunet.detect(img)
    if faces is None:
        return []
    return [tuple(int(v) for v in f[:4]) for f in faces if min(f[2], f[3]) >= min_size]


class OrtYuNet:
    """YuNet run by onnxruntime, so it can use an NVIDIA GPU (CUDA).

    Pre- and post-processing copy OpenCV's FaceDetectorYN, so the boxes match
    the CPU path.
    """

    STRIDES = (8, 16, 32)

    def __init__(self, providers, align=32):
        import onnxruntime as ort
        # Inputs are zero-padded to a multiple of `align`. 32 is what the
        # network needs (and what OpenCV does). A coarser value gives fewer
        # distinct input sizes, which matters on the GPU: the first image of
        # each new size pays a one-off cuDNN set-up of a few seconds.
        self.align = align
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(YUNET_DYNAMIC), opts, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.names = [o.name for o in self.sess.get_outputs()]

    def detect(self, img, min_size=24):
        h, w = img.shape[:2]
        a = self.align
        ph, pw = ((h - 1) // a + 1) * a, ((w - 1) // a + 1) * a
        # The model takes the uint8 BGR image as is (it transposes and casts to
        # float itself, on the GPU), padded with zeros at the bottom/right.
        if (ph, pw) != (h, w):
            img = cv2.copyMakeBorder(img, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
        blob = np.ascontiguousarray(img)[None]
        out = dict(zip(self.names, self.sess.run(None, {"input": blob})))

        boxes, scores = [], []
        for s in self.STRIDES:
            cols = pw // s
            cls = np.clip(out[f"cls_{s}"][0, :, 0], 0, 1)
            obj = np.clip(out[f"obj_{s}"][0, :, 0], 0, 1)
            score = np.sqrt(cls * obj)
            keep = np.nonzero(score >= SCORE_THRESHOLD)[0]
            if not keep.size:
                continue
            bb = out[f"bbox_{s}"][0, keep]
            c, r = keep % cols, keep // cols
            cx, cy = (c + bb[:, 0]) * s, (r + bb[:, 1]) * s
            bw, bh = np.exp(bb[:, 2]) * s, np.exp(bb[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1))
            scores.append(score[keep])
        if not boxes:
            return []
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes(boxes.astype(np.int32).tolist(), scores.tolist(),
                               SCORE_THRESHOLD, NMS_THRESHOLD, top_k=TOP_K)
        faces = boxes[np.asarray(idx, int).reshape(-1)]
        return [tuple(int(v) for v in f) for f in faces if min(f[2], f[3]) >= min_size]


_ort = None
# The first run of each new input size on the GPU costs a few seconds of cuDNN
# set-up; after that a frame takes milliseconds. With gpu_after=N, an image
# size goes to the GPU only once N images of that size have been seen, so a
# dataset of mixed sizes is never much slower than the CPU, and one with many
# same-size images still gets the speed-up. Video has one size: N = 0.
_gpu_after = 0
_size_counts = {}
gpu_stats = {"gpu": 0, "cpu": 0}


def use_gpu(align=32, gpu_after=0):
    """Switch face detection to onnxruntime on an NVIDIA GPU.

    Returns a message describing the device in use. Falls back to the CPU
    (OpenCV) detector with a warning if onnxruntime-gpu or CUDA is missing.
    """
    global _ort, _gpu_after
    try:
        import onnxruntime as ort
    except ImportError:
        return "WARNING: --gpu needs onnxruntime-gpu (pip install -r requirements-gpu.txt); using CPU"
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return "WARNING: onnxruntime has no CUDA support (install onnxruntime-gpu); using CPU"
    if hasattr(ort, "preload_dlls"):
        # CUDA/cuDNN from the nvidia-* pip packages, if installed. It prints a
        # line per missing library, which is noise when they come from PATH.
        import contextlib, io
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ort.preload_dlls()
        except Exception:
            pass
    # HEURISTIC picks fast kernels without timing every candidate; "DEFAULT"
    # sets up faster but runs ~10x slower per frame.
    det = OrtYuNet([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                    "CPUExecutionProvider"], align)
    if det.provider != "CUDAExecutionProvider":
        return "WARNING: CUDA could not start (driver or CUDA/cuDNN libraries missing); using CPU"
    _ort, _gpu_after = det, gpu_after
    return "Face detection on GPU (onnxruntime CUDA)"


def detect_faces(img, min_size=24):
    """Return a list of (x, y, w, h) face boxes."""
    if _ort is not None:
        a = _ort.align
        key = (-(-img.shape[0] // a), -(-img.shape[1] // a))
        _size_counts[key] = _size_counts.get(key, 0) + 1
        if _size_counts[key] > _gpu_after:
            gpu_stats["gpu"] += 1
            return _ort.detect(img, min_size)
        gpu_stats["cpu"] += 1
    return detect_faces_yunet(img, min_size)


def blur_region(img, box, method="gaussian", strength=3, pad=0.15, shape="oval"):
    """Blur one face box in place.

    pad   -- extra margin around the detected box, as a fraction of its size
    shape -- "rect" blurs the whole box, "oval" only an ellipse inside it
    """
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(x - px, 0), max(y - py, 0)
    x1, y1 = min(x + w + px, img.shape[1]), min(y + h + py, img.shape[0])
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return

    if method == "pixelate":
        blocks = max(2, 16 // strength)
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        # Blur a downscaled copy and scale it back up: looks the same as a huge
        # kernel on the full-size face, but stays fast on high-resolution photos.
        # Heavier blurs need less detail, so they work on a smaller copy
        # (64/32/16 px for strength 1/2/3) with the same kernel-to-face ratio.
        rh, rw = roi.shape[:2]
        s = min(max(strength, 1), 3)
        scale = min(1.0, (64 >> (s - 1)) / max(rh, rw))
        small = cv2.resize(roi, (max(1, round(rw * scale)), max(1, round(rh * scale))),
                           interpolation=cv2.INTER_AREA)
        k = max(small.shape[:2]) // (4 - s) | 1  # odd kernel
        small = cv2.GaussianBlur(small, (k, k), 0)
        blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_CUBIC)

    if shape == "oval":
        # Ellipse centred on the original face box, sized to the padded box,
        # with a soft edge. Drawn on a small copy and scaled up: the edge is a
        # smooth gradient anyway, and this is much faster on large faces.
        rh, rw = roi.shape[:2]
        ms = min(1.0, 64 / max(rh, rw))
        mw, mh = max(1, round(rw * ms)), max(1, round(rh * ms))
        mask = np.zeros((mh, mw), np.float32)
        sub = 16  # sub-pixel precision for cv2.ellipse (shift = 4 bits)
        center = (round((x + w / 2 - x0) * ms * sub), round((y + h / 2 - y0) * ms * sub))
        axes = (round((w / 2 + px) * ms * sub), round((h / 2 + py) * ms * sub))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA, 4)
        sigma = max(1, min(rw, rh) // 40) * ms
        if sigma >= 0.3:
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_LINEAR)
        roi[:] = cv2.blendLinear(blurred, roi, mask, 1 - mask)
    else:
        roi[:] = blurred


def padded_box(box, pad):
    """Return the (x0, y0, x1, y1) area blur_region() touches, unclipped."""
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    return x - px, y - py, x + w + px, y + h + py


def blurred_fraction(img_shape, faces, pad):
    """Fraction of the image covered by the (padded) face boxes."""
    H, W = img_shape[:2]
    boxes = [padded_box(b, pad) for b in faces]
    boxes = [(max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)) for x0, y0, x1, y1 in boxes]
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 0.0
    # Count the union only inside the area the boxes span (cheap on video frames).
    ox, oy = min(b[0] for b in boxes), min(b[1] for b in boxes)
    mask = np.zeros((max(b[3] for b in boxes) - oy, max(b[2] for b in boxes) - ox), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = 1
    return float(np.count_nonzero(mask)) / (H * W)


# Threads that load, blur and save images while faces are being detected.
IO_THREADS = 4

# Warn when the face boxes cover more than this share of a photo.
COVERAGE_WARN = 0.5


def process_dataset(in_dir, out_dir, method, strength, min_size,
                    pad=0.15, shape="oval"):
    in_dir, out_dir = Path(in_dir).resolve(), Path(out_dir).resolve()
    # Never write over the original dataset.
    if out_dir == in_dir or in_dir in out_dir.parents:
        sys.exit("OUTPUT_DIR must not be INPUT_DIR or a folder inside it "
                 "(that would overwrite or mix with the original images).")
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows, total_faces = [], 0

    def blur_and_save(img, faces, dst, row):
        for box in faces:
            blur_region(img, box, method, strength, pad, shape)
        if not cv2.imwrite(str(dst), img):
            row[1] = "error (write failed)"
            print(f"  ERROR: could not write {dst}")

    # Loading, and blurring + saving, run in worker threads (JPEG/PNG coding is
    # single-threaded in OpenCV); face detection stays in this thread, in order.
    # At most WINDOW images are in flight, so memory stays bounded.
    WINDOW = 8
    with ThreadPoolExecutor(IO_THREADS) as pool:
        loads, saves = deque(), deque()
        todo = iter(files)
        for src in todo:
            loads.append((src, pool.submit(cv2.imread, str(src))))
            if len(loads) >= WINDOW:
                break
        i = 0
        while loads:
            src, fut = loads.popleft()
            nxt = next(todo, None)
            if nxt is not None:
                loads.append((nxt, pool.submit(cv2.imread, str(nxt))))
            i += 1
            rel = str(src.relative_to(in_dir))
            img = fut.result()
            if img is None:
                print(f"[{i}/{len(files)}] SKIP (unreadable) {src}")
                report_rows.append([rel, "error", 0, 0])
                continue

            faces = detect_faces(img, min_size)
            covered = blurred_fraction(img.shape, faces, pad)
            dst = out_dir / src.relative_to(in_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            row = [rel, "ok", len(faces), round(covered * 100, 1)]
            report_rows.append(row)
            saves.append(pool.submit(blur_and_save, img, faces, dst, row))
            while len(saves) > WINDOW:
                saves.popleft().result()

            total_faces += len(faces)
            warn = f"  WARNING: {covered:.0%} of the image blurred" if covered > COVERAGE_WARN else ""
            print(f"[{i}/{len(files)}] {len(faces)} face(s)  {rel}{warn}")
        for f in saves:
            f.result()

    with open(out_dir / "blur_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "status", "faces", "blurred_pct"])
        writer.writerows(report_rows)

    print(f"\nDone: {len(files)} images, {total_faces} faces blurred -> {out_dir}")
    if _ort is not None:
        print(f"GPU detected faces in {gpu_stats['gpu']} images, CPU in {gpu_stats['cpu']} "
              f"(sizes with fewer than {_gpu_after} images stay on the CPU)")


def main():
    ap = argparse.ArgumentParser(description="Detect and blur faces in an image dataset.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=int, default=3, help="1 (light) .. 3 (heavy)")
    ap.add_argument("--min-size", type=int, default=24, help="smallest face size in px")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="margin around each face as a fraction of its size (0 = tight)")
    ap.add_argument("--shape", choices=["rect", "oval"], default="oval",
                    help="blur the whole box or an oval that follows the face")
    ap.add_argument("--gpu", action="store_true",
                    help="run face detection on an NVIDIA GPU (needs requirements-gpu.txt)")
    args = ap.parse_args()
    if not 0 <= args.pad <= 1:
        ap.error(f"--pad is a fraction of the face size between 0 and 1, not pixels "
                 f"(got {args.pad}; 0.15 adds a 15% margin). Large values blur the whole image.")
    if not YUNET_MODEL.exists():
        sys.exit(f"Model not found: {YUNET_MODEL}")
    if args.gpu:
        # Images: pad to multiples of 128 so mixed sizes share GPU set-ups
        # (same faces as with 32, within 2 px, on the test set).
        print(use_gpu(align=128, gpu_after=GPU_AFTER_IMAGES))
    process_dataset(args.input_dir, args.output_dir, args.method, args.strength,
                    args.min_size, args.pad, args.shape)


if __name__ == "__main__":
    main()
