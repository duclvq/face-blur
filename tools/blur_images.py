#!/usr/bin/env python3
"""Detect faces in every image of a folder and blur them.

Usage:
    python tools/blur_images.py INPUT_DIR OUTPUT_DIR [--model face_det_lite|yunet|...]
                                [--method gaussian|pixelate] [--strength N]

The folder structure of INPUT_DIR is mirrored into OUTPUT_DIR, with a
blur_report.csv. Images with no detected faces are copied unchanged.
"""
import argparse
import csv
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the library is in the repo root
from detector import (DEFAULT_MODEL, MODELS, YUNET_MODEL, blur_region,  # noqa: E402
                      detect_faces, use_model)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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


def main():
    ap = argparse.ArgumentParser(description="Detect and blur faces in an image dataset.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=int, default=3, help="1 (light) .. 3 (heavy)")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                    help="face detector: yolo11n_640 (default; YOLO11n-face, AGPL-3.0 weights), "
                         "yolo11n_1280 (finds small faces, about 4x slower), face_det_lite "
                         "(Qualcomm Lightweight-Face-Detection, misses small faces) or yunet "
                         "(simple scenes with large faces). All but yunet run on the Qualcomm "
                         "NPU with onnxruntime-qnn")
    ap.add_argument("--min-size", type=int, default=None,
                    help="smallest face size in px (default: 24 for yunet, no limit otherwise)")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="margin around each face as a fraction of its size (0 = tight)")
    ap.add_argument("--shape", choices=["rect", "oval"], default="oval",
                    help="blur the whole box or an oval that follows the face")
    ap.add_argument("--device", choices=["cpu", "gpu", "npu"], default="cpu",
                    help="where the detector runs: cpu (default), gpu (NVIDIA, needs onnxruntime-gpu) or npu (Qualcomm, needs onnxruntime-qnn)")
    ap.add_argument("--precision", choices=["fp32", "a16w8"], default=None,
                    help="which build of the model: fp32 (default) or a16w8 (quantized, for the "
                         "Qualcomm NPU). Not for yunet")
    args = ap.parse_args()
    if not 0 <= args.pad <= 1:
        ap.error(f"--pad is a fraction of the face size between 0 and 1, not pixels "
                 f"(got {args.pad}; 0.15 adds a 15% margin). Large values blur the whole image.")
    if args.min_size is None:
        args.min_size = 24 if args.model == "yunet" else 0
    if not YUNET_MODEL.exists():
        sys.exit(f"Model not found: {YUNET_MODEL}")
    print(use_model(args.model, args.device, args.precision))
    process_dataset(args.input_dir, args.output_dir, args.method, args.strength,
                    args.min_size, args.pad, args.shape)


if __name__ == "__main__":
    main()
