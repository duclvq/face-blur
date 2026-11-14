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
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# YuNet: small (230 KB) neural face detector bundled in models/ (MIT license).
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


_yunet = None


def detect_faces_yunet(img, min_size=24, score=0.6):
    """Return a list of (x, y, w, h) face boxes using YuNet."""
    global _yunet
    h, w = img.shape[:2]
    if _yunet is None:
        _yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), score, 0.3, 5000)
    _yunet.setInputSize((w, h))
    _, faces = _yunet.detect(img)
    if faces is None:
        return []
    return [tuple(int(v) for v in f[:4]) for f in faces if min(f[2], f[3]) >= min_size]


def detect_faces(img, min_size=24):
    """Return a list of (x, y, w, h) face boxes."""
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
        rh, rw = roi.shape[:2]
        scale = min(1.0, 64 / max(rh, rw))
        small = cv2.resize(roi, (max(1, round(rw * scale)), max(1, round(rh * scale))),
                           interpolation=cv2.INTER_AREA)
        k = max(small.shape[:2]) // max(1, 4 - min(strength, 3)) | 1  # odd kernel
        small = cv2.GaussianBlur(small, (k, k), 0)
        blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)

    if shape == "oval":
        # Ellipse centred on the original face box, sized to the padded box.
        rh, rw = roi.shape[:2]
        mask = np.zeros((rh, rw), np.float32)
        center = (x + w // 2 - x0, y + h // 2 - y0)
        axes = (w // 2 + px, h // 2 + py)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, min(rw, rh) // 40))[..., None]  # soft edge
        roi[:] = (blurred * mask + roi * (1 - mask)).astype(roi.dtype)
    else:
        roi[:] = blurred


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

    for i, src in enumerate(files, 1):
        dst = out_dir / src.relative_to(in_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src))
        if img is None:
            print(f"[{i}/{len(files)}] SKIP (unreadable) {src}")
            report_rows.append((str(src.relative_to(in_dir)), "error", 0))
            continue

        faces = detect_faces(img, min_size)
        for box in faces:
            blur_region(img, box, method, strength, pad, shape)
        cv2.imwrite(str(dst), img)

        total_faces += len(faces)
        report_rows.append((str(src.relative_to(in_dir)), "ok", len(faces)))
        print(f"[{i}/{len(files)}] {len(faces)} face(s)  {src.relative_to(in_dir)}")

    with open(out_dir / "blur_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "status", "faces"])
        writer.writerows(report_rows)

    print(f"\nDone: {len(files)} images, {total_faces} faces blurred -> {out_dir}")


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
    args = ap.parse_args()
    if not YUNET_MODEL.exists():
        sys.exit(f"Model not found: {YUNET_MODEL}")
    process_dataset(args.input_dir, args.output_dir, args.method, args.strength,
                    args.min_size, args.pad, args.shape)


if __name__ == "__main__":
    main()
