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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# YuNet: small (230 KB) neural face detector bundled in models/ (MIT license).
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"

# Haar cascades ship with OpenCV; kept as a fallback (--detector haar).
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


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


def detect_faces(img, min_size=24, detector="yunet"):
    """Return a list of (x, y, w, h) face boxes."""
    if detector == "yunet":
        return detect_faces_yunet(img, min_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kw = dict(scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))

    boxes = list(FRONTAL.detectMultiScale(gray, **kw))
    boxes += list(PROFILE.detectMultiScale(gray, **kw))
    # Profile cascade only finds faces looking one way; flip to catch the other side.
    w_img = gray.shape[1]
    for (x, y, w, h) in PROFILE.detectMultiScale(cv2.flip(gray, 1), **kw):
        boxes.append((w_img - x - w, y, w, h))

    if not boxes:
        return []
    # Merge overlapping detections from the different cascades.
    rects = [list(map(int, b)) for b in boxes]
    merged, _ = cv2.groupRectangles(rects + rects, groupThreshold=1, eps=0.3)
    return [tuple(r) for r in merged]


def blur_region(img, box, method="gaussian", strength=3, pad=0.15):
    """Blur one face box in place, padded a little so edges are covered."""
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
        roi[:] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        k = max(roi.shape[0], roi.shape[1]) // max(1, 4 - min(strength, 3)) | 1  # odd kernel
        roi[:] = cv2.GaussianBlur(roi, (k, k), 0)


def process_dataset(in_dir, out_dir, method, strength, min_size, detector="yunet"):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
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
            report_rows.append((str(src), "error", 0))
            continue

        faces = detect_faces(img, min_size, detector)
        for box in faces:
            blur_region(img, box, method, strength)
        cv2.imwrite(str(dst), img)

        total_faces += len(faces)
        report_rows.append((str(src), "ok", len(faces)))
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
    ap.add_argument("--detector", choices=["yunet", "haar"], default="yunet",
                    help="yunet = accurate (default), haar = older OpenCV cascades")
    args = ap.parse_args()
    if args.detector == "yunet" and not YUNET_MODEL.exists():
        sys.exit(f"Model not found: {YUNET_MODEL}  (or use --detector haar)")
    process_dataset(args.input_dir, args.output_dir, args.method, args.strength,
                    args.min_size, args.detector)


if __name__ == "__main__":
    main()
