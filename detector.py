"""Face detectors and the blur itself.

use_model(name) picks the detector for this process (face_det_lite,
yolo11n_640, yolo11n_1280, yunet; use_npu() for the
Qualcomm NPU builds); then
detect_faces_scored(img) returns (x, y, w, h, score) boxes, and
blur_region(img, box) blurs one box in place.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# YuNet: small (230 KB) neural face detector bundled in models/ (MIT license).
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
# The same network with a free input size, for onnxruntime (OrtYuNet; source
# of the fixed-size NPU models in tools/make_yunet_qnn.py).
YUNET_DYNAMIC = YUNET_MODEL.with_name("face_detection_yunet_2023mar_dynamic.onnx")
SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K = 0.7, 0.3, 5000


_yunet = None


def detect_faces_yunet(img, min_size=24, score=SCORE_THRESHOLD):
    """Return a list of (x, y, w, h, score) face boxes using YuNet."""
    global _yunet
    h, w = img.shape[:2]
    if _yunet is None:
        _yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), score,
                                           NMS_THRESHOLD, TOP_K)
    _yunet.setInputSize((w, h))
    _yunet.setScoreThreshold(score)
    _, faces = _yunet.detect(img)
    if faces is None:
        return []
    return [tuple(int(v) for v in f[:4]) + (float(f[14]),)
            for f in faces if min(f[2], f[3]) >= min_size]


class OrtYuNet:
    """YuNet run by onnxruntime (any execution provider).

    Pre- and post-processing copy OpenCV's FaceDetectorYN, so the boxes match
    detect_faces_yunet. Used as the reference by tools/eval_yunet_qnn.py.
    """

    STRIDES = (8, 16, 32)

    def __init__(self, providers, align=32):
        import onnxruntime as ort
        # Inputs are zero-padded to a multiple of `align`. 32 is what the
        # network needs (and what OpenCV does).
        self.align = align
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(YUNET_DYNAMIC), opts, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.names = [o.name for o in self.sess.get_outputs()]

    def detect(self, img, min_size=24, thresh=SCORE_THRESHOLD):
        h, w = img.shape[:2]
        a = self.align
        ph, pw = ((h - 1) // a + 1) * a, ((w - 1) // a + 1) * a
        # The model takes the uint8 BGR image as is (it transposes and casts to
        # float itself), padded with zeros at the bottom/right.
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
            keep = np.nonzero(score >= thresh)[0]
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
                               thresh, NMS_THRESHOLD, top_k=TOP_K)
        idx = np.asarray(idx, int).reshape(-1)
        return [tuple(int(v) for v in f) + (float(sc),) for f, sc in zip(boxes[idx], scores[idx])
                if min(f[2], f[3]) >= min_size]


_npu = None


QNN_CACHE = Path.home() / ".cache" / "face-blur" / "qnn"


def use_npu(model="face_det_lite", size=None, precision=None, allow_cpu=False,
            cache_dir=QNN_CACHE, force_cpu=False):
    """Switch face detection to a Qualcomm Hexagon NPU (onnxruntime QNN EP,
    HTP backend).

    model: "face_det_lite" (default; Qualcomm Lightweight-Face-Detection,
    float), "yolo11n_640" / "yolo11n_1280" (YOLO11n-face, float) or "yunet"
    (fixed-size quantized YuNet from tools/make_yunet_qnn.py;
    size (w, h) / precision: see hexagon.SIZES / hexagon.PRECISIONS).
    allow_cpu: run the same model on the CPU EP when QNN is missing.
    force_cpu: always run it on the CPU EP.
    Returns a message naming the provider in use. Falls back to the normal
    CPU detector with a warning if QNN is unavailable.
    """
    global _npu
    try:
        import hexagon
    except ImportError:
        return "WARNING: --npu needs onnxruntime-qnn (pip install onnxruntime-qnn); using CPU"
    providers = ["CPUExecutionProvider"] if force_cpu else None
    if providers is None and not hexagon.qnn_available():
        if not allow_cpu:
            return "WARNING: onnxruntime has no QNN support (pip install onnxruntime-qnn); using CPU"
        providers = ["CPUExecutionProvider"]
    try:
        if model in hexagon.YOLO_SIZES:
            det = hexagon.QnnYolo11Face(model, providers=providers, cache_dir=cache_dir)
        elif model == "yunet":
            det = hexagon.QnnYuNet(size or hexagon.DEFAULT_SIZE, precision or hexagon.DEFAULT_PRECISION,
                                   providers=providers, cache_dir=cache_dir)
        else:
            det = hexagon.QnnFaceDetLite(providers=providers, cache_dir=cache_dir)
    except Exception as e:  # missing model, QNN backend library that fails to load, ...
        return f"WARNING: NPU detector could not start ({e}); using CPU"
    if not det.on_npu and not allow_cpu:
        return f"WARNING: QNN did not start (providers {det.providers}); using CPU"
    _npu = det
    return f"Face detection: {det.name} on {', '.join(det.providers)}"


# --------------------------------------------------------------------------- models

# yolo11n_640 (the default) / yolo11n_1280: YOLO11n-face (models/yolo11n_face/,
# AGPL-3.0 weights). 1280 finds small faces. Run on the Hexagon NPU when
# onnxruntime-qnn is installed, else on the CPU.
# face_det_lite: Qualcomm Lightweight-Face-Detection (BSD-3), models/face_det_lite/;
# NPU or CPU like YOLO. yunet: OpenCV YuNet, for simple scenes with large faces.
YOLO_MODELS = ("yolo11n_640", "yolo11n_1280")
MODELS = YOLO_MODELS + ("face_det_lite", "yunet")
DEFAULT_MODEL = "yolo11n_640"


_selected = False  # use_model() called; if not, the first detection loads DEFAULT_MODEL


def use_model(name, cpu=False):
    """Select the face detector. Returns a message describing it."""
    global _npu, _selected
    _npu = None
    _selected = True
    if name == "yunet":
        return "Face detector: YuNet (OpenCV)"
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {name!r}")
    msg = use_npu(name, allow_cpu=True, force_cpu=cpu)
    if msg.startswith("WARNING"):
        sys.exit(msg)
    return msg.replace("Face detection:", "Face detector:")


_threshold = None  # set by use_threshold(); None = the model's own threshold


def use_threshold(t=None):
    """Set the confidence a detection needs to count as a face (None = the
    model's default: yolo11n 0.3, face_det_lite 0.55, yunet 0.7)."""
    global _threshold
    if t is not None and not 0 < t < 1:
        raise ValueError("threshold must be between 0 and 1")
    _threshold = t


def _ensure_model():
    """No detector chosen yet (use_model / use_npu not called): load DEFAULT_MODEL."""
    if not _selected and _npu is None:
        use_model(DEFAULT_MODEL)


def face_threshold():
    """Score above which the current detector reports a face."""
    _ensure_model()
    if _threshold is not None:
        return _threshold
    if _npu is not None:
        return _npu.threshold
    return SCORE_THRESHOLD


# Tiling: besides the whole frame, the detector also runs on an N x N grid of
# overlapping tiles, so small faces are seen at a larger scale. Helps the
# detectors that shrink the frame to a fixed input (face_det_lite: 640x480,
# yolo11n_640: 640x384); costs 1 + N*N detector runs per frame.
_tiles = 1
TILE_OVERLAP = 0.25  # each tile reaches this share of its size into its neighbours


def use_tiles(n):
    """Detect on the whole frame plus an n x n grid of tiles (1 = off)."""
    global _tiles
    _tiles = max(1, int(n))


def tile_grid(W, H, n, overlap=TILE_OVERLAP):
    """(x0, y0, x1, y1) of an n x n grid of overlapping tiles."""
    def spans(size):
        step = size / n
        t = min(size, int(math.ceil(step * (1 + overlap))))
        return [(s, s + t) for s in (min(max(int(round(i * step - (t - step) / 2)), 0), size - t)
                                     for i in range(n))]
    return [(x0, y0, x1, y1) for y0, y1 in spans(H) for x0, x1 in spans(W)]


def merge_faces(faces, iou_thresh=0.3, inside=0.6):
    """Merge duplicates from overlapping tiles: boxes that overlap (IoU) or
    where one lies mostly inside the other are one face; the larger box is
    kept (a face cut by a tile edge gives a smaller, partial box)."""
    kept = []
    for f in sorted(faces, key=lambda f: -f[4]):
        x, y, w, h, _ = f
        for k, g in enumerate(kept):
            ix = max(0, min(x + w, g[0] + g[2]) - max(x, g[0]))
            iy = max(0, min(y + h, g[1] + g[3]) - max(y, g[1]))
            inter, a, b = ix * iy, w * h, g[2] * g[3]
            if inter and (inter / (a + b - inter) >= iou_thresh or inter / min(a, b) >= inside):
                if a > b:
                    kept[k] = (x, y, w, h, g[4])
                break
        else:
            kept.append(f)
    return kept


def _detect(img, min_size, thresh):
    _ensure_model()
    if _npu is not None:
        return _npu.detect(img, min_size, thresh)
    return detect_faces_yunet(img, min_size, thresh)


def detect_faces_scored(img, min_size=24, thresh=None):
    """Return a list of (x, y, w, h, score) face boxes with score >= thresh
    (default: the detector's normal threshold, see face_threshold)."""
    if thresh is None:
        thresh = face_threshold()
    faces = _detect(img, min_size, thresh)
    if _tiles > 1:
        H, W = img.shape[:2]
        for x0, y0, x1, y1 in tile_grid(W, H, _tiles):
            faces += [(x + x0, y + y0, w, h, sc)
                      for x, y, w, h, sc in _detect(img[y0:y1, x0:x1], min_size, thresh)]
        faces = merge_faces(faces)
    return faces


def detect_faces(img, min_size=24):
    """Return a list of (x, y, w, h) face boxes."""
    return [f[:4] for f in detect_faces_scored(img, min_size)]


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
