"""Face detectors and the blur itself.

use_model(name, device, precision) picks the detector for this process
(face_det_lite, yolo11n_640, yolo11n_1280, yunet), where it runs (DEVICES)
and which build (hexagon.PRECISIONS); use_npu() for
the quantized YuNet NPU builds; then
detect_faces_scored(img) returns (x, y, w, h, score) boxes, and
blur_region(img, box) blurs one box in place.
"""
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


DEVICES = ("cpu", "gpu", "npu")
PROVIDER = {"npu": "QNNExecutionProvider", "gpu": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}


def use_npu(model="face_det_lite", size=None, precision=None, device="npu", cache_dir=QNN_CACHE):
    """Run face detection through onnxruntime on `device`:
    "npu" (Qualcomm Hexagon, onnxruntime-qnn), "gpu" (NVIDIA, onnxruntime-gpu)
    or "cpu".

    model: "face_det_lite" (default; Qualcomm Lightweight-Face-Detection),
    "yolo11n_640" / "yolo11n_1280" (YOLO11n-face) or "yunet" (fixed-size
    quantized YuNet from tools/make_yunet_qnn.py; size (w, h): hexagon.SIZES).
    precision: "fp32" or "a16w8" ("a8w8" too for yunet); None is the
    model's default, fp32 for face_det_lite and yolo11n, a16w8 for yunet.
    Returns a message naming the provider in use, or a "WARNING: ..." message
    (and no change) when the device is missing or does not start.
    """
    global _npu
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    try:
        import hexagon
    except ImportError:
        return "WARNING: onnxruntime is not installed"
    if device == "npu":
        if not hexagon.qnn_available():
            return "WARNING: device 'npu' needs a Qualcomm device and onnxruntime-qnn (pip install onnxruntime-qnn)"
        providers = None  # QNN, then the CPU for nodes QNN does not take
    elif device == "gpu":
        providers = hexagon.cuda_providers()
        if providers is None:
            return ("WARNING: device 'gpu' needs an NVIDIA GPU and onnxruntime-gpu "
                    "(pip install onnxruntime-gpu[cuda,cudnn])")
    else:
        providers = ["CPUExecutionProvider"]
    try:
        if model in hexagon.YOLO_SIZES:
            det = hexagon.QnnYolo11Face(model, precision or "fp32", providers=providers, cache_dir=cache_dir)
        elif model == "yunet":
            det = hexagon.QnnYuNet(size or hexagon.DEFAULT_SIZE, precision or hexagon.DEFAULT_PRECISION,
                                   providers=providers, cache_dir=cache_dir)
        else:
            det = hexagon.QnnFaceDetLite(precision or "fp32", providers=providers, cache_dir=cache_dir)
    except Exception as e:  # missing model, a backend library that fails to load, ...
        return f"WARNING: the detector could not start on {device} ({e})"
    if det.provider != PROVIDER[device]:
        return f"WARNING: {PROVIDER[device]} did not start (providers {det.providers})"
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


def use_model(name, device="cpu", precision=None):
    """Select the face detector and where it runs: "cpu" (default), "gpu"
    (NVIDIA) or "npu" (Qualcomm), and which build: precision "fp32"
    (default) or "a16w8" (not for yunet). A device that is missing
    exits with a message. Returns a message describing the detector."""
    global _npu, _selected
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {name!r}")
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    _npu = None
    _selected = True
    if name == "yunet":
        if device in ("gpu", "npu"):
            sys.exit(f"WARNING: yunet runs on the CPU (OpenCV), not on the {device}; "
                     "for the NPU build see tools/blur_live.py --npu yunet")
        if precision:
            sys.exit(f"WARNING: yunet runs the float model in OpenCV, no {precision} build; "
                     "for the quantized NPU build see tools/blur_live.py --npu yunet")
        return "Face detector: YuNet (OpenCV, CPU)"
    msg = use_npu(name, precision=precision, device=device)
    if msg.startswith("WARNING"):
        sys.exit(msg)
    return msg.replace("Face detection:", "Face detector:")


_threshold = None  # set by use_threshold(); None = the model's own threshold


def use_threshold(t=None):
    """Set the confidence a detection needs to count as a face (None = the
    model's default: yolo11n 0.25, face_det_lite 0.55, yunet 0.7)."""
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
    return _detect(img, min_size, thresh)


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
