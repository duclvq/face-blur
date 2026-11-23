"""Face detection on a Qualcomm Hexagon NPU (onnxruntime QNN EP, HTP backend).

Three detectors:
  * QnnFaceDetLite: Qualcomm's Lightweight-Face-Detection (face_det_lite, BSD-3,
    float model from Qualcomm AI Hub, models/face_det_lite/). The default on
    Qualcomm hardware.
  * QnnYolo11Face: YOLO11n-face at 640x384 or 1280x736 (models/yolo11n_face/,
    AGPL-3.0 weights). Finds far more small faces than face_det_lite.
  * QnnYuNet: YuNet with fixed input sizes, quantized by tools/make_yunet_qnn.py.

The Hexagon HTP runs graphs with fixed input shapes, so the frame is
letterboxed (scaled to fit, zero-padded at the bottom/right) into one of the
fixed-size models built by tools/make_yunet_qnn.py, and the boxes are scaled
back to the frame. Decode and NMS are the same as detector.OrtYuNet
(OpenCV FaceDetectorYN).

    from hexagon import QnnYuNet
    det = QnnYuNet(size=(960, 544), precision="a16w8")
    print(det.providers)            # QNNExecutionProvider first, else a CPU fallback
    faces = det.detect(frame_bgr, min_size=8, thresh=0.7)   # [(x, y, w, h, score)]

See docs/hexagon.md. Only the CPU EP path has been run so far; nothing here
has been tested on Hexagon hardware yet.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
YUNET_DYNAMIC = MODELS_DIR / "face_detection_yunet_2023mar_dynamic.onnx"
SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K = 0.7, 0.3, 5000  # same as detector
STRIDES = (8, 16, 32)

# Fixed input sizes (width, height) built by tools/make_yunet_qnn.py. All are
# multiples of 32 with room for a 16:9 frame (1920x1080 -> 640x360 / 1280x720
# + padding). 960x544 is the default for live 1080p (finds about as many
# faces >= 32 px as 1280x736 with half the pixels); see docs/hexagon.md.
SIZES = ((640, 384), (960, 544), (1280, 736))
DEFAULT_SIZE = (960, 544)
# a8w8: uint8 activations / int8 per-channel weights (any HTP).
# a16w8: uint16 activations / int8 per-channel weights (newer HTPs; check the SoC).
# fp32: the float model; HTP runs it in fp16 (enable_htp_fp16_precision), HTPs with fp16 only.
PRECISIONS = ("a8w8", "a16w8", "fp32")
DEFAULT_PRECISION = "a16w8"


def model_path(size=DEFAULT_SIZE, precision=DEFAULT_PRECISION):
    w, h = size
    return MODELS_DIR / f"face_detection_yunet_2023mar_qnn_{w}x{h}_{precision}.onnx"


def letterbox(img, size):
    """Scale `img` to fit in `size` (w, h) keeping its aspect ratio and pad
    with zeros at the bottom/right. Returns (float32 NCHW blob, scale)."""
    W, H = size
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = min(W, round(w * scale)), min(H, round(h * scale))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    blob = np.zeros((1, 3, H, W), np.float32)
    blob[0, :, :nh, :nw] = img.transpose(2, 0, 1)
    return blob, scale


def decode(out, size, thresh=SCORE_THRESHOLD):
    """YuNet outputs -> (boxes Nx4 [x, y, w, h] in model pixels, scores), after NMS.
    Same maths as detector.OrtYuNet / OpenCV FaceDetectorYN."""
    W, _ = size
    boxes, scores = [], []
    for s in STRIDES:
        cols = W // s
        cls = np.clip(out[f"cls_{s}"][0, :, 0], 0, 1)
        obj = np.clip(out[f"obj_{s}"][0, :, 0], 0, 1)
        score = np.sqrt(cls * obj)
        keep = np.nonzero(score >= thresh)[0]
        if not keep.size:
            continue
        bb = out[f"bbox_{s}"][0, keep].astype(np.float32)
        c, r = keep % cols, keep // cols
        cx, cy = (c + bb[:, 0]) * s, (r + bb[:, 1]) * s
        bw, bh = np.exp(bb[:, 2]) * s, np.exp(bb[:, 3]) * s
        boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1))
        scores.append(score[keep])
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    idx = cv2.dnn.NMSBoxes(boxes.astype(np.int32).tolist(), scores.tolist(),
                           thresh, NMS_THRESHOLD, top_k=TOP_K)
    idx = np.asarray(idx, int).reshape(-1)
    return boxes[idx], scores[idx]


def qnn_backend_path():
    """HTP backend library name: QnnHtp.dll on Windows, libQnnHtp.so elsewhere.
    Set QNN_BACKEND_PATH to point at a specific file (e.g. from the QNN SDK)."""
    return os.environ.get("QNN_BACKEND_PATH") or ("QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so")


def qnn_provider_options(precision=DEFAULT_PRECISION):
    opts = {
        "backend_path": qnn_backend_path(),
        "htp_performance_mode": "burst",           # or "sustained_high_performance" for long runs
        "htp_graph_finalization_optimization_mode": "3",  # slower first compile, faster graph
    }
    if precision == "fp32":
        opts["enable_htp_fp16_precision"] = "1"    # run the float model in fp16 on HTP
    return opts


def open_session(path, providers, cache_dir=None, strict=False, log_level=3):
    """onnxruntime session for `path`, with the QNN context-binary cache.

    cache_dir: the first QNN run compiles the graph for the HTP (seconds) and
    writes <name>_ctx.onnx (+ a .bin) there; later runs load that instead. The
    cache is tied to the SoC and QNN SDK version; a stale one is recompiled.
    """
    import onnxruntime as ort
    use_qnn = any((p[0] if isinstance(p, tuple) else p) == "QNNExecutionProvider" for p in providers)

    def options():
        opts = ort.SessionOptions()
        opts.log_severity_level = log_level  # 0-1 lists the nodes QNN takes / leaves to the CPU
        if strict:
            opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return opts

    opts, load = options(), Path(path)
    if use_qnn and cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ctx = Path(cache_dir) / f"{Path(path).stem}_ctx.onnx"
        if ctx.exists():
            load = ctx
        else:
            opts.add_session_config_entry("ep.context_enable", "1")
            opts.add_session_config_entry("ep.context_file_path", str(ctx))
            opts.add_session_config_entry("ep.context_embed_mode", "0")
    try:
        return ort.InferenceSession(str(load), opts, providers=providers)
    except Exception:
        if load == Path(path):
            raise
        return ort.InferenceSession(str(path), options(), providers=providers)


class QnnYuNet:
    """YuNet on the Hexagon NPU through onnxruntime's QNN execution provider.

    size       -- fixed model input (w, h), one of SIZES
    precision  -- "a8w8", "a16w8" or "fp32" (see PRECISIONS)
    providers  -- override, e.g. ["CPUExecutionProvider"] to run the same QDQ
                  model on the CPU (for testing on a machine without QNN)
    cache_dir  -- where the compiled QNN context binary is kept ("ep.context"),
                  so later starts skip the slow graph compilation. None: off.
    strict     -- refuse to start if any node would fall back to the CPU EP
                  (session.disable_cpu_ep_fallback); useful when validating.

    `self.providers` lists the providers in use: if QNN did not load, the
    first entry is CPUExecutionProvider (a silent fallback otherwise).
    """

    def __init__(self, size=DEFAULT_SIZE, precision=DEFAULT_PRECISION, providers=None,
                 cache_dir=None, strict=False, log_level=3):
        self.size = tuple(size)
        self.precision = precision
        path = model_path(self.size, precision)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python tools/make_yunet_qnn.py")
        if providers is None:
            providers = [("QNNExecutionProvider", qnn_provider_options(precision)), "CPUExecutionProvider"]
        self.sess = open_session(path, providers, cache_dir, strict, log_level)
        self.providers = self.sess.get_providers()
        self.provider = self.providers[0]
        self.on_npu = self.provider == "QNNExecutionProvider"
        self.input = self.sess.get_inputs()[0].name
        self.names = [o.name for o in self.sess.get_outputs()]
        self.threshold = SCORE_THRESHOLD
        self.name = f"YuNet {path.name}"

    def detect(self, img, min_size=24, thresh=SCORE_THRESHOLD):
        """(x, y, w, h, score) face boxes in `img` pixels with score >= thresh."""
        blob, scale = letterbox(img, self.size)
        out = dict(zip(self.names, self.sess.run(None, {self.input: blob})))
        boxes, scores = decode(out, self.size, thresh)
        H, W = img.shape[:2]
        faces = []
        for (x, y, w, h), sc in zip(boxes / scale, scores):
            x0, y0 = max(int(x), 0), max(int(y), 0)
            x1, y1 = min(int(x + w), W), min(int(y + h), H)
            if x1 > x0 and y1 > y0 and min(int(w), int(h)) >= min_size:
                faces.append((x0, y0, x1 - x0, y1 - y0, float(sc)))
        return faces



# --------------------------------------------------------------------------- face_det_lite

LITE_DIR = MODELS_DIR / "face_det_lite"
LITE_MODEL = LITE_DIR / "face_det_lite.onnx"   # float; weights in face_det_lite.data
LITE_DLC = LITE_DIR / "face_det_lite.dlc"      # same model for QAIRT/QNN apps (not used here)
LITE_SIZE = (640, 480)                          # fixed model input (w, h), grayscale
LITE_STRIDE = 8
LITE_THRESHOLD = 0.55   # Qualcomm's app and evaluator
LITE_NMS = 0.2          # Qualcomm's evaluator
# skimage.color.rgb2gray weights (what Qualcomm's app uses), in BGR order
LITE_GRAY = np.array([0.0721, 0.7154, 0.2125], np.float32) / 255.0


class QnnFaceDetLite:
    """Qualcomm Lightweight-Face-Detection (face_det_lite) through onnxruntime.

    Float model from Qualcomm AI Hub (v0.67.0, the most accurate precision).
    On the HTP it runs in fp16 (enable_htp_fp16_precision), as AI Hub runs it.
    Pre/post-processing follows qai_hub_models/models/face_det_lite (app.py,
    utils.detect): grayscale in [0, 1], 640x480 input; a CenterNet-style
    heatmap at stride 8 whose 3x3 local maxima are faces; box offsets
    (left, top, right, bottom) in stride units.

    providers -- None: QNN (HTP) then CPU; or e.g. ["CPUExecutionProvider"]
    Other arguments as QnnYuNet.
    """

    def __init__(self, providers=None, cache_dir=None, strict=False, log_level=3):
        if not LITE_MODEL.exists():
            raise FileNotFoundError(f"{LITE_MODEL} missing")
        if providers is None:
            providers = [("QNNExecutionProvider", qnn_provider_options("fp32")), "CPUExecutionProvider"]
        self.size = LITE_SIZE
        self.sess = open_session(LITE_MODEL, providers, cache_dir, strict, log_level)
        self.providers = self.sess.get_providers()
        self.provider = self.providers[0]
        self.on_npu = self.provider == "QNNExecutionProvider"
        self.input = self.sess.get_inputs()[0].name
        self.names = [o.name for o in self.sess.get_outputs()]
        self.threshold = LITE_THRESHOLD
        self.name = "Lightweight-Face-Detection (face_det_lite, float)"

    def preprocess(self, img):
        """BGR uint8 frame -> (1x1x480x640 float32 blob in [0, 1], scale)."""
        W, H = self.size
        h, w = img.shape[:2]
        scale = min(W / w, H / h)
        nw, nh = min(W, round(w * scale)), min(H, round(h * scale))
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        blob = np.zeros((1, 1, H, W), np.float32)
        blob[0, 0, :nh, :nw] = img.astype(np.float32) @ LITE_GRAY if img.ndim == 3 else img / 255.0
        return blob, scale

    @staticmethod
    def decode(hm, box, thresh=LITE_THRESHOLD, stride=LITE_STRIDE):
        """Model outputs -> (boxes Nx4 x0, y0, x1, y1 in model pixels, scores), as utils.detect."""
        hm = 1.0 / (1.0 + np.exp(-hm[0, 0]))                      # sigmoid
        peak = hm == cv2.dilate(hm, np.ones((3, 3), np.uint8))    # = max_pool2d(3, 1, 1)
        ys, xs = np.nonzero(peak & (hm >= thresh))
        scores = hm[ys, xs]
        order = np.argsort(-scores)[:2000]
        ys, xs, scores = ys[order], xs[order], scores[order]
        l, t, r, b = box[0, :, ys, xs].T
        boxes = np.stack([xs - l, ys - t, xs + r, ys + b], 1) * stride
        return boxes.astype(np.float32), scores.astype(np.float32)

    def detect(self, img, min_size=0, thresh=LITE_THRESHOLD):
        """(x, y, w, h, score) face boxes in `img` pixels with score >= thresh."""
        blob, scale = self.preprocess(img)
        out = dict(zip(self.names, self.sess.run(None, {self.input: blob})))
        boxes, scores = self.decode(out["heatmap"], out["bbox"], thresh)
        if len(boxes) > 1:
            xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
            idx = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), thresh, LITE_NMS),
                             int).reshape(-1)
            boxes, scores = boxes[idx], scores[idx]
        H, W = img.shape[:2]
        faces = []
        for (x0, y0, x1, y1), sc in zip(boxes / scale, scores):
            x0, y0 = max(int(x0), 0), max(int(y0), 0)
            x1, y1 = min(int(round(x1)), W), min(int(round(y1)), H)
            if x1 > x0 and y1 > y0 and min(x1 - x0, y1 - y0) >= min_size:
                faces.append((x0, y0, x1 - x0, y1 - y0, float(sc)))
        return faces


# --------------------------------------------------------------------------- YOLO11n-face

YOLO_DIR = MODELS_DIR / "yolo11n_face"
YOLO_SIZES = {"yolo11n_640": (640, 384), "yolo11n_1280": (1280, 736)}  # fixed model input (w, h)
YOLO_THRESHOLD = 0.3   # no false faces seen at 0.3 on the test clips
YOLO_NMS = 0.45


class QnnYolo11Face:
    """YOLO11n-face (zjykzj/YOLO11Face, YOLO11n-pose trained on WIDER FACE;
    AGPL-3.0 weights, see models/yolo11n_face/README.md) through onnxruntime.

    Float model; on the HTP it runs in fp16. The frame is letterboxed (RGB,
    pad 114, / 255) into the fixed input; output (1, 20, N) is cx, cy, w, h,
    score and 5 keypoints, decoded with NMS.

    name -- "yolo11n_640" (640x384) or "yolo11n_1280" (1280x736)
    Other arguments as QnnFaceDetLite.
    """

    def __init__(self, name="yolo11n_640", providers=None, cache_dir=None, strict=False, log_level=3):
        self.size = YOLO_SIZES[name]
        path = YOLO_DIR / f"yolo11n_face_{self.size[0]}x{self.size[1]}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing")
        if providers is None:
            providers = [("QNNExecutionProvider", qnn_provider_options("fp32")), "CPUExecutionProvider"]
        self.sess = open_session(path, providers, cache_dir, strict, log_level)
        self.providers = self.sess.get_providers()
        self.provider = self.providers[0]
        self.on_npu = self.provider == "QNNExecutionProvider"
        self.input = self.sess.get_inputs()[0].name
        self.threshold = YOLO_THRESHOLD
        self.name = f"YOLO11n-face {self.size[0]}x{self.size[1]} (float)"

    def preprocess(self, img):
        """BGR uint8 frame -> (1x3xHxW float32 RGB blob in [0, 1], scale)."""
        W, H = self.size
        h, w = img.shape[:2]
        scale = min(W / w, H / h)
        nw, nh = min(W, round(w * scale)), min(H, round(h * scale))
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad = np.full((H, W, 3), 114, np.uint8)
        pad[:nh, :nw] = img
        return pad[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0, scale

    def detect(self, img, min_size=0, thresh=YOLO_THRESHOLD):
        """(x, y, w, h, score) face boxes in `img` pixels with score >= thresh."""
        blob, scale = self.preprocess(img)
        out = self.sess.run(None, {self.input: blob})[0][0]   # (20, N)
        keep = out[4] >= thresh
        if not keep.any():
            return []
        cx, cy, bw, bh = out[:4, keep] / scale
        scores = out[4, keep]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
        idx = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), thresh, YOLO_NMS),
                         int).reshape(-1)
        H, W = img.shape[:2]
        faces = []
        for (x, y, w, h), sc in zip(boxes[idx], scores[idx]):
            x0, y0 = max(int(x), 0), max(int(y), 0)
            x1, y1 = min(int(round(x + w)), W), min(int(round(y + h)), H)
            if x1 > x0 and y1 > y0 and min(x1 - x0, y1 - y0) >= min_size:
                faces.append((x0, y0, x1 - x0, y1 - y0, float(sc)))
        return faces


def qnn_available():
    """True when onnxruntime has the QNN execution provider (onnxruntime-qnn)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "QNNExecutionProvider" in ort.get_available_providers()


def cuda_providers():
    """Providers for an NVIDIA GPU (onnxruntime-gpu), or None without one."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return None
    if hasattr(ort, "preload_dlls"):  # CUDA and cuDNN DLLs from the nvidia-* pip packages
        ort.preload_dlls()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
