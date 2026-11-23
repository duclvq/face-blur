"""Small Python API for face-blur, for use in your own app.

    from faceblur import FaceBlurrer

    fb = FaceBlurrer()                         # yolo11n_640, gaussian blur
    out = fb.blur(frame)                       # numpy BGR image -> blurred copy
    fb.blur_file("in.jpg", "out.jpg")          # image file

    live = fb.stream(fps=30)                   # live frames (RTSP, webcam, ...)
    for frame in frames:
        res = live.push(frame)                 # no delay by default
        if res is None:                        # only with delay=: buffer still filling
            continue
        res.original, res.blurred, res.boxes   # input, blurred copy, [(x, y, w, h)]
        res.scores                             # confidence of each face
    rest = live.flush()                        # with delay=: the frames still held, at the end

See quickstart.py for a full live pipeline (RTSP, hardware decode, this API,
hardware encode, live view). The library itself does not decode or encode video.

The detector is chosen per process (detector keeps it in module state), so
use one model per process. Images are BGR numpy arrays, as from cv2.imread or
cv2.VideoCapture.
"""
from pathlib import Path
from types import SimpleNamespace

import cv2

import detector
from detector import DEFAULT_MODEL, MODELS, blur_region, detect_faces

__all__ = ["FaceBlurrer", "StreamBlurrer", "FrameResult", "MODELS", "DEFAULT_MODEL"]


class FaceBlurrer:
    """Detect and blur faces.

    model    -- "yolo11n_640" (default; YOLO11n-face, AGPL-3.0 weights),
                "yolo11n_1280" (finds small faces, about 4x slower),
                "face_det_lite" (Qualcomm Lightweight-Face-Detection, BSD-3;
                misses small faces) or "yunet" (simple scenes with large faces).
                All but yunet run on the Hexagon NPU with onnxruntime-qnn
    method   -- "gaussian" or "pixelate"
    strength -- 1 (light) .. 3 (heavy)
    pad      -- margin around each face, as a fraction of its size (0..1)
    shape    -- "oval" (follows the face) or "rect" (whole box, safer)
    min_size -- skip faces smaller than this many pixels
                (default: 24 for yunet, no limit otherwise)
    threshold -- confidence a detection needs to count as a face, 0..1
                (default: the model's own, yolo11n 0.3, face_det_lite 0.55,
                yunet 0.7). Lower finds more faces and more false ones
    tiles    -- also detect on an N x N grid of overlapping tiles (1 = off):
                finds smaller faces, costs 1 + N*N detector runs per frame
    gpu      -- use the Qualcomm NPU when there is one (falls back to the CPU)
    mask     -- True (default): stream results carry a blurred copy.
                False: no blurring at all; res.blurred is None and you mask
                the faces yourself from res.boxes (method, strength, pad and
                shape are then unused). On a failed frame (res.failed) no
                fallback blur is made either: handle it yourself
    """

    def __init__(self, model=DEFAULT_MODEL, method="gaussian", strength=3, pad=0.15,
                 shape="oval", min_size=None, gpu=True, tiles=1, threshold=None, mask=True):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, not {model!r}")
        if method not in ("gaussian", "pixelate"):
            raise ValueError("method must be 'gaussian' or 'pixelate'")
        if strength not in (1, 2, 3):
            raise ValueError("strength must be 1, 2 or 3")
        if not 0 <= pad <= 1:
            raise ValueError("pad is a fraction of the face size, between 0 and 1")
        if shape not in ("oval", "rect"):
            raise ValueError("shape must be 'oval' or 'rect'")
        self.model, self.method, self.strength = model, method, strength
        self.pad, self.shape, self.gpu, self.mask = pad, shape, gpu, mask
        self.min_size = (24 if model == "yunet" else 0) if min_size is None else min_size

        try:
            self.device = detector.use_model(model, cpu=not gpu)
            detector.use_tiles(tiles)
            detector.use_threshold(threshold)
        except SystemExit as e:  # detector exits when a model cannot start
            raise RuntimeError(str(e)) from None

    # ---------------------------------------------------------------- images

    def detect(self, img):
        """Face boxes in a BGR image, as a list of (x, y, w, h)."""
        return detect_faces(img, self.min_size)

    def blur_boxes(self, img, boxes, inplace=False):
        """Blur the given (x, y, w, h) boxes. Returns the blurred image."""
        out = img if inplace else img.copy()
        for box in boxes:
            blur_region(out, box, self.method, self.strength, self.pad, self.shape)
        return out

    def blur(self, img, inplace=False):
        """Detect and blur every face in a BGR image. Returns the blurred image."""
        return self.blur_boxes(img, self.detect(img), inplace)

    def blur_file(self, src, dst):
        """Blur an image file into dst. Returns the number of faces blurred."""
        src, dst = Path(src), Path(dst)
        if src.resolve() == dst.resolve():
            raise ValueError("dst must not be src (the original would be overwritten)")
        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"cannot read image: {src}")
        faces = self.detect(img)
        self.blur_boxes(img, faces, inplace=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), img):
            raise OSError(f"cannot write image: {dst}")
        return len(faces)

    # ---------------------------------------------------------------- live

    def stream(self, fps, detect_every=1, hold=0.3, low_score=0.3, delay=0):
        """A StreamBlurrer for live frames; see that class."""
        return StreamBlurrer(self, fps, detect_every, hold, low_score, delay)


class StreamBlurrer:
    """Blur a live sequence of frames (webcam, RTSP, video).

    Faces are tracked with ByteTrack (Kalman filter + two-stage matching:
    low-score detections, low_score .. the normal threshold, keep an existing
    face tracked; 0 = off). Between detector runs (detect_every) and for a
    face the detector just lost, the box follows optical flow; a lost face
    stays blurred for `hold` seconds (0 = only faces the detector found).

    push(frame) returns a FrameResult: the original frame, the blurred frame
    and the face boxes. The input frame is not changed.

    delay (frames, default 0 = none; e.g. 12, about 0.5 s at 25 fps) holds
    that many frames so that later detections fix earlier frames: a new face
    is followed back through the frames the detector missed it in, and a face
    found again is filled in the frames between. push() then returns the
    FrameResult of the frame `delay` frames earlier (None while the buffer
    fills); flush() returns the rest at the end of the stream. About 0.5 s
    of frames halves the faces left unblurred.
    """

    def __init__(self, blurrer, fps, detect_every=1, hold=0.3, low_score=0.3, delay=0):
        from tracker import Analyzer, Lookahead  # imported here: only needed for streams
        if fps <= 0 or detect_every < 1 or hold < 0 or delay < 0 or int(delay) != delay:
            raise ValueError("need fps > 0, detect_every >= 1, hold >= 0, delay a whole number of frames >= 0")
        self.fb = blurrer
        opts = SimpleNamespace(hold=hold, detect_every=detect_every,
                               min_size=blurrer.min_size, low_score=low_score)
        self.analyzer = Analyzer(opts, fps)
        self.lookahead = Lookahead(self.analyzer, int(delay)) if delay > 0 else None

    def push(self, frame):
        """Analyse the next frame. Returns its FrameResult; with a delay, the
        FrameResult of the frame `delay` frames earlier (None at first)."""
        if self.lookahead is None:
            return self._result(frame, self.analyzer.push(frame))
        out = self.lookahead.push(frame)
        return self._result(*out[0]) if out else None

    def flush(self):
        """With a delay: the FrameResults still held, at the end of the stream."""
        return [self._result(*e) for e in self.lookahead.flush()] if self.lookahead else []

    def _result(self, frame, faces):
        return FrameResult(frame, faces or [], faces is None, self.fb)


class FrameResult:
    """One analysed frame.

    original -- the input frame, unchanged
    boxes    -- the faces to blur, as (x, y, w, h) in pixels
    scores   -- detector confidence of each face (at its last detection; a
                face followed by tracking keeps the score it was found with)
    failed   -- True if analysing the frame failed (then `blurred` is the
                whole frame pixelated, so nothing is shown unblurred)
    blurred  -- a blurred copy of the frame (made on first use);
                None with FaceBlurrer(mask=False)
    """

    def __init__(self, frame, faces, failed, blurrer):
        self.original, self.failed = frame, failed
        self.boxes = [f[:4] for f in faces]
        self.scores = [f[4] for f in faces]
        self._fb, self._blurred = blurrer, None

    @property
    def blurred(self):
        if not self._fb.mask:
            return None
        if self._blurred is None:
            out = self.original.copy()
            if self.failed:  # fail closed
                H, W = out.shape[:2]
                blur_region(out, (0, 0, W, H), "pixelate", 3, 0, "rect")
            self._blurred = self._fb.blur_boxes(out, self.boxes, inplace=True)
        return self._blurred
