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
    device   -- where the detector runs: "cpu" (default), "gpu" (NVIDIA,
                needs onnxruntime-gpu) or "npu" (Qualcomm, needs onnxruntime-qnn).
                A device that is missing raises RuntimeError. self.backend
                names the provider in use
    mask     -- True (default): stream results carry a blurred copy.
                False: no blurring at all; res.blurred is None and you mask
                the faces yourself from res.boxes (method, strength, pad and
                shape are then unused). On a failed frame (res.failed) no
                fallback blur is made either: handle it yourself
    """

    def __init__(self, model=DEFAULT_MODEL, method="gaussian", strength=3, pad=0.15,
                 shape="oval", min_size=None, device="cpu", threshold=None, mask=True):
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
        self.pad, self.shape, self.device, self.mask = pad, shape, device, mask
        self.min_size = (24 if model == "yunet" else 0) if min_size is None else min_size

        try:
            self.backend = detector.use_model(model, device)
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

    def stream(self, fps, low_score=0.3, delay=0, max_state_mb=100):
        """A StreamBlurrer for live frames; see that class."""
        return StreamBlurrer(self, fps, low_score, delay, max_state_mb)


class StreamBlurrer:
    """Blur a live sequence of frames (webcam, RTSP, video).

    Faces are tracked with ByteTrack (Kalman filter + two-stage matching:
    low-score detections, low_score .. the normal threshold, keep an existing
    face tracked; 0 = off). The detector runs on every frame. A face is
    blurred in the frames the detector finds it in; with a delay, the frames
    it was missed in are filled when it is found again.

    push(frame) returns a FrameResult: the original frame, the blurred frame
    and the face boxes. The input frame is not changed.

    Two halves, which may run in two threads:

      put(frame)  analyses a frame and adds it to the buffer (returns nothing)
      get()       takes the oldest frame out as a FrameResult, once at least
                  `delay` newer frames are behind it; None until then

    `delay` (frames, default 0; e.g. 12, about 0.5 s at 25 fps) is how many
    newer frames a frame waits for, so later detections fix it (a new face is
    followed back through the frames the detector missed it in; a face found
    again is filled in between). 12 frames left about a third fewer faces
    unblurred on two street clips. Change `live.delay`
    any time; `live.buffered` is the number of frames held, `live.ready` the
    number get() can return now. get(force=True) takes the oldest frame even
    with fewer newer ones behind it (e.g. to shrink the buffer, or when your
    output clock needs a frame now). flush() returns all the rest at the end.

    push(frame) = put(frame) then get(): with delay=0 it returns this frame's
    result at once; with a delay, None until `delay` frames are buffered.

    reset() forgets every tracked face. The next frame is analysed from
    scratch, as after a scene cut, and buffered frames keep their boxes. The
    stream also resets itself when its tracking state passes max_state_mb
    (default 100). live.state_mb gives the state size and live.resets counts
    both kinds of reset. Frames held for `delay` do not count toward
    max_state_mb; live.buffered gives their number.
    """

    def __init__(self, blurrer, fps, low_score=0.3, delay=0, max_state_mb=100):
        from tracker import Analyzer, Lookahead  # imported here: only needed for streams
        if fps <= 0 or max_state_mb <= 0:
            raise ValueError("need fps > 0, max_state_mb > 0")
        self.fb = blurrer
        opts = SimpleNamespace(min_size=blurrer.min_size,
                               low_score=low_score, max_state_mb=max_state_mb)
        self.analyzer = Analyzer(opts, fps)
        self.buffer = Lookahead(self.analyzer)
        self.delay = delay

    @property
    def delay(self):
        """Newer frames a frame waits for before get() returns it."""
        return self._delay

    @delay.setter
    def delay(self, frames):
        if frames < 0 or int(frames) != frames:
            raise ValueError("delay is a whole number of frames >= 0")
        self._delay = int(frames)

    @property
    def buffered(self):
        """Frames held now."""
        return len(self.buffer)

    @property
    def state_mb(self):
        """Memory held by the tracking state, in MB (estimate)."""
        return self.analyzer.state_bytes / 1e6

    @property
    def resets(self):
        """Times the tracking state was cleared: reset() calls and max_state_mb."""
        return self.analyzer.stats["resets"]

    def reset(self):
        """Forget every tracked face. Applied at the next put(); safe to call
        from the output thread."""
        self.analyzer.reset()

    @property
    def ready(self):
        """Frames get() can return now."""
        return max(0, len(self.buffer) - self._delay)

    def put(self, frame):
        """Analyse a frame and add it to the buffer."""
        self.buffer.put(frame)

    def get(self, force=False):
        """The oldest buffered frame as a FrameResult, if `delay` newer frames
        are behind it (or force=True and any frame is held); else None."""
        if not force and len(self.buffer) <= self._delay:
            return None
        e = self.buffer.pop()
        return self._result(*e) if e else None

    def push(self, frame):
        """put(frame), then get()."""
        self.put(frame)
        return self.get()

    def flush(self):
        """All the frames still held, as FrameResults (end of the stream)."""
        return [self._result(*e) for e in self.buffer.flush()]

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
