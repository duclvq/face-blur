"""Small Python API for face-blur, for use in your own app.

    from faceblur import FaceBlurrer

    fb = FaceBlurrer()                         # raccoon_s, gaussian blur
    out = fb.blur(frame)                       # numpy BGR image -> blurred copy
    fb.blur_file("in.jpg", "out.jpg")          # image file
    fb.blur_video("in.mp4", "out.mp4")         # video file (keeps sound with ffmpeg)

    live = fb.stream(fps=30)                   # live frames (webcam, RTSP, ...)
    for frame in frames:
        for out in live.push(frame):           # comes out `delay` seconds later
            show(out)
    for out in live.flush():                   # the last buffered frames
        show(out)

See examples/quickstart.py for a runnable guide.

The detector is chosen per process (blur_faces keeps it in module state), so
use one model per process. Images are BGR numpy arrays, as from cv2.imread or
cv2.VideoCapture.
"""
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

import blur_faces
from blur_faces import DEFAULT_MODEL, MODELS, blur_region, detect_faces

__all__ = ["FaceBlurrer", "StreamBlurrer", "MODELS", "DEFAULT_MODEL"]


class FaceBlurrer:
    """Detect and blur faces.

    model    -- "raccoon_s" (default, accurate, finds small faces), "raccoon_l"
                (larger) -- InsightFace, non-commercial research only --
                "yunet" (small and fast, MIT license, needs only OpenCV) or
                "face_det_lite" (Qualcomm Lightweight-Face-Detection, BSD-3;
                the default on Qualcomm hardware, runs on the Hexagon NPU)
    method   -- "gaussian" or "pixelate"
    strength -- 1 (light) .. 3 (heavy)
    pad      -- margin around each face, as a fraction of its size (0..1)
    shape    -- "oval" (follows the face) or "rect" (whole box, safer)
    min_size -- skip faces smaller than this many pixels
                (default: 24 for yunet, no limit for raccoon)
    gpu      -- use an NVIDIA GPU when available (falls back to the CPU)
    """

    def __init__(self, model=DEFAULT_MODEL, method="gaussian", strength=3, pad=0.15,
                 shape="oval", min_size=None, gpu=True):
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
        self.pad, self.shape, self.gpu = pad, shape, gpu
        self.min_size = (24 if model == "yunet" else 0) if min_size is None else min_size

        try:
            self.device = blur_faces.use_model(model, cpu=not gpu)
        except SystemExit as e:  # blur_faces exits when InsightFace is missing
            raise RuntimeError(str(e)) from None
        if model == "yunet" and gpu:
            # Each image size needs a few seconds of GPU set-up, so a size only
            # moves to the GPU after GPU_AFTER_IMAGES images of it (video frames
            # all share one size; a mix of photo sizes stays fast on the CPU).
            self.device += "; " + blur_faces.gpu_message(
                blur_faces.use_gpu(gpu_after=blur_faces.GPU_AFTER_IMAGES))

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

    # ---------------------------------------------------------------- video

    def blur_video(self, src, dst, detect_every=1, hold=10, analysis_fps=15):
        """Blur a video file into dst, like blur_video.py.

        Keeps the codec, sound and subtitles when ffmpeg is on PATH (video only
        without it). Returns a dict with frame and face counts; raises
        RuntimeError if the video could not be processed.

        detect_every, hold -- yunet: detect every N frames, keep a lost face
                              blurred for N frames
        analysis_fps       -- raccoon: full face scans per second of video
        """
        import blur_video  # imported here: it is only needed for video files
        src, dst = Path(src), Path(dst)
        if src.resolve() == dst.resolve():
            raise ValueError("dst must not be src (the original would be overwritten)")
        opts = SimpleNamespace(model=self.model, method=self.method, strength=self.strength,
                               pad=self.pad, shape=self.shape, min_size=self.min_size,
                               cpu=not self.gpu, detect_every=detect_every, hold=hold,
                               analysis_fps=analysis_fps)
        row = blur_video.process_video(src, dst, opts, shutil.which("ffmpeg") is not None,
                                       src.name)
        if row["status"] != "ok":
            raise RuntimeError(f"could not blur {src}: {row['note']}")
        return row

    # ---------------------------------------------------------------- live

    def stream(self, fps, delay=0.5, detect_every=1, hold=1.0, smooth=2, extend=False,
               low_score=0.1):
        """A StreamBlurrer for live frames; see that class."""
        return StreamBlurrer(self, fps, delay, detect_every, hold, smooth, extend, low_score)


class StreamBlurrer:
    """Blur a live sequence of frames (webcam, RTSP, ...) like blur_stream.py.

    Faces are tracked with optical flow between detector runs; low-score
    detections (low_score .. the normal threshold, 0 = off) keep an existing
    face tracked, as in ByteTrack. Frames come out
    `delay` seconds after they go in, so the frames between two detections of
    a face can be filled from both ends. A face is blurred from its first to
    its last detection only (frame-accurate); extend=True also blurs it before
    its first detection (within `delay`) and up to `hold` seconds after its
    last one, which is safer for privacy. If analysing a frame fails, that
    frame is blurred whole.

    push(frame) returns the frames that are ready (often none at first, then
    one per frame); flush() returns the rest at the end. Frames are blurred in
    place and returned.
    """

    def __init__(self, blurrer, fps, delay=0.5, detect_every=1, hold=1.0, smooth=2,
                 extend=False, low_score=0.1):
        from blur_stream import Analyzer  # imported here: only needed for streams
        if fps <= 0 or delay < 0 or detect_every < 1 or hold <= 0:
            raise ValueError("need fps > 0, delay >= 0, detect_every >= 1, hold > 0")
        self.fb = blurrer
        opts = SimpleNamespace(delay=delay, hold=hold, detect_every=detect_every,
                               min_size=blurrer.min_size, smooth=smooth if delay else 0,
                               extend=extend, low_score=low_score)
        self.analyzer = Analyzer(opts, fps)
        self.i = 0

    def push(self, frame):
        """Add the next frame. Returns a list of blurred frames ready to show."""
        out = self._finish(self.analyzer.push(self.i, frame, time.time()))
        self.i += 1
        return out

    def flush(self):
        """Return the frames still waiting in the delay buffer, blurred."""
        return self._finish(self.analyzer.flush())

    def _finish(self, items):
        frames = []
        for e, boxes in items:
            frame = e["frame"]
            e["gray"] = None
            if e["fail"]:
                H, W = frame.shape[:2]
                blur_region(frame, (0, 0, W, H), "pixelate", 3, 0, "rect")
            frames.append(self.fb.blur_boxes(frame, boxes, inplace=True))
        return frames
