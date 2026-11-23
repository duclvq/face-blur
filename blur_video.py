#!/usr/bin/env python3
"""Detect faces in a video (or every video of a folder) and blur them.

Usage:
    python blur_video.py INPUT OUTPUT [--model raccoon_s|raccoon_l|yunet]
                         [--method gaussian|pixelate] [--strength N]
                         [--analysis-fps N] [--detect-every N] [--hold N]

INPUT is one video file or a folder of videos. For a folder, its structure is
mirrored into the OUTPUT folder. For a single file, OUTPUT is the output file
(if it has a video extension) or a folder to put it in.

Face detection and blurring come from blur_faces.py, so both scripts behave the
same. Output keeps the input's file extension and, as far as possible, its
format: the same video codec, profile, pixel format, colour tags and frame rate.
If ffmpeg is on PATH it re-encodes the video in the source codec and copies the
original audio, subtitles and metadata; without it, OpenCV writes a video-only
file, trying the source codec first.
"""
import argparse
import csv
import functools
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

import cv2

from blur_faces import (COVERAGE_WARN, DEFAULT_MODEL, MODELS, RACCOON_MODELS, YUNET_MODEL,
                        blur_region, use_model,
                        blurred_fraction, detect_faces, gpu_message, use_gpu)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Video-only fallback when ffmpeg is missing: OpenCV codecs to try, per container.
OPENCV_FOURCCS = {".avi": ["XVID", "MJPG"], ".webm": ["VP80"]}
DEFAULT_FOURCCS = ["avc1", "mp4v"]


# --------------------------------------------------------------------------- boxes

def _same_face(a, b):
    """True if two (x, y, w, h) boxes most likely cover the same face."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union > 0 and inter / union > 0.3:
        return True
    # A face that moved or changed size a lot: centre of one inside the other.
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
    return (ax <= bcx <= ax + aw and ay <= bcy <= ay + ah) or \
           (bx <= acx <= bx + bw and by <= acy <= by + bh)


def update_tracks(tracks, detected, frame_idx, hold):
    """Merge a new detection into the list of [box, last_seen_frame] tracks.

    Fresh boxes replace the old box of the same face. A face the detector
    missed keeps its last box for up to `hold` frames, so a single missed
    detection does not flash an unblurred face.
    """
    kept = [t for t in tracks
            if frame_idx - t[1] <= hold and not any(_same_face(t[0], d) for d in detected)]
    return [[d, frame_idx] for d in detected] + kept


# --------------------------------------------------------------------------- privateframe

def analyze_privateframe(src, opts):
    """Find face boxes for every frame with InsightFace PrivateFrame.

    PrivateFrame detects faces with SCRFD, tracks them across frames, fills
    gaps (optical flow, forward and backward) and smooths the boxes. Only its
    analysis is used; blurring and encoding stay ours, so the output keeps the
    source format. Returns (meta, {frame_idx: [(x, y, w, h), ...]}).
    """
    try:
        from insightface.app.privateframe import analyze_streaming_pipeline
        from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH
    except ImportError:
        sys.exit(f"--model {opts.model} needs InsightFace: pip install -r requirements.txt\n"
                 "(or use --model yunet, which needs only OpenCV)")
    state = {"stage": None, "t": 0.0}
    tty = sys.stdout.isatty()

    def progress(cur, total, stage):
        now = time.time()
        if stage != state["stage"] or (tty and now - state["t"] > 0.5):
            state.update(stage=stage, t=now)
            pct = f" {100 * cur // total}%" if total else ""
            end = "" if tty else "\n"
            print(f"\r  analyzing ({stage}){pct}      ", end=end, flush=True)

    with tempfile.TemporaryDirectory(prefix="faceblur_") as tmp:
        result = Path(tmp) / "result.json"
        analyze_streaming_pipeline(
            config_path=DEFAULT_CONFIG_PATH, input_path=str(src),
            workdir=str(Path(tmp) / "work"), result_path=str(result),
            config_overrides={"models.name": opts.model,
                              "scan.max_analysis_fps": opts.analysis_fps,
                              "runtime.provider": "CPUExecutionProvider" if opts.cpu else "auto"},
            progress=progress)
        doc = json.loads(result.read_text())
    print("\r" + " " * 60 + "\r", end="")
    meta = doc["source_video"]["metadata"]
    boxes = {}
    for ob in doc["observations"]:
        x0, y0, x1, y1 = ob["box"]
        b = (int(x0), int(y0), int(round(x1 - x0)), int(round(y1 - y0)))
        if min(b[2], b[3]) >= opts.min_size:
            boxes.setdefault(ob["frame_idx"], []).append(b)
    return meta, boxes


# --------------------------------------------------------------------------- writers

# Source video codec -> (ffmpeg encoders to try, quality options).
# Quality is set high so the blur pass adds as little loss as possible.
ENCODERS = {
    "h264":       (["libx264"], ["-preset", "fast", "-crf", "18"]),
    "hevc":       (["libx265"], ["-preset", "fast", "-crf", "20", "-x265-params", "log-level=error"]),
    "vp9":        (["libvpx-vp9"], ["-crf", "32", "-b:v", "0", "-deadline", "realtime",
                                    "-cpu-used", "8", "-row-mt", "1"]),
    "vp8":        (["libvpx"], ["-crf", "10", "-deadline", "realtime", "-cpu-used", "8"]),
    "av1":        (["libsvtav1", "libaom-av1"], ["-crf", "30", "-preset", "8"]),
    "mpeg4":      (["mpeg4"], ["-q:v", "2"]),
    "mpeg2video": (["mpeg2video"], ["-q:v", "2"]),
    "mjpeg":      (["mjpeg"], ["-q:v", "2"]),
    "prores":     (["prores_ks"], []),
}
FALLBACK_CODEC = "h264"

# With --gpu: NVIDIA hardware encoders (NVENC) for the codecs they cover.
NVENC_ENCODERS = {
    "h264": (["h264_nvenc"], ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
    "hevc": (["hevc_nvenc"], ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0"]),
    "av1":  (["av1_nvenc"], ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "30", "-b:v", "0"]),
}
# Same pixel layout under the name a hardware encoder uses.
PIX_FMT_ALIASES = {"yuv420p10le": "p010le", "yuv444p10le": "yuv444p16le", "yuv420p": "nv12"}


def probe_video(src):
    """Return the first video stream's format details from ffprobe, or {}."""
    if not shutil.which("ffprobe"):
        return {}
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,codec_tag_string,profile,pix_fmt,bit_rate,color_range,"
         "color_space,color_primaries,color_transfer:format=bit_rate",
         "-of", "json", str(src)], capture_output=True, text=True)
    try:
        data = json.loads(out.stdout)
        info = data["streams"][0]
    except (ValueError, KeyError, IndexError):
        return {}
    info.setdefault("bit_rate", data.get("format", {}).get("bit_rate"))
    return info


@functools.lru_cache(maxsize=None)
def ffmpeg_encoders():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return {line.split()[1] for line in out.stdout.splitlines()
            if len(line.split()) > 1 and line.startswith(" V")}


@functools.lru_cache(maxsize=None)
def encoder_pix_fmts(encoder):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-h", f"encoder={encoder}"],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "pixel formats:" in line:
            return line.split(":", 1)[1].split()
    return []


@functools.lru_cache(maxsize=None)
def nvenc_works(args):
    """True if ffmpeg can open this NVENC encoder with these options. Being
    listed is not enough: it needs an NVIDIA GPU, a driver, and a GPU
    generation that supports the profile and pixel format (e.g. 10-bit H.264)."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                          "-i", "color=black:s=256x256:d=0.1", *args, "-f", "null", "-"],
                         capture_output=True)
    return out.returncode == 0


# ffprobe profile name -> encoder profile name, so e.g. a Main-profile H.264
# source does not come out as High (encoders pick their own default otherwise).
H264_PROFILES = {"constrained baseline": "baseline", "baseline": "baseline", "main": "main",
                 "high": "high", "high 10": "high10", "high 4:2:2": "high422",
                 "high 4:4:4 predictive": "high444"}
NVENC_H264_PROFILES = {"baseline": "baseline", "main": "main", "high": "high",
                       "high10": "high10", "high444": "high444p"}
NVENC_HEVC_PROFILES = {"main": "main", "main 10": "main10", "rext": "rext"}


def encoder_profile(encoder, src_profile):
    p = src_profile.lower()
    if encoder == "libx264":
        return H264_PROFILES.get(p)
    if encoder == "h264_nvenc":
        return NVENC_H264_PROFILES.get(H264_PROFILES.get(p, ""))
    if encoder == "hevc_nvenc":
        return NVENC_HEVC_PROFILES.get(p)
    return None


def _encoder_args(encoder, opts, codec, info, keep):
    """ffmpeg options for one encoder. Returns (args, pix_fmt_kept)."""
    args = ["-c:v", encoder, *opts]
    if codec == "vp8":  # libvpx needs a bitrate ceiling alongside -crf
        args += ["-b:v", str(info.get("bit_rate") or "8M")]
    if codec == "prores" and info.get("profile"):
        # "HQ", "4444 XQ", ... -> prores_ks profile names
        args += ["-profile:v", info["profile"].lower().replace(" ", "")]
    profile = encoder_profile(encoder, info.get("profile") or "")
    if keep and profile:
        args += ["-profile:v", profile]
    # Same pixel format (chroma subsampling / bit depth) when the encoder allows it.
    fmts = encoder_pix_fmts(encoder)
    pix_fmt = info.get("pix_fmt")
    if keep and pix_fmt in fmts:
        return args + ["-pix_fmt", pix_fmt], True
    if keep and PIX_FMT_ALIASES.get(pix_fmt) in fmts:
        return args + ["-pix_fmt", PIX_FMT_ALIASES[pix_fmt]], True
    if fmts:
        args += ["-pix_fmt", "yuv420p" if "yuv420p" in fmts else fmts[0]]
    return args, not fmts


def choose_encoder(info, gpu=False):
    """Pick ffmpeg video options that reproduce the source format.

    Returns (codec_args, codec_name, note); note is "" when the source codec
    is kept, or says what was used instead. With gpu=True, NVENC is used when
    it can produce the same codec, profile and pixel format; otherwise the
    CPU encoder.
    """
    src_codec = info.get("codec_name") or ""
    available = ffmpeg_encoders()
    codec = src_codec if src_codec in ENCODERS else None
    encoder, opts = None, []
    if codec:
        encoder = next((e for e in ENCODERS[codec][0] if e in available), None)
        opts = list(ENCODERS[codec][1])
    elif src_codec in available:  # a codec with a same-named encoder, default settings
        codec, encoder = src_codec, src_codec
    note = ""
    if encoder is None:
        note = (f"source codec {src_codec or 'unknown'} cannot be encoded by this ffmpeg, "
                f"used {FALLBACK_CODEC}")
        codec = FALLBACK_CODEC
        encoder = ENCODERS[codec][0][0]
        opts = list(ENCODERS[codec][1])

    args, _ = _encoder_args(encoder, opts, codec, info, keep=not note)
    if gpu and not note and codec in NVENC_ENCODERS:
        hw = next((e for e in NVENC_ENCODERS[codec][0] if e in available), None)
        if hw:
            hw_args, kept = _encoder_args(hw, list(NVENC_ENCODERS[codec][1]), codec, info, keep=True)
            if kept and nvenc_works(tuple(hw_args)):
                args = hw_args

    # Same codec tag (hvc1 vs hev1, XVID vs FMP4, ...) so players treat it the same.
    tag = info.get("codec_tag_string") or ""
    if not note and tag.isalnum() and len(tag) == 4:
        args += ["-tag:v", tag]

    # Same colour tags; the pixel values are converted with the same matrix
    # in both directions, so copying the tags keeps colours as they were.
    for key, flag in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                      ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc")):
        val = info.get(key)
        if val and val != "unknown":
            args += [flag, val]
    return args, codec, note


class FfmpegWriter:
    """Pipe raw frames to ffmpeg, which encodes them in the source video's format
    and copies the source audio, subtitles and metadata."""

    def __init__(self, dst, src, size, fps, info, gpu=False):
        w, h = size
        rate = Fraction(fps).limit_denominator(1001)
        vcodec, self.codec, self.note = choose_encoder(info, gpu)
        self.encoder = vcodec[1]
        pix_fmt = vcodec[vcodec.index("-pix_fmt") + 1] if "-pix_fmt" in vcodec else "yuv420p"
        # Chroma-subsampled formats need even sizes; pad odd ones by one pixel.
        subsampled = any(k in pix_fmt for k in ("420", "422", "411", "nv1", "nv2"))
        even = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] if subsampled and (w % 2 or h % 2) else []
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-framerate", f"{rate.numerator}/{rate.denominator}", "-i", "-",
               "-i", str(src),
               "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
               "-c:a", "copy", "-c:s", "copy", "-map_metadata", "1",
               *vcodec, *even, str(dst)]
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=self._err)

    def _error(self):
        self._err.seek(0)
        msg = self._err.read().decode(errors="replace").strip()
        return msg.splitlines()[-1] if msg else "ffmpeg failed"

    def write(self, frame):
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self._proc.wait()
            raise RuntimeError(self._error())

    def close(self):
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(self._error())

    def abort(self):
        self._proc.kill()
        self._proc.wait()


class OpenCVWriter:
    """Video-only writer used when ffmpeg is not installed."""

    def __init__(self, dst, size, fps, src_fourcc=""):
        # Try the source's own codec first, then common ones for the container.
        codes = OPENCV_FOURCCS.get(dst.suffix.lower(), DEFAULT_FOURCCS)
        if len(src_fourcc) == 4 and src_fourcc.isalnum():
            codes = [src_fourcc] + [c for c in codes if c != src_fourcc]
        for code in codes:
            self._w = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*code), fps, size)
            if self._w.isOpened():
                self.codec = code
                self.note = "" if code == src_fourcc else f"source codec {src_fourcc or 'unknown'} not writable by OpenCV, used {code}"
                return
        raise RuntimeError(f"OpenCV cannot write {dst.suffix} files")

    def write(self, frame):
        self._w.write(frame)

    def close(self):
        self._w.release()

    abort = close


def count_audio_streams(src):
    """Number of audio streams in src, or None if ffprobe is unavailable."""
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
                         capture_output=True, text=True)
    return len(out.stdout.split()) if out.returncode == 0 else None


# --------------------------------------------------------------------------- pipeline
#
# Three stages run at the same time on consecutive frames: a reader thread
# decodes, the main thread detects faces (on the GPU with --gpu), and a writer
# thread blurs and encodes. OpenCV, onnxruntime and the ffmpeg pipe all release
# the GIL, so the frame rate is set by the slowest stage instead of the sum.

_END = object()
QUEUE_FRAMES = 8


def _put(q, item, stop):
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _get(q, stop):
    while not stop.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            pass
    return _END


def _read_frames(cap, first, q, stop, errors):
    try:
        frame = first
        while frame is not None and _put(q, frame, stop):
            ok, frame = cap.read()
            if not ok:
                frame = None
    except Exception as e:  # pragma: no cover - decoder failure
        errors.append(e)
    finally:
        _put(q, _END, stop)


def _blur_and_write(q, writer, opts, stats, stop, errors):
    try:
        while True:
            item = _get(q, stop)
            if item is _END:
                return
            frame, boxes = item
            if boxes:
                cov = blurred_fraction(frame.shape, boxes, opts.pad)
                stats["cov_sum"] += cov
                stats["cov_max"] = max(stats["cov_max"], cov)
                stats["cov_over"] += cov > COVERAGE_WARN
                for box in boxes:
                    blur_region(frame, box, opts.method, opts.strength, opts.pad, opts.shape)
                stats["blurred"] += 1
            writer.write(frame)
            stats["written"] += 1
    except Exception as e:
        errors.append(e)
        stop.set()


def run_pipeline(cap, first, writer, opts, expected, t0, known_boxes=None):
    """Blur every frame of cap into writer. Returns the stats dict.

    known_boxes -- {frame_idx: boxes} from analyze_privateframe; when given,
    no detection runs here.
    """
    stats = dict(frames=0, written=0, blurred=0, max_faces=0,
                 cov_sum=0.0, cov_max=0.0, cov_over=0)
    stop, errors = threading.Event(), []
    q_read, q_write = queue.Queue(QUEUE_FRAMES), queue.Queue(QUEUE_FRAMES)
    reader = threading.Thread(target=_read_frames, args=(cap, first, q_read, stop, errors), daemon=True)
    blurrer = threading.Thread(target=_blur_and_write,
                               args=(q_write, writer, opts, stats, stop, errors), daemon=True)
    reader.start()
    blurrer.start()
    tracks, n, last_print, tty = [], 0, 0.0, sys.stdout.isatty()
    try:
        while True:
            frame = _get(q_read, stop)
            if frame is _END:
                break
            if known_boxes is not None:
                boxes = known_boxes.get(n, [])
                stats["max_faces"] = max(stats["max_faces"], len(boxes))
            else:
                if n % opts.detect_every == 0:
                    faces = detect_faces(frame, opts.min_size)
                    stats["max_faces"] = max(stats["max_faces"], len(faces))
                    tracks = update_tracks(tracks, faces, n, opts.hold)
                boxes = [box for box, _ in tracks]
            if not _put(q_write, (frame, boxes), stop):
                break
            n += 1
            if tty and time.time() - last_print > 0.5:
                last_print = time.time()
                pct = f" ({100 * n // expected}%)" if expected > 0 else ""
                print(f"\r  frame {n}{'/' + str(expected) if expected > 0 else ''}{pct}"
                      f"  {n / (last_print - t0):.1f} fps   ", end="", flush=True)
        _put(q_write, _END, stop)
        blurrer.join()
    finally:
        stop.set()
        reader.join()
        blurrer.join()
    if errors:
        raise errors[0]
    stats["frames"] = n
    return stats


# --------------------------------------------------------------------------- video

def process_video(src, dst, opts, use_ffmpeg, label):
    """Blur one video. Returns a report row (dict)."""
    row = {"file": label, "status": "error", "frames": 0, "fps": "", "width": "",
           "height": "", "frames_blurred": 0, "max_faces": 0, "max_blurred_pct": 0.0,
           "mean_blurred_pct": 0.0, "video_codec": "", "encoder": "", "audio": "", "note": ""}

    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        cap.release()
        row["note"] = "unreadable"
        print(f"  SKIP (unreadable) {src}")
        return row

    # Use the decoded frame's size: OpenCV applies rotation metadata, so a
    # portrait phone video comes out portrait.
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        fps, row["note"] = 25.0, "fps unknown, used 25"
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    row.update(fps=round(fps, 3), width=w, height=h)
    print(f"  {w}x{h} @ {fps:.3f} fps" + (f", ~{expected} frames" if expected > 0 else ""))

    t_start = time.time()
    known_boxes = None
    if opts.model in RACCOON_MODELS:
        try:
            meta, known_boxes = analyze_privateframe(src, opts)
        except (Exception, KeyboardInterrupt) as e:
            cap.release()
            if isinstance(e, KeyboardInterrupt):
                raise
            row["note"] = f"analysis failed: {e}"
            print(f"\n  ERROR: analysis failed: {e}")
            return row
        if (meta.get("width"), meta.get("height")) != (w, h):
            # E.g. rotation metadata applied by one decoder and not the other:
            # the boxes would not line up, so detect frame by frame instead.
            msg = (f"PrivateFrame saw {meta.get('width')}x{meta.get('height')}, OpenCV {w}x{h}; "
                   f"detected frame by frame instead (no tracking)")
            print(f"  WARNING: {msg}")
            row["note"] = "; ".join(filter(None, [row["note"], msg]))
            import blur_faces
            if blur_faces._raccoon is None:
                blur_faces.use_model(opts.model, opts.cpu)
            known_boxes = None
        else:
            print(f"  analysis: {sum(map(len, known_boxes.values()))} face boxes in "
                  f"{len(known_boxes)} frames, {time.time() - t_start:.1f} s")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name so an interrupted run never leaves a
    # half-written file that looks finished.
    tmp = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
    writer = None
    try:
        if use_ffmpeg:
            info = probe_video(src)
            writer = FfmpegWriter(tmp, src, (w, h), fps, info, not opts.cpu)
            src_codec = info.get("codec_name", "?")
        else:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            src_codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")
            writer = OpenCVWriter(tmp, (w, h), fps, src_codec)
        row["video_codec"] = src_codec if not writer.note else f"{src_codec} -> {writer.codec}"
        row["encoder"] = getattr(writer, "encoder", writer.codec)
        if writer.note:
            row["note"] = "; ".join(filter(None, [row["note"], writer.note]))
            print(f"  WARNING: {writer.note}")
        t0 = time.time()
        st = run_pipeline(cap, frame, writer, opts, expected, t0, known_boxes)
        writer.close()
        writer = None
        tmp.replace(dst)
    except (Exception, KeyboardInterrupt) as e:
        if writer is not None:
            writer.abort()
        tmp.unlink(missing_ok=True)
        if isinstance(e, KeyboardInterrupt):
            raise
        row["note"] = str(e)
        print(f"\n  ERROR: {e}")
        return row
    finally:
        cap.release()

    if use_ffmpeg:
        audio = count_audio_streams(src)
        row["audio"] = {None: "copied if present", 0: "none"}.get(audio, "copied")
    else:
        row["audio"] = "dropped (no ffmpeg)"
    n, blurred, max_faces = st["frames"], st["blurred"], st["max_faces"]
    cov_max, cov_over = st["cov_max"], st["cov_over"]
    row.update(status="ok", frames=n, frames_blurred=blurred, max_faces=max_faces,
               max_blurred_pct=round(cov_max * 100, 1),
               mean_blurred_pct=round(st["cov_sum"] / max(n, 1) * 100, 1))
    secs = time.time() - t_start
    print(f"\r  done: {n} frames, faces blurred in {blurred}, max {max_faces} per frame, "
          f"codec {row['video_codec']} ({row['encoder']}), audio {row['audio']}, {secs:.1f} s ({n / max(secs, 1e-9):.1f} fps)")
    if cov_over:
        print(f"  WARNING: more than {COVERAGE_WARN:.0%} of the picture blurred in {cov_over} "
              f"frame(s) (max {cov_max:.0%})")
    return row


# --------------------------------------------------------------------------- jobs

def plan_jobs(inp, out):
    """Return (jobs, report_path); jobs is a list of (src, dst, label).

    A folder run writes OUTPUT/blur_report.csv. A single video writes
    NAME_report.csv next to the output video, so nothing lands in a folder
    the user did not name as output (e.g. the input's own folder).
    """
    inp, out = Path(inp).resolve(), Path(out).resolve()
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    if inp.is_file():
        if inp.suffix.lower() not in VIDEO_EXTS:
            sys.exit(f"Not a supported video ({', '.join(sorted(VIDEO_EXTS))}): {inp}")
        dst = out if out.suffix.lower() in VIDEO_EXTS else out / inp.name
        if dst == inp:
            sys.exit("OUTPUT must not be the input file (that would overwrite the original).")
        return [(inp, dst, inp.name)], dst.with_name(f"{dst.stem}_report.csv")

    # Never write over the original videos.
    if out == inp or inp in out.parents:
        sys.exit("OUTPUT must not be INPUT or a folder inside it "
                 "(that would overwrite or mix with the original videos).")
    files = sorted(p for p in inp.rglob("*")
                   if p.suffix.lower() in VIDEO_EXTS and p.is_file()
                   and not p.name.startswith("."))  # skips macOS "._" metadata files
    if not files:
        sys.exit(f"No videos found in {inp}")
    jobs = []
    for src in files:
        rel = src.relative_to(inp)
        dst = out / rel
        if inp in dst.parents:  # e.g. INPUT is itself inside OUTPUT
            sys.exit(f"Refusing to write {dst}: it is inside INPUT.")
        jobs.append((src, dst, rel.as_posix()))
    return jobs, out / "blur_report.csv"


def main():
    ap = argparse.ArgumentParser(description="Detect and blur faces in videos.")
    ap.add_argument("input", help="a video file or a folder of videos")
    ap.add_argument("output", help="output folder (or output file for a single video)")
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=int, choices=[1, 2, 3], default=3,
                    help="1 (light) .. 3 (heavy)")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                    help="face detector: raccoon_s (default; InsightFace PrivateFrame with "
                         "tracking), raccoon_l (larger), or yunet (small and fast, for "
                         "low-end machines)")
    ap.add_argument("--analysis-fps", type=float, default=15, metavar="N",
                    help="raccoon models: full face scans per second of video (default 15; "
                         "use 30 for fast motion, or the video's frame rate to scan every frame)")
    ap.add_argument("--min-size", type=int, default=None,
                    help="smallest face size in px (default: 24 for yunet, no limit otherwise)")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="margin around each face as a fraction of its size, 0..1 (0 = tight)")
    ap.add_argument("--shape", choices=["rect", "oval"], default="oval",
                    help="blur the whole box or an oval that follows the face")
    ap.add_argument("--detect-every", type=int, default=1, metavar="N",
                    help="yunet: run face detection every N frames and reuse the boxes in "
                         "between (faster, but fast-moving faces can slip out)")
    ap.add_argument("--hold", type=int, default=10, metavar="N",
                    help="yunet: keep blurring a face for N frames after the detector "
                         "loses it, to avoid flicker (0 = off)")
    ap.add_argument("--cpu", action="store_true",
                    help="do not use a GPU (by default face detection and, for H.264/HEVC/AV1, "
                         "encoding use an NVIDIA GPU or Apple GPU when available)")
    ap.add_argument("--gpu", action="store_true", help=argparse.SUPPRESS)  # now the default
    args = ap.parse_args()
    if not 0 <= args.pad <= 1:
        ap.error(f"--pad is a fraction of the face size between 0 and 1, not pixels "
                 f"(got {args.pad}; 0.15 adds a 15% margin). Large values blur the whole frame.")
    if args.detect_every < 1 or args.hold < 0 or args.analysis_fps <= 0:
        ap.error("--detect-every must be >= 1, --hold >= 0 and --analysis-fps > 0")
    if args.min_size is None:
        args.min_size = 24 if args.model == "yunet" else 0
    if not YUNET_MODEL.exists():
        sys.exit(f"Model not found: {YUNET_MODEL}")

    jobs, report_path = plan_jobs(args.input, args.output)
    if args.model == "yunet":
        print("Face detector: YuNet (OpenCV)")
        if not args.cpu:
            print(gpu_message(use_gpu()))
    elif args.model not in RACCOON_MODELS:
        print(use_model(args.model, args.cpu))
    else:
        dev = "CPU only" if args.cpu else "GPU when available, else CPU"
        print(f"Face detector: {args.model} (InsightFace PrivateFrame: SCRFD + tracking; {dev})")
    use_ffmpeg = shutil.which("ffmpeg") is not None
    if not use_ffmpeg:
        print("WARNING: ffmpeg not found on PATH. Output videos will have NO AUDIO.\n"
              "         Install ffmpeg to keep the original sound.\n")

    rows = []
    for i, (src, dst, label) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {label}")
        rows.append(process_video(src, dst, args, use_ffmpeg, label))

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except OSError as e:
        print(f"WARNING: could not write blur_report.csv: {e}")

    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nDone: {ok}/{len(rows)} videos, {sum(r['frames'] for r in rows)} frames -> "
          f"{report_path.parent}")
    if ok < len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
