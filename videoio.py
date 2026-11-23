"""Video input and output for quickstart.py: hardware decode and encode
through ffmpeg processes (VideoSource, VideoSink).

    platform              decode                        encode
    Qualcomm Linux        h264_v4l2m2m / hevc_v4l2m2m   h264_v4l2m2m
    Windows on Snapdragon -hwaccel d3d11va              h264_mf (Media Foundation)
    macOS                 -hwaccel videotoolbox         h264_videotoolbox
    anything else         software                      libx264

The first one that works is used; sw=True forces software. Needs ffmpeg and
ffprobe on PATH. face-blur itself does not use this module.
"""
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]


# --------------------------------------------------------------------------- hardware choice

def probe(src):
    """(width, height, fps, codec) of the first video stream."""
    rtsp = ["-rtsp_transport", "tcp"] if src.startswith("rtsp://") else []
    for _ in range(3):  # a live stream joined between keyframes may not show its size yet
        out = subprocess.run(["ffprobe", "-v", "error", *rtsp, "-analyzeduration", "5000000",
                              "-probesize", "10000000", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,avg_frame_rate,r_frame_rate,codec_name",
                              "-of", "json", src], capture_output=True, text=True, timeout=60)
        s = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
        if s.get("width"):
            break
    else:
        sys.exit(f"cannot open {src}: {out.stderr.strip() or 'no video size'}")
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = s.get(key, "0/0").partition("/")
        if float(den or 0) > 0 and 0 < float(num) / float(den) <= 120:
            fps = float(num) / float(den)
            break
    return s["width"], s["height"], fps or 25.0, s.get("codec_name", "h264")


def ffmpeg_list(kind):
    return subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True, text=True).stdout


def decoder_args(codec, sw):
    """ffmpeg input options for a hardware decoder ([] = software)."""
    if sw:
        return []
    system = platform.system()
    if system == "Darwin":
        return ["-hwaccel", "videotoolbox"]
    if system == "Windows":
        return ["-hwaccel", "d3d11va"]
    if f"{codec}_v4l2m2m" in ffmpeg_list("decoders"):  # Qualcomm (Venus) and other SoCs
        return ["-c:v", f"{codec}_v4l2m2m"]
    return ["-hwaccel", "auto"]


def encoder_works(args):
    out = subprocess.run([*FFMPEG, "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
                          "-pix_fmt", "yuv420p", *args, "-f", "null", "-"], capture_output=True)
    return out.returncode == 0


def encoder_args(fps, sw):
    """ffmpeg options for the first H.264 encoder that works, hardware first."""
    gop = ["-g", str(max(1, round(fps))), "-bf", "0"]  # a keyframe every second, no B-frames
    candidates = [] if sw else [
        ["-c:v", "h264_v4l2m2m", "-b:v", "4M"],
        ["-c:v", "h264_mf", "-hw_encoding", "1", "-scenario", "live_streaming", "-b:v", "4M"],
        ["-c:v", "h264_videotoolbox", "-realtime", "1", "-b:v", "4M"],
    ]
    available = ffmpeg_list("encoders")
    for args in candidates:
        if args[1] in available and encoder_works(args):
            return args + gop
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-b:v", "4M"] + gop


def output_format(url):
    if url.startswith("rtsp://"):
        return ["-f", "rtsp", "-rtsp_transport", "tcp"]
    if url.startswith("rtmp://"):
        return ["-f", "flv"]
    if url.startswith(("srt://", "udp://", "tcp://")):
        return ["-f", "mpegts"]
    return []  # a file: format from its extension


# --------------------------------------------------------------------------- input

class VideoSource:
    """Decoded BGR frames from an RTSP URL (or any ffmpeg input), hardware
    decoded when possible. Iterate over it to get the frames.

    Decoding runs in an ffmpeg process read by a thread, which keeps only the
    newest frames: when the analysis is slower than the camera, old frames are
    dropped (never shown unblurred). A lost stream is reconnected. A video
    file plays at its own speed, like a camera.

    size, fps -- frame size (after `width` scaling) and frame rate
    dropped   -- frames dropped so far
    t         -- capture time (time.time()) of the frame just returned
    """

    def __init__(self, src, width=0, sw=False):
        w, h, fps, codec = probe(src)
        if width and width < w:  # e.g. analyse a 4K camera at 1080p
            w, h = width, round(h * width / w / 2) * 2
        self.src, self.size, self.fps, self.codec, self.sw = src, (w, h), fps, codec, sw
        self.q = queue.Queue(maxsize=2)
        self.dropped = 0
        self.t = None  # capture time of the last frame returned
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _cmd(self, sw):
        w, h = self.size
        if "://" in self.src:  # a live stream: no input buffering
            inp = ["-fflags", "nobuffer", "-flags", "low_delay"]
            if self.src.startswith("rtsp://"):
                inp += ["-rtsp_transport", "tcp"]
        else:
            inp = ["-re"]  # a file: play it at its own speed, like a camera
        # Joining a live stream between keyframes gives decode errors at first: keep going.
        return [*FFMPEG, *inp, *decoder_args(self.codec, sw), "-i", self.src, "-max_error_rate", "1",
                "-an", "-vf", f"scale={w}:{h}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]

    def _run(self):
        w, h = self.size
        nbytes = w * h * 3
        sw, retries = self.sw, 0
        while not self.stop.is_set():
            proc = subprocess.Popen(self._cmd(sw), stdout=subprocess.PIPE, bufsize=nbytes)
            got = 0
            while not self.stop.is_set():
                buf = proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                got += 1
                self._put((np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy(), time.time()))
            proc.kill()
            proc.wait()
            if self.stop.is_set():
                break
            if got == 0 and not sw:
                print("  hardware decode failed, using software decode")
                sw = True
                continue
            if "://" not in self.src:
                break  # end of a file
            retries = 0 if got else retries + 1
            if retries > 30:
                print("  ERROR: stream lost, giving up")
                break
            print("  stream lost, reconnecting...")
            time.sleep(1.0)
        self.q.put(None)

    def _put(self, item):
        while True:
            try:
                self.q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def __iter__(self):
        try:
            while (item := self.q.get()) is not None:
                self.t = item[1]  # capture time of this frame (time.time())
                yield item[0]
        finally:
            self.close()

    @property
    def decoder(self):
        return " ".join(decoder_args(self.codec, self.sw)) or "software"

    def close(self):
        self.stop.set()


# --------------------------------------------------------------------------- output

class VideoSink:
    """BGR frames -> H.264 (hardware encoder when possible) -> an rtsp://,
    srt://, udp://, rtmp:// URL or a video file. Frames are timestamped as
    they arrive, so the output keeps real time when frames were dropped."""

    def __init__(self, url, size, fps, sw=False):
        w, h = size
        self.args = encoder_args(fps, sw)
        self.proc = subprocess.Popen(
            [*FFMPEG, "-y", "-use_wallclock_as_timestamps", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-i", "-",
             "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", *self.args,
             *output_format(url), url], stdin=subprocess.PIPE)

    @property
    def encoder(self):
        return self.args[1]

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait()


# --------------------------------------------------------------------------- overlay

def draw_faces(frame, boxes, scores):
    """Draw each face box with its confidence above it."""
    scale = max(0.5, frame.shape[0] / 1300)
    for (x, y, w, h), s in zip(boxes, scores):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, f"{s:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return frame


class Stats:
    """Frame rate and analysis time, drawn on the frame (top left), with a
    graph of the last `history` frames below it: analysis time per frame
    (green, the dashed line is budget_ms, e.g. 1000 / fps) and face count
    (orange, own scale).

        stats = Stats(budget_ms=1000 / fps)
        t = stats.start()
        ... analyse the frame ...
        stats.draw(frame, t, faces=len(boxes), dropped=source.dropped)
    """

    def __init__(self, smooth=0.9, history=300, budget_ms=None):
        self.smooth = smooth
        self.fps = self.ms = self.dt = 0.0
        self.last = None
        self.budget = budget_ms
        self.hist_ms = deque(maxlen=history)
        self.hist_faces = deque(maxlen=history)

    def start(self):
        return time.perf_counter()

    def draw(self, frame, t_start, faces=0, dropped=0):
        now = time.perf_counter()
        ms = (now - t_start) * 1000
        k = self.smooth if self.ms else 0.0  # exponential averages, so the numbers are readable
        self.ms = k * self.ms + (1 - k) * ms
        if self.last is not None:  # average the frame interval, not 1/interval
            dt = now - self.last
            self.dt = k * self.dt + (1 - k) * dt if self.dt else dt
            self.fps = 1.0 / self.dt if self.dt > 0 else 0.0
        self.last = now
        self.hist_ms.append(ms)
        self.hist_faces.append(faces)
        text = f"{self.fps:.1f} fps | analysis {self.ms:.0f} ms | faces {faces} | dropped {dropped}"
        scale = max(0.5, frame.shape[0] / 1080)
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(frame, (0, 0), (w + 20, h + base + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2,
                    cv2.LINE_AA)
        self._graph(frame, 0, h + base + 20, max(w + 20, int(420 * scale)), int(130 * scale), scale)
        return frame

    def _graph(self, frame, x0, y0, gw, gh, scale):
        H, W = frame.shape[:2]
        gw, gh = min(gw, W - x0), min(gh, H - y0)
        if gw < 40 or gh < 30 or not self.hist_ms:
            return
        roi = frame[y0:y0 + gh, x0:x0 + gw]
        roi[:] = (roi * 0.25).astype(roi.dtype)  # darken, keep a little of the picture
        pad, fs = int(6 * scale), 0.45 * scale
        top, bottom = pad + int(14 * scale), gh - pad
        ms, faces = np.array(self.hist_ms), np.array(self.hist_faces, float)
        ms_top = max(ms.max(), self.budget or 0) * 1.15
        faces_top = max(faces.max(), 1) * 1.15
        n = self.hist_ms.maxlen
        xs = gw - 1 - (len(ms) - 1 - np.arange(len(ms))) * (gw - 1) / max(n - 1, 1)

        def ys(v, vmax):
            return bottom - v / vmax * (bottom - top)

        if self.budget:
            yb = int(ys(self.budget, ms_top))
            for xd in range(0, gw, int(10 * scale) + 2):
                cv2.line(roi, (xd, yb), (min(xd + int(5 * scale) + 1, gw - 1), yb), (150, 150, 150), 1)
        for vals, vmax, color in ((faces, faces_top, (0, 165, 255)), (ms, ms_top, (0, 255, 0))):
            pts = np.stack([xs, ys(vals, vmax)], 1).round().astype(np.int32)
            cv2.polylines(roi, [pts], False, color, max(1, int(scale * 1.5)), cv2.LINE_AA)
        label = f"time {ms[-1]:.0f} ms (max {ms.max():.0f})"
        if self.budget:
            label += f" | budget {self.budget:.1f}"
        cv2.putText(roi, label, (pad, pad + int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 1,
                    cv2.LINE_AA)
        flabel = f"faces {faces[-1]:.0f} (max {faces.max():.0f})"
        (fw, _), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(roi, flabel, (gw - fw - pad, pad + int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 165, 255), 1, cv2.LINE_AA)
