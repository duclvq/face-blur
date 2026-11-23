#!/usr/bin/env python3
"""GStreamer element that blurs faces, with a look-back delay.

    faceblurdelay  (video/x-raw, format=BGR in and out)
      delay         output delay in seconds (default 0.5)
      model         raccoon_s (default; face_det_lite on Qualcomm hardware),
                    raccoon_l, yunet or face_det_lite
      detect-every  run the face detector every N frames (default 1: every frame)
      extend        also blur before the first / after the last detection
                    of a face (default false: frame-accurate, gaps between
                    detections are filled)

Frames are held for `delay` seconds. A face found in a new frame is tracked
back through the held frames, so it is blurred from its first frame. Buffers
keep their timestamps (the stream is shifted by `delay`), the element adds
`delay` to the pipeline latency, and held frames are sent on at end of stream.
Detection and tracking are the same as blur_stream.py (via faceblur.py).

Use it from Python (register, then use it in any pipeline):

    import gst_faceblur
    gst_faceblur.register()
    Gst.parse_launch("... ! videoconvert n-threads=0 ! video/x-raw,format=BGR ! "
                     "faceblurdelay delay=0.5 ! videoconvert n-threads=0 ! ...")

Or run this file:

    python gst_faceblur.py INPUT [OUTPUT.mp4] [--show] [--delay 0.5] [--model yunet]

INPUT is a video file, a URI (rtsp://, http://, file:///...), or a webcam
index (0). A file plays at its own speed with --show (like a live source) and
as fast as possible without it.

Setup (GStreamer + Python bindings from conda-forge, in their own env):

    conda create -n faceblur-gst -c conda-forge python=3.12 pygobject gstreamer \\
        gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav \\
        gst-python numpy opencv onnxruntime scipy scikit-image onnx pyyaml tqdm requests
    conda run -n faceblur-gst pip install --no-deps insightface==2.1
    conda run -n faceblur-gst pip install onnxruntime-gpu      # NVIDIA GPU only
    conda run -n faceblur-gst --no-capture-output python gst_faceblur.py VIDEO --show

On Windows, run through `conda run` (or an activated env): otherwise the
GStreamer DLLs are not found.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, GObject, Gst, GstVideo  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faceblur import DEFAULT_MODEL, MODELS, FaceBlurrer  # noqa: E402

Gst.init(None)  # needed before any caps or templates are made; safe to call again
BGR = Gst.Caps.from_string("video/x-raw,format=BGR")


class FaceBlurDelay(Gst.Element):
    __gstmetadata__ = ("Face blur with look-back delay", "Filter/Effect/Video",
                       "Detects, tracks and blurs faces; output is delayed by `delay` seconds",
                       "face-blur")
    __gsttemplates__ = (
        Gst.PadTemplate.new("sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, BGR),
        Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, BGR),
    )
    __gproperties__ = {
        "delay": (float, "Delay", "Output delay in seconds", 0.0, 10.0, 0.5,
                  GObject.ParamFlags.READWRITE),
        "model": (str, "Model", "raccoon_s, raccoon_l, yunet or face_det_lite", DEFAULT_MODEL,
                  GObject.ParamFlags.READWRITE),
        "detect-every": (int, "Detect every", "Run the detector every N frames", 1, 1000, 1,
                         GObject.ParamFlags.READWRITE),
        "extend": (bool, "Extend", "Also blur before the first and after the last detection",
                   False, GObject.ParamFlags.READWRITE),
    }

    def __init__(self):
        super().__init__()
        self.props_ = {"delay": 0.5, "model": DEFAULT_MODEL, "detect-every": 1, "extend": False}
        self.sinkpad = Gst.Pad.new_from_template(self.get_pad_template("sink"), "sink")
        self.srcpad = Gst.Pad.new_from_template(self.get_pad_template("src"), "src")
        self.sinkpad.set_chain_function_full(self._chain)
        self.sinkpad.set_event_function_full(self._sink_event)
        self.srcpad.set_query_function_full(self._src_query)
        self.add_pad(self.sinkpad)
        self.add_pad(self.srcpad)
        self.fb = None          # FaceBlurrer, created on first caps (model load is slow)
        self.stream = None      # StreamBlurrer for the current caps
        self.info = None        # GstVideo.VideoInfo of the current caps
        self.held = deque()     # input buffers waiting, for their timestamps
        self.stats = {"in": 0, "out": 0}

    # ---------------------------------------------------------------- properties

    def do_get_property(self, prop):
        return self.props_[prop.name]

    def do_set_property(self, prop, value):
        self.props_[prop.name] = value

    # ---------------------------------------------------------------- data

    def _chain(self, pad, parent, buf):
        if self.stream is None:
            return Gst.FlowReturn.NOT_NEGOTIATED
        frame = self._to_array(buf)
        self.held.append(buf)
        self.stats["in"] += 1
        return self._send(self.stream.push(frame))

    def _send(self, frames):
        ret = Gst.FlowReturn.OK
        for frame in frames:
            src = self.held.popleft()  # frames come out in order
            out = Gst.Buffer.new_wrapped(self._to_bytes(frame))
            out.pts, out.dts, out.duration = src.pts, src.dts, src.duration
            out.offset, out.offset_end = src.offset, src.offset_end
            self.stats["out"] += 1
            ret = self.srcpad.push(out)
            if ret != Gst.FlowReturn.OK:
                break
        return ret

    def _to_array(self, buf):
        w, h, stride = self.info.width, self.info.height, self.info.stride[0]
        ok, m = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("cannot map buffer")
        try:
            rows = np.frombuffer(m.data, np.uint8, count=h * stride).reshape(h, stride)
            return rows[:, :w * 3].reshape(h, w, 3).copy()
        finally:
            buf.unmap(m)

    def _to_bytes(self, frame):
        h, w = frame.shape[:2]
        stride = self.info.stride[0]
        if stride == w * 3:
            return frame.tobytes()
        rows = np.zeros((h, stride), np.uint8)  # GStreamer pads BGR rows to 4 bytes
        rows[:, :w * 3] = frame.reshape(h, w * 3)
        return rows.tobytes()

    # ---------------------------------------------------------------- events

    def _sink_event(self, pad, parent, event):
        t = event.type
        if t == Gst.EventType.CAPS:
            if self.stream:  # caps change: send what was held under the old caps
                self._send(self.stream.flush())
            self.info = GstVideo.VideoInfo.new_from_caps(event.parse_caps())
            fps = self.info.fps_n / self.info.fps_d if self.info.fps_n and self.info.fps_d else 25.0
            if self.fb is None:
                self.fb = FaceBlurrer(model=self.props_["model"], pad=0.2)
                Gst.info(f"faceblurdelay: {self.fb.device}")
            self.stream = self.fb.stream(fps=fps, delay=self.props_["delay"],
                                         detect_every=self.props_["detect-every"],
                                         extend=self.props_["extend"])
        elif t == Gst.EventType.EOS:
            if self.stream:
                self._send(self.stream.flush())  # held frames go out before EOS
        elif t == Gst.EventType.FLUSH_STOP:
            self.held.clear()                    # seek: drop what was held
            if self.info:
                fps = self.info.fps_n / self.info.fps_d if self.info.fps_n else 25.0
                self.stream = self.fb.stream(fps=fps, delay=self.props_["delay"],
                                             detect_every=self.props_["detect-every"],
                                         extend=self.props_["extend"])
        return self.srcpad.push_event(event)

    def _src_query(self, pad, parent, query):
        if query.type == Gst.QueryType.LATENCY:
            # Tell the sinks about the delay, so delayed frames are not late.
            if not self.sinkpad.peer_query(query):
                return False
            live, lo, hi = query.parse_latency()
            extra = int(self.props_["delay"] * Gst.SECOND)
            query.set_latency(live, lo + extra, hi + extra if hi != Gst.CLOCK_TIME_NONE else hi)
            return True
        return Gst.Pad.query_default(pad, parent, query)


GObject.type_register(FaceBlurDelay)


def register():
    """Make `faceblurdelay` available to Gst.parse_launch / ElementFactory."""
    return Gst.Element.register(None, "faceblurdelay", Gst.Rank.NONE, FaceBlurDelay)


# --------------------------------------------------------------------------- demo app

def build_pipeline(src, output, show, delay, model, detect_every):
    if src.isdigit():  # webcam (Windows Media Foundation; v4l2src on Linux)
        source = f"mfvideosrc device-index={src}" if sys.platform == "win32" \
            else f"v4l2src device=/dev/video{src}"
    else:
        uri = src if "://" in src else Path(src).resolve().as_uri()
        source = f'uridecodebin uri="{uri}" caps="video/x-raw(ANY)"'
    sync = "true" if show else "false"  # a file plays in real time only when shown
    p = (f"{source} ! queue ! videoconvert n-threads=0 ! video/x-raw,format=BGR ! "
         f"faceblurdelay name=blur delay={delay} model={model} detect-every={detect_every} ! "
         f"videoconvert n-threads=0 ! tee name=t ")
    if show:
        p += "t. ! queue ! videoconvert n-threads=0 ! autovideosink sync=true "
    if output:
        out = Path(output).resolve().as_posix()
        p += (f't. ! queue ! videoconvert n-threads=0 ! x264enc tune=zerolatency speed-preset=veryfast ! '
              f'h264parse ! mp4mux ! filesink location="{out}" sync={sync} ')
    if not show and not output:
        p += "t. ! queue ! fakesink sync=false "
    return p


def main():
    ap = argparse.ArgumentParser(description="Blur faces with the faceblurdelay GStreamer element.")
    ap.add_argument("input", help="video file, URI (rtsp://, http://), or webcam index (0)")
    ap.add_argument("output", nargs="?", help="output .mp4 file")
    ap.add_argument("--show", action="store_true", help="show the blurred video (file plays in real time)")
    ap.add_argument("--delay", type=float, default=0.5, help="output delay in seconds")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--detect-every", type=int, default=1)
    a = ap.parse_args()
    if not a.output and not a.show:
        ap.error("give an OUTPUT file, or --show")

    Gst.init(None)
    register()
    desc = build_pipeline(a.input, a.output, a.show, a.delay, a.model, a.detect_every)
    print("Pipeline:", desc.replace(" ! ", " !\n    "))
    pipe = Gst.parse_launch(desc)
    blur = pipe.get_by_name("blur")

    def no_qos(_bin, _sub, element):
        # A late display would send QoS upstream and the decoder / videoconvert
        # would drop frames, also from the recording. Show late frames instead.
        if isinstance(element, GstVideo.VideoSink):
            element.set_property("qos", False)
            element.set_property("max-lateness", -1)
    pipe.connect("deep-element-added", no_qos)

    loop, t0, result = GLib.MainLoop(), time.time(), {"ok": True}

    def on_message(bus, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"\nERROR: {err.message}\n{dbg or ''}")
            result["ok"] = False
            loop.quit()
        elif msg.type == Gst.MessageType.LATENCY:
            pipe.recalculate_latency()

    def progress():
        s = blur.stats
        print(f"\r  frames in {s['in']}, out {s['out']} (held {s['in'] - s['out']}), "
              f"{s['out'] / max(time.time() - t0, 1e-9):.1f} fps   ", end="", flush=True)
        return True

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    GLib.timeout_add(1000, progress)
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n  stopping...")
    finally:
        q = Gst.Query.new_latency()
        live, lo, _ = q.parse_latency() if pipe.query(q) else (False, 0, 0)
        pipe.set_state(Gst.State.NULL)
    progress()
    print(f"\n  done in {time.time() - t0:.1f} s; pipeline latency {lo / Gst.SECOND:.2f} s"
          + (f"; detector: {blur.fb.device}" if blur.fb else ""))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
