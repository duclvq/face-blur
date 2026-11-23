# face-blur

Blur every face in a live feed from your own Python code, or in a folder of photos, for example to anonymize footage before sharing it. Runs on a CPU or on the NPU of Qualcomm chips.

The four built-in models on the same night street, at the default settings (detector on every frame, tracking, 0.5 s delay). Each panel shows the time per frame on a Mac mini M4 CPU and the number of faces blurred; [15 s clip as MP4](examples/video/models_compare.mp4):

![yolo11n_640, yolo11n_1280, face_det_lite and yunet on the same street video](examples/video/models_compare.gif)

**Speed per frame on the devices tested** (Qualcomm: detector on the NPU, measured with [Qualcomm AI Hub](https://aihub.qualcomm.com), November 2026; Mac: whole pipeline with tracking, CPU):

| Model | Mac mini M4 (CPU) | IQ-9075 | QCS8550 | Snapdragon X Elite | RB3 Gen 2 (QCS6490) |
|---|---|---|---|---|---|
| `yolo11n_640` (default) | 24 ms | 3.7 ms | 2.9 ms | 3.1 ms | 3.0 ms (w8a8) |
| `yolo11n_1280` | 67 ms | 14 ms | 12 ms | 12 ms | 475 ms (CPU) |
| `face_det_lite` | 18 ms | 2.7 ms | 1.9 ms | 2.0 ms | 1.4 ms (w8a8) |
| `yunet` | 26 ms | 0.6 to 4.6 ms | 0.5 to 4.2 ms | 0.4 to 4.0 ms | 15 to 170 ms (CPU) |

Details in [Qualcomm hardware](#qualcomm-hardware).

## Quick start

```bash
pip install -r requirements.txt     # plus ffmpeg on PATH
python quickstart.py --input clip.mp4                   # blur a video, show it in a window
```

From Python, face-blur is the analysis step of a live video pipeline: a decoded BGR frame goes in; the original frame, a blurred copy and the face boxes come out, 0.5 s later by default (the delay lets later frames fix earlier ones).

```
RTSP -> decode -> [ faceblur ] -> encode -> live view
```

```python
from faceblur import FaceBlurrer

fb = FaceBlurrer()                  # yolo11n_640; load the model once, then reuse it
live = fb.stream(fps=25)

for frame in frames:                # BGR numpy arrays (H x W x 3, uint8)
    res = live.push(frame)          # the frame from 0.5 s earlier
    if res is None:                 # the first 0.5 s: buffer filling
        continue
    res.original                    # the input frame, unchanged
    res.blurred                     # a blurred copy
    res.boxes                       # the blurred faces: [(x, y, w, h), ...]
    res.scores                      # the confidence of each face
for res in live.flush():            # end of stream: the frames still held
    ...
```

## Models

| `--model` | When to use it |
|---|---|
| `yolo11n_640` (default) | YOLO11n-face at 640x384. The best balance: finds most faces in crowds, fast on any device. |
| `yolo11n_1280` | YOLO11n-face at 1280x736. **For small faces** (people far away, wide shots of crowds). About 3 to 4x slower than `yolo11n_640`. |
| `face_det_lite` | Qualcomm Lightweight-Face-Detection (BSD-3). Fastest on the NPU, but misses small faces (`tiles=2` helps). |
| `yunet` | OpenCV YuNet (MIT). Only for simple scenes with few, large faces. |

All but `yunet` run on the Hexagon NPU of Qualcomm chips when `onnxruntime-qnn` is installed, and on the CPU elsewhere.

The YOLO11n-face weights come from [zjykzj/YOLO11Face](https://github.com/zjykzj/YOLO11Face) and were trained with Ultralytics, which is **AGPL-3.0**: check the license before using them in a product (see [`models/yolo11n_face`](models/yolo11n_face/README.md)).

## Options

The same options exist on the command line (`quickstart.py`, `tools/`) and in Python (`FaceBlurrer(...)` and `fb.stream(...)`).

### Detection and sampling: accuracy against speed and latency

| Command line | Python | Default | What it does | Trade-off |
|---|---|---|---|---|
| `--model` | `FaceBlurrer(model=)` | `yolo11n_640` | Face detector, see [Models](#models) | `yolo11n_1280`: more small faces, 3 to 4x slower |
| `--threshold T` | `FaceBlurrer(threshold=)` | model's own: yolo11n 0.3, face_det_lite 0.55, yunet 0.7 | Confidence a detection needs to count as a face (0 to 1) | Lower: more faces found, more false ones |
| `--tiles N` | `FaceBlurrer(tiles=)` | `1` (off) | Also detect on an N x N grid of overlapping tiles, so small faces are seen larger | 1 + N×N detector runs per frame. Helps `face_det_lite`; YOLO rarely needs it |
| `--min-size PX` | `FaceBlurrer(min_size=)` | `0` (`24` for yunet) | Skip faces smaller than PX pixels | Fewer false small faces, misses distant ones |
| `--detect-every N` | `fb.stream(detect_every=)` | `1` | Run the detector every N frames; tracking moves faces in between | N times less detector work; misses more fast or new faces |
| `--delay SEC` | `fb.stream(delay=)` | `0.5` | Hold frames so later detections fix earlier ones: a new face is followed back to the frames the detector missed, a face found again is filled in between | About half as many faces left unblurred; output SEC seconds late. `0` = no delay |
| `--hold SEC` (`blur_live.py`) | `fb.stream(hold=)` | `0.3` | A face the detector lost stays blurred this long, following optical flow | Longer: fewer flickers, blur may stay after the face has gone |
| `--low-score S` (`blur_live.py`) | `fb.stream(low_score=)` | `0.3` | ByteTrack second stage: detections between this and the threshold keep a known face tracked | `0` turns it off |
| `--cpu` | `FaceBlurrer(gpu=False)` | off | Detector on the CPU only, never the Qualcomm NPU | |

Typical settings:

```bash
python quickstart.py --input clip.mp4                                          # default: yolo11n_640, every frame, 0.5 s delay
python quickstart.py --input clip.mp4 --model yolo11n_1280                     # most accurate, small faces
python quickstart.py --input clip.mp4 --detect-every 5                         # slow machine: 3x less work, still 0.5 s delay
python quickstart.py --input rtsp://camera/stream --delay 0                    # lowest latency, less accurate
python quickstart.py --input clip.mp4 --model face_det_lite --tiles 2          # Qualcomm's detector, with tiles for small faces
```

### Blur (mask) and display

| Command line | Python | Default | What it does |
|---|---|---|---|
| `--method gaussian\|pixelate` | `FaceBlurrer(method=)` | `gaussian` | Smooth blur or pixel blocks |
| `--strength 1..3` | `FaceBlurrer(strength=)` | `3` | 1 = light, 3 = heavy |
| `--shape oval\|rect` | `FaceBlurrer(shape=)` | `oval` | Blur an oval that follows the face, or the whole box (safer) |
| `--pad F` | `FaceBlurrer(pad=)` | `0.15` | Extra margin around each face, as a fraction of its size (0 to 1; `0.4` also hides hair) |
| `--show-scores` (`quickstart.py`) | `videoio.draw_faces(frame, res.boxes, res.scores)` | off | Draw each face box with its confidence |

`quickstart.py` always draws the frame rate, analysis time and face count in the top left corner (`videoio.Stats`). See [`examples/options`](examples/options) for what each blur option looks like.

```bash
python quickstart.py --input clip.mp4 --method pixelate --shape rect           # pixel blocks over the whole box
python quickstart.py --input clip.mp4 --pad 0.4 --strength 2                   # larger, lighter blur that also hides hair
python quickstart.py --input clip.mp4 --show-scores                            # face boxes and confidences drawn
```

## Accuracy

Share of faces left unblurred, lower is better. Detector alone, every frame, on two crowded street clips (150 frames each):

| Model | Night street, small faces | Day street |
|---|---|---|
| `yolo11n_640` | 19% | 13% |
| `yolo11n_1280` | 9% | 10% |
| `face_det_lite` | 70% | 35% |
| `face_det_lite`, `tiles=2` | 21% | 22% |

Live stream with tracking, 300 frames of the night street clip, against `yolo11n_1280` run on every frame (1844 faces); time on a Mac mini M4 CPU:

| Model | `detect_every` | `delay=0` | `delay=0.5` (default) | Time per frame |
|---|---|---|---|---|
| `yolo11n_640` | 1 | 22% | **10%** | 20 to 25 ms |
| `yolo11n_640` | 5 | 37% | 22% | 7 to 12 ms |
| `face_det_lite` | 1 | 59% | 52% | 15 ms |
| `face_det_lite` | 5 | 68% | 58% | 6 ms |

- `delay=0.5` about halves the faces left unblurred with YOLO; the blurred area grows by only about 15%.
- `detect_every=5` with `delay=0.5` is 3x faster than `detect_every=1` without delay, and about as accurate.
- `threshold=0.4` gave no false faces with YOLO on these clips.

No face detector is perfect. Check a sample of the output by hand before sharing.

## Example usage

**A video or live stream** with [`quickstart.py`](quickstart.py) ([`videoio.py`](videoio.py) decodes and encodes with ffmpeg, on the video hardware when there is one):

```bash
python quickstart.py --input rtsp://camera/stream                                    # view in a window
python quickstart.py --input rtsp://camera/stream --output rtsp://server:8554/blurred # re-stream (to an RTSP server, e.g. MediaMTX)
python quickstart.py --input clip.mp4 --output out.mp4                               # save to a file
python quickstart.py                                                                 # the sample clip
```

A file plays at its own speed, like a camera: if the machine is too slow, frames are dropped (never shown unblurred). [`tools/blur_video.py`](tools/blur_video.py) blurs every frame of a file instead, however long it takes:

```bash
python tools/blur_video.py in.mp4 out.mp4 --model yolo11n_1280   # H.264 with ffmpeg, else mp4v; no sound
```

**A webcam preview** with `tools/blur_live.py` (no delay; shows the result, saves nothing):

```bash
python tools/blur_live.py 0 --show                         # webcam
python tools/blur_live.py rtsp://camera/stream --show      # IP camera
python tools/blur_live.py clip.mp4                         # timings only, no window
```

**A folder of images** with `run.sh` (Linux / macOS) or `run.bat` (Windows); the first run sets up a `.venv`, which takes a few minutes:

```bash
./run.sh ~/photos/dataset                                  # -> ~/photos/dataset_blurred
./run.sh ~/photos/dataset ~/photos/out --model yolo11n_1280 --method pixelate
python tools/blur_images.py INPUT_DIR OUTPUT_DIR [options] # the same without the launcher
```

Your originals are never changed: results go to a separate folder with the same structure, and `blur_report.csv` lists how many faces were found in each file.

**Python**, images and single frames:

```python
from faceblur import FaceBlurrer

fb = FaceBlurrer(model="yolo11n_1280", method="pixelate")
fb.blur_file("in.jpg", "out.jpg")     # an image file; returns the number of faces
out = fb.blur(frame)                  # a BGR array -> blurred copy
boxes = fb.detect(frame)              # [(x, y, w, h), ...]
print(fb.device)                      # which detector and device were picked
```

## Qualcomm hardware

Install the NPU build of onnxruntime: `pip uninstall -y onnxruntime && pip install onnxruntime-qnn`. The detector then runs on the Hexagon NPU (fp16) and falls back to the CPU when there is none. See [docs/hexagon.md](docs/hexagon.md).

Detector time per frame (one model run) on real devices with Qualcomm AI Hub (ONNX Runtime, NPU requested), November 2026:

| Model | IQ-9075 | QCS8550 | Snapdragon X Elite | RB3 Gen 2 (QCS6490) |
|---|---|---|---|---|
| `yolo11n_640` float (fp16 on the NPU) | 3.7 ms | 2.9 ms | 3.1 ms | 126 ms (CPU) |
| `yolo11n_640` w8a8 | | | | 3.0 ms |
| `yolo11n_1280` float | 14.3 ms | 11.5 ms | 11.9 ms | 475 ms (CPU) |
| `yolo11n_1280` a16w8 | 12.1 ms | 11.6 ms | 12.4 ms | 751 ms (CPU) |
| `face_det_lite` float | 2.7 ms | 1.9 ms | 2.0 ms | 74 ms (CPU) |
| `face_det_lite` a16w8 | 2.0 ms | 1.8 ms | 1.9 ms | 93 ms (CPU) |
| `face_det_lite` w8a8 | | | | 1.4 ms |
| YuNet 640x384 a8w8 | 0.6 ms | 0.5 ms | 0.4 ms | 15 ms (CPU) |
| YuNet 960x544 a16w8 | 2.6 ms | 2.2 ms | 2.1 ms | 93 ms (CPU) |
| YuNet 1280x736 a16w8 | 4.6 ms | 4.2 ms | 4.0 ms | 170 ms (CPU) |

- On IQ-9075, QCS8550 and X Elite the whole model runs on the NPU. `tiles=2` (5 runs per frame) stays around 10 to 13 ms there.
- RB3 Gen 2 (QCS6490): its NPU has no fp16 and the a16w8 graphs do not build, so float and a16w8 fall back to the CPU. **w8a8** (8-bit weights and activations, compiled with AI Hub to QNN DLC or a precompiled QNN ONNX) runs fully on its NPU. The w8a8 models are not in the repo yet and their accuracy is not measured yet.
- Run on the NPU, the face boxes match the CPU float run (99 to 100% for fp16, 98 to 100% for a16w8 on 16 test frames).
- `tools/blur_live.py --npu yunet` uses the quantized YuNet builds.

## Python API notes

- **Setup.** `faceblur.py` is a module in the repo root, not a pip package. Run your code from the repo root, or `sys.path.insert(0, "/path/to/face-blur")`.
- **Frames** are BGR `numpy.uint8` arrays (H x W x 3), as returned by `cv2.imread` and `cv2.VideoCapture.read`. The library does not decode or encode video.
- **Fail closed.** If analysing a frame fails, `res.failed` is `True` and `res.blurred` is the whole frame pixelated.
- **One model per process.** The detector is kept in module state, so a second `FaceBlurrer` with another model replaces the first one's. Create one `FaceBlurrer` and reuse it.
- **Not thread-safe.** Call one `FaceBlurrer` from one thread at a time.
- **Errors.** A wrong argument raises `ValueError`, a detector that cannot start `RuntimeError`.
- **Low-level access.** `detector.detect_faces_scored(img, min_size, thresh)` returns `(x, y, w, h, score)` for the model selected by `FaceBlurrer` or `detector.use_model(name)`.

## Layout

| | |
|---|---|
| `faceblur.py` | Public API: `FaceBlurrer`, `StreamBlurrer`. The only module an app needs to import |
| `detector.py` | Face detectors (YOLO11n-face, face_det_lite, YuNet) and the blur |
| `tracker.py` | Live tracking: ByteTrack + optical flow, and the delay buffer |
| `hexagon.py` | Qualcomm NPU (QNN) detectors |
| `bytetrack/` | ByteTrack (vendored, MIT) |
| `models/` | Model files |
| `quickstart.py` | Example: RTSP or file in, faceblur, live view or re-stream |
| `videoio.py` | Hardware decode / encode with ffmpeg, and the on-screen overlay, for the example |
| `tools/` | Command-line tools (`blur_images.py`, `blur_live.py`, `blur_video.py`) and NPU model scripts |

## Built with

[YOLO11Face](https://github.com/zjykzj/YOLO11Face) (YOLO11n-face, Ultralytics, AGPL-3.0) · [Qualcomm AI Hub](https://huggingface.co/qualcomm/Lightweight-Face-Detection) (Lightweight-Face-Detection) · [ByteTrack](https://github.com/FoundationVision/ByteTrack) (tracker, MIT) · [ONNX Runtime](https://onnxruntime.ai/) (runs the models on CPU or Qualcomm QNN) · [OpenCV](https://opencv.org/) (YuNet detector, blurring, image and video reading)
