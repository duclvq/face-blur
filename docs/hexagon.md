# Face detection on a Qualcomm Hexagon NPU (QNN)

Status: **prepared, not yet run on Hexagon hardware.** The models and code
have only been checked with onnxruntime's CPU execution provider (same
graphs). Expect to adjust things on the first real device.

## Default on Qualcomm: Lightweight-Face-Detection (face_det_lite)

`face_det_lite` is the default `--model` of every script and of `FaceBlurrer`:
Qualcomm's [Lightweight-Face-Detection](https://huggingface.co/qualcomm/Lightweight-Face-Detection)
(BSD-3-Clause, 878K parameters), Qualcomm AI Hub release v0.67.0, **float**
precision (the most accurate one; the HTP runs it in fp16, as AI Hub does).
It runs on the NPU with `--device npu` / `FaceBlurrer(device="npu")`, which
needs onnxruntime with the QNN execution provider (`onnxruntime-qnn`); the
default device is the CPU.

| File (`models/face_det_lite/`) | |
|---|---|
| `face_det_lite.onnx` + `face_det_lite.data` | ONNX model (weights in the .data file; keep both). Used by this repo through onnxruntime (QNN EP on Qualcomm, CPU EP elsewhere) |
| `face_det_lite.dlc` | The same model as a QNN DLC, for native QAIRT/QNN apps (`qnn-net-run`, SNPE). Not used by the Python code |
| `metadata_onnx.json`, `metadata_qnn_dlc.json`, `LICENSE` | From the AI Hub release |

Input: 640x480 grayscale in [0, 1] (the frame is letterboxed: a 16:9 1080p
frame becomes 640x360 + padding). Output: a stride-8 heatmap, box offsets and
landmarks; `hexagon.QnnFaceDetLite` decodes them exactly as Qualcomm's
`face_det_lite/utils.detect` (checked box for box on the repo images), score
threshold 0.55 and NMS 0.2 as in Qualcomm's evaluator.

AI Hub reports 1-3 ms per inference on the NPU for this float model on recent
Snapdragons (8 Gen 3, X Elite, QCS8550); there is no float build listed for
QCS6490-class chips (no fp16 HTP), use `--precision a8w8` there.

Against full-resolution YuNet on the street clip (CPU EP), it matched 85% of
YuNet's faces at 360p and 79% at 1080p (94% of faces >= 72 px at 1080p):
small faces are lost by the 640-px input.

## Quantized and compiled builds

Qualcomm AI Hub made these files from the float models (quantize and compile
jobs of 2026-11-23). `python tools/fetch_aihub_models.py` downloads them again;
the job of each file is in that script.

| File | Format | NPU time per inference |
|---|---|---|
| `yolo11n_face/yolo11n_face_640x384_a16w8.onnx` | QDQ ONNX | IQ-9075 3.5 ms, QCS8550 3.3 ms, X Elite 3.2 ms |
| `yolo11n_face/yolo11n_face_1280x736_a16w8.onnx` | QDQ ONNX | IQ-9075 12.1 ms, QCS8550 11.6 ms, X Elite 12.4 ms |
| `yolo11n_face/yolo11n_face_640x384_a8w8.onnx` | QDQ ONNX, scores split from boxes | |
| `yolo11n_face/yolo11n_face_1280x736_a8w8.onnx` | QDQ ONNX, scores split from boxes | |
| `yolo11n_face/qcs6490/yolo11n_face_640x384_a8w8.dlc` | QNN DLC for RB3 Gen 2 | RB3 Gen 2 3.2 ms |
| `yolo11n_face/qcs6490/yolo11n_face_1280x736_a8w8.dlc` | QNN DLC for RB3 Gen 2 | RB3 Gen 2 18.2 ms |
| `face_det_lite/face_det_lite_a16w8.onnx` | QDQ ONNX | IQ-9075 2.0 ms, QCS8550 1.8 ms, X Elite 1.9 ms |
| `face_det_lite/face_det_lite_a8w8.onnx` | QDQ ONNX | |
| `face_det_lite/qcs6490/face_det_lite_a8w8.dlc` | QNN DLC for RB3 Gen 2 | RB3 Gen 2 1.4 ms |

- The a8w8 YOLO models output boxes, scores and keypoints as three tensors
  (`a_boxes`, `b_scores`, `c_kpts`). In one 8-bit tensor, the box coordinates
  (0 to 1280) leave too few steps for the scores (0 to 1).
- Each DLC is the a8w8 ONNX above it, compiled for the QCS6490 HTP. Native
  QAIRT apps (`qnn-net-run`, SNPE) load it; the Python code does not.
- The RB3 Gen 2 runs only 8-bit models on its NPU. The a16w8 models fall back
  to its CPU (93 to 750 ms).
- `--precision a16w8` or `a8w8` (`FaceBlurrer(precision=)`) picks these ONNX
  files instead of the float one, for `yolo11n_640`, `yolo11n_1280` and
  `face_det_lite`. They run on any device.
- The a8w8 YOLO boxes are 8-bit over the whole input width: one step is 2.5 px
  at 640x384 and 5 px at 1280x736. Faces about 10 px wide in the model input
  come out with loose boxes, and NMS keeps some duplicates.
- On the CPU EP, the mean score difference from the float model on faces the
  float model finds is 0.025 for a8w8 YOLO, 0.015 for a8w8 face_det_lite and
  below 0.01 for a16w8 (`examples/input`).

## YuNet (quantized, fixed sizes)

## What is there

| File | |
|---|---|
| `tools/make_yunet_qnn.py` | Builds the NPU models from `models/face_detection_yunet_2023mar_dynamic.onnx` |
| `tools/eval_yunet_qnn.py` | Accuracy / speed against full-resolution float YuNet (`--qnn` to run on the NPU) |
| `hexagon.py` | `QnnYuNet` detector: letterbox, QNN session, decode + NMS as in `OrtYuNet` |
| `models/face_detection_yunet_2023mar_qnn_{W}x{H}_{a8w8,a16w8,fp32}.onnx` | Static-shape models |

Sizes: `640x384`, `960x544` (default), `1280x736`. A 1080p frame is scaled to
fit and zero-padded at the bottom/right. Precisions:

- `a16w8` (default): uint16 activations, int8 per-channel weights. Needs an HTP
  with 16-bit activation support (newer Snapdragons; check your SoC).
- `a8w8`: uint8 activations, int8 per-channel weights. Works on every HTP;
  loses more faces at low scores (`--low-score` association stage).
- `fp32`: float model, run in fp16 on the HTP (`enable_htp_fp16_precision`).
  Only on HTPs with fp16.

Pick `960x544 a16w8`; use `640x384` if the NPU is too slow (it misses faces
smaller than ~30 px at 1080p), `1280x736` for small faces, `a8w8` if 16-bit is
rejected.

## Install

- **Windows on Snapdragon (ARM64)**: `pip install onnxruntime-qnn` (ARM64 Python).
  The wheel ships the QNN libraries (`QnnHtp.dll` etc.).
- **Android / Linux (QCS6490, RB5, ...)**: build or get onnxruntime with the QNN
  EP, and put the QNN SDK libraries for your SoC (`libQnnHtp.so`,
  `libQnnHtpV??Stub.so`, `libQnnHtpV??Skel.so`, `libQnnSystem.so`) on
  `LD_LIBRARY_PATH` / `ADSP_LIBRARY_PATH`. Set `QNN_BACKEND_PATH` to the full
  path of `libQnnHtp.so` if it is not found by name.

Then copy the repo (the models are in `models/`; rebuild with
`python tools/make_yunet_qnn.py` on any machine if needed).

## Run

```bash
python tools/blur_live.py 0 --show --model face_det_lite --device npu   # face_det_lite on the NPU
python tools/blur_live.py 0 --show --npu                      # the same
python tools/blur_live.py 0 --show --npu yunet               # YuNet 960x544 a16w8
python tools/blur_live.py in.mp4 --show --npu yunet:640x384 --npu-precision a8w8
python quickstart.py --input in.mp4 --device npu --precision a8w8   # yolo11n_640 a8w8, for the RB3 Gen 2
```

From code: `FaceBlurrer(model="face_det_lite", device="npu")`,
`FaceBlurrer(device="npu", precision="a8w8")`, `detector.use_npu("face_det_lite")`,
`detector.use_npu("yunet", (960, 544), "a16w8")`, or `hexagon.QnnFaceDetLite` /
`hexagon.QnnYuNet`.

## What to check on the device

1. The start-up line must read `Face detection: YuNet ... on QNNExecutionProvider, CPUExecutionProvider`.
   `WARNING: ... using CPU` means QNN did not load and the normal CPU detector is used.
2. Every node should be on the HTP. Run with a verbose log and look for nodes
   assigned to `CPUExecutionProvider`:
   ```python
   from hexagon import QnnYuNet
   QnnYuNet((960, 544), "a16w8", log_level=1)   # 0 = verbose
   QnnYuNet((960, 544), "a16w8", strict=True)   # fails if any node falls back to the CPU
   ```
3. Accuracy and speed on the NPU: `python tools/eval_yunet_qnn.py --qnn`
   (compare with the CPU numbers below).
4. First start compiles the graph for the HTP (can take seconds). The compiled
   context is cached in `~/.cache/face-blur/qnn/*_ctx.onnx` (`ep.context_enable`)
   and reused; delete it after changing SoC, QNN SDK or onnxruntime version.
   For sustained runs try `htp_performance_mode="sustained_high_performance"`
   (in `hexagon.qnn_provider_options`).

## How the models are made

- Static shapes; the in-graph `Transpose`/`Cast(uint8 -> float)` of the
  dynamic model is removed (input is float32 NCHW, BGR 0..255, built on the
  host). `Cast` is not quantizable and the quantizer leaves a `Transpose` on a
  float graph input unquantized, so both would sit outside the QDQ graph. The
  input quantization is lossless (a8: scale 1, zero point 0).
- Landmark outputs (`kps_*`) dropped; opset 11 -> 21.
- `qnn_preprocess_model` + `get_qnn_qdq_config(per_channel=True, weight_type=QInt8)`,
  MinMax calibration on `examples/input` and every 10th frame of the street clip.

## Measured on CPU EP (Apple M4), threshold 0.7

Reference: float dynamic YuNet at full resolution. 1080p video = the street
clip upscaled to 1920x1080 (372 frames, calibration frames excluded); images =
`examples/input` (7 photos, 72 faces). `rec>=32`: recall of faces >= 32 px.

| model | images recall | images rec>=32 | video recall | video prec |
|---|---|---|---|---|
| 640x384 fp32 / a16w8 / a8w8 | 0.69 / 0.69 / 0.69 | 0.86 | 0.86 / 0.85 / 0.85 | 0.92 |
| 960x544 fp32 / a16w8 / a8w8 | 0.85 / 0.83 / 0.85 | 0.98 | 0.91 / 0.91 / 0.91 | 0.91 |
| 1280x736 fp32 / a16w8 / a8w8 | 0.93 / 0.93 / 0.93 | 0.98 | 0.93 / 0.93 / 0.93 | 0.91 |

Quantization alone (quantized vs fp32 of the same size, recall):
a16w8 0.99 at threshold 0.7, 0.97 at 0.1; a8w8 0.99 at 0.7, 0.89 at 0.1.
The loss is mostly from the smaller input, not from quantization. CPU timings
say nothing about the NPU.
