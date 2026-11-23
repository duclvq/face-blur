#!/usr/bin/env python3
"""Build Hexagon (onnxruntime QNN EP, HTP) ready YuNet models.

From models/face_detection_yunet_2023mar_dynamic.onnx this writes, for each
fixed input size W x H:

    models/face_detection_yunet_2023mar_qnn_{W}x{H}_fp32.onnx   float, static shapes
    models/face_detection_yunet_2023mar_qnn_{W}x{H}_a8w8.onnx   QDQ, uint8 act / int8 per-channel weights
    models/face_detection_yunet_2023mar_qnn_{W}x{H}_a16w8.onnx  QDQ, uint16 act / int8 per-channel weights

Graph changes against the dynamic model (all needed or useful for the HTP):
  * static shapes: input [1, 3, H, W], every output [1, anchors, C];
  * the in-graph preprocessing (Transpose NHWC -> NCHW, Cast uint8 -> float)
    is removed: the input is float32 NCHW, BGR pixel values 0..255, made on
    the host (hexagon.letterbox). Cast is not a quantizable op, and the
    quantizer leaves a Transpose that reads a float graph input unquantized,
    so both would stay float in the QDQ graph (CPU fallback or an fp16 island
    at the HTP partition's entry). Without them the first node is the input
    QuantizeLinear (scale 1, zero point 0 for 0..255 in a8w8: lossless).
    The QNN EP converts the NCHW graph to the HTP's NHWC layout itself.
  * the landmark heads (kps_*) are dropped: face-blur does not use them;
  * opset raised from 11 to 21 (QDQ per-channel and 16-bit Q/DQ need it).

Quantization: onnxruntime.quantization.execution_providers.qnn
(qnn_preprocess_model + get_qnn_qdq_config, i.e. the QNN EP's own QDQ rules),
MinMax calibration on letterboxed frames from examples/input and every 10th
frame of the street clip (left half; the other frames are used for evaluation).

Usage:
    python tools/make_yunet_qnn.py [--sizes 640x384 960x544 1280x736]
                                   [--precisions fp32 a8w8 a16w8] [--video PATH]
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import helper, version_converter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hexagon import PRECISIONS, SIZES, STRIDES, YUNET_DYNAMIC, letterbox, model_path  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
STREET = ROOT / "examples" / "video" / "street_before_after.mp4"
CALIB_EVERY = 10  # calibration uses frames 0, 10, 20, ... of the street clip


def street_frames(path=STREET, every=1, offset=0):
    """Frames of the street clip; for the before/after file, only the left
    (unblurred) half."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == offset:
            if Path(path).name == STREET.name:
                f = f[:, : f.shape[1] // 2]
            yield i, f
        i += 1
    cap.release()


def calibration_images(video=STREET):
    imgs = [cv2.imread(str(p)) for p in sorted((ROOT / "examples" / "input").rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS]
    imgs += [f for _, f in street_frames(video, CALIB_EVERY)]
    return imgs


def fixed_float_model(size):
    """Dynamic YuNet -> static-shape float model with a float32 NHWC input."""
    W, H = size
    m = onnx.load(str(YUNET_DYNAMIC))
    g = m.graph
    # Remove Transpose(NHWC -> NCHW) -> Cast(uint8 -> float): the graph input
    # feeds the first Conv directly.
    cast = next(n for n in g.node if n.op_type == "Cast")
    trans = next(n for n in g.node if n.output[0] == cast.input[0])
    for n in g.node:
        for i, name in enumerate(n.input):
            if name == cast.output[0]:
                n.input[i] = inp_name = trans.input[0]
    g.node.remove(cast)
    g.node.remove(trans)
    del g.value_info[:]
    # Static input; keep only the outputs face-blur decodes.
    inp = g.input[0]
    inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    assert inp.name == inp_name
    for d, v in zip(inp.type.tensor_type.shape.dim, (1, 3, H, W)):
        d.ClearField("dim_param")
        d.dim_value = v
    channels = {"cls": 1, "obj": 1, "bbox": 4}
    outs = []
    for o in g.output:
        kind, s = o.name.split("_")
        if kind in channels:
            outs.append(helper.make_tensor_value_info(
                o.name, onnx.TensorProto.FLOAT, [1, (H // int(s)) * (W // int(s)), channels[kind]]))
    del g.output[:]
    g.output.extend(outs)
    # Drop the nodes that only fed kps_*.
    needed = {o.name for o in g.output}
    keep = []
    for n in reversed(g.node):
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(n.input)
    del g.node[:]
    g.node.extend(reversed(keep))
    used = {i for n in g.node for i in n.input}
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    m = version_converter.convert_version(m, 21)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m)
    return m


class Reader:
    """onnxruntime CalibrationDataReader over letterboxed images."""

    def __init__(self, imgs, size, name="input"):
        self.blobs = [letterbox(im, size)[0] for im in imgs]
        self.name, self.i = name, 0

    def get_next(self):
        if self.i >= len(self.blobs):
            return None
        self.i += 1
        return {self.name: self.blobs[self.i - 1]}

    def rewind(self):
        self.i = 0


def quantize(float_path, out_path, imgs, size, precision):
    import logging
    logging.getLogger().setLevel(logging.ERROR)  # "consider pre-processing" noise
    from onnxruntime.quantization import QuantType, quantize as ort_quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
    act = {"a8w8": QuantType.QUInt8, "a16w8": QuantType.QUInt16}[precision]
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        src = pre if qnn_preprocess_model(str(float_path), str(pre)) else float_path
        cfg = get_qnn_qdq_config(str(src), Reader(imgs, size), activation_type=act,
                                 weight_type=QuantType.QInt8, per_channel=True)
        # Conv weights: per output channel, signed int8, symmetric (zero point
        # 0), which is what the HTP needs for per-channel. Biases are int32.
        ort_quantize(str(src), str(out_path), cfg)


def parse_size(s):
    w, h = (int(v) for v in s.lower().split("x"))
    if w % 32 or h % 32:
        raise argparse.ArgumentTypeError("sizes must be multiples of 32")
    return (w, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", nargs="+", type=parse_size, default=list(SIZES))
    ap.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    ap.add_argument("--video", type=Path, default=STREET, help="clip for calibration frames")
    o = ap.parse_args()
    imgs = calibration_images(o.video)
    print(f"{len(imgs)} calibration images")
    for size in o.sizes:
        fp = model_path(size, "fp32")
        onnx.save(fixed_float_model(size), str(fp))
        print("wrote", fp.relative_to(ROOT))
        for prec in o.precisions:
            if prec == "fp32":
                continue
            out = model_path(size, prec)
            quantize(fp, out, imgs, size, prec)
            print("wrote", out.relative_to(ROOT))
        if "fp32" not in o.precisions:
            fp.unlink()


if __name__ == "__main__":
    main()
