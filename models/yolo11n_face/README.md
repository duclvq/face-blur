# YOLO11n-face

`yolo11n-pose_widerface.pt` from [zjykzj/YOLO11Face](https://github.com/zjykzj/YOLO11Face) (release v1.0.0),
a YOLO11n-pose trained on WIDER FACE with [Ultralytics](https://github.com/ultralytics/ultralytics),
exported to ONNX (opset 17, float32) at two fixed input sizes:

| File | Input (w x h) |
|---|---|
| `yolo11n_face_640x384.onnx` | 640 x 384 |
| `yolo11n_face_1280x736.onnx` | 1280 x 736 |

Input: RGB, letterboxed (pad value 114 at the bottom/right), divided by 255, NCHW.
Output `(1, 20, N)`: cx, cy, w, h, face score, then 5 keypoints x (x, y, visibility).

**License:** the weights were trained with Ultralytics, which is AGPL-3.0
(https://www.gnu.org/licenses/agpl-3.0.html); the YOLO11Face repository is Apache-2.0.
Using these models in a product may require AGPL compliance or an Ultralytics Enterprise license.
