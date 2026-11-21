# face-blur

Blur every face in a folder of photos or videos, for example to anonymize a research dataset before sharing it.

| Before | After |
|:-:|:-:|
| ![original](examples/options/original.jpg) | ![blurred](examples/options/gaussian_3.jpg) |

## Quick start

**Linux / macOS**

```bash
git clone https://github.com/duclvq/face-blur.git
cd face-blur
./run.sh ~/photos/dataset          # a folder of images  -> ~/photos/dataset_blurred
./run.sh ~/videos/clips            # a folder of videos  -> ~/videos/clips_blurred
./run.sh ~/videos/party.mp4        # one video           -> ~/videos/party_blurred.mp4
```

**Windows** (images only for now)

```bat
run.bat C:\path\to\dataset
```

The first run installs everything into a local `.venv` and downloads the face model, which takes a few minutes. For videos, also install [ffmpeg](https://ffmpeg.org/download.html) so the output keeps its sound.

## Models

| `--model` | When to use it |
|---|---|
| `raccoon_s` (default) | Best results. Finds small and hard-to-see faces, and in videos follows faces from frame to frame. Slow on a laptop. |
| `raccoon_l` | Like `raccoon_s`, larger and slower. |
| `yunet` | Old or low-end machines. Much faster, but misses small faces. |

The raccoon models come from [InsightFace](https://github.com/deepinsight/insightface) and are for **non-commercial research only**. For commercial use, use `--model yunet` (MIT license).

## Options

Add them after the input (and optional output folder):

```bash
./run.sh ~/photos/dataset ~/photos/out --model yunet --method pixelate
```

| Option | Default | What it does |
|---|---|---|
| `--model` | `raccoon_s` | Face detector, see above |
| `--method gaussian\|pixelate` | `gaussian` | Smooth blur or pixel blocks |
| `--strength 1..3` | `3` | 1 = light, 3 = heavy |
| `--shape oval\|rect` | `oval` | Blur an oval around the face, or the whole box (safer) |
| `--pad 0..1` | `0.15` | Extra margin around each face (`0.4` also hides hair) |
| `--min-size N` | off (`24` for yunet) | Skip faces smaller than N pixels |
| `--cpu` | off | Don't use the GPU. By default an NVIDIA or Apple GPU is used when there is one, else the CPU |
| `--analysis-fps N` | `15` | Videos: face scans per second. Use `30` for fast motion |

See [`examples/options`](examples/options) for what each option looks like.

## Output

- Your originals are never changed. Results go to a separate folder with the same structure.
- `blur_report.csv` lists how many faces were found in each file.
- Videos keep their format, frame rate, size, sound and subtitles (needs ffmpeg).

No face detector is perfect. Check a sample of the output by hand before sharing.

## Running the scripts directly

```bash
pip install -r requirements.txt
python blur_faces.py INPUT_DIR OUTPUT_DIR [options]   # images
python blur_video.py INPUT OUTPUT [options]           # a video or a folder of videos
```

For an NVIDIA GPU: `pip uninstall -y onnxruntime && pip install -r requirements-gpu.txt` (driver 580 or newer). `run.sh` does this by itself.

## Built with

[InsightFace](https://github.com/deepinsight/insightface) (SCRFD face detector, PrivateFrame video tracking) · [ONNX Runtime](https://onnxruntime.ai/) (runs the models on CPU, NVIDIA CUDA or Apple CoreML) · [OpenCV](https://opencv.org/) (YuNet detector, blurring, image and video reading) · [ffmpeg](https://ffmpeg.org/) (video encoding, keeps sound and format)
