# face-blur

Detect faces in every image of a dataset folder and blur them, for example to anonymize a dataset before sharing it.

## Install

Requires Python 3.8+.

```bash
git clone https://github.com/duclvq/face-blur.git
cd face-blur
pip install -r requirements.txt
```

## Windows quick start

Double-click is not enough; give it a folder. Either drag your dataset folder onto `run.bat`, or run:

```bat
run.bat C:\path\to\dataset
run.bat C:\path\to\dataset C:\path\to\output --method pixelate
```

On the first run `run.bat` sets everything up by itself: it finds Python (installing Python 3.12 with `winget` if it is missing), creates a private `.venv` folder next to the script and installs OpenCV into it. Later runs start straight away. If no output folder is given, results go to `DATASET_blurred`. Delete `.venv` to force a clean reinstall.

## Usage

```bash
python blur_faces.py INPUT_DIR OUTPUT_DIR
```

Options:

| Option | Default | Description |
|---|---|---|
| `--method gaussian\|pixelate` | `gaussian` | Blur style |
| `--strength 1..3` | `3` | 1 = light, 3 = heavy |
| `--min-size N` | `24` | Smallest face to detect, in pixels |

Example:

```bash
python blur_faces.py ./my_dataset ./my_dataset_blurred --method pixelate
```

## Output

- `OUTPUT_DIR` mirrors the folder structure of `INPUT_DIR`. Every image is written out, and images without faces are unchanged.
- `OUTPUT_DIR/blur_report.csv` lists how many faces were found in each file and any files that could not be read.

Supported formats: jpg, jpeg, png, bmp, webp, tif, tiff.

## Limitations

Detection uses OpenCV's built-in Haar cascades (frontal and profile faces). They can miss small, tilted, dark or partly hidden faces. If you need strict anonymization, review a sample of the output by hand.
