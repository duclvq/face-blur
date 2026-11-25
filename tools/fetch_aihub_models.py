"""Download the quantized and compiled face detectors from Qualcomm AI Hub.

Each file under models/ below is the output of one AI Hub job (the jobs whose
NPU times and accuracy are in README.md and docs/hexagon.md). Needs
`pip install qai-hub onnx` and an AI Hub API token (`qai-hub configure`).

    python tools/fetch_aihub_models.py            # every file
    python tools/fetch_aihub_models.py --force    # overwrite existing files

Quantize jobs give a QDQ ONNX with its weights in a separate file; it is saved
here as one .onnx file. Compile jobs for the RB3 Gen 2 (QCS6490) give a QNN DLC.
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

JOBS = {
    "yolo11n_face/yolo11n_face_640x384_a8w8.onnx": "jprmexkvg",
    "yolo11n_face/yolo11n_face_1280x736_a8w8.onnx": "jp2qlo8xp",
    "yolo11n_face/yolo11n_face_640x384_a16w8.onnx": "jpe0kvn0g",
    "yolo11n_face/yolo11n_face_1280x736_a16w8.onnx": "jp3lemlzp",
    "yolo11n_face/qcs6490/yolo11n_face_640x384_a8w8.dlc": "jp8dzjoz5",
    "yolo11n_face/qcs6490/yolo11n_face_1280x736_a8w8.dlc": "jgkw36zyg",
    "face_det_lite/face_det_lite_a8w8.onnx": "jp08lo905",
    "face_det_lite/face_det_lite_a16w8.onnx": "jg97w42v5",
    "face_det_lite/qcs6490/face_det_lite_a8w8.dlc": "jgzqrzy4g",
}


def save_single_onnx(zip_path, dest):
    import onnx
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp)
        model = onnx.load(str(next(Path(tmp).rglob("*.onnx"))))  # loads the external weights too
    onnx.save(model, str(dest))


def fetch(rel_path, job_id):
    import qai_hub as hub
    dest = MODELS_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = Path(hub.get_job(job_id).get_target_model().download(str(Path(tmp) / "model")))
        if dest.suffix == ".onnx":
            save_single_onnx(downloaded, dest)
        else:
            shutil.copyfile(downloaded, dest)
    print(f"{rel_path} <- {job_id} ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()
    for rel_path, job_id in JOBS.items():
        if (MODELS_DIR / rel_path).exists() and not args.force:
            print(f"{rel_path} exists")
            continue
        fetch(rel_path, job_id)


if __name__ == "__main__":
    main()
