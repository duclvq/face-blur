# Examples

`input/` is a small sample dataset (with a subfolder per category). The outputs were produced with:

```bash
python tools/blur_images.py examples/input examples/output                                       # gaussian blur
python tools/blur_images.py examples/input/sports examples/output_pixelate --method pixelate    # pixelate
```

On Windows: `run.bat examples\input examples\output`

| Image | Faces found |
|---|---|
| `team/S69-31739.jpg` (Apollo 11 crew) | 3 |
| `team/jsc2008e049010.jpg` (Apollo 8 prime + backup crew) | 6 |
| `team/sts061-05-031.jpg` (STS-61 crew, in orbit) | 7 |
| `sports/football.jpg` | 1 |
| `crowd/street_parade.jpg` (Apollo 11 ticker-tape parade, New York, 1969) | 27 (10 with `--model yunet`) |
| `crowd/room_mission_control.jpg` (Mission Control after Apollo 11, 1969) | 19 (4 with `--model yunet`) |
| `crowd/film_set.jpg` (behind-the-scenes photos from a film set) | 25 (22 with `--model yunet`) |

Image credits: `team/*` and `crowd/*` are NASA photos (public domain, images.nasa.gov; the crowd photos are S70-17433 and S69-40022). `sports/football.jpg` is `messi5.jpg` from the OpenCV sample data. `crowd/film_set.jpg` is from https://i.redd.it/pa2mjlht85731.jpg (not public domain; copyright belongs to its owner).

`video/models_compare.*` is 15 s of https://www.youtube.com/watch?v=7F0Cy-RGOGY (Chongqing bar street at night; not public domain, copyright belongs to its owner) blurred by each built-in model; made with `FaceBlurrer.stream()` at its defaults.

`video/subway_before_after.*` is 12 s of https://www.youtube.com/watch?v=jN5OKuOTTJc (a subway train; not public domain, copyright belongs to its owner): the original on top, below it every frame through `FaceBlurrer(model="yolo11n_1280", device="gpu").stream(fps=60, delay=30, low_score=0.2, detect_every=2)` with the `videoio.Stats` overlay.

`video/street_before_after.*` shows `crowd_street.mp4` (from https://www.youtube.com/watch?v=b8QZJ5ZodTs, not public domain) before and after `tools/blur_video.py` with the default model.
