# Examples

`input/` is a small sample dataset (with a subfolder per category). The outputs were produced with:

```bash
python blur_faces.py examples/input examples/output                                       # gaussian blur
python blur_faces.py examples/input/sports examples/output_pixelate --method pixelate    # pixelate
```

On Windows: `run.bat examples\input examples\output`

| Image | Faces found |
|---|---|
| `team/S69-31739.jpg` (Apollo 11 crew) | 3 |
| `team/jsc2008e049010.jpg` (Apollo 8 prime + backup crew) | 6 |
| `team/sts061-05-031.jpg` (STS-61 crew, in orbit) | 7 |
| `sports/football.jpg` | 1 |

Image credits: `team/*` are NASA photos (public domain, images.nasa.gov). `sports/football.jpg` is `messi5.jpg` from the OpenCV sample data.
