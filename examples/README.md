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
| `crowd/street_parade.jpg` (Apollo 11 ticker-tape parade, New York, 1969) | 27 with `--model raccoon_s`, 10 with yunet |
| `crowd/room_mission_control.jpg` (Mission Control after Apollo 11, 1969) | 19 with `--model raccoon_s`, 4 with yunet |
| `crowd/film_set.jpg` (behind-the-scenes photos from a film set) | 25 with `--model raccoon_s`, 22 with yunet |

Image credits: `team/*` and `crowd/*` are NASA photos (public domain, images.nasa.gov; the crowd photos are S70-17433 and S69-40022). `sports/football.jpg` is `messi5.jpg` from the OpenCV sample data. `crowd/film_set.jpg` is from https://i.redd.it/pa2mjlht85731.jpg (not public domain; copyright belongs to its owner).

`video/street_before_after.*` shows `crowd_street.mp4` (from https://www.youtube.com/watch?v=b8QZJ5ZodTs, not public domain) before and after `blur_video.py --model raccoon_s`.
