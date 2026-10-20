# montage

A 76.8-second seamless looping video — warm, wooden, textural — meant as a **base layer / vibe background** for
other content, not a showcase of objects. Images come from the open collection of the
[National Museum of Taiwan History](https://collections.nmth.gov.tw/) (國立臺灣歷史博物館); music is generated
locally with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). Built end to end on an Apple Silicon Mac
(16 GB RAM) with Claude Code subagents.

The rendered `loop.mp4` (1080p30, ~76 MB) is not committed; rebuild it as below.

## How it works

**Images** (`images/`): 22 CC0 items from the museum's Open Data zone, in their large JPG variants. The museum's
per-item license badges were checked; CC BY 4.0 items exist on the site but only as 640 px previews, too soft for
full-frame crops, so the final cut uses CC0 only. 17 of the 22 are used (postcards with printed captions were
dropped). `credits.json` / `CREDITS.md` record title, registration number, license and source URL for each.

**Picture** (`video/build_loop.py`):
- Every shot is a crop that fills 16:9, no letterboxing. Each object gets a *hero* shot (composed around the focal
  point), a *detail* shot and a *texture* shot, 53 crops in all.
- Focal points and text are found locally with macOS Vision (`video/tools/vis.swift`: attention/objectness saliency
  and text boxes). Crop windows are kept clear of registration numbers, rulers and label strips.
- One shared grade: per-image white balance, then warm amber highlights, brown lifted blacks, desaturated cool
  colours, subtle vignette and light grain.
- Cuts every 2 or 4 beats (1.2 s / 2.4 s) as hard cuts, with a 4% push-in toward the focal target. The last shot
  is the first one played in reverse, so the last frame lands on frame 0 and the video loops seamlessly.
  Exactly 2304 frames at 30 fps.

**Music** (`music/`): ACE-Step 1.5 turbo DiT (MLX backend), instrumental, about 100 s generated, then cut to
exactly 76.8 s with the tail crossfaded onto the head (seamless audio loop). Beat tracking with librosa showed the
model actually produced a steady **75 bpm** even though 100 was requested. The cuts fall on a beat or half-beat, and
`music/align_beats.py` rotates the loop 27 ms so the pulse lines up with the cuts to within a frame.
Prompts, seeds and findings are in `music/README.md`.

## Rebuild

Needs macOS (Vision framework), Python 3 with `numpy`, `opencv-python`, `pillow`, ffmpeg and `swiftc`.

```bash
video/analyze.sh                  # compiles video/tools/vis, writes focal-point + text analysis to video/work/an/
python3 video/build_loop.py       # renders (about 12 minutes) -> video/loop.mp4, video/description.txt, video/work/contact.jpg
```

Optional: `python3 video/build_loop.py --check` validates inputs (images, analysis, credits, audio length) and prints
the shot plan without rendering; `python3 video/qc.py` prints duration, frame count, luma stats and the loop-seam
difference of the rendered file.

Music: see `music/README.md` (ACE-Step setup + generation, ~10 GB of weights), then `python3 music/align_beats.py`
writes `music/loop_aligned.wav`. That file is committed, so the two commands above reproduce the video without
installing ACE-Step.

### Reusing the pipeline for other images

Everything set-specific lives in `montage.json` (paths are relative to that file; pass another config as the last
argument to any command, e.g. `python3 video/build_loop.py other.json`):

| key | meaning |
|---|---|
| `images_dir`, `image_glob`, `drop` | where the images are, which files to take, image keys to skip (key = file name up to the first `_`) |
| `credits_file`, `license_filter` | JSON list with `file`, `title`, `accession_no`, `license`, `source_page` per image; only entries whose license starts with the filter are used |
| `manual_safe_rects` | `{key: [x0,y0,x1,y1]}` fractions, for images whose subject fills the frame (keeps crops off edges/labels) |
| `width`, `height`, `fps`, `bpm`, `loop_beats` | output format and the cut grid; loop length = `loop_beats * 60 / bpm` s |
| `shot_cycle_beats`, `dissolve`, `push_in` | shot-length cycle (must fill `loop_beats` exactly), short dissolve on long shots, push-in amount |
| `crop` | `min_width` (source px), crop sizes, `roles` per image and `extra_roles` per image key |
| `grade` | warm tint, black lift, gamma, saturation, vignette, grain, highlight knee, exposure targets |
| `x264`, `audio`, `out_dir`, `work_dir`, `out_name`, `seed` | encode targets (2-pass bitrate), audio file to mux, output locations, grain seed |
| `description` | `lines` for `description.txt` (`{credits}` expands to one line per used item using `credit_line`) |

Not parameterised: the 16:9 crop aspect, the saliency/gradient scoring weights, the focal-point drift, and the
hard-coded `swiftc`/Vision analysis (macOS only).

`build_loop.py` muxes the `audio` file if it exists, otherwise leaves `loop_silent.mp4`.

## Caveats

- Nobody has listened to the generated music in the build process; check the loop point and the tempo by ear.
- Cuts alternate between beats and half-beats at 75 bpm; they do not land on every beat.
- Crop and focal-point choices are automatic (saliency + gradient scores), so a few shots are weak or similar
  (e.g. two near-identical wooden-bar shots).

## License

Code: MIT. Images: CC0 1.0 from the National Museum of Taiwan History (attribution not required, listed as a
courtesy; see `images/CREDITS.md`). ACE-Step is Apache-2.0; check its terms for generated audio. YouTube
description text is in `video/description.txt`.
