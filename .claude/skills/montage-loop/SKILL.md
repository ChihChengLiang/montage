---
name: montage-loop
description: Build a seamless looping "vibe" montage video (cropped detail shots cut on a beat, one colour grade, generated instrumental loop) from an open-licensed image set. Use when asked to make a montage, background loop or B-roll base layer from a collection of images, or to reuse this repo's pipeline on a new image set.
---

# montage-loop

Turns a folder of openly licensed object photos into a seamless looping video (default 76.8 s, 1080p30, ~75 MB)
plus a matching ACE-Step music loop. Goal is a **base layer / vibe**, not a showcase of objects: tight crops of
texture and detail, cuts on the music's grid, one warm grade. Code is in this repo; this file is the workflow and the
judgment calls. Details of every config key are in `README.md` and `montage.json`.

macOS + Apple Silicon only (Vision framework for analysis, ACE-Step via MLX).

## Workflow

1. **Source images and verify licences per item.** Check each item's own licence badge (a collection-level
   statement is not enough). Prefer CC0/public domain. Only use full-resolution variants: crops need >= ~1400 px of
   source width, so 640 px previews are unusable for full-frame shots (that is why CC BY previews were dropped here).
   Download into `images/`, write `images/credits.json` (fields: `file`, `title`, `accession_no`, `license`,
   `source_page`) and `images/CREDITS.md`. Ask before bulk-downloading from a new site.
2. **Create `montage.json`** (copy and edit). Set `images_dir`, `image_glob`, `license_filter`, `drop`.
3. **Analyse** (focal points + text boxes, all local): `video/analyze.sh`.
4. **Check the plan, no render:** `python3 video/build_loop.py --check`. Look for images with too-small crops.
   Add `drop` entries for images with printed captions, rulers or labels baked in (postcards, scale strips) and
   `manual_safe_rects` for objects that fill the frame (fabric, documents). Re-run until it prints `OK`.
5. **Music** (see `music/README.md`; weights ~10 GB, check free disk first, `df -h`):
   - Generate ~100 s with `music/gen_loop.py`, cut to the exact loop length with a 2-beat equal-power crossfade
     of the tail onto the head (`music/make_loop.py`). Loop length = `loop_beats * 60 / bpm` seconds.
   - **Do not trust the requested bpm.** ACE-Step ignored `bpm=100` and produced 75 bpm. Run
     `music/detect_beats.py` (librosa) to get the real tempo, then `music/align_beats.py` to rotate the loop so
     the pulse lands on frame 0 (circular shift keeps the loop seamless). Use the aligned file as `audio`.
   - Choose cut lengths that are multiples of the music's eighth-note grid (0.4 s at 75 bpm; 1.2 s / 2.4 s cuts
     worked), so cuts fall on beats or half-beats. Cuts will not land on every beat; that is fine.
   - If the generation goes silent near the end, generate longer and use only the first part.
6. **Render** (about 12 min): `python3 video/build_loop.py`. Writes `video/loop.mp4` (2-pass, 60-100 MB target)
   and `video/description.txt`.
7. **QC before calling it done:** `python3 video/qc.py`, and look at `video/work/contact.jpg`. Check:
   - duration and frame count equal `loop_beats * 60 / bpm` and `fps`; audio length matches;
   - loop seam MAE is not larger than the normal frame-to-frame MAE;
   - brightness: YHIGH around 200-230, YMAX below ~235, no large clipped areas;
   - no registration numbers, rulers or labels visible in any shot; no near-duplicate adjacent shots.
   Nobody can listen in the build: tell the user to check the loop point and tempo by ear.
8. **Publish:** `video/description.txt` carries CC0/CC BY credits and the AI-music disclosure. Do not commit mp4s.

## Lessons learned (do not rediscover)

- **Exposure:** meter on the object, not the whole frame. Studio shots have pale backdrops; normalising to a
  mid-gray frame average turns them into dim beige-brown. Keep the backdrop warm cream and let the object set exposure.
- **White balance is rarely the problem** on museum studio shots (already near neutral). Neutralise the cast per
  image, then apply one shared grade.
- **Crop to fill 16:9**, never letterbox, for a vibe layer. Roles: hero (around the focal point), detail, texture.
- **Keep crops away from detected text** and the bottom ruler/label band (macOS Vision text boxes + margin).
- Loop closure: last shot is the first crop reversed, so the last frame lands on frame 0.
- `build_loop.py` regenerates `video/description.txt`; edit the description in `montage.json`, never the output file.
- The encode (2-pass x264 + grain) is not bit-reproducible; compare by PSNR/SSIM, not by hash.
- Disk: a fresh ACE-Step setup needs ~11 GB, an uncapped near-lossless render 770 MB. `uv cache clean` freed 26 GB once.
- Clean up: don't leave multi-hundred-MB intermediates in `video/work/`.

## Not parameterised (edit code if needed)

16:9 crop aspect, saliency/gradient scoring weights, focal drift factor, safe-region margins, white-balance and
exposure exponents. Analysis is macOS-only.
