# ACE-Step 1.5 local music (Apple Silicon, MLX DiT)

Setup (uv 0.5.10 prints a harmless `required-environments` warning):
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5 repo
    cd repo && uv sync            # ~1 min, .venv ~1.2GB
Weights auto-download on first run into repo/checkpoints (~9.4GB, from HF ACE-Step/Ace-Step1.5).

Generate (script: ../gen.py; turbo DiT, 8 steps, no LM / thinking=False):
    cd repo && .venv/bin/python ../gen.py 75      # arg = duration seconds
Output goes to out/*.wav (48kHz stereo); copied to track.wav, track.mp3 via ffmpeg.

Prompt: "calm elegant cinematic instrumental, traditional Taiwanese flavour, gentle guzheng and erhu melody over soft piano and warm strings, bamboo flute, slow, spacious, contemplative, museum and history documentary underscore, no vocals, no drums"
Params: lyrics "[Instrumental]", bpm 68, D Major, 4/4, seed 42, duration 75s.

Runtime: diffusion+decode ~146s for 75s of audio (16GB RAM). Model init/download extra.
Problems: first attempt died when the disk filled (only ~23GB free then); the first
run's DiT step was ~77s/it under memory/disk pressure. Rerun after freeing disk was fine.
LM (1.7B) was downloaded but not used.

## v2: seamless loop base layer (loop.wav / loop.mp3)
Old track kept as track_v1.wav/.mp3.
    cd repo && .venv/bin/python ../gen_loop.py 100     # -> out2/*.wav, copied to raw_100bpm.wav (100 s, 48k)
    cd .. && repo/.venv/bin/python make_loop.py        # -> loop.wav (PCM16), then:
    ffmpeg -i loop.wav -b:a 256k loop.mp3
Prompt: "subtle ambient texture-first instrumental, steady 100 bpm, plucked guzheng ostinato pattern, light woodblock and soft shaker percussion, warm soft pads, very sparse minimal melody, constant even energy, no build-up, no climax, no vocals, background music bed"
Params: bpm=100, D Major, 4/4, seed 7, turbo 8 steps, thinking=False. Gen ~173 s for 100 s audio.
Note: an 80 s generation goes silent after ~76.4 s (natural ending), so 100 s was generated and only the first 78 s used.
Loop: loop = raw[0:3686400] (128 beats @100bpm), with the first 57600 samples (2 beats) replaced by
equal-power crossfade: raw[0:X]*sin + raw[L:L+X]*cos, so the end flows into the head. Peak-normalised to -1 dBFS.
Verified: exactly 3686400 samples, 48 kHz, 76.8 s; loop+loop join step 0.014 vs max step elsewhere 0.057 (no click).
Caveats: tempo only roughly checked (autocorrelation peak ~101 bpm; the model is not sample-locked to the grid, so
beats may drift slightly from a true 100 bpm grid); energy is not perfectly flat (per-4s RMS varies, 8-beat RMS 0.054-0.124).

## v3: beat detection and alignment (loop_aligned.wav)
Real beat tracking (librosa, `detect_beats.py`) showed the generated music is a steady **75 bpm**, not 100:
beat period 0.800 s (grid fit 75.01 bpm, 21 ms RMS), even though bpm=100 was requested. The video cuts every
1.2 s / 2.4 s, which are multiples of the 0.4 s eighth-note grid of the 75 bpm music, so each cut lands on a beat or
exactly halfway between beats. The pulse sat ~27 ms (<1 frame) after the cuts, so `align_beats.py` rotates the
loop circularly by 1296 samples -> `loop_aligned.wav` (still exactly 3686400 samples; the loop stays seamless).
Run from the repo root with librosa + soundfile installed (e.g. `uv venv && uv pip install librosa soundfile`).

## v4: Gould bird plates loop (gould_loop_aligned.wav/.mp3)
Scripts (generalised): `gen_gould.py DUR BPM SEED...`, `make_loop_param.py RAW OUT PERIOD BEATS [XFADE_BEATS] [START]`,
`align_param.py IN OUT GRID_PERIOD`; librosa lives in `ana-venv` (uv venv; `uv pip install librosa soundfile`).
    cd repo && .venv/bin/python ../gen_gould.py 100 85 11 22     # ~170-210 s each
    ana-venv/bin/python detect_beats.py out_gould/seed22.wav
    ana-venv/bin/python make_loop_param.py gould_raw.wav gould_loop.wav 0.7061425 112
    ana-venv/bin/python align_param.py gould_loop.wav gould_loop_aligned.wav 0.7061425
Prompt: "gentle airy instrumental, morning garden, natural history, soft acoustic texture, light plucked harp and kalimba, occasional glockenspiel touches, soft warm pads, warm low strings, very sparse minimal melody, steady even unobtrusive energy, no drums, no vocals, no build-up, calm background music bed" (G Major, requested 85 bpm).
Seeds 11 and 22: both detected ~84.7-85 bpm; seed 22 chosen (2 s RMS CV 0.125 vs 0.314; beat-fit residual rms 13 ms vs 75 ms).
Detected beat period 0.7061 s (84.97 bpm), eighth 0.3531 s. N=112 beats -> 3796222 samples = 79.088 s.
Alignment: rolled by 2448 samples (51 ms); eighth-grid offset agreed (48 ms).

## v5: loop points from the MIDDLE of the generation (find_loop.py) -> loop_seam.*, gould_loop_seam_fit.*
Problem: loops cut from raw t=0 begin on the generation's intro and end ~79 s in (decay/ending), so the join is audible.
`find_loop.py RAW P T0 L_BEATS OUT [--step 1] [--max-end 94] [--old a.wav,b.wav]` tries every start = T0 + k*P on the beat grid
(start+L+xfade <= max-end), builds each loop (2-beat equal-power crossfade of head with the continuation raw[start+L:...]),
and scores the join: distance between the 2-bar windows after vs before the join for chroma, log-mel, spectral centroid,
log-RMS and onset density, each divided by the median distance between adjacent 2-bar windows elsewhere in the raw
(1.0 = as different as neighbouring music; lower = smoother). Also rejects 1-beat RMS ratio > 1.5 across the join and decaying ends.
- Gould: raw gould_raw.wav (seed 22), P=0.7061425, t0=0.051, 112 beats: best start 4.994 s, score 0.49 (old gould_loop_fit.wav: 0.97,
  its 1-beat RMS ratio at join 1.81 vs 1.40 now). Then `fit_loop.py` to 3,796,800 samples (79.1 s, +0.015% stretch, beat period 0.70625 s)
  -> gould_loop_seam_fit.wav/.mp3. Grid offset after the fit: 5 ms (< 1 frame), no roll.
- Wood: the old raw_100bpm.wav could not be made smooth (best score 2.56; old loop 1.34; only a ~12 s start range, non-stationary).
  Regenerated 110 s, loop-friendly prompt (static modal harmony, D pentatonic, repeating guzheng ostinato, woodblock, drone pad, no intro/ending/fade),
  `gen_seed.py 110 75 caption_wood2.txt out_wood2 31 32 33` (~200-240 s each). Detected 75.0 bpm (seeds 31, 33; seed 32 gave 152 bpm).
  Search on seed 31 (P=0.8): best start 23.186 s (start shifted -29 ms so the pulse lands at t=0; remaining offset 1 ms), score 0.44
  (seed 33 scored 0.35 but with a worse beat fit; old loop 1.34). Output loop_seam.wav/.mp3, exactly 3,686,400 samples = 76.8 s.
  `seam_check.py LOOP P` prints join step and per-beat RMS around the join of the tiled loop.
Caveat: all numbers are signal statistics; nobody has listened.
