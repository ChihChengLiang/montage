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
