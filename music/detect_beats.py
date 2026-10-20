import librosa, numpy as np, sys
y, sr = librosa.load(sys.argv[1], sr=22050, mono=True)
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=256, start_bpm=100, tightness=400, units='time')
print("dur", len(y)/sr, "tempo", np.atleast_1d(tempo)[0], "nbeats", len(beats))
d = np.diff(beats); print("median ibi", np.median(d), "=> bpm", 60/np.median(d), "std", d.std())
# best-fit grid: beats ~ t0 + k*P
k = np.round((beats-beats[0])/np.median(d))
P, t0 = np.polyfit(k, beats, 1)
print("fit period", P, "bpm", 60/P, "offset t0", t0, "resid ms max", 1000*np.abs(beats-(t0+k*P)).max(), "rms", 1000*np.sqrt(np.mean((beats-(t0+k*P))**2)))
# phase of onset-strength vs 0.6 s grid at several tempi
import itertools
for bpm in (98,99,100,101,102):
    p=60/bpm; best=max(((oenv[(np.arange(0,len(y)/sr-p,p)+o)*sr/256].astype(int) if False else 0) for o in [0]), default=0)
