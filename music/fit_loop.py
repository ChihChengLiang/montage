#!/usr/bin/env python3
"""Circular time-stretch of a seamless loop to an exact sample count (periodic cubic spline).
Usage: ana-venv/bin/python music/fit_loop.py in.wav out.wav target_samples"""
import sys, numpy as np, soundfile as sf
from scipy.interpolate import CubicSpline
x, sr = sf.read(sys.argv[1], always_2d=True); N = len(x); M = int(sys.argv[3])
t = np.arange(N + 1, dtype=np.float64)
tn = np.arange(M) * (N / M)
y = np.empty((M, x.shape[1]))
for c in range(x.shape[1]):
    cs = CubicSpline(t, np.append(x[:, c], x[0, c]), bc_type="periodic")
    y[:, c] = cs(tn)
sf.write(sys.argv[2], y.astype(np.float32), sr, subtype="PCM_24" if False else "PCM_16")
print("N", N, "->", M, "stretch", M / N - 1, "sr", sr)
def step(a): return np.abs(np.diff(a, axis=0)).max(0)
j = np.vstack([y[-4:], y[:4]]); print("max |step| at join:", np.abs(np.diff(j, axis=0)).max(), " typical max step elsewhere:", np.abs(np.diff(y, axis=0)).max(), " p99.9:", np.percentile(np.abs(np.diff(y, axis=0)), 99.9))
