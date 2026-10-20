#!/usr/bin/env python3
"""QC for a rendered loop: python3 video/qc.py [video/loop.mp4]
Prints duration, frame count, resolution, size, luma stats (YAVG/YHIGH/YMAX) at 5 evenly spaced times, and the loop-seam
difference (mean abs diff of last frame vs frame 0, with the normal frame-to-frame difference for scale)."""
import subprocess, sys, os, json, numpy as np
f = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "loop.mp4"))
def run(*a): return subprocess.run(a, capture_output=True, text=True).stdout
pr = json.loads(run("ffprobe", "-v", "error", "-count_packets", "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames,nb_read_packets,duration",
                    "-show_entries", "format=duration,size", "-of", "json", f))
v = next(s for s in pr["streams"] if s["codec_type"] == "video"); au = [s for s in pr["streams"] if s["codec_type"] == "audio"]
n = int(v.get("nb_frames") or v["nb_read_packets"]); dur = float(pr["format"]["duration"])
print(f"{os.path.basename(f)}: {v['width']}x{v['height']} @ {v['r_frame_rate']}  {n} frames  video {float(v['duration']):.3f}s  "
      + (f"audio {float(au[0]['duration']):.3f}s  " if au else "no audio  ") + f"file {int(pr['format']['size'])/1e6:.1f} MB ({int(pr['format']['size'])*8/dur/1e6:.2f} Mbps)")
fps = eval(v["r_frame_rate"])
for i in range(5):
    t = (i + 0.5) / 5 * dur
    o = run("ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", f, "-frames:v", "1", "-vf", "signalstats,metadata=print:file=-", "-f", "null", "-")
    st = {l.split("=")[0].split(".")[-1]: l.split("=")[1] for l in o.splitlines() if "YAVG" in l or "YHIGH" in l or "YMAX" in l}
    print(f"  t={t:6.2f}s  YAVG {float(st['YAVG']):6.1f}  YHIGH {st['YHIGH']:>4}  YMAX {st['YMAX']:>4}")
w, h = int(v["width"]), int(v["height"])
def frame(sel):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", f"select='{sel}'", "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(h, w, 3).astype(np.float32)
f0, f1, fl = frame("eq(n,0)"), frame("eq(n,1)"), frame(f"eq(n,{n-1})")
print(f"  loop seam: |last - first| MAE {np.abs(fl - f0).mean():.2f}   (normal consecutive-frame MAE: {np.abs(f1 - f0).mean():.2f})")
