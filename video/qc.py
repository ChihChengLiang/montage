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
def frames(sel, cnt):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-vf", f"select='{sel}'", "-vsync", "0", "-frames:v", str(cnt), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return list(np.frombuffer(b, np.uint8).reshape(-1, h, w, 3).astype(np.float32))
tail = frames(f"gte(n,{n-4})", 4); head = frames("lt(n,4)", 4)
mid = frames(f"between(n,{n//2},{n//2+3})", 4)
mae = lambda a, b: float(np.abs(a - b).mean())
seam = mae(tail[-1], head[0])
before = [mae(tail[i], tail[i + 1]) for i in range(3)]; after = [mae(head[i], head[i + 1]) for i in range(3)]
typ = np.mean([mae(mid[i], mid[i + 1]) for i in range(3)])
print(f"  loop seam: |F[N-1] - F[0]| MAE {seam:.2f}   (steps just before the seam {np.mean(before):.2f}, just after {np.mean(after):.2f}, typical mid-video step {typ:.2f})")
print(f"  seam is {'continuous' if seam <= 1.5 * max(np.mean(before), np.mean(after), 0.5) else 'ABRUPT'} (seam step {seam/max(np.mean(before),0.01):.2f}x the preceding step)")
