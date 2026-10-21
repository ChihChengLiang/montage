#!/usr/bin/env python3
"""5-second 'eye lock' opener: rapid cuts between bird plates with the eye pinned to one screen position and one size.

Usage: python3 video/eyelock.py [eyelock.json] [--check]     (--check: plan only, no render)
Inputs: images_gould/eyes.json (hand-labelled eyes: plate, bird, eye_xy [source px], eye_d [source px], facing, plate_size),
        safe rects of each plate (from gould.json manual_safe_rects), and optionally music/sting_info.json
        (bpm / beat_seconds / first_beat_offset) + music/eyelock_sting.wav.
Config: see eyelock.json. plan = [[start_beat, cuts_per_beat], ...] up to hold.start_beat, then one macro-eye hold."""
import json, os, sys, glob, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
args = [a for a in sys.argv[1:] if not a.startswith("--")]; flags = {a for a in sys.argv[1:] if a.startswith("--")}
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CP = os.path.abspath(args[0] if args else os.path.join(ROOT, "eyelock.json")); CFG = json.load(open(CP)); BASE = os.path.dirname(CP)
P = lambda x: x if os.path.isabs(x) else os.path.join(BASE, x)
W, H, FPS = CFG["width"], CFG["height"], CFG["fps"]
WK, OUT = P(CFG["work_dir"]), P(CFG["out_dir"]); os.makedirs(WK, exist_ok=True)
LK, PU, FL, GR, HD = CFG["lock"], CFG["punch"], CFG["flash"], CFG["grade"], CFG["hold"]
BEATS = CFG["duration_beats"]; BS = CFG["beat_seconds"]; OFF = CFG.get("first_beat_offset", 0.0)
HIT = None; info = P(CFG["sting_info"]) if CFG.get("sting_info") else None
if info and os.path.exists(info) and "--no-info" not in flags:
    si = json.load(open(info)); BS = si.get("beat_seconds", BS); OFF = si.get("first_beat_offset_s", si.get("first_beat_offset", si.get("offset", OFF))); HIT = si.get("hit_time_s"); print("using", info, "beat_seconds", BS, "offset", OFF)
TOTAL = round(BEATS * BS * FPS)
rng = np.random.default_rng(CFG["seed"])
safe_all = json.load(open(P(CFG["safe_rects_from"])))["manual_safe_rects"]
files = {f.split("/")[-1][:3]: f for f in glob.glob(P(CFG["images_dir"]) + "/g*.jpg")}
eyes = [e for e in json.load(open(P(CFG["eyes_file"]))) if e["status"] == "ok"]
TX, TY = LK["x"] * W, LK["y"] * H

def geometry(e, s):
    """source box (L,T,R,B) and flip flag so that the eye lands on (TX,TY) at scale s; None if outside the plate's safe rect"""
    w, h = e["plate_size"]; ex, ey = e["eye_xy"]; flip = e["facing"] != LK["facing"]
    L = ex + TX / s - W / s if flip else ex - TX / s
    T = ey - TY / s; box = (L, T, L + W / s, T + H / s)
    x0, y0, x1, y1 = [v * (w if i % 2 == 0 else h) for i, v in enumerate(safe_all[e["plate"]])]
    inside = box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1
    return box, flip, inside

# ---- plan: cut frames
cuts = []                                           # (beat, section_flag)
for (sb, cpb), nxt in zip(CFG["plan"], [p[0] for p in CFG["plan"][1:]] + [HD["start_beat"]]):
    k = 0
    while sb + k / cpb < nxt - 1e-9: cuts.append(sb + k / cpb); k += 1
cuts.append(HD["start_beat"])
starts = [round((OFF + b * BS) * FPS) for b in cuts]; starts[0] = 0
ends = starts[1:] + [TOTAL]
NSHOT = len(cuts)

# ---- choose eyes
hold = next(e for e in eyes if e["bird"] == HD["eye"])
pool = []
for e in eyes:
    if e is hold: continue
    s = LK["eye_d"] / e["eye_d"]
    if s > LK["max_scale"]: continue
    _, _, ok = geometry(e, s)
    if ok: pool.append((s, e))
pool.sort(key=lambda t: t[0]); need = NSHOT - 1
print(f"eyes labelled {len(eyes)}, usable at eye_d={LK['eye_d']}px: {len(pool)}, shots needed: {need}")
if len(pool) > need: pool = pool[:need]               # drop the most upscaled (softest) ones
while len(pool) < need: pool.append(pool[len(pool) % len(pool)])
imgs = {}
def plate(k):
    if k not in imgs:
        if len(imgs) > 4: imgs.pop(next(iter(imgs)))
        imgs[k] = Image.open(files[k]).convert("RGB")
    return imgs[k]

def render(e, s, f_in_shot, nf, kind, sec_start_hold=False):
    """frame for eye e at base scale s; punch/kick on cut, slow push-in on the hold"""
    u = f_in_shot
    if kind == "hold": k = 1 + HD["push_in"] * f_in_shot / max(nf - 1, 1)
    else:
        pf = PU["frames"]; k = 1 + PU["amount"] * max(0.0, 1 - f_in_shot / pf) ** 2 if pf > 0 else 1.0
    box, flip, _ = geometry(e, s * k)
    fr = plate(e["plate"]).resize((W, H), Image.BICUBIC, box=box)
    if flip: fr = fr.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(fr, dtype=np.float32) / 255.0

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = np.hypot((xx - TX) / (W / 2), (yy - TY) / (H / 2))
VIG = (1 - GR["vignette"] * np.clip((rr - GR["vignette_start"]) / (1 - GR["vignette_start"]), 0, 1) ** 1.5)[..., None]
def grade(a, gain):
    a = a * gain
    l = (a * [0.299, 0.587, 0.114]).sum(2, keepdims=True); a = l + (a - l) * GR["saturation"]
    a = (a - 0.5) * GR["contrast"] + 0.5
    a = a * VIG
    return a

def shot_gain(e, s):
    fr = render(e, s, 99, 99, "cut"); l = (fr * [0.299, 0.587, 0.114]).sum(2)
    return float(np.clip(GR["target_luma"] / max(l.mean(), 0.05), 0.55, 1.25)), float(l.mean()), fr[..., :].mean((0, 1))

# ---- order for contrast (alternate light/dark, warm/cool, never same plate twice in a row)
feat = []
for s, e in pool:
    g, lm, col = shot_gain(e, s); feat.append((lm, float(col[0] - col[2]), s, e, g))
med = np.median([f[0] for f in feat]); hi = sorted([f for f in feat if f[0] >= med], key=lambda f: -f[1]); lo = sorted([f for f in feat if f[0] < med], key=lambda f: f[1])
order = []
for i in range(max(len(hi), len(lo))):
    if i < len(hi): order.append(hi[i])
    if i < len(lo): order.append(lo[i])
for i in range(1, len(order)):                       # avoid same plate back to back
    if order[i][3]["plate"] == order[i - 1][3]["plate"]:
        j = next((j for j in range(i + 1, len(order)) if order[j][3]["plate"] != order[i - 1][3]["plate"]), None)
        if j: order[i], order[j] = order[j], order[i]
hs = HD["scale"] * LK["eye_d"] / hold["eye_d"]
print(f"hold eye {hold['bird']} scale {hs:.2f} (eye {HD['scale']*LK['eye_d']:.0f}px)")
plan = [dict(bird=f[3]["bird"], e=f[3], s=f[2], gain=f[4], kind="cut") for f in order] + [dict(bird=hold["bird"], e=hold, s=hs, gain=0, kind="hold")]
hl = shot_gain(hold, hs); plan[-1]["gain"] = hl[0]
for i, (p, a, b) in enumerate(zip(plan, starts, ends)):
    print(f"shot {i:2d} {p['bird']:5s} {p['kind']:4s} frames {a:3d}-{b-1:3d} ({b-a}f)  scale {p['s']:.2f}  beat {cuts[i]:.2f}")
print(f"cuts {NSHOT}, total frames {TOTAL} ({TOTAL/FPS:.3f}s)  beat_seconds {BS}  fastest shot {min(b-a for a,b in zip(starts,ends))} frames")
if "--check" in flags: sys.exit(0)

# ---- render
silent = f"{OUT}/{CFG['out_name']}_silent.mp4"
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-crf", str(CFG["crf"]), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent], stdin=subprocess.PIPE)
flash_beats = {round((OFF + b * BS) * FPS) for b in FL["beats"]}
if HIT and FL.get("on_hit", True): flash_beats.add(round(HIT * FPS))          # extra flash frame on the sting hit
n = 0; thumbs = []
for i, (p, a, b) in enumerate(zip(plan, starts, ends)):
    nf = b - a
    for f in range(nf):
        fr = render(p["e"], p["s"], f, nf, p["kind"]); out = grade(fr, p["gain"])
        out = out + rng.normal(0, GR["grain"], (H, W, 1)).astype(np.float32)
        if (a in flash_beats and f < FL["frames"]) or ((a + f) in flash_beats and a + f != a):
            out = out * (1 - FL["amount"]) + np.array(FL["color"], np.float32) / 255 * FL["amount"]
        o8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
        enc.stdin.write(o8.tobytes()); n += 1
        if f == min(PU["frames"], nf - 1): thumbs.append((p["bird"], o8))
enc.stdin.close(); enc.wait(); assert n == TOTAL, (n, TOTAL)
json.dump([(p["bird"], a, b, p["s"], p["e"]["facing"] != LK["facing"]) for p, a, b in zip(plan, starts, ends)], open(f"{WK}/shots.json", "w"))

# contact sheet with the lock crosshair
cols = 5; tw, th = 384, 216; rows = (len(thumbs) + cols - 1) // cols
S = Image.new("RGB", (cols * tw, rows * th), "black")
for i, (bid, a8) in enumerate(thumbs):
    t = Image.fromarray(a8).resize((tw, th)); d = ImageDraw.Draw(t); cx, cy = TX * tw / W, TY * th / H
    d.line([cx - 14, cy, cx + 14, cy], fill=(255, 0, 0), width=1); d.line([cx, cy - 14, cx, cy + 14], fill=(255, 0, 0), width=1); d.text((4, 3), f"{i} {bid}", fill=(255, 255, 0))
    S.paste(t, ((i % cols) * tw, (i // cols) * th))
S.save(f"{WK}/contact.jpg", quality=88)

# mux the sting if present
wav = P(CFG["audio"]) if CFG.get("audio") else None; out = f"{OUT}/{CFG['out_name']}.mp4"
if wav and os.path.exists(wav):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{TOTAL/FPS:.6f}", "-movflags", "+faststart", out], check=True)
    print("muxed", out)
else: print("no audio yet; silent:", silent)
