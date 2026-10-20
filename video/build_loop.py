#!/usr/bin/env python3
"""Seamless 128-beat (76.8 s @100bpm) warm-wood vibe loop from CC0 crops.
Needs: video/tools/vis (swift Vision helper: `swiftc -O video/tools/vis.swift -o video/tools/vis`) and
per-image analysis json in video/work/an (made by: for f in images/c*.jpg; do video/tools/vis $f > video/work/an/<name>.json; done).
Usage: python3 video/build_loop.py"""
import json, os, subprocess, glob, numpy as np, cv2
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG, VID, WK = f"{ROOT}/images", f"{ROOT}/video", f"{ROOT}/video/work"
os.makedirs(f"{WK}/mid", exist_ok=True); os.makedirs(f"{WK}/an", exist_ok=True)
W, H, FPS, BEAT = 1920, 1080, 30, 18
TOTAL = 128 * BEAT
rng = np.random.default_rng(11)
cred = {c["file"]: c for c in json.load(open(f"{IMG}/credits.json")) if c["license"].startswith("CC0")}
# postcards carry printed captions/low-res: dropped.  (c07,c16 small; c09,c19,c21 text-heavy postcards)
DROP = {"c07", "c09", "c16", "c19", "c21"}
# manual safe rects (x0,y0,x1,y1 as fractions) where the object fills the frame (fabric album / document): keep off backdrop edges + label strip
MANUAL = {"c12": (0.04, 0.05, 0.96, 0.83), "c22": (0.07, 0.04, 0.93, 0.87)}
MIN_W = 1300

def load(file):
    key = file[:3]; an = json.load(open(f"{WK}/an/{file[:-4]}.json"))
    im = Image.open(f"{IMG}/{file}").convert("RGB"); w, h = im.size
    # safe rect: above any detected text/ruler band (+margin), inside image edges
    if key in MANUAL: sx0, sy0, sx1, sy1 = [v * (w if i % 2 == 0 else h) for i, v in enumerate(MANUAL[key])]
    else:
        sy1 = h * 0.98
        for x, y, bw, bh in an["text"]:
            top = (1 - (y + bh)) * h
            if top > 0.5 * h: sy1 = min(sy1, top - 0.04 * h)
        sx0, sy0, sx1 = 0.02 * w, 0.02 * h, 0.98 * w
    safe = (sx0, sy0, sx1, sy1)
    s = 6; sm = np.asarray(im.resize((w // s, h // s), Image.BILINEAR), dtype=np.float32)
    g = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY); gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    G = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2)
    gh, gw = G.shape
    A = cv2.resize(np.array(an["attn_map"], np.float32), (gw, gh)); O = cv2.resize(np.array(an["obj_map"], np.float32), (gw, gh))
    F = 0.6 * A / (A.max() + 1e-6) + 0.4 * O / (O.max() + 1e-6)
    F[:int(sy0 / s)] = 0; F[int(sy1 / s):] = 0; F[:, :int(sx0 / s)] = 0; F[:, int(sx1 / s):] = 0
    boxes = [(x * w, (1 - y - bh) * h, (x + bw) * w, (1 - y) * h) for x, y, bw, bh in an["obj"]] or [(sx0, sy0, sx1, sy1)]
    fb = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    fb = (max(fb[0], sx0), max(fb[1], sy0), min(fb[2], sx1), min(fb[3], sy1))
    ys, xs = np.mgrid[0:gh, 0:gw]; tot = F.sum() + 1e-6
    focal = (float((F * xs).sum() / tot * s), float((F * ys).sum() / tot * s))
    # white balance from backdrop (low-saliency pixels inside safe rect); object-filling images: gentle gray-world
    mask = np.zeros((gh, gw), bool); mask[int(sy0 / s):int(sy1 / s), int(sx0 / s):int(sx1 / s)] = True
    bgm = mask & (F < 0.05)
    if bgm.sum() > 0.12 * mask.sum() and key not in MANUAL:
        ref = np.median(sm[bgm], 0); gains = ref.mean() / ref
    else:
        m = sm[mask].mean(0); gains = 1 + 0.5 * (m.mean() / m - 1)
    gains = np.clip(gains, 0.8, 1.3).astype(np.float32)
    bgl = float((np.median(sm[bgm], 0) * gains * [0.299, 0.587, 0.114]).sum() / 255) if (bgm.sum() > 0.12 * mask.sum() and key not in MANUAL) else None
    return dict(file=file, key=key, bg=bgl, im=im, w=w, h=h, safe=safe, G=G, F=F, s=s, fb=fb, focal=focal, gains=gains, c=cred[file])

def win_score(d, cx, cy, cw, mode):
    s = d["s"]; ch = cw * 9 / 16
    a, b, c, e = int((cx - cw/2) / s), int((cy - ch/2) / s), int((cx + cw/2) / s), int((cy + ch/2) / s)
    g = d["G"][b:e, a:c].mean(); f = d["F"][b:e, a:c].mean()
    return g * (1 + (3 * f if mode == "detail" else 0))

def clampc(d, cx, cy, cw):
    x0, y0, x1, y1 = d["safe"]; ch = cw * 9 / 16
    return min(max(cx, x0 + cw/2), x1 - cw/2), min(max(cy, y0 + ch/2), y1 - ch/2)

def fits(d, cw):
    x0, y0, x1, y1 = d["safe"]; return cw <= x1 - x0 and cw * 9 / 16 <= y1 - y0

def iou(a, b):
    w = min(a[0]+a[2]/2, b[0]+b[2]/2) - max(a[0]-a[2]/2, b[0]-b[2]/2); h = min(a[1]+a[2]*9/32, b[1]+b[2]*9/32) - max(a[1]-a[2]*9/32, b[1]-b[2]*9/32)
    return 0 if w <= 0 or h <= 0 else w * h / (min(a[2], b[2]) ** 2 * 9 / 16)

def plan(d, roles):
    """return list of shots dict(role, cx, cy, cw, tx, ty) for image d"""
    out, fb = [], d["fb"]; fcx, fcy = (fb[0]+fb[2])/2, (fb[1]+fb[3])/2
    fx, fy = d["focal"]; x0, y0, x1, y1 = d["safe"]
    for role in roles:
        if role == "hero":
            cw = max(0.5 * d["w"], (fb[2]-fb[0]) * 1.2, (fb[3]-fb[1]) * 16/9 * 1.2)
            cw = min(cw, x1 - x0, (y1 - y0) * 16 / 9)
            cx, cy = fcx + 0.06 * cw, fcy                      # slightly off-centre (thirds-ish), push toward the focal point
            tx, ty = fx, fy
        elif role == "wide":
            cw = min(x1 - x0, (y1 - y0) * 16 / 9); cx, cy = fcx, fcy; tx, ty = fx, fy
        else:
            cw = max(MIN_W, 0.38 * d["w"]) if role == "detail" else MIN_W * 1.02
            if not fits(d, cw): cw = min(x1 - x0, (y1 - y0) * 16 / 9) * 0.9
            best = None
            for cx_ in np.linspace(max(x0, fb[0]-0.1*cw), min(x1, fb[2]+0.1*cw), 16):
                for cy_ in np.linspace(max(y0, fb[1]-0.1*cw), min(y1, fb[3]+0.1*cw), 12):
                    cx2, cy2 = clampc(d, cx_, cy_, cw)
                    if any(iou((cx2, cy2, cw), (o["cx"], o["cy"], o["cw"])) > 0.45 for o in out if o["role"] != "hero"): continue
                    sc = win_score(d, cx2, cy2, cw, role)
                    if best is None or sc > best[0]: best = (sc, cx2, cy2)
            if best is None: continue
            cx, cy = best[1], best[2]
            if role == "detail": tx, ty = fx, fy
            else: tx, ty = cx + (0.05 if len(out) % 2 else -0.05) * cw, cy      # texture: gentle sideways drift
        cx, cy = clampc(d, cx, cy, cw)
        out.append(dict(role=role, cx=cx, cy=cy, cw=cw, tx=tx, ty=ty))
    return out

files = [f for f in sorted(os.listdir(IMG)) if f.startswith("c") and f.endswith(".jpg") and f[:3] not in DROP]
imgs = [load(f) for f in files]
need = 53; per = [["hero", "detail", "texture"] for _ in imgs]
for i in (2, 9): per[i] = ["hero", "detail", "texture", "wide"] if len(imgs) > i else per[i]
shots = []
for d, roles in zip(imgs, per):
    for p in plan(d, roles): shots.append((d, p))
print("images", len(imgs), "crops", len(shots), [(d["key"], sum(1 for x,_ in shots if x is d)) for d in imgs])
# trim/extend to exactly 53 unique shots (drop extra wides first)
while len(shots) > need:
    j = next((k for k in range(len(shots)-1, -1, -1) if shots[k][1]["role"] == "wide"), len(shots) - 1); shots.pop(j)
assert len(shots) == need, len(shots)
shots.append(shots[0])                                  # last = first, reversed -> seamless
pattern = ([2, 2, 2, 2, 4] * 10) + [2, 2, 2, 2]
assert len(pattern) == len(shots) and sum(pattern) == 128

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.hypot((xx - W/2) / (W/2), (yy - H/2) / (H/2))
VIG = (1 - 0.16 * np.clip(r - 0.5, 0, 1) ** 1.6)[..., None].astype(np.float32)
WARM = np.array([1.07, 0.98, 0.84], np.float32)         # amber/honey highlights
LIFT = np.array([0.055, 0.040, 0.028], np.float32)       # slight brown lifted blacks
KNEE, TOP = 0.84, 0.935                                   # soft highlight limiter (max ~235)

def grade(a, gain, expo):
    a = a / 255.0 * gain * expo
    l = (a * [0.299, 0.587, 0.114]).sum(2, keepdims=True)
    cool = np.clip((np.maximum(a[..., 1], a[..., 2]) - a[..., 0])[..., None] * 3.0, 0, 1)   # blues/greens
    a = l + (a - l) * (0.84 - 0.38 * cool)
    a = a * (1 + (WARM - 1) * (0.5 + 0.5 * np.clip(l, 0, 1)))   # warm, stronger in highlights
    a = np.clip(a, 0, 1.3) ** 0.9                                # midtone brightening
    sm_ = np.clip(a, 0, 1); a = a * 0.8 + 0.2 * (sm_ * sm_ * (3 - 2 * sm_))   # gentle S-curve
    a = a + LIFT * (1 - np.clip(l, 0, 1)) ** 2                   # brown-tinted shadows
    a = a * VIG
    hi = a > KNEE                                                # soft knee: no clipping to 255
    a = np.where(hi, KNEE + (TOP - KNEE) * (1 - np.exp(-(a - KNEE) / (TOP - KNEE))), a)
    a = a + rng.normal(0, 0.007, (H, W, 1)).astype(np.float32)   # light luma grain
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

inter = f"{WK}/inter.mp4"
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-crf", "10", "-preset", "veryfast", "-pix_fmt", "yuv420p", inter], stdin=subprocess.PIPE)
prev = None; n = 0; meta = []
for si, ((d, p), beats) in enumerate(zip(shots, pattern)):
    nf = beats * BEAT; last = si == len(shots) - 1; im = d["im"]
    cw0, ch0 = p["cw"], p["cw"] * 9 / 16
    # exposure normalise per shot from first-frame crop
    t = im.crop((p["cx"]-cw0/2, p["cy"]-ch0/2, p["cx"]+cw0/2, p["cy"]+ch0/2)).resize((192, 108))
    L = (np.asarray(t, np.float32) / 255 * d["gains"] * [0.299, 0.587, 0.114]).sum(2)
    p98 = float(np.percentile(L, 98))
    obj = np.abs(L - d["bg"]) > 0.10 if d["bg"] is not None else np.ones_like(L, bool)   # object mask = not-backdrop
    if obj.mean() > 0.05:
        ml = float(L[obj].mean()); expo = (0.42 / max(ml, 0.03)) ** 0.6 if d["bg"] is not None else (0.52 / max(ml, 0.03)) ** 0.8
    else: expo = 1.0
    expo = float(np.clip(expo, 0.9, 1.7)); expo = min(expo, 0.97 / max(p98, 0.05))   # backdrop stays bright cream, no clipping
    for f in range(nf):
        u = f / (nf - 1); u = (1 - u) if last else u
        z = 1 - 0.04 * u
        k = 0.7 * u                                      # drift toward target as we push in
        cx = p["cx"] + (p["tx"] - p["cx"]) * k * 0.5; cy = p["cy"] + (p["ty"] - p["cy"]) * k * 0.5
        cw = cw0 * z; cx, cy = clampc(d, cx, cy, cw)
        ch = cw * 9 / 16
        fr = np.asarray(im.resize((W, H), Image.BICUBIC, box=(cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2)), dtype=np.float32)
        out = grade(fr, d["gains"], expo)
        if beats == 4 and f < 3 and prev is not None:
            al = (f + 1) / 4; out = (prev.astype(np.float32) * (1 - al) + out * al).astype(np.uint8)
        enc.stdin.write(out.tobytes()); n += 1
        if f == nf // 2 and not last: Image.fromarray(out).save(f"{WK}/mid/{si:02d}.jpg", quality=88)
        if f == nf - 1: prev = out
    meta.append((si, d["c"]["accession_no"], p["role"], beats))
    print("shot", si, d["key"], p["role"], beats, flush=True)
enc.stdin.close(); enc.wait(); assert n == TOTAL, n

# contact sheet
cols = 6; tw, th = 320, 180; rows = (len(shots) - 1 + cols - 1) // cols
S = Image.new("RGB", (cols * tw, rows * (th + 14)), "white"); dr = ImageDraw.Draw(S)
for si, acc, role, b in meta[:-1]:
    x, y = (si % cols) * tw, (si // cols) * (th + 14)
    S.paste(Image.open(f"{WK}/mid/{si:02d}.jpg").resize((tw, th)), (x, y)); dr.text((x + 2, y + th + 1), f"{si} {acc} {role} {b}b", fill="black")
S.save(f"{WK}/contact.jpg", quality=85)

# final: 2-pass capped bitrate + loop.wav
wav = f"{ROOT}/music/loop_aligned.wav"; out = f"{VID}/loop.mp4"
pl = f"{WK}/x264pass"
base = ["ffmpeg", "-y", "-v", "error", "-i", inter]
subprocess.run(base + ["-c:v", "libx264", "-preset", "slow", "-b:v", "7700k", "-maxrate", "11000k", "-bufsize", "16000k", "-pass", "1", "-passlogfile", pl, "-an", "-f", "null", "/dev/null"], check=True)
subprocess.run(base + ["-c:v", "libx264", "-preset", "slow", "-b:v", "7700k", "-maxrate", "11000k", "-bufsize", "16000k", "-pass", "2", "-passlogfile", pl, "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", f"{VID}/loop_silent.mp4"], check=True)
if os.path.exists(wav):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{VID}/loop_silent.mp4", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", "76.8", "-movflags", "+faststart", out], check=True)
    print("muxed", out)
else: print("no music/loop_aligned.wav yet")

# description
seen = {}
for d, p in shots: seen.setdefault(d["c"]["accession_no"], d["c"])
L = ["Seamless 76.8 s loop (128 beats @ 100 bpm) of warm textures and details from the National Museum of Taiwan History collection.", "",
 "All images are released by the museum under CC0 1.0 (public domain dedication) — no attribution is required; credited here as a courtesy:",
 "https://creativecommons.org/publicdomain/zero/1.0/", "Source: https://collections.nmth.gov.tw/ (國立臺灣歷史博物館 / National Museum of Taiwan History)", ""]
for r_, c in seen.items(): L.append(f"- 「{c['title']}」({r_}) — {c['source_page']}")
L += ["", "Images are cropped, color-graded and animated for this video."]
open(f"{VID}/description.txt", "w").write("\n".join(L) + "\n")
