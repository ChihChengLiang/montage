#!/usr/bin/env python3
"""Seamless beat-grid vibe loop from a folder of images: saliency-aware crops, one shared warm grade, hard cuts on the beat.

Usage:
  python3 video/build_loop.py [config.json]            render (default config: ./montage.json at the repo root)
  python3 video/build_loop.py --analyze [config.json]  compile video/tools/vis (macOS Vision) and write per-image analysis json
  python3 video/build_loop.py --check   [config.json]  validate inputs and print the shot plan; renders nothing

Config (JSON; relative paths are relative to the config file; see montage.json):
  images_dir, credits_file, image_glob, drop[], manual_safe_rects{key:[x0,y0,x1,y1] fractions}, license_filter,
  width, height, fps, bpm (cut grid), loop_beats, shot_cycle_beats[], dissolve{min_beats,frames}, push_in,
  crop{min_width,hero_min_frac,detail_width_frac,texture_width_mult,roles,extra_roles{key:[roles]}},
  grade{...}, x264{...}, audio, out_dir, work_dir, out_name, seed, description{lines[], credit_line}.
An image "key" is its file name up to the first "_" (e.g. c01_xxx.jpg -> c01).
credits_file: JSON list of objects with fields file, title, accession_no, license, source_page (any others are ignored);
`file` matches the image file name; license_filter keeps entries whose license starts with that string ("" = all).
Needs: numpy, opencv-python, pillow, ffmpeg/ffprobe; macOS + swiftc for the Vision analysis step.
"""
import json, os, subprocess, sys, glob, numpy as np, cv2
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]; flags = {a for a in sys.argv[1:] if a.startswith("--")}
CFG_PATH = os.path.abspath(args[0] if args else os.path.join(os.path.dirname(HERE), "montage.json"))
CFG = json.load(open(CFG_PATH)); BASE = os.path.dirname(CFG_PATH)
def P(x): return x if os.path.isabs(x) else os.path.join(BASE, x)
IMG, WK, OUT_DIR = P(CFG["images_dir"]), P(CFG["work_dir"]), P(CFG["out_dir"])
WAV = P(CFG["audio"]) if CFG.get("audio") else None
W, H, FPS = CFG["width"], CFG["height"], CFG["fps"]
BEAT_S = CFG["beat_seconds"] if CFG.get("beat_seconds") else 60 / CFG["bpm"]   # optional exact beat period (non-integer frame grids)
BEAT = FPS * BEAT_S                                       # frames per beat (18.0 @ 30 fps / 100 bpm; may be fractional)
TOTAL = round(CFG["loop_beats"] * BEAT)                   # cut boundaries = round(cumulative beats * BEAT)
if CFG.get("total_frames"): assert TOTAL == CFG["total_frames"], f"loop_beats*beat = {CFG['loop_beats']*BEAT:.3f} frames, expected {CFG['total_frames']}"
LOOP_SECONDS = TOTAL / FPS
def fb(beats): return round(beats * BEAT)
GR, CR, X264 = CFG["grade"], CFG["crop"], CFG["x264"]
DROP = set(CFG.get("drop", []))
MANUAL = {k: tuple(v) for k, v in CFG.get("manual_safe_rects", {}).items()}
MIN_W = CR["min_width"]
os.makedirs(f"{WK}/mid", exist_ok=True); os.makedirs(f"{WK}/an", exist_ok=True)

def image_files():
    fs = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(IMG, CFG["image_glob"])))]
    return [f for f in fs if f.split("_")[0] not in DROP]

if "--analyze" in flags:
    vis = f"{HERE}/tools/vis"; src = f"{HERE}/tools/vis.swift"
    if not os.path.exists(vis) or os.path.getmtime(vis) < os.path.getmtime(src):
        subprocess.run(["swiftc", "-O", src, "-o", vis], check=True)
    for f in image_files():
        o = f"{WK}/an/{f[:-4]}.json"
        if os.path.exists(o) and "--force" not in flags: continue
        with open(o, "w") as fh: subprocess.run([vis, f"{IMG}/{f}"], stdout=fh, check=True)
        print("analysed", f, flush=True)
    sys.exit(0)

rng = np.random.default_rng(CFG["seed"])
_cred_all = json.load(open(P(CFG["credits_file"])))
cred = {c["file"]: c for c in _cred_all if c["license"].startswith(CFG.get("license_filter", ""))}

def load(file):
    key = file.split("_")[0]; an = json.load(open(f"{WK}/an/{file[:-4]}.json"))
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
    Fb = cv2.GaussianBlur(F, (0, 0), 3); py, px = np.unravel_index(int(Fb.argmax()), Fb.shape); peak = (float(px * s), float(py * s))
    # white balance from backdrop (low-saliency pixels inside safe rect); object-filling images: gentle gray-world
    mask = np.zeros((gh, gw), bool); mask[int(sy0 / s):int(sy1 / s), int(sx0 / s):int(sx1 / s)] = True
    bgm = mask & (F < 0.05)
    if bgm.sum() > 0.12 * mask.sum() and key not in MANUAL:
        ref = np.median(sm[bgm], 0); gains = ref.mean() / ref
    else:
        m = sm[mask].mean(0); gains = 1 + 0.5 * (m.mean() / m - 1)
    gains = np.clip(gains, 0.8, 1.3).astype(np.float32)
    bgl = float((np.median(sm[bgm], 0) * gains * [0.299, 0.587, 0.114]).sum() / 255) if (bgm.sum() > 0.12 * mask.sum() and key not in MANUAL) else None
    return dict(file=file, key=key, bg=bgl, im=im, w=w, h=h, safe=safe, G=G, F=F, s=s, fb=fb, focal=focal, peak=peak, gains=gains, c=cred[file])

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
            cw = max(CR["hero_min_frac"] * d["w"], (fb[2]-fb[0]) * 1.2, (fb[3]-fb[1]) * 16/9 * 1.2)
            if CR.get("hero_max_frac"): cw = min(cw, CR["hero_max_frac"] * d["w"])       # optional cap (illustrations: tighter hero)
            cw = min(cw, x1 - x0, (y1 - y0) * 16 / 9)
            if CR.get("hero_center") == "focal": fcx, fcy = fx, fy
            elif CR.get("hero_center") == "peak": fcx, fcy = d["peak"]; tx, ty = d["peak"]                        # optional: centre on the saliency centroid, not the bbox
            cx, cy = fcx + 0.06 * cw, fcy                      # slightly off-centre (thirds-ish), push toward the focal point
            tx, ty = fx, fy
        elif role == "wide":
            cw = min(x1 - x0, (y1 - y0) * 16 / 9); cx, cy = fcx, fcy; tx, ty = fx, fy
        else:
            cw = max(MIN_W, CR["detail_width_frac"] * d["w"]) if role == "detail" else MIN_W * CR["texture_width_mult"]
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

files = image_files()
imgs = [load(f) for f in files]
per = [list(CR["roles"]) + list(CR.get("extra_roles", {}).get(d["key"], [])) for d in imgs]
shots = []
for d, roles in zip(imgs, per):
    for p in plan(d, roles): shots.append((d, p))
print("images", len(imgs), "crops", len(shots), [(d["key"], sum(1 for x,_ in shots if x is d)) for d in imgs])
# shot lengths: cycle CFG["shot_cycle_beats"] until loop_beats is filled exactly; last shot = first shot reversed (seamless)
pattern, tot, i = [], 0, 0
cyc = CFG["shot_cycle_beats"]
while tot < CFG["loop_beats"]:
    pattern.append(cyc[i % len(cyc)]); tot += pattern[-1]; i += 1
assert tot == CFG["loop_beats"], f"shot_cycle_beats {cyc} cannot fill {CFG['loop_beats']} beats exactly (got {tot})"
need = len(pattern) - 1
assert len(shots) >= need, f"{len(shots)} crops available, {need} needed: add images or roles"
while len(shots) > need:                                  # trim extra crops, 'wide' ones first
    j = next((k for k in range(len(shots)-1, -1, -1) if shots[k][1]["role"] == "wide"), None)
    if j is None: break
    shots.pop(j)
for role in ("texture", "detail"):                        # then drop evenly spaced texture, then detail crops
    ex = len(shots) - need
    if ex <= 0: break
    idx = [k for k, (_, p_) in enumerate(shots) if p_["role"] == role]
    drop = {idx[int((m + 0.5) * len(idx) / min(ex, len(idx)))] for m in range(min(ex, len(idx)))}
    shots = [x for k, x in enumerate(shots) if k not in drop]
while len(shots) > need: shots.pop()
shots.append(shots[0])
assert len(pattern) == len(shots) and sum(pattern) == CFG["loop_beats"]

if "--check" in flags:
    ok = True
    def bad(m):
        global ok; ok = False; print("PROBLEM:", m)
    for f in files:
        if not os.path.exists(f"{IMG}/{f}"): bad(f"missing image {f}")
        if not os.path.exists(f"{WK}/an/{f[:-4]}.json"): bad(f"missing analysis {f[:-4]}.json (run --analyze)")
        if f not in cred: bad(f"{f} not in credits (or filtered by license_filter)")
    used = {d["file"] for d, _ in shots}
    for f in used:
        c = cred.get(f, {})
        for k in ("title", "accession_no", "license", "source_page"):
            if k not in c: bad(f"credits for {f} lack field {k}")
    if WAV is None or not os.path.exists(WAV): print("NOTE: audio missing:", WAV)
    else:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", WAV], capture_output=True, text=True).stdout)
        if abs(dur - LOOP_SECONDS) > 1 / FPS: bad(f"audio is {dur:.3f}s, video loop is {LOOP_SECONDS:.3f}s")
    t = 0; cb = 0
    for i, ((d, p), b) in enumerate(zip(shots, pattern)):
        print(f"shot {i:2d}  {d['key']}  {p['role']:7s} {b} beats  {fb(cb+b)-fb(cb)} frames  t={t/FPS:6.2f}s  crop {int(p['cw'])}px of {d['w']}x{d['h']}"); t += fb(cb+b) - fb(cb); cb += b
    if "--preview" in flags:                              # draw planned crops + safe rects on each image -> work/plan_preview.jpg
        ims = []
        for d in imgs:
            tt = d["im"].copy(); sc = 480 / max(tt.size); tt = tt.resize((int(tt.size[0]*sc), int(tt.size[1]*sc))); dr = ImageDraw.Draw(tt)
            dr.rectangle([v * sc for v in d["safe"]], outline=(0, 160, 255), width=2)
            for dd, p in shots[:-1]:
                if dd is d:
                    ch = p["cw"] * 9 / 16; col = {"hero": (255, 0, 0), "detail": (0, 200, 0), "texture": (255, 160, 0), "wide": (160, 0, 200)}[p["role"]]
                    dr.rectangle([(p["cx"]-p["cw"]/2)*sc, (p["cy"]-ch/2)*sc, (p["cx"]+p["cw"]/2)*sc, (p["cy"]+ch/2)*sc], outline=col, width=3)
            ims.append(tt)
        cols_ = 6; rows_ = (len(ims) + cols_ - 1) // cols_; Sx = Image.new("RGB", (cols_ * 490, rows_ * 490), "white")
        for i_, tt in enumerate(ims): Sx.paste(tt, ((i_ % cols_) * 490, (i_ // cols_) * 490))
        Sx.save(f"{WK}/plan_preview.jpg", quality=80); print("wrote", f"{WK}/plan_preview.jpg")
    print(f"images {len(files)}  shots {len(shots)}  frames {t} ({t/FPS:.3f}s)  beat={BEAT:g}f  {'OK' if ok and t == TOTAL else 'CHECK FAILED'}")
    sys.exit(0 if ok and t == TOTAL else 1)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.hypot((xx - W/2) / (W/2), (yy - H/2) / (H/2))
VIG = (1 - GR["vignette"] * np.clip(r - GR["vignette_start"], 0, 1) ** 1.6)[..., None].astype(np.float32)
WARM = np.array(GR["warm"], np.float32)                  # amber/honey highlights
LIFT = np.array(GR["lift"], np.float32)                  # slight brown lifted blacks
KNEE, TOP = GR["knee"], GR["top"]                        # soft highlight limiter (max ~ TOP*255)

def grade(a, gain, expo):
    a = a / 255.0 * gain * expo
    l = (a * [0.299, 0.587, 0.114]).sum(2, keepdims=True)
    cool = np.clip((np.maximum(a[..., 1], a[..., 2]) - a[..., 0])[..., None] * 3.0, 0, 1)   # blues/greens
    a = l + (a - l) * (GR["saturation"] - GR["cool_desat"] * cool)
    a = a * (1 + (WARM - 1) * (0.5 + 0.5 * np.clip(l, 0, 1)))   # warm, stronger in highlights
    a = np.clip(a, 0, 1.3) ** GR["gamma"]                        # midtone brightening
    sm_ = np.clip(a, 0, 1); a = a * (1 - GR["scurve"]) + GR["scurve"] * (sm_ * sm_ * (3 - 2 * sm_))   # gentle S-curve
    a = a + LIFT * (1 - np.clip(l, 0, 1)) ** 2                   # brown-tinted shadows
    a = a * VIG
    hi = a > KNEE                                                # soft knee: no clipping to 255
    a = np.where(hi, KNEE + (TOP - KNEE) * (1 - np.exp(-(a - KNEE) / (TOP - KNEE))), a)
    a = a + rng.normal(0, GR["grain"], (H, W, 1)).astype(np.float32)   # light luma grain
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

inter = f"{WK}/inter.mp4"
enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-crf", str(X264["intermediate_crf"]), "-preset", "veryfast", "-pix_fmt", "yuv420p", inter], stdin=subprocess.PIPE)
prev = None; n = 0; meta = []; cbeat = 0
for si, ((d, p), beats) in enumerate(zip(shots, pattern)):
    nf = fb(cbeat + beats) - fb(cbeat); cbeat += beats; last = si == len(shots) - 1; im = d["im"]
    cw0, ch0 = p["cw"], p["cw"] * 9 / 16
    # exposure normalise per shot from first-frame crop
    t = im.crop((p["cx"]-cw0/2, p["cy"]-ch0/2, p["cx"]+cw0/2, p["cy"]+ch0/2)).resize((192, 108))
    L = (np.asarray(t, np.float32) / 255 * d["gains"] * [0.299, 0.587, 0.114]).sum(2)
    p98 = float(np.percentile(L, 98))
    obj = np.abs(L - d["bg"]) > GR["object_mask_delta"] if d["bg"] is not None else np.ones_like(L, bool)   # object mask = not-backdrop
    if obj.mean() > 0.05:
        ml = float(L[obj].mean()); expo = (GR["object_target"] / max(ml, 0.03)) ** 0.6 if d["bg"] is not None else (GR["fill_target"] / max(ml, 0.03)) ** 0.8
    else: expo = 1.0
    expo = float(np.clip(expo, GR["expo_min"], GR["expo_max"])); expo = min(expo, GR["p98_cap"] / max(p98, 0.05))   # backdrop stays bright cream, no clipping
    for f in range(nf):
        u = f / (nf - 1); u = (1 - u) if last else u
        z = 1 - CFG["push_in"] * u
        k = 0.7 * u                                      # drift toward target as we push in
        cx = p["cx"] + (p["tx"] - p["cx"]) * k * 0.5; cy = p["cy"] + (p["ty"] - p["cy"]) * k * 0.5
        cw = cw0 * z; cx, cy = clampc(d, cx, cy, cw)
        ch = cw * 9 / 16
        fr = np.asarray(im.resize((W, H), Image.BICUBIC, box=(cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2)), dtype=np.float32)
        out = grade(fr, d["gains"], expo)
        if beats >= CFG["dissolve"]["min_beats"] and f < CFG["dissolve"]["frames"] and prev is not None:
            al = (f + 1) / (CFG["dissolve"]["frames"] + 1); out = (prev.astype(np.float32) * (1 - al) + out * al).astype(np.uint8)
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

# final: 2-pass capped bitrate (+ audio mux)
out = f"{OUT_DIR}/{CFG['out_name']}.mp4"; silent = f"{OUT_DIR}/{CFG['out_name']}_silent.mp4"
pl = f"{WK}/x264pass"
base = ["ffmpeg", "-y", "-v", "error", "-i", inter]
rate = ["-c:v", "libx264", "-preset", "slow", "-b:v", X264["bitrate"], "-maxrate", X264["maxrate"], "-bufsize", X264["bufsize"]]
subprocess.run(base + rate + ["-pass", "1", "-passlogfile", pl, "-an", "-f", "null", "/dev/null"], check=True)
subprocess.run(base + rate + ["-pass", "2", "-passlogfile", pl, "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", silent], check=True)
if WAV and os.path.exists(WAV):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", WAV, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", X264["audio_bitrate"], "-t", f"{LOOP_SECONDS:g}", "-movflags", "+faststart", out], check=True)
    print("muxed", out)
else: print("no audio at", WAV, "- silent version only")

# description
seen = {}
for d, p in shots: seen.setdefault(d["c"]["accession_no"], d["c"])
DS = CFG["description"]; L = []
for line in DS["lines"]:
    if line == "{credits}": L += [DS["credit_line"].format(**c) for c in seen.values()]
    else: L.append(line)
open(f"{OUT_DIR}/{CFG.get('description_file', 'description.txt')}", "w").write("\n".join(L) + "\n")
