#!/usr/bin/env python3
"""Re-download the images listed in a credits.json (fields: file, image_url) that are missing locally.
Used for sets whose scans are not committed (e.g. images_gould/, from Wikimedia Commons).
Usage: python3 scripts/fetch_images.py images_gould/credits.json [--only N]"""
import json, os, sys, time, urllib.request

cred_path = sys.argv[1]
only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
outdir = os.path.dirname(os.path.abspath(cred_path))
UA = "montage-loop/1.0 (https://github.com/ChihChengLiang/montage)"
todo = [c for c in json.load(open(cred_path)) if c.get("image_url") and not os.path.exists(f"{outdir}/{c['file']}")]
for c in todo[:only]:
    req = urllib.request.Request(c["image_url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(f"{outdir}/{c['file']}.part", "wb") as f:
        f.write(r.read())
    os.replace(f"{outdir}/{c['file']}.part", f"{outdir}/{c['file']}")
    print("fetched", c["file"]); time.sleep(1.5)
print(f"{len(todo)} missing, done")
