#!/bin/sh
# Compile video/tools/vis if needed and write video/work/an/*.json for every image in the config.
# Usage: video/analyze.sh [config.json] [--force]
cd "$(dirname "$0")/.." || exit 1
cfg=montage.json; extra=""
for a in "$@"; do case "$a" in --force) extra="--force";; *) cfg="$a";; esac; done
exec python3 video/build_loop.py --analyze $extra "$cfg"
