"""Fetch a small public photogrammetry dataset for end-to-end testing.

Default: the 11-image "Sceaux Castle" set (openMVG, used widely as a minimal real SfM
benchmark). Real Canon EOS images with genuine 3D structure — small enough to reconstruct
on CPU in a few minutes. No EXIF GPS, so the georeference stage falls back to scale/GCP
(exercising that path too).

Usage: python scripts/fetch_sample.py            # -> samples/sceaux/images/
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/master/images"
IMAGES = [f"100_71{n:02d}.JPG" for n in range(0, 11)]  # 100_7100 .. 100_7110
DEST = Path(__file__).resolve().parent.parent / "samples" / "sceaux" / "images"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(IMAGES, 1):
        out = DEST / name
        if out.exists() and out.stat().st_size > 0:
            print(f"[{i}/{len(IMAGES)}] cached {name}")
            continue
        url = f"{REPO}/{name}"
        print(f"[{i}/{len(IMAGES)}] downloading {name} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as r, out.open("wb") as f:
                f.write(r.read())
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}", file=sys.stderr)
            return 1
    total = sum(p.stat().st_size for p in DEST.glob("*.JPG"))
    print(f"done: {len(list(DEST.glob('*.JPG')))} images, {total/1e6:.1f} MB in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
