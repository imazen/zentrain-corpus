#!/usr/bin/env python3
"""Rebuild manifest.tsv from mlp-tune/ on disk.

Walks every PNG under mlp-tune/, captures sha256 + path + group +
content_class. Output is sorted by sha256 so re-runs produce identical
bytes (modulo ordering of os.walk, which is sorted explicitly).

Run after adding/removing images, or to verify the corpus hasn't drifted.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLP = ROOT / "mlp-tune"
OUT = ROOT / "manifest.tsv"

# group -> (content_class, source_label, license_id)
GROUPS = {
    "cid22-train": ("photo", "CID22-512px-train", "CC-BY-SA-4.0"),
    "clic-train": ("photo", "clic2025-native-train", "Unsplash"),
    "clic-1024-train": ("photo", "clic2025-1024px-resized-train", "Unsplash"),
    "kadid10k": ("mixed", "kadid10k-distortion-set", "Pixabay"),
    "gb82-photo": ("photo", "gb82-photos", "CC0-1.0"),
    "gb82-screen": ("screenshot", "gb82-screens", "CC0-1.0"),
    "size-dense-renders": ("photo", "zenwebp-size-dense-lanczos3", "CC-BY-SA-4.0"),
    "synthetic": ("synthetic", "zentrain-procedural", "CC0-1.0"),
}


def main() -> None:
    rows = []
    for group, (cc, src, lic) in GROUPS.items():
        gdir = MLP / group
        if not gdir.is_dir():
            print(f"WARN: missing group dir {gdir}")
            continue
        for p in sorted(gdir.rglob("*.png")):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            rel = p.relative_to(ROOT).as_posix()
            rows.append((sha, group, cc, src, lic, p.stat().st_size, rel))

    rows.sort(key=lambda r: r[0])
    with OUT.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sha256", "group", "content_class", "source", "license", "size_bytes", "path"])
        for r in rows:
            w.writerow(r)

    print(f"wrote {OUT} with {len(rows)} entries")
    by_group = {}
    for r in rows:
        by_group[r[1]] = by_group.get(r[1], 0) + 1
    for g, n in sorted(by_group.items()):
        print(f"  {g}: {n}")


if __name__ == "__main__":
    main()
