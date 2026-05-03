#!/usr/bin/env python3
"""Populate mlp-tune-fast/ from cross_codec_union.tsv.

Filters union to entries whose images are present in mlp-tune/, copies
those PNGs into mlp-tune-fast/<group>/<filename>, and writes a manifest.

Also writes a filtered subset TSV (cross_codec_union_filtered.tsv)
and a per_image_reasons_filtered.tsv keyed by sha256 + mlp-tune path
so downstream training tools can join cleanly.

Validation images and out-of-corpus images are dropped with stats.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/lilith/work/zentrain-corpus")
TUNE = ROOT / "mlp-tune"
TUNE_FAST = ROOT / "mlp-tune-fast"
MANIFEST = ROOT / "manifest.tsv"
VAL_MANIFEST = ROOT / "validate-manifest.tsv"

UNION_TSV = TUNE_FAST / "cross_codec_union.tsv"
REASONS_TSV = TUNE_FAST / "per_image_reasons.tsv"

# Outputs
FILTERED_UNION = TUNE_FAST / "cross_codec_union_filtered.tsv"
FILTERED_REASONS = TUNE_FAST / "per_image_reasons_filtered.tsv"
DROPPED_TSV = TUNE_FAST / "dropped_paths.tsv"


def load_manifest():
    sha_to_path = {}  # sha -> repo-relative mlp-tune path
    bn_to_paths: dict[str, list[str]] = defaultdict(list)
    for row in csv.DictReader(open(MANIFEST), delimiter="\t"):
        sha_to_path[row["sha256"]] = row["path"]
        bn_to_paths[os.path.basename(row["path"])].append(row["path"])
    return sha_to_path, bn_to_paths


def load_validate_set():
    val_sha = set()
    val_bn = set()
    for row in csv.DictReader(open(VAL_MANIFEST), delimiter="\t"):
        val_sha.add(row["sha256"])
        val_bn.add(os.path.basename(row["path"]))
    return val_sha, val_bn


def resolve_path(image_path: str, sha_to_path: dict, bn_to_paths: dict) -> str | None:
    """Return the repo-relative mlp-tune/... path that this raw image_path
    corresponds to, or None if it cannot be resolved into mlp-tune."""
    if image_path.startswith("sha:"):
        prefix = image_path[4:]
        for sha, p in sha_to_path.items():
            if sha.startswith(prefix):
                return p
        return None

    bn = os.path.basename(image_path)
    candidates = bn_to_paths.get(bn, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # disambiguate by hashing the source on disk if it exists
        if os.path.exists(image_path):
            sha = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
            for c in candidates:
                full = ROOT / c
                if full.exists():
                    fsha = hashlib.sha256(full.read_bytes()).hexdigest()
                    if fsha == sha:
                        return c
        # fallback: first candidate
        return candidates[0]
    # unknown basename — try sha lookup if file exists
    if os.path.exists(image_path):
        sha = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
        if sha in sha_to_path:
            return sha_to_path[sha]
    return None


def main() -> None:
    sha_to_path, bn_to_paths = load_manifest()
    val_sha, val_bn = load_validate_set()

    # Reverse: rel_path -> sha (for emitting manifest rows)
    path_to_sha = {p: s for s, p in sha_to_path.items()}

    # --- Pass 1: filter union ---
    union_rows = list(csv.DictReader(open(UNION_TSV), delimiter="\t"))
    print(f"raw union rows: {len(union_rows)}")

    kept_pairs: list[tuple[str, str, str]] = []   # (rel_path, size_class, original_path)
    dropped: list[tuple[str, str, str]] = []       # (image_path, size_class, reason)

    kept_paths: set[str] = set()
    for r in union_rows:
        ip = r["image_path"]
        sc = r["size_class"]
        rp = resolve_path(ip, sha_to_path, bn_to_paths)
        if rp is None:
            # Why?
            bn = os.path.basename(ip)
            if "validation" in ip or "final-test" in ip or bn in val_bn:
                reason = "validation_path"
            elif ip.startswith("sha:"):
                reason = "zenjxl_sha_outside_corpus"
            else:
                reason = "unknown_basename"
            dropped.append((ip, sc, reason))
            continue
        # Sanity: must not be in validate manifest
        if path_to_sha.get(rp) in val_sha:
            dropped.append((ip, sc, "resolved_to_validate"))
            continue
        kept_pairs.append((rp, sc, ip))
        kept_paths.add(rp)

    # Write filtered union
    with FILTERED_UNION.open("w") as f:
        f.write("path\tsize_class\toriginal_image_path\tsha256\n")
        for rp, sc, ip in sorted(kept_pairs):
            sha = path_to_sha.get(rp, "")
            f.write(f"{rp}\t{sc}\t{ip}\t{sha}\n")

    with DROPPED_TSV.open("w") as f:
        f.write("image_path\tsize_class\treason\n")
        for ip, sc, reason in sorted(dropped):
            f.write(f"{ip}\t{sc}\t{reason}\n")

    # Filtered reasons
    src = list(csv.DictReader(open(REASONS_TSV), delimiter="\t"))
    with FILTERED_REASONS.open("w") as f:
        f.write("path\tsha256\tcodecs_keeping\tsize_classes_kept\toriginal_image_path\n")
        kept_set = set()
        for r in src:
            ip = r["image_path"]
            rp = resolve_path(ip, sha_to_path, bn_to_paths)
            if rp is None:
                continue
            if path_to_sha.get(rp) in val_sha:
                continue
            sha = path_to_sha.get(rp, "")
            f.write(f"{rp}\t{sha}\t{r['codecs_keeping']}\t{r['size_classes_kept']}\t{ip}\n")
            kept_set.add(rp)

    print(f"filtered kept (image, sc) pairs: {len(kept_pairs)}")
    print(f"unique kept image paths: {len(kept_paths)}")
    print(f"dropped pairs: {len(dropped)}")
    by_reason = defaultdict(int)
    for _, _, reason in dropped:
        by_reason[reason] += 1
    for k, v in sorted(by_reason.items()):
        print(f"  drop reason {k}: {v}")

    # --- Pass 2: copy files ---
    n_copied = 0
    n_skipped_existing = 0
    total_bytes = 0
    for rp in sorted(kept_paths):
        # rp is "mlp-tune/<group>/<file>.png"
        src = ROOT / rp
        if not src.exists():
            print(f"WARN: missing source {src}")
            continue
        # Replace top-level 'mlp-tune' with 'mlp-tune-fast'
        parts = rp.split("/")
        assert parts[0] == "mlp-tune", parts
        dst_rel = "mlp-tune-fast/" + "/".join(parts[1:])
        dst = ROOT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            n_skipped_existing += 1
        else:
            shutil.copy2(src, dst)
            n_copied += 1
        total_bytes += src.stat().st_size

    print(f"copied {n_copied}, skipped_existing {n_skipped_existing}, total bytes {total_bytes/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
