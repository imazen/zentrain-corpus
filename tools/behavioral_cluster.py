"""Behavioral clustering across all 4 zen codecs.

For each (image, size_class) compute a behavioral signature: a vector
of best-config-id at each q (or zensim bucket for jxl) in a shared grid.
Two pairs are "behaviorally identical" if their signatures match within
a Hamming threshold; we keep one representative per cluster.

Cross-codec union: the set of (image, size_class) that any codec needed
to keep.  Sweep set = union -> mlp-tune-fast/.

This script is deliberately self-contained: read TSVs, write per-codec
subsets, write cross-codec union, write per_image_reasons.tsv.
"""
from __future__ import annotations
import csv
import sys
import os
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Codec configuration
# ---------------------------------------------------------------------------

ZEN = Path("/home/lilith/work/zen")

CODECS = {
    "zenwebp_v01": {
        "tsv": ZEN / "zenwebp/benchmarks/zenwebp_pareto_2026-04-30_combined.tsv",
        "q_col": "q",
        "metric_col": "zensim",
        "size_class_col": "size_class",
        "image_col": "image_path",
        "bytes_col": "bytes",
        "config_col": "config_id",
        # Shared discrete q grid for all q-based codecs (10..90 step 5)
        "q_grid": list(range(10, 95, 5)),
    },
    "zenwebp": {
        "tsv": ZEN / "zenwebp/benchmarks/zenwebp_pareto_2026-05-01_combined.tsv",
        "q_col": "q",
        "metric_col": "zensim",
        "size_class_col": "size_class",
        "image_col": "image_path",
        "bytes_col": "bytes",
        "config_col": "config_id",
        "q_grid": list(range(10, 95, 5)),
    },
    "zenjpeg": {
        "tsv": ZEN / "zenjpeg/benchmarks/zq_pareto_2026-04-29.tsv",
        "q_col": "q",
        "metric_col": "zensim",
        "size_class_col": "size_class",
        "image_col": "image_path",
        "bytes_col": "bytes",
        "config_col": "config_id",
        "q_grid": list(range(10, 95, 5)),
    },
    "zenavif": {
        "tsv": ZEN / "zenavif/benchmarks/rav1e_phase1a_2026-04-30.tsv",
        "q_col": "q",
        "metric_col": "zensim",
        "size_class_col": "size_class",
        "image_col": "image_path",
        "bytes_col": "bytes",
        "config_col": "config_id",
        "q_grid": list(range(10, 95, 5)),
    },
    "zenjxl": {
        "tsv": ZEN / "zenjxl/benchmarks/zenjxl_lossy_pareto_2026-05-01.tsv",
        # No q column -> bucket by zensim score instead
        "q_col": None,
        "metric_col": "zensim",
        "size_class_col": "size_class",
        # image_path column has "sha:HEX" form; we use it as-is
        "image_col": "image_path",
        "bytes_col": "bytes",
        "config_col": "config_id",
        # zensim grid (typical zensim 30..95 range): bucketize achieved zensim
        # rounded down to nearest 5, restricted to 30..90
        "zensim_grid": list(range(30, 95, 5)),
    },
}

THRESHOLDS = [0, 1, 2]
PRIMARY_THRESHOLD = 1  # we keep representatives at this threshold

OUT_DIR = Path("/home/lilith/work/zentrain-corpus/mlp-tune-fast")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_q_codec(cfg) -> dict:
    """Returns dict[(image, size_class)] -> dict[q -> (bytes_min, config_id)]"""
    tsv = cfg["tsv"]
    print(f"  loading {tsv.name} ({tsv.stat().st_size//1024//1024} MB)...", flush=True)
    out: dict[tuple, dict[int, tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: (10**18, -1))
    )
    n = 0
    skipped = 0
    t0 = time.time()
    with tsv.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n += 1
            try:
                img = row[cfg["image_col"]]
                sc = row[cfg["size_class_col"]]
                q = int(float(row[cfg["q_col"]]))
                b = int(row[cfg["bytes_col"]])
                cid = int(row[cfg["config_col"]])
            except (ValueError, KeyError):
                skipped += 1
                continue
            cur = out[(img, sc)][q]
            if b < cur[0]:
                out[(img, sc)][q] = (b, cid)
            if n % 2_000_000 == 0:
                dt = time.time() - t0
                print(f"    {n//1000}K rows ({dt:.1f}s)...", flush=True)
    dt = time.time() - t0
    print(f"    done: {n} rows, {len(out)} pairs, skipped={skipped} ({dt:.1f}s)", flush=True)
    return dict(out)


def load_zensim_codec(cfg) -> dict:
    """zenjxl-style: bucket each row by floor(zensim/5)*5; keep min-bytes config per bucket."""
    tsv = cfg["tsv"]
    print(f"  loading {tsv.name} ({tsv.stat().st_size//1024//1024} MB)...", flush=True)
    out: dict[tuple, dict[int, tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: (10**18, -1))
    )
    n = 0
    skipped = 0
    t0 = time.time()
    with tsv.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n += 1
            try:
                img = row[cfg["image_col"]]
                sc = row[cfg["size_class_col"]]
                z = float(row[cfg["metric_col"]])
                b = int(row[cfg["bytes_col"]])
                cid = int(row[cfg["config_col"]])
            except (ValueError, KeyError):
                skipped += 1
                continue
            if z != z:  # NaN
                continue
            # bucket: nearest 5 in grid range
            bucket = max(30, min(90, int(z // 5) * 5))
            cur = out[(img, sc)][bucket]
            if b < cur[0]:
                out[(img, sc)][bucket] = (b, cid)
            if n % 2_000_000 == 0:
                dt = time.time() - t0
                print(f"    {n//1000}K rows ({dt:.1f}s)...", flush=True)
    dt = time.time() - t0
    print(f"    done: {n} rows, {len(out)} pairs, skipped={skipped} ({dt:.1f}s)", flush=True)
    return dict(out)


def signatures(per_pair: dict, grid: list[int]):
    sigs = {}
    for key, qmap in per_pair.items():
        sig = tuple(qmap.get(q, (10**18, -1))[1] for q in grid)
        if all(c == -1 for c in sig):
            continue
        sigs[key] = sig
    return sigs


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y and x != -1 and y != -1)


def cluster_within_sc(sigs: dict, thresh: int):
    by_sc = defaultdict(list)
    for k, s in sigs.items():
        by_sc[k[1]].append((k, s))
    clusters = []
    for sc, items in by_sc.items():
        used = [False] * len(items)
        for i in range(len(items)):
            if used[i]:
                continue
            grp = {items[i][0]}
            used[i] = True
            sig_i = items[i][1]
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                if hamming(sig_i, items[j][1]) <= thresh:
                    grp.add(items[j][0])
                    used[j] = True
            clusters.append(grp)
    return clusters


def pick_representatives(clusters: list, per_pair: dict, grid: list[int]):
    keep = set()
    mid_q = grid[len(grid) // 2]  # mid-range q point as reference
    for grp in clusters:
        best = None
        best_b = 10**18
        for key in grp:
            qmap = per_pair[key]
            ref = qmap.get(mid_q)
            if ref is None or ref[1] == -1:
                # fall back to any present q
                for q in grid:
                    cand = qmap.get(q)
                    if cand and cand[1] != -1:
                        ref = cand
                        break
            if ref is None or ref[1] == -1:
                continue
            if ref[0] < best_b:
                best_b = ref[0]
                best = key
        if best is not None:
            keep.add(best)
    return keep


def report_codec(name: str, sigs: dict, per_pair: dict, grid: list[int], log):
    log(f"\n=== {name} ===")
    log(f"  (image, size_class) pairs: {len(sigs)}")
    by_sc_total = defaultdict(int)
    for k in sigs:
        by_sc_total[k[1]] += 1
    log(f"  per size_class total: {dict(by_sc_total)}")
    cluster_results = {}
    for thresh in THRESHOLDS:
        clusters = cluster_within_sc(sigs, thresh)
        keep = pick_representatives(clusters, per_pair, grid)
        cluster_results[thresh] = (clusters, keep)
        log(f"  Hamming<= {thresh}: clusters={len(clusters)} kept={len(keep)} "
            f"reduction={100*(1-len(keep)/len(sigs)):.1f}%")
    # detailed breakdown for primary threshold
    clusters, keep = cluster_results[PRIMARY_THRESHOLD]
    by_sc_keep = defaultdict(int)
    for k in keep:
        by_sc_keep[k[1]] += 1
    log(f"  primary threshold = {PRIMARY_THRESHOLD}, per size_class:")
    for sc in sorted(by_sc_total):
        if by_sc_total[sc]:
            drop = 100 * (1 - by_sc_keep[sc] / by_sc_total[sc])
            log(f"    {sc}: {by_sc_total[sc]} -> {by_sc_keep[sc]}  ({drop:.0f}% drop)")
    bigs = sorted(clusters, key=len, reverse=True)[:5]
    log(f"  largest clusters at H={PRIMARY_THRESHOLD} (top 5):")
    for grp in bigs:
        sample = list(grp)[:3]
        log(f"    size={len(grp)} sc={list(grp)[0][1]} sample={[Path(s[0]).name for s in sample]}")
    return cluster_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_lines: list[str] = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    per_codec_keep = {}
    per_codec_total = {}
    for name, cfg in CODECS.items():
        log(f"\n[{name}]")
        if cfg.get("q_col"):
            per_pair = load_q_codec(cfg)
            grid = cfg["q_grid"]
        else:
            per_pair = load_zensim_codec(cfg)
            grid = cfg["zensim_grid"]
        sigs = signatures(per_pair, grid)
        results = report_codec(name, sigs, per_pair, grid, log)
        keep = results[PRIMARY_THRESHOLD][1]
        per_codec_keep[name] = keep
        per_codec_total[name] = sigs

        # Write per-codec subset TSV
        out = OUT_DIR / f"{name}_subset.tsv"
        with out.open("w") as f:
            f.write("image_path\tsize_class\n")
            for img, sc in sorted(keep):
                f.write(f"{img}\t{sc}\n")
        log(f"  -> wrote {out} ({len(keep)} rows)")

    # ---- Cross-codec union ----
    log("\n=== Cross-codec union ===")
    union_pairs = set()
    for k in per_codec_keep.values():
        union_pairs.update(k)
    log(f"  union (image, size_class) pairs: {len(union_pairs)}")

    # Total observed
    total_pairs = set()
    for sigs in per_codec_total.values():
        total_pairs.update(sigs.keys())
    log(f"  total observed pairs across codecs: {len(total_pairs)}")
    if total_pairs:
        log(f"  reduction vs total observed: "
            f"{100*(1-len(union_pairs)/len(total_pairs)):.1f}%")

    # Per-image reasons
    per_image: dict[str, dict] = defaultdict(lambda: {"codecs": set(), "scs": set()})
    for codec, kept in per_codec_keep.items():
        for img, sc in kept:
            per_image[img]["codecs"].add(codec)
            per_image[img]["scs"].add(sc)

    log(f"  unique source images in union: {len(per_image)}")

    # union TSV
    union_tsv = OUT_DIR / "cross_codec_union.tsv"
    with union_tsv.open("w") as f:
        f.write("image_path\tsize_class\n")
        for img, sc in sorted(union_pairs):
            f.write(f"{img}\t{sc}\n")
    log(f"  -> wrote {union_tsv}")

    reasons_tsv = OUT_DIR / "per_image_reasons.tsv"
    with reasons_tsv.open("w") as f:
        f.write("image_path\tcodecs_keeping\tsize_classes_kept\n")
        for img, recs in sorted(per_image.items()):
            cs = ",".join(sorted(recs["codecs"]))
            ss = ",".join(sorted(recs["scs"]))
            f.write(f"{img}\t{cs}\t{ss}\n")
    log(f"  -> wrote {reasons_tsv}")

    # Save log
    log_path = OUT_DIR / "behavioral_cluster.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nlog saved -> {log_path}")


if __name__ == "__main__":
    main()
