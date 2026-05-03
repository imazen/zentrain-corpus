# mlp-tune-fast

A behavioral-clustered subset of `mlp-tune/` for fast-iteration sweeps.

`mlp-tune-fast/` is a strict subset of `mlp-tune/` chosen so that the next
codec-side Pareto sweep can run in ~5x fewer GPU-minutes while still
seeing the full Pareto-pick diversity of the corpus.

It is **not** a replacement for `mlp-tune/`. Final-quality picker training
should still run on the full `mlp-tune/` corpus. This subset is for
iterating on hyperparameters, sweep grids, and feature designs where the
4-6 hour vast.ai cycle is the bottleneck.

## How it was built

For each of the 4 zen lossy codecs (zenwebp v0.1, zenwebp v0.2, zenjpeg,
zenavif, zenjxl) we have a Pareto sweep TSV of bytes / quality / config
across many configurations and quality dials. From each TSV we computed:

1. **Per-(image, size_class) behavioral signature.** For q-based codecs
   (webp, jpeg, avif), the signature is a tuple of the smallest-bytes
   `config_id` at each q in {10, 15, ..., 90} that meets the metric
   threshold. For zenjxl (no `q` column) the signature buckets achieved
   `zensim` to the nearest 5 in [30, 90] and stores the smallest-bytes
   `config_id` per bucket.
2. **Greedy Hamming clustering within each `size_class`.** Two pairs whose
   signatures match within Hamming <=1 are collapsed into one cluster; we
   keep the smallest-bytes representative at the mid-grid q.
3. **Cross-codec union.** A pair survives if any codec's clustering
   kept it. Validation paths (`*/validation/*`, `*/final-test/*`) and
   out-of-corpus shas are filtered out: `mlp-tune-fast/` is constrained
   to be a strict subset of `mlp-tune/`.

Script: `tools/behavioral_cluster.py` (cluster) + `tools/populate_tune_fast.py`
(filter to mlp-tune-only and copy files).

## What's preserved

- **Pareto behavior at the discrete q grid.** Every (image, size_class)
  whose smallest-bytes-Pareto-config differs at >= 1 of the 17 grid
  points (or 13 zensim buckets for jxl) from every other pair in its
  size class is kept.
- **Cross-codec union.** A pair only needs to be useful to one of
  {webp v0.1, webp v0.2, jpeg, avif, jxl} to be kept.
- **All `cid22-train`, `clic-train`, `gb82-photo`, `gb82-screen`** —
  these groups are kept in full because every image had distinct
  Pareto behavior in at least one codec.

## What's lost

- **Smooth interpolation between q targets** — the clustering quantizes
  on the {10..90 step 5} grid; pairs that diverge only between adjacent
  q points get collapsed.
- **Pairs that have identical Pareto behavior** to another pair in the
  same `size_class`. The representative is the smallest-bytes one at
  the mid-grid q (=50), so the kept image tends to be the
  "compresses-easiest" of its cluster.
- **All 60 `synthetic` images** — they were not present in any of the
  4 codec Pareto TSVs (sweep didn't run on them).
- **46 of 81 `kadid10k`** images, **5 of 264 `size-dense-renders`**,
  and **15 of 32 `clic-1024-train`** — these collapsed into other
  representatives.

## Final size

| metric | mlp-tune | mlp-tune-fast | reduction |
|---|---|---|---|
| images | 713 | 587 | 17.7% |
| MB | 313 | 246 | 21.4% |
| (image, size_class) pairs | 3296 (observed) | 1826 | **44.6%** |

The (image, size_class) reduction is what matters for sweep cost — 1826
pairs is what the next sweep iteration encodes.

## Per-codec contribution

| codec | pairs kept (at H=1) |
|---|---|
| zenwebp v0.1 | 660 |
| zenwebp v0.2 | 783 |
| zenjpeg | 494 |
| zenavif | 211 |
| zenjxl | 368 |

Sum > 1826 because pairs are shared across codecs (one (image, sc) can
be kept by multiple codecs).

## Files

- `manifest-fast.tsv` (in repo root) — sha256 + group + path index.
- `cross_codec_union.tsv` — raw union of all per-codec kept pairs
  (includes validation and out-of-corpus paths). Reference only.
- `cross_codec_union_filtered.tsv` — union restricted to mlp-tune
  (`path`, `size_class`, `original_image_path`, `sha256`). **This is
  what sweep tooling reads.**
- `per_image_reasons_filtered.tsv` — per-image: which codecs care, which
  size classes. Useful for stratified sweeps that cover only the
  specific size classes each image is needed for.
- `dropped_paths.tsv` — pairs that resolved out of corpus (validation /
  jxl-outside).
- `<codec>_subset.tsv` — per-codec kept (image_path, size_class) before
  cross-codec union, for codec-specific sweeps.
- `behavioral_cluster.log` — full cluster sizes, per-threshold stats,
  per size_class drop rates from the cluster run.

## When to use vs full mlp-tune/

| scenario | corpus |
|---|---|
| Final picker training | `mlp-tune/` |
| Sweeping codec hyperparameters | `mlp-tune-fast/` |
| Iterating picker architecture | `mlp-tune-fast/` |
| Calibration tables that ship as `const`s | `mlp-tune/` (full) |
| Validation / held-out eval | `mlp-validate/` (always) |

The contract is identical: tools that load `mlp-tune-fast/*` must refuse
to load anything under `mlp-validate/`. The split is enforced by
filesystem (no shared shas — verified at populate time).
