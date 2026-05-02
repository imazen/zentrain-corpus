# zentrain-corpus

Frozen training corpus for the zen codec MLP/picker stack. Every image used
to fit a zenwebp/zenjpeg/zenavif/zenjxl picker (or analogous tuned head) is
committed here as bytes — no runtime generation, no symlinks into other
repos — so a checkout at any commit reproduces a specific oracle exactly.

Held-out validation lives in a separate sibling tree (`mlp-validate/`)
so the train/validate boundary is enforced at the filesystem level.
**Training tools must refuse to load any path under `mlp-validate/`** —
that's the contract of this repo. Codec-in-the-loop A/B harnesses do
the inverse: they load only `mlp-validate/` and never touch
`mlp-tune/`.

## Layout

```
mlp-tune/
├── cid22-train/          209 photos, 512×512, CC-BY-SA 4.0
├── clic-train/           32  photos, native res, Unsplash
├── clic-1024-train/      32  photos, 1024×1024 Lanczos resized, Unsplash
├── kadid10k/             81  mixed (KADID-10k distortion set), Pixabay
├── gb82-photo/           25  photos, CC0
├── gb82-screen/          10  screenshots, CC0
├── size-dense-renders/   264 photos, log-spaced sizes 32..1536px, CC-BY-SA 4.0
└── synthetic/            60  procedural (gradients/checker/noise/hue), CC0
                         ────
                         713 images, ~313 MB

mlp-validate/
├── cid22-val/            41  photos, 512×512, CC-BY-SA 4.0
├── clic-final-test/      30  photos, native res, Unsplash
└── clic-1024-final-test/ 30  photos, 1024×1024 Lanczos resamples of final-test, Unsplash
                         ────
                         101 images, ~177 MB
```

`manifest.tsv` (training) and `validate-manifest.tsv` (held-out) are
the canonical indexes — sha256, group, content class, source label,
license, file size, repo-relative path. Both sorted by sha256 for
deterministic regeneration.

## Conventions

- All images are PNG, 8-bit sRGB. No EXIF, no ICC profile beyond sRGB.
- Filenames are stable (never renamed after first commit).
- Adding images means adding a new group dir and re-running
  `tools/regenerate_manifest.py`. Don't mix sources within a group.
- Removing images is reserved for license/attribution corrections —
  open an issue first.

## Regeneration

The corpus is committed as bytes; nothing in this repo is regenerated at
training time. The two scripts below exist only for first-time setup or
maintenance:

```bash
# Pre-render the synthetic group (one-shot, deterministic)
python3 tools/render_synthetic.py

# Rebuild manifest.tsv after any group is modified
python3 tools/regenerate_manifest.py
```

`render_synthetic.py` uses fully seeded NumPy RNGs and PIL `optimize=True`
PNG compression, so re-running on a clean checkout produces byte-identical
PNGs (modulo `numpy`/`Pillow` version differences — pin both before any
regeneration).

## Per-codec splits

Stratified train/val/test splits live in `splits/<codec>_<version>.tsv`,
each referencing rows from `manifest.tsv` by sha256. Pickers consume these,
not the raw `mlp-tune/` tree. Splits are produced by codec-specific tools
in `~/work/zen/zenanalyze/zentrain/tools/` and are committed alongside the
trained picker artifact for reproducibility.

## Licensing

Per-image licensing is captured in `manifest.tsv`. Group-level license
files are in each subfolder. Aggregate attribution is in
`LICENSE-AGGREGATE.md`. The CC-BY-SA 4.0 obligation propagates to any
downstream picker artifact trained on this corpus — derivative works
must remain CC-BY-SA 4.0 with attribution to the listed sources.
