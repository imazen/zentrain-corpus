#!/usr/bin/env python3
"""Pre-render the synthetic training corpus deterministically.

Run once; commit the output PNGs. Training NEVER runs this script — it
exists only so a maintainer can regenerate identical bytes if the
corpus ever needs to be rebuilt.

All randomness is fully seeded (numpy RandomState) so re-running on a
clean checkout produces byte-identical PNGs (modulo PIL/numpy version).

Output layout:
  mlp-tune/synthetic/
    gradient_h_{w}x{h}.png        — horizontal luma ramp
    gradient_v_{w}x{h}.png        — vertical luma ramp
    gradient_diag_{w}x{h}.png     — diagonal luma ramp
    gradient_rgb_{w}x{h}.png      — three-channel ramps offset 120°
    checker_{cell}_{size}x{size}.png
    noise_uniform_seed{s}_{size}x{size}.png
    noise_gaussian_seed{s}_{size}x{size}.png
    hue_sweep_{size}x{size}.png   — HSV wheel, V=1.0
    color_stripes_{w}x{h}.png     — vertical RGB stripes
    thin_lines_{spacing}_{w}x{h}.png — black-on-white lines (high-frequency)
    solid_white_{w}x{h}.png       — flat 255
    solid_black_{w}x{h}.png       — flat 0
    solid_gray_{w}x{h}.png        — flat 128
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "mlp-tune" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [256, 512, 1024]


def save(arr: np.ndarray, name: str) -> None:
    Image.fromarray(arr).save(OUT / name, optimize=True)


def gradient_h(w: int, h: int) -> np.ndarray:
    g = np.linspace(0, 255, w, dtype=np.uint8)
    return np.broadcast_to(g, (h, w)).copy()


def gradient_v(w: int, h: int) -> np.ndarray:
    g = np.linspace(0, 255, h, dtype=np.uint8).reshape(h, 1)
    return np.broadcast_to(g, (h, w)).copy()


def gradient_diag(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = ((xx + yy) / (w + h - 2) * 255).astype(np.uint8)
    return g


def gradient_rgb(w: int, h: int) -> np.ndarray:
    """Three offset luma ramps so each channel hits a different range."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx / (w - 1)) * 255
    g = (yy / (h - 1)) * 255
    b = ((xx + yy) / (w + h - 2)) * 255
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def checker(cell: int, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    pat = ((xx // cell) + (yy // cell)) & 1
    return (pat * 255).astype(np.uint8)


def noise_uniform(seed: int, size: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)


def noise_gaussian(seed: int, size: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    arr = rng.normal(loc=128, scale=40, size=(size, size, 3))
    return arr.clip(0, 255).astype(np.uint8)


def hue_sweep(size: int) -> np.ndarray:
    """HSV wheel: hue from angle, saturation from radius, V=1.0."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # 0..1
    s = np.clip(r / (size / 2.0), 0, 1)
    v = np.ones_like(s)
    h6 = theta * 6.0
    c = v * s
    x = c * (1 - np.abs((h6 % 2) - 1))
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    masks = [(h6 < 1), (h6 < 2), (h6 < 3), (h6 < 4), (h6 < 5), (h6 < 6)]
    chans = [
        (c, x, np.zeros_like(x)),
        (x, c, np.zeros_like(x)),
        (np.zeros_like(x), c, x),
        (np.zeros_like(x), x, c),
        (x, np.zeros_like(x), c),
        (c, np.zeros_like(x), x),
    ]
    used = np.zeros((size, size), dtype=bool)
    for m, (rr, gg, bb) in zip(masks, chans):
        m = m & ~used
        rgb[..., 0] = np.where(m, rr, rgb[..., 0])
        rgb[..., 1] = np.where(m, gg, rgb[..., 1])
        rgb[..., 2] = np.where(m, bb, rgb[..., 2])
        used = used | m
    m = v - c
    rgb += m[..., None]
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def color_stripes(w: int, h: int) -> np.ndarray:
    """Six vertical RGB stripes covering the canvas."""
    palette = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )
    stripe_w = max(1, w // len(palette))
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        arr[:, i, :] = palette[min(i // stripe_w, len(palette) - 1)]
    return arr


def thin_lines(spacing: int, w: int, h: int) -> np.ndarray:
    """Black-on-white vertical lines — pure high-frequency content."""
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[:, ::spacing] = 0
    return arr


def solid(value: int, w: int, h: int) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def main() -> None:
    print(f"writing to {OUT}")

    # Gradients — luma + rgb at each size
    for s in SIZES:
        save(gradient_h(s, s), f"gradient_h_{s}x{s}.png")
        save(gradient_v(s, s), f"gradient_v_{s}x{s}.png")
        save(gradient_diag(s, s), f"gradient_diag_{s}x{s}.png")
        save(gradient_rgb(s, s), f"gradient_rgb_{s}x{s}.png")

    # Checkerboards
    for s in SIZES:
        for cell in (4, 16, 64):
            save(checker(cell, s), f"checker_{cell}_{s}x{s}.png")

    # Noise — three seeds × two distributions × three sizes
    for seed in (0xC0DE, 0xBEEF, 0xFACE):
        for s in SIZES:
            save(noise_uniform(seed, s), f"noise_uniform_seed{seed:04x}_{s}x{s}.png")
            save(noise_gaussian(seed, s), f"noise_gaussian_seed{seed:04x}_{s}x{s}.png")

    # Hue sweep
    for s in SIZES:
        save(hue_sweep(s), f"hue_sweep_{s}x{s}.png")

    # Stripes + thin lines
    for s in SIZES:
        save(color_stripes(s, s), f"color_stripes_{s}x{s}.png")
        for sp in (2, 4, 8):
            save(thin_lines(sp, s, s), f"thin_lines_sp{sp}_{s}x{s}.png")

    # Solids — small set
    for s in (256, 512):
        save(solid(255, s, s), f"solid_white_{s}x{s}.png")
        save(solid(0, s, s), f"solid_black_{s}x{s}.png")
        save(solid(128, s, s), f"solid_gray_{s}x{s}.png")

    n = len(list(OUT.glob("*.png")))
    print(f"wrote {n} synthetic images")


if __name__ == "__main__":
    main()
