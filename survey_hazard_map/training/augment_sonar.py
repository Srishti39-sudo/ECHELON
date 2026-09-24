#!/usr/bin/env python3
"""Offline sonar augmentation for enlarging small pipe / cylinder sets.

    python training/augment_sonar.py --images datasets/sss/images/train \\
        --labels datasets/sss/labels/train --out datasets/sss_aug --copies 2

    from augment_sonar import speckle, gain_drift, resolution_jitter, paste_with_shadow

READ THIS FIRST: SYNTHETIC AUGMENTATION IS NOT REAL DATA
    Everything here re-renders examples you already have. Speckle, gain and
    resolution changes widen the nuisance variation around a real object, and
    that genuinely helps a small set generalise across sonars. Copy-paste
    places a real object crop on a real seabed, with a shadow drawn to agree
    with the new geometry, but the shadow is a model of one and the seam is
    not how the object and seabed actually interact acoustically (no
    scour, no burial, no multipath). A detector trained heavily on pastes can
    learn the paste. So:
      - augment train only; never put an augmented or pasted image in val;
      - keep pastes to a minority of training images;
      - report results on real, unaugmented, line-split validation data.

ALL FUNCTIONS
    take and return uint8 greyscale H x W arrays, take an explicit
    numpy Generator so a run is reproducible, and never move a pixel's
    position unless they also return the moved boxes (only paste_with_shadow
    adds a box; nothing here rotates or rescales the frame).

    Boxes are YOLO rows: (cls, cx, cy, w, h), normalised to the image.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def speckle(image: np.ndarray, rng: np.random.Generator, looks: float = 4.0) -> np.ndarray:
    """Multiplicative speckle: intensity * Gamma(looks, 1/looks), mean 1.

    `looks` is the equivalent number of looks: 1 is fully developed single-look
    speckle (very harsh), higher is smoother. Real side-scan imagery already
    has speckle, so this ADDS roughness; keep looks >= 3 unless the source is
    unusually smooth.
    """
    noise = rng.gamma(looks, 1.0 / looks, image.shape)
    return np.clip(image.astype(np.float32) * noise, 0, 255).astype(np.uint8)


def gain_drift(image: np.ndarray, rng: np.random.Generator, along: float = 0.15,
               across: float = 0.25, nadir_col: float | None = None) -> np.ndarray:
    """Slow gain changes: a random walk along track, and a TVG-like ramp across.

    along   peak relative gain wander between rows (0.15 = +/-15%)
    across  strength of an across-track ramp that brightens (or dims) with
            range from nadir; nadir is the image centre if not given
    """
    h, w = image.shape
    walk = np.cumsum(rng.normal(0, 1, h))
    walk = walk - walk.mean()
    walk = walk / (np.abs(walk).max() + 1e-9) * along
    rows = 1.0 + np.convolve(walk, np.ones(31) / 31, mode="same")
    centre = w / 2.0 if nadir_col is None else float(nadir_col)
    distance = np.abs(np.arange(w) - centre) / max(centre, w - centre)
    cols = 1.0 + rng.uniform(-across, across) * (distance - 0.5)
    out = image.astype(np.float32) * rows[:, None] * cols[None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def resolution_jitter(image: np.ndarray, rng: np.random.Generator,
                      low: float = 0.5, high: float = 0.9, axis: str = "random") -> np.ndarray:
    """Lose resolution and restore the size: boxes are unchanged.

    Downsamples by a random factor in [low, high] along the across-track axis
    (x), the along-track axis (y) or both, then resizes back. Mimics a sonar
    with coarser range bins or a faster tow.
    """
    import cv2

    h, w = image.shape
    factor = rng.uniform(low, high)
    choice = rng.choice(["x", "y", "both"]) if axis == "random" else axis
    fx = factor if choice in ("x", "both") else 1.0
    fy = factor if choice in ("y", "both") else 1.0
    small = cv2.resize(image, (max(1, int(w * fx)), max(1, int(h * fy))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def paste_with_shadow(seabed: np.ndarray, crop: np.ndarray, x: int, y: int, cls: int,
                      rng: np.random.Generator, nadir_col: float | None = None,
                      height_ratio: float | None = None, shadow_level: float = 0.15,
                      mask: np.ndarray | None = None
                      ) -> tuple[np.ndarray, tuple[int, float, float, float, float]]:
    """Paste an object crop at (x, y) with a shadow on the far side from nadir.

    crop           grey object crop, ideally the highlight only (cut the
                   original shadow away, or it will be doubled)
    mask           where the crop is object (bool, same shape); default: crop
                   pixels brighter than the crop's median
    nadir_col      nadir column of the seabed image; default its centre. The
                   shadow extends toward larger x if the paste is right of
                   nadir, smaller x if left, which is the rule hazard_verify
                   checks.
    height_ratio   shadow length / object width; default random 0.6-1.5. A
                   longer shadow at longer range is what flat-seabed geometry
                   gives, and the default scales it by range fraction.
    shadow_level   shadow intensity as a fraction of local seabed

    Returns the new image and the YOLO row for the pasted object's box. The
    box covers the object only, not its shadow, matching how this project's
    detectors are labelled; change it if your labels include shadows.
    """
    out = seabed.astype(np.float32).copy()
    h, w = out.shape
    ch, cw = crop.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + cw), min(h, y + ch)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("paste location is outside the image")
    part = crop[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    if mask is None:
        mask = crop > np.median(crop)
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]

    centre = w / 2.0 if nadir_col is None else float(nadir_col)
    far = 1 if (x0 + x1) / 2.0 >= centre else -1
    range_fraction = abs((x0 + x1) / 2.0 - centre) / max(centre, w - centre, 1.0)
    ratio = rng.uniform(0.6, 1.5) if height_ratio is None else float(height_ratio)
    length = int(max(3, round((x1 - x0) * ratio * (0.5 + range_fraction))))

    local = float(np.median(out[y0:y1, max(0, x0 - cw):min(w, x1 + cw)]))
    # Shadow: for each row, from the object's far-most pixel outward.
    for r in range(y1 - y0):
        cols = np.flatnonzero(m[r])
        if cols.size == 0:
            continue
        edge = x0 + (cols.max() + 1 if far > 0 else cols.min() - 1)
        start, stop = (edge, min(w, edge + length)) if far > 0 else (max(0, edge - length + 1), edge + 1)
        if stop > start:
            speck = rng.rayleigh(1.0, stop - start) / math.sqrt(math.pi / 2)
            out[y0 + r, start:stop] = local * shadow_level * speck
    region = out[y0:y1, x0:x1]
    region[m] = part[m]
    out[y0:y1, x0:x1] = region

    ys, xs = np.nonzero(m)
    bx0, bx1 = x0 + xs.min(), x0 + xs.max() + 1
    by0, by1 = y0 + ys.min(), y0 + ys.max() + 1
    row = (int(cls), (bx0 + bx1) / 2 / w, (by0 + by1) / 2 / h, (bx1 - bx0) / w, (by1 - by0) / h)
    return np.clip(out, 0, 255).astype(np.uint8), row


def augment_once(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random intensity-only augmentation chain. Boxes are unchanged."""
    out = image
    if rng.uniform() < 0.8:
        out = gain_drift(out, rng)
    if rng.uniform() < 0.6:
        out = resolution_jitter(out, rng)
    if rng.uniform() < 0.7:
        out = speckle(out, rng, looks=float(rng.uniform(3, 8)))
    return out


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path, help="TRAIN images only")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if "val" in args.images.parts or "test" in args.images.parts:
        print("refusing to augment a val/test split: augmented images must never be evaluated on")
        return 1
    rng = np.random.default_rng(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        grey = np.asarray(Image.open(path).convert("L"))
        label = args.labels / (path.stem + ".txt")
        text = label.read_text() if label.is_file() else ""
        for k in range(args.copies):
            name = f"{path.stem}_aug{k}"
            Image.fromarray(augment_once(grey, rng)).save(args.out / "images" / f"{name}.jpg", quality=92)
            (args.out / "labels" / f"{name}.txt").write_text(text)
            count += 1
    print(f"wrote {count} augmented images to {args.out} (labels copied unchanged; "
          "these are synthetic variations, keep them out of validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
