"""Confidence scoring and noise filtering: does the image agree with the detector?

A YOLO box is a claim that something is there. On side-scan sonar the claim
fails in a small number of well-understood ways, and every one of them leaves
a physical signature that can be measured in the strip the box came from:

    NADIR AND WATER COLUMN   Between the nadir and the first seabed return no
                             seabed object can appear on a slant-range record.
                             The boundary is the strongest straight edge in
                             the image, which is exactly what a detector
                             trained on wrecks latches onto. On this project's
                             own sample strip all of the detections are this.
    NATURAL SHADOW           A dark patch with no bright return in front of it
                             is a shadow or a depression, not a reflector.
    ROCK CLUTTER             A rock casts a perfectly good highlight and shadow.
                             What gives a rock field away is its neighbours:
                             many similar blobs, oriented every which way.
    DROPOUT / MOTION         Rows the navigation says were interpolated, lost
                             or smeared by attitude carry invented texture.

and one way the claim is SUPPORTED:

    PROUD OBJECT             A man-made object standing off the seabed returns
                             a highlight and then casts a shadow on the side
                             AWAY from nadir. Straight edges, coherent
                             orientation and mesh texture add to it.

Each cue is a number between 0 and 1, and every raw measurement it was computed
from is stored beside it. The cues are fused with the detector's calibrated
probability in logit space:

    confidence_pct = 100 * sigmoid( logit(p_calibrated)
                                     + sum(weight * applicability * (score - neutral)) )

and every term of that sum is written into `verification.terms`, so the
percentage can be recomputed by hand from the record. The weights live in
hazard_config under "Verification".

WHAT THIS IS NOT
    Every cue here is a HEURISTIC. None is a trained classifier, none has been
    validated on a labelled sonar benchmark, and the weights were set against
    synthetic seabeds and two public records. The output is a better-informed
    ranking, not a probability anyone has measured. The calibrated detector
    probability is the only number in the formula with a statistical meaning,
    and only when models/calibration.json exists.

NOTHING IS DELETED
    A detection the evidence argues against is marked `suppressed` with plain-
    English reasons. It stays in the list. Suppression needs BOTH a low fused
    score and at least one hard artefact reason, so a faint contact that is
    merely unremarkable is never hidden. The detector's own confidence is kept
    untouched as `verification.detector_confidence`.

API
    ctx  = strip_context(image_path, sidecar)           once per strip
    cal  = load_calibration("models/calibration.json")  None when absent
    blk  = verify_detection(detection, ctx, cal)        pure; no mutation
    summ = verify_survey(detections, {strip: ctx}, cal) in place; adds
           verification, confidence_pct, suppressed, dimensions.height_m
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_severity import normalize_class

log = logging.getLogger("deepecho.hazard")

# Degraded-row reasons from the ingest sidecar. Anything other than "ok" is
# degraded; these are the names the sidecar contract uses.
DEGRADED_QUALITIES = ("interpolated", "dropout", "attitude")


def _r(value: Any, digits: int = 4) -> Any:
    """Round for the record, leaving None and non-finite values explicit."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(1.0 - cfg.VERIFY_PROB_EPS, max(cfg.VERIFY_PROB_EPS, float(p)))
    return math.log(p / (1.0 - p))


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    if sigma <= 0:
        return array.astype(np.float32, copy=False)
    return cv2.GaussianBlur(array.astype(np.float32, copy=False), (0, 0), float(sigma),
                            borderType=cv2.BORDER_REFLECT)


# ===========================================================================
# Strip context
# ===========================================================================

@dataclass
class StripContext:
    """Everything about one strip that does not depend on a detection.

    Built once per strip because loading a 3600 x 2758 image and tracing its
    water column is the expensive part; every detection on the strip reuses it.

    Fields that could not be determined are None (or NaN per row) and `basis`
    says why. Nothing here is a plausible default dressed up as a measurement:
    no nadir means no nadir, and the cues that need one say so.
    """

    strip: str
    image_path: str | None
    grey: np.ndarray                       # float32, 0..255, H x W
    nadir_col: float | None
    nadir_basis: dict[str, Any]
    wc_left: np.ndarray | None             # per row, first seabed column port side (NaN unknown)
    wc_right: np.ndarray | None            # per row, first seabed column starboard side
    m_per_px_across: float | None = None
    m_per_px_along: float | None = None
    port_is_left: bool | None = None
    slant_range_corrected: bool | None = None
    altitude_m: np.ndarray | None = None   # per row, NaN unknown
    degraded: np.ndarray | None = None     # per row bool
    degraded_reason: list[str] | None = None
    degraded_rows: list[list[Any]] = field(default_factory=list)
    basis: dict[str, Any] = field(default_factory=dict)
    _cv_ref: dict[float, float] = field(default_factory=dict, repr=False)

    @property
    def height(self) -> int:
        return int(self.grey.shape[0])

    @property
    def width(self) -> int:
        return int(self.grey.shape[1])

    def water_mask(self, y0: int, y1: int, x0: int, x1: int,
                   margin: float = 0.0) -> np.ndarray | None:
        """True where a pixel of the window lies in the water column.

        Uses the traced per-row edges where they exist and a NADIR_BAND_PX band
        around the nadir column for rows where they do not. None when neither
        is known, which callers treat as "no water column to exclude".
        `margin` widens the band by that many pixels each side; image
        statistics use it so the traced edge's own uncertainty does not leak
        water-column darkness into a seabed measurement.
        """
        if self.nadir_col is None and self.wc_left is None:
            return None
        cols = np.arange(x0, x1, dtype=np.float32)[None, :]
        rows = slice(y0, y1)
        n = y1 - y0
        if self.wc_left is not None:
            left = self.wc_left[rows].astype(np.float32)
            right = self.wc_right[rows].astype(np.float32)
        else:
            left = np.full(n, np.nan, np.float32)
            right = np.full(n, np.nan, np.float32)
        if self.nadir_col is not None:
            missing = ~np.isfinite(left) | ~np.isfinite(right)
            left = np.where(missing, self.nadir_col - cfg.NADIR_BAND_PX, left)
            right = np.where(missing, self.nadir_col + cfg.NADIR_BAND_PX, right)
        left = np.nan_to_num(left - margin, nan=np.inf)[:, None]
        right = np.nan_to_num(right + margin, nan=-np.inf)[:, None]
        return (cols > left) & (cols < right)

    def cv_reference(self, sigma: float, k: int) -> float | None:
        """Median local coefficient of variation of plain seabed, at this blur and window.

        The clutter cue asks "is this neighbourhood rougher than this strip's
        seabed usually is", which needs a per-strip reference: speckle
        statistics change with the sonar, the gain, the JPEG and the display
        palette, so a theoretical Rayleigh value would be wrong on every real
        record. Measured with exactly the local-window statistic the cue uses
        (see _local_cv), on a grid of windows lying wholly on seabed outside
        degraded rows. A median, so windows that land on objects do not move it.
        """
        key = (round(float(sigma), 2), int(k))
        if key in self._cv_ref:
            return self._cv_ref[key]
        size, values = max(64, 3 * int(k)), []
        H, W = self.grey.shape
        step_y = max(size, H // 12)
        step_x = max(size, W // 16)
        for y in range(0, H - size, step_y):
            if self.degraded is not None and self.degraded[y:y + size].any():
                continue
            for x in range(0, W - size, step_x):
                water = self.water_mask(y, y + size, x, x + size)
                if water is not None and water.any():
                    continue
                window = _blur(self.grey[y:y + size, x:x + size], sigma)
                if float(window.mean()) > 2.0:
                    values.append(float(np.median(_local_cv(window, k)[k:-k, k:-k])))
        result = float(np.median(values)) if len(values) >= 4 else None
        self._cv_ref[key] = result
        return result


def _local_cv(image: np.ndarray, k: int) -> np.ndarray:
    """Per-pixel std / mean over a k x k window.

    Local rather than over the whole neighbourhood, because a neighbourhood
    hundreds of pixels wide also contains the gain ramp across the swath and
    would read as rough on perfectly plain seabed.
    """
    import cv2

    mean = cv2.boxFilter(image, cv2.CV_32F, (k, k), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(image * image, cv2.CV_32F, (k, k), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(sq - mean * mean, 0.0)) / np.maximum(mean, 1.0)


def _load_grey(image_path: Any) -> np.ndarray:
    """8-bit luma as float32. Rec. 601 via PIL, the same weights survey_preparation uses."""
    from PIL import Image

    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def _dark_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True in a 1-D boolean array."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def _close_1d(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill False gaps shorter than `gap` between True runs.

    The water column is not uniformly dark: S-7's record has a saturated 12 px
    zero-range line down the middle and faint surface-return lines either
    side. Closing bridges them so the band is found as one run.
    """
    from scipy.ndimage import binary_closing

    if gap <= 1:
        return mask
    return binary_closing(mask, structure=np.ones(int(gap), bool), border_value=0) | mask


def estimate_nadir(grey: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    """The nadir column of a plain image strip, or None if it is not clear.

    METHOD
        1. Column profile: the median of every column over all rows. A median,
           not a mean, so bright objects and annotation in a few rows do not
           move it.
        2. Median-filter the profile (width ~1.5% of the image) against speckle.
        3. Dark = below p10 + 0.5 * (p90 - p10) of the profile. Close gaps up to
           3% of the width, which bridges a bright zero-range line.
        4. Candidate bands: dark runs that do not touch either image edge (the
           water column has seabed on both sides; a dark margin does not) and
           whose centre lies in the central NADIR_SEARCH_CENTRAL_FRACTION.
        5. The widest candidate wins, and is accepted only if the seabed either
           side is at least NADIR_MIN_CONTRAST times brighter than the band and
           the band is at least NADIR_MIN_BAND_FRACTION of the width.

    The nadir is the centre of the band. That is exact for a record with
    symmetric port and starboard water column and is stated as an estimate
    either way. On a slant-range-corrected strip the band may be too narrow to
    find, and None is the honest answer there.
    """
    from scipy.ndimage import median_filter

    info: dict[str, Any] = {"method": "column-median profile, widest central dark band "
                                      "with seabed on both sides",
                            "basis": "heuristic estimate from the image"}
    if grey.ndim != 2 or grey.shape[1] < 32:
        info["reason"] = "image too small to estimate a nadir"
        return None, info

    H, W = grey.shape
    step = max(1, H // 400)
    profile = np.median(grey[::step], axis=0)
    width = max(5, int(W * 0.015) | 1)
    smooth = median_filter(profile, size=width, mode="nearest")
    p10, p90 = float(np.percentile(smooth, 10)), float(np.percentile(smooth, 90))
    threshold = p10 + 0.5 * (p90 - p10)
    dark = _close_1d(smooth < threshold, max(3, int(W * 0.03)))
    info.update({"profile_p10": _r(p10, 2), "profile_p90": _r(p90, 2),
                 "threshold": _r(threshold, 2)})

    lo = W * (0.5 - cfg.NADIR_SEARCH_CENTRAL_FRACTION / 2)
    hi = W * (0.5 + cfg.NADIR_SEARCH_CENTRAL_FRACTION / 2)
    candidates = [(s, e) for s, e in _dark_runs(dark)
                  if s > 0 and e < W and lo <= (s + e) / 2 <= hi]
    if not candidates:
        info["reason"] = "no dark band with seabed on both sides near the centre"
        return None, info

    start, end = max(candidates, key=lambda run: run[1] - run[0])
    # p25 rather than the median: a band is often brighter towards its middle
    # (surface and zero-range lines) and dimmer at its near-range edges.
    band_level = float(np.percentile(smooth[start:end], 25))
    side = max(8, (end - start) // 2)
    left_level = float(np.median(smooth[max(0, start - side):start]))
    right_level = float(np.median(smooth[end:min(W, end + side)]))
    contrast = min(left_level, right_level) / max(band_level, 1.0)
    info.update({"band": [int(start), int(end)], "band_level": _r(band_level, 2),
                 "seabed_level_left": _r(left_level, 2),
                 "seabed_level_right": _r(right_level, 2),
                 "contrast": _r(contrast, 3)})

    if (end - start) < cfg.NADIR_MIN_BAND_FRACTION * W:
        info["reason"] = "dark band too narrow to be a water column"
        return None, info
    if contrast < cfg.NADIR_MIN_CONTRAST:
        info["reason"] = (f"band contrast {contrast:.2f} below NADIR_MIN_CONTRAST "
                          f"{cfg.NADIR_MIN_CONTRAST}")
        return None, info

    nadir = (start + end - 1) / 2.0
    info["nadir_col"] = _r(nadir, 1)
    return nadir, info


def estimate_water_column(grey: np.ndarray, nadir_col: float,
                          band: list[int] | None = None
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per-row first-seabed-return columns either side of a known nadir.

    The water column narrows and widens with altitude, so one band for the whole
    strip would miss the boundary exactly where detections cluster on it.

    METHOD. Rows are taken in blocks of WATER_COLUMN_BLOCK_ROWS. In each block:
      1. column-median profile, median-filtered over 1.5% of the width against
         speckle (a median preserves steps).
      2. At each column x, on the port side, inner = profile[x + k] (towards
         nadir) and outer = the median of the profile over the next 5% of the
         width beyond x - k (away from nadir); mirrored on starboard. Using a
         long outer window means a step must lead into SUSTAINED seabed, so a
         bright zero-range line or bottom-tracking overlay a few tens of pixels
         wide is not taken for the bottom.
      3. Candidates raise the level by at least NADIR_MIN_CONTRAST x and 8 grey
         levels, with a step at least 40% of the block's strongest. The first
         seabed return is the peak of the candidate run CLOSEST to nadir:
         walking outward, the first big step up is the bottom, and a wreck's
         shadow edge further out must not be mistaken for it.
      4. If the image's own dark band is known, the search is limited to the
         band plus a quarter of its width either side.
    Block edges are median-smoothed over 9 blocks along the track, so an object
    abutting the water column for a few blocks does not drag the edge with it.
    A block with no step gets NaN, never a guess, and the nadir band fallback
    applies to those rows.

    Levels are not used directly because they differ between the two sides of
    one record: S-7's starboard water column is twice as bright as its port.
    """
    from scipy.ndimage import median_filter

    H, W = grey.shape
    block = cfg.WATER_COLUMN_BLOCK_ROWS
    n_blocks = max(1, math.ceil(H / block))
    lefts = np.full(n_blocks, np.nan)
    rights = np.full(n_blocks, np.nan)
    nadir = int(round(min(max(nadir_col, 0), W - 1)))
    k = max(4, W // 200)
    size = max(9, int(W * 0.015) | 1)
    long = max(15, int(W * 0.05) | 1)
    if band is not None:
        margin = int(0.25 * (band[1] - band[0])) + 16
        port_lo, stbd_hi = max(k, band[0] - margin), min(W - k, band[1] + margin)
    else:
        port_lo, stbd_hi = k, W - k

    def pick(inner: np.ndarray, outer: np.ndarray, xs: np.ndarray, toward: int) -> float:
        steps = outer - inner
        ok = (outer >= cfg.NADIR_MIN_CONTRAST * np.maximum(inner, 1.0)) & (steps >= 8)
        if not ok.any():
            return float("nan")
        strong = ok & (steps >= 0.4 * steps[ok].max())
        runs = _dark_runs(strong)
        start, end = max(runs, key=lambda r: r[0]) if toward > 0 else min(runs, key=lambda r: r[0])
        return float(xs[start + int(np.argmax(steps[start:end]))])

    for b in range(n_blocks):
        rows = grey[b * block:min(H, (b + 1) * block)]
        prof = median_filter(np.median(rows, axis=0), size=size, mode="nearest")
        wide = median_filter(prof, size=long, mode="nearest")
        xs = np.arange(port_lo, max(port_lo, nadir))
        if xs.size:
            inner = prof[np.minimum(xs + k, W - 1)]
            outer = wide[np.maximum(xs - k - long // 2, 0)]
            lefts[b] = pick(inner, outer, xs, +1)
        xs = np.arange(min(nadir + 1, stbd_hi), stbd_hi)
        if xs.size:
            inner = prof[np.maximum(xs - k, 0)]
            outer = wide[np.minimum(xs + k + long // 2, W - 1)]
            rights[b] = pick(inner, outer, xs, -1)

    def smooth(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if finite.sum() >= 3:
            filled = np.where(finite, values, np.interp(
                np.arange(len(values)), np.flatnonzero(finite), values[finite]))
            values = np.where(finite, median_filter(filled, size=9, mode="nearest"), np.nan)
        return np.repeat(values, block)[:H]

    left_rows, right_rows = smooth(lefts), smooth(rights)
    info = {
        "method": "per-block median profile; peak of the strong outward step into "
                  "sustained seabed closest to nadir",
        "block_rows": block,
        "blocks_traced_left": int(np.isfinite(lefts).sum()),
        "blocks_traced_right": int(np.isfinite(rights).sum()),
        "blocks": n_blocks,
        "search_band": None if band is None else [int(port_lo), int(stbd_hi)],
        "basis": "heuristic estimate from the image",
    }
    return left_rows, right_rows, info


def strip_context(image_path: Any, sidecar: dict[str, Any] | None = None, *,
                  strip: str | None = None, grey: np.ndarray | None = None) -> StripContext:
    """Load a strip once and gather everything the cues need to know about it.

    sidecar
        The ingest contract: nadir_col, m_per_px_across, m_per_px_along,
        port_is_left, rows[{row, altitude_m, quality}], degraded_rows
        [[start, end_inclusive, reason]], and optionally slant_range_corrected.
        None for a plain image, in which case the nadir is estimated and
        everything else is unknown.
    grey
        An already-loaded float grey array, for tests and callers that hold
        the pixels. image_path may then be None.
    """
    if grey is None:
        grey = _load_grey(image_path)
    grey = np.asarray(grey, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey[..., 0] * 0.299 + grey[..., 1] * 0.587 + grey[..., 2] * 0.114
    H, W = grey.shape
    sidecar = sidecar or {}
    basis: dict[str, Any] = {"sidecar": bool(sidecar)}

    nadir_col = sidecar.get("nadir_col")
    if nadir_col is not None:
        nadir_col = float(nadir_col)
        nadir_basis = {"source": "sidecar", "nadir_col": _r(nadir_col, 1)}
    else:
        nadir_col, nadir_basis = estimate_nadir(grey)
        nadir_basis["source"] = "estimated" if nadir_col is not None else "unknown"

    wc_left = wc_right = None
    if nadir_col is not None:
        wc_left, wc_right, wc_info = estimate_water_column(grey, nadir_col,
                                                           nadir_basis.get("band"))
        basis["water_column"] = wc_info

    def per_row(key: str) -> np.ndarray | None:
        rows = sidecar.get("rows") or []
        if not rows:
            return None
        values = np.full(H, np.nan)
        for row in rows:
            index, value = row.get("row"), row.get(key)
            if index is not None and value is not None and 0 <= int(index) < H:
                values[int(index)] = float(value)
        if not np.isfinite(values).any():
            return None
        finite = np.flatnonzero(np.isfinite(values))
        # Rows between reported ones take the linear interpolation of their
        # neighbours; rows beyond the first/last report stay NaN.
        inside = np.arange(finite[0], finite[-1] + 1)
        values[inside] = np.interp(inside, finite, values[finite])
        return values

    altitude = per_row("altitude_m")

    degraded = np.zeros(H, bool)
    reasons = [""] * H
    degraded_rows: list[list[Any]] = []
    for entry in sidecar.get("degraded_rows") or []:
        start, end, reason = int(entry[0]), int(entry[1]), str(entry[2])
        degraded_rows.append([start, end, reason])
        for r in range(max(0, start), min(H, end + 1)):
            degraded[r] = True
            reasons[r] = reason
    for row in sidecar.get("rows") or []:
        quality = str(row.get("quality") or "ok")
        index = row.get("row")
        if quality != "ok" and index is not None and 0 <= int(index) < H:
            degraded[int(index)] = True
            reasons[int(index)] = reasons[int(index)] or quality
    if sidecar:
        basis["degraded_rows"] = "sidecar"
    else:
        # A plain image has no navigation to flag rows, but a fully blank row
        # is visible in the pixels. Only that is inferred; smear and attitude
        # are not detectable this way and are not claimed.
        blank = np.percentile(grey[:, ::max(1, W // 512)], 99, axis=1) <= cfg.BLANK_ROW_MAX_LEVEL
        for start, end in _dark_runs(blank):
            degraded_rows.append([int(start), int(end - 1), "blank (estimated from image)"])
            degraded[start:end] = True
            for r in range(start, end):
                reasons[r] = "blank (estimated from image)"
        basis["degraded_rows"] = "estimated from image: fully blank rows only"

    def maybe(key: str) -> float | None:
        value = sidecar.get(key)
        return None if value is None else float(value)

    port_is_left = sidecar.get("port_is_left")
    return StripContext(
        strip=strip or (Path(image_path).stem if image_path is not None else "strip"),
        image_path=None if image_path is None else str(image_path),
        grey=grey, nadir_col=nadir_col, nadir_basis=nadir_basis,
        wc_left=wc_left, wc_right=wc_right,
        m_per_px_across=maybe("m_per_px_across"), m_per_px_along=maybe("m_per_px_along"),
        port_is_left=None if port_is_left is None else bool(port_is_left),
        slant_range_corrected=(None if sidecar.get("slant_range_corrected") is None
                               else bool(sidecar["slant_range_corrected"])),
        altitude_m=altitude, degraded=degraded, degraded_reason=reasons,
        degraded_rows=degraded_rows, basis=basis)


# ===========================================================================
# Calibration
# ===========================================================================

@dataclass
class Calibration:
    """Per-model (and optionally per-class) recalibration of detector scores.

    File format (written by training/calibrate.py):

        {"format": "deepecho-calibration", "format_version": 1,
         "models": {
            "known": {"method": "temperature", "temperature": 1.37,
                      "platt": {"a": 0.8, "b": -0.2},            # kept for audit
                      "n": 812, "ece_before": 0.11, "ece_after": 0.03,
                      "classes": {"ship": {"method": "platt", "a": 0.9, "b": 0.1,
                                           "n": 240}}},
            "*": {...}                                           # any model
         }}

    temperature:  p' = sigmoid(logit(p) / T)
    platt:        p' = sigmoid(a * logit(p) + b)

    Lookup order: models[model].classes[class] -> models[model] -> models["*"]
    -> identity. Model names are compared by file stem, lowercased, so
    "models/known.pt", "known.pt" and "known" are one model.

    WHAT IT MEANS. The fitted probability is P(this box matches a ground-truth
    object of this class at IoU >= 0.5 | score), measured on the validation set
    it was fitted on. It says nothing about objects the detector missed.
    """

    models: dict[str, dict[str, Any]]
    source: str | None = None

    @staticmethod
    def model_key(model: Any) -> str:
        if model is None:
            return "*"
        return Path(str(model)).stem.lower() or "*"

    def params(self, model: Any, cls: Any) -> tuple[dict[str, Any] | None, str]:
        key = self.model_key(model)
        entry = self.models.get(key)
        name = normalize_class(cls)
        if entry is not None:
            per_class = (entry.get("classes") or {}).get(name)
            if per_class and per_class.get("method"):
                return per_class, f"model:{key}/class:{name}"
            if entry.get("method"):
                return entry, f"model:{key}"
        if key != "*" and self.models.get("*", {}).get("method"):
            return self.models["*"], "model:*"
        return None, "uncalibrated: no entry for this model"

    @staticmethod
    def transform(confidence: float, params: dict[str, Any] | None) -> float:
        if not params:
            return float(confidence)
        method = params.get("method")
        z = _logit(confidence)
        if method == "temperature":
            return _sigmoid(z / float(params["temperature"]))
        if method == "platt":
            return _sigmoid(float(params["a"]) * z + float(params["b"]))
        if method == "identity":
            return float(confidence)
        raise ValueError(f"unknown calibration method {method!r}")

    def apply(self, confidence: float, model: Any = None, cls: Any = None) -> float:
        params, _ = self.params(model, cls)
        return self.transform(confidence, params)


def load_calibration(path: Any = None) -> Calibration | None:
    """The calibration file, or None when there is none.

    None is a normal state, not an error: an uncalibrated detector's scores are
    used as they are and every verification block says `uncalibrated`. A file
    that exists but is malformed IS an error and raises, because silently
    ignoring a calibration someone fitted is worse than stopping.
    """
    path = Path(cfg.CALIBRATION_PATH if path is None else path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "deepecho-calibration":
        raise ValueError(f"{path} is not a deepecho-calibration file")
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{path} has no 'models' object")
    calibration = Calibration(models={Calibration.model_key(k) if k != "*" else "*": v
                                      for k, v in models.items()}, source=path.name)
    # Validate every entry now, not at the first detection that hits it.
    for key, entry in calibration.models.items():
        for params in [entry, *(entry.get("classes") or {}).values()]:
            if params.get("method"):
                Calibration.transform(0.5, params)
    return calibration


def calibrate(confidence: float, model: Any, cls: Any,
              calibration: Calibration | None) -> tuple[float, dict[str, Any]]:
    """(calibrated probability, basis). Identity with basis 'uncalibrated' when absent."""
    if calibration is None:
        return float(confidence), {"method": "identity",
                                   "basis": "uncalibrated: no calibration file"}
    params, how = calibration.params(model, cls)
    p = Calibration.transform(confidence, params)
    basis = {"method": (params or {}).get("method", "identity"), "basis": how,
             "source": calibration.source}
    if params:
        basis.update({k: params[k] for k in ("temperature", "a", "b", "n") if k in params})
    return p, basis


# ===========================================================================
# Per-detection measurements
# ===========================================================================

def class_expectation(cls: Any) -> tuple[dict[str, Any], str]:
    """Physical expectations for a class, matched like the severity table."""
    name = normalize_class(cls)
    table = cfg.VERIFY_CLASS_EXPECTATIONS
    if name in table:
        return dict(table[name]), "exact"
    keys = [key for key in table if key and key in name]
    if keys:
        key = max(keys, key=len)
        return dict(table[key]), f"substring:{key}"
    return dict(cfg.VERIFY_DEFAULT_EXPECTATION), "default"


@dataclass
class _Geometry:
    x1: int
    y1: int
    x2: int
    y2: int
    far: int | None        # +1 far range at larger x, -1 at smaller x, None unknown
    shadow_len: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def xc(self) -> float:
        return (self.x1 + self.x2) / 2.0


def _geometry(detection: dict[str, Any], ctx: StripContext) -> _Geometry | None:
    box = detection.get("bbox_global")
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(max(0, math.floor(x1)))
    y1 = int(max(0, math.floor(y1)))
    x2 = int(min(ctx.width, math.ceil(x2)))
    y2 = int(min(ctx.height, math.ceil(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    far = None
    if ctx.nadir_col is not None:
        far = 1 if (x1 + x2) / 2.0 >= ctx.nadir_col else -1
    length = int(min(cfg.SHADOW_MAX_WINDOW_PX, max(x2 - x1, cfg.SHADOW_MIN_WINDOW_PX)))
    return _Geometry(x1, y1, x2, y2, far, length)


class _Window:
    """A padded crop of the strip, raw and blurred, addressed in strip coordinates."""

    def __init__(self, ctx: StripContext, g: _Geometry, sigma: float):
        # Wide enough for every sub-region: the clutter neighbourhood (box
        # grown by its own size or CLUTTER_MIN_PAD_PX), the shadow and near-side
        # windows (shadow_len), and the doubled background ring (<= box size).
        pad_x = max(g.w, cfg.CLUTTER_MIN_PAD_PX) + g.shadow_len + 8
        pad_y = max(g.h, cfg.CLUTTER_MIN_PAD_PX) + 8
        self.x0 = max(0, g.x1 - pad_x)
        self.x1 = min(ctx.width, g.x2 + pad_x)
        self.y0 = max(0, g.y1 - pad_y)
        self.y1 = min(ctx.height, g.y2 + pad_y)
        self.raw = ctx.grey[self.y0:self.y1, self.x0:self.x1]
        self.blur = _blur(self.raw, sigma)
        water = ctx.water_mask(self.y0, self.y1, self.x0, self.x1, cfg.WATER_STATS_MARGIN_PX)
        self.water = np.zeros(self.raw.shape, bool) if water is None else water
        degraded = np.zeros(self.raw.shape[0], bool)
        if ctx.degraded is not None:
            degraded = ctx.degraded[self.y0:self.y1]
        self.degraded = np.repeat(degraded[:, None], self.raw.shape[1], axis=1)

    def mask(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        """Boolean mask of a strip-coordinate rectangle, clipped to the window."""
        m = np.zeros(self.raw.shape, bool)
        ax0 = int(max(self.x0, math.floor(x0))) - self.x0
        ax1 = int(min(self.x1, math.ceil(x1))) - self.x0
        ay0 = int(max(self.y0, math.floor(y0))) - self.y0
        ay1 = int(min(self.y1, math.ceil(y1))) - self.y0
        if ax1 > ax0 and ay1 > ay0:
            m[ay0:ay1, ax0:ax1] = True
        return m


def _side_rect(g: _Geometry, side: int, length: int, inner_half: bool) -> tuple:
    """Rectangle beyond the box on `side` (+1 larger x), optionally with the box's half on that side."""
    if side > 0:
        x0 = g.xc if inner_half else g.x2
        return (x0, g.x2 + length, g.y1, g.y2)
    x1 = g.xc if inner_half else g.x1
    return (g.x1 - length, x1, g.y1, g.y2)


def _measure_background(win: _Window, g: _Geometry) -> dict[str, Any]:
    """Seabed statistics in a ring around the box.

    Excludes the box, the shadow window on the far side (a real shadow would
    otherwise lower the background it is compared against), the water column
    and degraded rows. Expanded once if the ring is too small, for instance a
    box hard against the water column.
    """
    for scale in (1.0, 2.0):
        pr = max(cfg.BG_RING_MIN_PX, 0.5 * g.h) * scale
        pc = max(cfg.BG_RING_MIN_PX, 0.5 * g.w) * scale
        ring = win.mask(g.x1 - pc, g.x2 + pc, g.y1 - pr, g.y2 + pr)
        ring &= ~win.mask(g.x1, g.x2, g.y1, g.y2)
        if g.far is not None:
            ring &= ~win.mask(*_side_rect(g, g.far, g.shadow_len, False))
        ring &= ~win.water & ~win.degraded
        if ring.sum() >= cfg.BG_MIN_PIXELS:
            values = win.blur[ring]
            median = float(np.median(values))
            return {"median": median, "p10": float(np.percentile(values, 10)),
                    "dark_fraction": _dark_fraction(values, median),
                    "p95": float(np.percentile(values, 95)), "pixels": int(ring.sum()),
                    "ring_scale": scale}
    return {"median": None, "pixels": int(ring.sum()),
            "reason": "fewer than BG_MIN_PIXELS seabed pixels around the box"}


def _dark_fraction(values: np.ndarray, bg_median: float) -> float:
    return float((values < (1.0 - cfg.SHADOW_DARK_FRACTION) * bg_median).mean())


def _shadow_side(win: _Window, g: _Geometry, bg: dict[str, Any], side: int) -> dict[str, Any]:
    """Shadow on one side: how much of the far half of the box plus a window beyond is shadow.

    shadow pixel = blurred level below (1 - SHADOW_DARK_FRACTION) * background
    excess       = shadow fraction here - shadow fraction on the background ring
    score        = excess / SHADOW_FULL_EXCESS, clipped to 0..1

    The far half of the box is included because a detector's box around a
    wreck usually contains the wreck's shadow; measuring only beyond the box
    would find plain seabed and call a textbook target shadowless. Subtracting
    the ring's own fraction means a strip full of dark patches does not hand
    every box a shadow.
    """
    region = win.mask(*_side_rect(g, side, g.shadow_len, True)) & ~win.degraded & ~win.water
    out: dict[str, Any] = {"pixels": int(region.sum())}
    if region.sum() < 16:
        out["reason"] = "shadow window empty (image edge or water column)"
        return out
    fraction = _dark_fraction(win.blur[region], bg["median"])
    excess = fraction - bg["dark_fraction"]
    out.update({"dark_fraction": fraction, "excess": excess,
                "score": _clip01(excess / cfg.SHADOW_FULL_EXCESS)})
    return out


def _shadow_length_px(win: _Window, g: _Geometry, bg_median: float) -> dict[str, Any]:
    """Run of dark pixels, per row, starting at the far edge of the object.

    The far edge is the last highlight pixel in the row (so a box that already
    contains the shadow, as a wreck's box usually does, is measured from the
    object and not from the box border); a row with no highlight starts at the
    box's far edge. The run starts within 3 px, tolerates 2 px gaps, and ends
    at 3 consecutive non-dark pixels. The median over rows is reported, from
    rows where a run started at all, which must be at least 30% of the box.
    """
    if g.far is None:
        return {"length_px": None, "reason": "nadir unknown, so the far side is unknown"}
    dark_level = (1.0 - cfg.SHADOW_DARK_FRACTION) * bg_median
    bright_level = (1.0 + cfg.HIGHLIGHT_LEVEL_K) * bg_median
    # Sigma 1, not the detection's working blur: a wider blur smears the
    # highlight into the shadow and moves both ends of the run, which biases
    # the height by several percent on a 20 px shadow.
    fine = _blur(win.raw, 1.0)
    lengths, starts = [], []
    for y in range(g.y1, g.y2):
        row = fine[y - win.y0]
        cols = np.arange(g.x1, g.x2) if g.far > 0 else np.arange(g.x2 - 1, g.x1 - 1, -1)
        values = row[cols - win.x0]
        bright = np.flatnonzero(values > bright_level)
        start = int(cols[bright[-1]]) + g.far if bright.size else (g.x2 if g.far > 0 else g.x1 - 1)
        x, run, misses, began = start, 0, 0, False
        limit = start + g.far * (g.shadow_len + g.w)
        while 0 <= x - win.x0 < win.blur.shape[1] and (x - limit) * g.far < 0:
            dark = row[x - win.x0] < dark_level
            if dark:
                began, misses = True, 0
                run = abs(x - start) + 1
            elif began:
                misses += 1
                if misses >= 3:
                    break
            elif abs(x - start) >= 3:
                break
            x += g.far
        if began:
            lengths.append(run)
            starts.append(start)
    if len(lengths) < 0.3 * g.h:
        return {"length_px": 0.0, "rows_with_shadow": len(lengths),
                "reason": "fewer than 30% of rows show a dark run beyond the object"}
    return {"length_px": float(np.median(lengths)), "rows_with_shadow": len(lengths),
            "measured_on": "sigma-1 blur",
            "far_edge_col": float(np.median(starts))}


def _structure_coherence(image: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """Dominant-orientation strength of the gradient field, 0 (isotropic) to 1 (one direction).

    Summed structure tensor over the region, not an average of local
    coherences: speckle gives a low value because its gradients point
    everywhere, a pipe or a hull edge gives a high one.
    """
    import cv2

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    if mask is not None:
        gx, gy = gx[mask], gy[mask]
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    trace = jxx + jyy
    if trace <= 1e-6:
        return None
    return math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / trace


def _edge_sharpness(win: _Window, g: _Geometry, bg: dict[str, Any]) -> dict[str, Any]:
    """How abruptly the dark region inside the box ends.

    A shadow CAST by an object is bounded by geometry: its edges are hard, a
    few pixels wide at the record's resolution. Shading on a depression, a sand
    wave or a sediment patch grades over tens of pixels. So the gradient on the
    boundary of the dark region, relative to the background level, separates a
    cast shadow from a natural dark patch far better than darkness alone does.

    Blur sigma 2.5 keeps speckle gradients well below a genuine step. The
    statistic is the 75th percentile of |gradient| on the dark region's
    boundary, per pixel and as a fraction of background, minus the same
    percentile on the background ring (the speckle floor).
    """
    import cv2

    pad = 4
    x0, x1 = max(win.x0, g.x1 - pad), min(win.x1, g.x2 + pad)
    y0, y1 = max(win.y0, g.y1 - pad), min(win.y1, g.y2 + pad)
    crop = _blur(win.raw[y0 - win.y0:y1 - win.y0, x0 - win.x0:x1 - win.x0], 2.5)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy) / bg["median"]
    inner = np.zeros(crop.shape, bool)
    inner[g.y1 - y0:g.y2 - y0, g.x1 - x0:g.x2 - x0] = True
    water = win.water[y0 - win.y0:y1 - win.y0, x0 - win.x0:x1 - win.x0]
    dark = (crop < (1.0 - cfg.SHADOW_DARK_FRACTION) * bg["median"]) & ~water
    fraction = float((dark & inner).sum()) / max(1, int(inner.sum()))
    if fraction < 0.03:
        return {"sharpness": None, "score": None, "dark_fraction": _r(fraction),
                "reason": "no dark region in the box"}
    eroded = cv2.erode(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dilated = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = (dilated & ~eroded) & inner & ~water
    if boundary.sum() < 8:
        return {"sharpness": None, "score": None, "dark_fraction": _r(fraction),
                "reason": "dark region has no measurable boundary"}
    sharp = float(np.percentile(grad[boundary], 75))
    ring = ~dilated & ~inner & ~water
    floor = float(np.percentile(grad[ring], 75)) if ring.sum() >= 16 else 0.0
    return {"sharpness": _r(sharp), "speckle_floor": _r(floor), "dark_fraction": _r(fraction),
            "score": _r(_clip01((sharp - floor - cfg.SHADOW_EDGE_SHARP_NULL)
                                / (cfg.SHADOW_EDGE_SHARP_FULL - cfg.SHADOW_EDGE_SHARP_NULL)))}


def _mesh_score(patch: np.ndarray, bg_median: float) -> dict[str, Any]:
    """Grid-like periodic texture, from the autocorrelation of the high-passed box.

    A mesh correlates with itself at its cell spacing in TWO directions, and
    between cells it anti-correlates. So the cue is not the autocorrelation
    itself (smooth patches, JPEG blocks and straight edges all correlate at
    short lags) but its PROMINENCE:

        prominence(lag) = ac(lag) - ac(lag / 2)

    A periodic peak rises above the dip halfway to it; a smoothly decaying
    correlation (speckle grain, blur) and a straight edge (flat along its
    length) do not. The score uses the most prominent lag and the most
    prominent lag at least 45 degrees away from it, and takes the weaker, so
    one set of parallel lines (a ripple field, a pipe's two edges) is not a
    mesh. Lags from 4 px to the box's shorter side / MESH_MIN_REPEATS, after a
    sigma-1 blur. Gated on texture: a high-pass
    standard deviation under 5% of the background is too faint to have a
    period worth believing (and is where JPEG 8 px blocking dominates).
    """
    h, w = patch.shape
    if min(h, w) < 4 * cfg.MESH_MIN_REPEATS:
        return {"score": 0.0, "reason": "box too small for a texture period"}
    # Sigma-1 pre-blur: uncorrelated speckle otherwise dominates the variance
    # and buries a faint mesh whose strands are only a couple of pixels wide.
    patch = _blur(patch, 1.0)
    hp = patch - _blur(patch, max(3.0, min(h, w) / 6.0))
    hp = hp - hp.mean()
    var = float((hp * hp).mean())
    texture = math.sqrt(var) / max(bg_median, 1.0)
    if texture < 0.05:
        return {"score": 0.0, "texture": _r(texture),
                "reason": "texture too faint (high-pass std < 5% of background)"}
    F = np.fft.fft2(hp, s=(2 * h, 2 * w))
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
    cy, cx = h, w
    dy, dx = np.mgrid[-cy:cy, -cx:cx]
    overlap = np.clip((w - np.abs(dx)) * (h - np.abs(dy)), 1, None)
    ac = ac / overlap / var
    half = ac[np.clip(np.round(dy / 2).astype(int) + cy, 0, 2 * cy - 1),
              np.clip(np.round(dx / 2).astype(int) + cx, 0, 2 * cx - 1)]
    prominence = ac - half
    radius = np.hypot(dx, dy)
    # At least MESH_MIN_REPEATS cells across the box's shorter side. Two or
    # three repeats of anything (letters of a display annotation, a pair of
    # boulders) correlate at their spacing; a mesh repeats many times.
    max_lag = min(h, w) / cfg.MESH_MIN_REPEATS
    annulus = (radius >= 4) & (radius <= max_lag) & (dy >= 0)
    values = np.where(annulus, prominence, -np.inf)
    i1 = int(np.argmax(values))
    theta = np.degrees(np.arctan2(dy, dx)) % 180.0
    diff = np.abs(theta - theta.flat[i1])
    diff = np.minimum(diff, 180.0 - diff)
    second = np.where(annulus & (diff >= 45.0), prominence, -np.inf)
    i2 = int(np.argmax(second))
    p1 = float(values.flat[i1])
    p2 = float(second.flat[i2]) if np.isfinite(second.flat[i2]) else 0.0
    return {"texture": _r(texture), "prominence_first": _r(p1), "prominence_second": _r(p2),
            "lag_first": [int(dx.flat[i1]), int(dy.flat[i1])],
            "lag_second": [int(dx.flat[i2]), int(dy.flat[i2])],
            "score": _r(_clip01((p2 - cfg.MESH_NULL) / (cfg.MESH_FULL - cfg.MESH_NULL)))}


def _man_made(win: _Window, g: _Geometry, bg_median: float) -> dict[str, Any]:
    """Straight edges, orientation coherence, highlight shape, mesh.

    regular = 0.45 * line_score + 0.35 * coherence_score + 0.20 * shape_score
    score   = max(regular, mesh_score)

    The mix is a heuristic. Edges carry the most weight because long straight
    segments are rare in natural seabed at object scale; coherence catches
    pipes and hulls whose edges are soft; solidity of the bright component is
    weak on its own (rocks are convex too) and weighted least. Mesh is an
    alternative route to "man-made", not an addition, so a net that has no
    straight edges is not penalised for it.
    """
    import cv2

    ys = slice(g.y1 - win.y0, g.y2 - win.y0)
    xs = slice(g.x1 - win.x0, g.x2 - win.x0)
    sigma = float(np.clip(min(g.w, g.h) / 20.0, 1.0, 2.0))
    local = _blur(win.raw, sigma)[ys, xs]
    scaled = np.clip(local / max(bg_median, 1.0) * 100.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(scaled, cfg.CANNY_LOW, cfg.CANNY_HIGH)
    min_len = int(max(cfg.HOUGH_MIN_LINE_PX, cfg.HOUGH_MIN_LINE_FRACTION * max(g.w, g.h)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, cfg.HOUGH_VOTES,
                            minLineLength=min_len, maxLineGap=cfg.HOUGH_MAX_GAP_PX)
    segments = np.zeros((0, 4)) if lines is None else np.asarray(lines).reshape(-1, 4)
    total = float(np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]).sum())
    perimeter = 2.0 * (g.w + g.h)
    line_ratio = total / perimeter
    line_score = _clip01(line_ratio / cfg.LINE_FULL_RATIO)

    coherence = _structure_coherence(local)
    coherence_score = 0.0 if coherence is None else _clip01(
        (coherence - cfg.COHERENCE_NULL) / (cfg.COHERENCE_FULL - cfg.COHERENCE_NULL))

    bright = (local > (1.0 + cfg.HIGHLIGHT_LEVEL_K) * bg_median).astype(np.uint8)
    solidity = elongation = None
    shape_score = 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[largest, cv2.CC_STAT_AREA])
        if area >= 20:
            points = np.column_stack(np.nonzero(labels == largest))[:, ::-1].astype(np.int32)
            hull_area = float(cv2.contourArea(cv2.convexHull(points)))
            solidity = min(1.0, area / hull_area) if hull_area > 0 else None
            if len(points) >= 5:
                cov = np.cov(points.T.astype(np.float64))
                ev = np.sort(np.linalg.eigvalsh(cov))
                elongation = 1.0 - math.sqrt(max(ev[0], 0) / max(ev[1], 1e-9))
            if solidity is not None:
                shape_score = _clip01((solidity - 0.7) / 0.25)

    mesh = _mesh_score(win.raw[ys, xs].astype(np.float32), bg_median)
    regular = 0.45 * line_score + 0.35 * coherence_score + 0.20 * shape_score
    return {
        "score": max(regular, mesh["score"]),
        "measurements": {
            "long_segments": int(len(segments)),
            "long_segment_length_px": _r(total, 1), "box_perimeter_px": _r(perimeter, 1),
            "line_ratio": _r(line_ratio), "line_score": _r(line_score),
            "hough_min_line_px": min_len,
            "coherence": _r(coherence), "coherence_score": _r(coherence_score),
            "highlight_solidity": _r(solidity), "highlight_elongation": _r(elongation),
            "shape_score": _r(shape_score),
            "regular": _r(regular), "mesh": mesh,
        },
        "basis": "heuristic: 0.45*lines + 0.35*coherence + 0.20*solidity, or mesh if higher",
    }


def _clutter(win: _Window, g: _Geometry) -> dict[str, Any]:
    """Rock field or natural clutter around the box.

    Neighbourhood: the box grown by its own width and height on every side
    (3x the box), by at least CLUTTER_MIN_PAD_PX so a small box still sees a
    few neighbours, minus the water column and degraded rows. In it:

      density      similar-sized bright and dark blobs (local threshold on the
                   neighbourhood median, opened 3x3), excluding any blob that
                   touches the box or its shadow window (that is the object
                   itself), a bright blob with a dark
                   blob on its far side counting as one highlight/shadow pair
                   and unpaired blobs counting half; saturates at
                   CLUTTER_BLOBS_FULL.
      incoherence  1 - structure-tensor coherence of the neighbourhood. Sand
                   ripples and a pipeline are coherent; a boulder field is not.
      roughness    median local CV (window ~2x the box's short side) relative
                   to this strip's plain-seabed value for the same blur and window.

      score = density * (0.4 + 0.6 * incoherence) * (0.6 + 0.4 * roughness)

    Density gates the rest: a smooth, empty neighbourhood is never clutter no
    matter how incoherent its speckle is. Heuristic, and a known confusion: a
    debris field around a wreck also has many blobs, which is why this cue is
    weighed against man-made regularity rather than applied alone.
    """
    import cv2

    pad_x = max(g.w, cfg.CLUTTER_MIN_PAD_PX)
    pad_y = max(g.h, cfg.CLUTTER_MIN_PAD_PX)
    region = win.mask(g.x1 - pad_x, g.x2 + pad_x, g.y1 - pad_y, g.y2 + pad_y)
    region &= ~win.water & ~win.degraded
    region_nobox = region & ~win.mask(g.x1, g.x2, g.y1, g.y2)
    pixels = int(region_nobox.sum())
    if pixels < cfg.BG_MIN_PIXELS:
        return {"score": 0.0, "measurements": {"pixels": pixels},
                "unknown": True, "reason": "too little seabed around the box to judge clutter"}

    median = float(np.median(win.blur[region_nobox]))
    kernel = np.ones((3, 3), np.uint8)
    box_area = g.w * g.h
    lo = max(cfg.CLUTTER_MIN_BLOB_PX, cfg.CLUTTER_SIZE_RATIO[0] * box_area)
    hi = cfg.CLUTTER_SIZE_RATIO[1] * box_area

    # The object's OWN highlight and shadow are not its neighbours. A box is
    # rarely a perfect fit: a detector may box the middle of a large return, or
    # only the highlight of an object whose shadow lies beyond. Any component
    # touching the box grown by a quarter of its size, or the far-side shadow
    # window, is taken to be the object itself and not counted.
    grow_x, grow_y = max(4, g.w // 4), max(4, g.h // 4)
    own = win.mask(g.x1 - grow_x, g.x2 + grow_x, g.y1 - grow_y, g.y2 + grow_y)
    if g.far is not None:
        own |= win.mask(*_side_rect(g, g.far, g.shadow_len, False))

    def blobs(mask: np.ndarray) -> list[tuple[float, float, int, int]]:
        m = cv2.morphologyEx((mask & region).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        self_labels = set(np.unique(labels[own]).tolist())
        out = []
        for i in range(1, n):
            if i in self_labels:
                continue
            area = stats[i, cv2.CC_STAT_AREA]
            if lo <= area <= hi:
                out.append((float(centroids[i][0]), float(centroids[i][1]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])))
        return out

    bright = blobs(win.blur > (1.0 + cfg.CLUTTER_BRIGHT_K) * median)
    dark = blobs(win.blur < (1.0 - cfg.CLUTTER_DARK_K) * median)
    used, pairs = set(), 0
    for bx, by, bw, bh in bright:
        for j, (dx_, dy_, dw, dh) in enumerate(dark):
            if j in used:
                continue
            offset = dx_ - bx
            side_ok = g.far is None or offset * g.far > 0
            if side_ok and abs(offset) <= 2 * max(bw, dw) + 2 and abs(dy_ - by) <= max(bh, dh):
                used.add(j)
                pairs += 1
                break
    similar = pairs + 0.5 * (len(bright) + len(dark) - 2 * pairs)
    density = _clip01(similar / cfg.CLUTTER_BLOBS_FULL)

    coherence = _structure_coherence(win.blur, region_nobox)
    incoherence = 1.0 - (coherence if coherence is not None else 0.0)

    k = int(np.clip(2 * min(g.w, g.h), 9, 65)) | 1
    cv_local = float(np.median(_local_cv(win.blur, k)[region_nobox]))
    sigma = float(np.clip(min(g.w, g.h) / 10.0, 1.0, 4.0))
    return {
        "_cv_local": cv_local, "_sigma": sigma, "_k": k,
        "density": density, "incoherence": incoherence,
        "measurements": {
            "pixels": pixels, "local_median": _r(median, 2),
            "bright_blobs": len(bright), "dark_blobs": len(dark),
            "highlight_shadow_pairs": pairs, "similar_blob_count": _r(similar, 1),
            "blob_area_range_px": [_r(lo, 1), _r(hi, 1)],
            "density": _r(density), "coherence": _r(coherence), "incoherence": _r(incoherence),
            "cv_local": _r(cv_local),
        },
    }


def _finish_clutter(clutter: dict[str, Any], ctx: StripContext) -> dict[str, Any]:
    if clutter.get("unknown"):
        return clutter
    reference = ctx.cv_reference(clutter.pop("_sigma"), clutter.pop("_k"))
    cv_local = clutter.pop("_cv_local")
    if reference:
        roughness = _clip01((cv_local / reference - 1.0) / 1.5)
    else:
        roughness = 0.0
    density, incoherence = clutter.pop("density"), clutter.pop("incoherence")
    score = density * (0.4 + 0.6 * incoherence) * (0.6 + 0.4 * roughness)
    clutter["measurements"].update({"cv_reference": _r(reference), "roughness": _r(roughness)})
    clutter["score"] = _clip01(score)
    clutter["basis"] = ("heuristic: density * (0.4 + 0.6*incoherence) * "
                        "(0.6 + 0.4*roughness)")
    return clutter


def _nadir_zone(ctx: StripContext, g: _Geometry, water_column_ok: bool) -> dict[str, Any]:
    """How much the box looks like the nadir / water-column artefact.

      wc_score      fraction of the box inside the water column, 0 at
                    NADIR_WC_FRACTION_FREE, 1 at NADIR_WC_FRACTION_FULL. Skipped
                    for water-column classes (fish).
      edge_score    box centre within NADIR_EDGE_MARGIN_PX of a traced first
                    seabed return: 1 - distance / margin.
      line_score    box centre within NADIR_BAND_PX of the nadir column.
      score         max of the three.

    Known limitation: the tall superstructure of a wreck can return before the
    seabed and appear inside the water column. Such a box would be penalised.
    """
    if ctx.nadir_col is None:
        return {"score": 0.0, "unknown": True,
                "reason": "nadir unknown: water-column position could not be judged"}
    water = ctx.water_mask(g.y1, g.y2, g.x1, g.x2)
    wc_fraction = float(water.mean()) if water is not None else 0.0
    wc_score = 0.0 if water_column_ok else _clip01(
        (wc_fraction - cfg.NADIR_WC_FRACTION_FREE)
        / (cfg.NADIR_WC_FRACTION_FULL - cfg.NADIR_WC_FRACTION_FREE))

    edge_distance = None
    edge_score = 0.0
    if ctx.wc_left is not None:
        rows = slice(g.y1, g.y2)
        edges = ctx.wc_right[rows] if g.far is not None and g.far > 0 else ctx.wc_left[rows]
        edges = edges[np.isfinite(edges)]
        if edges.size:
            edge_distance = float(np.median(np.abs(edges - g.xc)))
            edge_score = _clip01(1.0 - edge_distance / cfg.NADIR_EDGE_MARGIN_PX)

    line_distance = abs(g.xc - ctx.nadir_col)
    line_score = _clip01(1.0 - line_distance / cfg.NADIR_BAND_PX)

    side_extent = (ctx.width - ctx.nadir_col) if g.far and g.far > 0 else ctx.nadir_col
    range_fraction = line_distance / max(side_extent, 1.0)
    return {
        "score": max(wc_score, edge_score, line_score),
        "measurements": {
            "nadir_col": _r(ctx.nadir_col, 1), "nadir_source": ctx.nadir_basis.get("source"),
            "water_column_fraction": _r(wc_fraction), "wc_score": _r(wc_score),
            "water_column_class_exempt": water_column_ok,
            "first_return_distance_px": _r(edge_distance, 1), "edge_score": _r(edge_score),
            "nadir_distance_px": _r(line_distance, 1), "line_score": _r(line_score),
            "range_fraction": _r(range_fraction),
        },
        "far_range": range_fraction >= cfg.FAR_RANGE_FRACTION,
        "basis": "geometry: max(water-column fraction, first-return edge, nadir line)",
    }


def _dropout(ctx: StripContext, g: _Geometry) -> dict[str, Any]:
    if ctx.degraded is None:
        return {"score": 0.0, "unknown": True, "reason": "no row quality information"}
    rows = ctx.degraded[g.y1:g.y2]
    fraction = float(rows.mean()) if rows.size else 0.0
    reasons = sorted({ctx.degraded_reason[r] for r in range(g.y1, g.y2) if ctx.degraded[r]})
    return {"score": fraction,
            "measurements": {"degraded_row_fraction": _r(fraction), "degraded_reasons": reasons,
                             "source": ctx.basis.get("degraded_rows")},
            "basis": "fraction of the box's rows flagged degraded"}


def _height(ctx: StripContext, g: _Geometry, shadow: dict[str, Any]) -> dict[str, Any]:
    """Object height from shadow length, when the geometry allows it.

        h = altitude * Ls / (R + Ls)

    R  = ground range (m) from nadir to the far edge of the object
    Ls = shadow length (m) along the ground beyond that edge

    Similar triangles between the towfish at `altitude` over nadir, the top of
    the object and the end of its shadow, assuming a flat seabed. On a strip
    that has not been slant-range corrected, pixel distance from nadir is slant
    range and is converted with ground = sqrt(slant^2 - altitude^2). Needs
    altitude, across-track resolution and nadir; null with the missing inputs
    named otherwise.
    """
    missing = [name for name, value in (("altitude_m", ctx.altitude_m),
                                        ("m_per_px_across", ctx.m_per_px_across),
                                        ("nadir_col", ctx.nadir_col)) if value is None]
    if missing:
        return {"height_m": None, "basis": f"not computed: {', '.join(missing)} unknown"}
    altitude_rows = ctx.altitude_m[g.y1:g.y2]
    altitude_rows = altitude_rows[np.isfinite(altitude_rows)]
    if not altitude_rows.size:
        return {"height_m": None, "basis": "not computed: no altitude for these rows"}
    length = shadow.get("length_px")
    if not length:
        return {"height_m": None,
                "basis": f"not computed: no measurable shadow ({shadow.get('reason', 'length 0')})"}
    altitude = float(np.median(altitude_rows))
    res = ctx.m_per_px_across
    edge_px = abs(shadow["far_edge_col"] - ctx.nadir_col)
    end_px = edge_px + length
    if ctx.slant_range_corrected is False:
        def ground(px: float) -> float | None:
            slant = px * res
            return math.sqrt(slant * slant - altitude * altitude) if slant > altitude else None
        r_edge, r_end = ground(edge_px), ground(end_px)
        if r_edge is None or r_end is None:
            return {"height_m": None,
                    "basis": "not computed: object inside the water column in slant range"}
        range_basis = "slant range converted to ground range"
    else:
        r_edge, r_end = edge_px * res, end_px * res
        range_basis = ("pixels treated as ground range" if ctx.slant_range_corrected
                       else "pixels ASSUMED ground range (sidecar did not state "
                            "slant_range_corrected)")
    ls = r_end - r_edge
    height = altitude * ls / (r_edge + ls)
    return {"height_m": _r(height, 2), "altitude_m": _r(altitude, 2),
            "ground_range_m": _r(r_edge, 2), "shadow_length_m": _r(ls, 2),
            "shadow_length_px": _r(length, 1), "far_edge_col": _r(shadow["far_edge_col"], 1),
            "formula": "h = altitude * Ls / (R + Ls)",
            "basis": f"flat-seabed shadow geometry; {range_basis}"}


# ===========================================================================
# Fusion
# ===========================================================================

REASON_TEXT = {
    "nadir_zone": "sits on the nadir / water-column boundary ({detail}); nothing on the "
                  "seabed can appear there, and it is where side-scan false positives cluster",
    "natural_shadow": "the box is {dark:.0%} darker than the surrounding seabed with no bright "
                      "return and soft edges: a natural shadow or depression, not a reflecting "
                      "object",
    "rock_clutter": "{n} similar-sized highlight/shadow blobs within 3x the box, orientation "
                    "coherence {coh:.2f}: consistent with a rock field or natural clutter",
    "dropout": "{frac:.0%} of the box's rows are flagged degraded ({why})",
}


def verify_detection(detection: dict[str, Any], ctx: StripContext,
                     calibration: Calibration | None = None) -> dict[str, Any]:
    """The verification block for one detection. Does not modify the detection.

    The class used for physical expectations is the DETECTOR'S own call
    (`class_withheld` when a confidence floor relabelled it), because the
    question is whether the image looks like what the detector claimed.
    """
    detector_conf = float(detection.get("confidence", 0.0))
    model = detection.get("detector_model")
    claimed = detection.get("class_withheld") or detection.get("class") \
        or detection.get("object_class") or detection.get("class_normalized")
    claimed_norm = normalize_class(claimed)
    p, calibration_basis = calibrate(detector_conf, model, claimed_norm, calibration)
    expectation, expectation_match = class_expectation(claimed_norm)

    block: dict[str, Any] = {
        "version": cfg.VERIFY_VERSION,
        "status": "checked",
        "detector_confidence": _r(detector_conf),
        "detector_model": model,
        "class_evaluated": claimed_norm,
        "class_basis": ("class_withheld: the detector's own call before the confidence floor"
                        if detection.get("class_withheld") else "class"),
        "calibrated_probability": _r(p),
        "calibration": calibration_basis,
        "expectation": {**expectation, "match": expectation_match},
        "heuristic": True,
    }

    g = _geometry(detection, ctx)
    if g is None:
        z = _logit(p)
        block.update({"status": "not_checked",
                      "reason": "no usable bbox_global inside the strip",
                      "terms": [{"name": "prior", "contribution": _r(z)}],
                      "logit": _r(z), "confidence_pct": round(100 * _sigmoid(z), 1),
                      "hard_reasons": [], "reasons": [], "notes": [], "suppressed": False,
                      "height": {"height_m": None, "basis": "not computed: no box"}})
        return block

    sigma = float(np.clip(min(g.w, g.h) / 10.0, 1.0, 4.0))
    win = _Window(ctx, g, sigma)
    notes: list[str] = []
    evidence: dict[str, dict[str, Any]] = {}

    # --- 5. nadir zone -------------------------------------------------------
    nadir = _nadir_zone(ctx, g, bool(expectation.get("water_column")))
    if nadir.get("far_range"):
        notes.append("at far range near the swath edge, where signal-to-noise is lowest")
    nadir.pop("far_range", None)
    evidence["nadir_zone"] = nadir

    # --- 6. dropout ----------------------------------------------------------
    evidence["dropout"] = _dropout(ctx, g)

    # --- 1-2. background, highlight, shadow --------------------------------
    bg = _measure_background(win, g)
    height = {"height_m": None, "basis": "not computed: no seabed background"}
    if bg["median"] is None or bg["median"] < 2.0:
        reason = bg.get("reason", "background too dark to measure contrast against")
        for name in ("acoustic_shadow", "natural_shadow", "man_made_regularity", "rock_clutter"):
            evidence[name] = {"score": 0.0, "unknown": True, "reason": reason}
        notes.append(f"image cues not measured: {reason}")
    else:
        box = win.mask(g.x1, g.x2, g.y1, g.y2)
        box_values = win.blur[box]
        highlight_p95 = float(np.percentile(box_values, 95))
        hc = highlight_p95 / bg["median"] - 1.0
        hc_null = bg["p95"] / bg["median"] - 1.0
        highlight = _clip01((hc - hc_null) / cfg.HIGHLIGHT_FULL_EXCESS)

        if g.far is not None:
            far = _shadow_side(win, g, bg, g.far)
            near = _shadow_side(win, g, bg, -g.far)
            direction = "away from nadir"
        else:
            sides = {s: _shadow_side(win, g, bg, s) for s in (1, -1)}
            best = max(sides, key=lambda s: sides[s].get("score", 0.0))
            far, near = sides[best], sides[-best]
            direction = "unknown nadir: took the darker side"
            notes.append("nadir unknown, so the shadow side was taken as whichever side is darker; "
                         "this can only overstate shadow evidence")
        near_outside = {}
        if g.far is not None:
            # Directionality is judged OUTSIDE the box on the near side: a real
            # object's near side is lit seabed, a symmetric dark blob's is not.
            nominal = win.mask(*_side_rect(g, -g.far, g.shadow_len, False))
            m = nominal & ~win.water & ~win.degraded
            # At least a quarter of the nominal window must be seabed, or the
            # few pixels left say more about the edge of the mask than the object.
            if m.sum() >= max(16, 0.25 * nominal.sum()):
                fraction = _dark_fraction(win.blur[m], bg["median"])
                exc = fraction - bg["dark_fraction"]
                near_outside = {"dark_fraction": fraction, "excess": exc,
                                "score": _clip01(exc / cfg.SHADOW_FULL_EXCESS)}
        shadow_score = far.get("score", 0.0)
        near_score = near_outside.get("score", 0.0)

        box_median = float(np.median(box_values))
        darkness = 1.0 - box_median / bg["median"]
        dark_score = _clip01((darkness - cfg.DARK_BOX_NULL) / cfg.DARK_BOX_FULL)
        sharp = _edge_sharpness(win, g, bg)
        sharp_score = sharp["score"] if sharp["score"] is not None else 0.0
        natural = dark_score * (1.0 - highlight) * (1.0 - sharp_score)
        # A hard-edged shadow implies a hard-edged occluder, so edge sharpness
        # stands in for a highlight when the object itself is no brighter than
        # the seabed (S-7's hull on its record is not).
        support = max(highlight, sharp_score)
        proud = shadow_score * (0.5 + 0.5 * support) * (1.0 - 0.5 * near_score)

        shadow_length = _shadow_length_px(win, g, bg["median"])
        height = _height(ctx, g, shadow_length)

        man_made = _man_made(win, g, bg["median"])
        mesh = man_made["measurements"]["mesh"]["score"]

        evidence["acoustic_shadow"] = {
            "score": proud,
            "measurements": {
                "shadow_direction": direction,
                "background_median": _r(bg["median"], 2), "background_p10": _r(bg["p10"], 2),
                "background_p95": _r(bg["p95"], 2), "background_pixels": bg["pixels"],
                "shadow_window_px": g.shadow_len,
                "shadow_level": _r((1.0 - cfg.SHADOW_DARK_FRACTION) * bg["median"], 2),
                "background_shadow_fraction": _r(bg["dark_fraction"]),
                "far_shadow_fraction": _r(far.get("dark_fraction")),
                "far_excess": _r(far.get("excess")), "far_shadow_score": _r(shadow_score),
                "near_outside_shadow_fraction": _r(near_outside.get("dark_fraction")),
                "near_outside_excess": _r(near_outside.get("excess")),
                "near_outside_score": _r(near_score),
                "box_p95": _r(highlight_p95, 2), "highlight_contrast": _r(hc),
                "highlight_null": _r(hc_null), "highlight_score": _r(highlight),
                "edge_sharpness_score": _r(sharp_score), "object_support": _r(support),
                "shadow_length_px": shadow_length.get("length_px"),
                "shadow_rows": shadow_length.get("rows_with_shadow"),
            },
            "basis": ("heuristic: far_shadow * (0.5 + 0.5*max(highlight, edge_sharpness)) * "
                      "(1 - 0.5*near_side_shadow); shadow scores are shadow-pixel fractions in "
                      "excess of the background ring's"),
        }
        evidence["natural_shadow"] = {
            "score": natural,
            "measurements": {"box_median": _r(box_median, 2), "darkness": _r(darkness),
                             "dark_score": _r(dark_score), "highlight_score": _r(highlight),
                             "edge": sharp},
            "basis": "heuristic: dark_score * (1 - highlight_score) * (1 - edge_sharpness_score)",
        }
        evidence["man_made_regularity"] = man_made
        evidence["rock_clutter"] = _finish_clutter(_clutter(win, g), ctx)
        block["mesh_relief"] = _r(mesh)

    # --- fusion --------------------------------------------------------------
    prior = _logit(p)
    terms = [{"name": "prior", "value": _r(p), "contribution": _r(prior),
              "formula": "logit(calibrated_probability)"}]
    total = prior
    shadow_applicability = float(expectation.get("shadow", 0.5))
    mesh = block.get("mesh_relief") or 0.0
    shadow_applicability *= (1.0 - cfg.MESH_SHADOW_RELIEF * mesh)

    for name, (weight, neutral) in cfg.VERIFY_WEIGHTS.items():
        ev = evidence.get(name, {"score": 0.0, "unknown": True, "reason": "not measured"})
        applicability = 0.0 if ev.get("unknown") else (
            shadow_applicability if name == "acoustic_shadow" else 1.0)
        score = float(ev.get("score", 0.0))
        contribution = weight * applicability * (score - neutral)
        total += contribution
        terms.append({"name": name, "score": _r(score), "neutral": neutral, "weight": weight,
                      "applicability": _r(applicability), "contribution": _r(contribution)})
        ev["score"] = _r(score)

    confidence_pct = round(100.0 * _sigmoid(total), 1)

    hard, reasons = [], []
    for name, threshold in cfg.VERIFY_HARD_REASON_AT.items():
        ev = evidence.get(name, {})
        if ev.get("unknown") or float(ev.get("score") or 0.0) < threshold:
            continue
        hard.append(name)
        m = ev.get("measurements", {})
        if name == "nadir_zone":
            detail = []
            if (m.get("wc_score") or 0) > 0:
                detail.append(f"{m['water_column_fraction']:.0%} of the box inside the water column")
            if (m.get("edge_score") or 0) > 0:
                detail.append(f"centre {m['first_return_distance_px']:.0f} px from the first seabed return")
            if (m.get("line_score") or 0) > 0:
                detail.append(f"centre {m['nadir_distance_px']:.0f} px from the nadir column")
            reasons.append(REASON_TEXT[name].format(detail="; ".join(detail)))
        elif name == "natural_shadow":
            reasons.append(REASON_TEXT[name].format(dark=m.get("darkness") or 0.0))
        elif name == "rock_clutter":
            reasons.append(REASON_TEXT[name].format(n=m.get("similar_blob_count"),
                                                    coh=m.get("coherence") or 0.0))
        elif name == "dropout":
            reasons.append(REASON_TEXT[name].format(
                frac=m.get("degraded_row_fraction") or 0.0,
                why=", ".join(m.get("degraded_reasons") or []) or "degraded"))

    shadow_term = next(t for t in terms if t["name"] == "acoustic_shadow")
    if (shadow_term["contribution"] or 0) <= -0.2:
        reasons.append(
            f"no clear acoustic shadow on the far-range side, which a proud "
            f"'{claimed_norm}' is expected to cast (term {shadow_term['contribution']:+.2f})")
    for ev_name, ev in evidence.items():
        if ev.get("unknown"):
            notes.append(f"{ev_name} not measured: {ev.get('reason')}")

    suppressed = bool(hard) and confidence_pct < cfg.SUPPRESS_BELOW_PCT
    block.update({
        "evidence": evidence,
        "terms": terms,
        "logit": _r(total),
        "confidence_pct": confidence_pct,
        "formula": ("confidence_pct = 100 * sigmoid(sum of terms[].contribution); "
                    "contribution = weight * applicability * (score - neutral)"),
        "hard_reasons": hard,
        "reasons": reasons,
        "notes": notes,
        "suppressed": suppressed,
        "suppress_rule": (f"suppressed when confidence_pct < {cfg.SUPPRESS_BELOW_PCT} AND at "
                          f"least one hard reason ({', '.join(cfg.VERIFY_HARD_REASON_AT)}); "
                          "never deleted"),
        "height": height,
        "box_px": [g.x1, g.y1, g.x2, g.y2],
    })
    return block


def verify_survey(detections: list[dict[str, Any]], contexts: dict[str, StripContext],
                  calibration: Calibration | None = None) -> dict[str, Any]:
    """Verify every detection in place and return a summary.

    Adds to each detection:
      verification     the full block from verify_detection
      confidence_pct   the fused 0..100 score, one decimal
      suppressed       True only with a low score AND a hard artefact reason
      dimensions       {"height_m", "height_basis"} merged into any existing dict

    A detection whose strip has no context is marked `not_checked` with the
    calibrated detector probability as its confidence_pct, and is never
    suppressed: absence of evidence is not evidence of an artefact.
    `confidence` itself is never modified.
    """
    by_reason: dict[str, int] = {}
    checked = suppressed = not_checked = 0
    for detection in detections:
        ctx = contexts.get(str(detection.get("strip") or ""))
        if ctx is None:
            conf = float(detection.get("confidence", 0.0))
            claimed = normalize_class(detection.get("class_withheld") or detection.get("class"))
            p, basis = calibrate(conf, detection.get("detector_model"), claimed, calibration)
            block = {"version": cfg.VERIFY_VERSION, "status": "not_checked",
                     "reason": "no strip image available for this detection's strip",
                     "detector_confidence": _r(conf), "calibrated_probability": _r(p),
                     "calibration": basis,
                     "terms": [{"name": "prior", "value": _r(p), "contribution": _r(_logit(p))}],
                     "confidence_pct": round(100.0 * _sigmoid(_logit(p)), 1),
                     "hard_reasons": [], "reasons": [], "notes": [], "suppressed": False,
                     "height": {"height_m": None, "basis": "not computed: no strip image"}}
        else:
            block = verify_detection(detection, ctx, calibration)
        if block["status"] == "checked":
            checked += 1
        else:
            not_checked += 1
        detection["verification"] = block
        detection["confidence_pct"] = block["confidence_pct"]
        detection["suppressed"] = block["suppressed"]
        dims = detection.get("dimensions")
        dims = dict(dims) if isinstance(dims, dict) else {}
        dims["height_m"] = block["height"].get("height_m")
        dims["height_basis"] = block["height"].get("basis")
        detection["dimensions"] = dims
        if block["suppressed"]:
            suppressed += 1
        for reason in block["hard_reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1

    if suppressed:
        log.info("verification: %d of %d detection(s) suppressed (kept, flagged): %s",
                 suppressed, len(detections), by_reason)
    return {
        "checked": checked,
        "not_checked": not_checked,
        "suppressed": suppressed,
        "by_reason": dict(sorted(by_reason.items())),
        "suppress_below_pct": cfg.SUPPRESS_BELOW_PCT,
        "calibrated": calibration is not None,
        "version": cfg.VERIFY_VERSION,
        "note": ("by_reason counts hard reasons on every checked detection, suppressed or "
                 "not; every cue is a heuristic, see verification.terms on each detection"),
    }


def configuration() -> dict[str, Any]:
    """The verification knobs, for an export's configuration block."""
    return {
        "version": cfg.VERIFY_VERSION,
        "weights": {k: {"weight": w, "neutral": n} for k, (w, n) in cfg.VERIFY_WEIGHTS.items()},
        "hard_reason_at": dict(cfg.VERIFY_HARD_REASON_AT),
        "suppress_below_pct": cfg.SUPPRESS_BELOW_PCT,
        "class_expectations": {k: dict(v) for k, v in cfg.VERIFY_CLASS_EXPECTATIONS.items()},
        "nadir_band_px": cfg.NADIR_BAND_PX,
        "formula": "confidence_pct = 100 * sigmoid(logit(p_calibrated) + "
                   "sum(weight * applicability * (score - neutral)))",
        "basis": "configurable heuristic; not a trained or validated classifier",
    }
