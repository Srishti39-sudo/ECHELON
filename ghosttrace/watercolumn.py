"""Is the net still fishing? Water-column echo enrichment beside a detection.

A ghost net keeps catching for years: fish are drawn to the structure, get
caught, and their bodies draw scavengers, which get caught in turn. A net that
has stopped catching is a clean-up job; a net that is still catching is a
rescue. Side-scan sonar records the water column between the towfish and the
seabed on every ping, and fish, schools and scavengers show up there as echo
clusters. This module asks one narrow question of that record:

    Are there more echo clusters in the water column beside this object than
    there are, on average, elsewhere along the same line on the same side?

It does not identify fish, count them, or say they are entangled. The answer is
an enrichment ratio and a score derived from it, with every number that went
into both kept in `evidence`.

INPUT (contract A, from the ingest agent)
    {strip}.nav.json  nadir_col, m_per_px_along, port_is_left, rows[...],
                      water_column: {path, m_per_bin, max_range_m}
    {strip}.wc.npz    port, starboard: rows x bins float32, NaN beyond bottom
                      bottom_range_m: rows
    Rows align with the strip image rows, so a detection's bbox_global y range
    is directly a range of pings.

METHOD
    1. Side. The detection is on the port channel when its centre column is
       left of nadir_col and port_is_left is true (mirrored otherwise).
    2. Background, per range bin, over the whole line on that side: median and
       MAD of the finite cells. A per-bin background removes the range-dependent
       gain and spreading loss that a single global threshold would mistake for
       echoes near the transducer.
    3. Echo mask: robust z = (value - median) / (1.4826 * MAD) > WC_Z_THRESHOLD,
       excluding the first WC_RINGDOWN_BINS bins (transducer ringdown) and the
       WC_BOTTOM_GUARD_BINS bins above each ping's bottom return (bottom
       sidelobe), and every NaN cell (below the seabed).
    4. Near window: the detection's rows expanded along-track by WC_WINDOW_M
       each side (converted with m_per_px_along). Connected components (8-
       connectivity) of the mask with at least WC_MIN_CLUSTER_CELLS cells are
       echo clusters, labelled once over the whole side; a cluster belongs to
       the ping row of its centroid.
    5. Background: the cluster rate per ping row over every row on the same
       side that is clear of detections -- this detection's window excluded
       whole, every other detection excluded over its box plus
       WC_EXCLUDE_MARGIN_M -- scaled to the near window's length. Unavailable
       when fewer than WC_MIN_BACKGROUND_WINDOWS window-lengths of rows are
       clear.
    6. enrichment_ratio = near_clusters / (background_per_window + WC_EPSILON)
       score = 1 / (1 + exp(-K * (ln(max(enrichment, floor)) - ln(MID))))
       level from WC_LEVELS, except that "high" needs at least
       WC_MIN_CLUSTERS_HIGH clusters in the near window: on a quiet line one
       or two blobs can reach a high ratio, and that is not a school. Below
       the gate the level is capped at "moderate" and evidence.level_capped
       says why; the score is left as computed.

Every constant is in config_core with its rationale. The whole method is a
heuristic that has not been validated on real ghost-net data, and the output
says so in `limitations`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ghosttrace import config_core as cfg

BASIS = ("echo clusters (robust z-score over per-range-bin median/MAD background, "
         "connected components) in the water column beside the detection, compared "
         "with the cluster rate over every ping on the same side of the same line "
         "that is clear of detections, scaled to the same window length; "
         + cfg.HEURISTIC_LABEL)


def _unavailable(reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        "available": False,
        "score": None,
        "level": "unknown",
        "reason": reason,
        "evidence": {
            "echo_clusters_near": None, "echo_area_near_m2": None,
            "background_clusters_per_window": None, "enrichment_ratio": None,
            "window_m": cfg.WC_WINDOW_M, "side": None, **evidence},
        "basis": BASIS,
        "limitations": cfg.WC_LIMITATIONS,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def locate_sidecars(survey_dir: Any, strip: str,
                    manifest: dict[str, Any] | None = None) -> tuple[Path | None, Path | None]:
    """(nav sidecar path, water-column npz path) for a strip, or None for each.

    Preference: the manifest's survey.navigation.sidecars / water_columns
    (paths relative to the survey directory), then the sidecar's own
    water_column.path (relative to the sidecar), then the conventional
    <survey>/nav/{strip}.nav.json and {strip}.wc.npz.
    """
    survey_dir = Path(survey_dir)
    if manifest is None:
        manifest = _load_json(survey_dir / "manifest.json") or {}
    navigation = (manifest.get("survey") or {}).get("navigation") or {}

    def resolve(rel: Any, base: Path) -> Path | None:
        if not rel:
            return None
        p = Path(str(rel))
        p = p if p.is_absolute() else base / p
        return p if p.is_file() else None

    nav = resolve((navigation.get("sidecars") or {}).get(strip), survey_dir)
    if nav is None:
        nav = resolve(f"nav/{strip}.nav.json", survey_dir)
    wc = resolve((navigation.get("water_columns") or {}).get(strip), survey_dir)
    if wc is None and nav is not None:
        sidecar = _load_json(nav) or {}
        declared = (sidecar.get("water_column") or {}).get("path")
        wc = resolve(declared, nav.parent)
        if wc is None and declared:
            wc = resolve(Path(str(declared)).name, nav.parent)
    if wc is None:
        wc = resolve(f"nav/{strip}.wc.npz", survey_dir)
    return nav, wc


def _side_background(side: Any) -> tuple[Any, Any]:
    """Per-range-bin (median, scaled MAD) over the finite cells of one side."""
    import numpy as np
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(side, axis=0)
        mad = np.nanmedian(np.abs(side - median[None, :]), axis=0) * 1.4826
    # A bin that is flat (MAD 0) or entirely below the seabed has no usable
    # spread. Fall back to the side's overall MAD so the bin is not all-echo.
    finite = np.isfinite(side)
    overall = float(np.nanmedian(np.abs(side[finite] - np.nanmedian(side[finite])))) * 1.4826 \
        if finite.any() else 1.0
    floor = max(overall, 1e-6)
    mad = np.where(np.isfinite(mad) & (mad > 1e-9), mad, floor)
    return median, mad


def echo_mask(side: Any, bottom_range_m: Any, m_per_bin: float) -> Any:
    """Boolean rows x bins mask of cells counted as water-column echo."""
    import numpy as np

    side = np.asarray(side, dtype=np.float32)
    rows, bins = side.shape
    median, mad = _side_background(side)
    with np.errstate(invalid="ignore"):
        z = (side - median[None, :]) / mad[None, :]
        mask = np.isfinite(z) & (z > cfg.WC_Z_THRESHOLD)
    mask[:, :min(cfg.WC_RINGDOWN_BINS, bins)] = False
    if bottom_range_m is not None and m_per_bin and m_per_bin > 0:
        bottom = np.asarray(bottom_range_m, dtype=float)
        limit = np.floor(bottom / float(m_per_bin)) - cfg.WC_BOTTOM_GUARD_BINS
        limit = np.where(np.isfinite(limit), limit, bins).astype(int)
        per_row = np.full(rows, bins, dtype=int)
        n = min(rows, len(limit))
        per_row[:n] = limit[:n]
        mask &= np.arange(bins)[None, :] < per_row[:, None]
    return mask


def count_clusters(mask_window: Any) -> tuple[int, int]:
    """(clusters with >= WC_MIN_CLUSTER_CELLS cells, total cells in them)."""
    import numpy as np
    from scipy import ndimage

    if mask_window.size == 0 or not mask_window.any():
        return 0, 0
    labels, n = ndimage.label(mask_window, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel())[1:]
    keep = sizes >= cfg.WC_MIN_CLUSTER_CELLS
    return int(keep.sum()), int(sizes[keep].sum())


def _cluster_rows(mask: Any) -> tuple[Any, Any]:
    """(centroid ping row, cell count) for every cluster of >= WC_MIN_CLUSTER_CELLS cells."""
    import numpy as np
    from scipy import ndimage

    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= cfg.WC_MIN_CLUSTER_CELLS) + 1
    if keep.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    centroids = ndimage.center_of_mass(mask, labels, keep)
    rows = np.array([int(c[0]) for c in centroids], dtype=int)
    return rows, sizes[keep - 1]


def logistic_score(enrichment: float) -> float:
    """The documented score: logistic in ln(enrichment), 0.5 at WC_LOGISTIC_MID."""
    e = max(float(enrichment), cfg.WC_ENRICHMENT_FLOOR)
    x = cfg.WC_LOGISTIC_K * (math.log(e) - math.log(cfg.WC_LOGISTIC_MID))
    return 1.0 / (1.0 + math.exp(-x))


def level_for(score: float | None) -> str:
    if score is None:
        return "unknown"
    for name, floor in cfg.WC_LEVELS:
        if score >= floor:
            return name
    return cfg.WC_LEVELS[-1][0]


def gated_level(score: float | None, near_clusters: int) -> tuple[str, str | None]:
    """(level, why it was capped or None). "high" needs WC_MIN_CLUSTERS_HIGH clusters."""
    level = level_for(score)
    if level == "high" and int(near_clusters) < cfg.WC_MIN_CLUSTERS_HIGH:
        return "moderate", (f"score {score:.3f} would be high, but only {int(near_clusters)} echo "
                            f"cluster(s) sit near the object and 'high' needs at least "
                            f"{cfg.WC_MIN_CLUSTERS_HIGH}; capped at moderate")
    return level, None


def _row_window(detection: dict[str, Any], half_rows: int, n_rows: int) -> tuple[int, int] | None:
    box = detection.get("bbox_global")
    if box and len(box) == 4:
        y0, y1 = float(box[1]), float(box[3])
    elif detection.get("global_y") is not None:
        y0 = y1 = float(detection["global_y"])
    else:
        return None
    r0 = max(0, int(math.floor(min(y0, y1))) - half_rows)
    r1 = min(n_rows, int(math.ceil(max(y0, y1))) + half_rows + 1)
    return (r0, r1) if r1 > r0 else None


def activity_evidence(detection: dict[str, Any], survey_dir: Any, *,
                      other_detections: list[dict[str, Any]] | None = None,
                      manifest: dict[str, Any] | None = None,
                      cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """Water-column activity block for one detection. See module docstring.

    other_detections: every detection on the survey (targets or not). Those on
    the same strip are excluded from the control windows so a second net, or a
    wreck with its own fish, cannot inflate the background.
    cache: optional dict reused across calls to avoid reloading the npz.
    """
    import numpy as np

    strip = (detection.get("provenance") or {}).get("strip") or detection.get("strip")
    if not strip:
        return _unavailable("detection records no strip, so its water column cannot be found")

    cache = cache if cache is not None else {}
    key = f"strip:{strip}"
    if key not in cache:
        nav_path, wc_path = locate_sidecars(survey_dir, strip, manifest)
        if nav_path is None or wc_path is None:
            cache[key] = ("unavailable",
                          "no water-column record for this strip (plain image survey, or a "
                          "raw log ingested without water-column samples)"
                          if wc_path is None else
                          "water-column record present but no navigation sidecar, so "
                          "nadir column and along-track scale are unknown")
        else:
            sidecar = _load_json(nav_path)
            try:
                with np.load(wc_path) as npz:
                    arrays = {name: np.asarray(npz[name]) for name in
                              ("port", "starboard", "bottom_range_m") if name in npz.files}
            except Exception as exc:  # corrupt or unreadable npz
                cache[key] = ("unavailable", f"water-column file unreadable: {type(exc).__name__}")
            else:
                if sidecar is None:
                    cache[key] = ("unavailable", "navigation sidecar unreadable")
                elif "port" not in arrays or "starboard" not in arrays:
                    cache[key] = ("unavailable", "water-column file lacks port/starboard arrays")
                else:
                    cache[key] = ("ok", sidecar, arrays, {})
    entry = cache[key]
    if entry[0] != "ok":
        return _unavailable(entry[1])
    _, sidecar, arrays, masks = entry

    along = sidecar.get("m_per_px_along")
    nadir = sidecar.get("nadir_col")
    wc_meta = sidecar.get("water_column") or {}
    m_per_bin = wc_meta.get("m_per_bin")
    if not along or along <= 0:
        return _unavailable("sidecar has no m_per_px_along, so the along-track window "
                            "cannot be converted from metres to pings")
    if nadir is None:
        return _unavailable("sidecar has no nadir_col, so the detection's side is unknown")
    if not m_per_bin or m_per_bin <= 0:
        return _unavailable("sidecar water_column has no m_per_bin, so echo area and the "
                            "bottom guard cannot be computed")

    x = detection.get("global_x")
    if x is None and detection.get("bbox_global"):
        x = (float(detection["bbox_global"][0]) + float(detection["bbox_global"][2])) / 2.0
    if x is None:
        return _unavailable("detection has no across-track position")
    left = float(x) < float(nadir)
    port_is_left = bool(sidecar.get("port_is_left", True))
    side = "port" if left == port_is_left else "starboard"

    if side not in masks:
        masks[side] = echo_mask(arrays[side], arrays.get("bottom_range_m"), float(m_per_bin))
    mask = masks[side]
    n_rows = mask.shape[0]
    half_rows = max(1, int(round(cfg.WC_WINDOW_M / float(along))))

    near = _row_window(detection, half_rows, n_rows)
    if near is None:
        return _unavailable("detection rows fall outside the water-column record", side=side)
    length = near[1] - near[0]

    cell_area = float(m_per_bin) * float(along)

    # Clusters are labelled once over the whole side and each is assigned to
    # the ping row of its centroid, so near and background are counted by the
    # same rule and a school straddling a window edge is counted once.
    labels_key = f"labels:{side}"
    if labels_key not in masks:
        masks[labels_key] = _cluster_rows(mask)
    centroid_rows, cluster_cells = masks[labels_key]

    in_near = (centroid_rows >= near[0]) & (centroid_rows < near[1])
    near_clusters = int(in_near.sum())
    near_cells = int(cluster_cells[in_near].sum())

    # Background is a RATE over every ping row not excluded, scaled to the
    # near window's length. It used to be the mean over whole control windows
    # of the same length that touched no detection, which on a short line with
    # several contacts left no window at all. Exclusions: this detection's own
    # window in full, and every other detection on the strip (either side: a
    # big object can shadow both channels' water column) over its box plus
    # WC_EXCLUDE_MARGIN_M, so a second net's fish cannot inflate the background.
    free_rows = np.ones(n_rows, dtype=bool)
    free_rows[near[0]:near[1]] = False
    margin_rows = max(1, int(round(cfg.WC_EXCLUDE_MARGIN_M / float(along))))
    for other in other_detections or []:
        if other is detection or other.get("id") == detection.get("id"):
            continue
        if ((other.get("provenance") or {}).get("strip") or other.get("strip")) != strip:
            continue
        w = _row_window(other, margin_rows, n_rows)
        if w:
            free_rows[w[0]:w[1]] = False
    n_free = int(free_rows.sum())
    background_clusters = int(free_rows[np.clip(centroid_rows, 0, n_rows - 1)].sum()) \
        if len(centroid_rows) else 0

    evidence = {
        "echo_clusters_near": near_clusters,
        "echo_area_near_m2": round(near_cells * cell_area, 2),
        "background_clusters_per_window": None,
        "enrichment_ratio": None,
        "window_m": cfg.WC_WINDOW_M,
        "side": side,
        "window_rows": [near[0], near[1]],
        "background_rows": n_free,
        "background_clusters": background_clusters,
        "background_windows_equivalent": round(n_free / length, 2) if length else 0.0,
        "exclude_margin_m": cfg.WC_EXCLUDE_MARGIN_M,
        "epsilon": cfg.WC_EPSILON,
        "z_threshold": cfg.WC_Z_THRESHOLD,
        "min_cluster_cells": cfg.WC_MIN_CLUSTER_CELLS,
        "cell_area_m2": round(cell_area, 4),
        "area_note": "echo area is measured in the echogram plane (range x along-track), "
                     "not a physical cross-section of the scatterers",
    }
    if n_free < cfg.WC_MIN_BACKGROUND_WINDOWS * length:
        out = _unavailable(
            f"only {n_free} ping rows of this line are clear of detections, less than "
            f"{cfg.WC_MIN_BACKGROUND_WINDOWS:g} window(s) of {length}, so there is no "
            "background to compare against", side=side)
        out["evidence"].update({k: v for k, v in evidence.items()
                                if k not in ("background_clusters_per_window", "enrichment_ratio")})
        return out

    background = background_clusters / n_free * length
    enrichment = near_clusters / (background + cfg.WC_EPSILON)
    score = logistic_score(enrichment)
    level, capped = gated_level(score, near_clusters)
    evidence.update({
        "background_clusters_per_window": round(background, 4),
        "enrichment_ratio": round(enrichment, 4),
        "high_requires_clusters": cfg.WC_MIN_CLUSTERS_HIGH,
        "level_capped": capped,
    })
    return {
        "available": True,
        "score": round(score, 4),
        "level": level,
        "evidence": evidence,
        "formula": (f"enrichment = echo_clusters_near / (background_clusters_per_window + "
                    f"{cfg.WC_EPSILON}); score = 1/(1+exp(-{cfg.WC_LOGISTIC_K}*(ln(max("
                    f"enrichment,{cfg.WC_ENRICHMENT_FLOOR})) - ln({cfg.WC_LOGISTIC_MID}))))"),
        "basis": BASIS,
        "limitations": cfg.WC_LIMITATIONS,
    }
