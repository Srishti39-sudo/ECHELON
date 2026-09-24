"""Whole strips back from a tile set, and the navigation that came with them.

Module 2 is handed tiles and a manifest, not the strips they were cut from. Two
later stages need the strip itself rather than a 640-pixel window of it:

    verification   an acoustic shadow can fall across a tile boundary, and the
                   background a detection is judged against is wider than the
                   box it surrounds
    dimensions     a size in metres needs the strip's resolution, which only a
                   raw sonar log's sidecar records

WHY THE STRIP IS REBUILT FROM TILES
    The original image may be anywhere, or gone. The tiles are always beside
    the manifest, and they are the pixels the detector was actually shown --
    after any denoising -- so a cue measured on them judges exactly what the
    model saw. Tiles skipped for low content are left at zero, which is what
    they were: blank margin.

WHAT IS NEVER DONE HERE
    No pixel is resampled, rotated or moved. The rebuilt strip's grid is the
    survey's coordinate system, so bbox_global indexes it directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("deepecho.hazard")


def rebuild_strips(rows: list[dict[str, Any]], tiles_dir: Path) -> dict[str, Any]:
    """{strip: 2-D float32 grey array} assembled from the manifest's tiles.

    A strip whose tiles cannot be read is left out, and the stages that wanted
    it record that they could not check it. Overlapping tiles hold the same
    source pixels, so the order they are pasted in does not matter.
    """
    import numpy as np
    from PIL import Image

    by_strip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strip.setdefault(str(row.get("strip") or ""), []).append(row)

    strips: dict[str, Any] = {}
    for strip, strip_rows in by_strip.items():
        width = max(int(r.get("width") or 0) for r in strip_rows)
        height = max(int(r.get("height") or 0) for r in strip_rows)
        if width <= 0 or height <= 0:
            log.warning("strip %s has no recorded size in the manifest; it cannot be "
                        "rebuilt for verification", strip)
            continue
        canvas = np.zeros((height, width), dtype=np.float32)
        try:
            for row in strip_rows:
                with Image.open(Path(tiles_dir) / str(row["tile"])) as tile:
                    array = np.asarray(tile.convert("L"), dtype=np.float32)
                x, y = int(row["x"]), int(row["y"])
                h = min(array.shape[0], height - y)
                w = min(array.shape[1], width - x)
                canvas[y:y + h, x:x + w] = array[:h, :w]
        except Exception as exc:
            log.warning("strip %s could not be rebuilt from its tiles (%s: %s); its "
                        "detections will not be verified", strip, type(exc).__name__, exc)
            continue
        strips[strip] = canvas
    return strips


def load_sidecars(survey_meta: dict[str, Any], manifest_dir: Path) -> dict[str, dict[str, Any]]:
    """{strip: sidecar dict} for strips that came from a raw sonar log.

    The manifest records sidecar paths relative to its own directory. A plain
    image survey has none, and an empty result is the correct answer for it.
    """
    navigation = survey_meta.get("navigation") or {}
    sidecars: dict[str, dict[str, Any]] = {}
    for strip, relative in (navigation.get("sidecars") or {}).items():
        path = Path(manifest_dir) / str(relative)
        try:
            sidecars[str(strip)] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("sidecar for strip %s could not be read (%s); it is treated as "
                        "a strip with no navigation", strip, exc)
    return sidecars


def strip_resolutions(sidecars: dict[str, dict[str, Any]],
                      rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """{strip: (m_per_px_across, m_per_px_along)} wherever it is actually known.

    From the sidecar first, which is where a raw log's resampling recorded it,
    then from manifest columns. A strip with neither is absent, and its
    detections keep pixel sizes only.
    """
    resolutions: dict[str, tuple[float, float]] = {}
    for strip, sidecar in sidecars.items():
        across, along = sidecar.get("m_per_px_across"), sidecar.get("m_per_px_along")
        if across and along:
            resolutions[strip] = (float(across), float(along))
    for row in rows:
        strip = str(row.get("strip") or "")
        if strip in resolutions:
            continue
        across, along = row.get("m_per_px_across"), row.get("m_per_px_along")
        if across not in (None, "") and along not in (None, ""):
            resolutions[strip] = (float(across), float(along))
    return resolutions
