"""Tile-local boxes to survey-wide positions.

A detector reports a box in the coordinate system of the 640-pixel tile it was
shown. That number is meaningless the moment the tile is closed. This module
converts it once, into the survey frame, and everything after this point works
in survey coordinates only.

    bbox_center_x = (x1 + x2) / 2        within the tile
    bbox_center_y = (y1 + y2) / 2
    global_x      = tile_x + bbox_center_x
    global_y      = tile_y + bbox_center_y

global_x and global_y are the canonical spatial identity of a detection. They
exist for every detection in every survey, georeferenced or not, and they are
never overwritten. A latitude and longitude, when navigation allows one, is
attached BESIDE them and never in place of them.

ONE FRAME PER STRIP
    global_x and global_y are offsets inside their own strip. Two strips are
    two coordinate systems: pixel (400, 900) of strip A and pixel (400, 900) of
    strip B are different places on the seabed, and without navigation there is
    nothing that says how far apart. So everything spatial downstream --
    merging, gridding, hotspots -- is scoped to a single strip. Merging across
    strips on pixel proximity alone would be inventing a relationship between
    two frames that have none.

DIMENSIONS
    A strip ingested from a raw sonar log knows what a pixel measures, so a box
    can be given a size in metres: length_m along-track from its height in
    rows, width_m across-track from its width in columns. attach_dimensions
    does that from {strip: (m_per_px_across, m_per_px_along)}, which
    strip_resolutions reads out of a manifest. A strip with no known resolution
    gets explicit nulls and a basis string saying so; the pixel sizes are kept
    either way.
"""

from __future__ import annotations

import logging
from typing import Any

from survey_hazard_map.hazard_geo import Georeference

log = logging.getLogger("deepecho.hazard")

DIMENSION_BASIS_KNOWN = (
    "bounding box in pixels x the strip's resolution from its navigation sidecar: "
    "length_m = height_px * m_per_px_along (along-track), width_m = width_px * "
    "m_per_px_across (across-track, slant-range corrected ground range). The box is "
    "the detector's, so it may include an acoustic shadow or miss a faint edge.")
DIMENSION_BASIS_UNKNOWN = ("strip resolution unknown (no navigation sidecar and no "
                           "m_per_px supplied); dimensions are not inferred")


def to_global(detections: list[dict[str, Any]],
              resolutions: dict[str, tuple[Any, Any]] | None = None) -> list[dict[str, Any]]:
    """Add survey-frame coordinates to every raw detection, in place.

    With `resolutions` ({strip: (m_per_px_across, m_per_px_along)}), metric
    dimensions are attached as well; see attach_dimensions.
    """
    for detection in detections:
        x1, y1, x2, y2 = (float(v) for v in detection["bbox_tile"])
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        tile_x, tile_y = float(detection["tile_x"]), float(detection["tile_y"])
        detection.update({
            "bbox_center_x": round(center_x, 2),
            "bbox_center_y": round(center_y, 2),
            "global_x": round(tile_x + center_x, 2),
            "global_y": round(tile_y + center_y, 2),
            # The box itself in survey coordinates, so a viewer can draw it
            # over the strip without going back to the tile.
            "bbox_global": [round(tile_x + x1, 2), round(tile_y + y1, 2),
                            round(tile_x + x2, 2), round(tile_y + y2, 2)],
            "width_px": round(x2 - x1, 2),
            "height_px": round(y2 - y1, 2),
        })
    if resolutions is not None:
        attach_dimensions(detections, resolutions)
    return detections


def attach_dimensions(detections: list[dict[str, Any]],
                      resolutions: dict[str, tuple[Any, Any]]) -> int:
    """Attach length_m and width_m where the strip's resolution is known.

    length_m is along-track (image rows), width_m across-track (image columns),
    both from the box in pixels, which is kept. A strip with no known
    resolution gets explicit nulls and a basis saying why, never a guessed
    scale. Returns how many detections were given both dimensions.
    """
    measured = 0
    for detection in detections:
        across, along = resolutions.get(str(detection.get("strip") or ""), (None, None))
        width_px = detection.get("width_px")
        height_px = detection.get("height_px")
        if width_px is None or height_px is None:
            x1, y1, x2, y2 = (float(v) for v in detection["bbox_tile"])
            width_px, height_px = x2 - x1, y2 - y1
        width_m = None if across is None else round(float(width_px) * float(across), 3)
        length_m = None if along is None else round(float(height_px) * float(along), 3)
        detection["width_m"] = width_m
        detection["length_m"] = length_m
        detection["m_per_px_across"] = across
        detection["m_per_px_along"] = along
        basis = (DIMENSION_BASIS_KNOWN if width_m is not None or length_m is not None
                 else DIMENSION_BASIS_UNKNOWN)
        detection["dimension_basis"] = basis
        # The nested block is what the export and verification read and extend
        # (verification adds height_m); the flat keys above stay for callers
        # that already use them.
        block = detection.get("dimensions")
        block = dict(block) if isinstance(block, dict) else {}
        block.update({"length_m": length_m, "width_m": width_m,
                      "length_px": round(float(height_px), 2),
                      "width_px": round(float(width_px), 2),
                      "m_per_px_across": across, "m_per_px_along": along,
                      "basis": basis})
        detection["dimensions"] = block
        measured += width_m is not None and length_m is not None
    return measured


def strip_resolutions(rows: list[dict[str, Any]] | None = None,
                      survey_meta: dict[str, Any] | None = None) -> dict[str, tuple[Any, Any]]:
    """{strip: (m_per_px_across, m_per_px_along)} from a manifest.

    Read from the survey block's strips first and the tile rows second, so a
    manifest edited in a spreadsheet (CSV only) still yields resolutions.
    """
    resolutions: dict[str, tuple[Any, Any]] = {}
    for entry in (survey_meta or {}).get("strips", []) or []:
        strip = str(entry.get("strip") or "")
        if strip:
            resolutions[strip] = (entry.get("m_per_px_across"), entry.get("m_per_px_along"))
    for row in rows or []:
        strip = str(row.get("strip") or "")
        if strip and resolutions.get(strip, (None, None)) == (None, None):
            across, along = row.get("m_per_px_across"), row.get("m_per_px_along")
            if across is not None or along is not None:
                resolutions[strip] = (across, along)
    return resolutions


def attach_geo(detections: list[dict[str, Any]],
               references: dict[str, Georeference]) -> int:
    """Attach lat/lon where the strip has navigation. Returns how many got one.

    A detection on a strip with no navigation is given an explicit null rather
    than having the keys left out, so a consumer can tell "not located" from
    "field missing" without guessing.
    """
    located = 0
    for detection in detections:
        reference = references.get(detection.get("strip") or "")
        if reference is None:
            detection["latitude"] = None
            detection["longitude"] = None
            continue
        lat, lon = reference.locate(detection["global_x"], detection["global_y"])
        detection["latitude"] = lat
        detection["longitude"] = lon
        located += lat is not None

    if references and located < len(detections):
        log.info("%d of %d detections carry a geographic position; the rest are "
                 "on strips with no navigation", located, len(detections))
    return located
