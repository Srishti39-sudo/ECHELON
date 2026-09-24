"""export.json into a standalone map.html.

The file this writes has no dependencies once it exists. No server, no tile
provider, no network, no sibling files. The sonar imagery is embedded in it as
a data URI and the Leaflet and Folium assets come from the CDN links Folium
emits, with the survey itself readable even if those fail. Copy map.html onto a
USB stick, open it on a machine that has never seen this project, and it works.

TWO COORDINATE MODES, AND THEY LOOK DIFFERENT ON PURPOSE

A survey with no navigation is drawn on Leaflet's CRS.Simple, which is a plain
pixel plane, with the sonar strip itself as the base layer. There is no world
map underneath because there is no world position to put it at. The header says
"Relative Survey Coordinates (px)" and every coordinate on the map is written
as a pixel offset. Nothing is formatted as a latitude.

That is not a degraded mode. For relative survey data it is the correct map:
the contacts are drawn on the acoustic image they were found in, at the pixels
they were found at, which is more useful to a survey team than the same dots
floating over a road atlas would be. It also means the map needs nothing from
the internet, so it cannot fail on a conference floor.

A georeferenced survey is drawn on real coordinates instead, with the strip
placed at its geographic footprint and an optional street basemap under it. The
basemap is the only thing here that needs a network, it is off by default, and
the survey renders completely without it.

MULTIPLE STRIPS IN RELATIVE MODE
Two strips are two coordinate frames. To show them together they are laid out
side by side with a gutter, and the map says in writing that the separation is
a display convenience with no geographic meaning. Within each strip every
position is exact.

THE STRIP IMAGE IS STITCHED FROM THE TILES
Not read from the original survey file, which may be long gone by the time
anyone looks at the map. Tiles are pasted back at their manifest offsets, so
the base layer shows exactly what the detector was shown -- including the gaps
where MIN_CONTENT dropped a blank tile, which is information rather than a
defect.

A STRIP THAT DOES NOT RUN NORTH-UP IS RESAMPLED, FOR DISPLAY ONLY
A strip ingested from a raw XTF or JSF log is located per ping: every row has
its own position and heading, so the strip lies along the towfish track at
whatever angle, and with whatever curve, the vessel steered. Leaflet can only
stretch an image between two latitudes and two longitudes. So the strip is
forward-mapped through its own Georeference onto a north-up grid and written
as a transparent PNG covering the strip's footprint. That picture exists to
be looked at. Detections, hotspots and every exported position are computed
by the engine from the original strip pixels and the navigation, never from
the resampled picture, and the map says so in its footer.

VERIFICATION IS SHOWN, NOT HIDDEN
A detection hazard_verify marked `suppressed` is a likely false positive. It
is kept in export.json with its reasons and left out of hotspots, and here it
is drawn as a grey hollow marker on its own layer, off when the map opens,
with the reasons in its popup. An operator who disagrees with the filter can
switch the layer on and look. Exports written before verification existed
have none of these fields, and the map renders them exactly as it always did.
"""

from __future__ import annotations

import abc
import base64
import html as html_lib
import io
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

from survey_hazard_map import hazard_assets
from survey_hazard_map import hazard_config as cfg
from survey_hazard_map import hazard_theme as theme
from survey_hazard_map.hazard_geo import Georeference, radii_of_curvature, references_from_manifest

# Exact per-ping navigation, when a survey came from a raw sonar log. Resolved
# once here rather than assumed, as hazard_map does, so the map still builds
# on a checkout that predates raw ingest.
try:
    from survey_hazard_map.hazard_geo import references_for_survey
except ImportError:  # pragma: no cover - depends on the checkout
    references_for_survey = None

log = logging.getLogger("deepecho.hazard")

# Below this the transform is treated as axis-aligned and the sonar image can
# be placed as a north-up rectangle. Above it the strip is rotated relative to
# north, a rectangle would be a lie, and the imagery is resampled instead.
ROTATION_TOLERANCE = 0.05

# The resampled north-up picture of a rotated or ping-navigated strip. Its
# long side never exceeds this many pixels, whatever the strip's size.
RESAMPLE_MAX_EDGE = 2048
# Strip pixels are located through the Georeference on a lattice this many
# pixels apart along each axis, and interpolated in between. Every supported
# transform is linear across a row, so the across-track spacing can be coarse;
# ping navigation turns with the heading from row to row, so rows are dense.
RESAMPLE_LATTICE_ACROSS_PX = 32
RESAMPLE_LATTICE_ALONG_PX = 4
# A cell with no strip pixel is filled from its neighbours only when at least
# this many of its eight neighbours hold one. An interior pinhole has eight;
# a cell just outside a straight footprint edge has three, and stays empty.
RESAMPLE_FILL_MIN_NEIGHBOURS = 5

# Wording and styling for verification. Local rather than in hazard_theme so
# the theme file, which other code reads, keeps its existing keys.
FILTERED_LAYER = "Filtered false positives"
FILTERED_COLOR = theme.COLOR["text_faint"]
HARD_REASON_LABEL = {
    "nadir_zone": "nadir / water-column artefact",
    "natural_shadow": "natural shadow or depression",
    "rock_clutter": "rock field or natural clutter",
    "dropout": "degraded sonar rows (dropout or motion)",
}
RESAMPLED_NOTE = (
    "Sonar imagery for {strips} is resampled onto a north-up latitude/longitude "
    "grid for display only. Detections, hotspots and every exported position are "
    "computed from the original strip pixels and the survey's navigation, never "
    "from this picture.")


class MapError(RuntimeError):
    """The map could not be built from what it was given."""


# --- frames ----------------------------------------------------------------


class Frame(abc.ABC):
    """Survey coordinates to map coordinates, for one of the two modes.

    Abstract on purpose rather than by convention: every drawing routine below
    goes through `point` and `rect` and neither mode may quietly inherit a
    half-implementation. Both subclasses must answer both questions.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode

    @abc.abstractmethod
    def point(self, strip: str, gx: float, gy: float,
              lat: float | None, lon: float | None) -> list[float] | None:
        """Where a single position lands on the map, or None if it cannot."""

    @abc.abstractmethod
    def rect(self, strip: str, x0: float, y0: float, x1: float, y1: float
             ) -> list[list[float]] | None:
        """The map bounds of a pixel rectangle, or None if it cannot be placed."""


class PixelFrame(Frame):
    """CRS.Simple. Leaflet's y grows upward and a sonar image's grows down, so
    every y is negated. x is shifted per strip by the side-by-side layout."""

    def __init__(self, offsets: dict[str, float]) -> None:
        super().__init__("pixel")
        self.offsets = offsets

    def point(self, strip, gx, gy, lat=None, lon=None):
        return [-float(gy), float(gx) + self.offsets.get(strip, 0.0)]

    def rect(self, strip, x0, y0, x1, y1):
        shift = self.offsets.get(strip, 0.0)
        return [[-float(y1), float(x0) + shift], [-float(y0), float(x1) + shift]]

    def aligned(self, strip: str) -> bool:
        return True


class GeoFrame(Frame):
    """Real coordinates. Rectangles are located through the strip's own
    transform rather than assumed, so a rotated survey stays correct."""

    def __init__(self, references: dict[str, Georeference]) -> None:
        super().__init__("geo")
        self.references = references

    def point(self, strip, gx, gy, lat=None, lon=None):
        if lat is not None and lon is not None:
            return [float(lat), float(lon)]
        reference = self.references.get(strip)
        if reference is None:
            return None
        located = reference.locate(float(gx), float(gy))
        return None if located[0] is None else [located[0], located[1]]

    def rect(self, strip, x0, y0, x1, y1):
        reference = self.references.get(strip)
        if reference is None:
            return None
        corners = [reference.locate(x, y) for x, y in
                   ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        if any(c[0] is None for c in corners):
            return None
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        return [[min(lats), min(lons)], [max(lats), max(lons)]]

    def aligned(self, strip: str) -> bool:
        """Whether a pixel rectangle on this strip is a north-up rectangle."""
        reference = self.references.get(strip)
        return reference is None or _axis_aligned(reference)[0]

    def outline(self, strip: str, x0: float, y0: float, x1: float, y1: float,
                per_edge: int = 4) -> list[list[float]] | None:
        """A pixel rectangle's true footprint, as a closed ring of positions.

        For a strip that does not run north-up, rect() would draw the bounding
        box of the rotated cell, which is larger than the cell and in the wrong
        orientation. Each edge is sampled so a curving ping track bends it.
        """
        reference = self.references.get(strip)
        if reference is None:
            return None
        ring: list[list[float]] = []
        edges = (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)))
        for (ax, ay), (bx, by) in edges:
            for step in range(per_edge):
                t = step / per_edge
                lat, lon = reference.locate(ax + (bx - ax) * t, ay + (by - ay) * t)
                if lat is None or lon is None:
                    return None
                ring.append([float(lat), float(lon)])
        return ring


# --- imagery ---------------------------------------------------------------


def _stitch_canvas(tiles_dir: Path, rows: list[dict[str, Any]]) -> tuple[Any, int, int] | None:
    """One strip rebuilt from its tiles at full resolution, as a greyscale PIL
    image, with the strip's real (width, height). None when no tile exists."""
    from PIL import Image

    placed = [r for r in rows if (tiles_dir / str(r["tile"])).is_file()]
    if not placed:
        return None

    width = int(max(int(r["x"]) + int(r.get("tile_width") or cfg.TILE) for r in placed))
    height = int(max(int(r["y"]) + int(r.get("tile_height") or cfg.TILE) for r in placed))
    width = int(placed[0].get("width") or width)
    height = int(placed[0].get("height") or height)

    # Mid-grey rather than black for the gaps: a dropped low-content tile
    # should read as "not examined", not as "examined and found empty".
    canvas = Image.new("L", (width, height), color=44)
    for row in placed:
        with Image.open(tiles_dir / str(row["tile"])) as tile:
            canvas.paste(tile.convert("L"), (int(row["x"]), int(row["y"])))
    return canvas, width, height


def _jpeg_preview(canvas: Any) -> str:
    """A downscaled JPEG data URI of a stitched strip, for file size."""
    from PIL import Image

    width, height = canvas.size
    preview = canvas
    longest = max(width, height)
    if longest > theme.PREVIEW_MAX_EDGE:
        scale = theme.PREVIEW_MAX_EDGE / longest
        preview = canvas.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                                Image.LANCZOS)

    buffer = io.BytesIO()
    preview.convert("L").save(buffer, "JPEG", quality=theme.PREVIEW_JPEG_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _stitch_strip(tiles_dir: Path, rows: list[dict[str, Any]]) -> tuple[str, int, int] | None:
    """A data URI preview of one strip, rebuilt from its tiles.

    Returns (data_uri, original_width, original_height). The preview is
    downscaled for file size; the returned dimensions are the strip's real ones,
    because that is what the overlay is positioned with.
    """
    stitched = _stitch_canvas(tiles_dir, rows)
    if stitched is None:
        return None
    canvas, width, height = stitched
    return _jpeg_preview(canvas), width, height


def _axis_aligned(reference: Georeference) -> tuple[bool, bool]:
    """(can be drawn as a north-up rectangle, image rows run north to south).

    Only a transform that is linear in both pixel axes, with no rotation, can be
    drawn as a rectangle. Ping navigation follows the towfish heading row by
    row, so it never is; it has no coefficients to inspect, and reading them
    would fail. Along-track navigation does not resolve across-track position
    at all, so it cannot place imagery by any method.
    """
    mode = reference.mode
    if mode in ("along_track", "ping"):
        return False, True
    if mode == "corners":
        corners = reference.detail.get("corners") or {}
        try:
            tl, tr, bl = corners["top_left"], corners["top_right"], corners["bottom_left"]
        except KeyError:
            return False, True
        # Along a row latitude must not change, and down a column longitude
        # must not, each relative to the change along the other axis.
        lat_rotation = abs(tr[0] - tl[0]) / (abs(bl[0] - tl[0]) or 1e-12)
        lon_rotation = abs(bl[1] - tl[1]) / (abs(tr[1] - tl[1]) or 1e-12)
        aligned = lat_rotation < ROTATION_TOLERANCE and lon_rotation < ROTATION_TOLERANCE
        return aligned, bl[0] < tl[0]
    lat_coefficients = reference.detail.get("lat_coefficients")
    lon_coefficients = reference.detail.get("lon_coefficients")
    if not lat_coefficients or not lon_coefficients:
        return False, True
    lat_px, lat_py, _ = lat_coefficients
    lon_px, lon_py, _ = lon_coefficients
    lat_rotation = abs(lat_px) / (abs(lat_py) or 1e-12)
    lon_rotation = abs(lon_py) / (abs(lon_px) or 1e-12)
    aligned = lat_rotation < ROTATION_TOLERANCE and lon_rotation < ROTATION_TOLERANCE
    # lat falling as the row index rises means row 0 is the northern edge.
    return aligned, lat_py < 0


def _resamplable(reference: Georeference) -> bool:
    """Whether the strip's transform places every pixel, so it can be resampled.

    Along-track navigation collapses the across-track axis onto the track line;
    forward-mapping through it would paint the whole swath onto one line.
    """
    return reference.mode in ("ping", "affine", "corners")


def _lattice(size: int, step: int) -> Any:
    """Pixel-centre coordinates every `step` pixels, always including both ends."""
    import numpy as np

    points = np.arange(0, size, max(1, step), dtype=float)
    if points[-1] != size - 1:
        points = np.append(points, size - 1)
    if len(points) == 1:
        points = np.append(points, points[0])
    return points + 0.5


def _axis_weights(lattice: Any, samples: Any) -> tuple[Any, Any]:
    """Index of the lattice interval each sample falls in, and its fraction."""
    import numpy as np

    index = np.clip(np.searchsorted(lattice, samples, side="right") - 1, 0, len(lattice) - 2)
    span = lattice[index + 1] - lattice[index]
    fraction = np.where(span > 0, (samples - lattice[index]) / np.where(span > 0, span, 1.0), 0.0)
    return index, np.clip(fraction, 0.0, 1.0)


def _mercator_y(lat: Any) -> Any:
    """Web-Mercator northing, in radians. Leaflet stretches an ImageOverlay
    linearly in this, not in latitude, so output rows are spaced in it."""
    import numpy as np

    return np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))


def _resample_north_up(grey: Any, reference: Georeference,
                       max_edge: int = RESAMPLE_MAX_EDGE) -> dict[str, Any] | None:
    """A strip forward-mapped onto a north-up grid, as a transparent PNG.

    grey        H x W uint8 array, the stitched strip
    reference   the strip's Georeference; locate(x, y) -> (lat, lon)

    Returns {"data_uri", "bounds" [[south, west], [north, east]], "footprint"
    ring of [lat, lon], "size" (w, h), "stride"} or None when nothing locates.

    numpy only. Pixel centres (c + 0.5, r + 0.5), the convention every
    detection position uses, are located through reference.locate on a
    lattice and interpolated between lattice points, which is exact for the
    affine and corner transforms (linear along each axis) and for ping
    navigation across a row, and within millimetres along a few rows of a
    turning track. Each output cell averages the strip pixels that land in it,
    so a strip larger than the output is box-filtered rather than aliased.
    Cells no pixel reached are transparent, apart from pinholes surrounded by
    covered cells, which are filled from their neighbours.
    """
    import numpy as np
    from PIL import Image

    grey = np.asarray(grey, dtype=np.uint8)
    if grey.ndim != 2 or grey.size == 0:
        return None
    height, width = grey.shape

    # --- lattice of located pixel centres ---------------------------------
    lx = _lattice(width, RESAMPLE_LATTICE_ACROSS_PX)
    ly = _lattice(height, RESAMPLE_LATTICE_ALONG_PX)
    lat_grid = np.full((len(ly), len(lx)), np.nan)
    lon_grid = np.full((len(ly), len(lx)), np.nan)
    for j, y in enumerate(ly):
        for i, x in enumerate(lx):
            lat, lon = reference.locate(float(x), float(y))
            if lat is not None and lon is not None:
                lat_grid[j, i] = lat
                lon_grid[j, i] = lon
    finite = np.isfinite(lat_grid) & np.isfinite(lon_grid)
    if not finite.any():
        return None

    south, north = float(lat_grid[finite].min()), float(lat_grid[finite].max())
    west, east = float(lon_grid[finite].min()), float(lon_grid[finite].max())
    mid_lat = (south + north) / 2.0
    meridional, prime = radii_of_curvature(mid_lat)
    north_m = math.radians(north - south) * meridional
    east_m = math.radians(east - west) * prime * math.cos(math.radians(mid_lat))

    # --- how big a strip pixel is on the ground ---------------------------
    def spacing_m(dlat: Any, dlon: Any, pixels: float) -> float:
        metres = np.hypot(np.radians(dlat) * meridional,
                          np.radians(dlon) * prime * math.cos(math.radians(mid_lat)))
        metres = metres[np.isfinite(metres) & (metres > 0)]
        return float(np.median(metres)) / pixels if metres.size else float("nan")

    across_m = spacing_m(np.diff(lat_grid, axis=1), np.diff(lon_grid, axis=1),
                         float(lx[1] - lx[0]) if lx[1] > lx[0] else 1.0)
    along_m = spacing_m(np.diff(lat_grid, axis=0), np.diff(lon_grid, axis=0),
                        float(ly[1] - ly[0]) if ly[1] > ly[0] else 1.0)
    candidates = [v for v in (across_m, along_m) if math.isfinite(v) and v > 0]
    pixel_m = min(candidates) if candidates else max(east_m, north_m, 1e-9) / max(width, height)

    # --- the output grid ---------------------------------------------------
    extent_m = max(east_m, north_m, pixel_m)
    cell_m = max(pixel_m, extent_m / max_edge)
    out_w = int(min(max_edge, max(1, math.ceil(east_m / cell_m))))
    out_h = int(min(max_edge, max(1, math.ceil(north_m / cell_m))))

    # Strip pixels are box-averaged in blocks of `stride` first when the output
    # is so much coarser that every output cell would receive several blocks
    # anyway; each block then lands as one sample at its centre.
    stride = max(1, int(math.floor(0.5 * cell_m / pixel_m)))
    usable_h, usable_w = (height // stride) * stride, (width // stride) * stride
    if usable_h == 0 or usable_w == 0:
        stride, usable_h, usable_w = 1, height, width
    blocks = grey[:usable_h, :usable_w].astype(np.float32)
    if stride > 1:
        blocks = blocks.reshape(usable_h // stride, stride, usable_w // stride, stride).mean(axis=(1, 3))
    xs = np.arange(blocks.shape[1], dtype=float) * stride + stride / 2.0
    ys = np.arange(blocks.shape[0], dtype=float) * stride + stride / 2.0

    ix, wx = _axis_weights(lx, xs)
    merc_north = float(_mercator_y(north))
    merc_span = merc_north - float(_mercator_y(south))
    lon_span = east - west

    sums = np.zeros(out_w * out_h, dtype=np.float64)
    counts = np.zeros(out_w * out_h, dtype=np.float64)
    chunk = 256
    for start in range(0, len(ys), chunk):
        rows_y = ys[start:start + chunk]
        iy, wy = _axis_weights(ly, rows_y)
        wy = wy[:, None]

        def interpolate(grid: Any) -> Any:
            upper, lower = grid[iy], grid[iy + 1]
            top = upper[:, ix] * (1.0 - wx) + upper[:, ix + 1] * wx
            bottom = lower[:, ix] * (1.0 - wx) + lower[:, ix + 1] * wx
            return top * (1.0 - wy) + bottom * wy

        lat = interpolate(lat_grid)
        lon = interpolate(lon_grid)
        values = blocks[start:start + chunk]
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            continue
        col = ((lon[ok] - west) / lon_span * out_w) if lon_span > 0 else np.zeros(ok.sum())
        row = ((merc_north - _mercator_y(lat[ok])) / merc_span * out_h) if merc_span > 0 \
            else np.zeros(ok.sum())
        col = np.clip(np.floor(col), 0, out_w - 1).astype(np.int64)
        row = np.clip(np.floor(row), 0, out_h - 1).astype(np.int64)
        index = row * out_w + col
        sums += np.bincount(index, weights=values[ok], minlength=out_w * out_h)
        counts += np.bincount(index, minlength=out_w * out_h)

    sums = sums.reshape(out_h, out_w)
    counts = counts.reshape(out_h, out_w)
    covered = counts > 0
    image = np.zeros((out_h, out_w), dtype=np.float64)
    image[covered] = sums[covered] / counts[covered]

    # --- pinholes ------------------------------------------------------------
    # Two passes of a 3x3 neighbourhood mean over covered cells, applied only
    # where most neighbours are covered, so the footprint's edge is not grown.
    for _ in range(2):
        holes = ~covered
        if not holes.any():
            break
        padded_value = np.pad(np.where(covered, image, 0.0), 1)
        padded_cover = np.pad(covered.astype(np.float64), 1)
        near_value = np.zeros_like(image)
        near_cover = np.zeros_like(image)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                near_value += padded_value[dy:dy + out_h, dx:dx + out_w]
                near_cover += padded_cover[dy:dy + out_h, dx:dx + out_w]
        fill = holes & (near_cover >= RESAMPLE_FILL_MIN_NEIGHBOURS)
        if not fill.any():
            break
        image[fill] = near_value[fill] / near_cover[fill]
        covered = covered | fill

    luminance = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    alpha = np.where(covered, 255, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.dstack([luminance, alpha]), mode="LA").save(buffer, "PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    # The strip's own outline: along the top row, down the last column, back
    # along the bottom row and up the first column, dropping unlocated points.
    down = np.unique(np.linspace(0, len(ly) - 1, min(len(ly), 120)).round().astype(int))
    ring_index = ([(0, i) for i in range(len(lx))]
                  + [(j, len(lx) - 1) for j in down]
                  + [(len(ly) - 1, i) for i in range(len(lx) - 1, -1, -1)]
                  + [(j, 0) for j in down[::-1]])
    footprint = [[float(lat_grid[j, i]), float(lon_grid[j, i])] for j, i in ring_index
                 if finite[j, i]]

    return {
        "data_uri": f"data:image/png;base64,{encoded}",
        "bounds": [[south, west], [north, east]],
        "footprint": footprint,
        "size": (out_w, out_h),
        "stride": stride,
    }


# --- popups ----------------------------------------------------------------


def _row(label: str, value: Any) -> str:
    return (f"<tr><th>{label}</th><td>{value}</td></tr>")


def _esc(value: Any) -> str:
    """Text from the export, made safe to place in the page's HTML."""
    return html_lib.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _confidence_pct(detection: dict[str, Any]) -> float | None:
    """The export's 0-100 confidence, or None for an export that predates it."""
    return _number(detection.get("confidence_pct"))


def _dimensions(detection: dict[str, Any]) -> tuple[str, list[str]] | None:
    """(size text, basis notes) from the export's dimensions block, or None.

    Metres only where the engine measured them. A strip with no known
    resolution still has a box in pixels, and that is what is shown, labelled
    as pixels, rather than any conversion made up here.
    """
    dims = detection.get("dimensions")
    if not isinstance(dims, dict):
        return None
    length, width, height = (_number(dims.get(k)) for k in ("length_m", "width_m", "height_m"))
    notes = [str(dims[k]) for k in ("basis", "height_basis") if dims.get(k)]
    if length is not None and width is not None:
        text = f"{length:.1f} &times; {width:.1f}"
        text += f" &times; {height:.1f} m" if height is not None else " m"
        label = "L &times; W &times; H" if height is not None else "L &times; W"
        return f"{text} <span class='de-sub'>{label}</span>", notes
    parts = []
    length_px, width_px = _number(dims.get("length_px")), _number(dims.get("width_px"))
    if length_px is not None and width_px is not None:
        parts.append(f"{length_px:.0f} &times; {width_px:.0f} px "
                     f"<span class='de-sub'>L &times; W, not measured in metres</span>")
    if height is not None:
        parts.append(f"height {height:.1f} m")
    if not parts:
        return ("not measured", notes) if notes else None
    return " ".join(parts), notes


def _reason_list(items: Iterable[Any]) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    return "<ul class='de-reasons'>" + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>"


def _verification_rows(detection: dict[str, Any]) -> list[str]:
    """What the verification stage found, in words, for one detection."""
    verification = detection.get("verification")
    rows: list[str] = []
    if not isinstance(verification, dict):
        basis = detection.get("confidence_pct_basis")
        if basis:
            rows.append(_row("Verification", f"<span class='de-sub'>{_esc(basis)}</span>"))
        return rows

    reasons = verification.get("reasons") or []
    hard = [HARD_REASON_LABEL.get(str(r), str(r).replace("_", " "))
            for r in verification.get("hard_reasons") or []]
    if verification.get("status") == "not_checked":
        rows.append(_row("Verification", "not checked <span class='de-sub'>"
                         + _esc(verification.get("reason") or "no strip image") + "</span>"))
    elif detection.get("suppressed"):
        rows.append(_row("Filtered because",
                         (f"<b>{_esc(', '.join(hard))}</b>" if hard else "")
                         + _reason_list(reasons)))
    elif reasons:
        rows.append(_row("Verification", "checked, not filtered"
                         + _reason_list(reasons)))
    else:
        rows.append(_row("Verification", "checked <span class='de-sub'>no artefact "
                                         "evidence found</span>"))
    return rows


def _detection_popup(detection: dict[str, Any], frame: Frame) -> str:
    provenance = detection["provenance"]
    style = theme.tier_style(detection["severity_tier"])
    suppressed = bool(detection.get("suppressed"))
    pct = _confidence_pct(detection)
    verification = detection.get("verification") if isinstance(
        detection.get("verification"), dict) else {}
    detector = _number(verification.get("detector_confidence"))
    detector = detector if detector is not None else float(detection["confidence"])

    if pct is not None:
        basis = detection.get("confidence_pct_basis")
        confidence = (f"<b>{pct:.1f}%</b> <span class='de-sub'>"
                      + (_esc(basis) if basis else "verified: detector confidence fused "
                         "with the image evidence")
                      + "</span>")
    else:
        confidence = f"{detection['confidence']:.2f}"

    rows = [
        _row("Class", f"<b>{_esc(detection['object_class'])}</b>"),
        _row("Confidence", confidence),
    ]
    if pct is not None:
        rows.append(_row("Detector", f"{detector:.2f} <span class='de-sub'>the model's "
                                     f"own score, 0 to 1</span>"))
    size = _dimensions(detection)
    if size is not None:
        text, notes = size
        rows.append(_row("Size", text + "".join(f"<span class='de-sub de-basis'>"
                                                f"{_esc(n)}</span>" for n in notes)))
    rows.extend([
        _row("Severity", f"{detection['severity']:.4f} "
                         f"<span class='de-sub'>= {detection['class_weight']} weight "
                         f"&times; {detection['confidence']:.2f} "
                         f"{'detector ' if pct is not None else ''}confidence</span>"),
        _row("Tier", f"<span class='de-chip' style='background:{style['surface']};"
                     f"border-color:{style['border']};color:{style['text']}'>"
                     f"{style['label']}</span>"),
        _row("Survey position", f"x {detection['global_x']:.1f}, y {detection['global_y']:.1f} px"),
    ])
    rows.extend(_verification_rows(detection))
    # A geographic position is shown only when one genuinely exists. In a
    # relative survey this row is absent rather than blank, so there is nothing
    # on screen that could be mistaken for a fix.
    if detection.get("latitude") is not None and detection.get("longitude") is not None:
        rows.append(_row("Latitude", f"{detection['latitude']:.6f}"))
        rows.append(_row("Longitude", f"{detection['longitude']:.6f}"))

    rows.append(_row("Source tile", f"<span class='de-mono'>"
                                    f"{provenance['representative_tile']}</span>"))
    merged = provenance["merged_count"]
    rows.append(_row("Views merged", f"{merged}" if merged == 1 else
                     f"{merged} <span class='de-sub'>across "
                     f"{len(provenance['source_tiles'])} tile(s)</span>"))
    if suppressed:
        rows.append(_row("Action", "none: filtered as a likely false positive "
                                   "<span class='de-sub'>kept in export.json with its "
                                   "reasons; excluded from hotspots and the heatmap</span>"))
    else:
        rows.append(_row("Action", _esc(detection["recommended_action"])))

    head_colour = FILTERED_COLOR if suppressed else style["marker"]
    flag = "<span class='de-filtered-chip'>filtered</span>" if suppressed else ""
    return (f"<div class='de-popup'><div class='de-popup-head' "
            f"style='border-left-color:{head_colour}'>"
            f"<span class='de-mono'>{_esc(detection['id'])}</span>{flag}</div>"
            f"<table>{''.join(rows)}</table></div>")


def _hotspot_popup(hotspot: dict[str, Any], frame: Frame) -> str:
    style = theme.tier_style(hotspot["severity_tier"])
    centroid = hotspot["centroid"]

    rows = [
        _row("Priority", f"<b>Rank {hotspot['priority_rank']}</b>"),
        _row("Dominant hazard", f"<b>{hotspot['dominant_class']}</b>"),
        _row("Detections", hotspot["detection_count"]),
        _row("Total severity", f"{hotspot['total_severity']:.4f} "
                               f"<span class='de-sub'>summed over the cell, "
                               f"can exceed 1</span>"),
        _row("Worst single", f"{hotspot['max_severity']:.4f} "
                             f"<span class='de-sub'>0 to 1, sets the tier</span>"),
        _row(theme.TEXT["risk_label"],
             f"{hotspot['risk_score']:.4f} "
             f"<span class='de-sub'>index, not a percentage</span>"),
        _row("Tier", f"<span class='de-chip' style='background:{style['surface']};"
                     f"border-color:{style['border']};color:{style['text']}'>"
                     f"{style['label']}</span>"),
        _row("Centroid", f"x {centroid['global_x']:.1f}, y {centroid['global_y']:.1f} px"),
    ]
    if centroid.get("latitude") is not None:
        rows.append(_row("Latitude", f"{centroid['latitude']:.6f}"))
        rows.append(_row("Longitude", f"{centroid['longitude']:.6f}"))
    rows.append(_row("Action", f"<b>{hotspot['recommended_action']}</b>"))

    return (f"<div class='de-popup'><div class='de-popup-head' "
            f"style='border-left-color:{style['marker']}'>"
            f"<b>{hotspot['hotspot_id']}</b></div>"
            f"<table>{''.join(rows)}</table>"
            f"<p class='de-why'>{hotspot['rationale']}</p></div>")


# --- dashboard chrome ------------------------------------------------------

_CSS = """
html, body { margin: 0; padding: 0; height: 100%; background: ${background}; }
body { font-family: ${sans}; color: ${text}; }

/* Folium sizes its map div inline. The dashboard occupies the top and left
   edges, so the map is repositioned to the space that is left. */
.folium-map {
  position: fixed !important;
  top: var(--de-top) !important; left: var(--de-left) !important;
  right: 0 !important; bottom: 0 !important;
  width: auto !important; height: auto !important;
}
:root { --de-top: 56px; --de-left: 330px; }

/* Leaflet paints #ddd behind an empty map, which reads as a broken page. A
   portrait survey strip in a landscape window always leaves margins, so they
   have to look like canvas rather than like failure. */
.leaflet-container { background: ${surround} !important; }

.de-header {
  position: fixed; top: 0; left: 0; right: 0; height: 56px; z-index: 1200;
  display: flex; align-items: center; gap: 16px; padding: 0 18px;
  background: ${navy_deep}; color: ${text_inverse};
  border-bottom: 1px solid ${navy};
}
.de-wordmark { font-size: 17px; font-weight: 600; letter-spacing: 0.02em; }
.de-wordmark span { color: ${teal_tint}; font-weight: 400; }
.de-title {
  font-family: ${mono}; font-size: 12.5px; color: ${navy_tint};
  padding-left: 16px; border-left: 1px solid rgba(255,255,255,0.22);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.de-header-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.de-badge {
  font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 4px 9px; border-radius: 999px; white-space: nowrap;
  border: 1px solid rgba(255,255,255,0.28); color: ${navy_tint};
}
.de-badge-mode { background: ${teal_deep}; border-color: ${teal}; color: #fff; }
.de-badge-ok { background: rgba(255,255,255,0.08); }

/* With the banner up, everything below it moves down by its height. Without
   this the map slid under the banner and lost its top 36 pixels. */
body.de-has-demo { --de-top: 92px; }

.de-demo {
  position: fixed; top: 56px; left: 0; right: 0; z-index: 1200;
  background: ${alert_surface}; border-bottom: 1px solid ${alert_border};
  color: ${alert_text}; padding: 7px 18px; font-size: 12px;
  display: flex; align-items: baseline; gap: 12px;
}
.de-demo b { letter-spacing: 0.08em; font-size: 11.5px; }

.de-panel {
  position: fixed; top: var(--de-top); left: 0; bottom: 0; width: 330px; z-index: 1150;
  background: ${surface}; border-right: 1px solid ${border};
  overflow-y: auto; font-size: 13px;
}
.de-section { padding: 14px 16px; border-bottom: 1px solid ${border}; }
.de-section h2 {
  margin: 0 0 10px; font-size: 11px; letter-spacing: 0.08em;
  text-transform: uppercase; color: ${text_faint}; font-weight: 600;
}

.de-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: ${border}; }
.de-stat { background: ${surface}; padding: 9px 10px; }
.de-stat .v { font-size: 19px; font-weight: 600; font-family: ${mono}; line-height: 1.2; }
.de-stat .k { font-size: 10.5px; color: ${text_muted}; letter-spacing: 0.04em;
              text-transform: uppercase; margin-top: 2px; }
.de-stat.alert .v { color: ${alert_text}; }

.de-note {
  font-size: 11.5px; line-height: 1.5; color: ${text_muted};
  background: ${surface_muted}; border-left: 2px solid ${border_strong};
  padding: 8px 10px; margin: 10px 0 0;
}
.de-note.mode { background: ${teal_tint}; border-left-color: ${teal}; color: ${teal_deep}; }

.de-legend-row { display: flex; align-items: center; gap: 9px; padding: 4px 0; }
.de-dot { width: 12px; height: 12px; border-radius: 50%; flex: none;
          border: 2px solid rgba(0,0,0,0.25); }
.de-legend-row .lbl { font-weight: 500; }
.de-legend-row .rng { margin-left: auto; font-family: ${mono}; font-size: 11px;
                      color: ${text_faint}; }

.de-prio { padding: 0; }
.de-prio-item {
  display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid ${border};
  background: ${surface}; padding: 10px 16px; cursor: pointer; font: inherit;
  border-left: 3px solid transparent;
}
.de-prio-item:hover { background: ${surface_muted}; }
.de-prio-item.active { background: ${navy_tint}; border-left-color: ${navy}; }
.de-prio-top { display: flex; align-items: center; gap: 8px; }
.de-rank {
  font-family: ${mono}; font-size: 11px; font-weight: 600; padding: 1px 6px;
  border-radius: 3px; border: 1px solid; flex: none;
}
.de-prio-id { font-family: ${mono}; font-weight: 600; font-size: 12.5px; }
.de-first {
  margin-left: auto; font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase;
  color: ${teal_deep}; background: ${teal_tint}; border: 1px solid ${teal};
  border-radius: 999px; padding: 1px 7px;
}
.de-prio-class { margin-top: 4px; font-size: 12.5px; }
.de-prio-action { margin-top: 2px; font-size: 12px; color: ${text_muted}; }
.de-bar { height: 4px; background: ${surface_muted}; border-radius: 2px; margin-top: 7px; }
.de-bar i { display: block; height: 100%; border-radius: 2px; }
.de-prio-meta { margin-top: 5px; font-family: ${mono}; font-size: 10.5px; color: ${text_faint}; }

.de-empty { padding: 16px; font-size: 12.5px; color: ${text_muted}; line-height: 1.55; }
.de-disclaimer { font-size: 11px; line-height: 1.5; color: ${text_faint}; }
.de-disclaimer b { color: ${text_muted}; display: block; margin-bottom: 4px;
                   font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; }

/* Leaflet's own controls, brought into the design system. The layer control
   is rendered expanded, which hides its toggle button and with it the only
   image Leaflet's stylesheet would have asked the network for. */
.leaflet-control-layers-toggle { display: none !important; }
.leaflet-control-layers {
  font-family: ${sans}; font-size: 12.5px; color: ${text};
  border: 1px solid ${border} !important; border-radius: 5px !important;
  box-shadow: 0 1px 2px rgba(21,40,69,0.10) !important; padding: 9px 12px !important;
  background: ${surface} !important;
}
.leaflet-control-layers label { margin: 0; display: block; padding: 2px 0; cursor: pointer; }
.leaflet-control-layers label span { display: inline-flex; align-items: center; gap: 6px; }
.leaflet-control-layers-separator { border-top: 1px solid ${border}; margin: 6px 0; }
.leaflet-bar a, .leaflet-bar a:hover {
  color: ${navy}; border-bottom-color: ${border};
}
.leaflet-control-attribution {
  font-size: 10px; background: rgba(255,255,255,0.82) !important; color: ${text_faint};
}
.leaflet-control-attribution a { color: ${teal_deep}; }

/* Popups */
.de-popup { font-family: ${sans}; font-size: 12.5px; min-width: 250px; }
.de-popup-head {
  border-left: 3px solid ${border_strong}; padding: 1px 0 1px 8px; margin-bottom: 8px;
  font-size: 13px;
}
.de-popup table { border-collapse: collapse; width: 100%; }
.de-popup th {
  text-align: left; font-weight: 500; color: ${text_muted}; padding: 2px 10px 2px 0;
  white-space: nowrap; vertical-align: top; font-size: 11.5px;
}
.de-popup td { padding: 2px 0; vertical-align: top; }
.de-mono { font-family: ${mono}; font-size: 11.5px; }
.de-sub { color: ${text_faint}; font-size: 11px; }
.de-chip { border: 1px solid; border-radius: 3px; padding: 0 6px; font-size: 11px; }
.de-why {
  margin: 9px 0 0; padding-top: 8px; border-top: 1px solid ${border};
  font-size: 11.5px; line-height: 1.5; color: ${text_muted};
}
.leaflet-popup-content { margin: 11px 13px; }
.de-basis { display: block; margin-top: 3px; line-height: 1.4; max-width: 250px;
            white-space: normal; }
.de-reasons { margin: 3px 0 0; padding-left: 16px; font-size: 11.5px; line-height: 1.45;
              color: ${text_muted}; max-width: 250px; }
.de-reasons li { margin: 2px 0; }
.de-filtered-chip {
  margin-left: 8px; font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
  color: ${text_muted}; background: ${surface_muted}; border: 1px dashed ${border_strong};
  border-radius: 3px; padding: 0 5px;
}
.leaflet-tooltip.de-filtered-tip {
  background: ${surface_muted}; color: ${text_muted}; border: 1px dashed ${border_strong};
  box-shadow: none; font: 10.5px ${sans}; letter-spacing: 0.04em; padding: 1px 5px;
}
.leaflet-tooltip.de-filtered-tip::before { display: none; }
.de-stat.wide { grid-column: span 2; }
.de-badge-filtered { background: rgba(255,255,255,0.06); border-style: dashed; }

@media (max-width: 880px) {
  :root { --de-left: 0px; }
  .de-panel { top: auto; bottom: 0; width: 100%; height: 42%; border-right: 0;
              border-top: 1px solid ${border}; }
  .folium-map { bottom: 42% !important; }
}
"""


def _css() -> str:
    from string import Template

    tokens = dict(theme.COLOR)
    tokens["sans"] = theme.FONT_SANS
    tokens["mono"] = theme.FONT_MONO
    tokens["surround"] = theme.MAP_SURROUND
    return f"<style>{Template(_CSS).substitute(tokens)}</style>"


def _stat(value: Any, label: str, alert: bool = False) -> str:
    return (f"<div class='de-stat{' alert' if alert else ''}'>"
            f"<div class='v'>{value}</div><div class='k'>{label}</div></div>")


def _dashboard(export: dict[str, Any], title: str, demo: bool,
               multi_strip: bool, resampled: Iterable[str] = ()) -> str:
    summary = export["survey_summary"]
    resampled = list(resampled)
    # Present only in exports written since verification existed. An older
    # export has no such count, and the map then says nothing about filtering
    # rather than claiming nothing was filtered.
    verified = "suppressed_detections" in summary or any(
        "suppressed" in d for d in export["detections"])
    filtered = int(summary.get("suppressed_detections") or sum(
        1 for d in export["detections"] if d.get("suppressed")))
    hotspots = export["hotspots"]
    georeferenced = summary["georeferenced"]
    tiers = summary["detections_by_tier"]
    critical = tiers.get("critical", 0)

    mode_label = theme.TEXT["geo_mode"] if georeferenced else theme.TEXT["relative_mode"]
    mode_note = theme.TEXT["geo_note"] if georeferenced else theme.TEXT["relative_note"]

    header = (
        f"<div class='de-header'>"
        f"<div class='de-wordmark'>{theme.TEXT['app']}"
        f"<span> {theme.TEXT['subtitle']}</span></div>"
        f"<div class='de-title'>{_esc(title)}</div>"
        f"<div class='de-header-right'>"
        f"<span class='de-badge de-badge-mode'>{mode_label}</span>"
        + (f"<span class='de-badge de-badge-filtered'>{filtered} filtered</span>"
           if verified else "")
        + f"<span class='de-badge de-badge-ok'>Processed "
        f"{summary['tiles_processed']} tiles</span>"
        f"</div></div>")

    banner = ""
    if demo:
        # The export's own warning when it carries one: a survey imported with
        # simulated navigation has real sonar but invented positions, which the
        # generic "generated by the demo script" note would misstate.
        note = (export.get("metadata") or {}).get("demo_warning") or theme.TEXT["demo_note"]
        banner = (f"<div class='de-demo'><b>{theme.TEXT['demo_banner']}</b>"
                  f"<span>{html_lib.escape(str(note))}</span></div>")

    stats = (
        "<div class='de-section'><h2>Survey</h2><div class='de-stats'>"
        + _stat(summary["total_deduplicated_detections"], "Detections")
        + _stat(summary["total_hotspots"], "Hotspots")
        + _stat(critical, "Critical", alert=critical > 0)
        + _stat(f"{summary['total_severity']:.2f}", "Total severity")
        + (f"<div class='de-stat wide'><div class='v'>{filtered}</div>"
           f"<div class='k'>Filtered false positives</div></div>" if verified else "")
        + "</div>"
        + (f"<p class='de-note'>{filtered} detection(s) filtered as likely false "
           f"positives by verification. They stay in export.json with their reasons, "
           f"are left out of hotspots and the heatmap, and are drawn grey and hollow "
           f"on the &ldquo;{FILTERED_LAYER}&rdquo; layer, which is off when the map "
           f"opens.</p>" if verified and filtered else "")
        + (f"<p class='de-note'>{summary['duplicates_removed']} duplicate view(s) "
           f"merged from {summary['total_raw_detections']} raw boxes.</p>"
           if summary["duplicates_removed"] else "")
        + f"<p class='de-note mode'>{mode_note}</p>"
        + (f"<p class='de-note'>{theme.TEXT['multi_strip_note']}</p>"
           if multi_strip and not georeferenced else "")
        + "</div>")

    legend_rows = []
    for tier, floor in cfg.SEVERITY_TIERS:
        style = theme.tier_style(tier)
        # The NEAREST boundary above this tier, not the first one listed.
        # SEVERITY_TIERS is ordered high to low, so next() returned critical's
        # 0.75 for every tier and the legend read "low: below 0.75".
        above = [f for _t, f in cfg.SEVERITY_TIERS if f > floor]
        upper = min(above) if above else None
        span = f"{floor:.2f} and up" if upper is None else f"{floor:.2f} to {upper:.2f}"
        span = f"below {upper:.2f}" if floor == 0.0 and upper else span
        legend_rows.append(
            f"<div class='de-legend-row'>"
            f"<span class='de-dot' style='background:{style['fill']};"
            f"border-color:{style['marker']}'></span>"
            f"<span class='lbl'>{style['label']}</span>"
            f"<span class='rng'>{span}</span></div>")

    legend = ("<div class='de-section'><h2>" + theme.TEXT["legend_heading"] + "</h2>"
              + "".join(legend_rows)
              + f"<p class='de-note'>{theme.TEXT['heat_note']}</p></div>")

    if hotspots:
        items = []
        worst = max(h["total_severity"] for h in hotspots) or 1.0
        for hotspot in hotspots:
            style = theme.tier_style(hotspot["severity_tier"])
            first = (f"<span class='de-first'>{theme.TEXT['start_here']}</span>"
                     if hotspot["priority_rank"] == 1 else "")
            items.append(
                f"<button class='de-prio-item' data-hotspot='{hotspot['hotspot_id']}'>"
                f"<div class='de-prio-top'>"
                f"<span class='de-rank' style='background:{style['surface']};"
                f"border-color:{style['border']};color:{style['text']}'>"
                f"{hotspot['priority_rank']}</span>"
                f"<span class='de-prio-id'>{hotspot['hotspot_id']}</span>{first}</div>"
                f"<div class='de-prio-class'><b>{hotspot['dominant_class']}</b>"
                f" &middot; {hotspot['detection_count']} detection"
                f"{'s' if hotspot['detection_count'] != 1 else ''}</div>"
                f"<div class='de-prio-action'>{hotspot['recommended_action']}</div>"
                f"<div class='de-bar'><i style='width:"
                f"{max(3, round(100 * hotspot['total_severity'] / worst))}%;"
                f"background:{style['fill']}'></i></div>"
                f"<div class='de-prio-meta'>"
                f"{theme.TEXT['total_severity_label']} {hotspot['total_severity']:.3f}"
                f" &middot; {theme.TEXT['max_severity_label']} "
                f"{hotspot['max_severity']:.3f}"
                f" &middot; {theme.TEXT['risk_label'].lower()} "
                f"{hotspot['risk_score']:.3f}</div>"
                f"</button>")
        priority = ("<div class='de-section' style='padding-bottom:0'><h2>"
                    + theme.TEXT["priority_heading"] + "</h2></div>"
                    + "<div class='de-prio'>" + "".join(items) + "</div>")
    elif export["detections"] and filtered == len(export["detections"]):
        priority = (f"<div class='de-empty'>Every detection in this survey was filtered "
                    f"as a likely false positive, so there is nothing to rank. Switch on "
                    f"&ldquo;{FILTERED_LAYER}&rdquo; to see them and why.</div>")
    else:
        priority = f"<div class='de-empty'>{theme.TEXT['no_detections']}</div>"

    imagery_note = ""
    if resampled:
        imagery_note = (f"<p class='de-disclaimer'><b>Imagery</b>"
                        f"{RESAMPLED_NOTE.format(strips=_esc(', '.join(resampled)))}</p>")
    disclaimer = (f"<div class='de-section de-footer'>{imagery_note}"
                  f"<p class='de-disclaimer'>"
                  f"<b>{theme.TEXT['disclaimer_heading']}</b>{cfg.DISCLAIMER}</p></div>")

    return (header + banner
            + "<div class='de-panel'>"
            + stats + legend + priority + disclaimer + "</div>"
            # One class, read by the stylesheet, moves the map and the panel
            # down together when the banner is present.
            + ("<script>document.body.classList.add('de-has-demo');</script>"
               if demo else ""))


# --- the map ---------------------------------------------------------------

# folium builds every popup's content with jQuery, as $(`<div>...</div>`)[0].
# hazard_assets removes jQuery, because nothing else on the page needs it, and
# with it gone the first popup threw "$ is not defined" and stopped the whole
# map script: no layer control, no fitted view, no hotspot focus. This is the
# one thing that call needs -- HTML text to a DOM node -- without the library.
_POPUP_SHIM = ("<script>function deHtml(markup) {"
               "var holder = document.createElement('template');"
               "holder.innerHTML = String(markup).trim();"
               "return holder.content.childNodes; }</script>")
_JQUERY_POPUP_CALL = "$(`"
_SHIM_POPUP_CALL = "deHtml(`"

_JS = """
// Deferred until the document is parsed. folium writes this page's own script
// ahead of the map's, so run immediately it read the map variable before the
// map existed and every call below failed.
(function (start) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})(function () {
  var map = ${map};
  var targets = ${targets};

  function focus(id, button) {
    var target = targets[id];
    if (!target) { return; }
    document.querySelectorAll('.de-prio-item.active')
      .forEach(function (el) { el.classList.remove('active'); });
    if (button) { button.classList.add('active'); }
    var marker = window[target.marker];
    // Opened only after the view has settled. Opening mid-animation let the
    // popup's auto-pan run against the zoom, and it came to rest clipped under
    // the zoom control. The padding keeps it clear of that control.
    var opened = false;
    function open() {
      if (opened || !marker || !marker.openPopup) { return; }
      opened = true;
      var popup = marker.getPopup && marker.getPopup();
      if (popup && popup.options) {
        popup.options.autoPanPaddingTopLeft = L.point(56, 56);
        popup.options.autoPanPaddingBottomRight = L.point(24, 24);
      }
      marker.openPopup();
    }
    map.once('moveend', open);
    map.setView(target.center, target.zoom, { animate: true });
    // setView to the view already showing fires no moveend.
    window.setTimeout(open, 900);
  }

  document.querySelectorAll('.de-prio-item').forEach(function (button) {
    button.addEventListener('click', function () {
      focus(button.getAttribute('data-hotspot'), button);
    });
  });

  // Leaflet measures the container once, on creation. The dashboard resizes it
  // afterwards, so without this the map renders into the old rectangle and the
  // tiles sit offset until the first manual pan.
  window.setTimeout(function () { map.invalidateSize(); }, 60);
  window.addEventListener('resize', function () { map.invalidateSize(); });

  // The page is also embedded in the DeepEcho dashboard, in a frame served
  // from a different origin. These two are the whole interface it offers, and
  // both are read-only: the host can ask the map to look at a hotspot, and the
  // map says which one it is showing. Nothing here accepts code, a URL, or
  // anything that is not one of this survey's own hotspot identifiers, and an
  // id that is not in `targets` is ignored rather than acted on.
  window.addEventListener('message', function (event) {
    var data = event && event.data;
    if (!data || data.type !== 'deepecho:focus') { return; }
    var id = String(data.hotspot_id || '');
    if (!Object.prototype.hasOwnProperty.call(targets, id)) { return; }
    focus(id, document.querySelector('.de-prio-item[data-hotspot="' + id + '"]'));
  });

  // A hotspot named in the URL fragment is opened on load, so a link can point
  // at one without the host having to send a message at all.
  var fragment = (window.location.hash || '').replace(/^#/, '');
  if (Object.prototype.hasOwnProperty.call(targets, fragment)) {
    window.setTimeout(function () { focus(fragment, null); }, 120);
  }
});
"""


def _detection_marker(detection: dict[str, Any], position: list[float]):
    import folium

    style = theme.tier_style(detection["severity_tier"])
    merged = int(detection["provenance"]["merged_count"])
    radius = min(theme.MERGED_RADIUS_CAP_PX,
                 theme.DETECTION_RADIUS_PX + (merged - 1) * theme.MERGED_RADIUS_BONUS_PX)
    pct = _confidence_pct(detection)
    confidence = f" &middot; {pct:.1f}%" if pct is not None else ""
    popup = folium.Popup(_detection_popup(detection, None), max_width=340)

    if detection.get("suppressed"):
        # Grey, hollow and dashed: present, inspectable, and plainly not a
        # hazard the survey is asking anyone to act on. The label is permanent
        # so the layer reads correctly without hovering over every marker.
        return folium.CircleMarker(
            location=position,
            radius=radius,
            color=FILTERED_COLOR,
            weight=theme.DETECTION_WEIGHT,
            dash_array="3 3",
            fill=False,
            popup=popup,
            tooltip=folium.Tooltip(
                f"filtered &middot; {_esc(detection['object_class'])}{confidence}",
                sticky=False, permanent=True, direction="right", offset=[8, 0],
                class_name="de-filtered-tip"),
        )

    return folium.CircleMarker(
        location=position,
        radius=radius,
        color=style["marker"],
        weight=theme.DETECTION_WEIGHT,
        fill=True,
        fill_color=style["fill"],
        fill_opacity=theme.DETECTION_FILL_OPACITY,
        popup=popup,
        tooltip=(f"{_esc(detection['object_class'])}{confidence} &middot; severity "
                 f"{detection['severity']:.3f}"),
    )


def render_map(export: dict[str, Any], out_dir: Any, tiles_dir: Any = None,
               manifest: Any = None, *, title: str | None = None,
               demo: bool = False, basemap: bool = False, offline: bool = True,
               filename: str = "map.html") -> Path:
    """Write a standalone map.html for one survey.

    export      the dictionary build_hazard_map returned, or a loaded export.json
    out_dir     where map.html goes
    tiles_dir   the tiles, used to rebuild the sonar base layer
    manifest    the manifest; found beside the tiles if omitted
    title       the survey name shown in the header
    demo        stamps the map as synthetic. See demo_survey.py.
    basemap     in a georeferenced survey, include an OpenStreetMap layer.
                Off by default and off at open even when on, because it is the
                only thing on the map that needs a network.
    offline     inline Leaflet into the page and strip the libraries folium
                links but this map never uses. On by default. See
                hazard_assets for what is kept and what is dropped.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError as exc:
        raise MapError(
            "folium is not installed, so map.html cannot be built. Run "
            "`pip install folium`, or pass --no-map to run_survey.py if you only "
            "need export.json and actions.csv. The engine itself does not need it."
        ) from exc

    from survey_hazard_map.survey_preparation import load_manifest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    survey_meta: dict[str, Any] = {}
    manifest_dir: Path | None = None
    if manifest is not None:
        rows, survey_meta = load_manifest(manifest)
        manifest_dir = Path(manifest) if Path(manifest).is_dir() else Path(manifest).parent
    elif tiles_dir is not None:
        for candidate in (Path(tiles_dir).parent / cfg.MANIFEST_JSON,
                          Path(tiles_dir).parent / cfg.MANIFEST_CSV):
            if candidate.is_file():
                rows, survey_meta = load_manifest(candidate)
                manifest_dir = candidate.parent
                break

    detections = export["detections"]
    hotspots = export["hotspots"]
    georeferenced = bool(export["survey_summary"]["georeferenced"])
    grid = int(export["configuration"]["hotspots"]["GRID"])

    by_strip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strip.setdefault(str(row["strip"]), []).append(row)
    # A survey run without a manifest still maps: the detections name their own
    # strip, there is simply no imagery to draw under them.
    for detection in detections:
        by_strip.setdefault(str(detection["provenance"]["strip"] or ""), [])
    strips = sorted(by_strip)

    # --- frame and base map ------------------------------------------------
    if georeferenced:
        references: dict[str, Georeference] = {}
        if references_for_survey is not None and manifest_dir is not None:
            # The survey's own best navigation, which for a strip ingested from
            # a raw log is the per-ping sidecar: the transform the engine used
            # to place every detection, rather than a refit to tile centres.
            try:
                references, _ = references_for_survey(rows, survey_meta, manifest_dir)
            except Exception:  # the map falls back; it does not fail
                log.exception("survey navigation unavailable to the map; refitting "
                              "from the manifest's tile centres instead")
                references = {}
        if not references:
            references, _ = references_from_manifest(rows)
        frame: Frame = GeoFrame(references)
        located = [d for d in detections if d.get("latitude") is not None]
        centre = ([sum(d["latitude"] for d in located) / len(located),
                   sum(d["longitude"] for d in located) / len(located)]
                  if located else [0.0, 0.0])
        fmap = folium.Map(location=centre, zoom_start=15, tiles=None,
                          control_scale=True, prefer_canvas=True)
        if basemap:
            folium.TileLayer("OpenStreetMap", name="Street basemap", show=False,
                             control=True).add_to(fmap)
    else:
        offsets: dict[str, float] = {}
        cursor = 0.0
        for strip in strips:
            offsets[strip] = cursor
            widths = [int(r.get("width") or 0) for r in by_strip[strip]]
            local = [d["global_x"] for d in detections
                     if str(d["provenance"]["strip"] or "") == strip]
            cursor += (max(widths) if widths else (max(local) if local else cfg.TILE)) \
                + theme.STRIP_GUTTER_PX
        frame = PixelFrame(offsets)
        # min_zoom below 0 so a strip taller than the window can be zoomed out
        # to fit. CRS.Simple has no natural zoom floor.
        # zoomSnap=0 is the whole fix for "the map opens mostly empty".
        #
        # Leaflet snaps a fitBounds zoom down to a whole integer by default.
        # On CRS.Simple a whole zoom step is a factor of two, so a survey that
        # ideally fits at zoom -1.79 is rendered at -2, at 0.25 scale instead
        # of 0.29, and the wasted linear scale squares into wasted area. On a
        # 1600x2048 survey in a typical window that is the difference between
        # filling 14% of the viewport and filling 40% of it. There are no
        # raster tiles here to misalign, so nothing is paid for it.
        #
        # minZoom and maxZoom go in as Leaflet's own option names. folium's
        # min_zoom and max_zoom are handed only to a tile layer it creates, and
        # with tiles=None there is none, so the floor silently fell back to 0
        # and fitBounds was clamped to it: a tall strip opened zoomed into its
        # top corner with the zoom-out button disabled.
        fmap = folium.Map(location=[0, 0], zoom_start=0, minZoom=-6, maxZoom=6,
                          crs="Simple", tiles=None, prefer_canvas=True,
                          zoomSnap=0, zoomDelta=0.5)

    # --- sonar imagery -----------------------------------------------------
    imagery = folium.FeatureGroup(name=theme.LAYERS["strip"],
                                  show="strip" in theme.LAYERS_ON_BY_DEFAULT)
    drawn_bounds: list[list[float]] = []
    skipped_imagery: list[str] = []
    resampled: list[str] = []
    extents: dict[str, tuple[int, int]] = {}

    for strip in strips:
        strip_rows = by_strip[strip]
        stitched = (_stitch_canvas(Path(tiles_dir), strip_rows)
                    if tiles_dir and strip_rows else None)
        if stitched is None:
            continue
        canvas, width, height = stitched

        if georeferenced:
            reference = getattr(frame, "references", {}).get(strip)
            if reference is None:
                continue
            aligned, north_up = _axis_aligned(reference)
            if not aligned:
                warped = _resample_north_up(canvas, reference) if _resamplable(reference) else None
                if warped is None:
                    skipped_imagery.append(strip)
                    continue
                folium.raster_layers.ImageOverlay(
                    image=warped["data_uri"], bounds=warped["bounds"], opacity=1.0,
                    zindex=1, pixelated=False).add_to(imagery)
                if len(warped["footprint"]) >= 3:
                    folium.Polygon(
                        locations=warped["footprint"], color=theme.STRIP_OUTLINE,
                        weight=theme.STRIP_OUTLINE_WEIGHT, fill=False,
                        tooltip=(f"{_esc(strip)} &middot; {width} x {height} px, "
                                 f"resampled north-up for display")).add_to(imagery)
                    drawn_bounds.extend(warped["footprint"])
                else:
                    drawn_bounds.extend(warped["bounds"])
                extents[strip] = (width, height)
                resampled.append(strip)
                continue
            data_uri = _jpeg_preview(canvas)
            bounds = frame.rect(strip, 0, 0, width, height)
            origin = "upper" if north_up else "lower"
        else:
            data_uri = _jpeg_preview(canvas)
            bounds = frame.rect(strip, 0, 0, width, height)
            origin = "upper"

        if bounds is None:
            continue
        folium.raster_layers.ImageOverlay(
            image=data_uri, bounds=bounds, origin=origin, opacity=1.0,
            zindex=1, pixelated=False).add_to(imagery)
        # A tall strip in a wide window leaves margins however well it is
        # fitted. The outline makes the survey's extent legible against them
        # rather than letting the image dissolve into the background.
        folium.Rectangle(bounds=bounds, color=theme.STRIP_OUTLINE,
                         weight=theme.STRIP_OUTLINE_WEIGHT, fill=False,
                         tooltip=f"{strip} &middot; {width} x {height} px"
                         ).add_to(imagery)
        extents[strip] = (width, height)
        drawn_bounds.extend(bounds)

    imagery.add_to(fmap)

    # --- tile footprints ---------------------------------------------------
    tile_layer = folium.FeatureGroup(name=theme.LAYERS["tiles"],
                                     show="tiles" in theme.LAYERS_ON_BY_DEFAULT)
    for strip in strips:
        for row in by_strip[strip]:
            x0, y0 = int(row["x"]), int(row["y"])
            x1 = x0 + int(row.get("tile_width") or cfg.TILE)
            y1 = y0 + int(row.get("tile_height") or cfg.TILE)
            tooltip = (f"{_esc(row['tile'])} &middot; content "
                       f"{float(row.get('content_score') or 0):.3f}")
            if not frame.aligned(strip):
                ring = frame.outline(strip, x0, y0, x1, y1)
                if ring is None:
                    continue
                folium.Polygon(
                    locations=ring, color=theme.COLOR["teal"],
                    weight=theme.TILE_OUTLINE_WEIGHT, opacity=theme.TILE_OUTLINE_OPACITY,
                    fill=False, tooltip=tooltip).add_to(tile_layer)
                drawn_bounds.extend(ring)
                continue
            bounds = frame.rect(strip, x0, y0, x1, y1)
            if bounds is None:
                continue
            folium.Rectangle(
                bounds=bounds, color=theme.COLOR["teal"],
                weight=theme.TILE_OUTLINE_WEIGHT, opacity=theme.TILE_OUTLINE_OPACITY,
                fill=False, tooltip=tooltip).add_to(tile_layer)
            drawn_bounds.extend(bounds)
    tile_layer.add_to(fmap)

    # --- severity heat -----------------------------------------------------
    # Weighted by severity, never by count. A hundred tyres must not glow
    # hotter than one mine, and a count-weighted heatmap does exactly that.
    # A detection verification filtered as a likely false positive adds no
    # heat, for the same reason it adds nothing to a hotspot.
    suppressed = [d for d in detections if d.get("suppressed")]
    reportable = [d for d in detections if not d.get("suppressed")]
    verified = ("suppressed_detections" in export["survey_summary"]
                or any("suppressed" in d for d in detections))
    heat_points = []
    for detection in reportable:
        position = frame.point(strip=str(detection["provenance"]["strip"] or ""),
                               gx=detection["global_x"], gy=detection["global_y"],
                               lat=detection.get("latitude"), lon=detection.get("longitude"))
        if position is None:
            continue
        heat_points.append([position[0], position[1], float(detection["severity"])])

    if heat_points:
        heat_group = folium.FeatureGroup(name=theme.LAYERS["heat"],
                                         show="heat" in theme.LAYERS_ON_BY_DEFAULT)
        HeatMap(
            heat_points,
            # Normalised against the strongest severity present, so the scale is
            # the survey's own. max_zoom keeps leaflet.heat from renormalising
            # as the operator zooms, which would make heat mean different things
            # at different zooms.
            max_zoom=6,
            radius=max(8, round(grid * theme.HEAT_RADIUS_FRACTION)),
            blur=max(6, round(grid * theme.HEAT_BLUR_FRACTION)),
            min_opacity=theme.HEAT_MIN_OPACITY,
            gradient=theme.HEAT_GRADIENT,
        ).add_to(heat_group)
        heat_group.add_to(fmap)

    # --- detections --------------------------------------------------------
    all_layer = folium.FeatureGroup(name=theme.LAYERS["detections"],
                                    show="detections" in theme.LAYERS_ON_BY_DEFAULT)
    tier_layers = {}
    for tier, _floor in cfg.SEVERITY_TIERS:
        tier_layers[tier] = folium.FeatureGroup(
            name=theme.LAYERS.get(tier, theme.tier_style(tier)["label"]),
            show=tier in theme.LAYERS_ON_BY_DEFAULT)

    # Present whenever the export was verified, even when nothing was
    # filtered: "(0)" is a finding, and an absent layer would not say it.
    filtered_layer = (folium.FeatureGroup(name=f"{FILTERED_LAYER} ({len(suppressed)})",
                                          show=False) if verified else None)

    unplaced = 0
    for detection in detections:
        position = frame.point(strip=str(detection["provenance"]["strip"] or ""),
                               gx=detection["global_x"], gy=detection["global_y"],
                               lat=detection.get("latitude"), lon=detection.get("longitude"))
        if position is None:
            unplaced += 1
            continue
        if detection.get("suppressed") and filtered_layer is not None:
            _detection_marker(detection, position).add_to(filtered_layer)
            drawn_bounds.append(position)
            continue
        _detection_marker(detection, position).add_to(all_layer)
        layer = tier_layers.get(detection["severity_tier"])
        if layer is not None:
            _detection_marker(detection, position).add_to(layer)
        drawn_bounds.append(position)

    all_layer.add_to(fmap)
    for layer in tier_layers.values():
        layer.add_to(fmap)
    if filtered_layer is not None:
        filtered_layer.add_to(fmap)

    # --- hotspots ----------------------------------------------------------
    # A hotspot is a grid cell, so it is drawn as that cell rather than as a
    # circle, and a numbered badge sits at the centroid. Drawing a radius the
    # aggregation never used would imply a precision it does not have.
    hotspot_layer = folium.FeatureGroup(name=theme.LAYERS["hotspots"],
                                        show="hotspots" in theme.LAYERS_ON_BY_DEFAULT)
    targets: dict[str, dict[str, Any]] = {}

    for hotspot in hotspots:
        strip = str(hotspot["strip"] or "")
        cell_x, cell_y = hotspot["cell"]
        size = int(hotspot["cell_size_px"])
        style = theme.tier_style(hotspot["severity_tier"])

        # A cell on the edge of a survey runs past the end of the strip. The
        # part beyond the imagery contains nothing and cannot, so it is clipped:
        # it would otherwise stretch the fitted bounds and pull the whole map
        # further out than the data warrants.
        x0, y0 = cell_x * size, cell_y * size
        x1, y1 = x0 + size, y0 + size
        extent = extents.get(strip)
        if extent:
            x1, y1 = min(x1, extent[0]), min(y1, extent[1])
        has_area = x1 > x0 and y1 > y0
        # On a strip that does not run north-up the cell is drawn as its true,
        # rotated footprint; its bounding box would claim ground it never held.
        ring = (frame.outline(strip, x0, y0, x1, y1)
                if has_area and not frame.aligned(strip) else None)
        bounds = frame.rect(strip, x0, y0, x1, y1) if has_area and ring is None \
            and frame.aligned(strip) else None
        centre = frame.point(strip, hotspot["centroid"]["global_x"],
                             hotspot["centroid"]["global_y"],
                             hotspot["centroid"].get("latitude"),
                             hotspot["centroid"].get("longitude"))
        if centre is None:
            continue

        if ring is not None:
            folium.Polygon(
                locations=ring, color=style["marker"], weight=theme.HOTSPOT_WEIGHT,
                fill=True, fill_color=style["fill"],
                fill_opacity=theme.HOTSPOT_FILL_OPACITY,
                tooltip=f"{hotspot['hotspot_id']} &middot; rank "
                        f"{hotspot['priority_rank']}").add_to(hotspot_layer)
            drawn_bounds.extend(ring)
        elif bounds is not None:
            folium.Rectangle(
                bounds=bounds, color=style["marker"], weight=theme.HOTSPOT_WEIGHT,
                fill=True, fill_color=style["fill"],
                fill_opacity=theme.HOTSPOT_FILL_OPACITY,
                tooltip=f"{hotspot['hotspot_id']} &middot; rank "
                        f"{hotspot['priority_rank']}").add_to(hotspot_layer)
            drawn_bounds.extend(bounds)

        badge = folium.Marker(
            location=centre,
            icon=folium.DivIcon(
                icon_size=(30, 30), icon_anchor=(15, 15),
                html=f"<div style=\"width:26px;height:26px;border-radius:50%;"
                     f"background:{style['marker']};color:#fff;font:600 12px/26px "
                     f"{theme.FONT_MONO};text-align:center;"
                     f"border:2px solid rgba(255,255,255,0.9);"
                     f"box-sizing:content-box\">{hotspot['priority_rank']}</div>"),
            popup=folium.Popup(_hotspot_popup(hotspot, frame), max_width=360),
            tooltip=f"{hotspot['hotspot_id']} &middot; {_esc(hotspot['dominant_class'])}")
        badge.add_to(hotspot_layer)

        targets[hotspot["hotspot_id"]] = {
            "center": centre,
            "zoom": 1 if frame.mode == "pixel" else 18,
            "marker": badge.get_name(),
        }

    hotspot_layer.add_to(fmap)
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    if drawn_bounds:
        lats = [p[0] for p in drawn_bounds]
        lons = [p[1] for p in drawn_bounds]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(24, 24))

    # --- chrome ------------------------------------------------------------
    from string import Template

    title = title or export["metadata"].get("survey_id") or "Survey"
    multi_strip = len(strips) > 1

    root = fmap.get_root()
    root.header.add_child(folium.Element(_css()))
    root.header.add_child(folium.Element(_POPUP_SHIM))
    root.html.add_child(folium.Element(_dashboard(export, title, demo, multi_strip, resampled)))
    root.script.add_child(folium.Element(Template(_JS).substitute(
        map=fmap.get_name(), targets=json.dumps(targets))))
    root.title = f"{_esc(title)} | {theme.TEXT['app']} {theme.TEXT['subtitle']}"

    path = out_dir / filename
    # Only folium's popup template produces this sequence: popup text has its
    # backticks escaped by folium, so it cannot appear inside content.
    html = root.render().replace(_JQUERY_POPUP_CALL, _SHIM_POPUP_CALL)

    if offline:
        try:
            html, assets = hazard_assets.inline(html)
        except hazard_assets.AssetError as exc:
            # Loud, because the failure mode it causes is a blank white page on
            # a machine with no network, and that is discovered on stage.
            log.warning("map.html will load Leaflet from a CDN and will NOT render "
                        "offline: %s", exc)
            assets = {"inlined": [], "dropped": [], "remaining_external": ["CDN fallback"]}
    else:
        assets = {"inlined": [], "dropped": [], "remaining_external": ["CDN by request"]}

    path.write_text(html, encoding="utf-8")

    if unplaced:
        log.warning("%d detection(s) had no position to draw and are not on the map",
                    unplaced)
    if skipped_imagery:
        log.info("sonar imagery omitted for strip(s) whose navigation cannot place "
                 "every pixel: %s", ", ".join(skipped_imagery))
    if resampled:
        log.info("sonar imagery resampled north-up for display: %s", ", ".join(resampled))
    external = [u for u in assets["remaining_external"] if "openstreetmap" not in u.lower()]
    log.info("wrote %s (%.1f KB, %s, %d detections, %d hotspots, %s)",
             path.name, path.stat().st_size / 1024,
             "geo-referenced" if georeferenced else "relative pixels",
             len(detections), len(hotspots),
             "self-contained" if not external else f"{len(external)} external reference(s)")
    return path
