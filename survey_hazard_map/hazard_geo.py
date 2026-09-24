"""Pixels to latitude and longitude, or an honest refusal.

Everything in this module exists to keep one promise: the system never invents
a position. A tile that cannot be located carries lat = None and lon = None,
and the survey is marked "Relative Survey Coordinates". Relative pixel
coordinates are always preserved, with or without navigation, because they are
the survey's real spatial identity and a geographic fix is a decoration on top
of them.

Three navigation inputs are supported.

CONTROL POINTS (a CSV of pixel -> lat/lon fixes)
    Columns, case-insensitive, in any order:
        strip       the strip's name, matching the image file's stem
        pixel_x     column in the original strip, "x" also accepted
        pixel_y     row in the original strip, "y" also accepted
        latitude    decimal degrees, "lat" also accepted
        longitude   decimal degrees, "lon" or "lng" also accepted

    With three or more fixes that are not collinear, a first-order affine
    transform is fitted by least squares:

        lon = a*px + b*py + c
        lat = d*px + e*py + f

    Six parameters from a linear solve. The fit's residual is reported so a bad
    nav file shows up as a number rather than as quietly wrong positions.

    A side-scan nav file often holds one fix per ping row, so every fix shares
    the same pixel column and the points are collinear. An affine fit is then
    underdetermined across-track and would be a fabrication. That case falls
    back to one-dimensional interpolation along the track line, and the tile's
    position is the along-track position of its centre row. Across-track
    displacement is NOT resolved, and the export says so in
    `provenance.navigation.across_track_resolved`, because without a range
    scale there is nothing to resolve it from.

FOUR CORNERS (the geographic corners of the whole strip)
    A dict per strip with top_left, top_right, bottom_left and bottom_right,
    each [latitude, longitude] in decimal degrees. A tile centre at normalised
    position (u, v) is interpolated bilinearly between them.

PING NAVIGATION (a sidecar written by sonar_ingest, one record per image row)
    A strip produced from a raw XTF or JSF log carries its own navigation: for
    every row of the image, the towfish position, heading and attitude at that
    row, plus the strip's across- and along-track resolution in metres. This is
    the only mode that knows what a pixel measures, so it is the only one that
    resolves across-track position from physics rather than from a fit:

        row           cy - 0.5, interpolated between the two nearest rows
        position      lat/lon of the nadir track at that row
        heading       interpolated on the circle, so 359 -> 1 passes through 0
        offset_m      (cx - nadir_col) * m_per_px_across, positive = starboard
        bearing       heading + 90 degrees
        result        nadir position moved offset_m along bearing

    The last step uses a local flat-earth (east-north-up) approximation with
    the WGS84 radii of curvature at the nadir latitude. Over a side-scan swath
    of a few hundred metres the error of that approximation is millimetres,
    far below the sonar's own resolution. Rows with a null position (a rejected
    navigation fix) are interpolated from the nearest located rows when both
    lie within PING_NAV_MAX_FILL_M; beyond that the answer is (None, None).

    What this does NOT correct, and the sidecar's processing block says so:
    layback between the recorded sensor position and the true towfish
    position, pitch and yaw displacement of the beam footprint, refraction,
    and layover of tall objects. The flat-seabed slant-range correction
    already applied by sonar_ingest assumes the seabed is level across the
    swath.

Corner and control-point methods assume the strip covers a small enough area
that the seabed is flat and degrees are locally linear, which is true of a
survey strip and not true of a basin. None of the methods is valid across the
antimeridian or over a pole.

Every position is computed from the TILE CENTRE, never its top-left corner.
Locating a 640-pixel tile by its corner puts it half a tile off, consistently,
in the same direction, which is the kind of error that survives review because
everything looks plausible.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

# Below this, the second singular value of the centred control points is noise
# and the points are treated as lying on a line. Scaled against the first
# singular value, so it is a shape test and does not care about survey size.
COLLINEAR_RATIO = 1e-6

# Ping navigation: a row with no position is filled from the nearest located
# rows only when the unlocated run between them is no longer than this. A few
# rejected fixes are bridged; a long navigation outage is not papered over.
PING_NAV_MAX_FILL_M = float(os.environ.get("HAZARD_PING_NAV_MAX_FILL_M", "30"))

# The sidecar format this module reads. Checked, so a different JSON dropped
# beside an image is refused rather than half-understood.
PING_SIDECAR_FORMAT = "deepecho-strip-nav/1"

# WGS84. Used only for the local metre <-> degree conversion.
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3

_LAT_KEYS = ("latitude", "lat")
_LON_KEYS = ("longitude", "lon", "lng", "long")
_X_KEYS = ("pixel_x", "x", "px")
_Y_KEYS = ("pixel_y", "y", "py")


class NavigationError(ValueError):
    """The navigation input was given but cannot be used as supplied."""


def _pick(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return row[key]
    return None


def _as_pair(value: Any) -> tuple[float, float]:
    """A corner, as [lat, lon] or {"latitude": .., "longitude": ..}."""
    if isinstance(value, dict):
        lat = _pick(value, _LAT_KEYS)
        lon = _pick(value, _LON_KEYS)
        if lat is None or lon is None:
            raise NavigationError(f"corner {value!r} needs a latitude and a longitude")
        return float(lat), float(lon)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise NavigationError(f"corner {value!r} must be [latitude, longitude]")


def _check_degrees(lat: float, lon: float, where: str) -> None:
    if not (-90.0 <= lat <= 90.0):
        raise NavigationError(f"{where}: latitude {lat} is outside -90..90")
    if not (-180.0 <= lon <= 180.0):
        raise NavigationError(f"{where}: longitude {lon} is outside -180..180")


# --- Local metres <-> degrees ----------------------------------------------


def radii_of_curvature(lat_deg: float) -> tuple[float, float]:
    """(meridional, prime-vertical) WGS84 radii of curvature in metres."""
    s = math.sin(math.radians(lat_deg))
    w = 1.0 - _WGS84_E2 * s * s
    return _WGS84_A * (1.0 - _WGS84_E2) / (w ** 1.5), _WGS84_A / math.sqrt(w)


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float
                  ) -> tuple[float, float]:
    """Move a position by a local east/north offset in metres.

    Flat-earth (east-north-up) approximation about the starting point, using
    the WGS84 radii there. Accurate to millimetres over hundreds of metres,
    which is the only scale it is used at.
    """
    meridional, prime = radii_of_curvature(lat)
    dlat = math.degrees(north_m / meridional)
    dlon = math.degrees(east_m / (prime * max(math.cos(math.radians(lat)), 1e-12)))
    return lat + dlat, lon + dlon


def enu_delta(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """(east_m, north_m) of a position relative to a reference, flat-earth.

    The inverse of offset_latlon, with the same assumption and the same scale.
    """
    meridional, prime = radii_of_curvature(lat0)
    north = math.radians(lat - lat0) * meridional
    east = math.radians(lon - lon0) * prime * math.cos(math.radians(lat0))
    return east, north


class Georeference:
    """Locates a tile centre within one strip, or refuses to.

    `mode` is "affine", "along_track", "corners" or "ping". A strip with no
    usable navigation gets no Georeference at all and its tiles carry nulls.
    """

    def __init__(self, strip: str, mode: str, detail: dict[str, Any]) -> None:
        self.strip = strip
        self.mode = mode
        self.detail = detail
        # Per-row arrays for ping navigation. Kept off `detail` because detail
        # is what describe() puts in the export, and thousands of rows of
        # navigation do not belong in a provenance block.
        self._nav: dict[str, Any] | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_control_points(cls, strip: str, points: list[tuple[float, float, float, float]]
                            ) -> "Georeference":
        """points are (pixel_x, pixel_y, latitude, longitude)."""
        import numpy as np

        if len(points) < 2:
            raise NavigationError(
                f"strip {strip!r}: {len(points)} navigation fix(es); at least 2 are needed")

        array = np.asarray(points, dtype=float)
        pixels, lats, lons = array[:, :2], array[:, 2], array[:, 3]

        centred = pixels - pixels.mean(axis=0)
        singular = np.linalg.svd(centred, compute_uv=False)
        spread = float(singular[0])
        if spread <= 0.0:
            raise NavigationError(
                f"strip {strip!r}: every navigation fix is at the same pixel")
        collinear = float(singular[1]) / spread < COLLINEAR_RATIO

        if collinear or len(points) < 3:
            # One fix per ping row, the usual side-scan case. Project every fix
            # onto the track line and interpolate along it. Across-track is
            # left unresolved rather than guessed.
            direction = centred[np.argmax(np.abs(centred).sum(axis=1))]
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 0 else np.array([0.0, 1.0])
            t = centred @ direction
            order = np.argsort(t)
            return cls(strip, "along_track", {
                "origin": pixels.mean(axis=0).tolist(),
                "direction": direction.tolist(),
                "t": t[order].tolist(),
                "lat": lats[order].tolist(),
                "lon": lons[order].tolist(),
                "fixes": len(points),
                "across_track_resolved": False,
                "method": "1-D linear interpolation along the fitted track line, "
                          "extrapolated linearly from the end segment beyond the "
                          "outermost fix",
            })

        # Three or more fixes with genuine two-dimensional spread: fit an
        # affine transform and report how well it fits.
        design = np.column_stack([pixels[:, 0], pixels[:, 1], np.ones(len(pixels))])
        lon_coef, *_ = np.linalg.lstsq(design, lons, rcond=None)
        lat_coef, *_ = np.linalg.lstsq(design, lats, rcond=None)
        residual = float(np.max(np.hypot(design @ lat_coef - lats, design @ lon_coef - lons)))
        return cls(strip, "affine", {
            "lat_coefficients": lat_coef.tolist(),
            "lon_coefficients": lon_coef.tolist(),
            "fixes": len(points),
            "max_residual_degrees": round(residual, 9),
            "across_track_resolved": True,
            "method": "first-order affine transform fitted by least squares over "
                      "all navigation fixes",
        })

    @classmethod
    def from_corners(cls, strip: str, corners: dict[str, Any],
                     width: int, height: int) -> "Georeference":
        missing = [k for k in ("top_left", "top_right", "bottom_left", "bottom_right")
                   if k not in corners]
        if missing:
            raise NavigationError(f"strip {strip!r}: corners missing {', '.join(missing)}")
        resolved = {}
        for key in ("top_left", "top_right", "bottom_left", "bottom_right"):
            lat, lon = _as_pair(corners[key])
            _check_degrees(lat, lon, f"strip {strip!r} {key}")
            resolved[key] = (lat, lon)
        return cls(strip, "corners", {
            "corners": {k: list(v) for k, v in resolved.items()},
            "width": width,
            "height": height,
            "across_track_resolved": True,
            "method": "bilinear interpolation of the four strip corners over "
                      "normalised pixel position",
        })

    @classmethod
    def from_ping_nav(cls, strip: str, sidecar: dict[str, Any]) -> "Georeference":
        """Per-row navigation from a sonar_ingest sidecar. See PING NAVIGATION."""
        import numpy as np

        if not isinstance(sidecar, dict) or sidecar.get("format") != PING_SIDECAR_FORMAT:
            raise NavigationError(
                f"strip {strip!r}: navigation sidecar is not {PING_SIDECAR_FORMAT!r} "
                f"(format={sidecar.get('format') if isinstance(sidecar, dict) else None!r})")
        rows = sidecar.get("rows") or []
        if not rows:
            raise NavigationError(f"strip {strip!r}: navigation sidecar has no rows")
        try:
            nadir_col = float(sidecar["nadir_col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NavigationError(f"strip {strip!r}: sidecar has no usable nadir_col") from exc

        def column(key: str) -> "np.ndarray":
            return np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows],
                            dtype=float)

        across = sidecar.get("m_per_px_across")
        along = sidecar.get("m_per_px_along")
        across = None if across is None else float(across)
        along = None if along is None else float(along)

        lat, lon, heading = column("lat"), column("lon"), column("heading_deg")
        valid = np.isfinite(lat) & np.isfinite(lon)
        if not valid.any():
            raise NavigationError(
                f"strip {strip!r}: the sidecar records no latitude or longitude on any row "
                f"(coordinate units: {sidecar.get('coordinate_units')!r})")

        # Fill limit in rows. Without an along-track scale there is no way to
        # say how far a run of rows is, so nothing is filled.
        limit_rows = (PING_NAV_MAX_FILL_M / along) if along else 0.0
        lat_f = _fill_short_gaps(lat, limit_rows)
        lon_f = _fill_short_gaps(lon, limit_rows)
        # Heading is filled on the circle through its unit vector.
        rad = np.radians(heading)
        hx = _fill_short_gaps(np.sin(rad), limit_rows)
        hy = _fill_short_gaps(np.cos(rad), limit_rows)

        reference = cls(strip, "ping", {
            "source_file": sidecar.get("source_file"),
            "source_format": sidecar.get("source_format"),
            "synthetic": bool(sidecar.get("synthetic")),
            "width": sidecar.get("width"),
            "height": sidecar.get("height"),
            "rows": len(rows),
            "rows_with_position": int(valid.sum()),
            "rows_locatable": int((np.isfinite(lat_f) & np.isfinite(hx)).sum()),
            "nadir_col": nadir_col,
            "m_per_px_across": across,
            "m_per_px_along": along,
            "port_is_left": bool(sidecar.get("port_is_left", True)),
            "coordinate_units": sidecar.get("coordinate_units"),
            "max_fill_m": PING_NAV_MAX_FILL_M,
            "across_track_resolved": across is not None,
            "method": "per-row towfish navigation interpolated between the two nearest "
                      "rows; across-track offset (x - nadir_col) * m_per_px_across applied "
                      "along heading + 90 degrees with a local flat-earth ENU "
                      "approximation (WGS84 radii)",
        })
        reference._nav = {"lat": lat_f, "lon": lon_f, "hx": hx, "hy": hy,
                          "rows": rows}
        return reference

    # -- use ----------------------------------------------------------------

    def locate(self, center_x: float, center_y: float) -> tuple[float | None, float | None]:
        """Latitude and longitude of a tile or detection centre, in degrees."""
        if self.mode == "ping":
            return self._locate_ping(center_x, center_y)
        if self.mode == "corners":
            return self._locate_corners(center_x, center_y)
        if self.mode == "affine":
            return self._locate_affine(center_x, center_y)
        return self._locate_along_track(center_x, center_y)

    def metres_per_pixel(self) -> tuple[float | None, float | None]:
        """(across-track, along-track) metres per pixel, or None where unknown.

        Only ping navigation measures this. The other modes map pixels to
        degrees without ever knowing what a pixel is on the seabed, so they
        answer (None, None) rather than deriving a scale from a fit.
        """
        return self.detail.get("m_per_px_across"), self.detail.get("m_per_px_along")

    def _row_bracket(self, cy: float) -> tuple[int, int, float]:
        rows = len(self._nav["rows"])
        # Continuous pixel y: row r spans [r, r+1) and its navigation describes
        # its centre, r + 0.5. Clamped to the strip: a box centre can sit at
        # most half a pixel past the last row centre.
        t = min(max(float(cy) - 0.5, 0.0), float(rows - 1))
        lo = int(math.floor(t))
        hi = min(lo + 1, rows - 1)
        return lo, hi, t - lo

    def _locate_ping(self, cx: float, cy: float) -> tuple[float | None, float | None]:
        nav = self._nav
        across = self.detail.get("m_per_px_across")
        if nav is None or across is None:
            return None, None
        lo, hi, w = self._row_bracket(cy)
        values = []
        for key in ("lat", "lon", "hx", "hy"):
            a, b = float(nav[key][lo]), float(nav[key][hi])
            if not (math.isfinite(a) and math.isfinite(b)):
                return None, None
            values.append(a + (b - a) * w)
        lat, lon, hx, hy = values
        heading = math.degrees(math.atan2(hx, hy))
        offset = (float(cx) - self.detail["nadir_col"]) * across
        if not self.detail.get("port_is_left", True):
            offset = -offset
        bearing = math.radians(heading + 90.0)
        lat, lon = offset_latlon(lat, lon, offset * math.sin(bearing),
                                 offset * math.cos(bearing))
        return round(lat, 8), round(lon, 8)

    def nav_row(self, cy: float) -> dict[str, Any] | None:
        """The navigation record at a continuous pixel row, interpolated.

        Numeric fields are interpolated between the two nearest rows (heading
        on the circle); a field null on either row is null. `quality` is the
        worse of the two rows' flags, `time` is taken from the nearer row. None
        for a Georeference that is not ping navigation.
        """
        if self._nav is None:
            return None
        rows = self._nav["rows"]
        lo, hi, w = self._row_bracket(cy)
        a, b = rows[lo], rows[hi]
        out: dict[str, Any] = {"row": round(lo + w, 3)}
        for key in ("lat", "lon", "altitude_m", "speed_mps", "pitch_deg",
                    "roll_deg", "heave_m"):
            va, vb = a.get(key), b.get(key)
            out[key] = None if va is None or vb is None else va + (vb - va) * w
        ha, hb = a.get("heading_deg"), b.get("heading_deg")
        if ha is None or hb is None:
            out["heading_deg"] = None
        else:
            ra, rb = math.radians(ha), math.radians(hb)
            sx = math.sin(ra) + (math.sin(rb) - math.sin(ra)) * w
            sy = math.cos(ra) + (math.cos(rb) - math.cos(ra)) * w
            out["heading_deg"] = math.degrees(math.atan2(sx, sy)) % 360.0
        out["time"] = (a if w < 0.5 else b).get("time")
        order = ("ok", "attitude", "nav_jump", "interpolated", "dropout")
        qa, qb = a.get("quality"), b.get("quality")
        known = [q for q in (qa, qb) if q in order]
        out["quality"] = max(known, key=order.index) if known else (qa or qb)
        return out

    def _locate_corners(self, cx: float, cy: float) -> tuple[float, float]:
        detail = self.detail
        corners = detail["corners"]
        # Normalise against the last addressable pixel, so the far corner lands
        # exactly on the far corner rather than one pixel short of it.
        span_x = max(detail["width"] - 1, 1)
        span_y = max(detail["height"] - 1, 1)
        u = min(max(cx / span_x, 0.0), 1.0)
        v = min(max(cy / span_y, 0.0), 1.0)
        out = []
        for axis in (0, 1):
            top = (1 - u) * corners["top_left"][axis] + u * corners["top_right"][axis]
            bottom = (1 - u) * corners["bottom_left"][axis] + u * corners["bottom_right"][axis]
            out.append((1 - v) * top + v * bottom)
        return round(out[0], 8), round(out[1], 8)

    def _locate_affine(self, cx: float, cy: float) -> tuple[float, float]:
        a, b, c = self.detail["lat_coefficients"]
        d, e, f = self.detail["lon_coefficients"]
        return round(a * cx + b * cy + c, 8), round(d * cx + e * cy + f, 8)

    def _locate_along_track(self, cx: float, cy: float) -> tuple[float, float]:
        ox, oy = self.detail["origin"]
        dx, dy = self.detail["direction"]
        t = (cx - ox) * dx + (cy - oy) * dy
        return (round(_interpolate(t, self.detail["t"], self.detail["lat"]), 8),
                round(_interpolate(t, self.detail["t"], self.detail["lon"]), 8))

    def describe(self) -> dict[str, Any]:
        """What went into this fix, for the provenance block."""
        described = {"strip": self.strip, "mode": self.mode}
        described.update({k: v for k, v in self.detail.items()
                          if k not in {"t", "lat", "lon", "origin", "direction"}})
        return described


def _fill_short_gaps(values: Any, limit_rows: float) -> Any:
    """Linear fill of NaN runs bounded on both sides and no longer than limit.

    Never extrapolates past the first or last finite value: a strip that
    starts before its first fix has no position there, and says so.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.all() or not finite.any():
        return values.copy()
    index = np.arange(len(values))
    known = index[finite]
    filled = np.interp(index, known, values[finite])
    after = np.searchsorted(known, index)            # first known >= i
    before = after - 1
    has_both = (before >= 0) & (after < len(known))
    span = np.where(has_both, known[np.clip(after, 0, len(known) - 1)]
                    - known[np.clip(before, 0, len(known) - 1)], np.inf)
    usable = finite | (has_both & (span - 1 <= limit_rows))
    filled[~usable] = np.nan
    return filled


def _interpolate(t: float, knots: list[float], values: list[float]) -> float:
    """Linear interpolation, extrapolated from the end segment beyond the ends.

    numpy.interp clamps instead, which would put every tile past the last
    navigation fix at exactly that fix's position. Clamping looks safe and is
    not: it silently stacks detections on a single point. Extrapolating from
    the end segment is the same first-order assumption the interior already
    rests on, applied consistently.
    """
    if len(knots) == 1:
        return values[0]
    if t <= knots[0]:
        lo, hi = 0, 1
    elif t >= knots[-1]:
        lo, hi = len(knots) - 2, len(knots) - 1
    else:
        hi = next(i for i in range(1, len(knots)) if knots[i] >= t)
        lo = hi - 1
    span = knots[hi] - knots[lo]
    if span == 0:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (t - knots[lo]) / span


# --- Loading ---------------------------------------------------------------


def load_control_points(path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Navigation fixes from a CSV, grouped by strip."""
    path = Path(path)
    if not path.is_file():
        raise NavigationError(f"navigation file not found: {path}")

    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise NavigationError(f"{path.name} has no header row")
        # Normalise the header once so "Pixel_X" and "pixel_x" are the same.
        for raw in reader:
            row = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}
            strip = _pick(row, ("strip", "strip_name", "source", "image"))
            px, py = _pick(row, _X_KEYS), _pick(row, _Y_KEYS)
            lat, lon = _pick(row, _LAT_KEYS), _pick(row, _LON_KEYS)
            if None in (px, py, lat, lon):
                continue
            try:
                values = (float(px), float(py), float(lat), float(lon))
            except (TypeError, ValueError) as exc:
                raise NavigationError(f"{path.name}: non-numeric row {raw!r}") from exc
            _check_degrees(values[2], values[3], f"{path.name} row for {strip!r}")
            key = str(strip).strip() if strip is not None else ""
            grouped.setdefault(Path(key).stem if key else "", []).append(values)

    if not grouped:
        raise NavigationError(
            f"{path.name}: no usable rows. Needs strip, pixel_x, pixel_y, "
            f"latitude and longitude.")
    return grouped


def build_references(nav: Any, strips: dict[str, tuple[int, int]]
                     ) -> tuple[dict[str, Georeference], dict[str, Any]]:
    """Turn whatever the caller passed into one Georeference per strip.

    `strips` maps a strip name to its (width, height). Returns the references
    and a description of where they came from, for the provenance block. A
    strip the navigation does not mention simply gets no reference, and its
    tiles carry nulls; that is a partial survey, not an error.
    """
    if nav is None:
        return {}, {"source": None, "mode": "none",
                    "note": "No navigation supplied. Positions are relative survey pixels."}

    if isinstance(nav, (str, Path)):
        nav = {"mode": "control_points", "path": str(nav)}

    if not isinstance(nav, dict):
        raise NavigationError(
            "nav must be None, a path to a control-point CSV, or a dict")

    references: dict[str, Georeference] = {}

    # A bare {"strip": {"top_left": ...}} is accepted as corners, so the common
    # case does not need a mode field.
    mode = nav.get("mode")
    if mode is None:
        mode = "control_points" if "path" in nav else "corners"

    if mode == "control_points":
        path = nav.get("path") or nav.get("csv")
        if not path:
            raise NavigationError("control-point navigation needs a 'path'")
        grouped = load_control_points(Path(path))
        for strip in strips:
            points = grouped.get(strip) or grouped.get("")
            if not points:
                continue
            references[strip] = Georeference.from_control_points(strip, points)
        source = {"source": str(Path(path).name), "mode": "control_points"}

    elif mode == "corners":
        table = nav.get("strips", {k: v for k, v in nav.items() if k != "mode"})
        for strip, (width, height) in strips.items():
            corners = table.get(strip)
            if corners is None and len(table) == 1 and len(strips) == 1:
                # One strip, one corner set, no name match. Obvious intent.
                corners = next(iter(table.values()))
            if corners is None:
                continue
            references[strip] = Georeference.from_corners(strip, corners, width, height)
        source = {"source": "four-corner coordinates supplied by the caller",
                  "mode": "corners"}

    elif mode == "ping":
        sidecars = nav.get("sidecars") or {}
        if not isinstance(sidecars, dict) or not sidecars:
            raise NavigationError("ping navigation needs 'sidecars': {strip: path}")
        base = Path(nav["base_dir"]) if nav.get("base_dir") else None
        rejected: dict[str, str] = {}
        for strip, (width, height) in strips.items():
            entry = sidecars.get(strip)
            if entry is None:
                continue
            path = Path(entry)
            if base is not None and not path.is_absolute():
                path = base / path
            try:
                sidecar = load_ping_sidecar(path)
                # A sidecar describes one image, row for row and column for
                # column. Paired with an image of a different size it would
                # locate every pixel confidently and wrongly, so it is refused.
                if (width and height and sidecar.get("width") and sidecar.get("height")
                        and (int(sidecar["width"]), int(sidecar["height"]))
                        != (int(width), int(height))):
                    raise NavigationError(
                        f"sidecar is {sidecar['width']}x{sidecar['height']} px but the "
                        f"strip is {width}x{height} px")
                references[strip] = Georeference.from_ping_nav(strip, sidecar)
            except NavigationError as exc:
                rejected[strip] = str(exc)
        source = {"source": "per-ping navigation sidecars written by sonar_ingest",
                  "mode": "ping",
                  "sidecars": {k: str(v) for k, v in sidecars.items()}}
        if rejected:
            source["sidecars_rejected"] = rejected
    else:
        raise NavigationError(f"unknown navigation mode {mode!r}")

    if not references:
        detail = ""
        if mode == "ping" and source.get("sidecars_rejected"):
            detail = " Rejected: " + "; ".join(
                f"{k}: {v}" for k, v in sorted(source["sidecars_rejected"].items()))
        raise NavigationError(
            "navigation was supplied but matched none of the strips "
            f"({', '.join(sorted(strips)) or 'none'}). Check the strip names.{detail}")

    source["strips_located"] = sorted(references)
    source["strips_unlocated"] = sorted(set(strips) - set(references))
    source["references"] = [ref.describe() for ref in references.values()]
    source["across_track_resolved"] = all(
        ref.detail.get("across_track_resolved", True) for ref in references.values())
    source["assumptions"] = (
        "Locally flat seabed and locally linear degrees over the extent of one "
        "strip. Not valid across the antimeridian or over a pole.")
    if mode == "ping":
        source["assumptions"] += (
            " Ping navigation: position recorded for the sensor is used as the "
            "towfish position (no layback correction); across-track offsets use a "
            "flat seabed and a local flat-earth ENU approximation; pitch and yaw "
            "displacement of the beam footprint are not corrected.")
    return references, source


def load_ping_sidecar(path: Any) -> dict[str, Any]:
    """Read and minimally validate a sonar_ingest navigation sidecar."""
    path = Path(path)
    if not path.is_file():
        raise NavigationError(f"navigation sidecar not found: {path.name}")
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise NavigationError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(sidecar, dict) or sidecar.get("format") != PING_SIDECAR_FORMAT:
        raise NavigationError(f"{path.name} is not a {PING_SIDECAR_FORMAT} sidecar")
    return sidecar


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Used for reporting, never for merging."""
    radius = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(h)))


def references_from_manifest(rows: list[dict[str, Any]]
                             ) -> tuple[dict[str, Georeference], dict[str, Any]]:
    """Recover a per-strip transform from a manifest's located tile centres.

    Module 2 is given a tile set and a manifest, not the original navigation
    file, so it has to reconstruct the mapping to place a detection that sits
    somewhere inside a tile rather than at its centre.

    Each located tile contributes one control point: its centre pixel and the
    latitude and longitude recorded for it. The same fitter used for a raw
    navigation file then runs over those points, and reports its residual. For
    a manifest written by affine or along-track navigation the refit is exact.
    For four-corner navigation it is a first-order approximation of a bilinear
    surface, and `max_residual_degrees` is how far off it is, in degrees, at
    the worst tile. That number is in the export rather than in a comment,
    because an approximation nobody can measure is indistinguishable from an
    error.

    A strip with fewer than two located tiles cannot be fitted at all. Its
    detections inherit the position of the tile they were found in, which is a
    real recorded fix rather than an extrapolation, and the provenance says so.
    """
    points: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in rows:
        lat, lon = row.get("lat"), row.get("lon")
        if lat is None or lon is None or lat == "" or lon == "":
            continue
        strip = str(row.get("strip") or "")
        cx = row.get("center_x")
        cy = row.get("center_y")
        if cx is None or cy is None:
            # An older or hand-edited manifest without the centre columns.
            # Reconstruct it from the offset and the tile's own size.
            cx = float(row["x"]) + float(row.get("tile_width") or 0) / 2.0
            cy = float(row["y"]) + float(row.get("tile_height") or 0) / 2.0
        points.setdefault(strip, []).append((float(cx), float(cy), float(lat), float(lon)))

    references: dict[str, Georeference] = {}
    inherited: list[str] = []
    for strip, fixes in points.items():
        try:
            references[strip] = Georeference.from_control_points(strip, fixes)
        except NavigationError:
            inherited.append(strip)

    if not points:
        return {}, {"source": None, "mode": "none",
                    "note": "The manifest records no latitude or longitude. "
                            "Positions are relative survey pixels."}

    described = {
        "source": "refitted from the manifest's located tile centres",
        "mode": "manifest_refit",
        "strips_located": sorted(references),
        "strips_inheriting_tile_position": sorted(inherited),
        "references": [ref.describe() for ref in references.values()],
        "across_track_resolved": all(
            ref.detail.get("across_track_resolved", True) for ref in references.values()),
        "assumptions": (
            "Locally flat seabed and locally linear degrees over the extent of "
            "one strip. Not valid across the antimeridian or over a pole."),
    }
    return references, described


def references_for_survey(rows: list[dict[str, Any]], survey_meta: dict[str, Any] | None,
                          manifest_dir: Any) -> tuple[dict[str, Georeference], dict[str, Any]]:
    """The best navigation a prepared survey can offer, per strip.

    A strip ingested from a raw sonar log has a per-ping sidecar recorded in
    the manifest's survey block ("navigation": {"mode": "ping", "sidecars":
    {strip: "nav/<file>"}}), resolved against `manifest_dir`. That is the
    original navigation, row for row, so it is preferred over refitting a
    transform to tile centres, which for a curving towfish track would be a
    first-order approximation of a curve.

    Every strip without a usable sidecar falls back to references_from_manifest,
    exactly as before. A sidecar that is recorded but missing or unreadable is
    named in the provenance rather than skipped silently.
    """
    navigation = (survey_meta or {}).get("navigation") or {}
    sidecars = navigation.get("sidecars") if navigation.get("mode") == "ping" else None
    if not sidecars:
        return references_from_manifest(rows)

    base = Path(manifest_dir) if manifest_dir is not None else Path(".")
    sizes: dict[str, tuple[int, int]] = {}
    for row in rows:
        strip = str(row.get("strip") or "")
        if strip and strip not in sizes:
            sizes[strip] = (int(row.get("width") or 0), int(row.get("height") or 0))
    for entry in (survey_meta or {}).get("strips", []):
        strip = str(entry.get("strip") or "")
        if strip and strip not in sizes:
            sizes[strip] = (int(entry.get("width") or 0), int(entry.get("height") or 0))

    missing: dict[str, str] = {}
    present: dict[str, str] = {}
    for strip, relative in sidecars.items():
        path = Path(relative)
        path = path if path.is_absolute() else base / path
        if strip not in sizes:
            continue
        if path.is_file():
            present[strip] = str(path)
        else:
            missing[strip] = f"sidecar {Path(relative).name} not found beside the manifest"

    references: dict[str, Georeference] = {}
    source: dict[str, Any] = {"mode": "ping", "source": "per-ping navigation sidecars "
                              "recorded in the manifest"}
    if present:
        try:
            references, source = build_references(
                {"mode": "ping", "sidecars": present}, {s: sizes[s] for s in present})
        except NavigationError as exc:
            source = {"mode": "ping", "source": "per-ping navigation sidecars recorded in "
                      "the manifest", "error": str(exc)}
    # Report the sidecars as the manifest names them, never as absolute paths.
    source["sidecars"] = {k: str(v) for k, v in sidecars.items()}
    if missing:
        source["sidecars_missing"] = missing

    remaining = [r for r in rows if str(r.get("strip") or "") not in references]
    if remaining:
        refit, refit_source = references_from_manifest(remaining)
        if refit:
            references.update(refit)
            source["fallback"] = refit_source
            source["strips_located"] = sorted(references)
        elif not references:
            return refit, {**refit_source, "ping_navigation": source}

    source["across_track_resolved"] = all(
        ref.detail.get("across_track_resolved", True) for ref in references.values())
    source["references"] = [ref.describe() for ref in references.values()]
    source["strips_located"] = sorted(references)
    return references, source
