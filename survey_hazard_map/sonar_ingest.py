"""Raw side-scan logs in, geometrically honest strips out.

    from survey_hazard_map.sonar_ingest import ingest
    strips = ingest("survey/line01.xtf", "out/strips")
    for s in strips:
        s.image_path, s.nav_path, s.summary

The rest of the survey engine works on images. A raw sonar log is not an
image: it is a sequence of pings, each a time series of echo strength along a
slant range, recorded from a vehicle that heaves, pitches, rolls, speeds up,
slows down and occasionally stops recording. Turning that into a picture in
which one pixel is a fixed patch of seabed, and in which every row knows where
on Earth it is, is this module's whole job.

WHAT COMES OUT, PER CONTIGUOUS LINE
    {strip}.png         8-bit grey waterfall. Port on the left, starboard on
                        the right, nadir between them at column nadir_col. Row
                        0 is the first ping in time. Pixel value 0 means "no
                        data" and nothing else: every real measurement is
                        scaled into 1..255.
    {strip}.nav.json    the sidecar (format deepecho-strip-nav/1): one record
                        per image row with time, position, heading, attitude,
                        altitude, depth and a quality flag, the strip's
                        resolution in metres, the ranges of degraded rows, and
                        a processing block recording every step, parameter and
                        data source used, including what could not be verified.
    {strip}.wc.npz      the water column the slant-range correction discards
                        from the image, kept for GhostTrace: per output row,
                        port and starboard slant-range bins from the transducer
                        to the first bottom return.

THE PIPELINE, AND WHY EACH STEP IS THERE
    read            XTF through pyxtf; JSF through a minimal reader for message
                    type 80. Positions are taken ONLY when the file says they
                    are degrees. Projected or unknown units give null lat/lon
                    and the sidecar says why. A position is never guessed.
    split           a long time gap or a change of range setting starts a new
                    strip, because one image cannot honestly have two
                    across-track scales.
    bottom track    first strong, sustained return per side, median-filtered
                    along track. Falls back to the header altitude where the
                    tracker finds nothing. Every ping records which source won.
    slant range     ground = sqrt(slant^2 - altitude^2) onto a fixed grid of
                    m_per_px_across metres. This is what makes the across-track
                    axis metres rather than seconds, and it removes the water
                    column from the image by construction.
    gain            empirical gain normalisation: each ground-range bin divided
                    by its mean over the line, per side, per altitude band. This
                    removes the beam pattern, spreading and TVG residue, and
                    most of any constant port/starboard imbalance. A per-ping,
                    per-side balance then removes the part of the imbalance
                    that changes with roll.
    along track     rows resampled to fixed m_per_px_along metres of distance
                    travelled, from navigation, or speed * dt where navigation
                    is missing. A towfish that slows down no longer stretches
                    every object it passes.
    dropouts        time gaps, empty pings, attitude excursions and navigation
                    jumps are detected and every row carries a quality flag.
                    Short gaps are interpolated and say so; long ones are left
                    as zero rows and say so. Nothing is silently filled.
    speckle         Lee filter on the normalised intensity. Intensity only:
                    it moves no pixel.
    scale           robust percentiles to 1..255.

WHAT IS NOT CORRECTED, STATED RATHER THAN HIDDEN
    Layback: the sensor position recorded in the file is used as the towfish
    position. Pitch and yaw displacement of the footprint. Refraction.
    Layover of tall objects. A sloping seabed across the swath. These are in
    every sidecar's processing block under "geometry_not_corrected".

KNOBS
    Every threshold below is a module constant, overridable by environment
    variable HAZARD_INGEST_<NAME>, or per call through `options` using the
    lower-case name (options={"m_per_px_across": 0.1}). An unknown option key
    is an error, because a typo that silently does nothing is worse than a
    crash.
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("deepecho.hazard")

SONAR_SUFFIXES = (".xtf", ".jsf")
SIDECAR_FORMAT = "deepecho-strip-nav/1"
SIDECAR_SUFFIX = ".nav.json"
WATER_COLUMN_SUFFIX = ".wc.npz"


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(f"HAZARD_INGEST_{name}")
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"auto", "none"}:
        return None
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"HAZARD_INGEST_{name}")
    return default if raw is None or raw.strip() == "" else int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(f"HAZARD_INGEST_{name}")
    return default if raw is None or raw.strip() == "" else raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"HAZARD_INGEST_{name}")
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Resolution ------------------------------------------------------------

# Across-track ground resolution of the output strip, metres per pixel. None
# means "from the data": the median slant-range sample spacing, range / samples,
# which is the finest across-track detail the recording can support. Choosing
# finer than that interpolates; coarser discards.
M_PER_PX_ACROSS = _env_float("M_PER_PX_ACROSS", None)

# Along-track resolution, metres of distance travelled per row. None means
# "square pixels": the same as across-track, so a 2 m cylinder is the same
# number of pixels whichever way it lies, which is what a shape-trained
# detector needs.
M_PER_PX_ALONG = _env_float("M_PER_PX_ALONG", None)

# The automatic along-track choice never interpolates more than this many rows
# per ping spacing. Beyond it the image would be mostly invented between pings,
# so the rows are made coarser instead and the sidecar says why.
MAX_ALONG_UPSAMPLE = _env_float("MAX_ALONG_UPSAMPLE", 4.0)

# --- Dropouts --------------------------------------------------------------

# A gap between pings longer than this multiple of the median ping interval is
# a recording dropout, not a slow ping.
DROPOUT_GAP_FACTOR = _env_float("DROPOUT_GAP_FACTOR", 3.0)

# A ping whose mean raw energy is below this fraction of the line's median is
# empty: a dead channel, a blanked transmit, zero-filled data.
EMPTY_PING_ENERGY_FRAC = _env_float("EMPTY_PING_ENERGY_FRAC", 0.05)

# Attitude beyond these limits distorts the footprint enough that geometry and
# intensity are both suspect. Rows are kept and flagged "attitude", never
# dropped: the data is real, just degraded.
MAX_PITCH_DEG = _env_float("MAX_PITCH_DEG", 5.0)
MAX_ROLL_DEG = _env_float("MAX_ROLL_DEG", 8.0)
MAX_HEAVE_RATE = _env_float("MAX_HEAVE_RATE", 0.5)          # metres per second
# An excursion swings through zero between its peaks. Rows within this many
# seconds of a flagged ping are flagged too, so one excursion is one range.
ATTITUDE_HOLD_S = _env_float("ATTITUDE_HOLD_S", 0.5)

# A navigation fix implying a speed above this from the last accepted fix is a
# jump, not motion. About 15 knots: faster than any towfish survey.
MAX_SPEED_MPS = _env_float("MAX_SPEED_MPS", 7.5)
# After this many consecutive rejected fixes the rejector re-anchors on the
# newest fix, so a genuine step in navigation cannot reject the rest of a line.
NAV_REANCHOR_FIXES = _env_int("NAV_REANCHOR_FIXES", 10)
# Pings between accepted fixes further apart than this in time get no position.
NAV_MAX_INTERP_S = _env_float("NAV_MAX_INTERP_S", 10.0)
# Moving-average window over positions before along-track distance is summed.
# GPS jitter otherwise adds distance the towfish never travelled. 1 disables.
NAV_SMOOTH_PINGS = _env_int("NAV_SMOOTH_PINGS", 5)

# A run of rows inside a dropout no longer than this is interpolated from its
# neighbours and flagged "interpolated". Longer runs are zero rows, "dropout".
INTERP_MAX_ROWS = _env_int("INTERP_MAX_ROWS", 10)

# --- Lines -----------------------------------------------------------------

# A time gap longer than this ends one strip and starts another.
SPLIT_GAP_S = _env_float("SPLIT_GAP_S", 60.0)
# Relative change of slant range setting that starts a new strip.
SPLIT_RANGE_CHANGE = _env_float("SPLIT_RANGE_CHANGE", 0.01)
# A line with fewer usable pings than this is skipped and recorded.
MIN_LINE_PINGS = _env_int("MIN_LINE_PINGS", 20)

# --- Bottom tracking -------------------------------------------------------

BT_SMOOTH_SAMPLES = _env_int("BT_SMOOTH_SAMPLES", 5)
# Threshold between the ping's quiet level (5th percentile) and its bright
# level (95th): quiet + BT_THRESHOLD_FRAC * (bright - quiet).
BT_THRESHOLD_FRAC = _env_float("BT_THRESHOLD_FRAC", 0.3)
# The seabed keeps returning after its first echo; a fish, a school or a
# bubble does not. A candidate counts only when at least BT_PERSIST_FRAC of the
# BT_PERSIST_M of slant range after it stays above half the threshold margin.
# A fraction of samples, not a mean: one bright fish can lift a mean.
BT_PERSIST_M = _env_float("BT_PERSIST_M", 2.0)
BT_PERSIST_FRAC = _env_float("BT_PERSIST_FRAC", 0.7)
# The median filter along track is an outlier test, not a smoother: a
# detection within BT_OUTLIER_M of the filtered track is kept as measured,
# because real heave moves the seabed return from ping to ping and smoothing
# it away puts a bright nadir streak metres out into the ground-range image.
BT_MEDIAN_PINGS = _env_int("BT_MEDIAN_PINGS", 9)
BT_OUTLIER_M = _env_float("BT_OUTLIER_M", 0.3)
BT_MIN_ALTITUDE_M = _env_float("BT_MIN_ALTITUDE_M", 0.5)
# A header altitude is "plausible" when positive, inside the recorded range,
# and within this fraction of the tracked altitude nearby when there is one.
BT_HEADER_TOLERANCE_FRAC = _env_float("BT_HEADER_TOLERANCE_FRAC", 0.3)

# --- Gain ------------------------------------------------------------------

# The gain surface is a mean over (altitude band, ground-range bin), smoothed
# with a Gaussian of these widths by normalised convolution and interpolated
# linearly in altitude for each ping. Smooth in both directions because a mean
# over a few dozen speckled pings is itself speckled, and a speckled gain
# profile paints vertical stripes down every row that uses it.
EGN_ALTITUDE_BAND_M = _env_float("EGN_ALTITUDE_BAND_M", 0.5)
EGN_SMOOTH_ALTITUDE_M = _env_float("EGN_SMOOTH_ALTITUDE_M", 0.75)
EGN_SMOOTH_RANGE_M = _env_float("EGN_SMOOTH_RANGE_M", 0.3)
# Cells with less effective support than this many pings use the line-wide
# profile for that ground-range bin instead.
EGN_MIN_PINGS = _env_int("EGN_MIN_PINGS", 20)
# The mean is trimmed: after a first pass, pixels whose LOCAL MEAN ratio to the
# surface (EGN_TRIM_WINDOW pixels square) is below EGN_TRIM_LOW or above
# EGN_TRIM_HIGH are left out and the surface is recomputed. A wreck's highlight
# and shadow otherwise pull the gain for every ping flown at the same altitude.
# The test is on a local mean because single speckled pixels do not separate:
# a shadow pixel is often brighter than a seabed one, a shadow region never is.
# The same mask keeps shadows out of the per-ping roll balance.
EGN_TRIM_LOW = _env_float("EGN_TRIM_LOW", 0.33)
EGN_TRIM_HIGH = _env_float("EGN_TRIM_HIGH", 3.0)
EGN_TRIM_WINDOW = _env_int("EGN_TRIM_WINDOW", 5)
ROLL_BALANCE_PINGS = _env_int("ROLL_BALANCE_PINGS", 9)

# --- Speckle and scaling ---------------------------------------------------

SPECKLE = _env_str("SPECKLE", "lee")                          # "lee" or "none"
LEE_SIZE = _env_int("LEE_SIZE", 5)
CLIP_PERCENTILES = (_env_float("CLIP_LOW", 0.5), _env_float("CLIP_HIGH", 99.5))

# --- Positions -------------------------------------------------------------

# A file with no sensor position but a ship position: the ship is where the
# GPS was, not where the towfish was, and the difference (layback) can be tens
# of metres. Off by default, so such a file yields null positions and says so.
USE_SHIP_POSITION = _env_bool("USE_SHIP_POSITION", False)

# --- Water column ----------------------------------------------------------

WATER_COLUMN = _env_bool("WATER_COLUMN", True)
# Slant-range bin size for the water column. None = the native sample spacing.
WC_M_PER_BIN = _env_float("WC_M_PER_BIN", None)

# Hard guard against a corrupt header asking for an absurd image.
MAX_STRIP_WIDTH_PX = _env_int("MAX_STRIP_WIDTH_PX", 40000)

# --- Row quality -----------------------------------------------------------

QUALITY_OK = "ok"
QUALITY_ATTITUDE = "attitude"
QUALITY_NAV_JUMP = "nav_jump"
QUALITY_INTERPOLATED = "interpolated"
QUALITY_DROPOUT = "dropout"
# Worst last. "attitude" is real data from a disturbed vehicle; "nav_jump" is
# real data at a doubtful position; "interpolated" pixels were never measured;
# "dropout" rows hold nothing.
QUALITY_ORDER = (QUALITY_OK, QUALITY_ATTITUDE, QUALITY_NAV_JUMP, QUALITY_INTERPOLATED,
                 QUALITY_DROPOUT)

KNOTS_TO_MPS = 0.514444

_GEOMETRY_NOT_CORRECTED = [
    "layback: the recorded sensor position is used as the towfish position",
    "pitch and yaw displacement of the beam footprint",
    "sound-speed refraction",
    "layover: tall objects appear slightly nearer nadir than they are",
    "seabed slope across the swath: slant-range correction assumes a flat seabed",
]


def worst_quality(flags: Any) -> str | None:
    """The worst of an iterable of row quality flags, or None if there are none."""
    known = [f for f in flags if f in QUALITY_ORDER]
    return max(known, key=QUALITY_ORDER.index) if known else None


class SonarIngestError(ValueError):
    """A raw sonar file could not be turned into any strip."""


@dataclass
class IngestedStrip:
    strip: str
    image_path: Path
    nav_path: Path
    summary: dict[str, Any]


# --- options ---------------------------------------------------------------

_OPTION_NAMES = (
    "M_PER_PX_ACROSS", "M_PER_PX_ALONG", "MAX_ALONG_UPSAMPLE", "DROPOUT_GAP_FACTOR",
    "EMPTY_PING_ENERGY_FRAC", "MAX_PITCH_DEG", "MAX_ROLL_DEG", "MAX_HEAVE_RATE",
    "ATTITUDE_HOLD_S", "MAX_SPEED_MPS", "NAV_REANCHOR_FIXES", "NAV_MAX_INTERP_S",
    "NAV_SMOOTH_PINGS", "INTERP_MAX_ROWS", "SPLIT_GAP_S", "SPLIT_RANGE_CHANGE",
    "MIN_LINE_PINGS", "BT_SMOOTH_SAMPLES", "BT_THRESHOLD_FRAC", "BT_PERSIST_M", "BT_PERSIST_FRAC",
    "BT_MEDIAN_PINGS", "BT_OUTLIER_M", "BT_MIN_ALTITUDE_M", "BT_HEADER_TOLERANCE_FRAC",
    "EGN_ALTITUDE_BAND_M", "EGN_SMOOTH_ALTITUDE_M", "EGN_SMOOTH_RANGE_M", "EGN_MIN_PINGS",
    "EGN_TRIM_LOW", "EGN_TRIM_HIGH", "EGN_TRIM_WINDOW",
    "ROLL_BALANCE_PINGS",
    "SPECKLE", "LEE_SIZE", "CLIP_PERCENTILES", "USE_SHIP_POSITION", "WATER_COLUMN",
    "WC_M_PER_BIN", "MAX_STRIP_WIDTH_PX",
)


def _resolve_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Module constants, read at call time, with per-call overrides on top."""
    resolved = {name.lower(): globals()[name] for name in _OPTION_NAMES}
    for key, value in (options or {}).items():
        if key.lower() not in resolved:
            raise ValueError(f"unknown sonar_ingest option {key!r}; known: "
                             f"{', '.join(sorted(resolved))}")
        resolved[key.lower()] = value
    if resolved["speckle"] not in ("lee", "none"):
        raise ValueError(f"speckle must be 'lee' or 'none', not {resolved['speckle']!r}")
    return resolved


# --- speckle ---------------------------------------------------------------


def lee_filter(img: Any, size: int = LEE_SIZE, *, mask: Any = None) -> Any:
    """Lee speckle filter. Intensity only; no pixel moves.

        mean, var   local mean and variance over a size x size window
        noise       Cu^2 * mean^2, with Cu^2 the speckle's squared coefficient
                    of variation, estimated from the image itself as the
                    median of var / mean^2 over valid pixels. The median is
                    dominated by homogeneous seabed, where all variation is
                    speckle, which is exactly the quantity wanted.
        k           clip((var - noise) / var, 0, 1)
        out         mean + k * (img - mean)

    In flat speckle k is near 0 and a pixel becomes its local mean; on an
    edge, a highlight or a shadow boundary var greatly exceeds noise, k is near
    1 and the pixel is left alone. That is why this is used instead of a median:
    it smooths seabed without eroding the small bright returns and sharp shadow
    edges that identify man-made objects.

    `mask` marks valid pixels; invalid ones neither contribute to nor are
    changed by the filter. A 3-D array is filtered per channel. A uint8 input
    returns uint8, anything else float32.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    arr = np.asarray(img)
    if arr.ndim == 3:
        return np.stack([lee_filter(arr[..., c], size, mask=mask) for c in range(arr.shape[2])],
                        axis=-1)
    if arr.ndim != 2:
        raise ValueError("lee_filter needs a 2-D image or a 3-D image with channels last")
    size = int(size)
    out_uint8 = arr.dtype == np.uint8
    x = arr.astype(np.float64)
    valid = np.ones(x.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if size < 2 or not valid.any():
        return arr.copy() if out_uint8 else x.astype(np.float32)

    xv = np.where(valid, x, 0.0)
    weight = uniform_filter(valid.astype(np.float64), size, mode="reflect")
    covered = weight > 1e-6
    safe = np.where(covered, weight, 1.0)
    mean = np.where(covered, uniform_filter(xv, size, mode="reflect") / safe, 0.0)
    square = np.where(covered, uniform_filter(xv * xv, size, mode="reflect") / safe, 0.0)
    var = np.clip(square - mean * mean, 0.0, None)
    usable = valid & covered & (mean > 1e-12)
    if not usable.any():
        return arr.copy() if out_uint8 else x.astype(np.float32)
    cu2 = float(np.median(var[usable] / (mean[usable] ** 2)))
    noise = cu2 * mean * mean
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(var > 0, np.clip((var - noise) / var, 0.0, 1.0), 0.0)
    out = np.where(usable, mean + k * (x - mean), x)
    if out_uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out.astype(np.float32)


# --- raw readers -----------------------------------------------------------


@dataclass
class _RawFile:
    source_format: str
    meta: dict[str, Any]                      # per-ping numpy arrays, NaN = absent
    port: list[Any]                           # per-ping float32 samples, nearest first
    stbd: list[Any]
    port_r0: Any                              # slant range of sample 0, metres
    port_dr: Any                              # metres per sample
    stbd_r0: Any
    stbd_dr: Any
    port_range: Any                           # range setting, for line splitting
    stbd_range: Any
    coordinate_units: str
    synthetic: bool
    synthetic_basis: str | None
    reader: dict[str, Any] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def _epoch(year: int, month: int, day: int, hour: int, minute: int, second: int,
           microsecond: int = 0) -> float:
    try:
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second),
                        int(microsecond), tzinfo=timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return float("nan")


def read_xtf(path: Any) -> _RawFile:
    """Decode an XTF file's side-scan pings through pyxtf.

    Only sonar (type 0) and notes (type 1) packets are decoded. Navigation,
    attitude and raw serial packets are not used: positions and attitude come
    from each ping's own header, which is what every XTF writer populates.
    """
    import numpy as np

    try:
        import pyxtf
    except ImportError as exc:                                     # pragma: no cover
        raise SonarIngestError("reading XTF needs pyxtf (pip install pyxtf)") from exc

    path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            header, packets = pyxtf.xtf_read(
                str(path), types=[pyxtf.XTFHeaderType.sonar, pyxtf.XTFHeaderType.notes])
    except Exception as exc:
        raise SonarIngestError(f"{path.name}: not a readable XTF file ({type(exc).__name__}: "
                               f"{exc})") from exc

    pings = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not pings:
        raise SonarIngestError(f"{path.name}: XTF file contains no sonar pings")

    sonar_info = list(getattr(header, "sonar_info", []))
    kinds = [int(info.TypeOfChannel) for info in sonar_info]
    port_i = next((i for i, k in enumerate(kinds) if k == 1), None)
    stbd_i = next((i for i, k in enumerate(kinds) if k == 2), None)
    if port_i is None or stbd_i is None:
        raise SonarIngestError(
            f"{path.name}: XTF file has no port/starboard side-scan channel pair "
            f"(sonar channel types {kinds}; 1 = port, 2 = starboard)")

    nav_units = int(header.NavUnits)
    if nav_units == 3:
        coordinate_units = "degrees (XTF NavUnits=3, SensorX=longitude, SensorY=latitude)"
    elif nav_units == 0:
        coordinate_units = ("projected metres (XTF NavUnits=0); no projection is recorded in "
                            "a form this reader trusts, so latitude/longitude are null")
    else:
        coordinate_units = f"unknown (XTF NavUnits={nav_units}); latitude/longitude are null"

    texts = [header.NoteString.decode("latin-1", "replace")]
    for note in packets.get(pyxtf.XTFHeaderType.notes, []):
        texts.append(note.NotesText.decode("latin-1", "replace"))
    synthetic = any("SYNTHETIC" in text.upper() for text in texts)

    n = len(pings)
    meta = {key: np.full(n, np.nan) for key in
            ("time", "lat", "lon", "ship_lat", "ship_lon", "heading", "pitch", "roll", "heave",
             "altitude", "speed", "depth")}
    port, stbd = [], []
    geom = {key: np.full(n, np.nan) for key in
            ("port_r0", "port_dr", "stbd_r0", "stbd_dr", "port_range", "stbd_range")}
    unverified: list[str] = []
    weighted = False
    delayed = False
    skipped: list[dict[str, Any]] = []

    for j, ping in enumerate(pings):
        meta["time"][j] = _epoch(ping.Year, ping.Month, ping.Day, ping.Hour, ping.Minute,
                                 ping.Second, int(ping.HSeconds) * 10000)
        if nav_units == 3:
            meta["lon"][j], meta["lat"][j] = ping.SensorXcoordinate, ping.SensorYcoordinate
            meta["ship_lon"][j], meta["ship_lat"][j] = ping.ShipXcoordinate, ping.ShipYcoordinate
        meta["heading"][j] = ping.SensorHeading
        meta["pitch"][j] = ping.SensorPitch
        meta["roll"][j] = ping.SensorRoll
        meta["heave"][j] = ping.Heave
        meta["altitude"][j] = ping.SensorPrimaryAltitude
        meta["speed"][j] = ping.SensorSpeed * KNOTS_TO_MPS         # XTF: knots
        meta["depth"][j] = ping.SensorDepth
        velocity = float(ping.SoundVelocity)
        for side, index, store in (("port", port_i, port), ("stbd", stbd_i, stbd)):
            if ping.data is None or index >= len(ping.data):
                store.append(np.zeros(0, dtype=np.float32))
                skipped.append({"ping": j, "reason": f"{side} channel missing from ping"})
                continue
            chan = ping.ping_chan_headers[index]
            samples = np.asarray(ping.data[index], dtype=np.float32)
            if chan.Weight:
                samples = samples * np.float32(2.0 ** -float(chan.Weight))
                weighted = True
            count = len(samples)
            slant = float(chan.SlantRange)
            r0 = 0.0
            if chan.TimeDelay > 0:
                # XTF SoundVelocity is conventionally one-way (Isis writes 750);
                # a value above 1000 is taken as two-way. Unverified by design.
                one_way = velocity if 0 < velocity < 1000 else (velocity / 2 if velocity else 750.0)
                r0 = float(chan.TimeDelay) * one_way
                delayed = True
            if slant <= 0 and chan.TimeDuration > 0:
                one_way = velocity if 0 < velocity < 1000 else (velocity / 2 if velocity else 750.0)
                slant = r0 + float(chan.TimeDuration) * one_way
                unverified.append("slant range derived from TimeDuration * SoundVelocity "
                                  "because SlantRange was 0")
            store.append(samples)
            geom[f"{side}_r0"][j] = r0
            geom[f"{side}_dr"][j] = (slant - r0) / count if count and slant > r0 else np.nan
            geom[f"{side}_range"][j] = slant

    # Fields that are all exactly zero were never recorded, not measured zero.
    for key in ("heading", "pitch", "roll", "heave", "altitude", "speed", "depth"):
        if np.all(meta[key] == 0):
            meta[key][:] = np.nan
    for lat_key, lon_key in (("lat", "lon"), ("ship_lat", "ship_lon")):
        zero = (meta[lat_key] == 0) & (meta[lon_key] == 0)
        bad = zero | ~(np.abs(meta[lat_key]) <= 90) | ~(np.abs(meta[lon_key]) <= 180)
        meta[lat_key][bad] = np.nan
        meta[lon_key][bad] = np.nan
    meta["altitude"][meta["altitude"] <= 0] = np.nan

    reader: dict[str, Any] = {
        "format": "xtf", "library": f"pyxtf {_pyxtf_version()}",
        "packets_decoded": "sonar (0), notes (1); navigation, attitude and serial packets "
                           "are not used",
        "channels": {"port": port_i, "starboard": stbd_i, "sonar_channel_types": kinds},
        "sonar_name": header.SonarName.decode("latin-1", "replace").strip("\x00 "),
        "pings": n, "nav_units": nav_units,
        "speed_units": "SensorSpeed read as knots (XTF specification) and converted to m/s",
        "sample_order": "sample 0 is the earliest (nearest) sample on both channels",
    }
    if len(kinds) > 2:
        reader["channels_ignored"] = [i for i in range(len(kinds)) if i not in (port_i, stbd_i)]
    if weighted:
        unverified.append("XTF ping channel Weight applied as samples * 2^-Weight (pyxtf "
                          "convention); gain normalisation makes a constant weight irrelevant")
    if delayed:
        unverified.append("XTF TimeDelay converted to a start range with a one-way sound "
                          "velocity; SlantRange taken as the range of the last sample")

    if nav_units == 3 and not np.isfinite(meta["lat"]).any():
        if np.isfinite(meta["ship_lat"]).any():
            if USE_SHIP_POSITION:
                meta["lat"], meta["lon"] = meta["ship_lat"].copy(), meta["ship_lon"].copy()
                coordinate_units += "; SHIP position used for the towfish (no layback)"
                unverified.append("ship position used as towfish position; layback unknown")
            else:
                coordinate_units += ("; sensor position absent, ship position present but not "
                                     "used (layback unknown; set HAZARD_INGEST_USE_SHIP_POSITION=1 "
                                     "to accept it)")

    return _RawFile(
        source_format="xtf", meta=meta, port=port, stbd=stbd,
        port_r0=geom["port_r0"], port_dr=geom["port_dr"],
        stbd_r0=geom["stbd_r0"], stbd_dr=geom["stbd_dr"],
        port_range=geom["port_range"], stbd_range=geom["stbd_range"],
        coordinate_units=coordinate_units, synthetic=synthetic,
        synthetic_basis=("XTF NoteString or notes packet contains 'SYNTHETIC'"
                         if synthetic else None),
        reader=reader, unverified=sorted(set(unverified)), skipped=skipped)


def _pyxtf_version() -> str:
    try:
        from importlib.metadata import version
        return version("pyxtf")
    except Exception:                                             # pragma: no cover
        return "unknown"


# EdgeTech JSF, from the EdgeTech "JSF Data File Description", document 0004824.
# Every offset below is relative to the start of the 240-byte sonar data
# message header that follows the 16-byte message header. They are written
# from that specification but NOT validated against a real EdgeTech file in
# this repository, and every strip read this way says so under "unverified".
JSF_START_MARKER = 0x1601
JSF_SONAR_MESSAGE = 80
JSF_MESSAGE_HEADER = struct.Struct("<HBBHBBBBHi")          # 16 bytes
JSF_DATA_HEADER_SIZE = 240
_JSF_FIELDS = {
    # name: (offset, struct format, note)
    "ping_time_s": (0, "<i", "ping time, seconds since 1970-01-01 UTC"),
    "start_depth": (4, "<I", "starting depth (window offset) in samples"),
    "ping_number": (8, "<I", "ping number"),
    "data_format": (34, "<h", "0 = envelope int16, 1 = analytic real+imag int16 pairs"),
    "x": (80, "<i", "X coordinate or longitude"),
    "y": (84, "<i", "Y coordinate or latitude"),
    "coordinate_units": (88, "<h", "1 = X,Y mm; 2 = lat/lon in 0.0001 minutes of arc; "
                                   "3 = X,Y decimetres"),
    "annotation": (90, "24s", "annotation string"),
    "samples": (114, "<H", "number of data samples"),
    "sample_interval_ns": (116, "<I", "sampling interval in nanoseconds"),
    "depth_mm": (136, "<i", "towfish depth in millimetres"),
    "altitude_mm": (144, "<i", "altitude in millimetres"),
    "sound_speed": (148, "<f", "sound speed, m/s"),
    "year": (156, "<h", "year"),
    "day": (158, "<h", "day of year"),
    "hour": (160, "<h", "hour"),
    "minute": (162, "<h", "minute"),
    "second": (164, "<h", "second"),
    "weight": (168, "<h", "weighting factor N; data * 2^-N"),
    "heading": (172, "<H", "compass heading, 0.01 degree"),
    "pitch": (174, "<h", "pitch, scale 180/32768 degrees"),
    "roll": (176, "<h", "roll, scale 180/32768 degrees"),
    "nmea_speed": (194, "<h", "NMEA speed, 0.1 knot"),
    "ms_today": (200, "<I", "milliseconds today"),
}


def _jsf_field(buffer: bytes, base: int, name: str) -> Any:
    offset, fmt, _ = _JSF_FIELDS[name]
    return struct.unpack_from(fmt, buffer, base + offset)[0]


def read_jsf(path: Any) -> _RawFile:
    """A minimal EdgeTech JSF reader: sonar data messages (type 80) only.

    Message layout (spec 0004824): a 16-byte header -- start marker 0x1601,
    protocol version, session id, message type, command type, subsystem,
    channel (0 = port, 1 = starboard), sequence, reserved, byte count of the
    message that follows -- then for type 80 a 240-byte data header and the
    samples. Side-scan subsystems are 20 (low frequency) and 21 (high
    frequency); subsystem 0 is sub-bottom and is never used as side-scan.

    A file that does not start with the marker is refused with a clear error.
    A marker lost mid-file is re-synchronised by scanning forward, and the
    skipped bytes are recorded.
    """
    import numpy as np

    path = Path(path)
    data = path.read_bytes()
    if len(data) < 16:
        raise SonarIngestError(f"{path.name}: too short to be a JSF file")
    first = struct.unpack_from("<H", data, 0)[0]
    if first != JSF_START_MARKER:
        raise SonarIngestError(
            f"{path.name}: not an EdgeTech JSF file (first message marker 0x{first:04x}, "
            f"expected 0x{JSF_START_MARKER:04x})")

    messages: dict[int, dict[int, dict[int, tuple[int, int]]]] = {}
    skipped: list[dict[str, Any]] = []
    resync_bytes = 0
    other_types: dict[int, int] = {}
    pos = 0
    while pos + JSF_MESSAGE_HEADER.size <= len(data):
        (marker, _version, _session, mtype, _command, subsystem, channel, _seq, _res,
         size) = JSF_MESSAGE_HEADER.unpack_from(data, pos)
        if marker != JSF_START_MARKER:
            nxt = data.find(struct.pack("<H", JSF_START_MARKER), pos + 1)
            if nxt < 0:
                resync_bytes += len(data) - pos
                break
            resync_bytes += nxt - pos
            pos = nxt
            continue
        body = pos + JSF_MESSAGE_HEADER.size
        end = body + size
        if size < 0 or end > len(data):
            skipped.append({"offset": pos, "reason": "truncated message at end of file"})
            break
        if mtype == JSF_SONAR_MESSAGE:
            if size < JSF_DATA_HEADER_SIZE:
                skipped.append({"offset": pos, "reason": "sonar message shorter than its header"})
            else:
                ping_number = _jsf_field(data, body, "ping_number")
                messages.setdefault(subsystem, {}).setdefault(ping_number, {})[channel] = \
                    (body, size)
        else:
            other_types[mtype] = other_types.get(mtype, 0) + 1
        pos = end

    side_scan = {s: m for s, m in messages.items() if s != 0 and any(
        {0, 1} <= set(ch) for ch in m.values())}
    if not side_scan:
        raise SonarIngestError(
            f"{path.name}: JSF file has no side-scan sonar messages (type 80 with port and "
            f"starboard channels; subsystems found: {sorted(messages) or 'none'})")
    subsystem = 20 if 20 in side_scan else (21 if 21 in side_scan else min(side_scan))
    chosen = side_scan[subsystem]
    order = sorted(chosen)

    n = len(order)
    meta = {key: np.full(n, np.nan) for key in
            ("time", "lat", "lon", "heading", "pitch", "roll", "heave", "altitude", "speed",
             "depth")}
    port, stbd = [], []
    geom = {key: np.full(n, np.nan) for key in
            ("port_r0", "port_dr", "stbd_r0", "stbd_dr", "port_range", "stbd_range")}
    units_seen: set[int] = set()
    formats_seen: set[int] = set()
    synthetic = False

    for j, ping_number in enumerate(order):
        channels = chosen[ping_number]
        header_from = channels.get(0) or channels.get(1)
        base = header_from[0]
        seconds = _jsf_field(data, base, "ping_time_s")
        ms_today = _jsf_field(data, base, "ms_today")
        if seconds > 0:
            meta["time"][j] = seconds + (ms_today % 1000) / 1000.0
        else:
            year = _jsf_field(data, base, "year")
            day = _jsf_field(data, base, "day")
            try:
                start = datetime(int(year), 1, 1, tzinfo=timezone.utc).timestamp()
                meta["time"][j] = (start + (day - 1) * 86400 + _jsf_field(data, base, "hour") * 3600
                                   + _jsf_field(data, base, "minute") * 60
                                   + _jsf_field(data, base, "second"))
            except (ValueError, OverflowError):
                pass
        units = _jsf_field(data, base, "coordinate_units")
        units_seen.add(units)
        x, y = _jsf_field(data, base, "x"), _jsf_field(data, base, "y")
        if units == 2 and (x or y):
            meta["lon"][j] = x / 10000.0 / 60.0
            meta["lat"][j] = y / 10000.0 / 60.0
        meta["heading"][j] = _jsf_field(data, base, "heading") / 100.0
        meta["pitch"][j] = _jsf_field(data, base, "pitch") * 180.0 / 32768.0
        meta["roll"][j] = _jsf_field(data, base, "roll") * 180.0 / 32768.0
        altitude = _jsf_field(data, base, "altitude_mm")
        meta["altitude"][j] = altitude / 1000.0 if altitude > 0 else np.nan
        depth = _jsf_field(data, base, "depth_mm")
        meta["depth"][j] = depth / 1000.0 if depth > 0 else np.nan
        meta["speed"][j] = _jsf_field(data, base, "nmea_speed") / 10.0 * KNOTS_TO_MPS
        annotation = _jsf_field(data, base, "annotation").decode("latin-1", "replace")
        synthetic = synthetic or "SYNTHETIC" in annotation.upper()

        for side, channel, store in (("port", 0, port), ("stbd", 1, stbd)):
            if channel not in channels:
                store.append(np.zeros(0, dtype=np.float32))
                skipped.append({"ping": int(ping_number), "reason": f"{side} message missing"})
                continue
            body, size = channels[channel]
            fmt = _jsf_field(data, body, "data_format")
            formats_seen.add(fmt)
            count = _jsf_field(data, body, "samples")
            bytes_per = 4 if fmt == 1 else 2
            if fmt not in (0, 1, 2, 3, 4) or count * bytes_per > size - JSF_DATA_HEADER_SIZE:
                store.append(np.zeros(0, dtype=np.float32))
                skipped.append({"ping": int(ping_number), "reason": (
                    f"{side}: unsupported data format {fmt}" if fmt not in (0, 1, 2, 3, 4) else
                    f"{side}: {count} samples do not fit the message")})
                continue
            raw = data[body + JSF_DATA_HEADER_SIZE: body + JSF_DATA_HEADER_SIZE + count * bytes_per]
            if fmt == 1:
                pairs = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float32)
                samples = np.hypot(pairs[:, 0], pairs[:, 1])
            elif fmt in (0, 4):
                samples = np.frombuffer(raw, dtype="<u2").astype(np.float32)
            else:
                samples = np.abs(np.frombuffer(raw, dtype="<i2").astype(np.float32))
            weight = _jsf_field(data, body, "weight")
            if weight:
                samples = samples * np.float32(2.0 ** -float(weight))
            interval = _jsf_field(data, body, "sample_interval_ns") * 1e-9
            speed = _jsf_field(data, body, "sound_speed")
            sound = float(speed) if 1400.0 <= speed <= 1600.0 else 1500.0
            dr = interval * sound / 2.0
            r0 = _jsf_field(data, body, "start_depth") * dr
            store.append(samples)
            geom[f"{side}_dr"][j] = dr if dr > 0 else np.nan
            geom[f"{side}_r0"][j] = r0
            geom[f"{side}_range"][j] = r0 + count * dr

    if 2 in units_seen:
        coordinate_units = "minutes of arc x 10000 (JSF coordinate units 2), converted to degrees"
    else:
        coordinate_units = (f"JSF coordinate units {sorted(units_seen)} (projected or unknown); "
                            "latitude/longitude are null")
    if len(units_seen) > 1:
        coordinate_units += f"; mixed units within the file {sorted(units_seen)}"
    for key in ("heading", "pitch", "roll", "speed"):
        if np.all(meta[key] == 0):
            meta[key][:] = np.nan

    reader = {
        "format": "jsf", "library": "sonar_ingest minimal reader (EdgeTech spec 0004824)",
        "subsystem": subsystem, "subsystems_found": sorted(messages),
        "subsystems_ignored": sorted(s for s in messages if s != subsystem),
        "data_formats": sorted(formats_seen), "pings": n,
        "other_message_types": {str(k): v for k, v in sorted(other_types.items())},
        "resync_bytes_skipped": resync_bytes,
        "sample_order": "sample 0 is the earliest (nearest) sample on both channels",
    }
    unverified = [
        "JSF field offsets are taken from EdgeTech specification 0004824 and have not been "
        "validated against a real EdgeTech file in this repository",
        "JSF heave is not carried in message 80 and is null (attitude messages 2020/3000 "
        "are not decoded)",
        "JSF sound speed field used only when within 1400..1600 m/s, else 1500 m/s assumed",
        "JSF weighting factor applied as samples * 2^-N",
    ]
    return _RawFile(
        source_format="jsf", meta=meta, port=port, stbd=stbd,
        port_r0=geom["port_r0"], port_dr=geom["port_dr"],
        stbd_r0=geom["stbd_r0"], stbd_dr=geom["stbd_dr"],
        port_range=geom["port_range"], stbd_range=geom["stbd_range"],
        coordinate_units=coordinate_units, synthetic=synthetic,
        synthetic_basis=("JSF annotation contains 'SYNTHETIC'" if synthetic else None),
        reader=reader, unverified=unverified, skipped=skipped)


# --- helpers ---------------------------------------------------------------


def _fill_nan(x: Any) -> Any:
    import numpy as np

    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.all() or not finite.any():
        return x.copy()
    index = np.arange(len(x))
    return np.interp(index, index[finite], x[finite])


def _median_smooth(x: Any, size: int) -> Any:
    import numpy as np
    from scipy.ndimage import median_filter

    x = np.asarray(x, dtype=float)
    if size <= 1 or not np.isfinite(x).any():
        return x.copy()
    return median_filter(_fill_nan(x), size=int(size), mode="nearest")


def _nan_moving_average(x: Any, size: int) -> Any:
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    x = np.asarray(x, dtype=float)
    if size <= 1:
        return x.copy()
    finite = np.isfinite(x)
    total = uniform_filter1d(np.where(finite, x, 0.0), size, mode="nearest")
    weight = uniform_filter1d(finite.astype(float), size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / weight
    # At the ends the window shrinks symmetrically. Padding by repeating the
    # end value would pull the first and last positions inward and shift every
    # row of the strip along-track by a fraction of a ping.
    half = size // 2
    for i in list(range(min(half, len(x)))) + list(range(max(len(x) - half, half), len(x))):
        radius = min(i, len(x) - 1 - i)
        window = x[i - radius:i + radius + 1]
        good = np.isfinite(window)
        out[i] = window[good].mean() if good.any() else np.nan
    out[~finite] = np.nan
    return out


def _circular_interp(xq: Any, xp: Any, degrees: Any) -> Any:
    import numpy as np

    rad = np.radians(degrees)
    s = np.interp(xq, xp, np.sin(rad))
    c = np.interp(xq, xp, np.cos(rad))
    return np.degrees(np.arctan2(s, c)) % 360.0


def _enu_steps(lat: Any, lon: Any) -> tuple[Any, Any]:
    """East/north metres between consecutive positions, local flat-earth per step."""
    import numpy as np

    from survey_hazard_map.hazard_geo import _WGS84_A, _WGS84_E2

    phi = np.radians(lat[:-1])
    w = 1.0 - _WGS84_E2 * np.sin(phi) ** 2
    meridional = _WGS84_A * (1.0 - _WGS84_E2) / w ** 1.5
    prime = _WGS84_A / np.sqrt(w)
    north = np.radians(np.diff(lat)) * meridional
    east = np.radians(np.diff(lon)) * prime * np.cos(phi)
    return east, north


def _pad(channel: list[Any], index: Any) -> tuple[Any, Any]:
    import numpy as np

    counts = np.array([len(channel[i]) for i in index], dtype=int)
    width = max(int(counts.max()) if len(counts) else 0, 1)
    out = np.zeros((len(index), width), dtype=np.float32)
    for row, i in enumerate(index):
        out[row, :counts[row]] = channel[i]
    return out, counts


def _first_returns(arr: Any, counts: Any, r0: Any, dr: Any, opts: dict) -> Any:
    """Slant range of the first strong, sustained return per ping, NaN if none."""
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    n, width = arr.shape
    if n == 0:
        return np.zeros(0)
    valid = np.arange(width)[None, :] < counts[:, None]
    smooth = uniform_filter1d(arr, max(1, int(opts["bt_smooth_samples"])), axis=1,
                              mode="nearest")
    step = max(1, width // 1000)
    sampled = np.where(valid, smooth, np.nan)[:, ::step]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quiet = np.nanpercentile(sampled, 5, axis=1)
        bright = np.nanpercentile(sampled, 95, axis=1)
    frac = float(opts["bt_threshold_frac"])
    threshold = quiet + frac * (bright - quiet)
    sustain = quiet + 0.5 * frac * (bright - quiet)
    median_dr = float(np.nanmedian(dr)) if np.isfinite(dr).any() else 1.0
    persist = max(3, int(round(float(opts["bt_persist_m"]) / max(median_dr, 1e-6))))
    lifted = (valid & (smooth > sustain[:, None])).astype(np.int32)
    csum = np.concatenate([np.zeros((n, 1), dtype=np.int32), np.cumsum(lifted, axis=1)], axis=1)
    k = np.arange(width)
    end = np.minimum(k + persist, width)
    held = (csum[:, end] - csum[:, k]) / (end - k)[None, :]
    above = (valid & (smooth > threshold[:, None])
             & (held >= float(opts["bt_persist_frac"])) & (bright > quiet)[:, None])
    found = above.any(axis=1)
    first = np.argmax(above, axis=1)
    # Sub-sample edge: the half-height crossing of a 3-sample smooth between
    # the quiet level and the peak just after the candidate. For a step edge
    # the half-height point of a symmetric smooth sits on the step itself, so
    # this is unbiased where a fixed-threshold crossing is early. Near nadir a
    # tenth of a metre of altitude is a metre of ground range, so it matters.
    fine = uniform_filter1d(arr, 3, axis=1, mode="nearest")
    rows = np.arange(n)
    offsets = np.arange(-3, 6)
    window = np.clip(first[:, None] + offsets[None, :], 0, width - 1)
    values = fine[rows[:, None], window]
    peak = values[:, 3:].max(axis=1)
    level = quiet + 0.5 * (peak - quiet)
    over = values >= level[:, None]
    over[:, :1] = False
    j = np.argmax(over, axis=1)
    j = np.where(over.any(axis=1), j, 3)
    lo_v = values[rows, j - 1]
    hi_v = values[rows, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_pos = np.where(hi_v > lo_v, np.clip((level - lo_v) / (hi_v - lo_v), 0.0, 1.0), 1.0)
    position = np.clip(first + offsets[j - 1] + frac_pos, 0, width - 1)
    ranges = r0 + position * dr
    ranges = np.where(found & np.isfinite(ranges), ranges, np.nan)
    ranges[ranges < float(opts["bt_min_altitude_m"])] = np.nan
    return ranges


def _reverse_valid(arr: Any, counts: Any) -> Any:
    import numpy as np

    out = np.zeros_like(arr)
    for row, count in enumerate(counts):
        out[row, :count] = arr[row, :count][::-1]
    return out


# --- lines -----------------------------------------------------------------


def _split_lines(raw: _RawFile, opts: dict) -> list[tuple[Any, str]]:
    """Index arrays of contiguous lines, and why each one started."""
    import numpy as np

    n = len(raw.port)
    order = np.arange(n)
    time = raw.meta["time"]
    if np.isfinite(time).all():
        order = np.argsort(time, kind="stable")
    breaks: list[tuple[int, str]] = [(0, "start of file")]
    gap = float(opts["split_gap_s"])
    change = float(opts["split_range_change"])
    for pos in range(1, n):
        a, b = order[pos - 1], order[pos]
        dt = time[b] - time[a]
        if np.isfinite(dt) and dt > gap:
            breaks.append((pos, f"time gap of {dt:.1f} s > SPLIT_GAP_S={gap:g}"))
            continue
        for side in ("port_range", "stbd_range"):
            ra, rb = getattr(raw, side)[a], getattr(raw, side)[b]
            if np.isfinite(ra) and np.isfinite(rb) and abs(rb - ra) > change * max(ra, 1e-6):
                breaks.append((pos, f"{side.split('_')[0]} range setting changed "
                                    f"{ra:g} -> {rb:g} m"))
                break
    lines = []
    for i, (start, reason) in enumerate(breaks):
        stop = breaks[i + 1][0] if i + 1 < len(breaks) else n
        lines.append((order[start:stop], reason))
    return lines


def _ranges(mask: Any, reason: str) -> list[list[Any]]:
    out = []
    start = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append([start, i - 1, reason])
            start = None
    return out


def _iso(epoch: float) -> str | None:
    if not math.isfinite(epoch):
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds")


def _r(value: Any, digits: int) -> float | None:
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


# --- the line processor ----------------------------------------------------


def _process_line(raw: _RawFile, index: Any, strip: str, out_dir: Path, opts: dict,
                  line_info: dict[str, Any], source_name: str) -> IngestedStrip:
    import numpy as np
    from PIL import Image

    notes: list[str] = []
    unverified = list(raw.unverified)
    meta = {k: v[index] for k, v in raw.meta.items()}
    n = len(index)

    # --- time axis ---------------------------------------------------------
    time = meta["time"]
    time_known = bool(np.isfinite(time).any())
    if time_known and not np.isfinite(time).all():
        notes.append(f"{int((~np.isfinite(time)).sum())} ping(s) had no valid time; "
                     "interpolated by ping order")
        time = _fill_nan(time)
    clock = time if time_known else np.arange(n, dtype=float)
    dts = np.diff(clock)
    positive = dts[dts > 0]
    median_dt = float(np.median(positive)) if len(positive) else 1.0

    # --- samples -----------------------------------------------------------
    port, port_n = _pad(raw.port, index)
    stbd, stbd_n = _pad(raw.stbd, index)
    port_r0, port_dr = raw.port_r0[index], raw.port_dr[index]
    stbd_r0, stbd_dr = raw.stbd_r0[index], raw.stbd_dr[index]

    # --- empty pings -------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        energy = (port.sum(axis=1) + stbd.sum(axis=1)) / np.maximum(port_n + stbd_n, 1)
    has_samples = (port_n > 0) & (stbd_n > 0) & np.isfinite(port_dr) & np.isfinite(stbd_dr)
    reference = float(np.median(energy[has_samples & (energy > 0)])) \
        if (has_samples & (energy > 0)).any() else 0.0
    empty = ~has_samples | (energy <= float(opts["empty_ping_energy_frac"]) * reference)

    # --- bottom tracking ---------------------------------------------------
    port_track = _first_returns(port, port_n, port_r0, port_dr, opts)
    stbd_track = _first_returns(stbd, stbd_n, stbd_r0, stbd_dr, opts)
    port_track[empty], stbd_track[empty] = np.nan, np.nan

    # Both channels see the same seabed first return at nadir. If the port
    # channel's reversed samples agree with starboard far better than its
    # forward samples do, the file stores port far-to-near.
    port_order = "near-to-far (sample 0 nearest)"
    both = np.isfinite(stbd_track)
    if both.sum() >= 10:
        reversed_port = _reverse_valid(port, port_n)
        rev_track = _first_returns(reversed_port, port_n, port_r0, port_dr, opts)
        rev_track[empty] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fwd_err = np.nanmedian(np.abs(port_track - stbd_track))
            rev_err = np.nanmedian(np.abs(rev_track - stbd_track))
        if (np.isfinite(rev_err) and (not np.isfinite(fwd_err) or rev_err < 0.5 * fwd_err)
                and np.isfinite(rev_track).mean() > 0.5):
            port = reversed_port
            port_track = rev_track
            port_order = "far-to-near in the file; reversed on read"
            notes.append("port channel samples detected stored far-to-near and reversed")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prelim = np.nanmean(np.vstack([port_track, stbd_track]), axis=0)
    window = int(opts["bt_median_pings"])
    ref = _median_smooth(prelim, window)
    header_alt = meta["altitude"]
    disagree = (np.isfinite(port_track) & np.isfinite(stbd_track)
                & (np.abs(port_track - stbd_track) > 0.15 * np.maximum(ref, 1e-6)))
    # Both channels see the same nadir. When they disagree, the header decides
    # if it has an altitude; otherwise the later return wins, because anything
    # in the water column -- a school, a bubble cloud -- returns before the
    # seabed, and nothing returns before the seabed's own first echo from below.
    header_pick = np.where(np.abs(port_track - header_alt) <= np.abs(stbd_track - header_alt),
                           port_track, stbd_track)
    later = np.fmax(port_track, stbd_track)
    pick = np.where(np.isfinite(header_alt), header_pick, later)
    tracked = np.where(disagree, pick, prelim)
    detected = np.isfinite(tracked)
    smoothed = _median_smooth(tracked, window) if detected.any() else np.full(n, np.nan)

    max_slant = np.nanmax(np.vstack([raw.port_range[index], raw.stbd_range[index]]), axis=0)
    plausible = (np.isfinite(header_alt) & (header_alt > float(opts["bt_min_altitude_m"]))
                 & (header_alt < max_slant))
    near = np.isfinite(smoothed)
    plausible &= ~near | (np.abs(header_alt - smoothed)
                          <= float(opts["bt_header_tolerance_frac"]) * np.maximum(smoothed, 1e-6))

    altitude = np.full(n, np.nan)
    source = np.full(n, "none", dtype=object)
    outlier = detected & (np.abs(tracked - smoothed) > float(opts["bt_outlier_m"]))
    altitude[detected] = np.where(outlier[detected], smoothed[detected], tracked[detected])
    source[detected] = "bottom_track"
    outliers_replaced = int(outlier.sum())
    # A tracked altitude that disagrees badly with a plausible header altitude
    # has almost certainly locked onto something in the water column.
    header_ok = (np.isfinite(header_alt) & (header_alt > float(opts["bt_min_altitude_m"]))
                 & (header_alt < max_slant))
    overruled = detected & header_ok & (
        np.abs(altitude - header_alt)
        > float(opts["bt_header_tolerance_frac"]) * np.maximum(header_alt, 1e-6))
    altitude[overruled] = header_alt[overruled]
    source[overruled] = "header_altitude_track_rejected"
    detected = detected & ~overruled
    use_header = ~detected & plausible & (source != "header_altitude_track_rejected")
    altitude[use_header] = header_alt[use_header]
    source[use_header] = "header_altitude"
    use_interp = ~detected & ~use_header & near & (source != "header_altitude_track_rejected")
    altitude[use_interp] = smoothed[use_interp]
    source[use_interp] = "bottom_track_interpolated"
    no_altitude = ~np.isfinite(altitude) & ~empty
    source[empty] = "empty_ping"
    source_counts = {str(k): int((source == k).sum()) for k in
                     ("bottom_track", "header_altitude", "header_altitude_track_rejected",
                      "bottom_track_interpolated", "empty_ping", "none")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        both_known = detected & np.isfinite(header_alt)
        agreement = float(np.median(np.abs(smoothed[both_known] - header_alt[both_known]))) \
            if both_known.any() else None

    usable = ~empty & np.isfinite(altitude)
    u = np.nonzero(usable)[0]
    if len(u) < int(opts["min_line_pings"]):
        raise SonarIngestError(
            f"only {len(u)} usable ping(s) (empty={int(empty.sum())}, "
            f"no altitude={int(no_altitude.sum())}); MIN_LINE_PINGS={opts['min_line_pings']}")

    # --- navigation --------------------------------------------------------
    lat, lon = meta["lat"].copy(), meta["lon"].copy()
    has_pos = np.isfinite(lat) & np.isfinite(lon)
    rejected = np.zeros(n, dtype=bool)
    accepted = np.zeros(n, dtype=bool)
    repeated = 0
    last = None
    run = 0
    max_speed = float(opts["max_speed_mps"])
    from survey_hazard_map.hazard_geo import enu_delta
    for i in np.nonzero(has_pos)[0]:
        if last is not None and lat[i] == lat[last] and lon[i] == lon[last]:
            repeated += 1                       # the same fix again, not a new one
            continue
        if last is None:
            accepted[i], last = True, i
            continue
        east, north = enu_delta(lat[last], lon[last], lat[i], lon[i])
        dt = clock[i] - clock[last]
        distance = math.hypot(east, north)
        implied = distance / dt if time_known and dt > 0 else (0.0 if distance < 1e-3 else math.inf)
        if not time_known or implied <= max_speed:
            accepted[i], last, run = True, i, 0
        else:
            run += 1
            rejected[i] = True
            if run >= int(opts["nav_reanchor_fixes"]):
                rejected[i], accepted[i], last, run = False, True, i, 0
    if repeated:
        notes.append(f"{repeated} repeated navigation fix(es) treated as no new fix and "
                     "interpolated in time between distinct fixes")

    lat_p = np.full(n, np.nan)
    lon_p = np.full(n, np.nan)
    acc = np.nonzero(accepted)[0]
    if len(acc) >= 1:
        if len(acc) == 1:
            lat_p[acc], lon_p[acc] = lat[acc], lon[acc]
        else:
            t_acc = clock[acc]
            inside = (clock >= t_acc[0]) & (clock <= t_acc[-1])
            lat_p[inside] = np.interp(clock[inside], t_acc, lat[acc])
            lon_p[inside] = np.interp(clock[inside], t_acc, lon[acc])
            # No position across a long navigation outage.
            nxt = np.searchsorted(t_acc, clock, side="left")
            prv = np.clip(nxt - 1, 0, len(acc) - 1)
            nxt = np.clip(nxt, 0, len(acc) - 1)
            span = t_acc[nxt] - t_acc[prv]
            lat_p[span > float(opts["nav_max_interp_s"])] = np.nan
            lon_p[span > float(opts["nav_max_interp_s"])] = np.nan
        smooth_n = int(opts["nav_smooth_pings"])
        lat_p = _nan_moving_average(lat_p, smooth_n)
        lon_p = _nan_moving_average(lon_p, smooth_n)
    position_ok = np.isfinite(lat_p) & np.isfinite(lon_p)

    steps = np.full(max(n - 1, 0), np.nan)
    distance_source = {"navigation": 0, "speed_x_dt": 0, "median_step": 0}
    if n > 1 and position_ok.sum() >= 2:
        east, north = _enu_steps(np.where(position_ok, lat_p, 0.0), np.where(position_ok, lon_p, 0.0))
        ok = position_ok[:-1] & position_ok[1:]
        steps[ok] = np.hypot(east, north)[ok]
        distance_source["navigation"] = int(ok.sum())
    speed = meta["speed"]
    speed_source = "header (SensorSpeed)" if np.isfinite(speed).any() else None
    if n > 1:
        missing = ~np.isfinite(steps)
        mean_speed = (speed[:-1] + speed[1:]) / 2.0
        fill = missing & np.isfinite(mean_speed) & (dts > 0) & time_known
        steps[fill] = (mean_speed * dts)[fill]
        distance_source["speed_x_dt"] = int(fill.sum())
    along_known = bool(np.isfinite(steps).any()) if n > 1 else False
    if along_known:
        still = ~np.isfinite(steps)
        if still.any():
            steps[still] = float(np.nanmedian(steps))
            distance_source["median_step"] = int(still.sum())
            notes.append(f"{int(still.sum())} ping interval(s) had neither navigation nor "
                         "speed; the median step was used for their along-track distance")
        distance = np.concatenate([[0.0], np.cumsum(steps)])
    else:
        # No scale at all: the along-track axis is counted in ping intervals,
        # with time gaps still opening the gaps they represent.
        units = np.where(dts > 0, np.maximum(np.rint(dts / median_dt), 1.0), 1.0) \
            if n > 1 else np.zeros(0)
        distance = np.concatenate([[0.0], np.cumsum(units)])
        notes.append("no navigation and no speed: along-track resolution is unknown and rows "
                     "are spaced one per nominal ping interval")

    if speed_source is None and along_known and time_known and n > 1:
        derived = np.concatenate([[np.nan], steps / np.where(dts > 0, dts, np.nan)])
        speed = _nan_moving_average(_fill_nan(derived), max(3, int(round(2.0 / median_dt))))
        speed_source = "derived from along-track distance / time"

    heading = meta["heading"].copy()
    heading_source = "header (SensorHeading)" if np.isfinite(heading).any() else None
    if heading_source is None and position_ok.sum() >= 3:
        east, north = _enu_steps(_fill_nan(lat_p), _fill_nan(lon_p))
        course = np.degrees(np.arctan2(east, north)) % 360.0
        heading = np.concatenate([course, course[-1:]])
        heading[~position_ok] = np.nan
        heading_source = "course over ground from navigation (no heading recorded)"
        notes.append("heading absent from the file; course over ground used for the "
                     "across-track bearing")

    # --- flags per ping ----------------------------------------------------
    gap_after = np.zeros(n, dtype=bool)
    if n > 1 and time_known:
        gap_after[:-1] = dts > float(opts["dropout_gap_factor"]) * median_dt
    attitude = np.zeros(n, dtype=bool)
    for key, limit in (("pitch", "max_pitch_deg"), ("roll", "max_roll_deg")):
        values = meta[key]
        attitude |= np.isfinite(values) & (np.abs(values) > float(opts[limit]))
    if np.isfinite(meta["heave"]).sum() > 2 and time_known:
        rate = np.gradient(_fill_nan(meta["heave"]), clock)
        attitude |= np.isfinite(rate) & (np.abs(rate) > float(opts["max_heave_rate"]))
    hold = float(opts["attitude_hold_s"])
    if attitude.any() and hold > 0 and time_known:
        flagged_t = clock[attitude]
        pos_idx = np.searchsorted(flagged_t, clock)
        before = np.abs(clock - flagged_t[np.clip(pos_idx - 1, 0, len(flagged_t) - 1)])
        after = np.abs(flagged_t[np.clip(pos_idx, 0, len(flagged_t) - 1)] - clock)
        attitude |= np.minimum(before, after) <= hold

    # --- slant-range correction -------------------------------------------
    alt_u = altitude[u]
    dr_all = np.concatenate([port_dr[u], stbd_dr[u]])
    native_dx = float(np.nanmedian(dr_all))
    if opts["m_per_px_across"] is not None:
        dx = float(opts["m_per_px_across"])
        dx_basis = "option / HAZARD_INGEST_M_PER_PX_ACROSS"
    else:
        dx = native_dx
        dx_basis = "auto: median slant-range sample spacing (range / samples)"
    far = np.maximum(port_r0[u] + (port_n[u] - 1) * port_dr[u],
                     stbd_r0[u] + (stbd_n[u] - 1) * stbd_dr[u])
    ground_max = float(np.sqrt(np.clip(np.nanmax(far) ** 2 - np.nanmin(alt_u) ** 2, 0, None)))
    columns = int(math.ceil(ground_max / dx))
    if 2 * columns > int(opts["max_strip_width_px"]):
        raise SonarIngestError(
            f"strip would be {2 * columns} px wide (ground range {ground_max:.1f} m at "
            f"{dx:g} m/px), above MAX_STRIP_WIDTH_PX={opts['max_strip_width_px']}")
    ground = (np.arange(columns) + 0.5) * dx

    def to_ground(side: Any, counts: Any, r0: Any, dr: Any) -> Any:
        a = alt_u[:, None]
        slant = np.sqrt(ground[None, :] ** 2 + a * a)
        fidx = (slant - r0[u][:, None]) / dr[u][:, None]
        valid = (fidx >= 0) & (fidx <= (counts[u] - 1)[:, None])
        width = side.shape[1]
        i0 = np.clip(np.floor(fidx), 0, width - 1).astype(np.int64)
        i1 = np.clip(i0 + 1, 0, width - 1)
        w = np.clip(fidx - i0, 0.0, 1.0).astype(np.float32)
        rows = side[u]
        values = (np.take_along_axis(rows, i0, axis=1) * (1 - w)
                  + np.take_along_axis(rows, i1, axis=1) * w)
        return np.where(valid, values, np.nan).astype(np.float32)

    port_g = to_ground(port, port_n, port_r0, port_dr)
    stbd_g = to_ground(stbd, stbd_n, stbd_r0, stbd_dr)

    # --- gain normalisation ------------------------------------------------
    band_m = max(float(opts["egn_altitude_band_m"]), 1e-3)
    alt_min = float(np.nanmin(alt_u))
    band_of = np.floor((alt_u - alt_min) / band_m).astype(int)
    n_bands = int(band_of.max()) + 1
    min_pings = float(opts["egn_min_pings"])

    def egn_surface(values: Any, keep: Any) -> tuple[Any, int]:
        from scipy.ndimage import gaussian_filter, gaussian_filter1d

        sums = np.zeros((n_bands, values.shape[1]))
        counts = np.zeros((n_bands, values.shape[1]))
        np.add.at(sums, band_of, np.where(keep, values, 0.0))
        np.add.at(counts, band_of, keep.astype(float))
        sigma = (float(opts["egn_smooth_altitude_m"]) / band_m,
                 float(opts["egn_smooth_range_m"]) / dx)
        sums_s = gaussian_filter(sums, sigma, mode="nearest")
        counts_s = gaussian_filter(counts, sigma, mode="nearest")
        line_sum = gaussian_filter1d(sums.sum(axis=0), sigma[1], mode="nearest")
        line_count = gaussian_filter1d(counts.sum(axis=0), sigma[1], mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            surface = sums_s / counts_s
            line = np.where(line_count > 0, line_sum / line_count, np.nan)
        # Effective support of a smoothed cell, in pings: what the kernel
        # gathered, rescaled by the kernel's own peak weight.
        peak = gaussian_filter(np.pad(np.ones((1, 1)), ((20, 20), (20, 20))), sigma,
                               mode="constant").max()
        thin = counts_s / max(peak, 1e-12) < min_pings
        surface = np.where(thin, line[None, :], surface)
        surface[~(surface > 0)] = np.nan
        # Per ping, linear between the two nearest band centres.
        t = np.clip((alt_u - alt_min) / band_m - 0.5, 0.0, n_bands - 1.0)
        b0 = np.floor(t).astype(int)
        b1 = np.minimum(b0 + 1, n_bands - 1)
        wb = (t - b0)[:, None]
        return surface[b0] * (1 - wb) + surface[b1] * wb, int(thin.sum())

    def seabed_mask(ratio: Any) -> Any:
        """Pixels whose local-mean ratio is inside the trim band."""
        from scipy.ndimage import uniform_filter

        finite = np.isfinite(ratio)
        size = max(1, int(opts["egn_trim_window"]))
        total = uniform_filter(np.where(finite, ratio, 0.0), size, mode="nearest")
        weight = uniform_filter(finite.astype(float), size, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            local = total / weight
        return finite & (local >= float(opts["egn_trim_low"])) & \
            (local <= float(opts["egn_trim_high"]))

    def egn(values: Any) -> tuple[Any, dict, Any]:
        finite = np.isfinite(values)
        profile, _ = egn_surface(values, finite)
        with np.errstate(invalid="ignore", divide="ignore"):
            keep = seabed_mask(values / profile)
        profile, thin = egn_surface(values, keep)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (values / profile).astype(np.float32)
        keep = seabed_mask(out)
        used = {"altitude_bands": n_bands, "cells_using_line_profile": thin,
                "cells": int(n_bands * values.shape[1]),
                "pixels_trimmed": int((finite & ~keep).sum())}
        return out, used, keep

    port_e, port_used, port_keep = egn(port_g)
    stbd_e, stbd_used, stbd_keep = egn(stbd_g)

    def balance(values: Any, keep: Any) -> tuple[Any, Any]:
        # Median over the ping's seabed only (the EGN trim mask): a wreck's
        # shadow covering half a swath would otherwise read as a gain drop and
        # brighten the rest of that ping.
        trimmed = np.where(keep, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_ping = np.nanmedian(trimmed, axis=1)
        smooth = _median_smooth(per_ping, int(opts["roll_balance_pings"]))
        level = float(np.nanmedian(smooth)) if np.isfinite(smooth).any() else 1.0
        factor = smooth / level if level > 0 else np.ones_like(smooth)
        factor = np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
        return values / factor[:, None], factor

    port_e, port_factor = balance(port_e, port_keep)
    stbd_e, stbd_factor = balance(stbd_e, stbd_keep)
    ping_image = np.concatenate([port_e[:, ::-1], stbd_e], axis=1)   # port left, nadir at `columns`

    # --- water column -----------------------------------------------------
    wc = None
    if opts["water_column"]:
        wc_dr = float(opts["wc_m_per_bin"]) if opts["wc_m_per_bin"] is not None else native_dx
        max_bottom = float(np.nanmax(alt_u))
        bins = max(1, int(math.ceil(max_bottom / wc_dr)))
        centres = (np.arange(bins) + 0.5) * wc_dr

        def column(side: Any, counts: Any, r0: Any, dr: Any, factor: Any) -> Any:
            fidx = (centres[None, :] - r0[u][:, None]) / dr[u][:, None]
            valid = ((fidx >= 0) & (fidx <= (counts[u] - 1)[:, None])
                     & (centres[None, :] < alt_u[:, None]))
            width = side.shape[1]
            i0 = np.clip(np.floor(fidx), 0, width - 1).astype(np.int64)
            i1 = np.clip(i0 + 1, 0, width - 1)
            w = np.clip(fidx - i0, 0.0, 1.0).astype(np.float32)
            rows = side[u]
            values = (np.take_along_axis(rows, i0, axis=1) * (1 - w)
                      + np.take_along_axis(rows, i1, axis=1) * w) / factor[:, None]
            values = np.where(valid, values, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                profile = np.nanmedian(values, axis=0)
            profile[~(profile > 0)] = np.nan
            return (values / profile[None, :]).astype(np.float32)

        wc = {"port": column(port, port_n, port_r0, port_dr, port_factor),
              "stbd": column(stbd, stbd_n, stbd_r0, stbd_dr, stbd_factor),
              "m_per_bin": wc_dr, "centres": centres, "max_range": max_bottom}

    # --- along-track resampling -------------------------------------------
    d_u = distance[u]
    spacing = np.diff(d_u)
    median_spacing = float(np.median(spacing[spacing > 0])) if (spacing > 0).any() else None
    if along_known:
        if opts["m_per_px_along"] is not None:
            dy = float(opts["m_per_px_along"])
            dy_basis = "option / HAZARD_INGEST_M_PER_PX_ALONG"
        else:
            dy = dx
            dy_basis = "auto: equal to across-track (square pixels)"
            upsample = float(opts["max_along_upsample"])
            if median_spacing and upsample > 0 and median_spacing / dy > upsample:
                dy = median_spacing / upsample
                dy_basis = (f"auto: square pixels would interpolate more than "
                            f"MAX_ALONG_UPSAMPLE={upsample:g} rows per ping, so median ping "
                            f"spacing / {upsample:g}")
    else:
        dy = 1.0
        dy_basis = "unknown: one row per nominal ping interval"
    height = int(math.floor((d_u[-1] - d_u[0]) / dy)) + 1
    y = d_u[0] + np.arange(height) * dy
    if len(u) > 1:
        lo = np.clip(np.searchsorted(d_u, y, side="right") - 1, 0, len(u) - 2)
        hi = lo + 1
        span = d_u[hi] - d_u[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(span > 0, (y - d_u[lo]) / span, 0.0)
        w = np.clip(w, 0.0, 1.0)
    else:
        lo = hi = np.zeros(height, dtype=int)
        w = np.zeros(height)

    between = w > 0
    gap_rows = np.bincount(lo[between], minlength=max(len(u) - 1, 1))
    degraded_gap = np.zeros(max(len(u) - 1, 1), dtype=bool)
    gap_reason = np.full(max(len(u) - 1, 1), "", dtype=object)
    if len(u) > 1:
        skipped_between = np.diff(u) > 1
        timed = np.array([(clock[u[j + 1]] - clock[u[j]])
                          > float(opts["dropout_gap_factor"]) * median_dt
                          for j in range(len(u) - 1)]) if time_known else np.zeros(len(u) - 1, bool)
        sparse = gap_rows[:len(u) - 1] > int(opts["interp_max_rows"])
        degraded_gap[:len(u) - 1] = skipped_between | timed | sparse
        gap_reason[:len(u) - 1] = np.where(timed, "time_gap",
                                           np.where(skipped_between, "empty_pings",
                                                    np.where(sparse, "sparse_pings", "")))

    quality = np.full(height, QUALITY_OK, dtype=object)
    in_bad_gap = between & degraded_gap[lo]
    long_gap = gap_rows[lo] > int(opts["interp_max_rows"])
    quality[in_bad_gap & ~long_gap] = QUALITY_INTERPOLATED
    quality[in_bad_gap & long_gap] = QUALITY_DROPOUT
    nearest_u = np.where(w < 0.5, lo, hi)
    att_rows = attitude[u][nearest_u] & (quality == QUALITY_OK)
    quality[att_rows] = QUALITY_ATTITUDE
    nearest_all = np.clip(np.searchsorted(distance, y), 0, n - 1)
    prev_all = np.clip(nearest_all - 1, 0, n - 1)
    pick_prev = np.abs(distance[prev_all] - y) < np.abs(distance[nearest_all] - y)
    nearest_all = np.where(pick_prev, prev_all, nearest_all)
    nav_rows = rejected[nearest_all] & np.isin(quality, [QUALITY_OK, QUALITY_ATTITUDE])
    quality[nav_rows] = QUALITY_NAV_JUMP
    dropout_rows = quality == QUALITY_DROPOUT

    def resample(values: Any) -> Any:
        out = np.empty((height, values.shape[1]), dtype=np.float32)
        chunk = 2048
        for start in range(0, height, chunk):
            sl = slice(start, min(start + chunk, height))
            a, b = values[lo[sl]], values[hi[sl]]
            ww = w[sl][:, None].astype(np.float32)
            mixed = a * (1 - ww) + b * ww
            mixed = np.where(np.isnan(a) & ~np.isnan(b), b, mixed)
            mixed = np.where(np.isnan(b) & ~np.isnan(a), a, mixed)
            out[sl] = mixed
        out[dropout_rows] = np.nan
        return out

    image = resample(ping_image)
    valid = np.isfinite(image)

    # --- speckle and scaling ----------------------------------------------
    noise_cv2 = None
    if opts["speckle"] == "lee" and valid.any():
        filled = np.where(valid, image, 0.0)
        image = lee_filter(filled, int(opts["lee_size"]), mask=valid)
        sample = filled[valid]
        noise_cv2 = "estimated inside lee_filter from var/mean^2 of the image"
    low_p, high_p = opts["clip_percentiles"]
    values = image[valid]
    if values.size:
        step = max(1, values.size // 2_000_000)
        low, high = np.percentile(values[::step], [float(low_p), float(high_p)])
    else:
        low, high = 0.0, 1.0
    scale = 254.0 / (high - low) if high > low else 0.0
    grey = np.where(valid, np.clip(1.0 + (image - low) * scale, 1, 255), 0).astype(np.uint8)

    # --- per-row navigation -----------------------------------------------
    def row_interp(values: Any, where: Any = None) -> Any:
        finite = np.isfinite(values) if where is None else (np.isfinite(values) & where)
        out = np.full(height, np.nan)
        if not finite.any():
            return out
        xp, fp = distance[finite], values[finite]
        inside = (y >= xp[0]) & (y <= xp[-1])
        out[inside] = np.interp(y[inside], xp, fp)
        return out

    row_time = row_interp(clock) if time_known else np.full(height, np.nan)
    row_lat = row_interp(lat_p, position_ok)
    row_lon = row_interp(lon_p, position_ok)
    row_lat[quality == QUALITY_NAV_JUMP] = np.nan
    row_lon[quality == QUALITY_NAV_JUMP] = np.nan
    if np.isfinite(heading).any():
        hf = np.isfinite(heading)
        row_heading = _circular_interp(y, distance[hf], heading[hf])
        row_heading[(y < distance[hf][0]) | (y > distance[hf][-1])] = np.nan
    else:
        row_heading = np.full(height, np.nan)
    row_alt = np.interp(y, d_u, alt_u)
    row_speed = row_interp(speed)
    row_pitch = row_interp(meta["pitch"])
    row_roll = row_interp(meta["roll"])
    row_heave = row_interp(meta["heave"])
    row_depth = row_interp(meta["depth"])
    for series in (row_alt, row_pitch, row_roll, row_heave, row_depth):
        series[dropout_rows] = np.nan

    rows = []
    for k in range(height):
        depth_k = _r(row_depth[k], 3)
        alt_k = _r(row_alt[k], 3)
        rows.append({
            "row": k,
            "time": _iso(row_time[k]) if time_known else None,
            "lat": _r(row_lat[k], 8),
            "lon": _r(row_lon[k], 8),
            "heading_deg": _r(row_heading[k], 3),
            "altitude_m": alt_k,
            "speed_mps": _r(row_speed[k], 3),
            "pitch_deg": _r(row_pitch[k], 3),
            "roll_deg": _r(row_roll[k], 3),
            "heave_m": _r(row_heave[k], 3),
            "depth_m": depth_k,
            "seabed_depth_m": (round(depth_k + alt_k, 3)
                               if depth_k is not None and alt_k is not None else None),
            "quality": str(quality[k]),
        })

    degraded: list[list[Any]] = []
    for flag in (QUALITY_ATTITUDE, QUALITY_NAV_JUMP, QUALITY_INTERPOLATED, QUALITY_DROPOUT):
        degraded.extend(_ranges(quality == flag, flag))
    degraded.sort(key=lambda r: r[0])

    # --- write ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{strip}.png"
    nav_path = out_dir / f"{strip}{SIDECAR_SUFFIX}"
    Image.fromarray(grey, mode="L").save(image_path, "PNG")

    water_column_block = None
    if wc is not None:
        wc_path = out_dir / f"{strip}{WATER_COLUMN_SUFFIX}"
        wc_port = resample(wc["port"])
        wc_stbd = resample(wc["stbd"])
        bottom = row_alt.astype(np.float32)
        beyond = wc["centres"][None, :] >= np.where(np.isfinite(bottom), bottom, -1.0)[:, None]
        wc_port[beyond] = np.nan
        wc_stbd[beyond] = np.nan
        np.savez_compressed(wc_path, port=wc_port, starboard=wc_stbd, bottom_range_m=bottom,
                            m_per_bin=np.float32(wc["m_per_bin"]))
        water_column_block = {
            "path": wc_path.name,
            "m_per_bin": round(float(wc["m_per_bin"]), 6),
            "max_range_m": round(float(wc["max_range"]), 3),
            "bins": int(len(wc["centres"])),
            "rows": height,
            "arrays": {"port": "rows x bins float32", "starboard": "rows x bins float32",
                       "bottom_range_m": "rows float32, slant range of the first bottom "
                                         "return used for the row, NaN if unknown"},
            "note": "slant-range bins from the transducer to the first bottom return; "
                    "intensity after gain normalisation; not speckle-filtered",
            "normalisation": "same per-ping, per-side roll balance as the image, then each "
                             "slant bin divided by its median over the line (median, not "
                             "mean, so fish schools do not raise their own reference)",
            "bin_order": "bin 0 at the transducer on both channels (port is NOT mirrored)",
            "row_alignment": "row i is row i of the PNG and of the sidecar rows; dropout "
                             "rows and bins beyond the row's bottom range are NaN",
        }

    rows_by_quality = {q: int((quality == q).sum()) for q in QUALITY_ORDER}
    processing = {
        "reader": {**raw.reader, "port_sample_order": port_order},
        "line": line_info,
        "pixel_convention": "continuous pixel x spans column c as [c, c+1); ground offset "
                            "from nadir = (x - nadir_col) * m_per_px_across, positive to "
                            "starboard. Row k's navigation describes continuous y = k + 0.5.",
        "time": {"known": time_known, "median_ping_interval_s": round(median_dt, 6)},
        "bottom_track": {
            "method": "first sample above quiet + BT_THRESHOLD_FRAC * (bright - quiet) that "
                      "keeps BT_PERSIST_FRAC of the next BT_PERSIST_M above half that margin, "
                      "per side, refined to the half-height crossing; sides combined (header, "
                      "else the later return, decides a disagreement); detections further than BT_OUTLIER_M from a median filter "
                      "along track replaced by the filtered value; header altitude where the "
                      "tracker found nothing, or disagreed with a plausible header by more "
                      "than BT_HEADER_TOLERANCE_FRAC",
            "source_counts": source_counts,
            "outliers_replaced_by_median": outliers_replaced,
            "median_abs_difference_from_header_m": (round(agreement, 3)
                                                    if agreement is not None else None),
            "params": {k: opts[k] for k in ("bt_smooth_samples", "bt_threshold_frac",
                                            "bt_persist_m", "bt_persist_frac",
                                            "bt_median_pings", "bt_outlier_m",
                                            "bt_min_altitude_m", "bt_header_tolerance_frac")},
        },
        "slant_range_correction": {
            "formula": "ground = sqrt(slant^2 - altitude^2), resampled linearly onto "
                       "ground-range bin centres (j + 0.5) * m_per_px_across",
            "m_per_px_across": dx, "basis": dx_basis,
            "native_slant_m_per_sample": round(native_dx, 6),
            "ground_range_max_m": round(ground_max, 3),
            "water_column": "slant samples nearer than the altitude map to no ground range "
                            "and are excluded from the image by construction; they are kept "
                            "in the water-column file",
        },
        "gain_normalisation": {
            "method": "empirical gain normalisation: value / mean(value) per ground-range "
                      "bin, over the line, per side, resolved by altitude bands of "
                      "EGN_ALTITUDE_BAND_M; the mean surface is Gaussian-smoothed by "
                      "normalised convolution (EGN_SMOOTH_ALTITUDE_M, EGN_SMOOTH_RANGE_M) and "
                      "interpolated linearly in altitude per ping; cells with less than "
                      "EGN_MIN_PINGS effective pings use the line-wide profile; the mean is "
                      "trimmed in a second pass to pixels whose EGN_TRIM_WINDOW local-mean "
                      "ratio to the first-pass surface is within EGN_TRIM_LOW..EGN_TRIM_HIGH, "
                      "so large targets and shadows do not set the gain",
            "port": port_used, "starboard": stbd_used,
            "roll_balance": "per-side normalisation removes constant port/starboard "
                            "imbalance; the part that varies with roll is removed by dividing "
                            "each ping-side by its median normalised intensity (within the EGN "
                            "trim band), median-smoothed over ROLL_BALANCE_PINGS",
            "params": {k: opts[k] for k in ("egn_altitude_band_m", "egn_smooth_altitude_m",
                                            "egn_smooth_range_m", "egn_min_pings",
                                            "egn_trim_low", "egn_trim_high",
                                            "egn_trim_window",
                                            "roll_balance_pings")},
        },
        "along_track": {
            "m_per_px_along": dy if along_known else None, "basis": dy_basis,
            "distance_source_intervals": distance_source,
            "median_ping_spacing_m": (round(median_spacing, 4)
                                      if median_spacing is not None and along_known else None),
            "nav_smoothing_pings": int(opts["nav_smooth_pings"]),
            "nav_fixes_accepted": int(accepted.sum()),
            "nav_fixes_rejected": int(rejected.sum()),
            "heading_source": heading_source,
            "speed_source": speed_source,
            "method": "rows at fixed distance travelled; each row linearly interpolated "
                      "between the two usable pings that bracket it",
        },
        "dropouts": {
            "pings_in_line": int(n), "pings_used": int(len(u)),
            "empty_pings": int(empty.sum()), "pings_without_altitude": int(no_altitude.sum()),
            "time_gaps": [{"after_time": _iso(clock[j]) if time_known else None,
                           "seconds": round(float(dts[j]), 3)}
                          for j in np.nonzero(gap_after[:-1])[0]] if n > 1 else [],
            "attitude_pings": int(attitude.sum()),
            "rows_by_quality": rows_by_quality,
            "params": {k: opts[k] for k in ("dropout_gap_factor", "empty_ping_energy_frac",
                                            "max_pitch_deg", "max_roll_deg", "max_heave_rate",
                                            "attitude_hold_s", "max_speed_mps",
                                            "interp_max_rows")},
            "rules": {"interpolated": "rows inside a degraded gap (time gap, empty or unusable "
                                      "pings) of at most INTERP_MAX_ROWS rows",
                      "dropout": "rows inside a gap longer than INTERP_MAX_ROWS; left as zero "
                                 "pixels, attitude/altitude/depth null",
                      "attitude": "nearest ping exceeds MAX_PITCH_DEG, MAX_ROLL_DEG or "
                                  "MAX_HEAVE_RATE (held for ATTITUDE_HOLD_S); data kept",
                      "nav_jump": "nearest ping's navigation fix implied a speed above "
                                  "MAX_SPEED_MPS and was rejected; lat/lon null on the row"},
            "positions_in_dropouts": "lat/lon/time on interpolated and dropout rows are "
                                     "interpolated between the real fixes either side",
        },
        "speckle": {"filter": opts["speckle"], "size": int(opts["lee_size"]),
                    "noise": noise_cv2, "applied": "after gain normalisation and along-track "
                                                   "resampling, before 8-bit scaling"},
        "intensity_scaling": {"percentiles": [float(low_p), float(high_p)],
                              "range": [round(float(low), 6), round(float(high), 6)],
                              "encoding": "valid pixels scaled to 1..255; 0 means no data"},
        "geometry_not_corrected": list(_GEOMETRY_NOT_CORRECTED),
        "unverified": unverified,
        "notes": notes,
        "skipped": raw.skipped[:200],
        "skipped_total": len(raw.skipped),
    }
    if raw.synthetic:
        processing["synthetic_basis"] = raw.synthetic_basis

    sidecar = {
        "format": SIDECAR_FORMAT,
        "source_file": source_name,
        "source_format": raw.source_format,
        "synthetic": bool(raw.synthetic),
        "width": int(grey.shape[1]),
        "height": int(grey.shape[0]),
        "nadir_col": float(columns),
        "m_per_px_across": dx,
        "m_per_px_along": dy if along_known else None,
        "port_is_left": True,
        "coordinate_units": raw.coordinate_units,
        "rows": rows,
        "degraded_rows": degraded,
        "processing": processing,
    }
    if water_column_block is not None:
        sidecar["water_column"] = water_column_block
    nav_path.write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")

    summary = {
        "strip": strip, "source_file": source_name, "source_format": raw.source_format,
        "synthetic": bool(raw.synthetic), "line": line_info,
        "image": image_path.name, "sidecar": nav_path.name,
        "water_column": water_column_block["path"] if water_column_block else None,
        "width": sidecar["width"], "height": sidecar["height"],
        "nadir_col": sidecar["nadir_col"],
        "m_per_px_across": dx, "m_per_px_along": sidecar["m_per_px_along"],
        "coordinate_units": raw.coordinate_units,
        "rows_with_position": int(np.isfinite(row_lat).sum()),
        "rows_by_quality": rows_by_quality, "degraded_rows": degraded,
        "bottom_track_sources": source_counts,
        "pings_in_line": int(n), "pings_used": int(len(u)),
        "unverified": unverified,
    }
    log.info("ingested %s line %s -> %s: %dx%d px at %.3f x %s m/px, %d degraded row range(s)",
             source_name, line_info.get("index"), strip, sidecar["width"], sidecar["height"],
             dx, f"{dy:.3f}" if along_known else "unknown", len(degraded))
    return IngestedStrip(strip=strip, image_path=image_path, nav_path=nav_path, summary=summary)


# --- public entry ----------------------------------------------------------


def ingest(path: Any, out_dir: Any, *, options: dict | None = None) -> list[IngestedStrip]:
    """Turn one raw XTF or JSF file into one strip per contiguous line.

    Writes {strip}.png, {strip}.nav.json and (unless disabled) {strip}.wc.npz
    into out_dir. The strip is named after the file stem, suffixed _L01, _L02
    ... when the file holds more than one line. A line that cannot be processed
    is skipped and named in the log and in the other strips' processing blocks;
    if no line survives, SonarIngestError says why for each.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    opts = _resolve_options(options)
    suffix = path.suffix.lower()
    if suffix not in SONAR_SUFFIXES:
        raise SonarIngestError(f"{path.name}: not a supported sonar file "
                               f"(expected one of {', '.join(SONAR_SUFFIXES)})")
    if not path.is_file():
        raise FileNotFoundError(f"sonar file not found: {path}")

    raw = read_xtf(path) if suffix == ".xtf" else read_jsf(path)
    lines = _split_lines(raw, opts)
    results: list[IngestedStrip] = []
    failures: list[str] = []
    for number, (index, reason) in enumerate(lines, start=1):
        strip = path.stem if len(lines) == 1 else f"{path.stem}_L{number:02d}"
        info = {"index": number, "of": len(lines), "started_because": reason,
                "pings": int(len(index))}
        try:
            results.append(_process_line(raw, index, strip, out_dir, opts, info, path.name))
        except SonarIngestError as exc:
            failures.append(f"line {number}: {exc}")
            log.error("%s line %d skipped: %s", path.name, number, exc)
    if not results:
        raise SonarIngestError(f"{path.name}: no line could be ingested. " + "; ".join(failures))
    if failures:
        for item in results:
            item.summary["lines_skipped"] = failures
            try:
                sidecar = json.loads(item.nav_path.read_text(encoding="utf-8"))
                sidecar["processing"]["lines_skipped_in_file"] = failures
                item.nav_path.write_text(json.dumps(sidecar, separators=(",", ":")),
                                         encoding="utf-8")
            except (OSError, ValueError):                         # pragma: no cover
                pass
    return results
