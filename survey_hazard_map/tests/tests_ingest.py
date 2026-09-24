#!/usr/bin/env python3
"""Tests for raw side-scan ingest, ping navigation and the water column.

    python3 tests_ingest.py
    python3 tests_ingest.py --verbose

Exit code is 1 if any case fails. Plain functions and asserts, like
unit_tests.py and smoke_test.py, for the same reason: no framework dependency.

Everything runs against a SYNTHETIC XTF written by tools/make_synthetic_xtf.py
into a temporary directory that is deleted on the way out. The synthetic file is
built from geometry -- a towfish track, slant ranges cast onto a seabed height
field -- so its ground truth is exact and a georeferencing error shows up as a
number of metres rather than as an image that merely looks plausible.

WHAT IS CHECKED
    ingest      strip PNG, sidecar schema, uniform resolution, nadir in the
                middle, synthetic marking carried through
    defects     the injected time-gap dropout, empty pings and attitude
                excursion are found in degraded_rows at the right times
    geolocation truth positions -> pixel -> Georeference("ping").locate()
                lands within 2 * m_per_px; cylinder highlights found in the
                image land within a cylinder's size of the truth
    speckle     lee_filter cuts Rayleigh speckle variance without moving the mean
    water col.  .wc.npz aligned to strip rows, bottom range matches altitude,
                fish aggregation over the ghost net stands out against the
                control net
    formats     JSF round trip; projected XTF units give null positions;
                a long gap splits a file into two strips
    pipeline    prepare_survey on the .xtf, manifest navigation block, tile
                columns, then build_hazard_map with a stand-in detector gives
                non-null latitude and longitude near the truth
    regression  unit_tests.py still passes
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "survey_hazard_map" / "tools"))

CASES = []
REPORT: dict[str, object] = {}
_STATE: dict[str, object] = {}


def case(name):
    def register(fn):
        CASES.append((name, fn))
        return fn
    return register


def workspace() -> Path:
    if "workspace" not in _STATE:
        _STATE["workspace"] = Path(tempfile.mkdtemp(prefix="deepecho_ingest_"))
    return _STATE["workspace"]


def synthetic() -> tuple[Path, dict]:
    """The full-size synthetic XTF and its truth, generated once."""
    if "xtf" not in _STATE:
        from make_synthetic_xtf import make_synthetic_xtf

        xtf, truth = make_synthetic_xtf(workspace() / "raw")
        _STATE["xtf"] = xtf
        _STATE["truth"] = json.loads(truth.read_text(encoding="utf-8"))
    return _STATE["xtf"], _STATE["truth"]


def ingested():
    """(IngestedStrip, sidecar dict), ingested once."""
    if "strip" not in _STATE:
        from survey_hazard_map import sonar_ingest
        xtf, _ = synthetic()
        strips = sonar_ingest.ingest(xtf, workspace() / "ingest")
        assert len(strips) == 1, f"expected one line, got {len(strips)}"
        _STATE["strip"] = strips[0]
        _STATE["sidecar"] = json.loads(strips[0].nav_path.read_text(encoding="utf-8"))
    return _STATE["strip"], _STATE["sidecar"]


def truth_pixel(sidecar: dict, lat: float, lon: float) -> tuple[float, float]:
    """Invert a position to a continuous strip pixel using the sidecar rows.

    For each located row: the target's along/across offset from that row's
    nadir in the row's heading frame. The abeam row is where |along| is least;
    x and y follow from the offsets and the strip resolution.
    """
    from survey_hazard_map.hazard_geo import enu_delta

    best = None
    for row in sidecar["rows"]:
        if row["lat"] is None or row["heading_deg"] is None:
            continue
        east, north = enu_delta(row["lat"], row["lon"], lat, lon)
        h = math.radians(row["heading_deg"])
        along = east * math.sin(h) + north * math.cos(h)
        across = east * math.cos(h) - north * math.sin(h)
        if best is None or abs(along) < abs(best[1]):
            best = (row["row"], along, across)
    row, along, across = best
    return (sidecar["nadir_col"] + across / sidecar["m_per_px_across"],
            row + 0.5 + along / sidecar["m_per_px_along"])


def row_of_time(sidecar: dict, iso: str) -> int:
    target = datetime.fromisoformat(iso)
    best, best_dt = 0, None
    for row in sidecar["rows"]:
        if row["time"] is None:
            continue
        dt = abs((datetime.fromisoformat(row["time"]) - target).total_seconds())
        if best_dt is None or dt < best_dt:
            best, best_dt = row["row"], dt
    return best


# --- ingest ----------------------------------------------------------------

@case("an XTF ingests into a PNG and a sidecar with the documented schema")
def _():
    from PIL import Image

    strip, sidecar = ingested()
    assert strip.image_path.is_file() and strip.nav_path.is_file()
    for key in ("format", "source_file", "source_format", "synthetic", "width", "height",
                "nadir_col", "m_per_px_across", "m_per_px_along", "port_is_left",
                "coordinate_units", "rows", "degraded_rows", "processing"):
        assert key in sidecar, f"sidecar missing {key}"
    assert sidecar["format"] == "deepecho-strip-nav/1"
    assert sidecar["source_format"] == "xtf" and sidecar["port_is_left"] is True
    with Image.open(strip.image_path) as image:
        assert image.mode == "L"
        assert image.size == (sidecar["width"], sidecar["height"])
    assert len(sidecar["rows"]) == sidecar["height"]
    row_keys = {"row", "time", "lat", "lon", "heading_deg", "altitude_m", "speed_mps",
                "pitch_deg", "roll_deg", "heave_m", "depth_m", "quality"}
    assert all(row_keys <= set(r) for r in sidecar["rows"])
    assert [r["row"] for r in sidecar["rows"][:3]] == [0, 1, 2]
    assert {r["quality"] for r in sidecar["rows"]} <= {
        "ok", "attitude", "nav_jump", "interpolated", "dropout"}
    for step in ("bottom_track", "slant_range_correction", "gain_normalisation",
                 "along_track", "dropouts", "speckle", "unverified", "geometry_not_corrected"):
        assert step in sidecar["processing"], f"processing missing {step}"


@case("the strip has a uniform, recorded resolution and nadir in the middle")
def _():
    _, sidecar = ingested()
    _, truth = synthetic()
    across, along = sidecar["m_per_px_across"], sidecar["m_per_px_along"]
    assert across and along and across > 0 and along > 0
    # Auto resolution: slant sample spacing, square pixels.
    assert abs(across - truth["sonar"]["slant_m_per_sample"]) < 1e-6
    assert abs(along - across) < 1e-9
    assert abs(sidecar["nadir_col"] - sidecar["width"] / 2.0) <= 1.0
    REPORT["strip"] = (f"{sidecar['width']}x{sidecar['height']} px, {across} m/px across, "
                       f"{along} m/px along")


@case("a file marked synthetic stays marked synthetic")
def _():
    strip, sidecar = ingested()
    assert sidecar["synthetic"] is True and strip.summary["synthetic"] is True
    assert "SYNTHETIC" in sidecar["processing"]["synthetic_basis"]


@case("positions come from the file's degrees and every row is located")
def _():
    _, sidecar = ingested()
    assert sidecar["coordinate_units"].startswith("degrees")
    located = [r for r in sidecar["rows"] if r["lat"] is not None]
    assert len(located) == len(sidecar["rows"])
    assert all(9.0 < r["lat"] < 9.3 and 78.9 < r["lon"] < 79.2 for r in located)


@case("the bottom track matches the header altitude and names its source")
def _():
    _, sidecar = ingested()
    bottom = sidecar["processing"]["bottom_track"]
    assert bottom["source_counts"]["bottom_track"] > 0.9 * sum(bottom["source_counts"].values())
    assert bottom["median_abs_difference_from_header_m"] < 0.1, bottom
    REPORT["bottom_track_vs_header_m"] = bottom["median_abs_difference_from_header_m"]


# --- defects ---------------------------------------------------------------

@case("the injected time-gap dropout is in degraded_rows as 'dropout'")
def _():
    _, sidecar = ingested()
    _, truth = synthetic()
    gap = truth["injected"]["dropout_time_gap"]
    start, end = row_of_time(sidecar, gap["time_start"]), row_of_time(sidecar, gap["time_end"])
    hits = [r for r in sidecar["degraded_rows"] if r[2] == "dropout"
            and r[0] <= end + 5 and r[1] >= start - 5]
    assert hits, f"no dropout near rows {start}..{end}: {sidecar['degraded_rows']}"
    length_m = (hits[0][1] - hits[0][0] + 1) * sidecar["m_per_px_along"]
    assert abs(length_m - gap["distance_m"]) < 1.5, (length_m, gap["distance_m"])
    assert all(sidecar["rows"][k]["altitude_m"] is None for k in range(hits[0][0], hits[0][1] + 1))


@case("the empty pings are bridged and flagged 'interpolated'")
def _():
    _, sidecar = ingested()
    _, truth = synthetic()
    empty = truth["injected"]["empty_pings"]
    start, end = row_of_time(sidecar, empty["time_start"]), row_of_time(sidecar, empty["time_end"])
    assert any(r[2] == "interpolated" and r[0] <= end + 3 and r[1] >= start - 3
               for r in sidecar["degraded_rows"]), sidecar["degraded_rows"]


@case("the attitude excursion is in degraded_rows as 'attitude', and nothing else is")
def _():
    _, sidecar = ingested()
    _, truth = synthetic()
    att = truth["injected"]["attitude_excursion"]
    start, end = row_of_time(sidecar, att["time_start"]), row_of_time(sidecar, att["time_end"])
    ranges = [r for r in sidecar["degraded_rows"] if r[2] == "attitude"]
    covered = sum(max(0, min(r[1], end) - max(r[0], start) + 1) for r in ranges)
    assert covered >= 0.8 * (end - start + 1), (ranges, start, end)
    outside = sum(r[1] - r[0] + 1 for r in ranges) - covered
    assert outside <= 0.2 * (end - start + 1), (ranges, start, end)


# --- geolocation -----------------------------------------------------------

@case("ping navigation locates every truth target within 2 * m_per_px")
def _():
    from survey_hazard_map.hazard_geo import Georeference, haversine_m

    _, sidecar = ingested()
    _, truth = synthetic()
    reference = Georeference.from_ping_nav("s", sidecar)
    assert reference.detail["across_track_resolved"] is True
    assert reference.metres_per_pixel() == (sidecar["m_per_px_across"], sidecar["m_per_px_along"])
    limit = 2 * max(sidecar["m_per_px_across"], sidecar["m_per_px_along"])
    worst_roundtrip = worst_independent = 0.0
    for target in truth["targets"]:
        # Round trip through the sidecar rows.
        cx, cy = truth_pixel(sidecar, target["latitude"], target["longitude"])
        lat, lon = reference.locate(cx, cy)
        worst_roundtrip = max(worst_roundtrip,
                              haversine_m(lat, lon, target["latitude"], target["longitude"]))
        # Independent: pixel from the truth's own along-track distance and
        # cross-track offset, nothing from the sidecar but its scale.
        cx = sidecar["nadir_col"] + target["cross_track_m"] / sidecar["m_per_px_across"]
        cy = target["along_track_m"] / sidecar["m_per_px_along"] + 0.5
        lat, lon = reference.locate(cx, cy)
        worst_independent = max(worst_independent,
                                haversine_m(lat, lon, target["latitude"], target["longitude"]))
    assert worst_roundtrip <= limit, worst_roundtrip
    assert worst_independent <= limit, worst_independent
    REPORT["geolocation_roundtrip_worst_m"] = round(worst_roundtrip, 4)
    REPORT["geolocation_independent_worst_m"] = round(worst_independent, 4)


@case("cylinder highlights found in the image land within a cylinder of the truth")
def _():
    import numpy as np
    from PIL import Image

    from survey_hazard_map.hazard_geo import Georeference, haversine_m

    strip, sidecar = ingested()
    _, truth = synthetic()
    reference = Georeference.from_ping_nav("s", sidecar)
    image = np.asarray(Image.open(strip.image_path)).astype(float)
    errors = {}
    for target in truth["targets"]:
        if target["class"] != "cylinder":
            continue
        cx, cy = truth_pixel(sidecar, target["latitude"], target["longitude"])
        half = int(3.0 / sidecar["m_per_px_across"])
        x0, y0 = max(int(cx) - half, 0), max(int(cy) - half, 0)
        window = image[y0:int(cy) + half, x0:int(cx) + half]
        ys, xs = np.nonzero(window >= np.percentile(window, 99))
        lat, lon = reference.locate(xs.mean() + x0 + 0.5, ys.mean() + y0 + 0.5)
        errors[target["id"]] = round(haversine_m(lat, lon, target["latitude"],
                                                 target["longitude"]), 3)
    assert errors and all(e < 1.0 for e in errors.values()), errors
    REPORT["cylinder_highlight_offset_m"] = errors


@case("rows with a null position are bridged over a short gap, refused over a long one")
def _():
    from survey_hazard_map.hazard_geo import Georeference

    _, sidecar = ingested()
    patched = json.loads(json.dumps(sidecar))
    rows = patched["rows"]
    for k in range(1000, 1010):                                   # 1 m: bridged
        rows[k]["lat"] = rows[k]["lon"] = None
    for k in range(2000, 2000 + int(60 / patched["m_per_px_along"])):   # 60 m: refused
        rows[k]["lat"] = rows[k]["lon"] = None
    reference = Georeference.from_ping_nav("s", patched)
    assert reference.locate(patched["nadir_col"], 1005.5)[0] is not None
    assert reference.locate(patched["nadir_col"], 2100.5) == (None, None)
    row = reference.nav_row(1005.5)
    assert row["altitude_m"] is not None and row["quality"] in ("ok", "attitude")


@case("heading interpolation passes through north, not the long way round")
def _():
    from survey_hazard_map.hazard_geo import Georeference

    rows = [{"row": 0, "lat": 9.0, "lon": 79.0, "heading_deg": 359.0, "quality": "ok"},
            {"row": 1, "lat": 9.0, "lon": 79.0, "heading_deg": 1.0, "quality": "attitude"}]
    reference = Georeference.from_ping_nav("s", {
        "format": "deepecho-strip-nav/1", "width": 3, "height": 2, "nadir_col": 1.0,
        "m_per_px_across": 1.0, "m_per_px_along": 1.0, "port_is_left": True, "rows": rows})
    row = reference.nav_row(1.0)                                  # halfway between rows
    assert min(row["heading_deg"], 360 - row["heading_deg"]) < 1e-6, row
    assert row["quality"] == "attitude"
    # 10 m to starboard of a north-heading nadir is due east.
    lat, lon = reference.locate(11.0, 1.0)
    assert abs(lat - 9.0) < 1e-6 and lon > 79.0


# --- speckle ---------------------------------------------------------------

@case("lee_filter reduces Rayleigh speckle variance without shifting the mean")
def _():
    import numpy as np

    from survey_hazard_map.sonar_ingest import lee_filter

    rng = np.random.default_rng(3)
    field = 100.0 * rng.rayleigh(scale=1 / math.sqrt(math.pi / 2), size=(256, 256)) ** 2
    out = lee_filter(field, 5)
    assert out.shape == field.shape and out.dtype == np.float32
    assert out.var() < 0.3 * field.var(), (out.var(), field.var())
    assert abs(out.mean() - field.mean()) < 0.03 * field.mean()
    # A bright edge survives: k -> 1 where variance is structure, not speckle.
    step = np.full((64, 64), 50.0)
    step[:, 32:] = 200.0
    filtered = lee_filter(step, 5)
    assert filtered[10, 20] == 50.0 and filtered[10, 40] == 200.0
    # uint8 in, uint8 out; masked pixels untouched.
    grey = rng.integers(0, 255, (32, 32), dtype=np.uint8)
    mask = np.ones_like(grey, dtype=bool)
    mask[:, :4] = False
    assert lee_filter(grey, 5, mask=mask).dtype == np.uint8
    assert (lee_filter(grey, 5, mask=mask)[:, :4] == grey[:, :4]).all()


# --- water column ----------------------------------------------------------

@case("the water column is saved, row-aligned, and bounded by the bottom return")
def _():
    import numpy as np

    strip, sidecar = ingested()
    block = sidecar["water_column"]
    path = strip.nav_path.parent / block["path"]
    assert path.is_file()
    data = np.load(path)
    assert data["port"].shape == data["starboard"].shape
    assert data["port"].shape[0] == sidecar["height"] == data["bottom_range_m"].shape[0]
    assert data["port"].dtype == np.float32
    bins = data["port"].shape[1]
    assert abs(bins * block["m_per_bin"] - math.ceil(block["max_range_m"] / block["m_per_bin"])
               * block["m_per_bin"]) < 1e-6
    for key in ("path", "m_per_bin", "max_range_m", "note"):
        assert key in block
    # Bins beyond each row's own bottom range are NaN.
    centres = (np.arange(bins) + 0.5) * block["m_per_bin"]
    bottom = data["bottom_range_m"]
    beyond = centres[None, :] >= np.where(np.isfinite(bottom), bottom, -1)[:, None]
    assert np.isnan(data["starboard"][beyond]).all()
    # Dropout rows are NaN throughout.
    for start, end, reason in sidecar["degraded_rows"]:
        if reason == "dropout":
            assert np.isnan(data["port"][start:end + 1]).all()
            assert np.isnan(bottom[start:end + 1]).all()


@case("bottom_range_m matches the towfish altitude recorded in the XTF headers")
def _():
    import numpy as np
    import pyxtf

    strip, sidecar = ingested()
    xtf, _ = synthetic()
    data = np.load(strip.nav_path.parent / sidecar["water_column"]["path"])
    header, packets = pyxtf.xtf_read(str(xtf), types=[pyxtf.XTFHeaderType.sonar])
    pings = packets[pyxtf.XTFHeaderType.sonar]
    times = np.array([datetime(p.Year, p.Month, p.Day, p.Hour, p.Minute, p.Second,
                               p.HSeconds * 10000, tzinfo=timezone.utc).timestamp()
                      for p in pings])
    altitude = np.array([p.SensorPrimaryAltitude for p in pings])
    rows = [r for r in sidecar["rows"] if r["time"] and r["quality"] == "ok"]
    row_t = np.array([datetime.fromisoformat(r["time"]).timestamp() for r in rows])
    expected = np.interp(row_t, times, altitude)
    measured = data["bottom_range_m"][[r["row"] for r in rows]]
    error = np.abs(measured - expected)
    assert np.median(error) < 0.1 and np.percentile(error, 95) < 0.3, \
        (np.median(error), np.percentile(error, 95))
    REPORT["bottom_range_vs_header_altitude_m"] = {
        "median": round(float(np.median(error)), 3),
        "p95": round(float(np.percentile(error, 95)), 3)}
    assert all(r["depth_m"] is not None for r in rows)
    assert all(abs(r["seabed_depth_m"] - (r["depth_m"] + r["altitude_m"])) < 1e-2 for r in rows[:50])


@case("fish over the ghost-fishing net stand out against the control net")
def _():
    import numpy as np

    strip, sidecar = ingested()
    _, truth = synthetic()
    data = np.load(strip.nav_path.parent / sidecar["water_column"]["path"])
    net = {t["id"]: t for t in truth["targets"] if t["class"] == "net"}
    aggregated = next(t for t in net.values() if t["fish_aggregation"])
    control = next(t for t in net.values() if not t["fish_aggregation"])
    assert aggregated["side"] == control["side"] == "starboard"
    half = int(20.0 / sidecar["m_per_px_along"])                  # +/- 20 m along-track

    def mean_near(target):
        _, cy = truth_pixel(sidecar, target["latitude"], target["longitude"])
        side = data["starboard" if target["side"] == "starboard" else "port"]
        return float(np.nanmean(side[max(int(cy) - half, 0):int(cy) + half]))

    hot, cold = mean_near(aggregated), mean_near(control)
    assert hot > 2.0 * cold, (hot, cold)
    REPORT["water_column_mean_aggregated_vs_control"] = (round(hot, 3), round(cold, 3))


# --- formats ---------------------------------------------------------------

def _write_jsf(path: Path, raw, annotation: bytes = b"SYNTHETIC JSF test") -> None:
    """A JSF file laid out as sonar_ingest reads it, from decoded XTF pings.

    This proves the reader is self-consistent with the offsets it documents.
    It does NOT prove those offsets match EdgeTech's own files; the ingest
    marks that as unverified.
    """
    import numpy as np

    out = bytearray()
    for j in range(len(raw.port)):
        when = datetime.fromtimestamp(raw.meta["time"][j], tz=timezone.utc)
        midnight = when.replace(hour=0, minute=0, second=0, microsecond=0)
        for channel, samples, dr in ((0, raw.port[j], raw.port_dr[j]),
                                     (1, raw.stbd[j], raw.stbd_dr[j])):
            head = bytearray(240)
            struct.pack_into("<i", head, 0, int(raw.meta["time"][j]))
            struct.pack_into("<I", head, 8, j)
            struct.pack_into("<h", head, 34, 0)
            struct.pack_into("<i", head, 80, int(round(raw.meta["lon"][j] * 600000)))
            struct.pack_into("<i", head, 84, int(round(raw.meta["lat"][j] * 600000)))
            struct.pack_into("<h", head, 88, 2)
            struct.pack_into("24s", head, 90, annotation)
            struct.pack_into("<H", head, 114, len(samples))
            dr = float(dr) if np.isfinite(dr) else 0.1
            struct.pack_into("<I", head, 116, int(round(dr * 2 / 1500.0 * 1e9)))
            struct.pack_into("<i", head, 136, int(raw.meta["depth"][j] * 1000))
            struct.pack_into("<i", head, 144, int(raw.meta["altitude"][j] * 1000))
            struct.pack_into("<f", head, 148, 1500.0)
            struct.pack_into("<H", head, 172, int(round(raw.meta["heading"][j] * 100)) % 36000)
            struct.pack_into("<h", head, 174, int(round(raw.meta["pitch"][j] * 32768 / 180)))
            struct.pack_into("<h", head, 176, int(round(raw.meta["roll"][j] * 32768 / 180)))
            struct.pack_into("<h", head, 194, int(round(raw.meta["speed"][j] / 0.514444 * 10)))
            struct.pack_into("<I", head, 200, int((when - midnight).total_seconds() * 1000))
            body = bytes(head) + np.clip(samples, 0, 65535).astype("<u2").tobytes()
            out += struct.pack("<HBBHBBBBHi", 0x1601, 16, 0, 80, 0, 20, channel, 0, 0, len(body))
            out += body
    path.write_bytes(bytes(out))


@case("a JSF file laid out per the documented offsets ingests and locates")
def _():
    from survey_hazard_map import sonar_ingest
    from make_synthetic_xtf import make_synthetic_xtf

    small_dir = workspace() / "jsf"
    xtf, truth_path = make_synthetic_xtf(small_dir, pings=500, name="SYNTHETIC_small")
    raw = sonar_ingest.read_xtf(xtf)
    jsf = small_dir / "SYNTHETIC_small.jsf"
    _write_jsf(jsf, raw)
    strips = sonar_ingest.ingest(jsf, small_dir / "out")
    assert len(strips) == 1
    sidecar = json.loads(strips[0].nav_path.read_text(encoding="utf-8"))
    assert sidecar["source_format"] == "jsf" and sidecar["synthetic"] is True
    assert all(r["lat"] is not None for r in sidecar["rows"])
    assert any("not been validated" in u for u in sidecar["processing"]["unverified"])
    assert any(r[2] == "dropout" for r in sidecar["degraded_rows"])
    assert sidecar["rows"][0]["heave_m"] is None                 # not in message 80

    not_jsf = small_dir / "bad.jsf"
    not_jsf.write_bytes(b"\x00" * 64)
    try:
        sonar_ingest.ingest(not_jsf, small_dir / "bad")
    except sonar_ingest.SonarIngestError as exc:
        assert "not an EdgeTech JSF file" in str(exc)
    else:
        raise AssertionError("a non-JSF file was accepted")


@case("projected XTF coordinates give null positions, and say why")
def _():
    import pyxtf

    from survey_hazard_map import sonar_ingest
    from survey_hazard_map.hazard_geo import Georeference, NavigationError
    from make_synthetic_xtf import make_synthetic_xtf

    folder = workspace() / "projected"
    xtf, _ = make_synthetic_xtf(folder, pings=300, name="SYNTHETIC_projected")
    data = bytearray(xtf.read_bytes())
    struct.pack_into("<H", data, pyxtf.XTFFileHeader.NavUnits.offset, 0)
    xtf.write_bytes(bytes(data))
    strips = sonar_ingest.ingest(xtf, folder / "out")
    sidecar = json.loads(strips[0].nav_path.read_text(encoding="utf-8"))
    assert all(r["lat"] is None and r["lon"] is None for r in sidecar["rows"])
    assert "projected" in sidecar["coordinate_units"]
    # Speed still gives the along-track scale; only the position is refused.
    assert sidecar["m_per_px_along"] is not None
    try:
        Georeference.from_ping_nav("s", sidecar)
    except NavigationError:
        pass
    else:
        raise AssertionError("a sidecar with no positions produced a Georeference")


@case("a gap longer than SPLIT_GAP_S splits one file into two strips")
def _():
    from survey_hazard_map import sonar_ingest
    from make_synthetic_xtf import make_synthetic_xtf

    folder = workspace() / "split"
    xtf, _ = make_synthetic_xtf(folder, pings=400, name="SYNTHETIC_split")
    strips = sonar_ingest.ingest(xtf, folder / "out", options={"split_gap_s": 0.5})
    assert [s.strip for s in strips] == ["SYNTHETIC_split_L01", "SYNTHETIC_split_L02"]
    assert all(s.summary["line"]["of"] == 2 for s in strips)
    try:
        sonar_ingest.ingest(xtf, folder / "x", options={"no_such_knob": 1})
    except ValueError as exc:
        assert "unknown sonar_ingest option" in str(exc)
    else:
        raise AssertionError("an unknown option was silently ignored")


# --- dimensions ------------------------------------------------------------

@case("detection dimensions come from strip resolution, or are null with a reason")
def _():
    from survey_hazard_map.hazard_coords import attach_dimensions, strip_resolutions, to_global

    items = [{"bbox_tile": [10, 10, 30, 50], "tile_x": 0, "tile_y": 0, "strip": "a"},
             {"bbox_tile": [10, 10, 30, 50], "tile_x": 0, "tile_y": 0, "strip": "b"}]
    to_global(items, {"a": (0.1, 0.05)})
    assert items[0]["width_m"] == 2.0 and items[0]["length_m"] == 2.0
    assert items[1]["width_m"] is None and items[1]["length_m"] is None
    assert "unknown" in items[1]["dimension_basis"]
    assert items[0]["dimensions"]["length_m"] == 2.0 and items[0]["dimensions"]["length_px"] == 40.0
    assert items[1]["dimensions"]["width_m"] is None and items[1]["dimensions"]["basis"]
    assert attach_dimensions(items, {"b": (0.2, 0.2)}) == 1
    assert strip_resolutions([], {"strips": [{"strip": "a", "m_per_px_across": 0.1,
                                              "m_per_px_along": 0.1}]}) == {"a": (0.1, 0.1)}


# --- pipeline --------------------------------------------------------------

@case("prepare_survey ingests the XTF and records ping navigation in the manifest")
def _():
    from survey_hazard_map import hazard_config as cfg
    from survey_hazard_map.survey_preparation import load_manifest, prepare_survey

    xtf, _ = synthetic()
    out = workspace() / "survey"
    tiles_dir, manifest_csv = prepare_survey([xtf], out)
    rows, survey = load_manifest(out / cfg.MANIFEST_JSON)
    navigation = survey["navigation"]
    strip = xtf.stem
    assert navigation["mode"] == "ping"
    assert navigation["sidecars"] == {strip: f"nav/{strip}.nav.json"}
    assert navigation["water_columns"] == {strip: f"nav/{strip}.wc.npz"}
    assert (out / navigation["sidecars"][strip]).is_file()
    assert (out / navigation["water_columns"][strip]).is_file()
    assert (out / "strips" / f"{strip}.png").is_file()
    assert survey["coordinate_mode"] == cfg.COORD_MODE_GEO
    assert survey["synthetic_strips"] == [strip]
    entry = survey["strips"][0]
    assert entry["m_per_px_across"] and entry["m_per_px_along"] and entry["degraded_rows"]
    assert rows and all(r["lat"] is not None for r in rows)
    assert all(r["m_per_px_across"] == entry["m_per_px_across"] for r in rows)
    qualities = {r["quality"] for r in rows}
    assert "dropout" in qualities and "ok" in qualities, qualities
    csv_rows, _ = load_manifest(manifest_csv)
    assert isinstance(csv_rows[0]["quality"], str)
    blob = json.dumps(survey)
    assert str(workspace()) not in blob, "absolute path leaked into the manifest"
    _STATE["survey_out"] = out
    _STATE["tiles_dir"] = tiles_dir


@case("references_for_survey prefers the sidecar over a refit of tile centres")
def _():
    from survey_hazard_map.hazard_geo import haversine_m, references_for_survey, references_from_manifest
    from survey_hazard_map.survey_preparation import load_manifest

    out = _STATE.get("survey_out")
    assert out is not None, "depends on the prepare_survey case"
    rows, survey = load_manifest(out / "manifest.json")
    _, truth = synthetic()
    references, source = references_for_survey(rows, survey, out)
    refit, _ = references_from_manifest(rows)
    strip = next(iter(references))
    assert references[strip].mode == "ping" and source["mode"] == "ping"
    sidecar = json.loads((out / survey["navigation"]["sidecars"][strip]).read_text())
    worst_ping = worst_refit = 0.0
    for target in truth["targets"]:
        cx, cy = truth_pixel(sidecar, target["latitude"], target["longitude"])
        for ref, name in ((references[strip], "ping"), (refit[strip], "refit")):
            lat, lon = ref.locate(cx, cy)
            error = haversine_m(lat, lon, target["latitude"], target["longitude"])
            if name == "ping":
                worst_ping = max(worst_ping, error)
            else:
                worst_refit = max(worst_refit, error)
    assert worst_ping < 0.2, worst_ping
    REPORT["pipeline_ping_reference_worst_m"] = round(worst_ping, 3)
    REPORT["pipeline_tile_centre_refit_worst_m"] = round(worst_refit, 3)
    # Missing sidecar: falls back to the refit and says so.
    moved = out / survey["navigation"]["sidecars"][strip]
    hidden = moved.with_suffix(".hidden")
    moved.rename(hidden)
    try:
        fallback, fallback_source = references_for_survey(rows, survey, out)
        assert fallback[strip].mode != "ping" and "sidecars_missing" in fallback_source
    finally:
        hidden.rename(moved)


class TruthDetector:
    """Stands in for a checkpoint: reports each SYNTHETIC truth target in the
    tiles that contain it, at the pixel the truth inverts to. Reads no pixels."""

    name = "truth-stand-in (tests_ingest.py, not a model)"

    def __init__(self, sidecar: dict, truth: dict) -> None:
        from survey_hazard_map import hazard_config as cfg
        self.conf = cfg.CONF_THRESH
        self.targets = []
        for target in truth["targets"]:
            cx, cy = truth_pixel(sidecar, target["latitude"], target["longitude"])
            half = max(target["length_m"], target["width_m"]) / 2 / sidecar["m_per_px_across"]
            self.targets.append((cx, cy, min(half, 60.0), target["class"]))
        self.classes = sorted({t[3] for t in self.targets})

    def __call__(self, image_path: Path) -> list[dict]:
        from PIL import Image

        from survey_hazard_map.hazard_detect import parse_tile_name

        parsed = parse_tile_name(image_path.name)
        if parsed is None:
            return []
        _, tx, ty = parsed
        with Image.open(image_path) as tile:
            width, height = tile.size
        boxes = []
        for cx, cy, half, label in self.targets:
            lx, ly = cx - tx, cy - ty
            if not (0 <= lx < width and 0 <= ly < height):
                continue
            h = min(half, lx, width - lx, ly, height - ly)          # keep the centre exact
            boxes.append({"class": label, "confidence": 0.8,
                          "bbox": [lx - h, ly - h, lx + h, ly + h]})
        return boxes


@case("the full pipeline on the synthetic XTF gives non-null lat/lon near the truth")
def _():
    from survey_hazard_map.hazard_geo import haversine_m
    from survey_hazard_map.hazard_map import build_hazard_map

    out = _STATE.get("survey_out")
    assert out is not None, "depends on the prepare_survey case"
    _, truth = synthetic()
    sidecar = json.loads((out / "nav" / f"{synthetic()[0].stem}.nav.json").read_text())
    export = build_hazard_map("SYNTHETIC", _STATE["tiles_dir"], out,
                              detector=TruthDetector(sidecar, truth), demo=True)
    detections = export["detections"]
    assert detections, "no detections came out of the pipeline"
    assert all(d["latitude"] is not None and d["longitude"] is not None for d in detections)
    errors = {}
    for target in truth["targets"]:
        near = [haversine_m(d["latitude"], d["longitude"], target["latitude"],
                            target["longitude"]) for d in detections
                if d["object_class"] == target["class"]]
        errors[target["id"]] = round(min(near), 3) if near else None
    assert all(e is not None for e in errors.values()), errors
    # Loose on purpose: build_hazard_map may locate through the ping sidecar
    # (centimetres) or through the older refit of tile centres (metres on a
    # curving track). Either way nothing may be tens of metres out.
    assert max(errors.values()) < 10.0, errors
    REPORT["pipeline_export_error_m"] = errors
    REPORT["pipeline_export_navigation_mode"] = export["provenance"]["navigation"].get("mode")


# --- regression ------------------------------------------------------------

@case("unit_tests.py still passes")
def _():
    result = subprocess.run([sys.executable, str(ROOT / "survey_hazard_map" / "tests" / "unit_tests.py")],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    args = parser.parse_args()

    import logging
    import warnings

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(levelname)-7s %(message)s")
    warnings.simplefilter("ignore", RuntimeWarning)

    failures = []
    try:
        for name, fn in CASES:
            try:
                fn()
                print(f"  ok    {name}")
            except Exception as exc:
                failures.append((name, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    finally:
        if "workspace" in _STATE:
            if args.keep:
                print(f"\nworkspace kept at {_STATE['workspace']}")
            else:
                shutil.rmtree(_STATE["workspace"], ignore_errors=True)

    if REPORT:
        print("\nmeasured on SYNTHETIC data:")
        for key, value in REPORT.items():
            print(f"  {key}: {value}")
    print()
    if failures:
        print(f"FAILED  {len(failures)} of {len(CASES)} ingest tests")
        return 1
    print(f"PASSED  {len(CASES)}/{len(CASES)} ingest tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
