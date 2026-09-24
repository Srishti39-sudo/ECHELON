"""Fetch, clip and record every dataset GhostTrace's geographic stages bundle.

    .venv/bin/python tools/fetch_ghosttrace_data.py                 # everything
    .venv/bin/python tools/fetch_ghosttrace_data.py --only currents --days 14
    .venv/bin/python tools/fetch_ghosttrace_data.py --only land,protected,turtle

Writes into data/ghosttrace/:

    layers/*.geojson            vector layers (see ghosttrace/layers.py)
    bathymetry/etopo2022_*.nc   NOAA ETOPO 2022 15" elevation, clipped
    currents/hycom_*.nc         HYCOM ESPC-D-V02 surface + near-bottom u/v
    manifest.json               machine-readable record of every source
    SOURCES.md                  the same record for people
    .gitignore                  keeps non-redistributable layers out of git

WHY A SCRIPT AND NOT A FOLDER OF FILES
    Every file under data/ghosttrace can be rebuilt from this script, and the
    manifest records the URL, the licence, the access date, what was clipped
    and how. A dataset that cannot be traced to where it came from is not used.

HONESTY RULES THIS SCRIPT ENFORCES
    - No fabricated geometry. A curated feature is built only from a position
      in a named source (OpenStreetMap, a paper, a government document), and
      carries that citation and a geometry_quality that says how rough it is.
      When no citable position exists, the feature is left out and SOURCES.md
      says so.
    - No fabricated currents. If every current source fails, the script stops
      with an error. There is no fallback to a synthetic field.
    - Licences are respected. UNEP-WCMC layers may not be redistributed
      (UNEP-WCMC General Data License, clause 3), so they are fetched to the
      local machine and git-ignored, and the manifest marks them
      redistributable: false.

SOURCES, IN THE ORDER THEY WERE CHOSEN
    reefs        UNEP-WCMC Global Distribution of Warm-water Coral Reefs v4.1
                 through UNEP-WCMC's public ArcGIS FeatureServer, because the
                 Allen Coral Atlas (the CC BY 4.0 preference) requires an
                 account login to download. Supplemented by OpenStreetMap
                 natural=reef features, which are redistributable.
    seagrass     UNEP-WCMC Global Distribution of Seagrasses, same service,
                 same licence. Dugong forage habitat.
    protected    OpenStreetMap boundary=protected_area / national_park and
                 leisure=nature_reserve through the Overpass API, filtered to
                 areas that touch the sea or the coast.
    turtle       curated: olive ridley mass-nesting rookeries, positioned from
                 OpenStreetMap river mouths / sanctuary polygons and the
                 coastline, extents from the cited literature.
    dugong       curated: Palk Bay Dugong Conservation Reserve, coastal extent
                 from a cited description, anchors from OpenStreetMap.
    land         OpenStreetMap coastline polygonised within each box, with
                 Natural Earth 10 m land used to decide which side is land;
                 falls back to Natural Earth alone if the OSM build fails a
                 sanity check.
    harbours     OpenStreetMap harbours, ports, fishing ports and named jetties.
                 Global Fishing Watch is used instead only when GFW_API_TOKEN
                 is set (see fetch_gfw).
    bathymetry   NOAA NCEI ETOPO 2022 15 arc-second surface elevation, via
                 NCEI THREDDS OPeNDAP.
    currents     HYCOM ESPC-D-V02 global 1/12 degree, 3-hourly, via the
                 ncss.hycom.org NetCDF Subset Service (FMRC "best" time
                 series: analysis for past days, the latest run's forecast
                 after). INCOIS was checked first and does not expose its
                 HOOFS current forecasts programmatically; see SOURCES.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
import shapely
from shapely.geometry import (LineString, MultiLineString, Point, Polygon, box, mapping,
                              shape)
from shapely.ops import linemerge, polygonize, substring, unary_union

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ghosttrace import config_geo as cfg  # noqa: E402
from ghosttrace.layers import to_local, to_lonlat  # noqa: E402

OUT = cfg.DATA_DIR
UA = {"User-Agent": "DeepEcho-GhostTrace-fetcher/1.0 (SIH26057 research prototype)"}
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
TODAY = datetime.now(timezone.utc).date().isoformat()

# Padding around each region box when clipping, degrees. Features straddling
# the edge keep their shape a little past it, so a nearest-feature search at
# the edge of the box is not cut short by the clip.
CLIP_PAD_DEG = 0.25


def log(msg: str) -> None:
    print(f"[fetch {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def padded(region: str, pad: float = CLIP_PAD_DEG) -> tuple[float, float, float, float]:
    s, w, n, e = cfg.REGIONS[region]
    return s - pad, w - pad, n + pad, e + pad


# --- manifest ------------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    if cfg.MANIFEST_PATH.exists():
        return json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"format": "deepecho-ghosttrace-data/1", "sources": [], "regions_fetched": [],
            "not_used": [], "files": {}}


def save_manifest(m: dict[str, Any]) -> None:
    """Merge this run's records into the manifest on disk, then rewrite SOURCES.md.

    Merged rather than overwritten so two fetcher processes (for example a long
    currents download in the background and a layers run in the foreground) do
    not erase each other's records.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    disk = load_manifest()
    for s in m.get("sources", []):
        if s.get("id") in _TOUCHED_SOURCES:
            _put(disk, "sources", "id", s)
    for s in m.get("not_used", []):
        if s.get("name") in _TOUCHED_NOT_USED:
            _put(disk, "not_used", "name", s)
    disk["regions_fetched"] = sorted(set(disk.get("regions_fetched", [])) | set(m.get("regions_fetched", [])))
    for key in ("absent_features", "fishing_activity_source"):
        if m.get(key):
            disk[key] = m[key]
    m.clear()
    m.update(disk)
    m["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    files = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "SOURCES.md", ".gitignore"):
            files[str(p.relative_to(OUT))] = p.stat().st_size
            total += p.stat().st_size
    m["files"] = files
    m["total_bytes"] = total
    cfg.MANIFEST_PATH.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sources_md(m)
    write_gitignore(m)


# Records written by THIS process. save_manifest merges only these into the manifest on disk,
# so a long-running fetch cannot overwrite a newer record with the stale copy it loaded at start.
_TOUCHED_SOURCES: set[str] = set()
_TOUCHED_NOT_USED: set[str] = set()


def _put(m: dict[str, Any], key: str, field: str, record: dict[str, Any]) -> None:
    m[key] = [s for s in m.get(key, []) if s.get(field) != record[field]] + [record]


def upsert_source(m: dict[str, Any], record: dict[str, Any]) -> None:
    record.setdefault("accessed", TODAY)
    _TOUCHED_SOURCES.add(record["id"])
    _put(m, "sources", "id", record)


def upsert_not_used(m: dict[str, Any], record: dict[str, Any]) -> None:
    _TOUCHED_NOT_USED.add(record["name"])
    _put(m, "not_used", "name", record)


def write_gitignore(m: dict[str, Any]) -> None:
    lines = ["# Written by tools/fetch_ghosttrace_data.py.",
             "# These files come from sources whose licence forbids redistribution.",
             "# They are fetched locally and must not be committed."]
    for s in m["sources"]:
        if s.get("redistributable") is False:
            lines += [f"# {s['name']}"] + list(s.get("files", []))
    (OUT / ".gitignore").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sources_md(m: dict[str, Any]) -> None:
    out = ["# GhostTrace bundled data: sources",
           "",
           "Generated by `tools/fetch_ghosttrace_data.py` from `manifest.json`. Do not edit by hand;",
           "re-run the fetcher. Every file in this directory is listed under the source it came from.",
           "",
           f"Last updated: {m.get('updated')}. Total size on disk: {m.get('total_bytes', 0) / 1e6:.2f} MB.",
           "",
           "Regions with fetched layers: " + ", ".join(
               f"`{r}` {cfg.REGIONS[r]}" for r in m.get("regions_fetched", []) if r in cfg.REGIONS)
           + " as (south, west, north, east).",
           ""]
    for s in m["sources"]:
        out += [f"## {s['name']}", "",
                f"- **id**: `{s['id']}`",
                f"- **URL**: {s.get('url')}",
                f"- **Licence**: {s.get('licence')}",
                f"- **Redistributable in this repository**: {'yes' if s.get('redistributable', True) else 'NO (git-ignored, fetched locally)'}",
                f"- **Accessed**: {s.get('accessed')}",
                f"- **Used for**: {s.get('used_for')}"]
        if s.get("citation"):
            out.append(f"- **Citation**: {s['citation']}")
        if s.get("snapshot"):
            out.append(f"- **Snapshot**: {s['snapshot']}")
        if s.get("clipped"):
            out.append(f"- **Clipped / processed**: {s['clipped']}")
        if s.get("files"):
            sizes = m.get("files", {})
            out.append("- **Files**: " + ", ".join(
                f"`{f}` ({sizes.get(f, 0) / 1e3:.0f} kB)" for f in s["files"]))
        for note in s.get("notes", []):
            out.append(f"- {note}")
        out.append("")
    if m.get("not_used"):
        out += ["## Checked and not used", ""]
        for s in m["not_used"]:
            out += [f"### {s['name']}", "", f"- **Checked**: {s.get('checked', TODAY)}",
                    f"- **URL(s)**: {s.get('url')}", f"- **Finding**: {s.get('finding')}", ""]
    if m.get("absent_features"):
        out += ["## Features deliberately NOT bundled", "",
                "No citable geometry was found for these, so they are absent rather than invented.", ""]
        for a in m["absent_features"]:
            out.append(f"- **{a['name']}**: {a['reason']}")
        out.append("")
    (OUT / "SOURCES.md").write_text("\n".join(out), encoding="utf-8")


# --- HTTP ------------------------------------------------------------------------------------


def http_get(url: str, *, params: dict | None = None, timeout: float = 120, retries: int = 3,
             stream_to: Path | None = None) -> requests.Response | Path:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout, stream=stream_to is not None)
            r.raise_for_status()
            if stream_to is None:
                return r
            with open(stream_to, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
            return stream_to
        except Exception as exc:  # network errors are retried, then raised
            last = exc
            log(f"  attempt {attempt + 1}/{retries} failed for {url[:120]}: {type(exc).__name__}: {str(exc)[:160]}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url[:200]} ({last})")


def overpass(query: str) -> dict[str, Any]:
    last = None
    for endpoint in OVERPASS:
        for attempt in range(2):
            try:
                r = requests.post(endpoint, data={"data": query}, headers=UA, timeout=300)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    data = r.json()
                    data["_endpoint"] = endpoint
                    return data
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            log(f"  overpass {endpoint} attempt {attempt + 1} failed: {str(last)[:160]}")
            time.sleep(15)
    raise RuntimeError(f"Overpass failed on every endpoint: {last}")


def osm_geometry(el: dict[str, Any]):
    """Shapely geometry for an Overpass `out geom` element, or None."""
    if el["type"] == "node":
        return Point(el["lon"], el["lat"])
    if el["type"] == "way":
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry") or []]
        if len(coords) < 2:
            return None
        if len(coords) >= 4 and coords[0] == coords[-1]:
            poly = Polygon(coords)
            return poly if poly.is_valid else shapely.make_valid(poly)
        return LineString(coords)
    if el["type"] == "relation":
        outers, inners = [], []
        for mem in el.get("members") or []:
            if mem.get("type") != "way" or not mem.get("geometry"):
                continue
            coords = [(p["lon"], p["lat"]) for p in mem["geometry"] if p]
            if len(coords) < 2:
                continue
            (inners if mem.get("role") == "inner" else outers).append(LineString(coords))
        if not outers:
            return None
        outer_polys = list(polygonize(unary_union(outers)))
        if not outer_polys:
            return None
        geom = unary_union(outer_polys)
        if inners:
            inner_polys = list(polygonize(unary_union(inners)))
            if inner_polys:
                geom = geom.difference(unary_union(inner_polys))
        return geom if geom.is_valid else shapely.make_valid(geom)
    return None


def write_layer(name: str, kind: str, features: list[dict[str, Any]], *, source_id: str,
                regions: list[str], restricted: bool = False, precision: int = 5) -> Path:
    (OUT / "layers").mkdir(parents=True, exist_ok=True)
    path = OUT / "layers" / f"{name}.geojson"
    out_feats = []
    for f in features:
        g = f["geometry"]
        g = shapely.set_precision(g, 10 ** -precision) if g.geom_type != "Point" else g
        if g.is_empty:
            continue
        out_feats.append({"type": "Feature", "geometry": mapping(g), "properties": f["properties"]})
    payload = {"type": "FeatureCollection",
               "ghosttrace": {"layer": name, "kind": kind, "source_id": source_id,
                              "regions": regions, "restricted": restricted,
                              "written": datetime.now(timezone.utc).isoformat(timespec="seconds")},
               "features": out_feats}
    path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    log(f"  wrote {path.relative_to(REPO)}: {len(out_feats)} features, {path.stat().st_size / 1e3:.0f} kB")
    return path


# --- land ------------------------------------------------------------------------------------

NE_LAND_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
NE_ISLANDS_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_minor_islands.zip"


def _read_shapefile_zip(url: str, cache: Path) -> list[Any]:
    """Polygons from a zipped shapefile, read with a minimal pure-python reader.

    Natural Earth land files are polygon shapefiles (shape type 5). Reading
    them directly avoids a GDAL/fiona dependency for one file format.
    """
    import struct

    if not cache.exists():
        http_get(url, timeout=300, stream_to=cache)
    with zipfile.ZipFile(cache) as z:
        shp_name = next(n for n in z.namelist() if n.endswith(".shp"))
        data = z.read(shp_name)
    pos = 100
    polys = []
    while pos < len(data):
        _, length = struct.unpack(">ii", data[pos:pos + 8])
        content = data[pos + 8:pos + 8 + length * 2]
        pos += 8 + length * 2
        stype = struct.unpack("<i", content[:4])[0]
        if stype != 5:
            continue
        num_parts, num_points = struct.unpack("<ii", content[36:44])
        parts = struct.unpack(f"<{num_parts}i", content[44:44 + 4 * num_parts])
        pts_off = 44 + 4 * num_parts
        pts = np.frombuffer(content[pts_off:pts_off + 16 * num_points], dtype="<f8").reshape(-1, 2)
        rings = [pts[parts[i]:(parts[i + 1] if i + 1 < num_parts else num_points)] for i in range(num_parts)]
        # Shapefile rings: clockwise = outer, counter-clockwise = hole.
        shells, holes = [], []
        for r in rings:
            if len(r) < 4:
                continue
            ring = shapely.LinearRing(r)
            (holes if ring.is_ccw else shells).append(Polygon(r))
        if not shells:
            continue
        geom = unary_union(shells)
        if holes:
            geom = geom.difference(unary_union(holes))
        polys.append(geom)
    return polys


def fetch_land(m: dict[str, Any], regions: list[str], cache_dir: Path) -> dict[str, Any]:
    log("land: Natural Earth 10 m land + minor islands")
    ne = _read_shapefile_zip(NE_LAND_URL, cache_dir / "ne_10m_land.zip")
    ne += _read_shapefile_zip(NE_ISLANDS_URL, cache_dir / "ne_10m_minor_islands.zip")
    ne_tree = shapely.STRtree(ne)
    lands = {}
    method = {}
    for region in regions:
        s, w, n, e = padded(region)
        clip = box(w, s, e, n)
        ne_clip = unary_union([g.intersection(clip) for g in (ne[i] for i in ne_tree.query(clip))])
        log(f"  {region}: OpenStreetMap natural=coastline")
        land = None
        try:
            data = overpass(f"[out:json][timeout:240];way[\"natural\"=\"coastline\"]({s},{w},{n},{e});out geom;")
            lines, islands = [], []
            for el in data["elements"]:
                coords = [(p["lon"], p["lat"]) for p in el.get("geometry") or []]
                if len(coords) < 2:
                    continue
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    islands.append(Polygon(coords))
                lines.append(LineString(coords))
            noded = unary_union([l.intersection(clip) for l in lines] + [clip.boundary])
            pieces = list(polygonize(noded))
            keep = []
            ne_prepared = ne_clip
            shapely.prepare(ne_prepared)
            for p in pieces:
                if p.area <= 0:
                    continue
                overlap = p.intersection(ne_prepared).area / p.area
                inside_island = any(p.representative_point().within(i) for i in islands if i.is_valid)
                if overlap >= 0.5 or inside_island:
                    keep.append(p)
            osm_land = unary_union(keep).buffer(0)
            ne_area, osm_area = ne_clip.area, osm_land.area
            ratio = osm_area / ne_area if ne_area else float("inf")
            log(f"    OSM land area / Natural Earth land area = {ratio:.3f} ({len(lines)} coastline ways)")
            if 0.8 <= ratio <= 1.25:
                land = osm_land
                method[region] = (f"OSM coastline polygonised in the padded box, each piece kept as land "
                                  f"when >=50% covered by Natural Earth 10 m land or inside a closed OSM "
                                  f"island ring; area ratio to Natural Earth {ratio:.3f}")
            else:
                method[region] = f"Natural Earth only: OSM build failed the area sanity check (ratio {ratio:.3f})"
        except Exception as exc:
            method[region] = f"Natural Earth only: OSM coastline fetch failed ({type(exc).__name__}: {str(exc)[:120]})"
        if land is None:
            land = ne_clip
        lands[region] = land
        geoms = list(land.geoms) if land.geom_type == "MultiPolygon" else [land]
        feats = [{"geometry": g, "properties": {"name": None, "source": "OpenStreetMap + Natural Earth"
                                                 if "OSM coastline" in method[region] else "Natural Earth",
                                                 "source_id": "osm_coastline" if "OSM coastline" in method[region] else "natural_earth_land",
                                                 "geometry_quality": "mapped_polygon"}}
                 for g in geoms if g.area > 0]
        write_layer(f"land_{region}", "land", feats, source_id="osm_coastline", regions=[region])
    upsert_source(m, {
        "id": "natural_earth_land", "name": "Natural Earth 10m land and minor islands (v5)",
        "url": f"{NE_LAND_URL} ; {NE_ISLANDS_URL}", "licence": "Public domain (Natural Earth terms of use)",
        "redistributable": True, "used_for": "deciding which side of the OSM coastline is land; fallback land mask",
        "clipped": "; ".join(f"{r}: {method[r]}" for r in regions),
        "files": [f"layers/land_{r}.geojson" for r in regions]})
    upsert_source(m, {
        "id": "osm_coastline", "name": "OpenStreetMap natural=coastline (Overpass API)",
        "url": "https://overpass-api.de/api/interpreter", "licence": "ODbL 1.0, (c) OpenStreetMap contributors",
        "redistributable": True, "used_for": "land mask for particle stranding and coastline for curated nesting beaches",
        "clipped": f"coastline ways in each region box padded by {CLIP_PAD_DEG} deg",
        "files": [f"layers/land_{r}.geojson" for r in regions],
        "notes": ["Land polygons are simplified to 1e-5 deg (~1 m) coordinate precision.",
                  "Coastal lagoons that OSM maps as water inside the coastline (not as sea) are LAND in "
                  "this mask; Chilika Lake is one. A particle cannot enter them, and protected areas "
                  "lying wholly inside them (Chilika (Nalabana) WLS) are dropped by the marine filter."]})
    return lands


def load_land(region: str):
    path = OUT / "layers" / f"land_{region}.geojson"
    data = json.loads(path.read_text(encoding="utf-8"))
    return unary_union([shape(f["geometry"]) for f in data["features"]])


# --- protected areas -------------------------------------------------------------------------------


def fetch_protected(m: dict[str, Any], regions: list[str]) -> None:
    feats_all: dict[str, list] = {}
    kept_names = []
    for region in regions:
        s, w, n, e = cfg.REGIONS[region]
        log(f"protected areas: OSM in {region}")
        q = (f"[out:json][timeout:240];("
             f"relation[\"boundary\"~\"^(protected_area|national_park)$\"]({s},{w},{n},{e});"
             f"way[\"boundary\"~\"^(protected_area|national_park)$\"]({s},{w},{n},{e});"
             f"relation[\"leisure\"=\"nature_reserve\"]({s},{w},{n},{e});"
             f"way[\"leisure\"=\"nature_reserve\"]({s},{w},{n},{e});"
             f"node[\"leisure\"=\"nature_reserve\"]({s},{w},{n},{e}););out geom;")
        data = overpass(q)
        land = load_land(region)
        coast = land.boundary
        shapely.prepare(land)
        feats, seen = [], set()
        way_names = {el.get("tags", {}).get("name") for el in data["elements"] if el["type"] != "node"}
        for el in data["elements"]:
            key = (el["type"], el["id"])
            if key in seen:
                continue
            seen.add(key)
            tags = el.get("tags") or {}
            if el["type"] == "node" and tags.get("name") and any(
                    tags["name"] in (wn or "") for wn in way_names):
                continue  # a polygon for the same reserve is present
            g = osm_geometry(el)
            if g is None or g.is_empty:
                continue
            if g.geom_type in ("LineString", "MultiLineString"):
                continue  # an unclosed boundary way is not an area
            # Keep only areas a net could reach: touching the sea, or within ~1 km of the coast.
            touches_sea = not land.contains(g)
            near_coast = g.distance(coast) <= 0.01
            if not (touches_sea or near_coast):
                continue
            quality = "point" if g.geom_type == "Point" else "mapped_polygon"
            feats.append({"geometry": g, "properties": {
                "name": tags.get("name") or tags.get("name:en"),
                "designation": tags.get("protection_title") or tags.get("boundary") or tags.get("leisure"),
                "protect_class": tags.get("protect_class"),
                "osm_id": f"{el['type']}/{el['id']}",
                "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
                "source": "OpenStreetMap", "source_id": "osm_protected",
                "geometry_quality": quality,
                "marine_or_coastal_basis": "polygon extends beyond the land mask" if touches_sea
                else "within ~1 km of the coastline"}})
            kept_names.append(f"{tags.get('name')} ({el['type']}/{el['id']})")
        feats_all[region] = feats
        write_layer(f"protected_osm_{region}", "protected_area", feats, source_id="osm_protected",
                    regions=[region])
    upsert_source(m, {
        "id": "osm_protected", "name": "OpenStreetMap protected areas (Overpass API)",
        "url": "https://overpass-api.de/api/interpreter",
        "licence": "ODbL 1.0, (c) OpenStreetMap contributors", "redistributable": True,
        "used_for": "protected_area layer (habitat context, drift impacts)",
        "clipped": ("boundary=protected_area|national_park and leisure=nature_reserve intersecting each "
                    "region box; kept only when the area extends beyond the land mask or lies within "
                    "~1 km (0.01 deg) of the coastline, so inland bird sanctuaries are dropped"),
        "files": [f"layers/protected_osm_{r}.geojson" for r in regions],
        "notes": ["Kept: " + "; ".join(kept_names)]})
    absent = [a for a in m.get("absent_features", []) if a["name"] not in (
        "Gulf of Mannar Biosphere Reserve", "Palk Bay Dugong Conservation Reserve (legal boundary)")]
    absent += [
        {"name": "Gulf of Mannar Biosphere Reserve",
         "reason": "no polygon in OpenStreetMap (Overpass name search, " + TODAY + ") and no downloadable "
                   "boundary found; the Marine National Park polygon inside it is bundled"},
        {"name": "Palk Bay Dugong Conservation Reserve (legal boundary)",
         "reason": "not in OpenStreetMap; the notification G.O.(Ms) No.165 (21.09.2022) was not found "
                   "online with boundary coordinates. Only an approximate coastal extent is bundled "
                   "(layer dugong_curated)."}]
    m["absent_features"] = absent


# --- reefs and seagrass (UNEP-WCMC) -------------------------------------------------------------------

WCMC_SERVICE = "https://data-gis.unep-wcmc.org/server/rest/services/HabitatsAndBiotopes"
WCMC_LICENCE = ("UNEP-WCMC General Data License (excluding WDPA), https://www.unep-wcmc.org/en/general-data-license : "
                "no commercial use without permission (clause 2); no redistribution (clause 3); may be "
                "published only in non-downloadable form with citation (clauses 4-5)")


def _arcgis_query(layer_url: str, bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    s, w, n, e = bbox
    feats, offset = [], 0
    while True:
        params = {"where": "1=1", "geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope",
                  "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326,
                  "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000}
        r = http_get(f"{layer_url}/query", params=params, timeout=180)
        data = r.json()
        batch = data.get("features") or []
        feats += batch
        if len(batch) < 1000 and not data.get("exceededTransferLimit"):
            break
        offset += len(batch)
    return feats


def fetch_wcmc(m: dict[str, Any], regions: list[str], *, what: str) -> None:
    if what == "reef":
        service, sid, name, kind = ("Global_Distribution_of_Coral_Reefs", "wcmc_reefs",
                                    "UNEP-WCMC Global Distribution of Warm-water Coral Reefs v4.1", "reef")
        citation = ("UNEP-WCMC, WorldFish Centre, WRI, TNC (2021). Global distribution of warm-water coral "
                    "reefs, compiled from multiple sources including the Millennium Coral Reef Mapping "
                    "Project. Version 4.1. Cambridge (UK): UN Environment World Conservation Monitoring "
                    "Centre. https://doi.org/10.34892/t2wk-5t34")
    else:
        service, sid, name, kind = ("Global_Distribution_of_Seagrasses", "wcmc_seagrass",
                                    "UNEP-WCMC Global Distribution of Seagrasses", "seagrass")
        citation = ("UNEP-WCMC, Short FT (2021). Global distribution of seagrasses (version 7.1). Seventh "
                    "update to the data layer used in Green and Short (2003). Cambridge (UK): UN Environment "
                    "World Conservation Monitoring Centre. https://doi.org/10.34892/x6r3-d211")
    base = f"{WCMC_SERVICE}/{service}/FeatureServer"
    info = http_get(f"{base}?f=json", timeout=60).json()
    poly_layers = [l for l in info.get("layers", []) if l.get("geometryType") == "esriGeometryPolygon"]
    point_layers = [l for l in info.get("layers", []) if l.get("geometryType") == "esriGeometryPoint"]
    files, counts = [], {}
    for region in regions:
        bbox = padded(region)
        feats = []
        for lyr, quality in [(l, "mapped_polygon") for l in poly_layers] + [(l, "point") for l in point_layers]:
            log(f"{what}: {lyr['name']} in {region}")
            for f in _arcgis_query(f"{base}/{lyr['id']}", bbox):
                if not f.get("geometry"):
                    continue
                g = shape(f["geometry"])
                if not g.is_valid:
                    g = shapely.make_valid(g)
                s, w, n, e = bbox
                g = g.intersection(box(w, s, e, n)) if quality != "point" else g
                if g.is_empty:
                    continue
                p = f.get("properties") or {}
                feats.append({"geometry": g, "properties": {
                    # UNEP-WCMC uses the literal "Not Reported" for an unnamed feature; kept as null.
                    "name": (None if str(p.get("NAME") or p.get("name") or "").strip().lower()
                             in ("", "not reported", "none") else (p.get("NAME") or p.get("name"))),
                    "wcmc_layer": lyr["name"], "metadata_id": p.get("METADATA_I") or p.get("metadata_i"),
                    "family": p.get("FAMILY") or p.get("family"), "location": p.get("LOC_DEF") or p.get("loc_def"),
                    "source": name, "source_id": sid, "geometry_quality": quality}})
        counts[region] = len(feats)
        # Written even when empty: an empty layer records "searched, none mapped here", which
        # habitat.py reports as assessed, unlike a missing file ("not assessed").
        path = write_layer(f"{kind}_wcmc_{region}", kind, feats, source_id=sid, regions=[region], restricted=True)
        files.append(str(path.relative_to(OUT)))
    upsert_source(m, {
        "id": sid, "name": name, "url": base, "licence": WCMC_LICENCE, "redistributable": False,
        "citation": citation, "used_for": f"{kind} layer (habitat context, drift impacts)",
        "clipped": ("FeatureServer query by envelope for each region box padded by "
                    f"{CLIP_PAD_DEG} deg, polygons intersected with the box; features per region: {counts}"),
        "files": files,
        "notes": ["Fetched to this machine only and git-ignored; a fresh clone must run the fetcher.",
                  "DeepEcho / SIH26057 is a non-commercial research prototype; any commercial use needs "
                  "UNEP-WCMC's written permission."]})


def fetch_osm_reefs(m: dict[str, Any], regions: list[str]) -> None:
    files = []
    for region in regions:
        s, w, n, e = cfg.REGIONS[region]
        log(f"reef: OSM natural=reef in {region}")
        data = overpass(f"[out:json][timeout:180];nwr[\"natural\"=\"reef\"]({s},{w},{n},{e});out geom;")
        feats = []
        for el in data["elements"]:
            g = osm_geometry(el)
            if g is None or g.is_empty:
                continue
            tags = el.get("tags") or {}
            feats.append({"geometry": g, "properties": {
                "name": tags.get("name"), "reef_type": tags.get("reef") or tags.get("subsea"),
                "osm_id": f"{el['type']}/{el['id']}", "source": "OpenStreetMap", "source_id": "osm_reefs",
                "geometry_quality": "point" if g.geom_type == "Point" else
                ("mapped_polygon" if g.geom_type in ("Polygon", "MultiPolygon") else "mapped_line"),
                "note": "natural=reef in OSM includes rocky and sand reefs, not only coral"}})
        path = write_layer(f"reef_osm_{region}", "reef", feats, source_id="osm_reefs", regions=[region])
        files.append(str(path.relative_to(OUT)))
    upsert_source(m, {
        "id": "osm_reefs", "name": "OpenStreetMap natural=reef (Overpass API)",
        "url": "https://overpass-api.de/api/interpreter", "licence": "ODbL 1.0, (c) OpenStreetMap contributors",
        "redistributable": True, "used_for": "supplementary reef layer, present even without the UNEP-WCMC fetch",
        "clipped": "natural=reef elements intersecting each region box", "files": files,
        "notes": ["Very sparse in both regions; not a substitute for the UNEP-WCMC reef layer."]})
    upsert_not_used(m, {"name": "Allen Coral Atlas (CC BY 4.0)", "url": "https://allencoralatlas.org/atlas/",
                        "finding": ("the preferred reef source, but downloads require a registered login "
                                    "(Allen Coral Atlas FAQ, https://allencoralatlas.org/resources/); no "
                                    "no-login programmatic access was found, so it was not used")})


# --- curated: turtle nesting and dugong -------------------------------------------------------------


def _coast_substring(land, anchor: Point, start_m: float, end_m: float, lat0: float, lon0: float,
                     direction_hint_deg: float):
    """A stretch of coastline measured along the shore from the point nearest an anchor.

    start_m / end_m are distances along the coast from the anchor, in the
    direction whose initial bearing is closest to direction_hint_deg.
    """
    local_land = to_local(land, lat0, lon0)
    a = to_local(anchor, lat0, lon0)
    rings = []
    for poly in (local_land.geoms if local_land.geom_type == "MultiPolygon" else [local_land]):
        rings.append(LineString(poly.exterior.coords))
    ring = min(rings, key=lambda r: r.distance(a))
    L = ring.length
    s0 = ring.project(a)

    def pt(dist):
        return ring.interpolate((s0 + dist) % L)

    fwd = pt(500.0)
    p0 = ring.interpolate(s0)
    bearing = math.degrees(math.atan2(fwd.x - p0.x, fwd.y - p0.y)) % 360
    diff = abs((bearing - direction_hint_deg + 180) % 360 - 180)
    sign = 1.0 if diff <= 90 else -1.0
    steps = np.linspace(start_m, end_m, 60)
    line = LineString([pt(sign * d) for d in steps])
    return to_lonlat(line, lat0, lon0), float(ring.distance(a))


def fetch_curated(m: dict[str, Any], regions: list[str]) -> None:
    turtle_feats, dugong_feats = {r: [] for r in regions}, {r: [] for r in regions}
    notes = []
    if "odisha_coast" in regions:
        region = "odisha_coast"
        land = load_land(region)
        log("turtle nesting: OSM anchors (Rushikulya river, Devi river, Gahirmatha sanctuary)")
        data = overpass("[out:json][timeout:180];("
                        "way[\"waterway\"=\"river\"][\"name\"~\"Rushikulya|Devi River\"](19.0,84.7,21.0,87.6);"
                        "way[\"name\"=\"Gahirmatha (Marine) WLS\"](19.0,84.7,21.0,87.6););out geom;")
        rivers: dict[str, list] = {}
        gahir = None
        for el in data["elements"]:
            tags = el.get("tags") or {}
            g = osm_geometry(el)
            if tags.get("waterway") == "river":
                rivers.setdefault(tags.get("name"), []).append((el["id"], g))
            elif tags.get("name") == "Gahirmatha (Marine) WLS":
                gahir = (el["id"], g)

        def river_mouth(name):
            ways = rivers.get(name) or []
            best = None
            for wid, g in ways:
                for c in (g.coords[0], g.coords[-1]):
                    d = Point(c).distance(land.boundary)
                    if best is None or d < best[0]:
                        best = (d, Point(c), wid)
            return best

        # Rushikulya: Pandav, Choudhury & Kar (1994), Marine Turtle Newsletter 67:15-16:
        # "The rookery extends 6 km from Puruna Bandha village (1 km north of the Rushikulya River
        # mouth) to Kantiagada village."
        rm = river_mouth("Rushikulya")
        if rm and rm[0] < 0.02:
            lat0, lon0 = rm[1].y, rm[1].x
            line, snap = _coast_substring(land, rm[1], 1000.0, 7000.0, lat0, lon0, direction_hint_deg=45.0)
            turtle_feats[region].append({"geometry": line, "properties": {
                "name": "Rushikulya olive ridley mass-nesting beach", "species": "Lepidochelys olivacea",
                "source": "curated (OpenStreetMap anchor + Pandav et al. 1994)", "source_id": "curated_turtle",
                "geometry_quality": "approximate_line",
                "citation": ("Pandav B, Choudhury BC, Kar CS (1994). Discovery of a new sea turtle rookery in "
                             "Orissa, India. Marine Turtle Newsletter 67:15-16. "
                             "http://www.seaturtle.org/mtn/archives/mtn67/mtn67p15.shtml"),
                "construction": (f"coastline from 1 km to 7 km north-east of the Rushikulya river mouth (end "
                                 f"vertex of OSM way/{rm[2]}, snapped {snap:.0f} m to the coast), per the "
                                 "cited extent '6 km from Puruna Bandha village (1 km north of the river "
                                 "mouth) to Kantiagada village'. Village positions were not used."),
            }})
        else:
            notes.append("Rushikulya river mouth not found near the coast in OSM; nesting beach omitted")
        # Devi: NCCR / ICMAM-PD document: "Devi river mouth (Jatadhar Muhana to Kadera river mouth),
        # Cuttack district, 100 km South of Gahirmatha (Kar, 1982)". The two named mouths are not in OSM,
        # so only the Devi mouth itself is placed, as an approximate point.
        dm = river_mouth("Devi River")
        if dm and dm[0] < 0.03:
            turtle_feats[region].append({"geometry": dm[1], "properties": {
                "name": "Devi river mouth olive ridley mass-nesting site", "species": "Lepidochelys olivacea",
                "source": "curated (OpenStreetMap anchor + NCCR/ICMAM-PD)", "source_id": "curated_turtle",
                "geometry_quality": "approximate",
                "citation": ("ICMAM Project Directorate / NCCR, Ministry of Earth Sciences. Critical habitat "
                             "information system: Bhitarkanika and Gahirmatha, section on rookeries: 'Devi "
                             "river mouth (Jatadhar Muhana to Kadera river mouth)'. "
                             "https://www.nccr.gov.in/sites/default/files/Gahirmatha.PDF"),
                "construction": (f"seaward end vertex of OSM way/{dm[2]} (Devi River). The rookery's stated "
                                 "extent (Jatadhar Muhana to Kadera river mouth) could not be positioned: "
                                 "neither mouth is named in OSM. Treat as a point marker, not an extent."),
            }})
        else:
            notes.append("Devi river mouth not found near the coast in OSM; nesting site omitted")
        # Gahirmatha: the marine sanctuary was notified for the rookery. The NCCR document places the
        # rookery on '35-40 km of the coastline from Maipura river in the north and Hansua river mouth
        # in the south' and mass nesting on Nasi island since 1990; neither river mouth nor Nasi island
        # is named in OSM, so the beach is approximated as the coastline inside the OSM sanctuary polygon.
        if gahir:
            coast = land.boundary.intersection(gahir[1].buffer(0.005))
            if not coast.is_empty:
                turtle_feats[region].append({"geometry": coast, "properties": {
                    "name": "Gahirmatha olive ridley mass-nesting coast", "species": "Lepidochelys olivacea",
                    "source": "curated (OpenStreetMap sanctuary polygon + NCCR/ICMAM-PD)",
                    "source_id": "curated_turtle", "geometry_quality": "approximate_line",
                    "citation": ("ICMAM Project Directorate / NCCR, Ministry of Earth Sciences. Gahirmatha "
                                 "critical habitat document: rookery on 35-40 km of coastline from the Maipura "
                                 "river to the Hansua river mouth; mass nesting on Nasi island from 1990. "
                                 "https://www.nccr.gov.in/sites/default/files/Gahirmatha.PDF"),
                    "construction": (f"land-mask coastline within ~500 m of OSM way/{gahir[0]} 'Gahirmatha "
                                     "(Marine) WLS'. The sanctuary coast is longer than the active nesting "
                                     "beach, and the barrier islands shift; this over-covers rather than "
                                     "misses the rookery."),
                }})
        else:
            notes.append("Gahirmatha (Marine) WLS polygon not found in OSM; nesting coast omitted")
        write_layer(f"turtle_nesting_curated_{region}", "turtle_nesting", turtle_feats[region],
                    source_id="curated_turtle", regions=[region])

    if "gulf_of_mannar_palk_bay" in regions:
        region = "gulf_of_mannar_palk_bay"
        land = load_land(region)
        log("dugong: Palk Bay Dugong Conservation Reserve coastal extent (OSM anchors)")
        data = overpass("[out:json][timeout:180];(node[\"place\"][\"name\"~\"^(Adirampattinam|Ammapattinam)$\"]"
                        "(9.8,78.9,10.6,79.6););out;")
        anchors = {}
        for el in data["elements"]:
            nm = el["tags"]["name"]
            if nm not in anchors or el["tags"].get("place") in ("town", "village"):
                anchors[nm] = (el["id"], Point(el["lon"], el["lat"]), el["tags"].get("place"))
        if {"Adirampattinam", "Ammapattinam"} <= set(anchors):
            a_id, a_pt, _ = anchors["Adirampattinam"]
            b_id, b_pt, _ = anchors["Ammapattinam"]
            lat0, lon0 = (a_pt.y + b_pt.y) / 2, (a_pt.x + b_pt.x) / 2
            local_land = to_local(land, lat0, lon0)
            la, lb = to_local(a_pt, lat0, lon0), to_local(b_pt, lat0, lon0)
            ring = None
            for poly in (local_land.geoms if local_land.geom_type == "MultiPolygon" else [local_land]):
                r = LineString(poly.exterior.coords)
                if ring is None or r.distance(la) + r.distance(lb) < ring.distance(la) + ring.distance(lb):
                    ring = r
            sa, sb = ring.project(la), ring.project(lb)
            lo, hi = sorted((sa, sb))
            seg = substring(ring, lo, hi)
            if seg.length > 0.5 * ring.length:  # took the long way round the ring
                seg = linemerge([substring(ring, hi, ring.length), substring(ring, 0, lo)])
            line = to_lonlat(seg, lat0, lon0)
            dugong_feats[region].append({"geometry": line, "properties": {
                "name": "Palk Bay Dugong Conservation Reserve (approximate coastal extent)",
                "species": "Dugong dugon", "source": "curated (OpenStreetMap anchors + cited descriptions)",
                "source_id": "curated_dugong", "geometry_quality": "approximate_line",
                "citation": ("Government of Tamil Nadu, DIPR press release P.R. No. 1645, 21.09.2022: G.O.(Ms) "
                             "No.165 notifies a 448 sq km Dugong Conservation Reserve in Palk Bay covering the "
                             "coastal waters of Thanjavur and Pudukkottai districts "
                             "(https://dugong.cms.int/sites/default/files/2022-09-21%20PalkBay%20Dugong%20Reserve-Govt.%20Press%20Release.pdf); "
                             "Mongabay India (Sept 2025), 'Dugongs recovering, need cross-border efforts in "
                             "conservation': fishing hamlets 'along the Adirampattinam-Ammapattinam coastline "
                             "bordering the reserve' "
                             "(https://india.mongabay.com/2025/09/dugongs-recovering-need-cross-border-efforts-in-conservation/)"),
                "construction": (f"land-mask coastline between OSM node/{a_id} (Adirampattinam) and "
                                 f"node/{b_id} (Ammapattinam). This is the shore the reserve borders, NOT "
                                 "its seaward boundary, which is not public in machine-readable form; a "
                                 "net inside the reserve but far from this shore is not detected by it."),
            }})
        else:
            notes.append("Adirampattinam / Ammapattinam not found in OSM; dugong reserve extent omitted")
        write_layer(f"dugong_curated_{region}", "dugong", dugong_feats[region], source_id="curated_dugong",
                    regions=[region])

    upsert_source(m, {
        "id": "curated_turtle", "name": "Curated olive ridley mass-nesting sites (Odisha)",
        "url": "see per-feature citation", "licence": ("geometry derived from OpenStreetMap (ODbL 1.0); "
                                                        "extents from the cited publications"),
        "redistributable": True, "used_for": "turtle_nesting layer",
        "clipped": "built by fetch_curated(); every feature carries citation, construction and geometry_quality",
        "files": [f"layers/turtle_nesting_curated_{r}.geojson" for r in regions if r == "odisha_coast"],
        "notes": notes or ["all three rookeries placed"]})
    upsert_source(m, {
        "id": "curated_dugong", "name": "Curated dugong habitat (Palk Bay)",
        "url": "see per-feature citation", "licence": ("geometry derived from OpenStreetMap (ODbL 1.0); "
                                                        "description from the cited documents"),
        "redistributable": True, "used_for": "dugong layer",
        "clipped": "built by fetch_curated()",
        "files": [f"layers/dugong_curated_{r}.geojson" for r in regions if r == "gulf_of_mannar_palk_bay"],
        "notes": ["Seagrass (dugong forage habitat) comes from the UNEP-WCMC seagrass layer when fetched."]})


# --- harbours / fishing activity -----------------------------------------------------------------------


def fetch_gfw(m: dict[str, Any], regions: list[str], token: str) -> bool:
    """Apparent fishing effort from Global Fishing Watch, when a token is configured.

    Written against the published GFW API v3 4Wings report endpoint. It could
    not be exercised while building this module because no token was
    available, so it is best-effort: any failure falls back to OSM harbours
    and the manifest records the failure.
    """
    files = []
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=365)
    for region in regions:
        s, w, n, e = cfg.REGIONS[region]
        geojson = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        r = requests.post(
            "https://gateway.api.globalfishingwatch.org/v3/4wings/report",
            params={"spatial-resolution": "LOW", "temporal-resolution": "ENTIRE", "format": "JSON",
                    "datasets[0]": "public-global-fishing-effort:latest",
                    "date-range": f"{start},{end}"},
            json={"geojson": geojson}, headers={**UA, "Authorization": f"Bearer {token}"}, timeout=300)
        r.raise_for_status()
        path = OUT / "layers" / f"gfw_fishing_effort_{region}.json"
        path.write_text(r.text, encoding="utf-8")
        files.append(str(path.relative_to(OUT)))
    upsert_source(m, {"id": "gfw", "name": "Global Fishing Watch apparent fishing effort (API v3)",
                      "url": "https://gateway.api.globalfishingwatch.org/v3/4wings/report",
                      "licence": "CC BY-NC 4.0 (Global Fishing Watch terms)", "redistributable": False,
                      "used_for": "fishing exposure (raw report; not yet read by safety.py)",
                      "snapshot": f"{start} to {end}", "files": files})
    return True


def fetch_harbours(m: dict[str, Any], regions: list[str]) -> None:
    token = os.environ.get(cfg.GFW_API_TOKEN_ENV)
    gfw_state = "GFW_API_TOKEN not set: Global Fishing Watch not used; OSM harbours used for fishing exposure"
    if token:
        try:
            fetch_gfw(m, regions, token)
            gfw_state = "GFW_API_TOKEN set: Global Fishing Watch report fetched (OSM harbours also bundled)"
        except Exception as exc:
            gfw_state = f"GFW_API_TOKEN set but the GFW request failed ({type(exc).__name__}: {str(exc)[:120]}); OSM harbours used"
    files, counts = [], {}
    for region in regions:
        s, w, n, e = cfg.REGIONS[region]
        b = f"({s},{w},{n},{e})"
        log(f"harbours: OSM in {region}")
        q = (f"[out:json][timeout:240];(nwr[\"harbour\"]{b};nwr[\"seamark:type\"=\"harbour\"]{b};"
             f"nwr[\"landuse\"=\"harbour\"]{b};nwr[\"industrial\"=\"port\"]{b};nwr[\"leisure\"=\"marina\"]{b};"
             f"nwr[\"man_made\"=\"pier\"][\"name\"]{b};"
             f"nwr[\"name\"~\"fishing harbou?r|fishing port|fish landing|landing cent|jetty\",i]{b};);out center tags;")
        data = overpass(q)
        feats, seen = [], set()
        for el in data["elements"]:
            tags = el.get("tags") or {}
            if el["type"] == "node":
                lat, lon = el["lat"], el["lon"]
            else:
                c = el.get("center")
                if not c:
                    continue
                lat, lon = c["lat"], c["lon"]
            if tags.get("highway") or tags.get("building") and not tags.get("name"):
                continue
            key = (round(lat, 3), round(lon, 3))
            if key in seen:
                continue
            seen.add(key)
            name = tags.get("name") or tags.get("name:en")
            text = " ".join(str(v) for v in tags.values()).lower()
            fishing = "fish" in text
            feats.append({"geometry": Point(lon, lat), "properties": {
                "name": name, "fishing": fishing,
                "category": tags.get("seamark:harbour:category") or tags.get("harbour") or tags.get("industrial")
                or tags.get("landuse") or tags.get("leisure") or tags.get("man_made") or tags.get("amenity"),
                "osm_id": f"{el['type']}/{el['id']}", "source": "OpenStreetMap", "source_id": "osm_harbours",
                "geometry_quality": "point" if el["type"] == "node" else "approximate",
                "note": None if el["type"] == "node" else "centre of the OSM way/relation"}})
        counts[region] = len(feats)
        path = write_layer(f"harbour_osm_{region}", "harbour", feats, source_id="osm_harbours", regions=[region])
        files.append(str(path.relative_to(OUT)))
    upsert_source(m, {
        "id": "osm_harbours", "name": "OpenStreetMap harbours, ports, fishing ports and named jetties (Overpass API)",
        "url": "https://overpass-api.de/api/interpreter", "licence": "ODbL 1.0, (c) OpenStreetMap contributors",
        "redistributable": True, "used_for": "harbour layer: propeller hazard and fishing exposure proxy",
        "clipped": f"elements in each region box; counts {counts}; ways/relations reduced to their centre",
        "files": files,
        "notes": [gfw_state,
                  "OSM coverage of Indian fish landing centres is incomplete: absence of a harbour point is "
                  "not evidence of no fishing activity."]})
    m["fishing_activity_source"] = gfw_state


# --- bathymetry ----------------------------------------------------------------------------------------

ETOPO_OPENDAP = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/15s/15s_surface_elev_netcdf/"


def fetch_bathymetry(m: dict[str, Any], regions: list[str]) -> None:
    import netCDF4

    (OUT / "bathymetry").mkdir(parents=True, exist_ok=True)
    files = []
    for region in regions:
        s, w, n, e = padded(region, 0.1)
        # ETOPO 2022 15" tiles are 15 x 15 degrees, named by their NORTH-WEST corner.
        north_edge = int(math.ceil(n / 15.0) * 15)
        west_edge = int(math.floor(w / 15.0) * 15)
        if s < north_edge - 15:
            raise RuntimeError(f"{region} spans two ETOPO tiles; not handled")
        name = f"ETOPO_2022_v1_15s_N{north_edge:02d}E{west_edge:03d}_surface.nc"
        url = ETOPO_OPENDAP + name
        log(f"bathymetry: {name} for {region}")
        with netCDF4.Dataset(url) as ds:
            lat = ds.variables["lat"][:]
            lon = ds.variables["lon"][:]
            iy = np.where((lat >= s) & (lat <= n))[0]
            ix = np.where((lon >= w) & (lon <= e))[0]
            z = ds.variables["z"][iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
            attrs = {k: str(ds.getncattr(k)) for k in ds.ncattrs()}
        out = OUT / "bathymetry" / f"etopo2022_15s_{region}.nc"
        with netCDF4.Dataset(out, "w") as nc:
            nc.createDimension("lat", len(iy))
            nc.createDimension("lon", len(ix))
            vlat = nc.createVariable("lat", "f8", ("lat",))
            vlon = nc.createVariable("lon", "f8", ("lon",))
            vlat[:], vlon[:] = lat[iy], lon[ix]
            vlat.units, vlon.units = "degrees_north", "degrees_east"
            vz = nc.createVariable("z", "i2", ("lat", "lon"), zlib=True, complevel=6)
            vz[:] = np.clip(np.round(np.ma.filled(z, 0)), -32767, 32767).astype("i2")
            vz.units = "m"
            vz.long_name = "surface elevation relative to EGM2008 (negative below sea level), rounded to 1 m"
            nc.title = "NOAA ETOPO 2022 15 arc-second surface elevation, clipped for GhostTrace"
            nc.source_url = url
            nc.accessed = TODAY
            nc.region = region
            nc.clip_box_south_west_north_east = f"{s},{w},{n},{e}"
            nc.original_description = attrs.get("GDAL_TIFFTAG_IMAGEDESCRIPTION", "")
        files.append(str(out.relative_to(OUT)))
        log(f"  wrote {out.relative_to(REPO)} {out.stat().st_size / 1e3:.0f} kB, z range {float(z.min()):.0f}..{float(z.max()):.0f} m")
    upsert_source(m, {
        "id": "etopo_2022", "name": "NOAA NCEI ETOPO 2022 Global Relief Model, 15 arc-second, surface elevation",
        "url": ETOPO_OPENDAP, "licence": "Public domain (US Government work, NOAA NCEI)", "redistributable": True,
        "citation": ("NOAA National Centers for Environmental Information (2022). ETOPO 2022 15 Arc-Second "
                     "Global Relief Model. doi:10.25921/fd45-gt74"),
        "used_for": "seabed depth for the diver brief and propeller hazard when the survey has no depth",
        "clipped": "OPeNDAP index subset of each region box padded by 0.1 deg; elevation rounded to whole metres (int16)",
        "files": files})


# --- currents ----------------------------------------------------------------------------------------------

HYCOM_NCSS = "https://ncss.hycom.org/thredds/ncss/grid/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"


def _hycom_day(region: str, day: datetime, tmpdir: Path, server_end: datetime | None = None) -> dict[str, Any]:
    import netCDF4

    s, w, n, e = padded(region, 0.1)
    t0 = day
    # The last requested day may be partial: the server's forecast ends mid-day.
    t1 = day + timedelta(hours=21)
    if server_end is not None and server_end < t1:
        t1 = server_end
    params = {"var": "water_u,water_v", "north": n, "south": s, "west": w, "east": e,
              "time_start": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "time_end": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "accept": "netcdf4"}
    path = tmpdir / f"{region}_{day:%Y%m%d}.nc"
    url = f"{HYCOM_NCSS}?{urlencode(params)}"
    for attempt in range(4):
        try:
            http_get(url, timeout=900, retries=1, stream_to=path)
            with netCDF4.Dataset(path) as ds:
                tvar = ds.variables["time"]
                times = netCDF4.num2date(tvar[:], tvar.units, only_use_cftime_datetimes=False,
                                         only_use_python_datetimes=True)
                rvar = ds.variables["time_run"]
                runs = netCDF4.num2date(rvar[:], rvar.units, only_use_cftime_datetimes=False,
                                        only_use_python_datetimes=True)
                t1var = ds.variables.get("time1")
                if t1var is not None and not np.allclose(t1var[:], tvar[:]):
                    raise RuntimeError("water_u and water_v time axes differ")
                out = {"times": list(times), "runs": list(runs), "depth": ds.variables["depth"][:].astype(float),
                       "lat": ds.variables["lat"][:].astype(float), "lon": ds.variables["lon"][:].astype(float),
                       "u": ds.variables["water_u"][:], "v": ds.variables["water_v"][:],
                       "created_on": str(getattr(ds, "created_on", "")),
                       "generating_model": str(getattr(ds, "generating_model", "")),
                       "institution": str(getattr(ds, "institution", ""))}
            path.unlink(missing_ok=True)
            log(f"  currents {region} {day:%Y-%m-%d}: {len(out['times'])} steps, run(s) "
                f"{sorted({r.strftime('%Y-%m-%dT%H') for r in out['runs']})}")
            return out
        except Exception as exc:
            log(f"  currents {region} {day:%Y-%m-%d} attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc)[:160]}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"HYCOM day {day:%Y-%m-%d} for {region} failed after 4 attempts")


def fetch_currents(m: dict[str, Any], regions: list[str], *, days: int, start: datetime | None,
                   workers: int = 3) -> None:
    import netCDF4

    log("currents: HYCOM ESPC-D-V02 FMRC best time series (ncss.hycom.org)")
    r = http_get(f"{HYCOM_NCSS}/dataset.xml", timeout=120)
    text = r.text
    begin = datetime.fromisoformat(text.split("<begin>")[1].split("</begin>")[0].replace("Z", "+00:00"))
    end = datetime.fromisoformat(text.split("<end>")[1].split("</end>")[0].replace("Z", "+00:00"))
    if start is None:
        start = (begin + timedelta(hours=23)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = start.astimezone(timezone.utc)
    # --days counts calendar days from --start, inclusive. The final day may be
    # partial (the server's forecast ends mid-day); its steps up to the server
    # end are kept rather than dropping the whole day.
    last_day = min(start + timedelta(days=days - 1), end.replace(hour=0, minute=0, second=0, microsecond=0))
    day_list = []
    d = start
    while d <= last_day:
        day_list.append(d)
        d += timedelta(days=1)
    log(f"  server window {begin.isoformat()} .. {end.isoformat()}; fetching {day_list[0]:%Y-%m-%d} .. "
        f"{day_list[-1]:%Y-%m-%d} ({len(day_list)} days) for {regions}")
    (OUT / "currents").mkdir(parents=True, exist_ok=True)
    files, snapshots = [], {}
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for region in regions:
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                chunks = list(pool.map(lambda day: _hycom_day(region, day, tmpdir, end), day_list))
            times = [t for c in chunks for t in c["times"]]
            runs = [t for c in chunks for t in c["runs"]]
            order = np.argsort(np.array([t.timestamp() for t in times]))
            u = np.ma.concatenate([c["u"] for c in chunks], axis=0)[order]
            v = np.ma.concatenate([c["v"] for c in chunks], axis=0)[order]
            times = [times[i] for i in order]
            runs = [runs[i] for i in order]
            depth, lat, lon = chunks[0]["depth"], chunks[0]["lat"], chunks[0]["lon"]
            uf = np.ma.filled(u.astype("float32"), np.nan)
            vf = np.ma.filled(v.astype("float32"), np.nan)
            # NCSS writes below-bed and land values as NaN rather than as a masked fill, so
            # validity is tested on the values, not only on the mask.
            valid = np.isfinite(uf) & np.isfinite(vf)                         # (T, Z, Y, X)
            nz = valid.shape[1]
            any_valid = valid.any(axis=1)                                      # (T, Y, X)
            kb = nz - 1 - np.argmax(valid[:, ::-1], axis=1)                    # deepest valid level
            ti, yi, xi = np.meshgrid(np.arange(len(times)), np.arange(len(lat)), np.arange(len(lon)), indexing="ij")
            u_bot = np.where(any_valid, uf[ti, kb, yi, xi], np.nan)
            v_bot = np.where(any_valid, vf[ti, kb, yi, xi], np.nan)
            u_sur = np.where(valid[:, 0], uf[:, 0], np.nan)
            v_sur = np.where(valid[:, 0], vf[:, 0], np.nan)
            bottom_depth = np.where(any_valid[0], depth[kb[0]], np.nan).astype("float32")
            out = OUT / "currents" / f"hycom_espc_{region}.nc"
            epoch = "hours since 1970-01-01 00:00:00"
            with netCDF4.Dataset(out, "w") as nc:
                nc.createDimension("time", len(times))
                nc.createDimension("lat", len(lat))
                nc.createDimension("lon", len(lon))
                vt = nc.createVariable("time", "f8", ("time",))
                vt.units = epoch
                vt[:] = netCDF4.date2num(times, epoch)
                vr = nc.createVariable("run_time", "f8", ("time",))
                vr.units = epoch
                vr.long_name = "model run (analysis) time that produced each step"
                vr[:] = netCDF4.date2num(runs, epoch)
                vla = nc.createVariable("lat", "f8", ("lat",))
                vla[:] = lat
                vla.units = "degrees_north"
                vlo = nc.createVariable("lon", "f8", ("lon",))
                vlo[:] = lon
                vlo.units = "degrees_east"
                for name, arr, desc in (("u_surface", u_sur, "eastward velocity at z = 0 m"),
                                        ("v_surface", v_sur, "northward velocity at z = 0 m"),
                                        ("u_bottom", u_bot, "eastward velocity at the deepest valid model level"),
                                        ("v_bottom", v_bot, "northward velocity at the deepest valid model level")):
                    var = nc.createVariable(name, "i2", ("time", "lat", "lon"), zlib=True, complevel=6,
                                            fill_value=np.int16(-32768))
                    var.scale_factor = 0.001
                    var.add_offset = 0.0
                    var.units = "m s-1"
                    var.long_name = desc
                    var[:] = np.ma.masked_invalid(arr)
                vb = nc.createVariable("bottom_level_depth", "f4", ("lat", "lon"), zlib=True, fill_value=np.float32(np.nan))
                vb.units = "m"
                vb.long_name = "depth of the model level used for u_bottom / v_bottom (first time step)"
                vb[:] = bottom_depth
                nc.title = "HYCOM ESPC-D-V02 surface and near-bottom currents, clipped for GhostTrace"
                nc.source = "HYCOM.org THREDDS NetCDF Subset Service, FMRC_ESPC-D-V02_uv3z best time series"
                nc.source_url = HYCOM_NCSS
                nc.generating_model = chunks[0]["generating_model"]
                nc.institution = chunks[0]["institution"]
                nc.model_created_on = ";".join(sorted({c["created_on"] for c in chunks if c["created_on"]}))
                nc.model_runs = ";".join(sorted({r.strftime("%Y-%m-%dT%H:%MZ") for r in runs}))
                nc.accessed = datetime.now(timezone.utc).isoformat(timespec="seconds")
                nc.region = region
                nc.depth_levels_m = ",".join(f"{d:g}" for d in depth)
                nc.bottom_level_note = ("deepest z-level with valid data in each cell: a near-bottom velocity, "
                                        "up to one level spacing above the model bed, not a boundary-layer value")
                nc.forecast_note = ("steps whose time is later than run_time + 24 h come from the latest run's "
                                    "forecast, not an analysis")
                nc.synthetic = "false"
            latest_run = max(runs)
            n_fc = sum(1 for t, rr in zip(times, runs) if (t - rr) >= timedelta(hours=24))
            snapshots[region] = (f"{times[0]:%Y-%m-%dT%H:%MZ} to {times[-1]:%Y-%m-%dT%H:%MZ}, 3-hourly, "
                                 f"{len(times)} steps; runs {min(runs):%Y-%m-%dT%HZ}..{latest_run:%Y-%m-%dT%HZ}; "
                                 f"{n_fc} steps are forecast (>= 24 h after their run)")
            files.append(str(out.relative_to(OUT)))
            log(f"  wrote {out.relative_to(REPO)} {out.stat().st_size / 1e6:.2f} MB; {snapshots[region]}")
    upsert_source(m, {
        "id": "hycom_espc_d_v02", "name": "HYCOM ESPC-D-V02 Global 1/12 deg ocean analysis/forecast (NRL / FNMOC)",
        "url": HYCOM_NCSS, "licence": "Freely available (HYCOM.org THREDDS catalog: 'rights: Freely available'; "
                                      "distribution statement: Approved for public release; distribution unlimited)",
        "redistributable": True,
        "citation": "HYCOM consortium / US Naval Research Laboratory; ESPC-D V02 (HYCOM 2.2.99, CICE 5.1.2, expt_03.1), https://www.hycom.org/dataserver/espc-d-v02",
        "used_for": "surface and near-bottom currents for the drift forecast and the diver brief",
        "snapshot": "; ".join(f"{r}: {s}" for r, s in snapshots.items()),
        "clipped": ("NCSS subset per day (all 40 z-levels) for each region box padded by 0.1 deg; kept only the "
                    "z = 0 m level and the deepest valid level per cell; stored as int16 x 0.001 m/s, zlib"),
        "files": files,
        "notes": ["Grid: 0.08 deg longitude x 0.04 deg latitude (glby0.08); ~9 km, so channels between the Gulf "
                  "of Mannar islands and Pamban Pass are not resolved.",
                  "No wind is bundled: floating-mode drift is current-only (no windage)."]})
    upsert_not_used(m, {
        "name": "INCOIS (MoES) HOOFS ocean current forecasts",
        "url": ("https://incois.gov.in/site/datainfo/modelling/hoofs.jsp ; https://erddap.incois.gov.in/erddap/ ; "
                "https://las.incois.gov.in/las/ ; https://odis.incois.gov.in/"),
        "checked": "2026-09-13",
        "finding": ("preferred for Indian waters but not programmatically accessible. The HOOFS page describes the "
                    "system (IO-HOOFS 1/12 deg, NIO-HOOFS 1/48 deg) but links no data service. INCOIS ERDDAP lists "
                    "15 gridded datasets (AMSR-E, ASCAT, Argo products, Oceansat-2, QuikSCAT, TMI, NOAA OI SST, "
                    "IRS chlorophyll and 'INCOIS Value Added Products' with 1 deg 10-day geostrophic GEO_U/GEO_V "
                    "ending 2019-03-30) and no HOOFS currents. The INCOIS Live Access Server categories are Argo, "
                    "ASCAT, GODAS, IGORA, MaMetAtTIO, microwave, NIO climatology, NOAA SST, ocean carbonate, ocean "
                    "colour, OSCAT, QuikSCAT and TropFlux, with no HOOFS forecast. odis.incois.gov.in did not "
                    "resolve in DNS. The INCOIS data-holdings page lists HF radar currents as 'Registered access "
                    "through Website'. None of these gives hourly/3-hourly surface and subsurface currents "
                    "for a recent window without registration.")})
    upsert_not_used(m, {
        "name": "HYCOM ESPC-D-V02 archive via OPeNDAP (tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z/2026)",
        "url": "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z/2026", "checked": "2026-09-13",
        "finding": ("reachable, but a 14-day subset did not complete in over 15 minutes; a single 14-day NCSS "
                    "request also returned HTTP 500 after 5 minutes. Per-day NCSS requests (~45 s each) work "
                    "and are what this script uses.")})


# --- main ---------------------------------------------------------------------------------------------------

STEPS = ("land", "protected", "reefs", "seagrass", "curated", "harbours", "bathymetry", "currents")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=",".join(STEPS), help=f"comma list of steps: {','.join(STEPS)}")
    ap.add_argument("--regions", default=",".join(cfg.REGIONS))
    ap.add_argument("--days", type=int, default=14, help="calendar days of currents to fetch from --start, inclusive; the last day may be partial when the server forecast ends mid-day")
    ap.add_argument("--start", default=None, help="currents start date (UTC, YYYY-MM-DD); default: server window start")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.only.split(",") if s.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    for s in steps:
        if s not in STEPS:
            ap.error(f"unknown step {s}")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "ghosttrace-fetch-cache"
    cache.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    ok = True
    for step in steps:
        try:
            if step == "land":
                fetch_land(m, regions, cache)
            elif step == "protected":
                fetch_protected(m, regions)
            elif step == "reefs":
                fetch_osm_reefs(m, regions)
                try:
                    fetch_wcmc(m, regions, what="reef")
                except Exception as exc:
                    log(f"UNEP-WCMC reefs failed: {exc}")
                    ok = False
            elif step == "seagrass":
                fetch_wcmc(m, regions, what="seagrass")
            elif step == "curated":
                fetch_curated(m, regions)
            elif step == "harbours":
                fetch_harbours(m, regions)
            elif step == "bathymetry":
                fetch_bathymetry(m, regions)
            elif step == "currents":
                start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
                fetch_currents(m, regions, days=args.days, start=start, workers=args.workers)
            if step in ("land", "protected", "reefs", "curated", "harbours"):
                m["regions_fetched"] = sorted(set(m.get("regions_fetched", [])) | set(regions))
            save_manifest(m)
        except Exception as exc:
            ok = False
            log(f"STEP {step} FAILED: {type(exc).__name__}: {exc}")
            if step == "currents":
                log("No current data was written. GhostTrace does not substitute synthetic currents.")
            save_manifest(m)
    log(f"done; total bundled size {m.get('total_bytes', 0) / 1e6:.2f} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
