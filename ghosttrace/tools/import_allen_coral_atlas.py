"""Bring an Allen Coral Atlas download into the GhostTrace habitat layers.

    .venv/bin/python ghosttrace/tools/import_allen_coral_atlas.py <download.zip | folder | file.geojson | file.gpkg> [...]
        [--regions gulf_of_mannar_palk_bay,odisha_coast] [--keep-fragments]

The Atlas (https://allencoralatlas.org, CC BY 4.0) maps the world's shallow
reefs from satellite imagery, but a download needs a registered login, so the
fetcher cannot pull it. This tool takes the file a person downloaded and
writes, per configured region:

    layers/reef_aca_<region>.geojson       benthic class "Coral/Algae"
    layers/seagrass_aca_<region>.geojson   benthic class "Seagrass" (when present)

in the same shape as every other bundled layer, so habitat.py, drift.py and
the API pick them up with no code change. The manifest gains the source with
its citation and the "not used" entry for the Atlas is retired. Nothing else
on disk is touched: the UNEP-WCMC and OpenStreetMap reef layers stay, and
habitat_context reports the nearest reef across all of them.

WHAT IS READ
    GeoJSON (.geojson / .json) and GeoPackage (.gpkg, read through sqlite3
    without GDAL) inside a zip or a folder, or given directly. Only benthic
    maps are used; geomorphic maps are skipped by name. The benthic class is
    taken from the first of the properties `class`, `benthic_class`, `Class`.
    Anything that is not Coral/Algae or Seagrass (Rock, Rubble, Sand,
    Microalgal Mats) is dropped: rock and rubble are not what a net smothers.

WHAT IS WRITTEN
    Polygons are clipped to each region box plus the fetcher's padding, then
    dissolved (union) and exploded into single polygons, because the Atlas
    ships tens of thousands of raster-derived fragments and the nearest-
    feature search only needs the outline. --keep-fragments skips the union.
    Every feature carries source, source_id, geometry_quality, the benthic
    class and a note that the map is satellite-derived (shallow reefs only).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ghosttrace import config_geo as cfg  # noqa: E402
from ghosttrace.tools import fetch_ghosttrace_data as fetcher  # noqa: E402

SOURCE_ID = "allen_coral_atlas"
SOURCE_NAME = "Allen Coral Atlas benthic habitat map (CC BY 4.0)"
CITATION = ("Allen Coral Atlas (2022). Imagery, maps and monitoring of the world's tropical coral reefs. "
            "doi:10.5281/zenodo.3833242. https://allencoralatlas.org")
CLASS_KEYS = ("class", "benthic_class", "Class", "CLASS")
KIND_FOR_CLASS = {"coral/algae": "reef", "seagrass": "seagrass"}
GEO_SUFFIXES = (".geojson", ".json", ".gpkg")


# --- reading -------------------------------------------------------------------------------


def _is_benthic(name: str) -> bool:
    n = name.lower()
    return "geomorphic" not in n


def _gpkg_geometry(blob: bytes):
    """Geometry from a GeoPackage binary blob: 'GP' header, optional envelope, then WKB."""
    if not blob or blob[:2] != b"GP":
        return None
    flags = blob[3]
    env = (flags >> 1) & 0x07
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env, 0)
    return shapely.from_wkb(blob[8 + env_len:])


def _read_gpkg(path: Path) -> Iterable[dict[str, Any]]:
    con = sqlite3.connect(str(path))
    try:
        tables = [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")]
        for table in tables:
            if not _is_benthic(table):
                continue
            gcol = con.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
                               (table,)).fetchone()
            if not gcol:
                continue
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
            for row in con.execute(f'SELECT * FROM "{table}"'):
                rec = dict(zip(cols, row))
                geom = _gpkg_geometry(rec.pop(gcol[0], None))
                if geom is None or geom.is_empty:
                    continue
                yield {"geometry": geom, "properties": rec}
    finally:
        con.close()


def _read_geojson(path: Path) -> Iterable[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features") if isinstance(data, dict) else None
    if feats is None:
        return
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        try:
            geom = shape(g)
        except Exception:
            continue
        if geom.is_empty:
            continue
        yield {"geometry": geom, "properties": f.get("properties") or {}}


def _expand(inputs: list[Path], tmp: Path) -> list[Path]:
    """Every readable benthic file behind the given zips, folders and files."""
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out += _expand(sorted(p.iterdir()), tmp)
        elif p.suffix.lower() == ".zip":
            dest = tmp / p.stem
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
            out += _expand([dest], tmp)
        elif p.suffix.lower() in GEO_SUFFIXES and _is_benthic(p.name):
            out.append(p)
    return out


def read_features(paths: list[Path]) -> list[dict[str, Any]]:
    feats: list[dict[str, Any]] = []
    for p in paths:
        reader = _read_gpkg if p.suffix.lower() == ".gpkg" else _read_geojson
        n = 0
        for f in reader(p):
            feats.append(f)
            n += 1
        fetcher.log(f"read {p.name}: {n} features")
    return feats


def benthic_class(props: dict[str, Any]) -> str | None:
    for k in CLASS_KEYS:
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# --- writing -------------------------------------------------------------------------------


def _polygons(geom) -> list:
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [g for part in geom.geoms for g in _polygons(part)]
    return []


def build_layers(feats: list[dict[str, Any]], regions: list[str], *, dissolve: bool = True
                 ) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """{(kind, region): features} for the classes GhostTrace scores, clipped per region."""
    by_class: dict[str, list] = {k: [] for k in KIND_FOR_CLASS}
    seen: dict[str, int] = {}
    for f in feats:
        c = benthic_class(f["properties"])
        seen[c or "(no class)"] = seen.get(c or "(no class)", 0) + 1
        key = (c or "").lower()
        if key in by_class:
            by_class[key].append(f["geometry"])
    fetcher.log("benthic classes in the download: " + ", ".join(f"{k} x{v}" for k, v in sorted(seen.items())))

    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cls, kind in KIND_FOR_CLASS.items():
        geoms = by_class[cls]
        for region in regions:
            s, w, n, e = fetcher.padded(region)
            clip = box(w, s, e, n)
            inside = []
            for g in geoms:
                if not g.intersects(clip):
                    continue
                g2 = g.intersection(clip)
                if g2.is_empty:
                    continue
                inside.append(shapely.make_valid(g2) if not g2.is_valid else g2)
            if not inside:
                out[(kind, region)] = []
                continue
            if dissolve:
                merged = unary_union(inside)
                polys = _polygons(merged)
                quality = "mapped_polygon (dissolved from satellite-classified fragments)"
            else:
                polys = [p for g in inside for p in _polygons(g)]
                quality = "mapped_polygon (satellite-classified fragment)"
            out[(kind, region)] = [{
                "geometry": p,
                "properties": {"name": None, "benthic_class": cls.title() if cls != "coral/algae" else "Coral/Algae",
                               "source": "Allen Coral Atlas", "source_id": SOURCE_ID,
                               "geometry_quality": quality,
                               "citation": CITATION,
                               "note": ("satellite-derived benthic map of shallow (roughly < 15 m) reef; deeper "
                                        "reef is not mapped by this source")}}
                for p in polys if not p.is_empty]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="Atlas download zip(s), folder(s), or .geojson/.gpkg file(s)")
    ap.add_argument("--regions", default=",".join(cfg.REGIONS),
                    help="regions to write, from config_geo.REGIONS (default: all)")
    ap.add_argument("--keep-fragments", action="store_true", help="do not dissolve touching polygons")
    args = ap.parse_args(argv)

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    unknown = [r for r in regions if r not in cfg.REGIONS]
    if unknown:
        print(f"error: unknown region(s) {unknown}; known: {list(cfg.REGIONS)}", file=sys.stderr)
        return 2
    inputs = [Path(p) for p in args.inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print(f"error: not found: {missing}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="aca-") as tmp:
        files = _expand(inputs, Path(tmp))
        if not files:
            print("error: no benthic .geojson/.json/.gpkg found in the inputs (geomorphic maps are skipped)",
                  file=sys.stderr)
            return 2
        feats = read_features(files)
    if not feats:
        print("error: the inputs contain no features", file=sys.stderr)
        return 2

    layers = build_layers(feats, regions, dissolve=not args.keep_fragments)
    written: list[str] = []
    counts: dict[str, int] = {}
    for (kind, region), fs in layers.items():
        if not fs:
            fetcher.log(f"{kind}: nothing from the Atlas inside {region}; no file written")
            continue
        path = fetcher.write_layer(f"{kind}_aca_{region}", kind, fs, source_id=SOURCE_ID,
                                   regions=[region], restricted=False)
        written.append(str(path.relative_to(fetcher.OUT)))
        counts[f"{kind}/{region}"] = len(fs)
    if not written:
        print("error: the download has no Coral/Algae or Seagrass polygons inside the configured regions; "
              "check that the area you drew on the Atlas covers the region box", file=sys.stderr)
        return 1

    m = fetcher.load_manifest()
    fetcher.upsert_source(m, {
        "id": SOURCE_ID, "name": SOURCE_NAME, "url": "https://allencoralatlas.org/atlas/",
        "licence": "CC BY 4.0 (https://allencoralatlas.org/terms/); attribution required",
        "redistributable": True, "citation": CITATION,
        "used_for": "reef layer (benthic class Coral/Algae) and seagrass layer (benthic class Seagrass); "
                    "satellite-derived shallow-reef outlines beside the UNEP-WCMC and OSM layers",
        "clipped": "benthic polygons intersecting each region box plus padding, dissolved per region",
        "files": written, "feature_counts": counts,
        "how_obtained": "downloaded by a registered user from the Atlas map (My Areas > Download) and "
                        "imported with tools/import_allen_coral_atlas.py; not fetched programmatically",
        "notes": ["Shallow reef only (optical satellite depth limit); deeper reef remains unmapped here.",
                  "Benthic classes Rock, Rubble, Sand and Microalgal Mats were dropped."]})
    m["not_used"] = [s for s in m.get("not_used", []) if "Allen Coral Atlas" not in str(s.get("name"))]
    fetcher._TOUCHED_NOT_USED.update(s["name"] for s in m["not_used"])
    fetcher.save_manifest(m)
    # save_manifest merges only touched not_used records, so retire the old entry explicitly
    disk = fetcher.load_manifest()
    disk["not_used"] = [s for s in disk.get("not_used", []) if "Allen Coral Atlas" not in str(s.get("name"))]
    cfg.MANIFEST_PATH.write_text(json.dumps(disk, indent=2, ensure_ascii=False), encoding="utf-8")
    fetcher.write_sources_md(disk)

    for name, n in counts.items():
        print(f"  {name}: {n} polygons")
    print(f"wrote {len(written)} layer file(s) under {fetcher.OUT / 'layers'} and updated the manifest")
    print("re-run GhostTrace on each survey so habitat and drift use the new reef outlines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
