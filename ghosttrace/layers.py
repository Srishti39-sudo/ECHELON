"""The bundled geographic layers, loaded lazily, indexed, and honest about coverage.

GhostTrace asks the same question of every detection: what is this net lying
on, what is near it, and what could it reach. The answers come from vector
layers bundled under data/ghosttrace/layers/, written there by
tools/fetch_ghosttrace_data.py from real, named sources. This module loads
them, puts a spatial index over each one, and answers one prior question
before any other: DOES THE BUNDLED DATA COVER THIS POINT AT ALL.

That question matters more than it looks. A habitat search that finds nothing
within 50 km returns an empty list, and an empty list reads as "no reef near
this net". Outside the bundled boxes the truth is "nobody here looked", which
is a different sentence with the opposite operational meaning. region_for()
tells the two apart, and habitat.py and drift.py refuse to answer outside
coverage rather than reporting a clean bill of health.

LAYER FILES
    One GeoJSON FeatureCollection per layer, lon/lat WGS84, with a top-level
    "ghosttrace" block:

        {"layer": "reef_wcmc", "kind": "reef", "source_id": "...",
         "regions": ["gulf_of_mannar_palk_bay", ...], "restricted": false}

    and on every feature at least:

        name              str or null (never invented; unnamed stays null)
        source            short source name, e.g. "OpenStreetMap"
        source_id         key into the manifest's sources list
        geometry_quality  "surveyed_polygon" | "mapped_polygon" |
                          "approximate" | "approximate_line" | "point"
        citation          for curated features, where the location comes from

    kind is one of reef | seagrass | protected_area | turtle_nesting | dugong |
    harbour | land.

COVERAGE
    A point is covered when it lies inside one of config_geo.REGIONS AND the
    manifest records that layers were fetched for that region. A region box
    with no fetched data is not coverage.

    Coverage is per kind as well. The coral reef layer from UNEP-WCMC may not
    be redistributed (UNEP-WCMC General Data License, clause 3), so it is
    fetched locally and git-ignored; a fresh clone that has not run the
    fetcher has only the sparse OpenStreetMap reefs. region_for() lists which
    kinds are present, so a missing kind is reported as "not assessed" rather
    than "absent".

DISTANCES
    Distances are computed in an azimuthal equidistant projection centred on
    the query point (pyproj). That projection preserves distance and azimuth
    FROM ITS CENTRE exactly on the WGS84 ellipsoid, so the distance to the
    nearest point of a polygon is a geodesic distance and the bearing is a true
    azimuth, without a separate geodesic search along the polygon edge. Buffers
    used by the drift stage are built in a projection centred on the region,
    where distortion over a ~250 km box is well under 1%.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point, shape, mapping
from shapely.strtree import STRtree

from ghosttrace import config_geo as cfg

GEOD = Geod(ellps="WGS84")

LAYER_KINDS = ("reef", "seagrass", "protected_area", "turtle_nesting", "dugong", "harbour", "land")

NO_DATA_MESSAGE = "no bundled data for this location"


# --- projections ------------------------------------------------------------------


@lru_cache(maxsize=256)
def _aeqd(lat: float, lon: float) -> tuple[Transformer, Transformer]:
    """(to_local, to_lonlat) for an azimuthal equidistant CRS centred at lat, lon."""
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat:.6f} +lon_0={lon:.6f} +datum=WGS84 +units=m +no_defs")
    to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return to_local, to_lonlat


def _transform(geom, transformer: Transformer):
    def fn(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geom, fn)


def to_local(geom, lat0: float, lon0: float):
    """Geometry in lon/lat -> metres in an AEQD projection centred at lat0, lon0."""
    return _transform(geom, _aeqd(round(lat0, 6), round(lon0, 6))[0])


def to_lonlat(geom, lat0: float, lon0: float):
    """The inverse of to_local."""
    return _transform(geom, _aeqd(round(lat0, 6), round(lon0, 6))[1])


def degree_pad(lat: float, metres: float) -> tuple[float, float]:
    """(dlat, dlon) in degrees that certainly contain `metres` around a latitude.

    Padded by 2% so a bounding-box prefilter never drops a feature a precise
    distance test would keep.
    """
    dlat = metres / 110_574.0 * 1.02
    dlon = metres / (111_320.0 * max(math.cos(math.radians(abs(lat) + dlat)), 1e-6)) * 1.02
    return dlat, dlon


# --- layers -----------------------------------------------------------------------


@dataclass
class Layer:
    """One bundled layer: a kind, its features, and a spatial index over them."""

    name: str
    kind: str
    geometries: list[Any]
    properties: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    _tree: STRtree | None = field(default=None, repr=False)

    @classmethod
    def from_geojson(cls, path: Path) -> "Layer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = data.get("ghosttrace") or {}
        name = meta.get("layer") or Path(path).stem
        kind = meta.get("kind")
        if kind not in LAYER_KINDS:
            raise ValueError(f"{path}: layer kind {kind!r} is not one of {LAYER_KINDS}")
        geoms, props = [], []
        for feat in data.get("features", []):
            if not feat.get("geometry"):
                continue
            g = shape(feat["geometry"])
            if g.is_empty:
                continue
            if not g.is_valid:
                g = shapely.make_valid(g)
            geoms.append(g)
            p = dict(feat.get("properties") or {})
            p.setdefault("name", None)
            p.setdefault("source", meta.get("source"))
            p.setdefault("geometry_quality", meta.get("geometry_quality", "unspecified"))
            props.append(p)
        return cls(name=name, kind=kind, geometries=geoms, properties=props, meta=meta, path=Path(path))

    def __len__(self) -> int:
        return len(self.geometries)

    @property
    def tree(self) -> STRtree:
        if self._tree is None:
            self._tree = STRtree(self.geometries)
        return self._tree

    def query_bbox(self, south: float, west: float, north: float, east: float) -> np.ndarray:
        """Indices of features whose bounding boxes intersect the box."""
        if not self.geometries:
            return np.array([], dtype=int)
        return self.tree.query(shapely.box(west, south, east, north))

    def features(self) -> list[dict[str, Any]]:
        """GeoJSON features, for callers that want plain dicts."""
        return [{"type": "Feature", "geometry": mapping(g), "properties": p}
                for g, p in zip(self.geometries, self.properties)]

    def regions(self) -> list[str]:
        return list(self.meta.get("regions") or [])


class Bathymetry:
    """Gridded elevation (NOAA ETOPO 2022, 15 arc-second) for the bundled regions.

    depth_at() answers with a positive-down depth in metres, bilinearly
    interpolated, or None outside the bundled grids. Elevation above zero is
    reported as on_land rather than a negative depth. ETOPO is a global
    compilation: in shallow Indian coastal waters its values come from
    sparse soundings and satellite-derived bathymetry, so it is a planning
    figure, never a substitute for the survey's own depth.
    """

    def __init__(self, grids: list[dict[str, Any]], source: dict[str, Any] | None):
        self.grids = grids
        self.source = source or {}

    @classmethod
    def from_dir(cls, directory: Path, source: dict[str, Any] | None = None) -> "Bathymetry | None":
        import netCDF4  # imported lazily: layers without bathymetry need no netCDF

        grids = []
        for path in sorted(Path(directory).glob("*.nc")):
            with netCDF4.Dataset(path) as ds:
                lat = np.asarray(ds.variables["lat"][:], dtype=float)
                lon = np.asarray(ds.variables["lon"][:], dtype=float)
                z = np.ma.filled(ds.variables["z"][:].astype("float32"), np.nan)
                grids.append({"path": path.name, "lat": lat, "lon": lon, "z": z,
                              "attrs": {k: ds.getncattr(k) for k in ds.ncattrs()}})
        return cls(grids, source) if grids else None

    def depth_at(self, lat: float, lon: float) -> dict[str, Any]:
        for g in self.grids:
            lat_a, lon_a = g["lat"], g["lon"]
            if not (lat_a[0] <= lat <= lat_a[-1] and lon_a[0] <= lon <= lon_a[-1]):
                continue
            i = int(np.clip(np.searchsorted(lat_a, lat) - 1, 0, len(lat_a) - 2))
            j = int(np.clip(np.searchsorted(lon_a, lon) - 1, 0, len(lon_a) - 2))
            fy = (lat - lat_a[i]) / (lat_a[i + 1] - lat_a[i])
            fx = (lon - lon_a[j]) / (lon_a[j + 1] - lon_a[j])
            c = g["z"][i:i + 2, j:j + 2]
            z = (c[0, 0] * (1 - fy) * (1 - fx) + c[0, 1] * (1 - fy) * fx
                 + c[1, 0] * fy * (1 - fx) + c[1, 1] * fy * fx)
            z = float(z)
            return {
                "elevation_m": round(z, 1),
                "depth_m": round(-z, 1) if z < 0 else None,
                "on_land": z >= 0,
                "source": self.source.get("name", "NOAA ETOPO 2022 15 arc-second"),
                "source_url": self.source.get("url"),
                "grid": g["path"],
                "resolution": "15 arc-second (~460 m)",
                "method": "bilinear interpolation of the ETOPO 2022 surface elevation grid",
                "note": ("global compilation; shallow coastal values are approximate and are "
                         "no substitute for the survey's own depth"),
            }
        return {"elevation_m": None, "depth_m": None, "on_land": None, "source": None,
                "note": NO_DATA_MESSAGE}


class LayerSet:
    """Every bundled layer, by name and by kind, plus coverage and derived geometry."""

    def __init__(self, layers: Iterable[Layer], manifest: dict[str, Any] | None = None,
                 data_dir: Path | None = None):
        self.layers: dict[str, Layer] = {l.name: l for l in layers}
        self.manifest = manifest or {}
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._land_cache: dict[str, Any] = {}
        self._buffer_cache: dict[tuple, Any] = {}

    # -- construction ---------------------------------------------------------------

    @classmethod
    def from_dir(cls, data_dir: Path | None = None) -> "LayerSet":
        data_dir = Path(data_dir or cfg.DATA_DIR)
        manifest = {}
        mpath = data_dir / "manifest.json"
        if mpath.exists():
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        layers = [Layer.from_geojson(p) for p in sorted((data_dir / "layers").glob("*.geojson"))]
        return cls(layers, manifest, data_dir)

    # -- access -----------------------------------------------------------------------

    def by_kind(self, kind: str) -> list[Layer]:
        return [l for l in self.layers.values() if l.kind == kind]

    def kinds_present(self, region: str | None = None) -> list[str]:
        """Kinds with a layer file for the region. An empty layer counts: it was searched."""
        kinds = set()
        for l in self.layers.values():
            if region is None or region in l.regions():
                kinds.add(l.kind)
        return sorted(kinds)

    @property
    def data_sources(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("sources", []))

    def source_record(self, source_id: str | None) -> dict[str, Any] | None:
        for s in self.data_sources:
            if s.get("id") == source_id:
                return s
        return None

    @property
    def harbours(self) -> list[dict[str, Any]]:
        """Harbour / landing positions as plain records (polygons reduced to a point on them)."""
        out = []
        for layer in self.by_kind("harbour"):
            for g, p in zip(layer.geometries, layer.properties):
                pt = g if g.geom_type == "Point" else g.representative_point()
                out.append({"name": p.get("name"), "latitude": pt.y, "longitude": pt.x,
                            "source": p.get("source"), "layer": layer.name,
                            "geometry_quality": p.get("geometry_quality")})
        return out

    # -- coverage ---------------------------------------------------------------------

    def region_for(self, lat: float, lon: float) -> dict[str, Any]:
        """Whether bundled data covers a point, and which kinds are present there.

        Returns {"covered", "region", "kinds_present", "kinds_missing", "reason"}.
        """
        fetched = set(self.manifest.get("regions_fetched") or [])
        for name, (s, w, n, e) in cfg.REGIONS.items():
            if s <= lat <= n and w <= lon <= e:
                present = self.kinds_present(name)
                if name not in fetched or not present:
                    return {"covered": False, "region": name, "kinds_present": [],
                            "kinds_missing": list(LAYER_KINDS),
                            "reason": (f"{NO_DATA_MESSAGE}: region '{name}' is defined but its "
                                       "layers have not been fetched (run tools/fetch_ghosttrace_data.py)")}
                missing = [k for k in LAYER_KINDS if k not in present]
                return {"covered": True, "region": name, "kinds_present": present,
                        "kinds_missing": missing, "reason": None}
        return {"covered": False, "region": None, "kinds_present": [], "kinds_missing": list(LAYER_KINDS),
                "reason": (f"{NO_DATA_MESSAGE}: ({lat:.4f}, {lon:.4f}) lies outside every bundled "
                           f"region ({', '.join(cfg.REGIONS)})")}

    # -- derived geometry for the drift stage --------------------------------------------

    def land(self, region: str):
        """Union of land polygons for a region (lon/lat), or None when not bundled."""
        with self._lock:
            if region in self._land_cache:
                return self._land_cache[region]
            geoms = []
            for layer in self.by_kind("land"):
                if region in layer.regions() or not layer.regions():
                    geoms.extend(layer.geometries)
            land = shapely.union_all(geoms) if geoms else None
            if land is not None:
                shapely.prepare(land)
            self._land_cache[region] = land
            return land

    def buffered_features(self, kind: str, buffer_m: float, region: str) -> list[dict[str, Any]]:
        """Features of a kind in a region, each buffered by buffer_m, prepared for point tests.

        Buffers are built in an AEQD projection centred on the region box and
        projected back to lon/lat, so a 1 km buffer is 1 km on the ground to
        well under 1% across the box.
        """
        key = (kind, round(float(buffer_m), 3), region)
        with self._lock:
            if key in self._buffer_cache:
                return self._buffer_cache[key]
        s, w, n, e = cfg.REGIONS[region]
        lat0, lon0 = (s + n) / 2.0, (w + e) / 2.0
        out = []
        for layer in self.by_kind(kind):
            if layer.regions() and region not in layer.regions():
                continue
            for g, p in zip(layer.geometries, layer.properties):
                if buffer_m > 0:
                    local = to_local(g, lat0, lon0).buffer(buffer_m, quad_segs=8)
                    bg = to_lonlat(local, lat0, lon0)
                else:
                    bg = g if g.geom_type in ("Polygon", "MultiPolygon") else \
                        to_lonlat(to_local(g, lat0, lon0).buffer(1.0), lat0, lon0)
                shapely.prepare(bg)
                out.append({"geometry": bg, "layer": layer.name, "kind": kind, "properties": p})
        with self._lock:
            self._buffer_cache[key] = out
        return out


# --- defaults -------------------------------------------------------------------------

_DEFAULT: LayerSet | None = None
_DEFAULT_BATHY: Bathymetry | None = None
_DEFAULT_LOCK = threading.Lock()


def load_default_layers() -> LayerSet:
    """The LayerSet from config_geo.DATA_DIR, loaded once per process."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = LayerSet.from_dir(cfg.DATA_DIR)
        return _DEFAULT


def load_bathymetry() -> Bathymetry | None:
    """ETOPO 2022 grids from data/ghosttrace/bathymetry, or None when not fetched."""
    global _DEFAULT_BATHY
    with _DEFAULT_LOCK:
        if _DEFAULT_BATHY is None:
            layers_manifest = {}
            if cfg.MANIFEST_PATH.exists():
                layers_manifest = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
            src = next((s for s in layers_manifest.get("sources", []) if s.get("id") == "etopo_2022"), None)
            if cfg.BATHYMETRY_DIR.exists():
                _DEFAULT_BATHY = Bathymetry.from_dir(cfg.BATHYMETRY_DIR, src)
        return _DEFAULT_BATHY


def region_for(lat: float, lon: float, layers: LayerSet | None = None) -> dict[str, Any]:
    """Module-level shortcut: coverage of the bundled data at a point."""
    return (layers or load_default_layers()).region_for(lat, lon)


def point_geometry(lat: float, lon: float) -> Point:
    return Point(lon, lat)
