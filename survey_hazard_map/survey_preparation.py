"""Module 1. Sonar strips in, a positioned tile set out.

    from survey_hazard_map.survey_preparation import prepare_survey
    tiles_dir, manifest_path = prepare_survey(["survey/strip_a.png"], "out/")

A side-scan strip is long, thin and far larger than any detector's input. This
module cuts it into overlapping 640-pixel tiles, records where each tile came
from, and writes a manifest that lets every later stage put a detection back
where it was found.

The manifest is the contract. Everything downstream reads positions from it and
never re-derives them from a filename, so the pixel arithmetic in this file is
the only place it happens.

WHAT A TILE IS CALLED
    {strip}_{x}_{y}.jpg, where x and y are the tile's top-left offset in the
    ORIGINAL strip, in pixels. Not a tile index. A tile's name therefore states
    its position, and the name is reproducible from the manifest and back.

COVERAGE
    Offsets march in steps of STRIDE and stop as soon as the next tile would
    start past the end of the strip, so the last tile in each direction is
    partial and every pixel of the strip is covered exactly once or twice.
    Because STRIDE is smaller than TILE, a partial tile is never narrower than
    TILE - STRIDE, so the run never produces a useless sliver.

MEMORY
    Strips are opened one at a time and each tile is written before the next is
    cut. Peak memory is one strip plus one tile, not the whole survey, so the
    number of strips does not change the footprint.

DETERMINISM
    Strips are processed in sorted order, offsets ascend, and nothing depends
    on dictionary iteration or wall-clock time. The same inputs produce
    byte-identical tiles and the same manifest rows in the same order.

RAW SONAR LOGS
    An .xtf or .jsf path (or a directory holding them) is first ingested by
    sonar_ingest into out_dir/strips/: a slant-range-corrected, gain-normalised
    PNG per contiguous line, with a {strip}.nav.json sidecar of per-row
    navigation and a {strip}.wc.npz water column. Those PNGs are then tiled
    like any other strip. A file that cannot be ingested is recorded under
    strips_unreadable with its reason, exactly like an unreadable image.

PING NAVIGATION FROM SIDECARS
    Any image strip with a {stem}.nav.json beside it -- every ingested strip,
    and any strip a caller ingested elsewhere -- is located from that sidecar,
    row by row, unless navigation passed in `nav` names the strip. Explicit
    navigation wins per strip, not per survey, so four corners typed for a
    photograph do not switch off the recorded navigation of a raw log in the
    same run. Sidecars and water columns are copied to out_dir/nav/ and the
    manifest's navigation block records them relative to the manifest:

        "navigation": {"mode": "ping", "sidecars": {strip: "nav/<file>"},
                       "water_columns": {strip: "nav/<file>"}, ...}

    Every tile row then also carries m_per_px_across, m_per_px_along (null
    where unknown) and quality: the worst row quality flag inside the tile, so
    a detection made on interpolated or dropout rows can say so.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_geo import Georeference, NavigationError, build_references
from survey_hazard_map.sonar_ingest import (SIDECAR_SUFFIX, SONAR_SUFFIXES, WATER_COLUMN_SUFFIX, ingest,
                          lee_filter, worst_quality)

log = logging.getLogger("deepecho.hazard")

# Image formats taken seriously as survey strips when a directory is passed.
STRIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# Where ingested raw logs and copied navigation go, inside out_dir.
STRIPS_DIRNAME = "strips"
NAV_DIRNAME = "nav"

# A survey strip is legitimately enormous, and Pillow's default bomb guard
# fires at ~179 megapixels. Raised rather than disabled: an accidental 20
# gigapixel input should still stop rather than exhaust the machine.
MAX_STRIP_PIXELS = 4_000_000_000


def _resolve_strips(strip_paths: Any) -> list[Path]:
    """One path, several paths, or a directory. Always sorted, always files."""
    if strip_paths is None:
        raise ValueError("strip_paths is required")
    if isinstance(strip_paths, (str, Path)):
        strip_paths = [strip_paths]
    if not isinstance(strip_paths, Iterable):
        raise TypeError("strip_paths must be a path or an iterable of paths")

    resolved: list[Path] = []
    for entry in strip_paths:
        path = Path(entry)
        if path.is_dir():
            resolved.extend(sorted(p for p in path.iterdir() if p.is_file() and
                                   p.suffix.lower() in STRIP_SUFFIXES + SONAR_SUFFIXES))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"survey strip not found: {path}")

    if not resolved:
        raise ValueError("no survey strips found in the given paths")
    return sorted(dict.fromkeys(resolved), key=lambda p: (p.name, str(p)))


def _strip_names(paths: Sequence[Path]) -> dict[Path, str]:
    """A stable, unique name per strip, taken from the file stem.

    Two strips in different folders can share a stem. Since the stem becomes
    part of every tile filename, a collision would have one strip's tiles
    overwrite another's, so the second occurrence is suffixed.
    """
    names: dict[Path, str] = {}
    used: dict[str, int] = {}
    for path in paths:
        stem = path.stem
        count = used.get(stem, 0)
        used[stem] = count + 1
        names[path] = stem if count == 0 else f"{stem}-{count + 1}"
    return names


def _offsets(extent: int, tile: int, stride: int) -> list[int]:
    """Tile origins along one axis. See COVERAGE in the module docstring."""
    if extent <= tile:
        return [0]
    origins = [0]
    while origins[-1] + tile < extent:
        origins.append(origins[-1] + stride)
    return origins


def _denoise(array, percentiles: tuple[float, float]):
    """Lee speckle filter plus a conservative percentile stretch. Intensity only.

    The Lee filter (sonar_ingest.lee_filter) replaced a median filter: a median
    erodes exactly the small bright returns and sharp shadow edges that mark a
    man-made object, while Lee smooths homogeneous seabed and leaves edges
    alone. The window is sonar_ingest.LEE_SIZE.

    Geometry is untouched on purpose: a tile's pixel grid IS the survey's
    coordinate system, so any resize, rotation or pad would move every object
    in it away from its recorded position.
    """
    import numpy as np

    from survey_hazard_map import sonar_ingest
    filtered = np.asarray(lee_filter(np.asarray(array, dtype=np.float32),
                                     sonar_ingest.LEE_SIZE), dtype=np.float32)

    low, high = np.percentile(filtered, percentiles)
    if high <= low:
        # A flat tile has nothing to stretch. Returning it unchanged is
        # correct; dividing by zero and calling the result contrast is not.
        return filtered.astype(np.uint8)
    return np.clip((filtered - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _grey(array):
    """The 2-D grey view used for scoring, whatever the tile's mode."""
    import numpy as np

    if array.ndim == 2:
        return array
    # Rec. 601 luma. The weights matter less than being consistent, since the
    # score is compared against a threshold, not against another system.
    return (array[..., 0] * 0.299 + array[..., 1] * 0.587 + array[..., 2] * 0.114
            ).astype(np.float32)


class StripUnreadableError(ValueError):
    """Every strip in the survey failed to open."""


def _tile_strip(path: Path, strip: str, size: tuple[int, int],
                reference: "Georeference | None", tiles_dir: Path,
                rows: list[dict[str, Any]], *, resolution: tuple[Any, Any] = (None, None),
                row_quality: list[str] | None = None, denoise: bool | None = None) -> int:
    """Cut one strip into tiles, appending manifest rows. Returns tiles skipped.

    Separated from prepare_survey so that a strip which cannot be read fails on
    its own rather than taking the survey with it. Raises only for a problem
    with this strip; the caller decides what a failure means for the batch.
    """
    import numpy as np
    from PIL import Image

    width, height = size
    skipped = 0
    denoise = cfg.DENOISE if denoise is None else denoise

    with Image.open(path) as image:
        # Palette and bilevel images are promoted; anything else keeps its
        # mode so a grey strip stays a single channel on disk.
        if image.mode in ("P", "1", "I", "F", "LA", "RGBA", "CMYK"):
            image = image.convert("L" if image.mode in ("1", "I", "F", "LA") else "RGB")

        xs = _offsets(width, cfg.TILE, cfg.STRIDE)
        ys = _offsets(height, cfg.TILE, cfg.STRIDE)
        log.info("strip %s: %dx%d px, %d x %d tile grid",
                 strip, width, height, len(xs), len(ys))

        for y in ys:
            for x in xs:
                right, bottom = min(x + cfg.TILE, width), min(y + cfg.TILE, height)
                array = np.asarray(image.crop((x, y, right, bottom)))
                if denoise:
                    array = _denoise(array, cfg.DENOISE_CLIP_PERCENTILES)

                # Scored on the pixels as written, so the filter judges
                # exactly what the detector will be shown.
                grey = _grey(array)
                content_score = float(np.std(grey)) / 255.0
                if content_score < cfg.MIN_CONTENT:
                    skipped += 1
                    continue

                tile_w, tile_h = right - x, bottom - y
                center_x, center_y = x + tile_w / 2.0, y + tile_h / 2.0
                lat, lon = reference.locate(center_x, center_y) if reference else (None, None)

                name = f"{strip}_{x}_{y}.jpg"
                Image.fromarray(array).save(
                    tiles_dir / name, "JPEG", quality=cfg.TILE_JPEG_QUALITY)

                rows.append({
                    "tile": name,
                    "strip": strip,
                    "x": x,
                    "y": y,
                    "lat": lat,
                    "lon": lon,
                    "mean_intensity": round(float(np.mean(grey)), 3),
                    "width": width,
                    "height": height,
                    "tile_width": tile_w,
                    "tile_height": tile_h,
                    "center_x": center_x,
                    "center_y": center_y,
                    "source_image": path.name,
                    "denoised": denoise,
                    "content_score": round(content_score, 6),
                    "m_per_px_across": resolution[0],
                    "m_per_px_along": resolution[1],
                    # The worst row flag inside the tile, or null when the strip
                    # has no per-row quality (a plain image).
                    "quality": (worst_quality(row_quality[y:bottom]) if row_quality else None),
                })

    return skipped


def _ingest_raw(paths: list[Path], strips_dir: Path, options: dict | None,
                unreadable: list[dict[str, str]]) -> tuple[list[Path], list[dict[str, Any]]]:
    """Ingest every raw sonar path. Returns (strip images, ingest records)."""
    images: list[Path] = []
    records: list[dict[str, Any]] = []
    for raw in paths:
        try:
            produced = ingest(raw, strips_dir, options=options)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            unreadable.append({"strip": raw.stem, "source_image": raw.name, "error": reason})
            log.error("sonar file %s could not be ingested and was skipped: %s", raw.name, reason)
            continue
        for item in produced:
            images.append(Path(item.image_path))
            records.append({"source_file": raw.name, "strip": item.strip,
                            "image": f"{STRIPS_DIRNAME}/{Path(item.image_path).name}",
                            "summary": item.summary})
    return images, records


def _sidecar_for(path: Path) -> Path | None:
    candidate = path.with_name(f"{path.stem}{SIDECAR_SUFFIX}")
    return candidate if candidate.is_file() else None


def _split_resolution(value: Any, strip: str) -> tuple[float | None, float | None]:
    """nav["m_per_px"]: a number, [across, along], or {strip: either}."""
    if isinstance(value, dict):
        value = value.get(strip)
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return float(value), float(value)


def prepare_survey(strip_paths: Any, out_dir: Any, nav: Any = None, *,
                   ingest_options: dict | None = None) -> tuple[Path, Path]:
    """Cut survey strips into positioned tiles and write the manifest.

    strip_paths
        A path, a list of paths, or a directory. Images are tiled directly;
        .xtf and .jsf raw sonar logs are ingested first (see RAW SONAR LOGS).
    out_dir
        Created if absent. Tiles go in out_dir/tiles, the manifest beside them,
        ingested strips in out_dir/strips and navigation sidecars in out_dir/nav.
    nav
        None for a relative survey, which is the default and produces null
        lat/lon everywhere -- except for strips with a navigation sidecar,
        which are located from it. A path to a control-point CSV, or a dict of
        four-corner coordinates per strip. See hazard_geo for both formats.
        A dict may also carry "m_per_px": a number, [across, along] or
        {strip: either}, recording the resolution of plain images that have no
        sidecar; it does not locate anything on its own.
    ingest_options
        Per-run overrides for sonar_ingest, e.g. {"m_per_px_across": 0.1}.

    Returns (tiles_dir, manifest_csv_path).
    """
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_STRIP_PIXELS

    started = datetime.now(timezone.utc)
    resolved = _resolve_strips(strip_paths)

    out_dir = Path(out_dir)
    tiles_dir = out_dir / cfg.TILES_DIRNAME
    tiles_dir.mkdir(parents=True, exist_ok=True)

    unreadable: list[dict[str, str]] = []
    raw_paths = [p for p in resolved if p.suffix.lower() in SONAR_SUFFIXES]
    paths = [p for p in resolved if p.suffix.lower() not in SONAR_SUFFIXES]
    ingested: list[dict[str, Any]] = []
    if raw_paths:
        images, ingested = _ingest_raw(raw_paths, out_dir / STRIPS_DIRNAME, ingest_options,
                                       unreadable)
        paths = sorted(dict.fromkeys(paths + images), key=lambda p: (p.name, str(p)))
    if not paths:
        raise StripUnreadableError(
            "no survey strip could be read. "
            + "; ".join(f"{u['source_image']} ({u['error']})" for u in unreadable))
    names = _strip_names(paths)

    # Sizes first, from the headers alone: the georeference for a four-corner
    # strip needs the strip's dimensions before any pixel is read.
    #
    # This is also where an unreadable file is caught, because opening a header
    # is the cheapest way to find out. A file that fails here is dropped from
    # the survey with its reason recorded, rather than taking the batch with it.
    sizes: dict[str, tuple[int, int]] = {}
    for path in list(paths):
        try:
            with Image.open(path) as image:
                sizes[names[path]] = (image.width, image.height)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            unreadable.append({"strip": names[path], "source_image": path.name,
                               "error": reason})
            log.error("strip %s could not be opened and was skipped: %s",
                      path.name, reason)
            paths.remove(path)

    if not sizes:
        raise StripUnreadableError(
            "no survey strip could be opened. "
            + "; ".join(f"{u['source_image']} ({u['error']})" for u in unreadable))

    # --- per-strip navigation sidecars ------------------------------------
    m_per_px = None
    if isinstance(nav, dict) and "m_per_px" in nav:
        nav = dict(nav)
        m_per_px = nav.pop("m_per_px")
        if not {k for k in nav if k != "mode"} or nav.get("mode") == "none":
            nav = None

    nav_dir = out_dir / NAV_DIRNAME
    sidecar_src: dict[str, Path] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    for path in paths:
        strip = names[path]
        found = _sidecar_for(path)
        if found is None:
            continue
        try:
            sidecars[strip] = json.loads(found.read_text(encoding="utf-8"))
            sidecar_src[strip] = found
        except (OSError, ValueError) as exc:
            unreadable.append({"strip": strip, "source_image": found.name,
                               "error": f"navigation sidecar unreadable: {exc}"})
            log.error("sidecar %s could not be read; strip %s is unlocated", found.name, strip)

    copied: dict[str, str] = {}
    water_columns: dict[str, str] = {}
    if sidecar_src:
        nav_dir.mkdir(parents=True, exist_ok=True)
    for strip, src in sidecar_src.items():
        target = nav_dir / f"{strip}{SIDECAR_SUFFIX}"
        payload = sidecars[strip]
        wc_block = payload.get("water_column") or {}
        wc_src = src.parent / str(wc_block.get("path") or f"{src.name[:-len(SIDECAR_SUFFIX)]}"
                                                             f"{WATER_COLUMN_SUFFIX}")
        wc_target = nav_dir / f"{strip}{WATER_COLUMN_SUFFIX}"
        if wc_src.is_file():
            if not (wc_target.exists() and wc_target.resolve() == wc_src.resolve()):
                shutil.copy2(wc_src, wc_target)
            water_columns[strip] = f"{NAV_DIRNAME}/{wc_target.name}"
        renamed = wc_block and wc_block.get("path") != wc_target.name and wc_src.is_file()
        if target.exists() and target.resolve() == src.resolve():
            pass
        elif renamed:
            # The strip was renamed to stay unique; keep the sidecar's own
            # pointer to its water column true.
            payload = dict(payload, water_column=dict(wc_block, path=wc_target.name))
            target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        else:
            shutil.copy2(src, target)
        copied[strip] = f"{NAV_DIRNAME}/{target.name}"

    # --- references: explicit navigation per strip, else the sidecar ------
    references: dict[str, Georeference] = {}
    explicit_source: dict[str, Any] | None = None
    if nav is not None:
        try:
            references, explicit_source = build_references(nav, sizes)
        except NavigationError as exc:
            # Navigation that matches none of the strips is an error for a
            # survey of images. Where sidecars locate the rest, it is recorded
            # and the sidecar strips still carry their recorded positions.
            if not sidecars or "matched none" not in str(exc):
                raise
            explicit_source = {"source": (Path(nav).name if isinstance(nav, (str, Path))
                                          else "navigation supplied by the caller"),
                               "mode": (nav.get("mode") if isinstance(nav, dict)
                                        else "control_points"),
                               "error": str(exc)}

    ping_strips = {s: str(p) for s, p in sidecar_src.items() if s not in references}
    ping_source: dict[str, Any] | None = None
    if ping_strips:
        try:
            ping_refs, ping_source = build_references(
                {"mode": "ping", "sidecars": ping_strips}, {s: sizes[s] for s in ping_strips})
            references.update(ping_refs)
        except NavigationError as exc:
            ping_source = {"mode": "ping", "error": str(exc)}
            log.error("navigation sidecars could not be used: %s", exc)

    if ping_source is not None and "error" not in ping_source:
        nav_source = dict(ping_source)
        nav_source["sidecars"] = {s: copied[s] for s in ping_strips if s in copied}
        nav_source["water_columns"] = {s: water_columns[s] for s in ping_strips
                                       if s in water_columns}
        if explicit_source is not None:
            nav_source["other_navigation"] = explicit_source
        nav_source["strips_located"] = sorted(references)
        nav_source["strips_unlocated"] = sorted(set(sizes) - set(references))
        nav_source["across_track_resolved"] = all(
            ref.detail.get("across_track_resolved", True) for ref in references.values())
        nav_source["references"] = [ref.describe() for ref in references.values()]
    elif explicit_source is not None:
        nav_source = explicit_source
        if copied:
            nav_source["ping_sidecars_not_used"] = copied
            if water_columns:
                nav_source["water_columns"] = water_columns
    else:
        nav_source = {"source": None, "mode": "none",
                      "note": "No navigation supplied. Positions are relative survey pixels."}
        if ping_source is not None:
            nav_source["ping_navigation_error"] = ping_source.get("error")
            nav_source["ping_sidecars_not_used"] = copied
        if water_columns:
            nav_source["water_columns"] = water_columns

    if references:
        log.info("navigation: %s, located %d of %d strips",
                 nav_source["mode"], len(references), len(sizes))
    else:
        log.info("navigation: none supplied, survey is in relative pixel coordinates")

    # --- per-strip resolution and quality ---------------------------------
    strip_extra: dict[str, dict[str, Any]] = {}
    for path in paths:
        strip = names[path]
        payload = sidecars.get(strip)
        if payload is not None:
            processing = payload.get("processing") or {}
            quality = [str(r.get("quality")) for r in payload.get("rows") or []]
            despeckled = (processing.get("speckle") or {}).get("filter") == "lee"
            strip_extra[strip] = {
                "resolution": (payload.get("m_per_px_across"), payload.get("m_per_px_along")),
                "resolution_basis": "navigation sidecar",
                "row_quality": quality if len(quality) == sizes[strip][1] else None,
                "despeckled": despeckled,
                "survey": {
                    "source_file": payload.get("source_file"),
                    "source_format": payload.get("source_format"),
                    "synthetic": bool(payload.get("synthetic")),
                    "sidecar": copied.get(strip),
                    "water_column": water_columns.get(strip),
                    "nadir_col": payload.get("nadir_col"),
                    "coordinate_units": payload.get("coordinate_units"),
                    "degraded_rows": payload.get("degraded_rows") or [],
                    "rows_by_quality": (processing.get("dropouts") or {}).get("rows_by_quality"),
                },
            }
        else:
            across, along = _split_resolution(m_per_px, strip)
            strip_extra[strip] = {
                "resolution": (across, along),
                "resolution_basis": ("supplied by the caller (nav m_per_px)"
                                     if across is not None else None),
                "row_quality": None, "despeckled": False, "survey": {},
            }

    rows: list[dict[str, Any]] = []
    skipped = 0
    denoise_skipped: list[str] = []

    for path in paths:
        strip = names[path]
        extra = strip_extra[strip]
        # An ingested strip was Lee-filtered at ingest; filtering its tiles a
        # second time would only blur it, so DENOISE leaves it alone and the
        # survey block says so.
        denoise = cfg.DENOISE and not extra["despeckled"]
        if cfg.DENOISE and extra["despeckled"]:
            denoise_skipped.append(strip)
        # A survey is a batch, and one bad file in it is a bad file, not a
        # failed survey. A truncated strip, or a JPEG that never finished
        # copying, is skipped with its reason recorded and the rest of the
        # survey still runs. Silence would be worse than either failure mode,
        # so the skip and its reason go into the manifest and the log.
        try:
            skipped += _tile_strip(path, strip, sizes[strip], references.get(strip),
                                   tiles_dir, rows, resolution=extra["resolution"],
                                   row_quality=extra["row_quality"], denoise=denoise)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            unreadable.append({"strip": strip, "source_image": path.name, "error": reason})
            log.error("strip %s could not be read and was skipped: %s", path.name, reason)

    if unreadable and not rows:
        raise StripUnreadableError(
            "no survey strip could be read. "
            + "; ".join(f"{u['source_image']} ({u['error']})" for u in unreadable))

    manifest_csv = out_dir / cfg.MANIFEST_CSV
    manifest_json = out_dir / cfg.MANIFEST_JSON
    columns = list(cfg.MANIFEST_REQUIRED_COLUMNS) + [
        c for c in (rows[0] if rows else {}) if c not in cfg.MANIFEST_REQUIRED_COLUMNS]

    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        # An empty cell, not the string "None": a reader that sees "None" in a
        # latitude column has been handed a value that looks like data.
        writer.writerows([{k: ("" if v is None else v) for k, v in row.items()} for row in rows])

    strips_block = []
    for p in paths:
        strip = names[p]
        extra = strip_extra[strip]
        entry = {"strip": strip, "source_image": p.name,
                 "width": sizes[strip][0], "height": sizes[strip][1],
                 "m_per_px_across": extra["resolution"][0],
                 "m_per_px_along": extra["resolution"][1],
                 "resolution_basis": extra["resolution_basis"]}
        entry.update(extra["survey"])
        strips_block.append(entry)

    survey = {
        "generated_at": started.isoformat(timespec="seconds"),
        "engine": cfg.ENGINE_NAME,
        "processing_version": cfg.PROCESSING_VERSION,
        "strips": strips_block,
        "tiles_written": len(rows),
        "tiles_skipped_low_content": skipped,
        "strips_unreadable": unreadable,
        "coordinate_mode": cfg.COORD_MODE_GEO if references else cfg.COORD_MODE_RELATIVE,
        "navigation": nav_source,
        "tiling": {"tile": cfg.TILE, "stride": cfg.STRIDE, "overlap": cfg.TILE - cfg.STRIDE,
                   "min_content": cfg.MIN_CONTENT, "denoise": cfg.DENOISE,
                   "denoise_filter": "Lee speckle filter + percentile stretch",
                   "content_score": "standard deviation of the tile's grey levels / 255, "
                                    "measured on the tile as written"},
    }
    if denoise_skipped:
        survey["tiling"]["denoise_skipped_already_despeckled"] = sorted(denoise_skipped)
    if ingested:
        survey["ingested"] = [
            {"source_file": r["source_file"], "strip": r["strip"], "image": r["image"],
             "sidecar": copied.get(r["strip"]), "water_column": water_columns.get(r["strip"])}
            for r in ingested]
    synthetic = sorted(s for s, e in strip_extra.items() if e["survey"].get("synthetic"))
    if synthetic:
        survey["synthetic_strips"] = synthetic
        survey["synthetic_warning"] = (
            "SYNTHETIC DATA: these strips come from files marked synthetic by their own "
            "headers (for example tools/make_synthetic_xtf.py). No sonar recorded them and "
            "nothing on them is evidence of anything.")
    manifest_json.write_text(
        json.dumps({"survey": survey, "tiles": rows}, indent=2), encoding="utf-8")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info("prepared %d strips: %d tiles written, %d skipped below MIN_CONTENT=%.3f, %.2fs",
             len(paths), len(rows), skipped, cfg.MIN_CONTENT, elapsed)
    if not rows:
        log.warning("no tile met MIN_CONTENT=%.3f; lower it or check the strips",
                    cfg.MIN_CONTENT)

    return tiles_dir, manifest_csv


def load_manifest(path: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a manifest back, from either the JSON or the CSV.

    Returns (tile rows, survey metadata). The JSON is preferred because it
    keeps types and carries the survey block; the CSV is accepted so a manifest
    edited in a spreadsheet still loads.
    """
    path = Path(path)
    if path.is_dir():
        path = path / cfg.MANIFEST_JSON if (path / cfg.MANIFEST_JSON).is_file() \
            else path / cfg.MANIFEST_CSV
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload, {}
        return payload.get("tiles", []), payload.get("survey", {})

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value == "" or value is None:
                    row[key] = None
                elif key in ("tile", "strip", "source_image", "quality"):
                    row[key] = value
                elif key == "denoised":
                    row[key] = str(value).strip().lower() in {"1", "true", "yes"}
                else:
                    try:
                        row[key] = int(value) if key in ("x", "y", "width", "height",
                                                         "tile_width", "tile_height") \
                            else float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)

    sidecar = path.parent / cfg.MANIFEST_JSON
    survey = {}
    if sidecar.is_file():
        survey = json.loads(sidecar.read_text(encoding="utf-8")).get("survey", {})
    return rows, survey
