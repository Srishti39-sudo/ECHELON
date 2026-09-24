"""Module 2. A tile set in, a ranked hazard picture out.

    from survey_hazard_map.hazard_map import build_hazard_map
    export = build_hazard_map("models/known.pt", "out/tiles", "out/")

Nothing in this module imports a dashboard, a web framework, a retriever or a
language model. It is a library, it runs offline, and the only heavy dependency
is whatever the detector needs. That separation is the point:

    THE HAZARD MAP ANSWERS   where things are and how urgent they are
    THE RAG ASSISTANT ANSWERS what a thing is and what is known about it

Those two questions have different evidence and different failure modes. A
severity score is arithmetic over a detector's output and is reproducible. A
grounded answer about an object is retrieval over a document corpus. Merging
them would let a confident sentence raise a priority, or a priority imply a
fact, and neither is something either system can support.

THE PIPELINE
    tiles + manifest
      -> detector, loaded once, run over every tile      hazard_detect
      -> raw boxes into survey coordinates               hazard_coords
      -> severity = class_weight * confidence            hazard_severity
      -> duplicates from overlapping tiles merged        hazard_dedup
      -> geographic position attached, or null           hazard_geo
      -> detections aggregated into ranked hotspots      hazard_hotspots
      -> export.json and actions.csv                     hazard_export
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_coords import attach_geo, to_global
from survey_hazard_map.hazard_dedup import deduplicate, merge_across_models
from survey_hazard_map.hazard_detect import Detector, UltralyticsDetector, list_tiles, run_detector
from survey_hazard_map.hazard_export import (build_export, build_summary, run_configuration,
                           safe_model_reference, utc_now, write_actions, write_export,
                           write_reports)
from survey_hazard_map.hazard_geo import build_references, references_from_manifest
from survey_hazard_map.hazard_hotspots import build_hotspots
from survey_hazard_map.hazard_severity import apply_confidence_floor, score_detection
from survey_hazard_map.hazard_strips import load_sidecars, rebuild_strips, strip_resolutions
from survey_hazard_map.survey_preparation import load_manifest

# Both arrive with raw sonar ingest. Resolved once here rather than assumed, so
# the engine still runs over an image survey on a checkout that predates them.
try:
    from survey_hazard_map.hazard_geo import references_for_survey
except ImportError:  # pragma: no cover - depends on the checkout
    references_for_survey = None
try:
    from survey_hazard_map.hazard_coords import attach_dimensions
except ImportError:  # pragma: no cover - depends on the checkout
    attach_dimensions = None

log = logging.getLogger("deepecho.hazard")


def _find_manifest(tiles_dir: Path, manifest: Any) -> Path | None:
    """The manifest the caller named, or the one sitting beside the tiles."""
    if manifest is not None:
        return Path(manifest)
    for candidate in (tiles_dir.parent / cfg.MANIFEST_JSON,
                      tiles_dir.parent / cfg.MANIFEST_CSV,
                      tiles_dir / cfg.MANIFEST_JSON,
                      tiles_dir / cfg.MANIFEST_CSV):
        if candidate.is_file():
            return candidate
    return None


def _inherit_tile_positions(detections: list[dict[str, Any]],
                            rows: dict[str, dict[str, Any]],
                            already_located: set[str]) -> int:
    """Give a detection its tile's recorded fix when its strip could not be fitted.

    Used only for a strip with too few located tiles to fit a transform. The
    position is a real recorded one, just coarser than the detection: it is the
    centre of the tile, not the object inside it.
    """
    inherited = 0
    for detection in detections:
        if detection.get("latitude") is not None:
            continue
        if str(detection.get("strip") or "") in already_located:
            continue
        row = rows.get(detection["representative_tile"])
        if not row or row.get("lat") in (None, "") or row.get("lon") in (None, ""):
            continue
        detection["latitude"] = float(row["lat"])
        detection["longitude"] = float(row["lon"])
        detection["position_precision"] = "tile centre"
        inherited += 1
    return inherited


EventCallback = Callable[[dict[str, Any]], Any]


def _emit(on_event: EventCallback | None, event: dict[str, Any]) -> None:
    """Hand one progress event to an observer, and survive whatever it does.

    The observer is how a live interface watches a survey run -- in DeepEcho,
    backend/survey_job.py appending events to a file a browser tails. It is
    strictly a spectator. A raising callback is logged and the survey carries
    on, because the survey is the product and the progress display is not; an
    export that fails to exist because a log line could not be written would
    be the wrong way round.
    """
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:  # an observer's bug is not the survey's failure
        log.exception("on_event callback raised on a %r event; the survey continues",
                      event.get("type"))


def _provisional_event(raw: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    """One raw box, as the live view is allowed to see it before merging.

    Provisional in the plain sense: this is a single detector call on a single
    tile, before cross-model merging, class floors, deduplication or scoring.
    Two overlapping tiles will show the same object twice here, and that is
    correct for this moment in the run. The final detections replace these.

    The position is computed with the same per-strip transform the export will
    use when one exists, and is null otherwise. It is never approximated from
    a tile or from anything else, so a provisional marker on a map is where the
    export will also put it, or it is not on a map at all.
    """
    x1, y1, x2, y2 = (float(v) for v in raw["bbox_tile"])
    tile_x, tile_y = float(raw["tile_x"]), float(raw["tile_y"])
    bbox_global = [round(tile_x + x1, 2), round(tile_y + y1, 2),
                   round(tile_x + x2, 2), round(tile_y + y2, 2)]
    gx, gy = tile_x + (x1 + x2) / 2.0, tile_y + (y1 + y2) / 2.0

    latitude = longitude = None
    reference = references.get(str(raw.get("strip") or ""))
    if reference is not None:
        try:
            latitude, longitude = reference.locate(gx, gy)
        except Exception:  # a provisional position is not worth a failure
            latitude = longitude = None

    return {
        "type": "detection",
        "provisional": True,
        "id": raw["id"],
        "class": raw["class"],
        "confidence": raw["confidence"],
        "detector_model": raw.get("detector_model"),
        "strip": raw.get("strip"),
        "tile": raw.get("tile"),
        "bbox_global": bbox_global,
        "global_x": round(gx, 2),
        "global_y": round(gy, 2),
        "latitude": latitude,
        "longitude": longitude,
    }


def build_hazard_map(model_path: Any, tiles_dir: Any, out_dir: Any, manifest: Any = None,
                     *, detector: Detector | None = None, nav: Any = None,
                     conf: float | None = None, merge_dist: float | None = None,
                     grid: int | None = None, top_n: int | None = None,
                     demo: bool = False, title: str | None = None,
                     on_event: EventCallback | None = None,
                     verify: bool = True) -> dict[str, Any]:
    """Detect, deduplicate, score, aggregate and export one survey.

    model_path
        A YOLOv8 checkpoint. Loaded once, for the whole run.
    tiles_dir
        The directory prepare_survey() wrote.
    out_dir
        export.json and actions.csv are written here.
    manifest
        The manifest path. Found automatically beside the tiles if omitted.
        Without one, tile positions are recovered from the {strip}_{x}_{y}
        filenames and there is no geographic position at all.

    detector
        A callable replacing the built-in loader, for a checkpoint hosted
        elsewhere -- in DeepEcho it is the subprocess worker that keeps torch
        away from faiss. See hazard_detect for the interface.
    nav
        Navigation for this run, in prepare_survey's formats. Normally omitted:
        positions come from the manifest. Supply it to use the original
        transform instead of the one refitted from the manifest.
    conf, merge_dist, grid, top_n
        Per-run overrides of CONF_THRESH, MERGE_DIST, GRID and TOP_N_HOTSPOTS.
    title
        A human label for the survey, shown wherever one is listed. Separate
        from survey_id, which is an identifier and stays stable.
    demo
        Marks every output as synthetic. Set by demo_survey.py and by nothing
        else. It is a first-class field rather than a note added afterwards,
        so a demo export cannot be mistaken for a real one by a consumer that
        only reads the JSON.
    on_event
        An optional observer, called with one dict per event so a live view can
        follow the run. Omitted, nothing below changes in any way. Given, it
        receives, in order:
            {"type": "stage", "stage": "detect", "message"}
            {"type": "progress", "tiles_done", "tiles_total"}        per tile
            {"type": "detection", "provisional": True, ...}          per raw box
            {"type": "stage", "stage": "dedup" | "geo" | "hotspots" | "export"}
            {"type": "final_detections", "detections": [...]}        after export
        The final list is export["detections"], so the live view ends on the
        same objects the artefact holds. An observer that raises is logged and
        ignored; see _emit.
    verify
        Run hazard_verify over the deduplicated detections: acoustic shadow,
        rock clutter, nadir and dropout evidence fused into confidence_pct,
        with artefacts marked suppressed and kept out of hotspots. On by
        default because the problem it answers -- shadows, rocks and nadir
        lines read as objects -- is present in every real survey. Off, every
        detection still carries confidence_pct, taken from the detector and
        labelled as unverified.

    Returns the export dictionary, the same object written to export.json.
    """
    started = time.perf_counter()
    tiles_dir = Path(tiles_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs ------------------------------------------------------------
    manifest_path = _find_manifest(tiles_dir, manifest)
    rows: list[dict[str, Any]] = []
    survey_meta: dict[str, Any] = {}
    if manifest_path is not None:
        rows, survey_meta = load_manifest(manifest_path)
        log.info("manifest: %s, %d tiles", manifest_path.name, len(rows))
    else:
        log.warning("no manifest found beside %s; tile positions will be read from "
                    "filenames and no geographic position is available", tiles_dir)

    by_tile = {str(row["tile"]): row for row in rows}
    tiles = list_tiles(tiles_dir)
    if not tiles:
        raise FileNotFoundError(f"no tile images in {tiles_dir}")

    # --- detection ---------------------------------------------------------
    if detector is None:
        # .onnx checkpoints run through onnxruntime with no torch at all, which
        # is what an edge or onboard deployment needs; .pt stays on Ultralytics.
        # make_detector dispatches on the extension. Without onnxruntime
        # installed the .pt path is unchanged.
        try:
            from survey_hazard_map.hazard_detect_onnx import make_detector
        except ImportError:
            detector = UltralyticsDetector(model_path, conf=conf)
        else:
            detector = make_detector(model_path, conf=conf)
    threshold = float(getattr(detector, "conf", cfg.CONF_THRESH if conf is None else conf))

    on_tile = None
    if on_event is not None:
        # The transform used for provisional positions. Built here only when
        # someone is watching, so a run with no observer does exactly the work
        # it always did. It is the same construction the geo stage below uses,
        # so a provisional marker and its final position agree.
        try:
            if nav is not None:
                early_sizes = {str(row["strip"]): (int(row.get("width") or 0),
                                                   int(row.get("height") or 0))
                               for row in rows}
                live_references, _ = build_references(nav, early_sizes)
            else:
                live_references, _ = references_from_manifest(rows)
        except Exception:
            log.exception("provisional positions unavailable; live detections "
                          "will carry null latitude and longitude")
            live_references = {}

        def on_tile(index: int, total: int, name: str, boxes: list[dict]) -> None:
            _emit(on_event, {"type": "progress", "tiles_done": index,
                             "tiles_total": total, "tile": name})
            for box in boxes:
                _emit(on_event, _provisional_event(box, live_references))

        _emit(on_event, {"type": "stage", "stage": "detect",
                         "message": f"running {getattr(detector, 'name', 'the detector')} "
                                    f"over {len(tiles)} tiles"})

    raw, failed_tiles = (run_detector(detector, tiles, by_tile) if on_tile is None
                         else run_detector(detector, tiles, by_tile, on_tile=on_tile))
    raw_count = len(raw)

    # --- survey coordinates and severity ----------------------------------
    to_global(raw)
    # Two checkpoints seeing one object in one tile is folded first, before
    # anything counts detections. Otherwise a wreck both models found is two
    # contacts and doubles its own hotspot's severity.
    raw = merge_across_models(raw)
    # The floor runs BEFORE scoring, because withholding a class changes what
    # the object is called and therefore what it weighs. It never drops a box.
    withheld = 0
    for detection in raw:
        apply_confidence_floor(detection)
        withheld += "class_withheld" in detection
        score_detection(detection)
    if withheld:
        log.info("%d detection(s) had their class withheld for sitting below its "
                 "per-class confidence floor; all are still reported as unidentified",
                 withheld)

    # --- deduplication -----------------------------------------------------
    _emit(on_event, {"type": "stage", "stage": "dedup",
                     "message": f"merging {raw_count} raw boxes seen across "
                                f"overlapping tiles"})
    detections = deduplicate(raw, merge_dist=merge_dist)

    # --- geographic position, or an honest null ---------------------------
    _emit(on_event, {"type": "stage", "stage": "geo",
                     "message": f"attaching positions to {len(detections)} "
                                f"deduplicated detections where navigation allows"})
    if nav is not None:
        sizes = {str(row["strip"]): (int(row.get("width") or 0), int(row.get("height") or 0))
                 for row in rows}
        references, nav_source = build_references(nav, sizes)
        nav_source["source"] = f"{nav_source.get('source')} (supplied to build_hazard_map)"
    elif references_for_survey is not None and manifest_path is not None:
        # Prefers the per-ping navigation a raw sonar log recorded, which
        # resolves across-track position exactly, over refitting a transform
        # to tile centres. Falls back to the refit for image surveys.
        references, nav_source = references_for_survey(rows, survey_meta, manifest_path.parent)
    else:
        references, nav_source = references_from_manifest(rows)

    attach_geo(detections, references)
    inherited = _inherit_tile_positions(detections, by_tile, set(references))
    if inherited:
        nav_source["detections_inheriting_tile_position"] = inherited
        log.info("%d detection(s) took their tile's recorded position because their "
                 "strip had too few fixes to fit a transform", inherited)

    georeferenced = any(d.get("latitude") is not None for d in detections)
    coordinate_mode = cfg.COORD_MODE_GEO if georeferenced else cfg.COORD_MODE_RELATIVE

    # --- size in metres, where the strip's resolution is known ------------
    manifest_dir = manifest_path.parent if manifest_path is not None else tiles_dir.parent
    sidecars = load_sidecars(survey_meta, manifest_dir)
    resolutions = strip_resolutions(sidecars, rows)
    if attach_dimensions is not None:
        attach_dimensions(detections, resolutions)

    # --- verification: shadow, clutter, nadir, dropouts -> confidence_pct --
    # After deduplication, so each physical object is judged once, and before
    # hotspots, so a detection the evidence says is an artefact does not raise
    # a hotspot's priority. Nothing is deleted: a suppressed detection stays in
    # the export with its reasons.
    verification_summary: dict[str, Any] = {"ran": False}
    if verify and detections:
        _emit(on_event, {"type": "stage", "stage": "verify",
                         "message": "checking every detection for an acoustic shadow, "
                                    "rock clutter, nadir artefacts and motion dropouts"})
        try:
            from survey_hazard_map.hazard_verify import load_calibration, strip_context, verify_survey

            greys = rebuild_strips(rows, tiles_dir)
            contexts = {strip: strip_context(None, sidecars.get(strip), strip=strip, grey=grey)
                        for strip, grey in greys.items()}
            calibration = load_calibration()
            verification_summary = {"ran": True, **verify_survey(detections, contexts,
                                                                 calibration)}
            verification_summary["calibration"] = (
                "none found; confidence_pct starts from the raw detector confidence"
                if calibration is None else "models/calibration.json")
        except Exception as exc:
            # A verification failure costs the evidence, not the survey. The
            # export says it did not run rather than pretending it did.
            log.exception("verification failed; detections are exported unverified")
            verification_summary = {"ran": False,
                                    "error": f"{type(exc).__name__}: {exc}"}

    for detection in detections:
        # Every exported detection states a 0-100 confidence. Unverified, it is
        # the detector's own number and the basis says so.
        if "confidence_pct" not in detection:
            detection["confidence_pct"] = round(float(detection["confidence"]) * 100.0, 1)
            detection["confidence_pct_basis"] = "detector confidence, not verified"
        detection.setdefault("suppressed", False)

    reportable = [d for d in detections if not d.get("suppressed")]

    # --- hotspots ----------------------------------------------------------
    _emit(on_event, {"type": "stage", "stage": "hotspots",
                     "message": "aggregating detections into ranked hotspots"})
    hotspots = build_hotspots(reportable, references, grid=grid)

    # --- export ------------------------------------------------------------
    _emit(on_event, {"type": "stage", "stage": "export",
                     "message": "writing export.json and actions.csv"})
    strips = sorted({str(row["strip"]) for row in rows} or
                    {str(d.get("strip") or "") for d in detections})
    summary = build_summary(
        raw_count=raw_count, detections=detections, hotspots=hotspots,
        strips=strips, tiles=len(tiles), georeferenced=georeferenced,
        coordinate_mode=coordinate_mode)

    model_reference = safe_model_reference(model_path)
    elapsed = time.perf_counter() - started

    configuration = run_configuration()
    configuration["detection"]["CONF_THRESH"] = threshold
    if merge_dist is not None:
        configuration["deduplication"]["MERGE_DIST"] = float(merge_dist)
    if grid is not None:
        configuration["hotspots"]["GRID"] = int(grid)

    metadata = {
        "engine": cfg.ENGINE_NAME,
        "processing_version": cfg.PROCESSING_VERSION,
        "processed_at": utc_now(),
        "processing_seconds": round(elapsed, 3),
        "survey_id": survey_meta.get("survey_id") or out_dir.resolve().name,
        "title": title or survey_meta.get("title") or out_dir.resolve().name,
        "coordinate_mode": coordinate_mode,
        "confidence_threshold": threshold,
        **model_reference,
        "detector_classes": list(getattr(detector, "classes", []) or []),
        "detector": getattr(detector, "name", type(detector).__name__),
        "demo": demo,
    }
    if demo:
        metadata["data_source"] = "SYNTHETIC DEMO DATA"
        metadata["demo_warning"] = (
            "Every detection in this file was generated by demo_survey.py. No "
            "sonar was recorded, no real object was detected, and nothing here "
            "is evidence of anything.")

    provenance = {
        "demo": demo,
        "processing_version": cfg.PROCESSING_VERSION,
        "severity_policy_version": cfg.SEVERITY_POLICY_VERSION,
        "coordinate_mode": coordinate_mode,
        "coordinate_note": (
            "global_x and global_y are pixel offsets within their own strip and "
            "are present for every detection. Latitude and longitude are null "
            "wherever navigation does not support one, and are never inferred."),
        "navigation": nav_source,
        "source_strips": survey_meta.get("strips", [{"strip": s} for s in strips]),
        "tile_count": len(tiles),
        "tiles_in_manifest": len(rows),
        "tiles_processed": len(tiles) - len(failed_tiles),
        # Empty on a clean run. A populated list says the survey is incomplete
        # and names which tiles are missing from it.
        "tiles_failed": failed_tiles,
        "manifest": None if manifest_path is None else manifest_path.name,
        "tiling": survey_meta.get("tiling", {}),
        "model": model_reference,
        "confidence_threshold": threshold,
        "severity_formula": "severity = class_weight * confidence",
        "class_confidence_floors": dict(cfg.CLASS_CONFIDENCE_FLOOR),
        "class_floor_rule": (
            "a detection below its class's floor is relabelled "
            f"'{cfg.DOWNGRADE_LABEL}' and never dropped; the original call is "
            "kept in the detection's downgraded_from. This changes the claim, "
            "not the direction of the score: it lowers severity only for "
            "classes weighted above UNKNOWN_CLASS_SEVERITY."),
        "class_floors_withheld": withheld,
        "ranking_rule": "hotspots sorted by total_severity descending; "
                        "H001 is the highest priority",
        "audit_note": (
            "Every hotspot carries a `rationale` naming the detection that set "
            "its severity, and every detection carries the tiles it was seen in "
            "and how many views were merged into it."),
        "verification": verification_summary,
        "suppression_rule": (
            "a detection marked suppressed is kept in `detections` with its "
            "reasons and excluded from hotspots and the action list; nothing "
            "is deleted"),
        "strip_resolutions_m_per_px": {s: {"across": a, "along": b}
                                       for s, (a, b) in resolutions.items()},
    }

    export = build_export(metadata=metadata, summary=summary, detections=detections,
                          hotspots=hotspots, provenance=provenance,
                          configuration=configuration)

    export_path = write_export(out_dir, export)
    actions_path = write_actions(out_dir, hotspots, top_n=top_n)
    write_reports(out_dir, export)

    tiers = summary["detections_by_tier"]
    hotspot_tiers = summary["hotspots_by_tier"]
    log.info(
        "survey complete | strips=%d tiles=%d raw=%d deduplicated=%d duplicates_removed=%d "
        "| detections critical=%d medium=%d low=%d "
        "| hotspots=%d (critical=%d medium=%d low=%d) "
        "| total_severity=%.3f coordinates=%s time=%.2fs",
        summary["strips_processed"], summary["tiles_processed"], raw_count,
        len(detections), summary["duplicates_removed"],
        tiers.get("critical", 0), tiers.get("medium", 0), tiers.get("low", 0),
        len(hotspots), hotspot_tiers.get("critical", 0), hotspot_tiers.get("medium", 0),
        hotspot_tiers.get("low", 0),
        summary["total_severity"], coordinate_mode, elapsed)
    if demo:
        log.warning("SYNTHETIC DEMO DATA: this export was generated from a "
                    "simulated survey and is not evidence of anything")
    log.info("wrote %s and %s", export_path.name, actions_path.name)

    # After the files exist, not before: the final list a live view settles on
    # is then the one a download of export.json will also contain.
    _emit(on_event, {"type": "final_detections",
                     "detections": export.get("detections", [])})

    return export
