#!/usr/bin/env python3
"""Run a whole survey from the command line.

    python3 run_survey.py --strips survey/ --model models/known.pt --out out/

    # with four-corner navigation
    python3 run_survey.py --strips survey/ --model models/known.pt --out out/ \
        --corners 12.92,74.85 12.92,74.862 12.908,74.85 12.908,74.862

    # with a navigation CSV
    python3 run_survey.py --strips survey/ --model models/known.pt --out out/ \
        --nav nav.csv

    # raw side-scan logs: ingested, slant-range corrected, located per ping
    python3 run_survey.py --strips survey/line01.xtf survey/line02.jsf \
        --model models/known.pt --out out/

    # tiles already prepared, just rebuild the hazard map
    python3 run_survey.py --tiles out/tiles --model models/known.pt --out out/

This is the path to use before a demo. Processing a survey takes as long as it
takes, and doing it live behind an HTTP request is how a demo ends up watching
a spinner. Generate export.json ahead of time and let the interface load the
artefact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from survey_hazard_map.hazard_map import build_hazard_map
from survey_hazard_map.hazard_mapview import render_map
from survey_hazard_map.sonar_ingest import SONAR_SUFFIXES
from survey_hazard_map.survey_preparation import STRIP_SUFFIXES, prepare_survey


def _corner_pairs(values: list[str]) -> dict:
    """--corners TL TR BL BR, each as lat,lon."""
    if len(values) != 4:
        raise SystemExit("--corners needs exactly four lat,lon pairs: TL TR BL BR")
    keys = ("top_left", "top_right", "bottom_left", "bottom_right")
    corners = {}
    for key, value in zip(keys, values):
        try:
            lat, lon = (float(part) for part in value.split(","))
        except ValueError:
            raise SystemExit(f"--corners entry {value!r} must be 'latitude,longitude'")
        corners[key] = [lat, lon]
    return corners


def _report_ingest(out_dir: Path) -> None:
    """Say where ingested strips, sidecars and water columns went."""
    from survey_hazard_map.hazard_config import MANIFEST_JSON

    try:
        survey = json.loads((out_dir / MANIFEST_JSON).read_text(encoding="utf-8"))["survey"]
    except (OSError, ValueError, KeyError):
        return
    for item in survey.get("ingested", []):
        print(f"ingested {item['source_file']} -> {out_dir / item['image']}"
              + (f", navigation {out_dir / item['sidecar']}" if item.get("sidecar") else "")
              + (f", water column {out_dir / item['water_column']}"
                 if item.get("water_column") else ""))
    for bad in survey.get("strips_unreadable", []):
        if Path(bad.get("source_image", "")).suffix.lower() in SONAR_SUFFIXES:
            print(f"could not ingest {bad['source_image']}: {bad['error']}")
    if survey.get("synthetic_strips"):
        print("SYNTHETIC DATA in this survey: " + ", ".join(survey["synthetic_strips"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-scan survey to ranked hazard map.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--strips", nargs="+",
                        help="survey strip images, raw .xtf/.jsf side-scan logs, or a "
                             "directory of either")
    parser.add_argument("--tiles", help="skip preparation and use these tiles")
    parser.add_argument("--model", required=True, nargs="+", metavar="CHECKPOINT",
                        help="YOLOv8 checkpoint. Give several to run them all over "
                             "every tile: DeepEcho's known.pt names what it "
                             "recognises and anomaly.pt carries the 'other' class "
                             "that is the unidentified-object signal.")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--nav", help="navigation CSV of pixel-to-lat/lon fixes")
    parser.add_argument("--corners", nargs="+", metavar="LAT,LON",
                        help="four strip corners: top-left top-right bottom-left "
                             "bottom-right. Applies to every strip in the run.")
    parser.add_argument("--conf", type=float, help="override CONF_THRESH")
    parser.add_argument("--merge-dist", type=float, help="override MERGE_DIST")
    parser.add_argument("--grid", type=int, help="override GRID")
    parser.add_argument("--top-n", type=int, help="limit actions.csv to N hotspots")
    parser.add_argument("--no-map", action="store_true", help="skip map.html")
    parser.add_argument("--basemap", action="store_true",
                        help="include a street basemap layer. Georeferenced surveys "
                             "only, needs a network, and is off at open regardless.")
    parser.add_argument("--title", help="survey name shown in the map header")
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)-7s %(message)s")

    if not args.strips and not args.tiles:
        parser.error("give --strips to prepare a survey, or --tiles to reuse one")
    if args.nav and args.corners:
        parser.error("--nav and --corners are two ways to say the same thing; pick one")

    out_dir = Path(args.out)
    nav = None
    if args.nav:
        nav = args.nav
    elif args.corners:
        # One corner set applied to every strip. build_references matches it by
        # name where it can, and a single-strip run needs no name at all.
        nav = {"mode": "corners", "strips": {}, "_corners": _corner_pairs(args.corners)}

    if args.strips:
        if nav and "_corners" in nav:
            corners = nav.pop("_corners")
            strips = [Path(p) for p in args.strips]
            names = []
            for entry in strips:
                candidates = sorted(entry.iterdir()) if entry.is_dir() else [entry]
                # Corners are for images only. A raw log carries its own
                # per-ping navigation, and a typed rectangle must not replace it.
                names.extend(q.stem for q in candidates
                             if q.suffix.lower() in STRIP_SUFFIXES)
            if not names:
                parser.error("--corners applies to strip images; raw .xtf/.jsf logs "
                             "carry their own navigation")
            nav["strips"] = {name: corners for name in names}
        tiles_dir, manifest = prepare_survey(args.strips, out_dir, nav=nav)
        _report_ingest(out_dir)
    else:
        tiles_dir, manifest = Path(args.tiles), None

    models = args.model if len(args.model) > 1 else args.model[0]
    export = build_hazard_map(models, tiles_dir, out_dir, manifest,
                              conf=args.conf, merge_dist=args.merge_dist,
                              grid=args.grid, top_n=args.top_n, title=args.title)

    map_path = None
    if not args.no_map:
        map_path = render_map(export, out_dir, tiles_dir, manifest,
                              title=args.title, basemap=args.basemap)

    summary = export["survey_summary"]
    top = summary["highest_priority_hotspot"]
    print(f"\n{summary['total_deduplicated_detections']} objects in "
          f"{summary['total_hotspots']} hotspots, "
          f"{summary['coordinate_mode'].lower()}")
    if top:
        print(f"start at {top['hotspot_id']}: {top['detection_count']} detections, "
              f"dominant class {top['dominant_class']}, {top['recommended_action'].lower()}")
    written = ["export.json", "actions.csv"] + (["map.html"] if map_path else [])
    print(f"written to {out_dir}/: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
