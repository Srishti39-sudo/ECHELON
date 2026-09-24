#!/usr/bin/env python3
"""The whole pipeline, with no survey and no checkpoint.

    python3 demo_survey.py                 # writes ./demo_out/
    python3 demo_survey.py --out /tmp/demo
    python3 demo_survey.py --geo           # with four-corner navigation

Every stage runs for real: tiling, the manifest, global coordinates, severity,
deduplication, hotspots, ranking, export.json, actions.csv and map.html. The
only simulated part is the two inputs at the very front, the sonar and the
detector, and both are simulated deterministically so the demo is identical
every time it is shown.

NOTHING HERE IS EVIDENCE OF ANYTHING
The output is labelled as synthetic in four independent places, because a
synthetic detection that reads like a real one is worse than no detection at
all: everything downstream treats a record as evidence.

    1  export.json carries  metadata.demo: true, a data_source of
       "SYNTHETIC DEMO DATA", and a demo_warning in words
    2  provenance.demo is true
    3  map.html shows a banner across the top that cannot be dismissed
    4  the run writes to its own directory, defaulting to ./demo_out, and
       refuses to write into a directory that already holds a real export

Point four is the one that matters most. Demo output never lands in the same
directory as real output, so no operator can end up looking at a mixture.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_map import build_hazard_map
from survey_hazard_map.hazard_mapview import render_map
from survey_hazard_map.survey_preparation import prepare_survey

log = logging.getLogger("deepecho.hazard")

# The simulated survey. Deterministic: one seed, one layout, one result.
SEED = 26057
STRIP_W, STRIP_H = 1800, 2600
NADIR_X = STRIP_W // 2
NADIR_HALF_WIDTH = 110

# Simulated contacts, as (x, y, class, confidence).
#
# Laid out to exercise the parts of the picture a demo has to show: a genuine
# priority-one cluster, a wreck that is serious but not urgent, a large low
# severity debris field that must NOT outrank the cluster, an unidentified
# object that must not be dressed up as a confirmed hazard, and one contact
# sitting in a tile overlap so deduplication has something to do on stage.
CONTACTS = [
    # The cluster that should come first: ordnance, plus company. All three sit
    # inside one 512 px grid cell on purpose. Spread across a cell boundary
    # they would split into two hotspots, each correctly ranked and neither
    # showing the point being made.
    (1180.0, 430.0, "mine", 0.88),
    (1240.0, 470.0, "uxo", 0.71),
    (1320.0, 440.0, "debris", 0.55),

    # A wreck site: one serious contact with associated debris and snagging
    # hazards. Ranks second. Serious, and correctly below the ordnance.
    (520.0, 1180.0, "shipwreck", 0.82),
    (610.0, 1250.0, "debris", 0.61),
    (600.0, 1330.0, "chain", 0.48),
    (700.0, 1400.0, "cable", 0.55),

    # Unidentified, alone in its own cell. It must rank in the middle of the
    # table: high enough to be looked at, and never presented as a confirmed
    # hazard the way a mine is.
    (1350.0, 900.0, "unknown", 0.64),

    # Ghost gear, an entanglement risk rather than an explosive one. The drum
    # sits in a tile overlap, so deduplication has something visible to do.
    (1090.0, 1610.0, "drum", 0.66),
    (1430.0, 1980.0, "net", 0.59),
    (1500.0, 2040.0, "net", 0.52),

    # A debris field. The MOST contacts of any cell in this survey, and the
    # whole argument of the demo is that it still ranks below the ordnance
    # cluster and below the wreck. Consequence, not headcount.
    (380.0, 2180.0, "tire", 0.74),
    (450.0, 2230.0, "bottle", 0.69),
    (430.0, 2300.0, "debris", 0.63),
    (350.0, 2260.0, "tire", 0.58),
    (500.0, 2320.0, "bottle", 0.54),
]

CORNERS = {
    "top_left": [12.9260, 74.8480],
    "top_right": [12.9260, 74.8640],
    "bottom_left": [12.9020, 74.8480],
    "bottom_right": [12.9020, 74.8640],
}


def make_sonar_strip(path: Path) -> None:
    """A simulated side-scan waterfall. Deterministic from SEED.

    Drawn to the grammar of a real side-scan record, because a demo image that
    looks nothing like sonar invites the wrong question from a judge:

      * a nadir band down the centre, directly beneath the towfish, where there
        is no seabed return
      * backscatter that strengthens with range as grazing angle falls
      * along-track striping from vessel motion
      * targets as a bright return with an acoustic shadow behind them, and the
        shadow always falls AWAY from the nadir, which is the direction the
        sound came from
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(SEED)
    ys, xs = np.mgrid[0:STRIP_H, 0:STRIP_W]
    across = np.abs(xs - NADIR_X).astype(np.float32)

    # Backscatter rising with range, flattening at the far edge.
    strip = 42.0 + 92.0 * (1.0 - np.exp(-across / 340.0))

    # Speckle. Sonar noise is multiplicative, not additive, so the far field is
    # noisier than the near field, which is what the eye reads as texture.
    strip *= rng.gamma(shape=9.0, scale=1 / 9.0, size=strip.shape)

    # Along-track striping from vessel motion and gain changes.
    strip *= (1.0 + 0.05 * np.sin(ys / 23.0) + 0.03 * rng.normal(0, 1, (STRIP_H, 1)))

    # The nadir band: no seabed return directly under the towfish.
    nadir = np.clip(1.0 - (across / NADIR_HALF_WIDTH) ** 2, 0.0, 1.0)
    strip = strip * (1.0 - nadir) + 16.0 * nadir

    for x, y, label, _confidence in CONTACTS:
        x, y = int(x), int(y)
        # Bigger classes get bigger returns, which is what the operator expects
        # to see beside a class name.
        size = {"shipwreck": 46, "mine": 15, "uxo": 17, "drum": 16,
                "net": 30, "chain": 26, "cable": 28}.get(label, 13)
        direction = 1 if x >= NADIR_X else -1
        highlight = ((xs - x) ** 2 / (size * 1.5) ** 2 + (ys - y) ** 2 / size ** 2) <= 1.0
        strip[highlight] = 236.0

        # The shadow: as long as the object is tall, thrown away from nadir.
        shadow_len = int(size * 3.4)
        x0 = x + direction * int(size * 1.4)
        x1 = x0 + direction * shadow_len
        lo, hi = sorted((max(0, min(x0, x1)), min(STRIP_W, max(x0, x1))))
        strip[max(0, y - size):y + size, lo:hi] *= 0.13

    Image.fromarray(np.clip(strip, 0, 255).astype("uint8"), mode="L").save(path)


class SimulatedDetector:
    """Stands in for a YOLOv8 checkpoint. Reports the contacts as drawn.

    It reads no pixels, and it says so in its name, which is written into
    export.json as the detector that produced the run. There is no path by
    which this output can be mistaken for a model's.
    """

    name = "SIMULATED (demo_survey.py, not a trained model)"
    conf = cfg.CONF_THRESH

    def __init__(self) -> None:
        self.classes = sorted({c for _, _, c, _ in CONTACTS})

    def __call__(self, image_path: Path) -> list[dict]:
        from PIL import Image

        from survey_hazard_map.hazard_detect import parse_tile_name

        parsed = parse_tile_name(image_path.name)
        if parsed is None:
            return []
        _, tile_x, tile_y = parsed
        with Image.open(image_path) as tile:
            tile_w, tile_h = tile.size

        boxes = []
        for gx, gy, label, confidence in CONTACTS:
            lx, ly = gx - tile_x, gy - tile_y
            if not (0 <= lx < tile_w and 0 <= ly < tile_h):
                continue
            if confidence < self.conf:
                continue
            half = 24.0 if label in ("shipwreck", "net", "chain", "cable") else 15.0
            boxes.append({"class": label, "confidence": confidence,
                          "bbox": [lx - half, ly - half * 0.7, lx + half, ly + half * 0.7]})
        return boxes


def _refuse_to_mix(out_dir: Path) -> None:
    """Never write demo output on top of a real survey."""
    export = out_dir / cfg.EXPORT_JSON
    if not export.is_file():
        return
    try:
        existing = json.loads(export.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    if not existing.get("metadata", {}).get("demo"):
        raise SystemExit(
            f"{export} holds a real survey export. Demo output will not be written "
            f"on top of it. Choose another --out directory.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline on a simulated survey.",
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="demo_out", help="output directory")
    parser.add_argument("--geo", action="store_true",
                        help="attach four-corner navigation, to demonstrate the "
                             "geo-referenced path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)-7s %(message)s")

    out_dir = Path(args.out)
    _refuse_to_mix(out_dir)
    survey_dir = out_dir / "simulated_survey"
    survey_dir.mkdir(parents=True, exist_ok=True)

    strip = survey_dir / "DEMO_SYNTHETIC_strip.png"
    log.info("simulating a side-scan strip: %d x %d px, %d contacts",
             STRIP_W, STRIP_H, len(CONTACTS))
    make_sonar_strip(strip)

    nav = {"mode": "corners", "strips": {strip.stem: CORNERS}} if args.geo else None
    tiles_dir, manifest = prepare_survey([strip], out_dir, nav=nav)

    export = build_hazard_map("SIMULATED", tiles_dir, out_dir, manifest,
                              detector=SimulatedDetector(), demo=True,
                              title="Simulated survey DEMO-2026-09")
    map_path = render_map(export, out_dir, tiles_dir, manifest,
                          title="Simulated survey DEMO-2026-09", demo=True)

    summary = export["survey_summary"]
    top = summary["highest_priority_hotspot"]
    print("\n  SYNTHETIC DEMO DATA. Nothing below is a real detection.\n")
    print(f"  {summary['total_deduplicated_detections']} contacts in "
          f"{summary['total_hotspots']} hotspots, "
          f"{summary['duplicates_removed']} duplicate view(s) merged")
    print(f"  {summary['detections_by_tier'].get('critical', 0)} critical, "
          f"{summary['detections_by_tier'].get('medium', 0)} medium, "
          f"{summary['detections_by_tier'].get('low', 0)} low")
    print(f"  coordinates: {summary['coordinate_mode']}")
    if top:
        print(f"  start at {top['hotspot_id']}: {top['dominant_class']}, "
              f"{top['detection_count']} contacts, {top['recommended_action'].lower()}")
    print(f"\n  open {map_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
