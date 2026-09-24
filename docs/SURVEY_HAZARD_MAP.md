# Survey Hazard Intelligence

The hazard-map subsystem for DeepEcho (SIH26057). It takes a side-scan sonar
survey and a trained YOLOv8 checkpoint and produces a ranked, auditable picture
of where the hazards are and which one to look at first.

    THE HAZARD MAP ANSWERS    WHERE things are, and HOW URGENT they are
    THE RAG ASSISTANT ANSWERS WHAT a thing is, and WHAT TO DO about it

Those stay separate throughout. A severity score is arithmetic over a
detector's output and can be recomputed by hand from the record it sits in. A
grounded answer is retrieval over a document corpus, with different evidence and
different ways of being wrong. Merged, a confident sentence could raise a
priority, or a priority could imply a fact, and neither system can support that.

## Three things this will not do

Read these before the numbers, because they govern how much the numbers mean.

**The coordinates are not GPS.** Unless you supply navigation, every position
is a pixel offset inside the sonar strip, derived from survey geometry and
nothing else. The export says `"coordinate_mode": "Relative Survey Coordinates"`,
every latitude and longitude is `null`, and the map never formats anything as a
fix. Supply navigation and geographic positions are interpolated from it, under
documented assumptions stated in the export. Nothing is ever inferred, defaulted
or carried over from a previous survey.

**Severity is a heuristic, not a standard.** The class weights, the tier
boundaries and the recommended actions are a configurable policy chosen for this
project. They are not Navy, Coast Guard, NOAA or IMO procedure and they carry no
authority. They are calibrated against nothing: they are a considered ordering of
consequence, and the per-class confidence floors rest on a single observed false
positive each. Everything in `hazard_config.py` is meant to be retuned against a
labelled validation set. The same sentence is in `configuration.disclaimer` of
every export and in a column of every `actions.csv`, so a file that travels on
its own still carries the caveat.

**A detection is not a fact.** YOLO output is a prediction. What it is worth
depends on the checkpoint, its training data, the quality of the sonar, the
quality of the navigation, whether the class definitions match what is actually
on this seabed, and where the confidence threshold sits. Ten tiles of a real
side-scan waterfall record in this repository produce four detections, and all
four appear to be false positives on nadir and shadow boundaries. The engine
ranks what it is given; it cannot tell you the detector was wrong.

## Pipeline

    Raw .xtf / .jsf side-scan logs          Strip images (.png .jpg .tif)
      │                                        │
      ▼  ingest()                sonar_ingest.py
    strips/{strip}.png + {strip}.nav.json + {strip}.wc.npz
      │                                        │
      └──────────────┬─────────────────────────┘
                     ▼  prepare_survey()                 survey_preparation.py
    Positioned tiles + manifest.csv / manifest.json (+ nav/ sidecars)
      │
      ▼  YOLOv8 or ONNX, one or more checkpoints         hazard_detect.py, hazard_detect_onnx.py
    Tile detections
      │
      ▼  global coordinates, tile-local -> survey        hazard_coords.py
      ▼  cross-model merge, one box per object per tile  hazard_dedup.py
      ▼  class confidence floors, then severity          hazard_severity.py
      ▼  global deduplication across overlapping tiles   hazard_dedup.py
      ▼  geographic position (ping, refit, or null)      hazard_geo.py
      ▼  length_m / width_m where resolution is known    hazard_coords.py, hazard_strips.py
      ▼  verification: confidence_pct, suppressed        hazard_verify.py
      ▼  hotspots over the detections NOT suppressed     hazard_hotspots.py
      │
      ▼  build_hazard_map()                              hazard_map.py
    ┌─────────────────────────────┬────────────────────────────────────────┐
    │ export.json  actions.csv    │  map.html (hazard_mapview.render_map)  │
    │ report.csv   report.geojson │  Dashboard /map  ──►  RAG handoff       │
    └─────────────────────────────┴────────────────────────────────────────┘

The order above is the order `build_hazard_map()` runs its stages in. Two points
in it are easy to misread. Verification runs **after** deduplication, so each
physical object is judged once, and **before** hotspots, so a detection the
evidence marks as an artefact does not raise a hotspot's priority. And nothing
after detection deletes a record: floors relabel, deduplication merges with
provenance, and verification flags.

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python survey_hazard_map/tests/unit_tests.py && ./.venv/bin/python survey_hazard_map/tests/smoke_test.py
```

`numpy` and `pillow` are the only hard dependencies. `folium` is needed to build
`map.html` and never to read one. `ultralytics` is needed only to run a real
checkpoint, because `build_hazard_map()` accepts any callable that returns boxes,
and every test in this repository runs without it.

## Module APIs

### Module 1

```python
from survey_preparation import prepare_survey

tiles_dir, manifest_path = prepare_survey(strip_paths, out_dir, nav=None)
```

| Argument | Meaning |
| --- | --- |
| `strip_paths` | A path, a list of paths, or a directory of images and/or raw `.xtf` / `.jsf` logs. |
| `out_dir` | Created if absent. Tiles go in `out_dir/tiles`, ingested strips in `out_dir/strips`, navigation sidecars in `out_dir/nav`. |
| `nav` | `None`, a path to a control-point CSV, or a dict of four corners. |
| `ingest_options` | Per-run overrides for `sonar_ingest`, e.g. `{"m_per_px_across": 0.1}`. |

From a shell:

```bash
python -m survey_hazard_map.run_survey --strips survey/ --model survey_hazard_map/models/known.pt --out out/
python -m survey_hazard_map.run_survey --strips survey/line01.xtf survey/line02.jsf \
    --model survey_hazard_map/models/known.pt --out out/          # raw logs, located per ping
```

### Raw sonar ingest

`sonar_ingest.py` turns a raw side-scan log into a strip image the rest of the
engine can tile, and records what every row of that image is. XTF is read with
pyxtf; JSF with a minimal reader for message type 80. For each contiguous line
(a long time gap or a change of range setting starts a new strip) it writes:

| File | What it is |
| --- | --- |
| `{strip}.png` | 8-bit waterfall. Port left, starboard right, nadir at column `nadir_col`. Row 0 is the first ping. Pixel value 0 means no data; every measurement is scaled into 1..255. |
| `{strip}.nav.json` | The sidecar, format `deepecho-strip-nav/1`: one record per image row (time, lat, lon, heading, attitude, altitude, depth, quality flag), `m_per_px_across`, `m_per_px_along`, `port_is_left`, degraded row ranges, and a processing block naming every step and parameter. |
| `{strip}.wc.npz` | The water column the slant-range correction removes from the image, kept for GhostTrace. |

The steps, in order: read (positions only when the file says they are degrees;
otherwise null with the reason), split, bottom track, slant-range correction
onto a fixed `m_per_px_across` ground grid, empirical gain normalisation,
along-track resampling to fixed `m_per_px_along` metres of distance travelled,
dropout and attitude flagging per row, a Lee speckle filter, and robust 8-bit
scaling. Not corrected, and written into every sidecar under
`geometry_not_corrected`: layback, pitch and yaw displacement of the footprint,
refraction, layover of tall objects, and a sloping seabed across the swath.
Every threshold is a module constant overridable as `HAZARD_INGEST_<NAME>`.

`prepare_survey()` ingests raw paths first, tiles the resulting PNGs like any
other strip, copies sidecars and water columns to `out_dir/nav/`, and records
them in the manifest's survey block as
`"navigation": {"mode": "ping", "sidecars": {strip: "nav/<file>"}, ...}`. A log
that cannot be ingested is listed under `strips_unreadable` with its reason.

### Module 2

```python
from hazard_map import build_hazard_map

export = build_hazard_map(model_path, tiles_dir, out_dir, manifest=None,
                          detector=None, nav=None, conf=None,
                          merge_dist=None, grid=None, top_n=None,
                          demo=False, title=None)
```

`model_path` takes one checkpoint or several. `detector` replaces the built-in
loader with any callable taking an image path and returning
`[{"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]`, which is how
the engine runs inside DeepEcho without pulling torch into a process that has
already loaded faiss.

Both entry points are importable on their own and neither needs the dashboard,
the assistant, a database, a key or a network.

```bash
python -m survey_hazard_map.run_survey --strips survey/ --model survey_hazard_map/models/known.pt models/anomaly.pt \
    --out out/ --title "Mangaluru approach"
python -m survey_hazard_map.demo_survey                       # the whole pipeline, no survey needed
```

## Configuration

Every threshold lives in `hazard_config.py` and nothing else hard-codes a value.

| Setting | Default | What it does |
| --- | --- | --- |
| `TILE` | 640 | Tile edge, matching the checkpoints' training size. |
| `STRIDE` | 512 | Step between tiles, giving 128 px of overlap. |
| `MIN_CONTENT` | 0.02 | Keep a tile when its grey-level standard deviation over 255 is at least this. |
| `DENOISE` | False | 3x3 median plus a 1st-to-99th percentile stretch. Intensity only. |
| `CONF_THRESH` | 0.30 | Boxes below this are never read out of the model. |
| `CLASS_CONFIDENCE_FLOOR` | per class | Below its floor a class is withheld, not dropped. |
| `DETECTOR_MERGE_IOU` | 0.5 | Cross-model overlap, same tile. |
| `MERGE_DIST` | 60 px | Survey-wide merge distance, same class and strip. |
| `GRID` | 512 px | Hotspot cell edge. |
| `SEVERITY` | table | Class weight, the consequence of being wrong. |
| `UNKNOWN_CLASS_SEVERITY` | 0.8 | A class the table has never seen. Never 0.0. |
| `SEVERITY_TIERS` | 0.75 / 0.40 | critical, medium, low. |
| `RISK_DENSITY_WEIGHT` | 0.25 | How much the objects beyond the worst one add. |
| `TOP_N_HOTSPOTS` | 0 (all) | Rows in `actions.csv`. |

## Tiling and coverage

Tiles are named exactly `{strip}_{x}_{y}.jpg`, where x and y are the tile's
top-left offset in the original strip, in pixels. Not an index. The name parses
back even when the strip's own name contains underscores, though the manifest is
the contract and filenames are only a fallback.

Offsets step by `STRIDE` and stop once the next tile would start past the end, so
the last tile in each direction is partial and every pixel is covered. Because
`STRIDE` is smaller than `TILE`, a partial tile is never narrower than
`TILE - STRIDE`, so a run never produces a useless sliver.

Strips are opened one at a time and each tile is written before the next is cut,
so peak memory is one strip plus one tile rather than the whole survey.

`MIN_CONTENT` is stated exactly because a silent filter is how a contact goes
missing. A tile is kept when

    content_score = standard deviation of the tile's 8-bit grey levels / 255

is at or above the threshold, measured on the tile as written, so what is scored
is what the detector will be shown.

## Coordinate provenance

Without navigation: `lat` and `lon` are `null` everywhere, the mode is
`Relative Survey Coordinates`, and that is a complete, usable result.

**Control-point CSV.** Columns `strip, pixel_x, pixel_y, latitude, longitude`.
Three or more non-collinear fixes fit a first-order affine transform by least
squares, and the worst residual is reported in the export. A nav file with one
fix per ping row is collinear, so an affine fit would be underdetermined
across-track; that falls back to one-dimensional interpolation along the track
line and sets `across_track_resolved: false` rather than guessing a range scale
it does not have.

**Four corners.** `top_left`, `top_right`, `bottom_left`, `bottom_right`, each
`[latitude, longitude]`. A tile centre is interpolated bilinearly between them.

**Ping navigation.** A strip ingested from a raw log is located from its own
sidecar, row by row (`Georeference.from_ping_nav`, mode `"ping"`). For a pixel
position `(x, y)`: the navigation at continuous row `y - 0.5` is interpolated
between the two nearest rows (heading on the circle), the across-track offset
`(x - nadir_col) * m_per_px_across` is applied along `heading + 90°` (negated
when `port_is_left` is false), using a local flat-earth east-north-up step with
WGS84 radii. This is the only mode that resolves across-track position from
the sonar's own geometry rather than from a fit. Rows with a null position are
bridged only across gaps no longer than `HAZARD_PING_NAV_MAX_FILL_M` (30 m by
default); beyond that the position is null. A sidecar whose width and height
differ from the strip's is refused.

`build_hazard_map()` calls `hazard_geo.references_for_survey(rows, survey_meta,
manifest_dir)`: every strip with a readable sidecar recorded in the manifest
gets ping navigation, and every other strip falls back to
`references_from_manifest`, the refit of an affine or along-track transform to
the manifest's located tile centres. A recorded sidecar that is missing is named
in `provenance.navigation.sidecars_missing`, not skipped silently. Explicit
navigation passed as `nav` wins per strip.

All of them assume a locally flat seabed and locally linear degrees over one strip.
Neither is valid across the antimeridian or over a pole, and the export says so.
Every position is computed from the **tile centre**, never its corner: locating a
640-pixel tile by its corner puts it half a tile out, consistently, in one
direction, which is the kind of error that survives review.

Two strips are two coordinate frames. Without navigation nothing says how far
apart they are, so deduplication and hotspots are scoped to a single strip.

## Severity provenance

One formula, and it is the whole of it:

    severity = class_weight * confidence

Every record keeps `class_weight`, `confidence`, `severity` and `severity_basis`
side by side, so any score in the export can be recomputed by hand from the
record itself. Matching is exact on the normalised class name, else the longest
table key contained in it, else `UNKNOWN_CLASS_SEVERITY`. So `moored_mine`,
`sea mine` and `Mine` all reach the mine weight without being listed, and a class
the policy has never been taught is treated as unidentified rather than weighted
out of existence.

A class below its confidence floor is relabelled `unknown` and the original call
is kept in `downgraded_from`. Nothing is dropped, because a deleted box hides a
contact from the operator.

Read this before changing a floor: withholding a class does **not** uniformly
lower severity. It lowers it only for classes weighted above
`UNKNOWN_CLASS_SEVERITY`. A withheld `human` gets quieter, a withheld `aircraft`
gets louder, and both are the policy working. "Downgrade" describes the claim,
not the score.

## Deduplication

Two stages, answering different questions. Neither does the other's job.

| Stage | Question | Test | Merges classes |
| --- | --- | --- | --- |
| Cross-model | did two checkpoints see the same box in this tile? | box overlap (IoU) | yes |
| Survey-wide | did one object appear across overlapping tiles? | centre distance | never |

The cross-model stage merges across classes because the checkpoints do not share
a vocabulary: on a real record of the submarine S-7, `known.pt` called the wreck
"ship" at 0.82 and `anomaly.pt` called the same box "shipwreck" at 0.39,
overlapping at IoU 0.82. Counted separately that is one submarine with double its
severity. The losing call is kept as a structured `second_opinion`.

The survey-wide stage never merges classes, because across tiles two different
classes near each other are two objects. It stays distance-based rather than
overlap-based because a sonar return's box shape varies with range, so the same
object at two ranges would fail an overlap test.

Detections are taken highest-confidence first and each either joins an accepted
representative or becomes one; membership is only tested against representatives.
Without that, A merges B, B merges C, and a line of separate objects collapses
into one contact hundreds of pixels long.

## Verification and confidence

`hazard_verify.py` asks, for every deduplicated detection, whether the strip
image agrees with the detector. The strip is rebuilt from the tiles
(`hazard_strips.rebuild_strips`), so the evidence is measured on exactly the
pixels the detector saw, together with the sidecar where one exists. Each cue is
a score from 0 to 1 with its raw measurements stored beside it:

| Cue | Weight, neutral | What it measures |
| --- | --- | --- |
| `acoustic_shadow` | +1.2, 0.35 | A highlight followed by a shadow on the side away from nadir, as a proud object casts. |
| `man_made_regularity` | +1.0, 0.30 | Straight edges, coherent orientation, mesh texture. |
| `natural_shadow` | -2.5, 0 | A dark box with no bright return: a shadow or depression. |
| `rock_clutter` | -2.5, 0 | Many similar blobs nearby with incoherent orientations. |
| `nadir_zone` | -4.0, 0 | The box sits in the water column or on the nadir line. |
| `dropout` | -2.0, 0 | The box's rows are flagged degraded in the sidecar. |

They are fused with the detector's probability in logit space:

    confidence_pct = 100 * sigmoid( logit(p_calibrated)
                                    + sum(weight * applicability * (score - neutral)) )

`p_calibrated` is the raw detector confidence unless `models/calibration.json`
exists. Every term is written to `verification.terms`, so the percentage can be
recomputed by hand; the detector's own number is kept, untouched, as
`confidence` and as `verification.detector_confidence`.

**Suppression, not deletion.** A detection is marked `suppressed: true` only when
`confidence_pct` is below `SUPPRESS_BELOW_PCT` (35 by default,
`HAZARD_SUPPRESS_BELOW_PCT`) **and** at least one hard reason fired
(`VERIFY_HARD_REASON_AT`: `nadir_zone` 0.5, `natural_shadow` 0.5, `rock_clutter`
0.45, `dropout` 0.25). A faint contact with no artefact evidence against it is
never hidden. A suppressed detection stays in `detections` with plain-English
`verification.reasons` and the `hard_reasons` codes, is counted in
`survey_summary.suppressed_detections`, is excluded from hotspots and
`actions.csv`, and is included and flagged in `report.csv`.

With `verify=False`, or if verification raises, every detection still carries
`confidence_pct` (the detector confidence times 100) with
`confidence_pct_basis: "detector confidence, not verified"`, and
`provenance.verification` records `ran: false` and any error. A detection on a
strip with no image is `status: "not_checked"` and never suppressed.

Every cue is a heuristic. None is a trained classifier and none has been
validated on a labelled sonar benchmark; the output is a better-informed
ranking, not a measured probability. On this repository's public waterfall
strip, verification suppresses both nadir-boundary "shipwreck" detections and
leaves the S-7 wreck at 82.4%.

## Dimensions

Where a strip's resolution is known (a ping sidecar records `m_per_px_across`
and `m_per_px_along`), `hazard_coords.attach_dimensions` sets
`length_m = height_px * m_per_px_along` (along-track) and
`width_m = width_px * m_per_px_across` (across-track, slant-range-corrected
ground range). The box is the detector's, so it may include a shadow or miss a
faint edge; the `basis` string says so. Verification adds `height_m` from
flat-seabed shadow geometry where a shadow and an altitude allow it, with
`height_basis` explaining either the method or why it was not computed. On a
strip with no resolution, `length_m` and `width_m` are null, the box is kept in
pixels (`length_px`, `width_px`), and the basis says dimensions are not inferred.

## Hotspots and ranking

The survey is divided into a `GRID`-pixel square grid per strip, and a cell
holding at least one non-suppressed detection becomes a hotspot. A fixed frame
rather than a grown cluster, so two runs produce the same hotspots in the same
order.

**Known limitation: the grid splits clusters.** `hazard_hotspots.build_hotspots`
assigns a detection to the cell `(floor(global_x / GRID), floor(global_y /
GRID))` of its own strip and never looks at neighbouring cells. So:

- Two detections either side of a cell boundary land in different hotspots,
  even a few pixels apart. Both are still reported and ranked, but each hotspot
  carries only part of the cluster's `total_severity`, and a cluster straddling
  a corner can be split four ways and rank lower than it should.
- The grid is in pixels, not metres. On a ping strip at 0.1 m/px a 512 px cell
  is 51.2 m of seabed; on an image survey with no resolution its ground size is
  unknown, and two strips at different resolutions get cells of different sizes.
- Cells never cross strips, so a cluster at the edge of two adjacent survey
  lines is always two hotspots.
- On a strip that is not north-up the cell is a pixel square rotated with the
  track; the map draws it as that rotated footprint.

Mitigations available today: raise `GRID` (`--grid`) when objects cluster more
widely than the grid resolves, and read `max_severity` and the detections list
beside `total_severity`. `demo_survey.py` places its priority-one cluster inside
a single cell on purpose so the demo shows ranking rather than this edge.

Hotspots rank by `total_severity`, descending, never by count. The dominant class
is severity-weighted for the same reason, so three pieces of debris beside a mine
still leaves the mine dominant and the action is the mine's.

**The honest edge of that rule**, tested in `unit_tests.py` so nobody discovers it
on stage: enough low-severity objects do outrank one high-severity object. Five
tyres at 0.9 confidence sum to 1.350 against one mine's 0.900, so the tyre field
ranks first. What separates them is `max_severity` and `risk_score`, both of which
put the mine ahead, and both of which are in the export and on the map beside the
total.

Derived metrics, all arithmetic on the numbers above:

| Metric | What it is |
| --- | --- |
| `risk_score` | `max_severity + RISK_DENSITY_WEIGHT * (total_severity - max_severity)`. An index, not a percentage: it has no upper bound. |
| `detection_density` | Detections per megapixel of cell area. |
| `hazard_diversity` | Distinct classes in the cell. |
| `confidence_mean`, `confidence_max` | The detector's own certainty. |
| `severity_per_detection` | Separates one bad object from many mild ones. |
| `spatial_extent` | Bounding box of the detection centres, not of the cell. |
| `priority_rank` | Position in the ranking. 1 is first. |
| `rationale` | The above in a sentence, naming the detection that set the severity. |

Identifiers are `H001`, `H002`, assigned after ranking, so the id and the rank
never disagree.

## Output structure

```
out/
├── strips/                 ingested raw logs: {strip}.png, .nav.json, .wc.npz
├── nav/                    sidecars and water columns the manifest points at
├── tiles/                  {strip}_{x}_{y}.jpg
├── manifest.csv            tile,strip,x,y,lat,lon,mean_intensity + extras
├── manifest.json           the same rows, plus a survey block
├── export.json             the contract below
├── actions.csv             the worklist, in rank order
├── report.csv              every detection, one flat row each
├── report.geojson          every located detection, as points
└── map.html                standalone, needs no network
```

`strips/` and `nav/` exist only when raw logs were ingested. `manifest.csv`
always begins with `tile, strip, x, y, lat, lon, mean_intensity`, in that order.
Added after them: `width`, `height`, `tile_width`, `tile_height`, `center_x`,
`center_y`, `source_image`, `denoised`, `content_score`, `m_per_px_across`,
`m_per_px_along`, `quality`.

### report.csv and report.geojson

Written by `hazard_export.write_reports` after `export.json`: one record per
detection, **including suppressed ones**, flagged rather than left out, because a
report that silently omits what the filter removed cannot be audited. Columns,
in order:

| Column | Value |
| --- | --- |
| `detection_id`, `object_class` | As in the export. |
| `confidence_pct` | The 0-100 figure. |
| `suppressed`, `suppression_reasons` | The flag, and `verification.reasons` joined with `; ` (empty unless suppressed). |
| `latitude`, `longitude` | Empty when unlocated. |
| `position_basis` | The coordinate mode, or `not georeferenced: no navigation for this strip`. |
| `length_m`, `width_m`, `height_m` | From `dimensions`; empty when not measured. |
| `dimension_basis` | Every `*basis` value of `dimensions`, joined with `; `. |
| `width_px`, `height_px` | The box in pixels. |
| `severity_tier`, `recommended_action` | As in the export. |
| `strip`, `global_x`, `global_y` | Relative survey position. |
| `detector_confidence` | The detector's raw 0-1 confidence. |
| `confidence_basis` | `confidence_pct_basis`, or a statement that the figure is verified. |
| `policy_basis` | The heuristic caveat, on every row. |

`report.geojson` is a FeatureCollection of Point features with the same
properties (minus latitude and longitude, which are the geometry). Only
detections with a real position are features: a point at (0, 0) would be a
fabricated location. The collection's `properties` carry `survey_id`, `title`,
`processed_at`, `demo`, `coordinate_mode`, `detections_total`,
`detections_located`, `detections_unlocated` and the disclaimer.

### export.json

```jsonc
{
  "metadata":       { "engine", "processing_version", "processed_at",
                      "survey_id", "title", "coordinate_mode",
                      "confidence_threshold", "model_name", "model_path",
                      "detector_classes", "demo" },
  "survey_summary": { "total_raw_detections", "total_deduplicated_detections",
                      "duplicates_removed", "suppressed_detections",
                      "total_hotspots", "total_severity",
                      "highest_priority_hotspot", "highest_severity_class",
                      "class_distribution", "detections_by_tier",
                      "hotspots_by_tier", "georeferenced", "coordinate_mode",
                      "strips_processed", "strips", "tiles_processed",
                      "model_confidence_threshold" },
  "detections": [ { "id", "object_class", "confidence", "class_weight",
                    "severity", "severity_tier", "severity_basis",
                    "recommended_action", "global_x", "global_y",
                    "bbox_global", "width_px", "height_px",
                    "latitude", "longitude",
                    "confidence_pct", "confidence_pct_basis?", "suppressed",
                    "dimensions": { "length_m", "width_m", "height_m",
                                    "length_px", "width_px", "m_per_px_across",
                                    "m_per_px_along", "basis", "height_basis" },
                    "verification?": { "status", "detector_confidence",
                                       "calibrated_probability", "evidence",
                                       "terms", "confidence_pct", "hard_reasons",
                                       "reasons", "notes", "suppressed",
                                       "suppress_rule", "height", ... },
                    "class_withheld?", "downgraded_from?",
                    "provenance": { "strip", "representative_tile",
                                    "source_tiles", "merged_count",
                                    "merged_from", "tile_offset",
                                    "bbox_tile", "second_opinion?" } } ],
  "hotspots":   [ { "hotspot_id", "priority_rank", "strip", "cell",
                    "centroid", "detection_count", "total_severity",
                    "max_severity", "severity_tier", "dominant_class",
                    "recommended_action", "risk_score", "detection_density",
                    "hazard_diversity", "confidence_mean", "confidence_max",
                    "severity_per_detection", "spatial_extent",
                    "detection_ids", "top_detection", "rationale" } ],
  "configuration": { "tiling", "detection", "deduplication",
                     "hotspots", "severity_policy", "disclaimer" },
  "provenance":    { "coordinate_mode", "navigation", "source_strips",
                     "tile_count", "tiles_processed", "tiles_failed",
                     "model", "severity_formula", "ranking_rule",
                     "class_confidence_floors", "audit_note",
                     "verification", "suppression_rule",
                     "strip_resolutions_m_per_px" }
}
```

Exports written before verification have no `confidence_pct`, `suppressed`,
`dimensions`, `verification` or `suppressed_detections`. The map and the
dashboard page both render such exports as before, without a filtered count.

The base contract `{"detections": [], "hotspots": []}` is a subset, so a consumer
written against the minimal shape keeps working. Every field is load-bearing: a
value that could not be determined is `null` with something in `provenance`
saying why, rather than a plausible default that reads like measurement.

## The map

`map.html` depends on nothing once it exists. Leaflet is inlined and the sonar
imagery is embedded as a data URI, so it opens from a USB stick on a machine that
has never seen this project. Folium links eleven files from four CDNs by default,
which renders as a blank rectangle offline; `hazard_assets.py` inlines the three
that are used and removes the eight that are not.

A survey with no navigation is drawn on a pixel plane with the sonar strip as the
base layer and no world map underneath, because there is no world position to put
one at. A navigated survey is drawn on real coordinates with an optional street
basemap, off by default.

Layers, each toggleable: sonar imagery, survey tiles, severity heatmap, all
detections, critical, medium, low, filtered false positives, and hotspots. The
heatmap is weighted by severity and never by count, so a hundred tyres cannot
glow hotter than one mine.

**Detection popups** show `confidence_pct` as `NN.N%` with its basis, the
detector's own 0-1 score, the size as `L × W (× H) m` with the basis notes in
small text (or the box in pixels, labelled as not measured in metres), the
severity arithmetic, and what verification found.

**Filtered detections.** A suppressed detection is drawn as a grey, dashed,
hollow marker with a permanent "filtered" label, on the layer
"Filtered false positives (N)", which is off when the map opens. Its popup lists
the hard reasons and the plain-English reasons. Suppressed detections add no
heat and are not in "All detections" or the tier layers. The header badge and
the survey panel state the filtered count. The layer exists whenever the export
was verified, so "(0)" is shown as a finding; an export from before verification
has no such layer.

**Strips that do not run north-up.** Leaflet can only stretch an image between
two latitudes and two longitudes. A strip whose navigation is ping mode (or a
rotated affine or corner transform) is therefore resampled for display:
`_resample_north_up` locates strip pixel centres through the strip's own
`Georeference.locate` on a lattice (every 32 px across, every 4 rows along) and
interpolates between lattice points, box-averages strip pixels into a north-up
grid spaced in Web-Mercator northing, fills pinholes whose neighbours are
covered, and embeds the result as a transparent PNG data URI no larger than
2048 px on its long side. It uses numpy only; a 1000 x 3000 strip takes well
under a second. `render_map` uses `references_for_survey`, so a raw-log survey is
drawn with the exact per-ping transform the engine used, not the tile-centre
refit. Tile footprints and hotspot cells on such a strip are drawn as their true
rotated polygons. The map footer states that the imagery is resampled for
display only: every detection, hotspot and exported position comes from the
original pixels and navigation. Along-track navigation, which does not resolve
across-track position, still omits imagery.

Folium 0.20 builds popup content with a jQuery `$()` call, which the offline
page does not carry, so `render_map` swaps that call for a four-line DOM helper;
the panel script waits for `DOMContentLoaded` so it runs after the map exists;
and a relative map sets Leaflet's own `minZoom`/`maxZoom`, since folium only
passes `min_zoom` to a tile layer and a map with `tiles=None` has none.

## Dashboard integration

Route `/map` in the DeepEcho dashboard. Everything lives in
`frontend/src/survey/`: the page, five components, a config file holding every
URL and label, an API module that is the only thing that fetches, a handoff
module, `detections.js` for reading the verification fields, and a stylesheet
scoped to `.sv-`. Delete that directory and three lines
and the feature is gone.

| Endpoint | Returns |
| --- | --- |
| `GET /survey` | Every processed survey, enough for a picker. |
| `GET /survey/{id}/export` | The export document, unmodified. |
| `GET /survey/{id}/map` | The standalone `map.html`. |
| `GET /survey/{id}/actions.csv` | The worklist. |
| `GET /survey/{id}/report.csv`, `/report.geojson` | The per-detection reports. |
| `GET /survey/{id}/export.json`, `/map.html` | The same files as downloads. |
| `GET /survey/{id}/strips/{name}.png` | A browser-sized strip preview written by a survey job. |
| `POST /survey/process` | Runs the engine. Behind a flag, off by default. |
| `POST /survey/jobs`, `GET /survey/jobs/{id}`, `GET /survey/jobs/{id}/events` | Upload and watch a survey run. See Live survey jobs. |

The page reads `export.json` once and renders it. No number is recomputed in the
browser and no threshold is duplicated: the engine decided the severities, the
ranking and the actions and recorded why, and the page displays that decision.
Filter options are derived from the loaded export, so a survey full of classes
nobody has seen still filters correctly.

Verification on the page: a "Filtered False Positives" stat reads
`survey_summary.suppressed_detections` (a dash, "export predates verification",
on an older export). The selected hotspot's detections table shows
`confidence_pct` with the detector score beneath it, the size from `dimensions`,
and a "withheld" or "filtered" chip where it applies; each row expands to the
verification reasons and the measurement basis. Suppressed detections belong to
no hotspot, so they are listed in their own "Filtered false positives" section
below, closed until asked for, each with its reasons. The parsing lives in
`frontend/src/survey/detections.js`, the table in
`components/DetectionTable.jsx`.

## Live survey jobs

The Live survey page (`/mission`, `frontend/src/pages/SurveyMission.jsx` through
`services/missionApi.js`) uploads files and watches the real run.

1. `POST /survey/jobs` (`<feature>/routes/jobs.py`) takes a multipart upload:
   `files` (`.xtf .jsf .png .jpg .jpeg .tif .tiff`), optional `nav` CSV or
   `corners` JSON (image strips only), `title`, `survey_id`, `conf`. It checks
   size (`DEEPECHO_MAX_SURVEY_UPLOAD_BYTES`), file count
   (`DEEPECHO_MAX_SURVEY_FILES`) and concurrency (`DEEPECHO_MAX_SURVEY_JOBS`),
   writes the files and `data/uploads/<id>/job.json`, starts
   `python -m survey_hazard_map.survey_job --job <id>` with no shell, and returns 202. The
   route exists when `config.ENABLE_UPLOAD` holds (a detector can really run)
   or `DEEPECHO_ENABLE_SURVEY_JOBS` forces it either way.
2. The worker (`backend/survey_job.py`) owns the job. It is a subprocess because
   torch and faiss cannot share a process on macOS. It emits `stage` events
   `ingest` (strip events with size, nadir, resolution, track and a preview
   PNG), `tile` (`prepare_survey`), then runs `build_hazard_map` with
   `on_event`, which streams `progress` per tile, a provisional `detection` per
   raw box, the `dedup`, `geo`, `verify`, `hotspots` and `export` stages, and
   `final_detections`; then `report` (`render_map`), then `ghosttrace`, and
   finally `done` with the summary (including `suppressed_detections`) and the
   downloadable files that exist, or `error`. A job refuses to run without a
   real checkpoint.
3. Every event is appended to `data/surveys/<id>/events.jsonl` with `seq` and
   `ts`; status is in `data/surveys/<id>/job.json`. `GET /survey/jobs/{id}`
   returns the state and last event; `GET /survey/jobs/{id}/events` streams
   Server-Sent Events, replaying from `?after=N` or `Last-Event-ID`, with a
   heartbeat, and ends after `done` or `error`.
4. The finished job is an ordinary survey directory, so it appears in
   `GET /survey` and opens on `/map` like any other.

### POST /survey/process

`POST /survey/process` runs `run_survey.py` in a subprocess rather than in the
API process. faiss and torch each ship their own libomp and on macOS whichever
initialises second aborts the server mid-request; the application has already
loaded faiss for the assistant. Client-supplied strip paths are resolved and
required to sit inside an allowed root, so `..`, an absolute path and a symlink
all fail the same way.

## RAG handoff

The map hands a hotspot to the assistant as a structured object and stops there.
No retrieval, no prompt and no knowledge live on this side of the line.

```js
{
  hotspot_id: "H001",
  dominant_class: "ship",
  severity: 0.4937,              // the hotspot's max_severity, 0 to 1
  confidence: 0.8228,
  centroid: { global_x: 168.6, global_y: 291.4 },
  lat: null,                     // null unless the survey has navigation
  lon: null,
  recommended_action: "Flag navigation hazard",

  // additive, safe to ignore
  severity_tier: "medium", priority_rank: 1, detection_count: 1,
  total_severity: 0.4937, coordinate_mode: "Relative Survey Coordinates",
  survey_id: "s7-submarine", demo: false, evidence_tile: "..._0_0.jpg",

  // verification, copied from export.json; null or [] on an older export
  confidence_pct: 82.4,          // of the detection that sets the severity
  confidence_pct_basis: null,
  dimensions: { length_m: null, width_m: null, length_px: 520.59, ... },
  suppressed: false,
  verification_reasons: [],
  detections: [ { id, object_class, confidence, confidence_pct, dimensions,
                  suppressed, verification_reasons, hard_reasons } ],
  survey_suppressed_detections: 0
}
```

It does not travel as a detection record. The assistant maps `object_class` onto
its own label and looks severity up in its own table, so the same hotspot would
carry one urgency on the map and another beside the answer, with nobody able to
see both at once. The map owns urgency; the assistant displays what it is given.

`total_severity` is a sum over a grid cell and routinely exceeds 1. It must never
be rendered as a severity. `severity_tier` is already computed and should be
taken rather than re-derived, so the two systems cannot drift.

## Testing

```bash
python survey_hazard_map/tests/unit_tests.py         # unit tests, no files, no model, under a second
python survey_hazard_map/tests/smoke_test.py         # end-to-end checks on synthetic sonar
python survey_hazard_map/tests/tests_mapview.py      # map.html: ping resampling, verification, older exports
python3 validate_output.py out/   # validates one survey's artefacts
python3 real_model_test.py    # a real checkpoint over a few tiles, or skips
```

The four do not overlap on purpose. Unit tests exercise single functions against
values chosen by hand. The smoke test drives the whole pipeline with synthetic
strips and a stand-in detector that reports the objects they were drawn from, so
the assertions are exact rather than approximate. `validate_output.py` reads a
survey's files the way a stranger would and believes nothing it was not shown;
it is checked against eight deliberate sabotages. `real_model_test.py` covers the
one seam a mock cannot: that a real checkpoint loads and its output has the shape
the engine expects. It skips with a stated reason when torch is absent, because
that is not a failure.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `map.html` is a blank white page | Built without network access and Leaflet could not be vendored. Build one map with a network to populate `vendor/`, then it works offline forever. |
| `OMP: Error #15` | torch and faiss in one process. Use `detector=` with a subprocess, or call `run_survey.py` from a shell. |
| No tiles written | Every tile scored below `MIN_CONTENT`. Set it to 0.0 to keep them all. |
| Every lat/lon is null | No navigation was supplied. That is correct behaviour, not a bug. |
| `ModuleNotFoundError: folium` | Only needed to build a map. `pip install folium`, or pass `--no-map`. |
| A strip is missing from the survey | It could not be opened. `manifest.json` lists it under `strips_unreadable` with the reason. |
| Hotspot ranking looks wrong | Read `rationale` on the hotspot. It names the detection that set the severity. |

## Files

| File | What it is |
| --- | --- |
| `hazard_config.py` | Every knob. Nothing else hard-codes a value. |
| `survey_preparation.py` | Module 1. Strips to positioned tiles plus a manifest. |
| `hazard_map.py` | Module 2. The orchestrator and public entry point. |
| `hazard_detect.py` | The detector interface and the Ultralytics loader. |
| `hazard_coords.py` | Tile-local boxes to survey coordinates. |
| `hazard_dedup.py` | Cross-model and survey-wide deduplication. |
| `hazard_severity.py` | The formula, the tiers, the floors and the actions. |
| `hazard_hotspots.py` | Spatial aggregation, ranking and derived metrics. |
| `hazard_geo.py` | Pixels to latitude and longitude, or a refusal. |
| `hazard_export.py` | The JSON contract, the summary and `actions.csv`. |
| `hazard_mapview.py` | `export.json` to a standalone `map.html`, including north-up resampling of ping strips. |
| `sonar_ingest.py` | Raw XTF/JSF logs to strips, navigation sidecars and water columns. |
| `hazard_strips.py` | Strips rebuilt from tiles, sidecars and resolutions for later stages. |
| `hazard_verify.py` | Verification: `confidence_pct`, suppression and reasons, `height_m`. |
| `backend/survey_job.py` | The worker behind `POST /survey/jobs`. |
| `hazard_theme.py` | Every colour and label the map uses. |
| `hazard_assets.py` | Inlines Leaflet so the map works offline. |
| `run_survey.py` | Command-line runner over the whole pipeline. |
| `demo_survey.py` | The pipeline on a simulated survey. |
| `unit_tests.py`, `smoke_test.py`, `tests_mapview.py`, `validate_output.py`, `real_model_test.py` | See Testing. |
| `vendor/` | Leaflet and the heat plugin. Commit it. |
