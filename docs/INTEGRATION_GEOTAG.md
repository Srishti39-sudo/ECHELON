# Integrating geotag.py / sonar_pipeline.py

The teammate's `models/` folder owns detection and geotagging: the trained
detector (`sonar_detector.py`), the shadow check (`shadow_check.py`), the anomaly
channel (`anomaly.py`) and `geotag.py`, chained by `sonar_pipeline.py`. It writes
`hazards.json`.

This repository owns the three features that consume detections: the Survey
Hazard Map, GhostTrace and the Assistant. `import_geotag.py` connects the two.
It turns `hazards.json` into a normal survey directory, so nothing else has to
change.

```
sonar_pipeline.py  ─►  report/hazards.json + <name>_waterfall.png
import_geotag.py   ─►  data/surveys/<id>/  manifest, tiles, export.json, actions.csv,
                                           report.csv/.geojson, map.html,
                                           nav/<strip>.nav.json, nav/<strip>.wc.npz,
                                           ghosttrace.json/.geojson
```

## Running it

```bash
# 1. teammate's pipeline (from models/)
python sonar_pipeline.py survey.xtf --weights best.pt --calib calibration.json \
       --shadow --anomaly bg_embed.pkl --out report/

# 2. import (from this repository). --xtf is the preferred path.
.venv/bin/python import_geotag.py report/hazards.json data/surveys/<id> \
       --xtf survey.xtf [--sensor-depth 20] [--title "Line 01"]

.venv/bin/python validate_output.py data/surveys/<id> --strict
```

The import options are `--waterfall`, `--xtf`, `--nav-csv` with `--nadir-col`,
`--survey-id`, `--title`, `--verify auto|theirs|ours|none`, `--sensor-depth`,
`--altitude` (passed to `read_xtf`), `--modules` (the folder that holds
`geotag.py`; the default is `../models`), `--weights` (a name for the model
reference), `--overwrite`, `--no-ghosttrace` and `--no-map`. The library call is
`import_geotag(hazards_json, out_dir, **same options)`, and it returns the
export dict.

The import refuses a directory that already has files in it unless you pass
`--overwrite`. Even with `--overwrite`, it only deletes a directory that looks
like a survey, and never one that holds the input files.

Choosing the navigation:

1. `--xtf`: the navigation is re-read exactly with `geotag.read_xtf`. This is
   the preferred path.
2. `--nav-csv`: uses the `NavTable.to_csv` columns. `roll_deg`, `pitch_deg`,
   `heave_m` and `gap_before` are read if they are present.
3. Neither: the only positions are each hazard's own `lat`/`lon`. There is no
   sidecar and no water column, and GhostTrace activity reports it is
   unavailable and says why.

The waterfall defaults to `<hazards dir>/<image>_waterfall.png`, then
`<image>`. The annotated PNG is never used.

## What is read from hazards.json, and why

| hazards.json field | becomes | why |
|---|---|---|
| `id` | `id = <strip>_g<id>` | geotag ids are run-local ints. Prefixing the strip keeps them unique and stable for a given file. |
| `classification` | `object_class`. `unknown_anomaly` becomes `unknown`. | severity, action and GhostTrace target selection |
| `confidence_pct` | `confidence_pct`, and `confidence = pct/100` | It is already calibrated, and already fused with the shadow score when that ran. It is copied as is, never recomputed. |
| `anomaly_score` | `confidence_pct = 100 × score` (the basis says so), and `verification.anomaly_score` | open-set channel |
| `box_xyxy_px` | `bbox_global`, `global_x/y`, `width_px/height_px`, tile, `bbox_tile` | survey-frame coordinates. The representative tile is the manifest tile that contains the box centre. |
| `lat`, `lon` | `latitude`, `longitude`, **unchanged** | geotag is the authority for positions |
| `length_m`, `width_m` | `dimensions.length_m/width_m` | geotag is the authority for sizes. The basis is `geotag.py (edge ground-range difference; windowed along-track spacing)`. |
| `height_m` | `dimensions.height_m` | `shadow_check.py` geometry |
| `shadow_score`, `verdict`, `vetoed` | `verification` block and `suppressed` | the "theirs" verification path |
| `side`, `ground_range_m`, `slant_range_m`, `ping`, `ping_time` (converted to Z), `vehicle_altitude_m`, `heading_deg`, `lat_dms`, `lon_dms`, `man_made`, `navigation` | `detections[].provenance.geotag` | audit trail |
| `meta.simulated_navigation`, or `SIMULATED` in any `navigation` | `metadata.demo = true`, `data_source`, `demo_warning`, a warning, and `synthetic` in the sidecar | A position from a simulated track must never read as a real one. |
| `meta.notes`, `meta.limitations` | `provenance.geotag_notes`, `provenance.geotag_limitations`. Any `WARNING` note is also copied to `metadata.warnings`. | For example, the altitude-0 warning. |

Absolute paths, such as `meta.source` or `nav.source`, are cut down to file
names, because export.json travels.

After the fields are mapped, the records go through the same stages as
`build_hazard_map`, in the same order: class confidence floor, then scoring,
then deduplication, then verification, then hotspots from the unsuppressed
detections. After that come export.json, actions.csv, report.csv/.geojson,
`render_map` and `run_ghosttrace(out_dir, surveys_root=out_dir.parent)`.

Detections have their own positions, so hotspots are built with no
georeference. `_centroid_geo` averages the positions of the member detections.

### The navigation sidecar and water column

`geotag.read_xtf` renders **slant range**, but the `deepecho-strip-nav/1`
contract and hazard_geo's `ping` mode assume ground range. The sidecar the
import writes therefore:

- has `slant_range_corrected: false`, `slant_m_per_sample`,
  `m_per_px_across: null`, and `m_per_px_along` set to the median along-track
  spacing between rows, with repeated fixes interpolated first;
- has a row for each image row: `time` (ISO with Z), `lat`, `lon`,
  `heading_deg`, `altitude_m` (null if the whole line recorded 0), `depth_m`
  (the `--sensor-depth` value, otherwise null), `seabed_depth_m`, `roll_deg`,
  `pitch_deg`, `heave_m`, and `quality`. `quality` is `dropout` where geotag
  flagged `gap_before`, and those rows are summarised in `degraded_rows`.
- is recorded under `manifest.survey.navigation` with mode
  `geotag_slant_range`. `hazard_geo.references_for_survey` only uses mode
  `ping`, so nothing re-locates a pixel from the sidecar. Tile centres are
  positioned with `NavTable.pixel_to_latlon`, which is geotag's own geometry.

The water column (`<strip>.wc.npz`: `port`, `starboard`, `bottom_range_m`,
`m_per_bin`) is cut from the waterfall. For each row, the samples whose slant
range is less than the altitude are the water column. The 8-bit log-scaled grey
levels are used as they are, not converted back to linear, and the sidecar's
`water_column.note` says so.

## Verification: one path, never both

| `--verify` | what happens |
|---|---|
| `theirs` | Uses `shadow_score`/`verdict`/`vetoed` from hazards.json. `suppressed = vetoed`. If no `vetoed` flag is present, a `no-shadow` verdict counts as the veto. `confidence_pct` is not changed. The source is recorded as `shadow_check.py (teammate)`. |
| `ours` | Runs `hazard_verify.verify_survey` over the waterfall. The nadir comes from the NavTable. **No `m_per_px_across`** is passed, so no metric height is computed from slant geometry, and geotag/shadow heights are kept. No second calibration is applied, because sonar_detector is calibrated upstream. |
| `none` | Uses `confidence_pct` as supplied, marked "not verified". |
| `auto` (default) | `theirs` if any record has `shadow_score` or `verdict`. Otherwise `ours` if a waterfall exists. Otherwise `none`. |

The chosen path and the reason for it are recorded in
`provenance.verification.path` and `.choice`.

## Class vocabulary

These are the trained classes: shipwreck, aircraft, human, pipeline,
fishing_gear, mine_like_object, plus unknown_anomaly.

- `mine_like_object` has weight 1.0 and the action "Keep clear; send for expert
  identification as possible ordnance". It has an exact key in
  `hazard_config.SEVERITY`/`ACTIONS`, because the substring rule would otherwise
  match "mine" and give "Deploy EOD team". For the Assistant it is mapped to
  "suspected mine-like object". That label keeps high severity and routes to
  `rag_assistant/kb/naval-mine-identification.md` and `rag_assistant/kb/unidentified-object-protocol.md`,
  but the assistant treats it as a classifier output, never as a confirmed mine.
- `fishing_gear` gets the net family's action, "Schedule ghost-gear recovery".
  It is a GhostTrace target (`fishing-gear`) and maps to "derelict fishing gear"
  for the Assistant.
- `pipeline` uses the existing pipeline entry.
- `unknown_anomaly` becomes `unknown` (weight 0.8, expert identification, and
  the unidentified-object path in the Assistant).

## Status of geotag.py (verified 2026-09-14)

**Fixed upstream on 2026-09-14.** These two bugs were confirmed earlier today
and are now fixed in `models/geotag.py`:

1. *Width:* it used to multiply by ground/slant instead of slant/ground, so a 2 m
   object at 2–4 m range reported 0.26 m. It now takes the difference of the
   ground ranges at the box edges, and adds the two ranges when the box
   straddles nadir.
2. *Length:* it used the ping-to-ping GPS distance at one row, which gave 0.0
   with repeated fixes. It now averages the along-track spacing over a window,
   and `read_xtf` interpolates repeated fixes by time.

Verified selftest numbers: width 2.40 m at 3 m range (true value 2.0; the error
is pixel quantisation near nadir), 2.05 m at 11 m, 2.02 m at 31 m. Lengths are
within 0.2 m and positions within 0.07 m.

The import trusts these values. When a navigation table is available, it also
recomputes width and length independently. Any disagreement over 25% (and over
0.1 m) goes to `provenance.import.dimension_cross_check` and
`metadata.warnings` as a warning, and geotag's value is kept. In the tests there
are none.

## Requested additions (for the teammate)

1. **Keep vetoed detections, with a reason.** Today `sonar_pipeline.py` drops
   shadow-vetoed boxes, and `geotag.geotag` does not carry `vetoed`. Please keep
   them and pass `vetoed` (and ideally `veto_reason`) through geotag's record.
   The import already honours the flag, and the map shows vetoed boxes on its
   "Filtered false positives" layer with the reason instead of losing them.
2. **Write the navigation CSV by default.** Call `NavTable.to_csv(out /
   'nav.csv')` in `write_report`, including `roll_deg`, `pitch_deg`, `heave_m`
   and `gap_before` in `FIELDS`. Image and GeoTIFF runs then have a navigation
   input to import.
3. **Sensor depth.** `read_xtf` has `SensorDepth`. Please add `depth_m` to the
   rows. Until then, pass `--sensor-depth`. GhostTrace needs it for seabed depth.
4. **Stable ids.** Ids are run-local ints (1..n). Please add an id derived from
   the source and the box, e.g. `<stem>_<ping>_<x1>`, so a re-run gives the same
   ids.
5. **Timezone.** `ping_time` from `read_xtf` is UTC-aware (`+00:00`). Please
   keep that for CSV and simulated navigation too, preferably as `Z`. The import
   treats naive times as UTC.
6. **Water column.** Please export the real slant-range water-column samples
   (the samples nearer than altitude, per side, before gain) as `<stem>.wc.npz`.
   The import currently derives them from the 8-bit log-scaled waterfall, which
   is only an approximation.
