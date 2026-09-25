# ghosttrace.json — output contract (`deepecho-ghosttrace/1`)

GhostTrace turns each detected ghost net / debris object in a processed survey
into a rescue decision. `ghosttrace/engine.py::run_ghosttrace` (or
`run_ghosttrace.py --survey DIR`) writes two files into the survey directory,
beside `export.json`:

- `ghosttrace.json` — this contract.
- `ghosttrace.geojson` — the same decisions for a map (see the end of this file).

Both are written atomically (temp file + rename), so a reader never sees a
half-written file.

A realistic, fully populated example lives at
`tests/fixtures/ghosttrace_example.json`. It is **synthetic**
(`"synthetic_example": true`): built by `tests_ghosttrace_core.py
--write-fixture` from a synthetic survey pair and fake habitat/drift/safety
stages.

## Rules every consumer can rely on

1. **Unknown is null plus a reason, never a plausible default.** A stage that
   could not run is `{"available": false, "reason": "..."}`. A position that
   does not exist is `null`.
2. **Every score keeps its terms.** `priority.score` is recomputable from
   `priority.terms`; `activity.score` from `activity.evidence` and
   `activity.formula`.
3. **Heuristics are labelled.** Weights, thresholds and neutral values are a
   configurable heuristic (`ghosttrace/config_core.py`), not fitted to real
   ghost-gear recoveries and not an official procedure. The UI must show
   `caveats`.
4. **Synthetic is labelled.** `demo` / `synthetic_inputs` true means nothing in
   the file is evidence of a real object. Alerts then carry `[SYNTHETIC]` in
   the subject and a first line saying so.
5. **No invented contacts.** An authority is named only if `kb/` names it for
   that situation; otherwise `name` is `"authority not in corpus"`.
6. **Additive only.** Keys may be added in later versions of format `/1`;
   none listed here will be removed or change meaning without a format bump.
   Keys marked *(extra)* below are additions beyond the minimum contract.

## Top level

| key | type | meaning |
|---|---|---|
| `format` | string | always `"deepecho-ghosttrace/1"` |
| `survey_id` | string | `export.metadata.survey_id`, else the directory name |
| `generated_at` | ISO-8601 UTC string | when this file was produced |
| `demo` | bool | copied from `export.metadata.demo` |
| `synthetic_inputs` | bool | `demo`, or `metadata.data_source` says SYNTHETIC, or any navigation sidecar has `synthetic: true` |
| `data_sources` | array | `{name, url, licence, snapshot, used_for}`; any may be `null` when the providing stage did not state it. Includes the geo stages' layers/currents and the kb documents actually cited by alerts |
| `caveats` | string[] | plain-English caveats for the UI. Always shown |
| `targets` | array | one per selected detection, **sorted by `priority.rank`** |
| `removed_since_previous` | array | prior targets inside this survey's coverage with no match |
| `change_summary` | object | `{compared_with: [survey_id], new, moved, persistent, removed}` |
| `recovery_plan` | object | visiting order and legs |
| `summary` | object | counters, see below |
| `run` *(extra)* | object | `core_version`, `horizon_hours`, `n_particles`, `seed`, `drift_dt_minutes`, `survey_time`, `survey_time_basis`, `stages` (`{name: {available, source, reason}}`), `failures` (`[{detection_id, stage, error, trace}]`), `priority_weights`, `priority_neutral`, `priority_tiers`, `heuristic` |

### Which detections become targets

Classes matching `GHOSTTRACE_CLASSES` on whole hyphen-delimited tokens (plural
tolerant): `net, ghost-gear, fishing-gear, rope, debris, tyre, tire, drum,
container`. `unknown` only with `GHOSTTRACE_INCLUDE_UNKNOWN=1` (an unidentified
object falls under the corpus's do-not-approach protocol). Detections with
`suppressed: true` are excluded unless `GHOSTTRACE_KEEP_SUPPRESSED=1`, and
counted in `summary.suppressed_excluded`.

## `targets[]`

| key | type | meaning |
|---|---|---|
| `detection_id` | string | `export.detections[].id` |
| `object_class` | string | as exported |
| `latitude`, `longitude` | number \| null | as exported; null when the survey is not georeferenced |
| `confidence_pct` | number \| null | 0-100. Verification's `confidence_pct` when present, else detector `confidence * 100` |
| `confidence_basis` *(extra)* | string | which of the two, stated |
| `dimensions` | object \| null | `{length_m, width_m, height_m}` from verification, else null |
| `suppressed` | bool | as exported (false for every target unless suppressed ones are kept) |
| `strip` *(extra)* | string \| null | source strip |
| `seabed_depth_m` *(extra)* | number \| null | towfish `depth_m + altitude_m` at the detection's ping row |
| `seabed_depth_basis` *(extra)* | string | how it was obtained, or why it is null |
| `activity` | object | water-column activity, below |
| `habitat` | object | geo stage `habitat_context` output, or `{available:false, reason}` |
| `drift` | object | geo stage `drift_forecast` output (+ extras), or `{available:false, reason, requested_mode, mode_basis}` |
| `people` | object | geo stage `people_safety` output, or `{available:false, reason}` |
| `change` | object | below |
| `priority` | object | below |
| `alert` | object | below |
| `errors` *(extra)* | string[] | per-target stage failures (`"drift: ValueError: ..."`); empty when clean |

### `activity`

```json
{"available": true, "score": 0.8889, "level": "high",   // "high" also needs evidence.echo_clusters_near >= evidence.high_requires_clusters (3); otherwise capped at "moderate" and evidence.level_capped says so
 "evidence": {"echo_clusters_near": 8, "echo_area_near_m2": 5.0,
              "background_clusters_per_window": 0.5, "enrichment_ratio": 8.0,
              "window_m": 25.0, "side": "port",
              "window_rows": [305, 526], "background_rows": 1320,
              "background_clusters": 3, "background_windows_equivalent": 5.97,
              "exclude_margin_m": 5.0, "epsilon": 0.5,
              "z_threshold": 3.0, "min_cluster_cells": 4, "cell_area_m2": 0.0312,
              "area_note": "..."},
 "formula": "enrichment = echo_clusters_near / (background_clusters_per_window + 0.5); score = 1/(1+exp(-1.5*(ln(max(enrichment,0.001)) - ln(2.0))))",
 "basis": "...", "limitations": "..."}
```

- `level`: `high` (score >= 0.70), `moderate` (>= 0.40), `low`, or `unknown`
  when `available` is false (then `score` is null and `reason` says why — e.g.
  a plain image survey has no water column).
- `evidence` keys after `side` are *(extra)*. `echo_area_near_m2` is echogram
  area (range x along-track), not a physical cross-section.
- `limitations` must be shown with any activity claim: echoes may be fish,
  bubbles, suspended sediment or turbulence; not validated on real ghost-net
  data; evidence of aggregation near the object, not proof of entanglement.

### `habitat`, `drift`, `people` (geo stages)

Passed through from the geo agent's modules; see their docs for the full
shape. Minimum shapes the core relies on:

- `habitat`: `{covered, inside: [...], nearest: [{layer, kind, name, distance_m, bearing_deg, source, geometry_quality}], score (0..1), terms}`
- `drift`: `{snapshots: [{t_hours, points, cone50, cone90}], impacts: [{kind, name, probability, first_arrival_hours, source}], stranding_probability, assumptions, limitations, current_source, ...}`
  plus core extras: `requested_mode` (`seabed` | `floating`), `mode_basis`,
  `start_time`, `start_time_basis`.
- `people`: `{propeller_hazard: {level, reasons, terms}, diver_brief: {...}}`
  - `diver_brief.thresholds` *(extra)*: the limits the brief actually applied,
    each with where it comes from:
    `{recreational_depth_m, advanced_depth_m, max_current_mps, max_current_knots,
    depth_check_uses, depth_label, depth_basis, current_label, current_basis,
    current_window_hours, basis}`. `within_recreational_limit` is seabed depth
    <= `advanced_depth_m` (PADI Advanced Open Water, 30 m) and
    `within_open_water_limit` is <= `recreational_depth_m` (PADI Open Water,
    18 m); `current_ok_for_divers` is the modelled near-bottom current <=
    `max_current_mps` (1 knot, US OSHA 29 CFR 1910.424(b)(3), a US workplace
    rule used as a planning analogue, not Indian law). `*_label` is `cited`
    for the documented defaults and `heuristic` when overridden by environment.

Any of the three may instead be `{"available": false, "reason": "..."}` —
module not installed, target has no position, or the stage raised on this
target. Consumers must check `available === false` before reading other keys
(a successful stage output may omit `available`).

Drift mode is `seabed` unless the class name has a floating/midwater hint,
verification says `midwater: true`, or `dimensions.height_m >= 5`.

### `change`

```json
{"status": "moved", "previous_survey_id": "s1", "previous_detection_id": "S1_P2",
 "previous_latitude": 11.60412, "previous_longitude": 92.69951,
 "moved_m": 45.0, "basis": "matched by Hungarian assignment ..."}
```

| status | meaning |
|---|---|
| `persistent` | matched to an earlier target, displacement <= 20 m (inside combined positional uncertainty; not movement) |
| `moved` | matched, displacement > 20 m and <= 75 m gate |
| `new` | no match, and the position was inside an earlier survey's coverage |
| `first_survey` | no match, and no earlier overlapping survey covered this position |
| `unmatched_no_prior` | target has no position, so it cannot be compared |

`previous_*` and `moved_m` are null unless matched (`new` names the covering
earlier survey in `previous_survey_id`). `previous_latitude` /
`previous_longitude` *(extra)* are the matched prior target's position, set for
`persistent` and `moved` only, so a map can draw the displacement.

Survey order in time comes from the earliest ping time in each survey's
navigation sidecars, falling back to `export.metadata.processed_at` only when
no ping times are recorded (the basis says which). A survey recorded later but
processed earlier is therefore still "later". Matching: Hungarian assignment on
geodesic distance, gated at `MATCH_RADIUS_M` (75 m), same class family (gear vs
debris), plan-area ratio <= 4 when both have dimensions.

### `removed_since_previous[]`

`{previous_survey_id, previous_detection_id, object_class, latitude, longitude, basis}`.
Only prior targets **inside the current survey's coverage** (convex hull of
located tile centres, located detections and sampled track — which
under-states the swath, so edge objects are left undecided). Something outside
coverage is never called removed. "Removed" means not seen again: recovered,
buried, moved beyond the gate, or missed — sonar alone cannot tell which.

### `priority`

```json
{"score": 0.6253, "rank": 1, "tier": "urgent",
 "terms": {
   "activity":       {"value": 0.8889, "weight": 0.25, "contribution": 0.222225, "measured": true, "basis": "..."},
   "habitat":        {"value": 0.7,    "weight": 0.20, "contribution": 0.14,     "measured": true, "basis": "..."},
   "drift_impact":   {...}, "people_risk": {...}, "size": {...}, "change": {...}, "recoverability": {...},
   "confidence":     {"value": 0.88, "weight": null, "contribution": null, "role": "multiplier", "basis": "..."}},
 "formula": "score = confidence.value * sum(weight * value for every other term) (= confidence.value * sum(contribution))",
 "weighted_sum": 0.7106, "tiers": {"urgent": 0.55, "high": 0.35, "routine": 0.0}, "basis": "..."}
```

- Recompute: `terms.confidence.value * Σ terms[k].contribution` (k ≠ confidence) equals `score` to ±0.001.
- `measured: false` means the input was unavailable and the stated neutral value was used.
- `confidence_missing: true` (and `terms.confidence.measured: false`) means no `confidence_pct` reached the scorer. The multiplier is then `PRIORITY_NEUTRAL_CONFIDENCE` (0.5), not 0, so the target stays in the queue; the basis text starts with `CONFIDENCE MISSING`. It is a data fault, never a low-risk result.
- `rank` is 1..n by score descending, ties by `detection_id`.
- `tier`: `urgent` >= 0.55, `high` >= 0.35, else `routine`.

### `alert`

```json
{"authorities": [{"name": "authority not in corpus",
                  "role": "fisheries authority responsible for derelict or abandoned fishing gear",
                  "situation": "fisheries",
                  "contact_basis": "no document in kb/ names this authority; ..."}],
 "subject": "[SYNTHETIC] GhostTrace urgent: suspected net at 11.60094, 92.69966 (survey s2)",
 "draft_text": "…multi-line plain text…",
 "generated_by": "template",
 "citations": [{"doc": "kb/reporting-authorities-india.md", "section": "What a report carries",
                "title": "...", "source_url": "...", "status": "verified"}],
 "basis": "deterministic template ..."}
```

- `situation` *(extra)* on each authority: `fisheries`, `protected_habitat`,
  `navigation_hazard`, `unknown_contents`, `possible_pollution`, `unidentified_object`.
- `authorities[].name` is either a name found verbatim in `kb/` or exactly
  `"authority not in corpus"`. No phone numbers or email addresses are ever
  produced.
- `generated_by: "assistant"` only when `GHOSTTRACE_USE_ASSISTANT=1` and the
  grounded assistant answered; citations then also contain
  `{"assistant_source": {...}}` entries. Any failure falls back to the
  template and says why in `basis`.
- `draft_text` always ends with the "Automated detection - verify before
  action" disclaimer.

## `recovery_plan`

```json
{"start": {"name": "Synthetic Harbour", "latitude": 11.58, "longitude": 92.69, "source": "..."},
 "order": ["S2_D1", "S2_D2", "S2_D3", "S2_Q"],
 "legs": [{"from": "Synthetic Harbour", "to": "S2_D1", "distance_km": 2.555}, ...],
 "total_km": 3.554, "method": "...", "notes": ["Legs are straight geodesic lines; ..."]}
```

- `start` is the nearest harbour (from the geo layers) to the targets'
  centroid; with no harbour layer it is the first target
  (`name: "target <id>"`, `source` says so, and `legs` begins from it). `null`
  only when no target has a position.
- `order`: every located target, tiers strictly urgent → high → routine;
  nearest-neighbour + 2-opt within a tier.
- Legs are straight lines; routes around land are not computed. No return leg.
- Targets without position are named in `notes`.

## `summary`

| key | meaning |
|---|---|
| `targets` | number of targets |
| `urgent`, `high` | tier counts |
| `actively_fishing` | targets with `activity.level == "high"` |
| `near_sensitive_habitat` | habitat available and (`inside` non-empty, or `score` >= 0.5, or a `nearest` within 2 km) |
| `propeller_hazards` | `people.propeller_hazard.level` in high/severe/moderate/medium |
| `suppressed_excluded` *(extra)* | suppressed detections not turned into targets |
| `stage_failures` *(extra)* | length of `run.failures` |

## Events (`on_event`)

Every event is `{"type": "ghosttrace", "stage": ..., "status": "done", ...}`.
Stages in order: `load`, `stages`, `select`, `target` (one per target, with
`index`, `total`, `detection_id`, `activity_level`, `errors`), `change`,
`priority`, `alerts`, `recovery`, `write` (or `complete` when `write=False`).

## `ghosttrace.geojson`

A `FeatureCollection` (`format: "deepecho-ghosttrace/1+geojson"`, `survey_id`,
`synthetic`). Coordinates are `[lon, lat]`. Features by `properties.kind`:

- `target` — Point; `detection_id, object_class, rank, tier, score, confidence_pct, activity_level, change_status, propeller_hazard, synthetic`.
- `drift_cone90` — Polygon (or MultiPolygon): the 90% cone at the final drift snapshot; `detection_id, t_hours, mode`.
- `recovery_route` — LineString from the harbour (when there is one) through `recovery_plan.order`; `total_km, start, note`.
