# Running DeepEcho end to end

How to run the whole system, how the Live survey picks its detection
pipeline, and exactly what to drop where when the team's trained detector
arrives. Nothing in the code has to change for the switch.

## 1. Run it

```bash
# backend (from the repository root)
.venv/bin/python -m uvicorn backend.app.main:app --port 8000

# frontend
cd frontend && npm run dev            # http://localhost:5173, Live survey at /mission
```

Upload `survey_hazard_map/samples/synthetic_xtf/SYNTHETIC_mannar_line01.xtf` on the Live survey
page. Before the upload, the page shows which pipeline the job will use and
why (from `GET /survey/jobs/capabilities`). While the job runs, the header
badge shows the pipeline that actually ran, and "Why this pipeline" shows the
reason recorded by the job.

The checks, in increasing cost:

```bash
.venv/bin/python survey_hazard_map/tests/tests_jobs.py --fast      # routes, selection, team pipeline with a fake detector
.venv/bin/python survey_hazard_map/tests/tests_jobs.py             # + a real known.pt/anomaly.pt run on the S-7 sample
.venv/bin/python survey_hazard_map/tests/tests_e2e.py              # the whole system in one process (about 1.5 min)
DEEPECHO_PIPELINE=auto .venv/bin/python survey_hazard_map/tests/tests_e2e.py   # the same, on whatever auto selects
```

`tests_e2e.py` starts the real FastAPI app in-process with TestClient, so it
opens no port. All data goes to a temporary directory. It checks:
capabilities, then the XTF upload through `POST /survey/jobs`, then the SSE
stream followed to `done` (stages in order, ending with `ghosttrace`), then
every download, then `/survey`, `/survey/<id>/export` (and
`validate_output --strict`), `/ghosttrace/<id>`, `/ghosttrace/layers/reef`,
`POST /detect` with a sample tile (skipped if the route is off), and
`POST /chat` with a GhostTrace context built from the survey's
`ghosttrace.json`. If every LLM provider is rate-limited or offline, `/chat`
returns the labelled `retrieval_only` fallback and the test checks that shape
instead, so a rate limit never fails the test. At the end it prints a summary
table and exits with 1 on any failure.

## 2. How the pipeline is chosen

One function decides: `backend/survey_job.py: select_pipeline()`. The worker
calls it for each job, and the capabilities route calls it for each request.
Both look at the files on disk at that moment, so no restart is needed.

| `DEEPECHO_PIPELINE` | runs |
|---|---|
| `auto` (default) | `teammate` only when **all** of `sonar_pipeline.py`, `geotag.py`, `sonar_detector.py`, the weights and the calibration file exist; otherwise `echelon` |
| `echelon` | always the DeepEcho engine (`models/known.pt` + `models/anomaly.pt`) |
| `teammate` | the team pipeline, or a failed job whose error names the missing files. An explicit request is never quietly swapped for the other engine. |

Where the team's files are looked for:

| variable | default |
|---|---|
| `DEEPECHO_TEAMMATE_MODULES` | `<repo>/../models` (here `/Users/mysterymaven/models`) if that folder exists, else `<repo>/models` |
| `DEEPECHO_TEAMMATE_WEIGHTS` | `<modules>/best.pt`, else `<repo>/models/best.pt` |
| `DEEPECHO_TEAMMATE_CALIB` | `calibration.json` beside the weights |
| `DEEPECHO_TEAMMATE_TIMEOUT` | `3600` seconds for one `sonar_pipeline.py` run |

`sonar_detector.py` and `geotag.py` count if they are in the modules folder
or in the weights' folder, because `sonar_pipeline.py` imports from both.
The optional stages are switched on only when their files exist. `--shadow`
needs `shadow_check.py`. `--anomaly` needs `anomaly.py` and a `bg_embed.pkl`
(or `.npz`) beside the modules. If there is a background embedding but no
`anomaly.py`, a warning is logged and `--anomaly` is not passed.

What is recorded, in three places:

- **events**: a `pipeline` event (`pipeline`, `label`, `reason`, `requested`),
  `pipeline` on every stage event, and `pipeline`, `pipeline_label`,
  `pipeline_reason` and `inputs_not_processed` on the `done` event.
- **job.json** (`GET /survey/jobs/<id>`): `pipeline`, `pipeline_label`,
  `pipeline_reason`, `pipeline_requested` and `inputs_not_processed`.
- **`GET /survey/jobs/capabilities`**: `selected`, `label`, `reason`,
  `error`, and for each pipeline `available`, `reason`, `found`, `missing`,
  `shadow`, `anomaly`, `modules`, `weights` and `calib`.

### The team path, step by step

For one upload:

1. **detect**: `sonar_pipeline.py <source> --weights W --calib C --modules M
   [--shadow] [--anomaly bg_embed.pkl] [--nav nav.csv] --out <upload>/teammate/pipeline/<stem>`
   runs in a subprocess with a fixed argv, no shell, `PYTHONDONTWRITEBYTECODE=1`
   (so nothing is written into the team's folder) and the timeout above.
   Every stdout/stderr line is streamed as a `log` event prefixed
   `sonar_pipeline:`. If the run exits non-zero, times out or writes no
   `hazards.json`, the job fails and the last output lines are in the error.
2. **geotag**: `import_geotag.import_geotag(hazards.json, staging, xtf=... |
   nav_csv=..., waterfall=..., verify="auto", overwrite=True, ghosttrace=False)`
   writes into a staging folder named after the job. The result is then moved
   into `data/surveys/<id>/`. It is not written straight there because the
   importer replaces its output directory, and that directory holds
   `job.json` and the `events.jsonl` the browser is reading.
3. **verify / export / report**: these report what the importer did. The
   verification path is `theirs` when `shadow_check.py` ran and `ours`
   otherwise. The export, report.csv/.geojson, actions.csv and map.html are
   written, and the final detections are sent to the page.
4. **ghosttrace**: runs on the finished survey with the real surveys root,
   the same way as on the DeepEcho path. That is why the importer's own
   GhostTrace run is off: its root would have been the staging folder, and
   change tracking would have had nothing to compare against.

The team path takes one source per job, because `sonar_pipeline.py` writes one
`hazards.json` and `import_geotag` builds one survey from it:

- **XTF**: the first `.xtf` runs. Every other uploaded file is listed in
  `inputs_not_processed` with a reason, logged as a warning, and shown on the
  page as "N uploaded files were not processed". Upload those as their own
  surveys.
- **Images** (`.png/.jpg`): run on the team path only with a navigation CSV
  that has geotag `NavTable` columns (`lat, lon, heading_deg, altitude_m,
  slant_range_m, samples_per_side`). A pixel-to-lat/lon fix CSV, four corners,
  or no navigation at all means the job runs on the DeepEcho engine, and the
  reason says so ("image uploads need a navigation CSV with geotag NavTable
  columns ...").
- `.jsf` and `.tif` uploads always run on the DeepEcho engine.

## 3. When best.pt, calibration.json and sonar_detector.py arrive

### State on 2026-09-15

`/Users/mysterymaven/models/` already has `sonar_pipeline.py`, `geotag.py`,
`shadow_check.py` and `anomaly.py`. It is **missing `sonar_detector.py`,
`best.pt` and `calibration.json`** at the top level, so `auto` runs the
DeepEcho engine and the job logs:

```
pipeline: DeepEcho engine (known.pt + anomaly.pt); auto: teammate pipeline unavailable: sonar_detector.py, best.pt, calibration.json not delivered yet (looked in /Users/mysterymaven/models); running the DeepEcho engine
```

The team's trained kit is in the subfolders, under a different weights name:

| folder | contents |
|---|---|
| `models/demo/` | `marine.pt`, `sonar_detector.py`, `calibration.json` (identity: A=1, B=0), `bg_embed.pkl`, `anomaly.py`, `shadow_check.py`, pipeline and geotag copies |
| `models/run_v1/` | `marine.pt` (same bytes), `best.onnx`, `sonar_detector.py`, `calibration.json` (Platt: A=1.028, B=0.946) |

The two calibration files disagree. The team has to say which one is final;
this project does not pick one for them.

### Option A: drop the files in (no configuration)

Put these beside `sonar_pipeline.py` in `/Users/mysterymaven/models/`:

```
/Users/mysterymaven/models/sonar_detector.py
/Users/mysterymaven/models/best.pt              # the trained weights (marine.pt, renamed)
/Users/mysterymaven/models/calibration.json     # the calibration chosen for best.pt
/Users/mysterymaven/models/bg_embed.pkl         # optional: enables --anomaly (anomaly.py is already there)
```

(If the sibling `models/` folder does not exist on a machine,
`<repo>/models/` is used instead: `sonar_pipeline.py`, `geotag.py`,
`sonar_detector.py`, `best.pt` and `calibration.json` all go there.)

### Option B: point at the delivered folder (no copying)

```bash
export DEEPECHO_TEAMMATE_MODULES=/Users/mysterymaven/models/demo
export DEEPECHO_TEAMMATE_WEIGHTS=/Users/mysterymaven/models/demo/marine.pt
# DEEPECHO_TEAMMATE_CALIB defaults to demo/calibration.json
.venv/bin/python -m uvicorn backend.app.main:app --port 8000
```

This was run on 2026-09-15 against `SYNTHETIC_mannar_line01.xtf`. The job
completed in about 100 s, and the survey passed
`validate_output --strict` (1040 checks).

### What you should then see

`GET /survey/jobs/capabilities` returns `"selected": "teammate"` and
`"label": "team detector (best.pt)"`. The page's badge reads
**Pipeline: team detector (best.pt)**. The job log reads, in order (from the
Option B run; with Option A the names are `best.pt` and the top-level folder):

```
stage ingest      reading 1 raw sonar log
pipeline: team detector (marine.pt); auto: team pipeline complete: sonar_pipeline.py, geotag.py, sonar_detector.py, marine.pt, calibration.json found in /Users/mysterymaven/models/demo
stage detect      team pipeline on SYNTHETIC_mannar_line01.xtf: sonar_pipeline.py with marine.pt + shadow_check.py, anomaly.py with bg_embed.pkl
running python sonar_pipeline.py SYNTHETIC_mannar_line01.xtf --weights marine.pt --calib calibration.json --modules demo --shadow --anomaly bg_embed.pkl --out SYNTHETIC_mannar_line01
sonar_pipeline: read_xtf: 1 ping drop-outs (time gaps > 2.5x median interval) flagged
sonar_pipeline:   # 1 shipwreck           47.7%  9°07'10.60" N, 79°03'03.99" E  no-shadow  h=0.18
sonar_pipeline.py finished in 89.4 s: 1 detection(s), 138 unknown anomal(ies), 0 vetoed by the shadow check, navigation xtf/csv
stage geotag      geotag.py positions from hazards.json, built into a DeepEcho survey (import_geotag: ...)
stage verify      verification path 'theirs': auto: at least one record carries shadow_score or verdict; 1 suppressed
stage export      139 detection(s), 6 hotspot(s), Geo-referenced; export.json written
stage report      report.csv, report.geojson, actions.csv and map.html written by the importer
stage ghosttrace  checking detected nets and debris: ...
```

To go back to the DeepEcho engine without removing anything, set
`DEEPECHO_PIPELINE=echelon`.

### Seen on that run, for the team

- **Anomaly channel.** `anomaly.py` with the delivered `bg_embed.pkl` flagged
  138 windows of the synthetic Mannar line as `unknown_anomaly` at 100%. The
  embedding was presumably fitted on the USGS mosaic, and the synthetic
  waterfall is outside that distribution. This needs a look before a demo on
  XTF input; without `bg_embed.pkl` beside the modules, `--anomaly` is not
  passed.
- **Dimensions.** Seven anomaly boxes had a geotag `length_m` of 53.41 m against
  an independent 28.03 m. These are recorded as import warnings, and geotag's
  value is kept.

## 4. Troubleshooting

| symptom | cause |
|---|---|
| job fails: `DEEPECHO_PIPELINE=teammate requested, but teammate pipeline unavailable: ...` | the forced team path, with the files named in the message missing |
| job fails: `sonar_pipeline.py exited with status N: ...` | the team pipeline crashed; its output is in the live log and in `worker.log` |
| job fails: `sonar_pipeline.py did not finish within 3600 s` | raise `DEEPECHO_TEAMMATE_TIMEOUT`; the process group was killed |
| `.png` upload ran on the DeepEcho engine | no NavTable navigation CSV; the reason is on the page |
| capabilities 404 | survey jobs are disabled (`DEEPECHO_ENABLE_SURVEY_JOBS=0`, or neither pipeline can run) |
