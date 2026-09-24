# SIH26057 demo runbook

What to run, in what order, and what to say about each part. Every synthetic
input is labelled synthetic on screen; say so out loud as well.

## Start

```
.venv/bin/python -m uvicorn backend.app.main:app --port 8000
cd frontend && npm run dev            # http://localhost:5173
```

Regenerate the two GhostTrace demo surveys if needed (under a minute; also
rewrites `survey_hazard_map/samples/synthetic_xtf/SYNTHETIC_mannar_line0{1,2}.*`):

```
.venv/bin/python ghosttrace/tools/make_ghosttrace_demo.py
```

Line 01 is recorded 2026-09-06 04:30 UTC and the repeat 2026-09-09 04:30 UTC,
both inside the bundled HYCOM window (2026-09-05 00:00 to 2026-09-19 12:00 UTC,
`data/ghosttrace/manifest.json`), so every drift runs on the model currents for
its own dates over the full 240 h. After re-fetching currents
(`tools/fetch_ghosttrace_data.py --only currents --start YYYY-MM-DD --days N`),
move `LINE01_START` in the demo script; it refuses to build if the window no
longer covers the surveys.

Currents re-fetched 2026-09-15: the bundled HYCOM window is now 2026-09-06 00:00 to 2026-09-21 12:00 UTC (runs to 2026-09-13T12Z; steps after 14 Sep are forecast), `LINE01_START` unchanged, demo surveys regenerated (repeat: 1 moved, 1 new, 1 removed); pipeline auto-switch in `docs/E2E.md`.

## 1. The four problem-statement modules (Live survey, `/mission`)

Upload `survey_hazard_map/samples/synthetic_xtf/SYNTHETIC_mannar_line01.xtf`.

| PS module | What the audience sees |
|---|---|
| Detection | tiles processed live, boxes landing on the map as they are found |
| Confidence + noise filter | every contact has a 0-100% confidence; nadir, shadow and rock artefacts are marked "filtered" with the reason (toggle to show them) |
| Geotagging report | lat/lon from the XTF ping headers, length x width in metres, height from the acoustic shadow; download `report.csv` / `report.geojson` |
| Dashboard | stage stepper, live map, detection table, downloads, offline `map.html` |

Robustness to name: slant-range correction, gain normalisation, Lee speckle
filter, along-track resampling (varying resolution), and dropout / heave /
pitch / roll rows flagged and handled (`degraded_rows` in the sidecar).

Real-data proof point: on the public waterfall-strip sample the two nadir false
positives drop from 49% / 41% to 0.1% / 0.8% and are filtered, while the real
S-7 submarine stays at 82%.

Edge: `run_survey.py --model models/known.onnx models/anomaly.onnx` runs the
whole pipeline with no torch (numbers in `docs/EDGE.md`, measured on an M1).

## 2. GhostTrace (`/ghosttrace/demo-ghosttrace-mannar`, then `-repeat`)

Say first: the detector in these surveys is a ground-truth stand-in, because the
shipped models have no net class. Everything after detection is real.

1. **Is it killing now?** Net #1 (truth T5) shows 3.3x more water-column echo
   clusters than the line's background (moderate, 0.68); net #2 (T7, the
   control) shows 1.3x (low). Open Activity to show the evidence and the
   formula.
2. **What is it near?** Gulf of Mannar Marine National Park, about 6 km (OSM).
3. **Where will it go?** Forecast on the seabed: the real HYCOM near-bottom
   currents for 6-16 September (at most 0.09 m/s against an assumed 0.25 m/s
   mobility threshold) are too weak to
   move it. Switch to **Scenario: if refloated** and press play: it reaches a
   reef in about 18 h and the Marine National Park in about 21 h.
4. **Who is at risk?** People tab: propeller hazard and the diver brief. The
   checklist shows the limits actually applied and where each comes from
   (PADI 18 / 30 m; 1 knot from US OSHA 29 CFR 1910.424(b)(3), a US workplace
   rule used as a planning analogue).
5. **What to do?** Priority "Why?" breakdown, recovery route from the nearest
   harbour, and the alert draft naming the Wildlife Warden, Ramanathapuram and
   the fisheries authorities, each from a verified source in `rag_assistant/kb/`.
6. **What changed three days later?** Switch the survey picker to
   `demo-ghosttrace-mannar-repeat` (line 02, same track, recorded 9 September).
   Change since last survey: compared with `demo-ghosttrace-mannar`, 1 moved,
   1 new, 1 removed. On the map the dashed arrow runs from T5's old position
   to its new one (**moved 40 m**); the x marks T7, gone from a spot the repeat
   covered ("removed" means not seen again: recovered, buried, moved beyond the
   75 m gate or missed; sonar cannot tell which); T8 is **new**.
   Say honestly: the moved net's school is in the synthetic record, but in the
   shallower water it moved to the schools merge into fewer, larger echo
   clusters (13 clusters over 52 m², against a busier background), so the
   cluster-count heuristic scores it **low** there. The score was left as
   measured, not tuned.
7. **Ask the assistant.** In the target card or detail press **Ask the
   assistant about this net**. The assistant opens with this net's context
   (priority terms, activity, habitat, drift, people, change, authorities and
   the caveats, including SYNTHETIC) and the question "Why is this net ranked
   ... priority, and who should be told?". GhostTrace owns the numbers; the
   assistant explains them from the knowledge base and does not re-rank.

## Say these limits before a judge asks

- The water-column activity score is a heuristic, not validated on real
  ghost-net recordings; echoes may be fish, bubbles or sediment.
- The drift model is ~9 km HYCOM, no wind; seabed mobility threshold is an
  uncalibrated assumption. The refloat run is a scenario, not a forecast. The
  currents after 13 September are model forecast, not analysis.
- Change tracking matches by position and class family only; tens of metres of
  navigation disagreement between two real surveys can look like movement
  (movement is only called above 20 m). The repeat's changes are synthetic and
  written into the file on purpose.
- No public side-scan dataset has real labelled ghost nets.
- UNEP-WCMC reef layers inform the analysis but are not served to the map
  (their licence forbids redistribution).
- INCOIS (MoES) HOOFS currents have no public programmatic access; HYCOM is
  used instead and the finding is documented in `data/ghosttrace/SOURCES.md`.

## Before the final demo

Train the pipe / cylinder / net checkpoint (`training/README.md`), calibrate it
(`training/calibrate.py` writes `models/calibration.json`, which the confidence
score picks up automatically), drop it in `models/`, and rerun the Live survey
upload with the real model instead of the stand-in.
After `models/sonar_pipeline.py`, import its output with `.venv/bin/python import_geotag.py report/hazards.json data/surveys/<id> --xtf survey.xtf` (see `docs/INTEGRATION_GEOTAG.md`), then `validate_output.py data/surveys/<id> --strict`.
