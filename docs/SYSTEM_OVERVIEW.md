# DeepEcho: System Overview

*Side-scan sonar hazard detection, grounded decision support, and ghost-net rescue planning. Problem statement SIH26057.*

Written 16 September 2026 from the code as it stands. Every number here is taken from the repository, the training kit's results file, or a test run on this machine.

---

## 1. What DeepEcho is

A survey vessel tows a side-scan sonar and records the seabed as a long strip of acoustic imagery. Somewhere in that strip may be a shipwreck, a lost mine, a pipeline, a drowned aircraft, human remains, or a derelict fishing net. DeepEcho reads the strip, finds those objects, checks each one against the physics of how sound bounces, works out where it is on the earth and how severe it is, ranks the results, drafts the report, and answers the operator's questions from published references with citations.

It answers three questions, kept deliberately separate because each has a different kind of evidence and a different way of being wrong:

| Question | Subsystem | Evidence |
|---|---|---|
| **What is there, and where?** | Detector plus geotag | A trained neural network and survey geometry |
| **How bad is it, and what do I do first?** | Severity engine, hotspots, GhostTrace | Arithmetic over the detector's output and a configurable policy |
| **What does it mean, and who do I tell?** | Assistant | Retrieval over a curated corpus of published documents, with citations |

A confident sentence from the assistant can never raise a severity. A severity can never imply a fact. That separation is the design.

---

## 2. Architecture at a glance

```
                     ┌──────────────────────────── React dashboard (Vite, port 5173) ────────────────────────────┐
                     │ Live survey · Live Feed · Detections · Survey Hazard Map · Alerts · History · Assistant · GhostTrace │
                     └──────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                                │ HTTP + Server-Sent Events
                     ┌──────────────────────────────────────────▼─────────────────────────────────────────────────┐
                     │ FastAPI backend (uvicorn, port 8000)  backend/app/main.py                                  │
                     │  /detect  /survey/jobs  /survey  /detections  /history  /stats  /chat  /chat/stream  /ghosttrace │
                     └───┬───────────────────┬──────────────────────┬──────────────────────────┬──────────────────┘
                         │                   │                      │                          │
              ┌──────────▼────────┐  ┌───────▼─────────┐   ┌────────▼──────────┐    ┌──────────▼──────────┐
              │ Detector worker   │  │ Survey job      │   │ RAG engine        │    │ GhostTrace engine   │
              │ (subprocess)      │  │ (subprocess)    │   │ rag.py + chat.py  │    │ ghosttrace/*.py     │
              │ marine.pt, YOLO11s│  │ ingest → tile → │   │ FAISS HNSW index  │    │ activity · habitat  │
              │ backend/detect.py │  │ detect → verify │   │ over kb/ corpus   │    │ drift · safety ·    │
              │ detector_worker.py│  │ → geotag → score│   │ Gemini / Groq     │    │ change · priority · │
              └──────────┬────────┘  │ → export → map  │   │ Mission Copilot   │    │ recovery · alert    │
                         │           └───────┬─────────┘   └────────┬──────────┘    └──────────┬──────────┘
                         │                   │                      │                          │
              ┌──────────▼───────────────────▼──────────────────────▼──────────────────────────▼──────────────┐
              │ Local storage: data/deepecho.db (SQLite) · data/surveys/<id>/ (files) · data/uploads/ · index.faiss │
              └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two processes never share memory on purpose. FAISS and PyTorch each bundle their own OpenMP runtime, and on macOS the second to load aborts the process. The detector therefore runs in its own subprocess, and so does every survey job.

---

## 3. The pipeline, end to end

A survey job runs these stages in order. Each emits an event the Live survey page shows as it happens.

1. **Ingest.** A raw `.xtf` or `.jsf` log is decoded ping by ping into a waterfall image plus a navigation table: for every image row, the towfish position, heading, altitude, slant range and sample count. A plain image is loaded as pixels with no navigation. A georeferenced GeoTIFF mosaic is read with its raster transform. (`sonar_ingest.py`, `survey_hazard_map/models/marine/geotag.py`)
2. **Tile.** The strip is cut into 640 px squares, stepping 512 px, so every tile overlaps its neighbour by 128 px. An object under 128 px across is guaranteed to appear whole in at least one tile. (`survey_preparation.py`)
3. **Detect.** `marine.pt` runs on every tile at its native 640 px training size and returns boxes with class and confidence. (`hazard_detect.py`, `survey_hazard_map/models/marine/sonar_detector.py`)
4. **Verify.** Two things. Boxes seen twice across the overlap are merged into one object. Then each contact is checked against the image itself: an acoustic shadow on the far side, a bright face, sharp edges, and whether it sits in the water column or on dropout rows. The detector's score is fused with that evidence into the percentage the operator sees. A contact with low fused confidence *and* a hard reason against it is marked suppressed, never deleted. (`hazard_dedup.py`, `hazard_verify.py`, `survey_hazard_map/models/marine/shadow_check.py`)
5. **Geotag.** Each box's centre pixel becomes a latitude and longitude. From an XTF: slant range to ground range with the flat-seabed correction, offset perpendicular to heading, geodesic on WGS-84. From a GeoTIFF: the raster transform. From a bare image: no position, and the export says "Relative Survey Coordinates" rather than guessing. Sizes come out in metres where a range scale exists. (`hazard_geo.py`, `survey_hazard_map/models/marine/geotag.py`)
6. **Score.** Severity is class weight times confidence. The word for it, Critical, Medium or Low, comes from a tier table, raised to a per-class floor for navigation and life-safety classes: a wreck is never reported below Critical. (`hazard_severity.py`, `hazard_config.py`)
7. **Hotspots.** Contacts are grouped into 512 px cells and ranked by a risk score: the worst object plus a fraction of everything else in the cell. Each hotspot carries a recommended action and a written rationale. (`hazard_hotspots.py`)
8. **Export.** `export.json`, `report.csv`, `report.geojson`, `actions.csv` and a self-contained `map.html` are written. (`hazard_export.py`, `hazard_mapview.py`)
9. **GhostTrace.** For every fishing-gear or debris contact: is it still catching, what habitat is near, where would it drift, who could it hurt, has it moved since the last survey, and in what order should a boat recover them. Ends with a draft alert. (`ghosttrace/engine.py`)

A single tile uploaded on the Live Feed page runs steps 3, 4 and 6 only, and stores the result in SQLite.

---

## 4. The model

### 4.1 marine.pt

| | |
|---|---|
| Architecture | Ultralytics YOLO11s, 9.4 M parameters, 19 MB |
| Task | Object detection, seven classes (since 2026-09-25: the live checkpoint is `final.pt`, the six-class `marine.pt` fine-tuned with `ghost_net`; scorecard in `survey_hazard_map/models/final/RESULTS.md`) |
| Classes | `shipwreck`, `aircraft`, `human`, `pipeline`, `fishing_gear`, `mine_like_object`, `ghost_net` |
| Input | 640 × 640 px tiles; large strips are tiled with 128 px overlap and merged with class-wise NMS |
| Training | Google Colab, T4 GPU, early-stopped at epoch 76, best epoch 51 |
| Calibration | Identity. Raw confidence on the held-out split had an expected calibration error of 0.041; Platt scaling made it worse, so none is applied |
| Speed | 16.6 ms per tile on a T4; about 0.5 s per tile on an M1 CPU; a 145-megapixel mosaic, 558 tiles, in about 11 minutes on CPU |
| Location | `survey_hazard_map/models/marine/marine.pt`, with `calibration.json` and the pipeline scripts beside it |
| Licence note | Ultralytics YOLO11 is AGPL-3.0; a closed commercial deployment would need an enterprise licence |

### 4.2 Training data

Six public datasets were merged into one class list. No images are redistributed; only the weights and code are ours.

| Dataset | Classes taken | Licence |
|---|---|---|
| SCTD 1.0, Sonar Common Target Detection Dataset | shipwreck, aircraft, human | research use |
| Side-scan sonar imaging for mine detection, Teledyne Gavia AUV 2010–2021 | mine-like objects, empty seafloor negatives | CC-BY-SA-4.0 |
| Ghost-pot side-scan dataset, GhostVision, Delaware Inland Bays | fishing gear (derelict crab pots) | CC-BY-SA-4.0 |
| DRISHTI side-scan splits (pipe, background, wreck tiles only) | pipeline, negatives, shipwreck | CC-BY-SA-4.0 |
| SubPipe, OceanScan-MST | pipeline | CC-BY-4.0 |
| AI4Shipwrecks, University of Michigan and NOAA Thunder Bay | shipwreck | CC-BY-4.0 |
| Side Scan Sonar (Ship, Plane), Roboflow Universe | shipwreck, aircraft | CC-BY-4.0 |
| SeabedObjects-KLSG | evaluation only | research use |

The full attribution with citations is in the training kit's `DATA_ATTRIBUTION.md`.

### 4.3 Results on the held-out split

1,106 tiles and 1,379 objects, never seen in training, split per source with survey sequences kept contiguous. Confidence 0.25, IoU 0.5.

| Class | AP50 | Precision / Recall |
|---|---|---|
| pipeline | 0.99 | 0.98 / 0.99 |
| aircraft | 0.90 | 1.00 / 0.90 |
| mine-like object | 0.60 | 0.69 / 0.61 |
| shipwreck | 0.52 | 0.62 / 0.48 |
| fishing gear | 0.34 | 0.41 / 0.31 |
| human (3 test objects) | 0.33 | thin class |
| **overall mAP50** | **0.61** | mAP50-95 0.41 |
| false alarms per empty seafloor tile | 0.30 | |

### 4.4 What it replaced

Two earlier stand-in checkpoints, `known.pt` (YOLOv8s on SCTD: aircraft, human, ship) and `anomaly.pt` (YOLOv8n: aircraft, fish, other, shipwreck), are retired. On the same held-out split they scored 0.20 and 0.22 AP50 on wrecks against marine.pt's 0.52, fired on almost nothing for aircraft (precision 0.04), and had no pipeline, mine or gear classes at all. Their class names remain in the vocabulary map so older stored detections still resolve.

### 4.5 Honest limitations

- No public labelled dataset of ghost nets in side-scan sonar exists. `fishing_gear` is trained on derelict crab pots and should be read as "derelict fishing gear". Fine-tuning on real net imagery is the first thing to do with field access.
- `aircraft` (about 120 boxes) and `human` (35 boxes) are thin classes.
- Fishing gear recall is worst on large boxes, traced to label quality in the source set, not resolution.
- The raw `.xtf` path is validated on synthetic surveys only, because no public raw side-scan file exists. The real-data path is validated on a USGS GeoTIFF mosaic.

---

## 5. Where each model and service is used

| Component | What it is | Where it runs | What it does |
|---|---|---|---|
| **final.pt** | The team's detection stack: YOLO11s, YOLO26s and SAM 2.1, seven classes | Detector subprocess and survey job subprocess | Every box on every page comes from it |
| **Shadow check** | Physics heuristic, no model | Verify stage | Acoustic-shadow score, height estimate, false-positive veto |
| **Groq (default)** running `openai/gpt-oss-120b` | Hosted LLM, free tier | Assistant answers, query translation, copilot tool planning | Answers in 2–3 s. Set by `DEEPECHO_PROVIDER=groq` |
| **Gemini** `gemini-3.8-flash` | Hosted LLM, free tier | Same roles, as the failover | Free tier was taking 20–160 s on the night before the demo, so it is the fallback, not the default |
| **Embeddings** | TF-IDF projected to dense vectors, built locally | `rag.py index` | Retrieval needs no API call. Sentence-transformers or Gemini embeddings can be switched in |
| **FAISS HNSW** | Approximate nearest-neighbour index | `index.faiss` | Retrieval over 76 passages from 10 documents; also the object catalog |
| **Shapely STR-tree** | Exact 2-D spatial index | GhostTrace habitat lookup | Nearest reef, park or nesting site to a net, in milliseconds |

Two rules govern the LLM everywhere it appears. It answers only from the retrieved passages and cites `[S1]`-style tags after every claim. It never computes a figure, never names an authority the sources do not connect to that hazard, and states "not specified in the sources" instead of filling a gap. When every provider fails, the assistant returns the retrieved passages verbatim with nothing generated, and labels the answer retrieval-only.

---

## 6. Features, one by one

### Live survey (`/mission`)
Upload a raw `.xtf` log, a georeferenced GeoTIFF, or strip images with an optional navigation CSV. The backend picks the pipeline, starts a worker, and the page follows it live: stage cards, tile progress, provisional boxes appearing on the map as each tile returns, then the final deduplicated list, the summary tiles, the detections table and the seven downloadable reports.

### Live Feed (`/live`)
Drop one sonar tile. marine.pt runs, the verifier scores it, and every contact is shown with its box, class, fused percentage, raw score and severity. The result is stored so it appears in History and Detections.

### Detections (`/detections`)
Every stored single-tile contact across all time, worst first. Severity is looked up from the class under the current policy, never read out of the model's confidence. Suppressed false positives are hidden by default and one checkbox away.

### Survey Hazard Map (`/map`)
Every processed survey as an interactive map: contacts, hotspots ranked by risk, the towfish track, and the strip imagery where navigation can place every pixel. A hotspot can be handed to the assistant.

### Alerts (`/alerts`) and History (`/history`)
Alerts lists contacts that are unidentified or high severity and not suppressed. History lists every scan with its tile and lets you delete one.

### Assistant (`/assistant`)
Two modes chosen automatically. **References** is retrieval-augmented generation over the corpus with citations that open the source PDF at the passage. **Mission Copilot** is a tool-calling loop over the survey files on disk: which net first across all surveys, what changed between two surveys, which contacts were filtered and why. Tools are deterministic and read-only; their records are cited as `[D1]`-style data citations, separately from documents. A detection, hotspot or GhostTrace target can be attached so follow-ups stay about that object. Answers can be requested in several Indian languages; retrieval stays in English.

### GhostTrace (`/ghosttrace`)
For every fishing-gear or debris contact, seven assessments and a decision:

| Assessment | Method | Output |
|---|---|---|
| Activity | Water-column echo clusters beside the net versus the rest of the line (median/MAD background, robust z-score, connected components, logistic score) | Low / Moderate / High, with every count kept |
| Habitat | Nearest reef, seagrass, protected area, turtle or dugong site from bundled layers; exponential decay over 5 km | Score, nearest features with distance and bearing |
| Drift | Monte Carlo particle simulation over bundled HYCOM currents, 500 particles, 10 days | 50 % and 90 % cones, stranding probability, habitat impacts |
| People | Points for shallow water, floating gear, harbour within 5 km, drift to harbour, net over 20 m; plus a diver brief | Propeller hazard Low / Moderate / High / Unknown, recommended recovery method |
| Change | Match against the previous survey of the same area by position and size | New / moved (with metres) / persistent / removed |
| Priority | Weighted sum of activity 0.25, habitat 0.20, drift 0.15, people 0.15, size 0.10, change 0.10, recoverability 0.05, times detection confidence | Score 0–1; Urgent ≥ 0.55, High ≥ 0.35, else Routine |
| Recovery | Tiers in order, nearest neighbour within a tier, geodesic legs | A numbered route from the nearest harbour |
| Alert | Situation → authority table, each name string-checked against the cited corpus document at run time | Draft subject and body; no invented contacts, ever |

Where an input is unavailable, the term takes a stated neutral value and the record says "not assessed". Outside the regions with bundled data the answer is "not covered", never "nothing nearby".

---

## 7. How the numbers are made honest

- **Two confidences, always shown.** The raw detector score and the evidence-fused percentage sit side by side. The fusion is a logit-space sum of shadow, highlight, edge, nadir and dropout terms, each with its measurements written into the record.
- **Class confidence floors.** A class the model is not sure enough about is relabelled `unknown` and the original call is kept in `downgraded_from`. The box is never dropped.
- **Class tier floors.** Wrecks, aircraft, mines, ordnance and human remains are never reported below Critical once the class survives its floor. The export records whether the score or the floor chose the word.
- **Suppression, not deletion.** A likely false positive stays in every file with `suppressed: true` and the reasons in words.
- **No invented positions.** A tile with no navigation gets null coordinates and the survey is marked relative.
- **Every policy in the export.** Class weights, tier boundaries, floors and actions are written into `export.json`, so any score can be recomputed by hand.

---

## 8. How data is stored locally

Everything lives under the repository. No cloud service is required.

| Store | Path | What is in it |
|---|---|---|
| **SQLite database** | `data/deepecho.db` | Two tables. `scans`: one row per single-tile upload (file, size, optional position, status, models). `detections`: one row per box (class, confidence, box corners, anomaly flag, severity, and the full detector record as JSON). Written by the Live Feed route, read by Detections, History, Alerts and Stats |
| **Uploaded tiles** | `data/uploads/<scan-id>.png` | The image each scan was run on |
| **Survey folders** | `data/surveys/<survey-id>/` | Per survey: `export.json`, `report.csv`, `report.geojson`, `actions.csv`, `map.html`, `manifest.json`, `job.json` and `events.jsonl` (the live event log), `strips/` (browser previews), `tiles/`, `nav/` (navigation sidecars and water column), `ghosttrace.json`, `ghosttrace.geojson`, `worker.log` |
| **Survey uploads** | `data/uploads/<survey-id>/` | The raw log or images as uploaded, plus the team pipeline's intermediate output |
| **Knowledge base** | `rag_assistant/kb/*.md` | Ten curated documents the assistant answers from, each with authority, status and scope warning |
| **Source publications** | `rag_assistant/sources/*.pdf, *.txt` | The originals every citation links to, served read-only at `/sources` |
| **Retrieval index** | `index.faiss`, `index.json`, `vectors.npy` (+ `catalog.*`) | Built by `rag.py index`; git-ignored, rebuilt in seconds |
| **Object catalog** | `rag_assistant/catalog/objects.json` | Known object types with hazard class, descriptor and what confirms or rules each out |
| **GhostTrace layers** | `data/ghosttrace/layers/*.geojson` | Reefs, seagrass, protected areas, turtle and dugong sites, harbours, land, for the Gulf of Mannar / Palk Bay and Odisha regions; fetched by `tools/fetch_ghosttrace_data.py` |
| **Model kit** | `survey_hazard_map/models/marine/` | `marine.pt`, `calibration.json`, `sonar_detector.py`, `geotag.py`, `sonar_pipeline.py`, `shadow_check.py` |
| **Secrets** | `.env` | API keys and the provider choice; git-ignored |

Supabase is supported as an optional alternative to SQLite. If `SUPABASE_URL` and `SUPABASE_KEY` are set the same routes write there with the same row shapes; if not, nothing changes and `/health` says storage is local.

---

## 9. Backend: which file does what

### API layer (`backend/app/`)

| File | Responsibility |
|---|---|
| `main.py` | The FastAPI application: CORS, lifespan warm-up of index and detector, router wiring, `/health` |
| `store.py` | The storage interface with SQLite and Supabase implementations |
| `routes/detection.py` | `POST /detect`: single tile in, records out, persisted |
| `routes/jobs.py` | `POST /survey/jobs`: upload, validate, start the worker; status and SSE event stream; pipeline capabilities |
| `routes/survey.py` | Read processed surveys: list, export, map, actions, replay |
| `routes/history.py` | Scans, detections list, delete |
| `routes/stats.py` | Dashboard counts |
| `routes/hazard.py` | Hazard summary from stored scans |
| `routes/rag.py` | `POST /chat`, `POST /chat/stream`, `POST /rag/query` |
| `routes/ghosttrace.py` | Read and re-run GhostTrace for a survey |
| `routes/telemetry.py` | Field-device telemetry ingest for GhostTrace targets |
| `services/detection_service.py` | Row shaping and verification fields for stored detections |
| `services/rag_service.py` | Bridge from the API to the chat engine |

### Engine layer (`backend/` and repository root)

| File | Responsibility |
|---|---|
| `backend/config.py` | Every knob for the API and assistant: paths, providers, class map, confidence floors, severity vocabulary |
| `rag_assistant/chat.py` | The conversation engine: intent, routing between reference and copilot modes, provider failover, streaming, grounding checks, unsourced-number detection |
| `backend/copilot_tools.py` | The Mission Copilot's read-only tools over survey files |
| `backend/detect.py` | Runs the detector worker, merges boxes, applies floors, verifies against the tile |
| `backend/detector_worker.py` | The subprocess that holds marine.pt (torch or ONNX backend) |
| `backend/survey_job.py` | The survey worker: pipeline selection, stages, events, the team pipeline runner |
| `backend/schemas.py` | Pydantic request and response models |
| `rag.py` | The retrieval engine and provider clients: indexing, embedders, FAISS, Gemini and Groq backends, the grounding system prompt |
| `hazard_config.py` | Every knob for the survey engine: tiling, floors, weights, tiers, actions |
| `hazard_map.py` | The survey engine orchestrator |
| `sonar_ingest.py` | Raw XTF/JSF logs into strips with per-ping navigation |
| `survey_preparation.py` | Tiling and the manifest |
| `hazard_detect.py` | Detector over a tile set |
| `hazard_dedup.py` | One record per physical object across overlaps |
| `hazard_verify.py` | Confidence fusion and false-positive suppression |
| `hazard_geo.py` | Pixels to latitude and longitude |
| `hazard_severity.py` | Weights, tiers, class floors, actions |
| `hazard_hotspots.py` | Cells, ranking, rationale |
| `hazard_export.py` | JSON, CSV, GeoJSON, summary |
| `hazard_mapview.py` | The self-contained offline map |
| `import_geotag.py` | Turns the team pipeline's `hazards.json` into a full survey directory |
| `ghosttrace/*.py` | One module per GhostTrace stage, listed in section 6 |

### Frontend (`frontend/src/`)

| Area | Files |
|---|---|
| Pages | `pages/SurveyMission.jsx`, `LiveFeed.jsx`, `Detections.jsx`, `MapPage.jsx`, `SurveyReport.jsx`, `Alerts.jsx`, `History.jsx`, `Assistant.jsx`, `GhostTracePage.jsx`, `Dashboard.jsx` |
| Assistant | `assistant/components/ChatWindow.tsx`, `MessageBubble.tsx`, `CitationPanel.tsx`, `Composer.tsx`; `assistant/lib/api.ts` (SSE client) |
| Survey map | `survey/SurveyHazardMap.jsx` and `survey/components/` |
| GhostTrace | `ghosttrace/GhostTracePanel.jsx`, `components/GhostTraceMap.jsx`, `TargetCard.jsx`, `TargetDetail.jsx`, `handoff.js` |
| API clients | `services/api.js`, `survey/api.js`, `ghosttrace/api.js` |

---

## 10. Tech stack

| Layer | Technology |
|---|---|
| Detector | PyTorch, Ultralytics YOLO11, ONNX Runtime export for edge, OpenCV, NumPy |
| Sonar geometry | pyxtf (XTF reader), pyproj (WGS-84 geodesics), rasterio (GeoTIFF) |
| Survey engine | NumPy, SciPy, Pillow, folium and Leaflet for the offline map |
| GhostTrace | Shapely, pyproj, SciPy, netCDF4 (currents and bathymetry) |
| Retrieval | FAISS (HNSW), TF-IDF dense embeddings; optional sentence-transformers or Gemini embeddings |
| Generation | Groq API (`openai/gpt-oss-120b`), Google Gemini API (`gemini-3.8-flash`) as failover |
| API | FastAPI, uvicorn, Pydantic v2, Server-Sent Events for streaming |
| Storage | SQLite (default), Supabase (optional), JSON/CSV/GeoJSON files per survey |
| Frontend | React 19, Vite, React Router, React-Leaflet, react-markdown, lucide icons, TypeScript for the assistant module |
| Deployment | Docker with `serve`, `full` and `edge` profiles; Render blueprint for API plus static dashboard |
| Language | Python 3.13, JavaScript/TypeScript |

### Not used: LangChain and LangGraph

The assistant is written directly against the provider SDKs and FAISS. The retrieval loop, the provider failover, the tool-calling loop for the Mission Copilot, the citation ledger and the grounding checks are all plain Python in `rag.py`, `rag_assistant/chat.py` and `backend/copilot_tools.py`.

This was a choice, not an omission. Every rule the system depends on, that a number must be cited, that an authority must appear in the cited document, that a tool's records get numbered in the order they ran, is enforced by code the team wrote and can read in full. A framework's abstractions would sit between those rules and the evidence, and the evaluation suite tests the rules directly. The cost is that there is no drop-in agent graph; the benefit is that every step is auditable.

---

## 11. Evaluation and tests

**Assistant evaluation** (`eval/run.py`, 53 cases). Every check is mechanical, none asks a model to grade a model: citations resolve to real passages, no number appears that is absent from the sources, refusals refuse, grounded answers are grounded, intent and severity match, coverage gaps are declared. Cases cover refusal, standoff distances, unidentified objects, GhostTrace handoffs, the copilot, offline fallback and streaming. Exit code 1 on any failure so it can gate a commit.

**Engine tests** run on this machine on 15 September 2026:

| Suite | Result |
|---|---|
| `unit_tests.py` (severity, geo, tiling, hotspots) | 40 / 40 |
| `smoke_test.py` (end-to-end survey engine) | 191 / 191 checks |
| `tests_jobs.py` (survey jobs, pipeline selection, team pipeline) | 18 / 18 |
| `tests_detect_verify.py` (tile upload and verification) | 8 / 8 |
| `tests_import_geotag.py` | 23 / 23 |
| `geotag.py --selftest` (planted objects recovered in synthetic XTF) | pass, position error under 0.1 m |

---

## 12. Deployment

- **Local:** two terminals. Backend `DEEPECHO_ENABLE_UPLOAD=1 .venv/bin/python -m uvicorn backend.app.main:app --port 8000`, frontend `npm run dev`. Keys in `.env`.
- **Docker:** three profiles. `serve` is the assistant and survey reader with no torch, fits 512 MB. `full` adds tile upload and survey processing. `edge` is a headless survey runner on ONNX with no torch at all.
- **Render:** the blueprint runs the API as a container and the dashboard as static files.

---

## 13. Known limitations

- Ghost nets specifically are not in the training data; the gear class is crab pots.
- GhostTrace's habitat, current and bathymetry layers are bundled for two Indian regions. Elsewhere it reports "not covered" until the fetcher is run for that coast.
- Water-column activity needs the raw log; images cannot supply it, and the method is unvalidated on real ghost-net recoveries.
- Drift is a model forecast over a 9 km current grid, not a measurement.
- The corpus names no Indian ghost-gear reporting hotline because none exists in the sources; alerts say so.
- Recovery routes are straight legs; they do not route around land or shoals.
- The Dockerfile's edge stage still copies the retired ONNX files; it should copy the marine export instead.

---

## 14. Future scope

1. **Fine-tune on real ghost-net imagery.** A few hundred labelled tiles from a fisheries partner or a net-recovery NGO would move the weakest class most. The class map, severity and GhostTrace already accept a `ghost_net` class.
2. **Real raw logs.** Validate the XTF path on a real Klein or EdgeTech recording with GPS, and the water-column activity check against a known net.
3. **More regions.** Run the GhostTrace data fetcher for the Konkan, Gujarat and Andaman coasts so habitat, harbours and currents exist there.
4. **Streaming ingest.** The activity check keeps a running median per range bin; the same maths can score a net as pings arrive during the survey rather than after.
5. **Corpus growth.** Each new document that names an authority for a situation makes the alert table richer; the run-time string check keeps it honest.
6. **ONNX edge deployment.** The exported model already runs without torch; package it for an onboard computer.
7. **Field telemetry loop.** The telemetry route accepts surface-device reports for a target; close the loop from alert to recovery confirmation.
8. **Change tracking at scale.** Match targets across many surveys of a working ground to build a per-net history.

---

## 15. Demo checklist

| Show | Where | Why it lands |
|---|---|---|
| A tile through the detector | Live Feed, any image in the demo kit | Raw score and fused score side by side, severity from class |
| A geotagged survey | Live survey, `synthetic_A_east_5Hz.xtf` | Every contact with latitude, longitude and metres; the track on the map |
| Real imagery with the veto working | Survey report `usgs-cat-island-ds563` | Four survey-line crossings suppressed as "no shadow", one 22 m target kept |
| GhostTrace end to end | `demo-ghosttrace-mannar-repeat` | A net that moved 40 m, a propeller hazard, a recovery route, an alert that admits the corpus gap |
| Grounded answers | Assistant, "Ask the assistant" on a net | Citations that open the PDF; refusal to invent a number |

Start the backend from your own terminal and leave it running. Groq is the provider. Open each page once while online so the map tiles are cached.
