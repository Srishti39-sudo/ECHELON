# Integrating with the DeepEcho assistant

For another application, or another agent, that needs to call this system. It
describes the HTTP surface and the rules that surface is under. Nothing here
requires reading the Python.

Repository: `mythri2405/Echelons`, branch `dashboard`.

## What this is

Two halves that must not be confused.

**The detector** answers *what is there*. Two YOLOv8 checkpoints in `models/`,
run together over each tile.

**The assistant** answers *what does it mean and what do I do*, grounded in a
fixed corpus of nine published marine documents and nothing else. It cites
every claim and refuses when the corpus is silent. That refusal is the product,
not a limitation. Do not build anything that routes around it.

## Running it

```bash
DEEPECHO_ENABLE_UPLOAD=1 .venv/bin/python -m uvicorn backend.main:app --port 8000
```

Default base URL `http://127.0.0.1:8000`. Add your origin to
`DEEPECHO_CORS_ORIGINS` if you are calling from a browser on another port.

## Endpoints

### `POST /detect`

`multipart/form-data`, one field `tile`, an image. Returns every contact found.

```json
{"stub": false, "models": ["anomaly", "known"], "filename": "tile.png",
 "bytes": 946465,
 "detections": [
   {"object_class": "shipwreck", "confidence": 0.7574,
    "bbox": [130.0, 139.9, 555.7, 178.2],
    "detector_model": "known", "detector_class": "ship",
    "second_opinion": "the anomaly model called this same box 'other' at 0.54"}]}
```

`bbox` is `[x, y, w, h]` in pixels. `stub` true means no weights were loaded and
the record is synthetic; say so wherever you display it.

`downgraded_from`, when present, means the detector's class was below the
confidence this system requires for that class, so the contact is reported as
`unknown` with the original call preserved. Show it. Never re-promote it.

Read "downgrade" as a change of claim, not a de-escalation. Withholding a class
lowers the assessed risk only where that class outranks "unidentified". Below
that, it raises it, because an object nobody has identified is treated as more
serious than a confirmed wreck, and that is deliberate. Anyone adopting this
pattern expecting quieter output will be surprised by the aircraft case.

### `POST /chat`

```json
{"message": "what is this?",
 "history": [{"role": "user", "content": "..."},
             {"role": "assistant", "content": "..."}],
 "detection_record": { ...a detection from /detect, unchanged... }}
```

History is client-side; the backend stores nothing. Pass a detection record
straight through from `/detect`.

```json
{"answer": "...", "intent": "question|explain|anomaly|report",
 "object_class": "shipwreck", "confidence": 0.7574,
 "is_anomaly": false, "severity": "low|medium|high|unknown",
 "grounded": true, "refusal": true, "coverage_gap": false,
 "sources": [{"n": 1, "id": "...", "title": "...", "section": "...",
              "snippet": "...", "authority": "...", "status": "verified",
              "score": 0.21, "pdf_url": "/sources/....pdf"}],
 "matches": [], "query": "...", "provider": "groq", "model": "..."}
```

### `POST /chat/stream`

Same request. Server-sent events, `data: {json}` per frame, typed `meta`,
`sources`, `delta`, `done`, `error`. Sources arrive before the first word.
`grounded` exists only on `done`, because it cannot be known until the answer
is complete. The `done` frame is validated against the same model `/chat`
returns, so the two cannot drift.

### `GET /health`

Index status, corpus size, provider, and which detector models are loaded. Use
it to decide whether to show an upload control at all.

### `GET /sources/<file>`

The publications themselves, read-only. A citation's `pdf_url` points here.

## Four flags, and what a UI owes each one

Getting these wrong is the only way to make this system dangerous.

| Flag | Meaning | Required treatment |
|---|---|---|
| `grounded: false` | No citation in the answer resolves | Mark unverified, tell the user to confirm manually |
| `refusal: true` | The answer declines to fill a gap the corpus does not cover | Say the detail is unavailable, not omitted |
| `coverage_gap: true` | No document exists about this class of object | Say so before anything else |
| `is_anomaly: true` | The object is unidentified | Never render a match as an identification |

Two more rules. `severity` is looked up from a table, never inferred from the
text, so do not recompute it. Similarity scores in `matches` are rankings and
must never be shown as percentages or confidences.

Citation markers in `answer` appear as `[S1]`, and providers also emit `[ S1 ]`,
`(S1)`, `【S1】` and `[S1, S3, S6]`. Parse bracket groups and read `S<digits>`
out of them; matching the literal will silently miss most of them.

## Detector classes

`known.pt` emits aircraft, human, ship. `anomaly.pt` emits aircraft, fish,
other, shipwreck. `other` is the detector saying it saw something it cannot
name, and routes to the unidentified-object path.

Those are not the corpus's words. `DETECTOR_CLASS_MAP` in `backend/config.py`
translates them, and `CLASS_COVERAGE` records which translated classes the
corpus actually has a document for. Fish is currently uncovered.

## Constraints worth knowing before you design around them

- **Free-tier rate limits are the binding constraint.** Forty evaluation cases
  exhaust both Groq and Gemini for hours. Budget a paid key for anything that
  calls this in a loop.
- **faiss and torch cannot share a process** on macOS; both bundle libomp and
  the second to initialise aborts. Detector inference runs in a subprocess for
  this reason. Do not import ultralytics into the API process.
- **Retrieval cannot tell on-topic from off-topic by score.** An unrelated
  question scores about as high as a real one. The refusal behaviour comes
  entirely from the prompt, so anything that bypasses `/chat` loses it.

## Changing behaviour

Every threshold, class map, severity value and piece of operator-facing text is
in `backend/config.py` or `frontend/src/assistant/config/`. Change behaviour
there. `rag.py` holds retrieval, chunking and the grounding prompt, and is the
safety-critical part; it has one additive change in this whole project.

Run `python rag_assistant/eval/run.py --workers 1 --delay 3` after any change.
