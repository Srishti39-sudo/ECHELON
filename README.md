# DeepEcho RAG assistant

The detection model answers *what's there*. This answers *what does it mean, and
what do I do*, grounded only in the curated corpus under `rag_assistant/kb/`.

Two ways in. `rag.py` is the command-line engine and answers one question at a
time. The chat application in `backend/` and `frontend/` wraps the same engine
in a conversation, with follow-ups that keep context and citations an operator
can open. See [The chat application](#the-chat-application).

## Layout

One folder per feature. Each is a Python package with its own engine, routes and tests, so a teammate can work inside one without touching the others.

```
survey_hazard_map/   raw log → tiles → detector → verification → geotag → hazard map
                     engine modules, routes/, models/, samples/, training/, tools/, tests/
ghosttrace/          which net to recover first: activity, habitat, drift, people, change
                     engine package, routes/ (api + telemetry), tools/, tests/
rag_assistant/       Beacon: grounded answers with citations
                     rag.py, chat.py, routes/, kb/, sources/, catalog/, eval/, tests/
backend/             what all three share: config.py, schemas.py, procs.py, app/main.py
frontend/            the dashboard (src/survey, src/ghosttrace, src/assistant)
data/                surveys and layers, shared by all three at run time
```

`backend/app/main.py` only wires the three routers together; the feature switch `DEEPECHO_FEATURES` (see docs/DOCKER.md) decides which of them one process serves. Run anything as a module from the repo root: `python -m survey_hazard_map.run_survey`, `python -m rag_assistant.rag index`, `python -m ghosttrace.run_ghosttrace`. Tests: `python <feature>/tests/<file>.py`.

Ownership is in `.github/CODEOWNERS`; work on a feature branch named after the folder.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env               # then paste your keys into .env
.venv/bin/python -m rag_assistant.rag index
```

Keys live in `.env`, which is gitignored. Only the provider you actually use
needs one.

```
GEMINI_API_KEY=...                 # free key: https://aistudio.google.com/apikey
GROQ_API_KEY=...                   # free key: https://console.groq.com/keys
DEEPECHO_PROVIDER=gemini           # gemini | groq
```

`.env` is read at startup by a few lines of stdlib, so there is no extra
dependency. A variable already exported in your shell wins over the file, which
makes a one-off override easy:

```bash
DEEPECHO_PROVIDER=groq python -m rag_assistant.rag ask "..."
```

The whole pipeline runs on free tiers. Retrieval is local by default and costs
nothing at all; generation is Gemini Flash.

`--no-llm` on any command shows the retrieved sources without calling any model,
so the pipeline is demonstrable before a key exists. With no dependencies
installed at all, `index` falls back to exact sparse retrieval and still runs.

## Providers

The grounding rules live in one system prompt shared by every backend, so
switching provider changes the cost and the latency, never what the assistant is
allowed to say.

| `--provider` | Default model | Key |
|---|---|---|
| `gemini` (default) | `gemini-3.8-flash` | `GEMINI_API_KEY` |
| `groq` | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| `nvidia` | `nvidia/nemotron-3-super-120b-a12b` (`DEEPECHO_NVIDIA_MODEL`) | `NVIDIA_API_KEY` |

```bash
python -m rag_assistant.rag ask "..." --provider groq
python -m rag_assistant.rag ask "..." --model gemini-3.5-flash
export DEEPECHO_PROVIDER=groq      # or set the default for the session
```

Temperature is 0 everywhere. This is grounded extraction from retrieved text,
not composition, and sampling only invites drift away from the sources.

Gemini's free tier returns 503 under load, often enough to hit one mid-demo.
The client is configured with the SDK's own retry: five attempts, exponential
backoff with jitter, on 408, 429 and 5xx. If it still fails, `gemini-2.5-flash`
answers in about a second and is the fastest fallback.

Groq is the one to reach for if a Gemini safety filter trips on ordnance
content, and it is also the honest route to the sovereign and on-premises pitch,
since the same open weights can run on your own hardware later with no change
above this seam.

`nvidia` talks to NVIDIA's hosted NIM endpoints over the OpenAI protocol. The
same NIM container runs on an NVIDIA GPU on-premises, so pointing
`NVIDIA_BASE_URL` at it is the whole change for an at-sea assistant with no
internet. The catalogue at build.nvidia.com moves; set `DEEPECHO_NVIDIA_MODEL`
to whatever it currently lists. List what your account can actually reach with
`client.models.list()`; the model literals baked into the SDK are not a
guarantee of access.

## Embeddings

Five embedders, and the index layer does not care which you use.

| `--embedder` | Cost | Notes |
|---|---|---|
| `tfidf-dense` (default) | free, local, offline | no fidelity loss, dimension is the vocabulary size |
| `sentence-transformers` | free, local, offline | 384-d semantic vectors, used automatically when installed |
| `gemini` | free tier, network | `gemini-embedding-001` at 768-d, a call per query |
| `nvidia` | free credits, network; on-prem NIM | `nemotron-3-embed-1b`, 2048-d, multilingual retrieval (`DEEPECHO_NVIDIA_EMBED_MODEL`) |
| `random-projection` | free, local, offline | fixed dimensions, lossy, measured below |

`--embedder gemini` uses the retrieval task types the model expects, embedding
documents and queries differently, and renormalises because truncated Gemini
embeddings are not unit length. It is opt-in rather than automatic: making every
index build and every query hit a rate-limited network service is not a good
default for a demo you need to run on stage.

`--embedder nvidia` is the one to reach for when questions arrive in Hindi or
Tamil: the model places a Tamil question next to the English passage that
answers it, so retrieval no longer depends on the translation step. It needs
`NVIDIA_API_KEY`; the same model runs on-premises as a NIM container with only
`NVIDIA_BASE_URL` changed.

### Reranking

Retrieval answers "which chunks are near this query". A reranker reads query and
passage together and answers "does this passage answer it", for a handful of
candidates. `rag_assistant/rerank.py` fetches 20, reranks with
`llama-nemotron-rerank-vl-1b-v2` (`DEEPECHO_RERANK_MODEL`, `NVIDIA_RERANK_URL`), keeps `TOP_K`. It is on automatically when
`NVIDIA_API_KEY` is set (`DEEPECHO_RERANK=on|off|auto`), and advisory: any
failure returns retrieval's order untouched, so an answer never depends on a
second network call.

Measure before trusting either switch:

```bash
python rag_assistant/eval/retrieval_bench.py             # the index as built
python rag_assistant/eval/retrieval_bench.py --rerank    # plus the reranker
```

24 operator-phrased questions, two in Hindi and Tamil, each with the document
that answers it. `hit@1`, `hit@k` and MRR, no model grading anything.

## The four functions

```bash
python -m rag_assistant.rag explain --detection detection.json
python -m rag_assistant.rag ask "what is the disposal procedure for unexploded ordnance?"
python -m rag_assistant.rag anomaly --describe "cylindrical, 2 m, partially buried, hard shadow"
python -m rag_assistant.rag report --detection detection.json
```

`search` and `match` expose the two retrieval halves on their own, and `bench`
measures the index:

```bash
python -m rag_assistant.rag search "who do I report a mine to" -k 5
python -m rag_assistant.rag match --describe "large structure, debris scatter" --top 3
python -m rag_assistant.rag bench --ef 8 16 32 64 128
```

## Retrieval

`rag_assistant/kb/*.md` are split on `##` headings and packed into ~900 character chunks with
overlap. Each chunk becomes a TF-IDF vector over unigrams and bigrams, L2
normalised, and is stored in a **FAISS HNSW index** under inner product, which on
unit vectors is cosine. HNSW is a navigable small-world graph: a query descends
coarse layers and refines, so cost grows with log(n) instead of n. The same code
path serves this 37-chunk corpus and a corpus of millions.

Build parameters are `M=32`, `efConstruction=200`; `efSearch` defaults to 64 and
is the query-time recall dial (`--ef-search`).

### Layers, and what is swappable

| Layer | Default | Alternatives |
|---|---|---|
| Embedder | `tfidf-dense` | `sentence-transformers`, `random-projection` |
| Index | `faiss-hnsw` | `exact` (brute force), sparse stdlib fallback |
| Metric | cosine | pearson, set at index time |

Nothing above couples to anything below it. Installing `sentence-transformers`
switches the embedder to 384-dimensional semantic vectors automatically, and the
index code does not change.

### HNSW is approximate, so it is measured

`bench` runs the query set against both the HNSW index and exact brute-force
search over the identical vectors, and reports how often the approximate index
returns what exact search returns.

```
 efSearch   recall@k   top-1 agree
        8      1.000       10/10
       64      1.000       10/10
```

Recall is 1.000 at every `efSearch` on this corpus, which is what a graph index
does when the corpus is smaller than the candidate list. The number matters as
the corpus grows. On a safety corpus a missed neighbour can be the standoff
distance document, so raise `efSearch` until recall is 1.000 and keep it there.

### On the default embedder

`tfidf-dense` uses the vocabulary size as the dimension, so cosine in the index
is exactly the cosine of the sparse representation and nothing is traded away for
speed. `random-projection` gives fixed dimensions instead, and it is lossy.
Measured on this corpus against the exact ranking:

| dim | agreement@6 | top-1 |
|---|---|---|
| 512 | 0.567 | 8/10 |
| 1024 | 0.633 | 9/10 |
| 2048 | 0.700 | 10/10 |

That is why it is opt-in. Past roughly tens of thousands of vocabulary terms,
move to `sentence-transformers` rather than to projection.

### Cosine and Pearson

Cosine is the default and the right choice: vectors are L2 normalised, so the
inner product is the cosine and length has no vote. Pearson is cosine on
mean-centred vectors, so `index --metric pearson` centres each vector before
normalising and the whole index becomes a Pearson index. Metric is a property of
the index, not of the query. On this corpus the two rank almost identically,
which is expected for sparse TF-IDF where the mean sits near zero. Pearson earns
its keep against dense embeddings carrying a per-dimension bias.

### Diversity

At most two chunks per source document are kept, after over-fetching four times
`k`, so one verbose document cannot crowd out the protocol or reporting document
that the answer also needs.

## Anomalies

An unclassified detection has no label, so there is no protocol to look up by
name. The anomaly path makes three separate moves and keeps them separate.

**Retrieve the generic protocol.** The unknown-object procedure comes out of
`rag_assistant/kb/` like any other answer: treat as potentially hazardous, hold separation, do
not disturb, report, log for expert review.

**Rank the nearest known objects.** `rag_assistant/catalog/objects.json` holds known object
classes. Each carries a descriptor, the hazard class, what would confirm it,
what would rule it out, and the `rag_assistant/kb/` document that governs it, so a match keeps
the citation chain intact. Matching runs in one of two spaces:

- **embedding** when the detection record carries an `embedding` array and every
  catalog entry carries one too. This is the real path, and it uses your
  detection model's own space.
- **descriptor** otherwise, matching the operator's description against the
  entry text. Weaker, but it runs today without a trained embedding head.

The two are never mixed. A detection embedding against a descriptor-only catalog
is an error, not a silent fallback, and a catalog where only some entries carry
embeddings is rejected at index time.

**Generate an honest answer.** The model gets the protocol and the ranked
matches, and is held to saying "unidentified" first, presenting matches only as
ranked possibilities with their discriminators, and naming the escalation.

### The scores are not percentages

A cosine similarity of 0.78 is not "78% similar", and rendering it as a
percentage makes a ranking read as a confidence to an operator on deck. Matches
are printed as similarity scores with a rank, labelled in the prompt as "NOT a
probability, NOT a confidence, NOT an identification", and the model is
forbidden from converting one into a percentage or treating a higher score as
making an identity more likely true.

Similarity ranks candidates against each other. It says nothing about whether
the right answer is in the catalog at all.

## The guardrails

The system prompt is the safety-critical part of this system.

- Answer only from the `SOURCES` block. The model's own recollection is
  inadmissible.
- Cite `[S1]` after every distance, timing, procedure step and authority name.
- If the sources do not cover it, say so, then give only the universal fallback:
  do not approach, do not touch, do not recover, hold separation, report.
- Never invent a standoff distance, a disposal step, or a contact detail. A
  missing number is "not specified in the sources".
- Flag any citation whose document is marked `status: PLACEHOLDER`.

For an unclassified object, four more:

- Never state or imply an identity. Say "unidentified" first.
- Present nearest matches as possibilities, always with what would confirm and
  what would rule each one out. Never "this is a" or "likely a".
- Never convert a similarity score into a probability, a confidence, or a
  percentage.
- Escalate to a qualified human expert, and say the object stays unidentified
  until that expert rules.

No document in `rag_assistant/kb/` is a placeholder any more. All seven are written from real
publications, carry `status: verified`, and name the file in `rag_assistant/sources/` they came
from. The rule stays in the prompt because the index still warns on a
`PLACEHOLDER` document and the assistant must flag one if it ever appears.

What the corpus deliberately still lacks is numbers. Most of these publications
do not state a universal standoff distance, and none of them names an Indian
authority for an ordnance report. Ask for either and the assistant says so
rather than inventing one. That is the system working, not a gap to paper over.

## Adding real sources

Drop a `.md` or `.txt` file in `rag_assistant/kb/` with front matter, then re-index.

```yaml
---
title: Underwater UXO Handling
authority: <issuing body>
source_url: <url>
source_file: sources/<the file it was written from>
doc_type: safety procedure
status: verified
retrieved: 2026-09-10
---
```

Put the publication itself in `rag_assistant/sources/` and add a row to `rag_assistant/sources/PROVENANCE.md`.
Nothing in `rag_assistant/sources/` is indexed; it is evidence, and it is what lets any quote
in an answer be traced back to a real document. The chat interface links it: a
citation whose document names a `source_file` opens the original PDF.

`python -m rag_assistant.rag index` prints a warning listing every document still marked
`PLACEHOLDER`.

Catalog entries in `rag_assistant/catalog/objects.json` follow the same discipline. Every
entry cites the `rag_assistant/kb/` document that governs it and carries its own `status`, and
`length_m` is deliberately `null` throughout: a fabricated size range would be
read as evidence by an operator.

## The survey hazard map

A second subsystem, alongside the assistant and sharing nothing with it but the
dashboard it appears in. It takes a side-scan survey and a checkpoint and
produces a ranked picture of where the hazards are and which one to look at
first.

    THE HAZARD MAP ANSWERS    where things are, and how urgent they are
    THE RAG ASSISTANT ANSWERS what a thing is, and what is known about it

They stay separate on purpose. A severity score is arithmetic over a detector's
output and can be recomputed by hand. A grounded answer is retrieval over this
corpus. Merged, a confident sentence could raise a priority, or a priority could
imply a fact, and neither system can support that. A hotspot handed to the
assistant travels as its own context object and carries its own severity, so the
same contact cannot show one urgency on the map and another beside the answer.

```bash
pip install -r requirements-survey.txt

# process a survey
python -m survey_hazard_map.run_survey --strips samples/sidescan-s7-submarine.jpg \
    --model survey_hazard_map/models/marine/marine.pt --out data/surveys/s7-submarine

# or see the whole pipeline with no survey and no checkpoint
python -m survey_hazard_map.demo_survey --out data/surveys/demo-synthetic
```

Then open the dashboard at `/map`. The page reads `export.json` through
`/survey/{id}/export` and renders it; nothing is recomputed in the browser.

Three things it will not do, stated here because they govern how much its
numbers mean:

* Positions are pixel offsets inside the sonar strip unless navigation is
  supplied. They are **not GPS**, every latitude is `null`, and the export says
  `"coordinate_mode": "Relative Survey Coordinates"`.
* Severity is a configurable heuristic chosen for this project. It is **not**
  Navy, Coast Guard, NOAA or IMO procedure and carries no authority.
* A YOLO detection is a prediction, not a fact. Ten tiles of the real waterfall
  record in `survey_hazard_map/samples/` produce four detections and all four look like false
  positives on nadir boundaries.

Full documentation, including the JSON contract, the severity policy, the
coordinate assumptions and the handoff shape: [docs/SURVEY_HAZARD_MAP.md](docs/SURVEY_HAZARD_MAP.md).

```bash
python survey_hazard_map/tests/unit_tests.py                      # 40 unit tests
python survey_hazard_map/tests/smoke_test.py                      # 191 end-to-end checks
python3 validate_output.py data/surveys/s7-submarine
python3 real_model_test.py                 # real checkpoint, or skips with a reason
```

## The chat application

The CLI answers one question and exits. The chat application is the same engine
with a conversation around it: one input box, follow-ups that keep context, and
citations an operator can open.

```
backend/     One FastAPI app over two engines.
  app/              the Supabase-backed shell: uploads, persistence, history
    main.py           the application, CORS, health, router wiring
    routes/           detection, rag, hazard, history, stats
    services/         persistence, and the bridge to the assistant
    supabase_client.py  optional: absent credentials cost history, not the app
  config.py         every knob: paths, retrieval, severity, detector classes
  schemas.py        pydantic request and response models
  chat.py           the seam: routing, history, retrieval, citations, grounding
  detect.py         sonar tile in, detection records out
  detector_worker.py the models, in their own process
frontend/    One Vite app: the dashboard, with the assistant as a page in it.
  src/pages/Assistant.jsx    the route
  src/assistant/             the chat, self-contained
    components/              ChatWindow, MessageBubble, CitationPanel, Composer
    config/theme.ts          every colour, font and spacing value
    config/copy.ts           every string the operator reads
    config/settings.ts       API base URL and feature flags
    lib/api.ts               the only module that talks to the backend
    assistant.css            scoped under .dq-assistant
eval/        forty cases and the runner
```

The chat stylesheet is scoped under a single root class. The dashboard and the
chat both defined `.status-dot` and both defined `.app`, and an unscoped merge
would have quietly restyled the sidebar.

Retrieval, chunking, the system prompt and the corpus are untouched by all of
it. `rag.py` gained one thing: a `stream()` method beside each provider's
`complete()`, so an answer can be delivered as it is written.

### Running it

Three commands, all local.

```bash
.venv/bin/pip install -r requirements-server.txt
DEEPECHO_ENABLE_UPLOAD=1 .venv/bin/python -m uvicorn backend.app.main:app --port 8000

cd frontend && npm install && npm run dev      # http://localhost:5173
```

The assistant is the last item in the sidebar. Do not run `npm run build` while
`npm run dev` is running; they share a build directory and the dev server starts
returning 500.

Set `DEEPECHO_PROVIDER=groq` in `.env` for the demo. Gemini's free tier returns
503 under load often enough to hit one mid-answer, and Groq answers in about a
second.

### Running the whole thing in Docker

Nothing to install but Docker. One container per feature, a gateway in front,
and the dashboard:

```bash
cp .env.example .env          # optional: add a Gemini or Groq key
docker compose up --build     # then open http://localhost:5173
```

The assistant, the hazard map and GhostTrace are separate containers, so each
can be rebuilt or restarted alone (`docker compose up -d --build ghosttrace`).
Live reload for editing, the routing table, and deploying a feature on its own
are in [docs/DOCKER.md](docs/DOCKER.md).

### Deploying

Two services. The API runs in a container because torch is 583 MB on disk and
165 MB resident, and a native Python build handles that badly. The dashboard
does not need one: `npm run build` produces static files.

```bash
docker build --build-arg PROFILE=serve -t deepecho .
docker run -p 8000:8000 --env-file .env deepecho
```

Two build profiles, and the choice is about memory rather than features.

| Profile | Contains | Size | Fits |
|---|---|---|---|
| `serve` (default) | assistant, corpus, pre-generated surveys | ~250 MB | a 512 MB instance |
| `full` | adds YOLOv8 and survey processing | ~1.2 GB | 2 GB and up |

On `serve` there is no torch in the image, so `DEEPECHO_ENABLE_UPLOAD` stays 0
and `/detect` is not registered at all. A deployment that cannot detect anything
should not advertise a detection endpoint, even one that answers honestly from
the stub.

`render.yaml` is a blueprint for both services. Supabase credentials, the model
provider keys and the dashboard origin are the only values it needs, and every
one of them is marked `sync: false` so nothing secret lives in the repository.
Without Supabase the API still serves the assistant and the surveys, and
`/health` reports why history is unavailable.

The image builds the vector index rather than shipping one. A committed index
that has drifted from the corpus is worse than no index, and it takes a second.

### Setting up on another machine

A fresh clone is missing three things by design: the vector index, the API keys,
and the detector dependencies. All three are one command each.

```bash
git clone https://github.com/Srishti39-sudo/ECHELON.git
cd ECHELON
git checkout dashboard        # the default branch; what the demo and the deploy run

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-server.txt -r requirements-ghosttrace.txt
# requirements-detector.txt (PyTorch, ~2 GB) is NOT needed: the committed
# marine.onnx runs the detector under onnxruntime. Install it only to train.

cp .env.example .env        # then put a key in it, see below
.venv/bin/python -m rag_assistant.rag index

cd frontend && npm install && cd ..
```

Then two terminals:

```bash
DEEPECHO_ENABLE_UPLOAD=1 .venv/bin/python -m uvicorn backend.app.main:app --port 8000
cd frontend && npm run dev
```

Open `http://localhost:5173` and pick Assistant in the sidebar.

**The index is not in the repo.** `index.faiss`, `index.json` and `vectors.npy`
are built from `rag_assistant/kb/` and are gitignored, because a stale committed index that
disagrees with the corpus is worse than no index. `rag.py index` rebuilds them
in about a second and prints what it indexed.

**To get the same assistant as the demo machine**, not just a working one, match its
three choices. The demo machine embeds with NVIDIA Nemotron, reranks with NVIDIA, and
answers with Groq. A bare `rag index` with no keys falls back to a local embedder and
retrieves differently, and a bare `.env` answers with Gemini, whose free tier returns
503 under load. So, with the team's keys in `.env` (shared privately, never through git):

```
DEEPECHO_PROVIDER=groq          # in .env
.venv/bin/python -m rag_assistant.rag index --embedder nvidia
```

`/health` then reports `embedder: nvidia`, `index: faiss-hnsw`, `provider: groq`, which
is what the demo machine reports. A shell variable overrides `.env`, so if `/health`
disagrees with the file, check the terminal that started the server.

**Keys are not in the repo either.** `.env` is gitignored and always should be.
A free Groq key from https://console.groq.com/keys is enough, and Groq is the
one to use: it answers in about a second where Gemini's free tier returns 503
under load. Put `GROQ_API_KEY=...` and `DEEPECHO_PROVIDER=groq` in `.env`.

**The model and the sources are in the repo.** `survey_hazard_map/models/marine/marine.pt`
is committed with its calibration and with `marine.onnx`, its ONNX export (parity with the
checkpoint measured in `docs/edge_parity.json`). The detector worker picks the `.onnx` when
onnxruntime imports, so a fresh clone uploads a sonar image or a raw `.xtf` log and watches
the model detect with no PyTorch installed. The publications in `rag_assistant/sources/` are
committed too, so citations resolve to real files. Every demo survey under `data/surveys/`
ships with its `export.json` and `ghosttrace.json`, so the dashboard, the hazard map and the
GhostTrace page show the same results on a clone as on the machine that produced them.

**Four habitat layers are not in the repo.** The UNEP-WCMC coral reef and seagrass layers
under `data/ghosttrace/layers/` are fetched, not committed, because their licence forbids
redistribution. The committed GhostTrace results were computed with them. A clone that
presses "Run GhostTrace" without them falls back to the sparse OpenStreetMap reef layer and
scores habitat lower; the output says so in `habitat.notes`. To re-run with the same inputs:

```bash
.venv/bin/pip install -r requirements-ghosttrace.txt
.venv/bin/python -m ghosttrace.tools.fetch_ghosttrace_data --only reefs,seagrass
```

**Without a key**, retrieval still works and can be demonstrated:
`python -m rag_assistant.rag search "who do I report a mine to" -k 5` needs no network at all.

### Endpoints

| Route | What it does |
|---|---|
| `POST /chat` | One turn, answered whole |
| `POST /rag/query` | The same engine on a simpler contract; `retrieve_only` skips the model |
| `GET /hazard/map`, `/history`, `/stats` | Stored scans and detections, needs Supabase |
| `POST /chat/stream` | The same turn, streamed as server-sent events |
| `POST /detect` | A sonar tile in, detection records out. Behind `DEEPECHO_ENABLE_UPLOAD` |
| `GET /health` | Index status, corpus size, provider, detector state |
| `GET /sources/...` | The original publications, read-only |

The streaming route emits `data: {json}` frames typed `meta`, `sources`,
`delta`, `done` and `error`. Retrieval finishes before generation starts, so the
citations go out before the first word and the panel fills while the answer is
still arriving. `grounded` can only be known once the whole answer exists, so it
rides in the `done` frame alone. That frame is validated against the same model
`POST /chat` returns, so the two cannot drift apart.

### What the interface has to show

Four states must never look like a confident answer, and each is rendered
differently from one.

- **Ungrounded.** The answer carries no citation that resolves. Marked
  unverified, with an instruction to confirm manually.
- **Partly outside the references.** The answer is cited but declines to fill a
  gap. Marked as such, because a missing standoff distance is unavailable, not
  omitted.
- **Unidentified object.** Nearest known objects are listed as possibilities
  with what would confirm and what would rule each one out, under a heading
  saying nothing below identifies anything. Similarity is shown as a score and
  never as a percentage.
- **Placeholder detection.** While the detector is a stub, every record it
  produces is labelled synthetic wherever it appears.

### Evaluation

Forty cases in `eval/cases.jsonl`, run by `eval/run.py`, checked mechanically.

```bash
python rag_assistant/eval/run.py                      # in-process, no server needed
python rag_assistant/eval/run.py --url http://127.0.0.1:8000
python rag_assistant/eval/run.py --category refusal --verbose
python rag_assistant/eval/run.py --repeat 3           # generation is not deterministic
```

Nothing in the suite asks a model to grade another model. A suite whose purpose
is evidence cannot rest on the same machinery it is testing, so every check is a
regex, a set membership, or a string lookup against the text that was actually
retrieved. The exit code is 1 on any failure, so it can gate a commit.

| Category | Cases | What it holds the assistant to |
|---|---|---|
| refusal | 8 | The corpus is silent, so the answer says so and gives only the fallback |
| grounding | 7 | The corpus does cover it, and the answer cites it |
| anomaly | 6 | Never an identity, never a similarity rendered as a percentage |
| routing | 6 | The four intents resolve without the operator picking one |
| coverage | 4 | A class with no governing document is flagged, not answered around |
| authority | 5 | A body is named only where a source connects it to that hazard |
| detector | 4 | Severity is looked up from the table, not read out of prose |

Two checks run on every case whether it asks for them or not.

**Citations resolve.** Every `[Sn]` in the answer must point at a source that was
really retrieved. A marker past the end of the list means the model numbered
something it was never given.

**No invented numbers.** Every quantity carrying a unit is extracted from the
answer and must appear in the retrieved text. A standoff distance, a depth or a
delay that no source stated is the exact failure this system exists to prevent,
and it is detectable without judgement. Citation markers and ordered-list
numbering are stripped first, or `[S3]` and `3.` become quantities.

The suite earned its place on its first run by finding a real bug: the engine
read `label` while the API spoke `object_class`, and the mapping lived only in
the HTTP layer. Anything calling the engine directly had its classified contacts
silently treated as anomalies. `_prepare_turn` now normalises the field itself.

### The detector

One checkpoint, run over every tile.

| Checkpoint | Base | Classes | Held-out mAP50 |
|---|---|---|---|
| `survey_hazard_map/models/marine/marine.pt` | YOLO11s | shipwreck, aircraft, human, pipeline, fishing_gear, mine_like_object | 0.61 (pipeline 0.99, aircraft 0.90, mine-like 0.60, shipwreck 0.52, fishing gear 0.34) |

The kit beside it in `survey_hazard_map/models/marine/` is what the survey pipeline runs:
`sonar_detector.py` (tiled inference, NMS, calibrated confidence),
`geotag.py` (ping headers from an `.xtf`, or a NavTable CSV, to WGS-84
positions and sizes in metres), `shadow_check.py` (acoustic-shadow physics,
no model) and `sonar_pipeline.py`, which chains them. `calibration.json` is
the identity: raw confidence on the test split had an ECE of 0.041, and Platt
scaling made it worse, so none is applied.

Two earlier stand-in checkpoints, `known.pt` and `anomaly.pt`, are retired.
Their class names remain in `DETECTOR_CLASS_MAP` so stored detections from
that era still resolve, and the merge in `backend/detect.py` still handles
several models if one is added again.

Inference runs in a subprocess. faiss and torch each bundle their own libomp,
and on macOS the second to initialise aborts the process. The documented
workaround is a flag whose own description says it may silently produce
incorrect results, which is not a trade this system should make, so the two
libraries are kept in separate processes instead. `backend/detector_worker.py`
loads the weights once and stays up.

### Per-class confidence floors

Measured across seven public side-scan tiles, six of them wrecks and none of
them containing a human:

| class | n | min | max | note |
|---|---|---|---|---|
| ship | 8 | 0.318 | 0.829 | the workhorse class, behaves well |
| other | 3 | 0.359 | 0.545 | the anomaly model's "I cannot name it" |
| shipwreck | 1 | 0.843 | 0.843 | |
| human | 1 | 0.463 | 0.463 | fired on debris beside a wreck, a false positive |
| aircraft | 1 | 0.322 | 0.322 | fired on a wreck, a false positive |

Both false positives came from `known.pt` and both sat below 0.5, while the
correct ship calls clustered higher. `CLASS_CONFIDENCE_FLOOR` in
`backend/config.py` sets a floor per class from that, and from consequence:
"human remains" is the highest-consequence claim in the vocabulary, it is a
legal and humanitarian assertion, and no document in the corpus supports any
procedure for it. It gets the strictest floor at 0.75.

A class below its floor is **downgraded, never dropped**. Deleting the box would
hide a contact from the operator, which is worse than reporting one without a
name. The contact becomes `unknown`, routes to the unidentified-object protocol,
and carries a `downgraded_from` note recording exactly what the model called it
and why that call was not taken at face value. Nothing is hidden and nothing is
asserted.

One observation per false-positive class is not a fitted threshold. These are
precautionary, and the right next step is to retune them against a labelled
validation set.

### The vocabulary gap

The detector recognises aircraft, human remains, fish and ships. The corpus is
about ordnance, wrecks, debris and unidentified objects. Those lists only
partly overlap, and pretending otherwise is how a body becomes a mine.

`CLASS_COVERAGE` in `backend/config.py` records which detected classes the
corpus actually has a document for. A class marked `False` puts a note in the
prompt telling the model to say plainly that no reference covers it, and sets
`coverage_gap` on the response so the interface can show it. Only `shipwreck`
and the unknown class are covered today. An aircraft wreck, a fish shoal and
human remains are not, and the assistant says so instead of answering from a
document about something else.

Filling that gap is a corpus job, not a code job. The most valuable additions
are a document on wrecks containing human remains and the reporting obligation
that attaches to them, and an Indian procedure for reporting suspected ordnance.

## The final model: seven classes

`survey_hazard_map/models/final/` holds the model this submission is built on:

| File | What it is |
|---|---|
| `final.pt` | YOLO11s, 9.4 M parameters, 19 MB. `marine.pt` (six classes) fine-tuned 30 epochs with a seventh class, `ghost_net`, on procedurally rendered nets with acoustic shadows composited onto real seabed tiles. Loads with all seven class names. |
| `calibration.json` | Identity calibration: raw confidence already had a lower calibration error (0.038) than Platt scaling (0.043). The pipeline reads it beside the weights. |
| `RESULTS.md` | The full scorecard: final vs marine, marine vs the two original checkpoints, a YOLO26 vs YOLO11 architecture benchmark, calibration and speed, and what to say honestly. |
| `DATA_ATTRIBUTION.md` | Every dataset the model saw, with its licence. No images are redistributed here. |

Per class, on 1,533 held-out test tiles never used in training, confidence 0.25, IoU 0.5 (AP50, precision / recall):

| Class | final.pt | marine.pt (6 classes) |
|---|---|---|
| pipeline | **0.99** (0.98 / 0.99) | 0.99 |
| ghost_net | **0.93** (0.89 / 0.90), synthetic held-out regime, 648 boxes | no class |
| aircraft | 0.88 (0.82 / 0.90) | 0.90 |
| mine_like_object | **0.66** (0.69 / 0.62) | 0.60 |
| shipwreck | **0.58** (0.69 / 0.54) | 0.53 |
| human (3 test objects) | 0.33 | 0.33 |
| fishing_gear | 0.25 (0.37 / 0.27) | 0.33 |
| **all classes, Ultralytics validator** | **mAP50 0.655 · mAP50-95 0.466** | mAP50 0.605 · mAP50-95 0.408 |
| false alarms per empty seafloor tile | **0.26** | 0.30 |

Speed: 11.3 ms per tile on a Tesla T4, about 0.5 s per tile on an Apple-silicon CPU.

Two things `RESULTS.md` says that the table cannot: the `ghost_net` number is measured on synthetic
nets rendered under a different parameter regime and on different surveys than the training nets, because
no public real ghost-net side-scan dataset exists; and `fishing_gear` fell from 0.33 to 0.25 when the net
class was added, because tangled nets and crab-pot strings overlap visually. Both are stated on the slide
and in the speaker notes rather than hidden.

**Which model runs in the app.** The pipeline is wired to `survey_hazard_map/models/marine/marine.pt`
with its ONNX export, because that is the checkpoint the parity, verification and end-to-end tests were
measured against. To run the seven-class model live: copy `final.pt` and `calibration.json` over
`marine.pt` and its calibration file, re-run `survey_hazard_map/tools/export_onnx.py`, and re-run the
survey test suites. That swap is a one-minute change kept out of the demo build on purpose.

## Licence

All rights reserved, Team Echelon, 2026. The repository is public so it can be viewed and run for the
evaluation of this Smart India Hackathon 2026 submission and for no other purpose; see `LICENSE`.
Third-party components keep their own licences, listed in `NOTICE`. The detector is built with
Ultralytics YOLO, which is AGPL-3.0, and its terms apply to the parts of this software that use it.

