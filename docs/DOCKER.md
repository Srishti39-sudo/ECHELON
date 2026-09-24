# Running DeepEcho in Docker

One command brings up the whole project on any machine with Docker, and each
feature runs in its own container so it can be edited, rebuilt, restarted and
deployed without touching the others.

```
browser ──> frontend   :5173   the dashboard (static files, or Vite in dev)
       └──> gateway    :8000   sends each API path to the feature that owns it
              ├── assistant   :8001   /chat  /rag  /sources
              ├── hazard      :8002   /survey  /hazard  /history  /stats
              │                       /detections  /detect  /scans
              └── ghosttrace  :8003   /ghosttrace
```

## First run

You need Docker Desktop (or Docker Engine) with **Compose 2.24 or newer**:
`docker compose version`.

```bash
git clone <repo> && cd ECHELON
cp .env.example .env             # optional, see below
docker compose up --build
```

Open <http://localhost:5173>. The first build downloads the Python and Node
packages and takes a few minutes; after that it is cached.

`.env` is optional. With no keys the maps, surveys and GhostTrace work fully and
the assistant answers in its offline mode (retrieved passages, no generated
text). For generated answers put a `GEMINI_API_KEY` or `GROQ_API_KEY` in `.env`.
Keys are passed to the containers at run time and are never baked into an image.

Port 8000 is also where a locally started `uvicorn` listens. Stop one before
starting the other.

## Editing with live reload

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

or put `COMPOSE_FILE=compose.yaml:compose.dev.yaml` in `.env` once (`;` instead
of `:` on Windows) and keep typing `docker compose up`.

Your working tree is mounted into the containers:

| You edit | What happens |
|---|---|
| `backend/**`, `survey_hazard_map/**`, `ghosttrace/**`, `rag_assistant/**` | the three API containers reload on save |
| `frontend/src/**` | Vite hot-reloads the page |
| `rag_assistant/kb/` or `rag_assistant/sources/` | delete `index.json`, then `docker compose restart assistant` rebuilds the index |
| a `requirements-*.txt` | `docker compose up -d --build` |
| `frontend/package.json` | `docker compose restart frontend` (it runs `npm ci` on start) |
| `docker/gateway.conf` | `docker compose restart gateway` |

On Windows keep the clone inside WSL2 (`\\wsl$\...`), not under `C:\`. File
change events do not cross from the Windows filesystem into a Linux container,
so neither reload would fire.

## Working on one feature

```bash
docker compose up gateway frontend ghosttrace     # start only what you need
docker compose up -d --build hazard               # rebuild + restart one feature
docker compose logs -f assistant                  # follow one feature's logs
docker compose restart ghosttrace
docker compose stop hazard                        # the rest keeps working
```

The gateway looks a feature up on every request, so it does not care which ones
are running. A path whose feature is down answers 502 and everything else
carries on.

Each feature also has its own port, with its own interactive API docs:
<http://localhost:8001/docs> (assistant), <http://localhost:8002/docs> (hazard),
<http://localhost:8003/docs> (ghosttrace). Health for each is at
`http://localhost:8000/_health/<feature>`.

## How the split works

The three API containers are **one image** (`./Dockerfile`) started three times
with a different `DEEPECHO_FEATURES`. `backend/config.py` reads it and
`backend/app/main.py` mounts only that feature's routers and warms only what it
needs: the GhostTrace container never loads the retrieval index, and only the
hazard container opens the history database.

It is one image on purpose. The features share code (history asks the
assistant's severity policy; the assistant's copilot reads the surveys), so
three separate codebases would mean three copies of it. What is separate is the
process, and that is what restarting, scaling and deploying act on.

Unset, `DEEPECHO_FEATURES` means `all`, so `uvicorn backend.app.main:app` on a
laptop behaves exactly as before.

`./data` is mounted into all three. Surveys are the shared ground: a job the
hazard container finishes is what GhostTrace ranks and what the copilot answers
from. They stay ordinary files on your machine.

### Adding or moving a route

Three places, and they have to agree:

1. the feature's own `routes/` folder holds the router; `backend/app/main.py` includes it under that feature's `if`
2. `docker/gateway.conf` — the `map` line that sends the path prefix there
3. for a new feature: a name in `config.ALL_FEATURES` and a service in `compose.yaml`

If the gateway sends a path to a container that does not serve it, FastAPI
answers 404. If no line matches, the gateway answers 404 and says so.

## The detector (`full` profile)

The default image has no torch, so `/detect` and survey jobs are not registered.
To get them:

```bash
DEEPECHO_PROFILE=full docker compose up --build
```

That image is about 1.3 GB and the hazard container wants 2 GB of memory. It
needs the checkpoints in `models/`.

## Deploying a feature on its own

Every feature is the same image with one variable, so any container host works:

```bash
docker build -t deepecho-api .
docker run -p 8000:8000 --env-file .env -e DEEPECHO_FEATURES=ghosttrace deepecho-api
```

On separate hosts there is no gateway to put them back under one address. Either
run `docker/gateway.conf` somewhere with the three `proxy_pass` targets changed
to the real hostnames, or deploy one container with `DEEPECHO_FEATURES=all`,
which is what `render.yaml` does today. Set `DEEPECHO_CORS_ORIGINS` to the
dashboard's origin, and build the dashboard with `VITE_API_BASE_URL` pointing at
the gateway: it is compiled into the bundle, so changing it means rebuilding.

Surveys written by a hazard container on one host are not visible to a
GhostTrace container on another. Split across hosts, `data/` needs shared
storage, or the features that must see each other's output stay together.

## When something is wrong

| Symptom | Cause |
|---|---|
| dashboard loads, every request fails with a CORS error | the gateway or that feature is down (a 502 carries no CORS headers, so the browser reports CORS). `docker compose ps` |
| CORS error with everything up | the dashboard is not on port 5173. Set `DEEPECHO_CORS_ORIGINS` in `.env` |
| `port is already allocated` | a local `uvicorn` or `npm run dev` is still running |
| `env_file … required` or `!reset` errors | Compose older than 2.24 |
| assistant `/health` says `degraded` in dev | no index yet; `docker compose restart assistant` builds it |
| container exits with `DEEPECHO_FEATURES names unknown feature` | a typo in the feature name |
