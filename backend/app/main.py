"""The DeepEcho API.

One application over two engines. The shell owns uploads, persistence and
history through Supabase; the engines own detection and grounded answering. The
routers below are the seam and nothing crosses it except data.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.schemas import HealthResponse

# Each feature is its own package with its own routes; main.py only wires them.
from rag_assistant import chat
from rag_assistant.routes.rag import router as rag_router
from survey_hazard_map import detect
from survey_hazard_map.routes import stats
from survey_hazard_map.routes import jobs as survey_jobs
from survey_hazard_map.routes.detection import router as detection_router
from survey_hazard_map.routes.hazard import router as hazard_router
from survey_hazard_map.routes.history import router as history_router
from survey_hazard_map.routes.survey import router as survey_router
from survey_hazard_map.store import storage_status
from ghosttrace.routes.api import router as ghosttrace_router
from ghosttrace.routes.telemetry import router as telemetry_router

log = logging.getLogger("deepecho")

# The feature groups this process serves; see config.FEATURES.
ASSISTANT = "assistant" in config.FEATURES
HAZARD = "hazard" in config.FEATURES
GHOSTTRACE = "ghosttrace" in config.FEATURES


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the index and the detector, in that order and never the other way.

    faiss and torch each ship their own copy of libomp, and on macOS whichever
    initialises second aborts the process. Detector inference lives in a
    subprocess for exactly that reason, so the two never share an address space.
    Warming here also moves index and weight loading off the first request.

    Only what this process serves is warmed (config.FEATURES): a GhostTrace
    container has no use for the index, and should not hold it in memory.
    """
    log.info("features: %s", ", ".join(sorted(config.FEATURES)))

    if ASSISTANT:
        try:
            chat.get_retriever()
            chat.get_catalog()
        except chat.EngineError as exc:
            log.warning("index not loaded at startup: %s", exc)

    if HAZARD and config.ENABLE_UPLOAD:
        loaded = detect.load_models()
        log.info("detector: %s", ", ".join(sorted(loaded)) or "none, using stub")

    if HAZARD:
        log.info("storage: %s", storage_status())
    yield


app = FastAPI(
    lifespan=lifespan,
    title="deepEcho",
    description="AI-powered underwater sonar detection and grounded decision support",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The publications the corpus was written from, served read-only so a citation
# can open the document behind it. This is what makes a claim checkable rather
# than merely attributed.
if ASSISTANT and config.SOURCES_DIR.is_dir():
    app.mount(config.SOURCES_MOUNT, StaticFiles(directory=config.SOURCES_DIR), name="sources")


@app.get("/")
def home():
    return {
        "message": "deepEcho backend is running",
        "features": sorted(config.FEATURES),
        "assistant": "/chat, /chat/stream, /rag/query",
        "detection": "/detect",
        "survey": "/survey, /hazard/map, /history, /stats, /detections",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Is the index loaded, what is behind it, and is storage reachable."""
    features = sorted(config.FEATURES)
    upload_enabled = HAZARD and config.ENABLE_UPLOAD
    # History belongs to the hazard feature, and so does its database. The
    # other containers never open it: under compose it is one SQLite file on a
    # shared mount, which is safe with one writer process and not with three.
    storage = storage_status() if HAZARD else "not served by this process"
    detector = "disabled"
    detector_models: list[str] = []
    if upload_enabled:
        loaded = detect.load_models()
        detector_models = sorted(loaded)
        detector = "loaded" if loaded else "stub"

    # A process that does not serve the assistant has no index to report, and
    # that is not a fault: it is ready for what it does serve. Asking for the
    # retriever here would also load the index on the first health probe.
    retriever = None
    if ASSISTANT:
        try:
            retriever = chat.get_retriever()
        except chat.EngineError:
            pass
    if retriever is None:
        return HealthResponse(
            status="degraded" if ASSISTANT else "ready",
            corpus_loaded=False, documents=0, chunks=0,
            embedder="none", index="none", catalog_entries=0, catalog_space=None,
            provider=config.PROVIDER, model=config.MODEL or "provider default",
            detector=detector, detector_models=detector_models,
            upload_enabled=upload_enabled, storage=storage,
            features=features,
        )

    catalog = chat.get_catalog()
    return HealthResponse(
        status="ready",
        corpus_loaded=True,
        documents=len({c.meta.get("doc_id") for c in retriever.chunks}),
        chunks=len(retriever.chunks),
        embedder=getattr(getattr(retriever, "embedder", None), "kind", retriever.mode),
        index=getattr(getattr(retriever, "index", None), "kind", retriever.mode),
        catalog_entries=len(catalog.entries) if catalog else 0,
        catalog_space=catalog.space if catalog else None,
        provider=config.PROVIDER,
        model=config.MODEL or "provider default",
        detector=detector,
        detector_models=detector_models,
        upload_enabled=upload_enabled,
        storage=storage,
        features=features,
    )


if ASSISTANT:
    app.include_router(rag_router)

# The upload path is behind one flag, and the flag has to gate the route rather
# than only the model loading. Without this the serve container, which ships no
# torch, still advertises /detect and answers it from the stub: a synthetic
# detection, correctly labelled, from a deployment that cannot detect anything.
# Off means the route does not exist.
if HAZARD:
    if config.ENABLE_UPLOAD:
        app.include_router(detection_router)
    app.include_router(history_router)
    app.include_router(hazard_router)
    app.include_router(stats.router)

# The survey hazard map. Reads pre-generated surveys off disk, so it needs no
# database, no detector and no key, and cannot fail at startup.
#
# Note for whoever tidies this up: /hazard/map above answers a similar-sounding
# question from Supabase scan rows, with severity from a detection count. This
# router answers it from a processed survey, with severity from class weight
# times confidence. They are two different models of the same idea and the
# project should eventually keep one. Nothing here touches the other.
#
# Survey jobs (upload a log, watch it process, download the reports) go first:
# /survey/jobs/{id} would otherwise be read as /survey/{survey_id}/... by the
# router below. Gated like /detect -- the route exists only where a detector
# can really run; see routes/jobs.py for the flag and why.
if HAZARD:
    if survey_jobs.ENABLED:
        app.include_router(survey_jobs.router)
    app.include_router(survey_router)

# GhostTrace: which detected ghost net to recover first, and why. Reads
# ghosttrace.json beside a survey's export; its run route has its own flag, see
# routes/ghosttrace.py. Prefix /ghosttrace, so it cannot shadow /survey.
if GHOSTTRACE:
    app.include_router(ghosttrace_router)
    # Field telemetry: drifter tags and recovery confirmations for GhostTrace
    # targets. Same feature, since every route is keyed by a survey's targets.
    app.include_router(telemetry_router)
