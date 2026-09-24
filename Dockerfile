# The DeepEcho backend.
#
# Two build profiles, chosen at build time, because torch is 583 MB on disk and
# 165 MB resident and most deployments do not need it:
#
#   --build-arg PROFILE=serve    the assistant, the corpus, and pre-generated
#                                surveys. No torch. Roughly 250 MB.
#   --build-arg PROFILE=full     adds the YOLOv8 checkpoints and survey
#                                processing. Roughly 1.2 GB.
#
# Serve is the default. On a 512 MB instance it is the only one that fits.
#
# A third, separate target runs the survey engine on a vehicle or edge box:
#
#   --target edge                the survey engine and the detector, offline,
#                                with no torch and no assistant. The YOLOv8
#                                weights run as ONNX under onnxruntime.
#
#   python tools/export_onnx.py            # once, on a machine with ultralytics
#   docker build --target edge -t deepecho-edge .
#   docker run --rm -v "$PWD/survey:/survey" deepecho-edge \
#       --strips /survey/strips --model models/known.onnx models/anomaly.onnx \
#       --out /survey/out
#
# It is a build TARGET rather than a third PROFILE value because it shares
# almost nothing with the other two: no corpus, no index, no FastAPI, no LLM
# client. See docs/EDGE.md for what it does and does not include, and the
# measured numbers.
#
# The stages are ordered so that `docker build .` with no --target still builds
# the serve/full image exactly as before: the default target is the LAST stage.


# --- Checkpoints for serve/full ----------------------------------------------
# models/ can also hold ONNX exports (tens of MB each, several variants). The
# serve and full images cannot run them (neither installs onnxruntime), so they
# are filtered out here rather than being copied into a layer where deleting
# them later would not shrink the image. Anything else in models/ still goes.
FROM python:3.13-slim AS checkpoints
COPY survey_hazard_map/models/ /models/
RUN find /models -name "*.onnx" -delete && rm -rf /models/trt_cache


# --- Edge: survey engine + ONNX detector, no torch ----------------------------
FROM python:3.13-slim AS edge

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPECHO_DETECTOR_BACKEND=onnx

WORKDIR /app

# onnxruntime's Linux CPU wheels carry their own thread pool and need no
# OpenMP runtime, so unlike the image below there is no apt step at all.
# opencv-python-headless and scipy are the verification cues' dependencies
# (hazard_verify.py); both are self-contained manylinux wheels.
COPY requirements-survey.txt requirements-edge.txt ./
RUN pip install -r requirements-survey.txt -r requirements-edge.txt \
        opencv-python-headless scipy

# The whole survey engine package, so a module added later is not silently
# missing from the vehicle image. None of them import torch at module load;
# the ones that could (hazard_detect's ultralytics path) do so lazily and only
# for a .pt. backend/ is here for config.py only.
COPY backend/config.py backend/__init__.py ./backend/
COPY survey_hazard_map/ ./survey_hazard_map/
COPY vendor/ ./vendor/

# The detector runs as ONNX here. Run survey_hazard_map/tools/export_onnx.py
# first so survey_hazard_map/models/marine/ holds the export; the .pt is not
# usable in this image. The int8 variant is left out on purpose: it did not
# keep parity (docs/EDGE.md).

ENTRYPOINT ["python", "-m", "survey_hazard_map.run_survey"]
CMD ["--help"]


# --- Serve / full: the API --------------------------------------------------
FROM python:3.13-slim AS base

ARG PROFILE=serve
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPECHO_HOST=0.0.0.0

WORKDIR /app

# libgomp is faiss's OpenMP runtime. On the full profile torch brings its own,
# which is why detector inference runs in a subprocess: the two cannot
# initialise in one address space. See backend/detector_worker.py.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# requirements-ghosttrace.txt is installed on every profile. The image is run
# once per feature (compose.yaml, DEEPECHO_FEATURES), and one image that can be
# any of the three is simpler to build, cache and ship than three that cannot.
COPY requirements.txt requirements-server.txt requirements-survey.txt requirements-detector.txt \
     requirements-ghosttrace.txt ./
RUN pip install -r requirements.txt -r requirements-server.txt -r requirements-survey.txt \
        -r requirements-ghosttrace.txt \
 && if [ "$PROFILE" = "full" ]; then pip install -r requirements-detector.txt; fi

# One package per feature. The assistant first: its corpus changes least, so
# the layer (and the index built from it) caches across code edits.
COPY backend/ ./backend/
COPY rag_assistant/ ./rag_assistant/

# The index is built here rather than committed. A stale index that disagrees
# with the corpus is worse than no index, and building takes about a second.
RUN python -m rag_assistant.rag index

COPY survey_hazard_map/ ./survey_hazard_map/
COPY ghosttrace/ ./ghosttrace/
COPY data/ ./data/
COPY vendor/ ./vendor/

# The checkpoints are 27 MB and only the full profile can use them. Taken from
# the filtering stage above so ONNX exports never reach this image.
COPY --from=checkpoints /models/ ./survey_hazard_map/models/

EXPOSE 8000

# Render and most hosts inject PORT. Falling back to 8000 keeps `docker run`
# working with no environment at all.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
