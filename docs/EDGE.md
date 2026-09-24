# DeepEcho on the edge

SIH26057 asks that the solution be "optimized to run efficiently, potentially
allowing deployment on edge devices or onboard a marine drone without requiring
heavy cloud computing dependencies." This page is what that means for DeepEcho
in practice. Every figure on it was measured, and each one says what it was
measured on. Anything that was not measured says so.

## What needs a network, and what does not

The survey pipeline is local end to end. The assistant is the only part that
calls out, and the survey pipeline does not use it.

| Stage | Module | Runs offline | Needs |
|---|---|---|---|
| Tiling a sonar strip, manifest | `survey_preparation.py` | yes | numpy, pillow |
| Detection (both YOLOv8 models) | `hazard_detect_onnx.py` | yes | onnxruntime, numpy, pillow |
| Verification cues (nadir, shadow, clutter) | `hazard_verify.py` | yes | numpy, opencv-headless, scipy |
| Georeferencing from corners or a nav CSV | `hazard_geo.py`, `hazard_coords.py` | yes | numpy |
| Cross-model merge, deduplication | `hazard_dedup.py` | yes | stdlib |
| Severity, hotspots, recommended action | `hazard_severity.py`, `hazard_hotspots.py` | yes | stdlib |
| `export.json`, `actions.csv` | `hazard_export.py` | yes | stdlib |
| `map.html` | `hazard_mapview.py` | yes, with `vendor/` committed | folium |
| Grounded assistant (`/chat`) | `rag.py`, `backend/` | **no** | a Groq or Gemini API key |

The assistant is useful to an analyst ashore. A vehicle does not need it to
detect, verify, place, rank and report a contact.

## The detector without torch

The YOLOv8 checkpoints are exported to ONNX once, on any machine with
ultralytics, and after that they run under onnxruntime with numpy and pillow.

```bash
# once, where ultralytics is installed
.venv/bin/python survey_hazard_map/tools/export_onnx.py            # models/known.onnx, models/anomaly.onnx + parity check
.venv/bin/python survey_hazard_map/tools/export_onnx.py --int8 --fp16 --dynamic   # the other variants too

# on the edge machine
pip install -r requirements-edge.txt            # onnxruntime, numpy, pillow; no torch
python survey_hazard_map/tests/tests_edge.py                            # checks that need ultralytics print SKIP
python survey_hazard_map/tools/benchmark_edge.py                  # measure this machine
```

`hazard_detect_onnx.OnnxDetector` has the same interface and output as
`hazard_detect.UltralyticsDetector`, and `make_detector(paths)` picks one by
file extension (a mix of `.pt` and `.onnx` also works, in the order given).
In a virtualenv with only `requirements-edge.txt` and numpy/pillow installed,
`build_hazard_map(..., detector=make_detector([known.onnx, anomaly.onnx]))` ran
the whole engine over `survey_hazard_map/samples/tiles` (detect, merge, deduplicate, score,
export) and `sys.modules` held no `torch` or `ultralytics` afterwards.
`tests_edge.py` checks the same for the detector on its own.

The site-packages for that virtualenv (onnxruntime 1.30, numpy, pillow and
their dependencies) came to **145 MB** on macOS arm64. torch's package directory
alone is **583 MB** in the project's main venv.

For the API, `DEEPECHO_DETECTOR_BACKEND` chooses what `backend/detector_worker.py`
runs:

| Value | Behaviour |
|---|---|
| `auto` (default) | per model, the `.onnx` beside the `.pt` if it exists and onnxruntime imports, else the `.pt` |
| `onnx` | the `.onnx`; a missing one stops the worker with the file name |
| `torch` | the `.pt`, as before |

The worker's handshake reports `backend`, `backends` and `providers`, so what
ran is recorded rather than assumed. Protocol and box format are unchanged.

Execution providers are picked in the order TensorRT, CUDA, CoreML, CPU, from
whichever the installed onnxruntime offers. `HAZARD_ONNX_PROVIDERS` overrides
that (for example `CPUExecutionProvider`). `HAZARD_ONNX_THREADS` sets intra-op
threads; on a vehicle computer that also runs navigation and acquisition, this
is the setting that stops detection from starving them.

## Does the ONNX detect the same things? Measured

`tools/export_onnx.py` runs every image in `survey_hazard_map/samples/` and every tile under
`data/surveys/*/tiles/` through both backends: 80 images, of which 53 are
640x640 tiles and 27 are other shapes (edge tiles and the two whole strips).
Boxes are paired per image and per model, same class, by IoU, at thresholds of
0.10, 0.25 and 0.30. The ONNX side runs on `CPUExecutionProvider`. The full
output is in [`edge_parity.json`](edge_parity.json). onnxruntime 1.30.0,
ultralytics 8.4.146, Apple M1.

At a confidence threshold of 0.25:

| Variant | 640x640 tiles: boxes torch / onnx | max conf delta | min IoU | boxes that differ | Other shapes: max conf delta | Verdict |
|---|---|---|---|---|---|---|
| known fp32 | 9 / 9 | 0.0000 | 1.000 | 0 | 0.058 | parity |
| anomaly fp32 | 9 / 9 | 0.0000 | 1.000 | 0 | 0.120 | parity |
| known fp16 | 9 / 9 | 0.0018 | 0.996 | 0 | 0.058 | parity |
| anomaly fp16 | 9 / 9 | 0.0026 | 0.994 | 0 | 0.123 | parity |
| known dynamic fp32 | 9 / 9 | 0.0000 | 1.000 | 0 | 0.0005 | parity |
| anomaly dynamic fp32 | 9 / 9 | 0.0000 | 1.000 | 0 | 0.019 | parity |
| known int8 | 9 / 9, 2 of them different | 0.099 | 0.935 | 4 | 0.063 | **no parity** |
| anomaly int8 | 9 / 11 | 0.189 | 0.810 | 2 | 0.112 | **no parity** |

"0.0000" means identical at the four decimals both backends report, and IoU 1.000
means identical box coordinates at two decimals.

Three things in that table need reading correctly.

**fp32 is exact on 640x640 tiles.** Every box, every class, every confidence.

**Tiles that are not 640x640 differ, and the reason is not numerical.**
ultralytics' `predict()` sets `rect=True`, so torch pads a 640x359 edge tile
only to 640x384. A static ONNX accepts only 1x3x640x640, so the same tile
arrives with a much larger grey border and the convolutions near the edge see
different context. Confidences on those tiles moved by up to 0.12, and no box
appeared or disappeared at 0.25 or 0.30 on these images. The `--dynamic` export
takes the rect canvas like torch does and cuts the difference to 0.019. What is
left comes from resizing the whole-strip images, where OpenCV's fixed-point
bilinear and this numpy bilinear differ by at most one grey level (checked
against ultralytics' own `LetterBox` in `tests_edge.py`). Static is still the
default, because TensorRT builds its fastest engine for one fixed shape.

**int8 did not keep parity.** Dynamic quantisation moved confidences by up to
0.19 and added or dropped boxes at the operating threshold. Excluding the
detection head from quantisation, per-channel weights, and a static QDQ
calibration on the sample tiles were also tried on raw scores; none got the
largest score change below about 0.12. The int8 files are written so the trade
can be evaluated on real survey data, and `export_onnx.py` exits 3 to flag them.
Do not deploy int8 as if it matched fp32.

## How fast, measured

`tools/benchmark_edge.py`, both models per tile (the survey workload): the
full cost of each call, meaning JPEG decode, letterbox, known.onnx, anomaly.onnx
and NMS, on the ten real 640x640 tiles in `survey_hazard_map/samples/tiles`, 50 timed runs after
3 warm-up calls. Each variant runs in its own process. The full output is in
[`edge_benchmark.json`](edge_benchmark.json).

**Device: Apple M1 (4 performance + 4 efficiency cores), 8 GB, macOS 26.5,
Python 3.13.3, torch 2.14.0, onnxruntime 1.30.0.** Other jobs were running on
the machine: the 1-minute load average before each variant was 2.2 to 6.7 on 8
cores. Repeated runs moved individual latencies by roughly 20 %, so treat
differences smaller than that as noise.

| Variant | Model MB | Cold start s | First tile ms | p50 ms | p95 ms | Tiles/s | Peak RSS MB | Survey km/h | x real time |
|---|---|---|---|---|---|---|---|---|---|
| torch .pt, 1 thread | 28.8 | 1.43 | 1426 | 235 | 304 | 4.13 | 572 | 190 | 25.7 |
| torch .pt, 4 threads | 28.8 | 1.44 | 1428 | 251 | 433 | 3.64 | 602 | 168 | 22.6 |
| onnx fp32 CPU, 1 thread | 57.0 | 0.31 | 1234 | 770 | 787 | 1.30 | 299 | 60 | 8.1 |
| onnx fp32 CPU, 4 threads | 57.0 | 0.21 | 556 | 449 | 554 | 2.15 | 308 | 99 | 13.4 |
| onnx fp16 CPU, 4 threads | 28.6 | 0.21 | 568 | 449 | 786 | 2.05 | 277 | 95 | 12.8 |
| onnx int8 CPU, 4 threads | 14.8 | 0.24 | 322 | 230 | 308 | 3.98 | 258 | 183 | 24.8 |
| onnx fp32 CoreML | 57.0 | 4.99 | 41 | 37 | 44 | 26.57 | 382 | 1224 | 165 |

"torch, 1 thread" is how the survey engine runs torch today. ultralytics sets
`OMP_NUM_THREADS=1` on import unless it is already set.

Read honestly, the table says:

* **On this CPU, onnxruntime fp32 is slower than torch**, by about 2x at 4
  threads and 3x at 1. Changing onnxruntime's graph-optimisation level did not
  close the gap, which points at the convolution kernels on this platform. That
  was not investigated further. This was measured on an M1 only. It says
  nothing either way about a Cortex-A76 or a Jetson's CPU, where the comparison
  has not been run.
* **The measured edge gains are footprint and startup.** Resident memory is
  about half (572 to 299 MB), cold start drops from 1.4 s to 0.2 to 0.3 s, the
  install is 145 MB instead of torch's 583 MB plus ultralytics, and the image
  carries no torch.
* **An accelerator changes the picture.** The same fp32 file on CoreML ran at
  26.6 tiles/s, 6x torch on CPU. On a spot check of one tile, its raw class
  scores were within 0.0004 of the CPU provider's. It pays about 5 s at load to compile. CoreML is Apple-only, but the
  same file reaches TensorRT on a Jetson, which is the point of exporting to
  ONNX rather than to one vendor's format.
* **Every variant keeps up with the survey.** The slowest row, fp32 on one CPU
  thread, processes eight times as fast as the survey arrives.

### What "x real time" assumes

The survey figures use the benchmark's defaults, which can be changed with
`--swath`, `--resolution` and `--speed`:

* 100 m swath per side, 0.1 m per pixel across track, so 2,000 px across both
  channels, which makes 4 tiles across at 640 px with a 512 px stride;
* the waterfall resampled to square pixels, so 1000 / (512 x 0.1) = 19.5 rows
  of tiles per km, or **78.1 tiles per km of survey line**;
* a vessel at 4 knots covers 7.41 km of line per hour, which needs **0.16
  tiles/s**.

That assumption is generous to the workload. At 100 m range the two-way travel
time limits the ping rate to about 7.5 Hz, so at 4 knots pings are about 0.27 m
apart along track. Resampling to 0.1 m creates more tiles than the sonar
actually resolves. Detection is also not the only work per survey. Tiling,
verification, deduplication, scoring and export are not in these numbers.
They run over the detector's output, not once per network pass, but they have
not been benchmarked on edge hardware.

## Deploying

### Docker, ARM64 or x86 CPU

The `edge` target in the `Dockerfile` contains the survey engine and the ONNX
detector. It has no torch, no FastAPI, no corpus and no LLM client.

```bash
.venv/bin/python survey_hazard_map/tools/export_onnx.py            # the image needs models/known.onnx and anomaly.onnx
docker build --target edge -t deepecho-edge .
docker run --rm -v "$PWD/survey:/survey" deepecho-edge \
    --strips /survey/strips --model models/known.onnx models/anomaly.onnx --out /survey/out
```

The build fails at the `COPY models/known.onnx ...` line if the export has not
been run. That is deliberate. `docker build .` without `--target` still produces
the serve/full API image, and ONNX files are filtered out of that image.
Docker was not available on the machine these numbers came from, so the image
has not been built or measured here.

### Raspberry Pi 5 / other ARM64 CPU (not tested on hardware here)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-edge.txt -r requirements-survey.txt
export HAZARD_ONNX_PROVIDERS=CPUExecutionProvider
export HAZARD_ONNX_THREADS=3          # leave a core for acquisition and nav
.venv/bin/python survey_hazard_map/tools/benchmark_edge.py --threads 1 3
.venv/bin/python survey_hazard_map/tests/tests_edge.py
```

* Use fp32 first. int8 is about a quarter of the size and ran about 2x faster
  on the M1 CPU, but it did not keep parity (see above). Evaluate it on your own
  labelled tiles before trusting it.
* `anomaly` is a YOLOv8n (8.1 GFLOPs) and `known` a YOLOv8s. On the M1 CPU at 4
  threads, anomaly alone took about 87 ms per tile and known about 300 ms.
  On a very constrained CPU, running `anomaly` in real time and `known` on
  flagged tiles is a design option. It has not been built.
* Run the benchmark on the device. The M1 numbers above do not transfer.

### NVIDIA Jetson Orin (not tested on hardware here)

None of these steps have been run on a Jetson for this project. They follow
NVIDIA's and onnxruntime's documented paths and are written so they can be
checked.

**Option A: onnxruntime with the TensorRT execution provider.** No code changes
are needed. `OnnxDetector` puts TensorRT first when it is available.

```bash
# onnxruntime-gpu built for your JetPack (NVIDIA's Jetson wheels or an l4t-ml
# container); the plain `onnxruntime` wheel from requirements-edge.txt is CPU-only
python3 -c "import onnxruntime as o; print(o.get_available_providers())"   # expect Tensorrt, CUDA

export HAZARD_TRT_FP16=1                       # default; 0 for fp32 engines
export HAZARD_TRT_CACHE=/data/trt_cache        # default models/trt_cache
python survey_hazard_map/tools/benchmark_edge.py --threads 0    # first run builds engines: expect minutes
```

The TensorRT engine is built on first use and cached, so the first tile after a
fresh cache is slow and later starts are not. The detector's `provider`
attribute and the worker's handshake show whether TensorRT was actually
registered. onnxruntime quietly falls back to CUDA or CPU when it was not.

**Option B: a raw TensorRT engine with trtexec.**

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/known.onnx   --saveEngine=models/known.fp16.engine   --fp16
/usr/src/tensorrt/bin/trtexec --onnx=models/anomaly.onnx --saveEngine=models/anomaly.fp16.engine --fp16
# dynamic export: give it a profile
/usr/src/tensorrt/bin/trtexec --onnx=models/known.dynamic.onnx --fp16 \
    --minShapes=images:1x3x320x320 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640
```

trtexec prints its own latency. This repository has **no runner for a `.engine`
file**. Option A is the integrated path. Option B is for measuring the ceiling.

Whichever option you use, check fp16 accuracy on the device. On the M1 CPU, the
fp16 ONNX kept parity (max confidence delta 0.0026 on 640x640 tiles), but a
TensorRT fp16 engine is a different graph and has not been measured.

## Getting contacts off the vehicle

An AUV or USV does not need to send the survey home to be useful. It needs to
send the contacts. The pipeline's output is already structured, so a contact
report is a projection of one detection record:

```json
{"v":1,"id":"s7_0_0_d0","t":1789290692,"cls":"ship","c":0.82,"sev":"medium","lat":12.91234,"lon":74.85612,"wm":24.8,"hm":52.1}
```

That record is **126 bytes** as compact JSON: id, time, class, confidence,
severity tier, position to five decimal places (about 1 m) and size in metres.
For comparison, measured on `data/surveys/s7-submarine/export.json`, a full
detection record with its provenance and second opinion is 823 bytes, and that
survey's whole export is 7.9 KB.

What that buys on common links (vendor-typical figures, not measured here):

* **Iridium SBD** carries up to 340 bytes per mobile-originated message, so two
  contact reports fit in one message.
* **Acoustic modems** run from tens of bits per second at long range to a few
  kbit/s at short range. At 100 bit/s, a 126-byte report takes about 10 s before
  framing and error-correction overhead. The severity ranking decides which
  contacts go first, so the highest-tier contacts should be sent before the
  rest.
* Imagery chips, the full export and `map.html` stay on the vehicle for
  recovery or a high-bandwidth link.

The compact format above is a proposal sized from real records. The pipeline
does not emit it yet.

## Files, and what to commit

| File | Size | Committed? |
|---|---|---|
| `models/known.pt`, `models/anomaly.pt` | 22.5 + 6.3 MB | yes, as plain git objects (the repo has no `.gitattributes` and no LFS) |
| `models/known.onnx`, `models/anomaly.onnx` | 44.7 + 12.3 MB | recommended **not** to commit; see below |
| `models/*.int8.onnx`, `*.fp16.onnx`, `*.dynamic.onnx` | 3.4 to 44.9 MB each | no |
| `docs/edge_parity.json`, `docs/edge_benchmark.json` | small | yes: they are the evidence for this page |

The fp32 ONNX is twice the size of the `.pt` because ultralytics stores
checkpoints in half precision. Every ONNX is regenerated from its `.pt` in about
two seconds, and each export embeds its creation timestamp in its metadata, so
re-exporting produces a new binary every time. Committing them as plain git
objects would add about 57 MB of history per re-export. If a deployment really
cannot run the export step, commit only the two fp32 files through Git LFS.
Either way, `models/*.onnx` belongs in `.gitignore` or `.gitattributes`. This
change does not edit those files.

## Limits, stated once

* Parity was measured on 80 images from three surveys, two of which come from
  the same public waterfall record. That shows the export reproduces the model,
  not that the model is right. See `survey_hazard_map/samples/README.md` for what these
  detections actually are.
* All timings are from one Apple M1 with other work running. Nothing was
  measured on a Raspberry Pi, a Jetson, or in the Docker image.
* int8 is not equivalent to fp32 for these models.
