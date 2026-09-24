"""The detector, in its own process.

Run as `python -m survey_hazard_map.detector_worker`. Reads one JSON request per line on
stdin, writes one JSON response per line on stdout.

This exists for an unglamorous reason. faiss and torch each bundle their own
copy of libomp, and on macOS the second one to initialise aborts the process
with OMP Error #15. The vendor's own workaround, KMP_DUPLICATE_LIB_OK, is
documented as unsafe and as possibly producing silently incorrect results,
which is not a trade a safety system should make. Keeping the two libraries in
separate processes removes the conflict instead of suppressing it, and costs
one subprocess that loads its weights once and stays up.

The worker is deliberately dumb. It returns raw boxes with the class names the
checkpoints were trained with. Class mapping, merging and severity stay in the
main process, where the configuration lives.

TWO BACKENDS, ONE PROTOCOL
    The same checkpoints can run under torch (ultralytics, the .pt files) or
    under onnxruntime (the .onnx exports written by tools/export_onnx.py, run
    by hazard_detect_onnx.OnnxDetector with no torch at all). Chosen by
    DEEPECHO_DETECTOR_BACKEND:

        torch   the .pt checkpoints, as before
        onnx    the .onnx beside each .pt (or the configured path itself, if it
                already ends in .onnx); a model with no .onnx is an error
        auto    per model: the .onnx when one exists beside the .pt and
                onnxruntime imports, otherwise the .pt. The default, so a
                machine that has run the export picks up the lighter path and
                one that has not keeps working exactly as it did.

    Requests, responses and box fields are identical on both. The handshake
    gains `backend` (torch, onnx, or mixed), `backends` (per model) and
    `providers` (per model: the onnxruntime execution providers registered in
    priority order, or "torch-cpu"), so /health and a log line can say what
    actually ran rather than what was configured.

    The ONNX backend does not need this process boundary for the libomp reason:
    onnxruntime's default builds use their own thread pool and do not link
    libomp (checked on the macOS arm64 wheel with otool). The worker is kept
    anyway, because one protocol is simpler than two, and because a crash in a
    native inference library then takes down a subprocess the API respawns,
    not the API.

    On a 640x640 tile the two backends return identical boxes on every tile in
    this repository; on tiles of other shapes the static ONNX differs by up to
    about 0.12 in confidence. docs/EDGE.md has the measurement and the reason.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from backend import config

BACKENDS = ("auto", "onnx", "torch")


def requested_backend() -> str:
    value = os.environ.get("DEEPECHO_DETECTOR_BACKEND", "auto").strip().lower() or "auto"
    if value not in BACKENDS:
        raise ValueError(f"DEEPECHO_DETECTOR_BACKEND={value!r}; expected one of {', '.join(BACKENDS)}")
    return value


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def resolve(path: Path, backend: str) -> tuple[str, Path]:
    """(backend, file) for one configured model path."""
    if path.suffix.lower() == ".onnx":
        if backend == "torch":
            raise ValueError(f"{path.name} is an ONNX file but DEEPECHO_DETECTOR_BACKEND=torch")
        return "onnx", path
    onnx_path = path.with_suffix(".onnx")
    if backend == "onnx":
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"DEEPECHO_DETECTOR_BACKEND=onnx but {onnx_path.name} does not exist; "
                f"run tools/export_onnx.py")
        return "onnx", onnx_path
    if backend == "auto" and onnx_path.is_file() and _onnxruntime_available():
        return "onnx", onnx_path
    return "torch", path


class _TorchModel:
    backend = "torch"

    def __init__(self, path: Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.classes = list(self.model.model.names.values())
        self.providers = ["torch-cpu"]

    def boxes(self, image) -> list[tuple[str, float, list[float]]]:
        out = []
        for result in self.model.predict(source=image, conf=config.DETECTOR_CONFIDENCE,
                                         imgsz=config.DETECTOR_IMGSZ, verbose=False):
            for box in result.boxes:
                out.append((result.names[int(box.cls)], float(box.conf),
                            [float(v) for v in box.xyxy[0]]))
        return out


class _OnnxModel:
    backend = "onnx"

    def __init__(self, path: Path) -> None:
        sys.path.insert(0, str(config.ROOT))
        from survey_hazard_map.hazard_detect_onnx import OnnxDetector

        self.detector = OnnxDetector(path, conf=config.DETECTOR_CONFIDENCE,
                                     imgsz=config.DETECTOR_IMGSZ)
        self.classes = list(self.detector.classes)
        self.providers = next(iter(self.detector.providers.values()))

    def boxes(self, image) -> list[tuple[str, float, list[float]]]:
        import numpy as np

        return [(cls, conf, xyxy) for _, cls, conf, xyxy
                in self.detector.detect_raw(np.asarray(image, dtype=np.uint8))]


def load_models() -> dict:
    backend = requested_backend()
    models = {}
    for name, path in config.DETECTOR_MODELS.items():
        if not path.exists() and not (backend != "torch" and path.with_suffix(".onnx").is_file()):
            continue
        kind, file = resolve(path, backend)
        models[name] = _OnnxModel(file) if kind == "onnx" else _TorchModel(file)
    return models


def predict(models: dict, image_bytes: bytes) -> list[dict]:
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    boxes = []
    for name, model in models.items():
        for cls, conf, (x1, y1, x2, y2) in model.boxes(image):
            boxes.append({
                "model": name,
                "cls": cls,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
            })
    return boxes


def handshake(models: dict) -> dict:
    kinds = {name: model.backend for name, model in models.items()}
    distinct = set(kinds.values())
    return {
        "ready": True,
        "models": sorted(models),
        "classes": {name: model.classes for name, model in models.items()},
        "backend": distinct.pop() if len(distinct) == 1 else ("mixed" if distinct else "none"),
        "backends": kinds,
        "providers": {name: model.providers for name, model in models.items()},
    }


def main() -> None:
    try:
        models = load_models()
    except Exception as exc:
        print(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return

    print(json.dumps(handshake(models)), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            with open(request["image_path"], "rb") as handle:
                boxes = predict(models, handle.read())
            print(json.dumps({"ok": True, "boxes": boxes}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
