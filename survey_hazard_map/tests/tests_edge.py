#!/usr/bin/env python3
"""Checks for the torch-free edge detector. Plain asserts, no test framework.

    .venv/bin/python tests_edge.py

Exit 0 when everything passes, 1 on any failure. A check that cannot run on
this machine (no ultralytics to compare against, no .pt to export from) is
printed as SKIP with its reason, never silently counted as a pass.

What is checked, and why each one matters:

* the letterbox and its inverse, against hand-computed numbers and against
  ultralytics' own LetterBox when it is installed, because an off-by-one pad
  moves every box by a pixel;
* NMS on hand-made boxes, including the strict `>` at exactly the threshold
  and boxes of different classes that overlap, because that is where a numpy
  reimplementation usually drifts from torchvision;
* decoding a synthetic YOLOv8 head, so the xywh-to-corners step and the strict
  confidence threshold are pinned without needing a network;
* the Detector interface attributes the survey engine reads;
* OnnxDetector against UltralyticsDetector on the real sonar tiles in samples/,
  box for box;
* a missing class-name table fails loudly instead of numbering classes;
* make_detector dispatches by extension, including a mixed pair;
* the backend worker resolves backends as documented and returns identical
  records from torch and onnx on the same tile;
* importing and running the ONNX detector never imports torch.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from survey_hazard_map.hazard_detect import DetectorError  # noqa: E402
from survey_hazard_map.hazard_detect_onnx import (OnnxDetector, decode_yolov8, letterbox, load_rgb,  # noqa: E402
                                make_detector, nms, unletterbox)

MODELS = ROOT / "models"
SAMPLE_TILES = sorted((ROOT / "survey_hazard_map" / "samples" / "tiles").glob("*.jpg"))
WHOLE_STRIP = ROOT / "survey_hazard_map" / "samples" / "sidescan-s7-submarine.jpg"

results: list[tuple[str, str, str]] = []


class Skip(Exception):
    pass


def check(fn):
    try:
        fn()
        results.append(("PASS", fn.__name__, ""))
    except Skip as exc:
        results.append(("SKIP", fn.__name__, str(exc)))
    except Exception as exc:  # AssertionError included
        results.append(("FAIL", fn.__name__, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()
    return fn


def have_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


_tmp = Path(tempfile.mkdtemp(prefix="deepecho-edge-tests-"))


def onnx_models() -> list[Path]:
    """models/*.onnx if exported, else a throwaway export in a temp dir."""
    wanted = [MODELS / "known.onnx", MODELS / "anomaly.onnx"]
    if all(p.is_file() for p in wanted):
        return wanted
    if not have_ultralytics() or not all(p.with_suffix(".pt").is_file() for p in wanted):
        raise Skip("no .onnx in models/ and no ultralytics + .pt to export one from")
    from tools.export_onnx import export_fp32

    out = []
    for p in wanted:
        if not (_tmp / p.name).is_file():
            shutil.copy(p.with_suffix(".pt"), _tmp / p.with_suffix(".pt").name)
            export_fp32(_tmp / p.with_suffix(".pt").name, 640, 17)
        out.append(_tmp / p.name)
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --- Pure numpy units ---------------------------------------------------------

@check
def letterbox_square_tile_is_untouched():
    img = np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img)
    assert out.shape == (640, 640, 3) and gain == 1.0 and pad == (0, 0)
    assert np.array_equal(out, img)


@check
def letterbox_edge_tile_pads_like_ultralytics():
    img = np.full((359, 640, 3), 7, dtype=np.uint8)
    out, gain, pad = letterbox(img)
    # dh = 281 split as round(140.5 - 0.1) = 140 on top, round(140.5 + 0.1) = 141 below.
    assert out.shape == (640, 640, 3), out.shape
    assert gain == 1.0 and pad == (0, 140), (gain, pad)
    assert (out[:140] == 114).all() and (out[140:499] == 7).all() and (out[499:] == 114).all()
    rect, _, rect_pad = letterbox(img, auto=True)
    assert rect.shape == (384, 640, 3) and rect_pad == (0, 12), (rect.shape, rect_pad)


@check
def letterbox_round_trip_recovers_original_boxes():
    for h, w in [(871, 1676), (359, 640), (640, 198), (2758, 3600), (640, 640)]:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for auto in (False, True):
            _, gain, pad = letterbox(img, auto=auto)
            boxes = np.array([[0, 0, w, h], [w * 0.1, h * 0.2, w * 0.4, h * 0.9]], dtype=np.float64)
            forward = boxes.copy()
            forward[:, [0, 2]] = forward[:, [0, 2]] * gain + pad[0]
            forward[:, [1, 3]] = forward[:, [1, 3]] * gain + pad[1]
            back = unletterbox(forward, gain, pad, (h, w))
            assert np.allclose(back, boxes, atol=1e-6), (h, w, auto, back, boxes)


@check
def letterbox_matches_ultralytics_letterbox():
    if not have_ultralytics():
        raise Skip("ultralytics not installed")
    import cv2
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(1)
    for h, w in [(359, 640), (640, 198), (871, 1676), (300, 500)]:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for auto in (False, True):
            theirs = LetterBox((640, 640), auto=auto, stride=32)(image=img[..., ::-1].copy())[..., ::-1]
            ours, _, _ = letterbox(img, auto=auto)
            assert theirs.shape == ours.shape, (h, w, auto, theirs.shape, ours.shape)
            diff = np.abs(theirs.astype(int) - ours.astype(int)).max()
            # Identical when no resize happens; within one grey level when OpenCV's
            # fixed-point bilinear and this float bilinear both resample.
            assert diff <= 1, (h, w, auto, diff)
    del cv2


@check
def nms_suppresses_overlap_and_keeps_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10.5]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.95], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [3, 2], keep


@check
def nms_threshold_is_strict():
    # Two 10x10 boxes offset so IoU is exactly 0.5: 100 overlap... use widths.
    a = [0, 0, 10, 10]
    b = [0, 0, 10, 20]            # intersection 100, union 200, IoU exactly 0.5
    boxes = np.array([a, b], dtype=np.float64)
    assert abs(iou(a, b) - 0.5) < 1e-12
    assert nms(boxes, np.array([0.9, 0.8]), 0.5).tolist() == [0, 1]   # not > 0.5: both kept
    assert nms(boxes, np.array([0.9, 0.8]), 0.49).tolist() == [0]     # > 0.49: suppressed
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).tolist() == []


def _head(rows: list[tuple[float, float, float, float, list[float]]], nc: int) -> np.ndarray:
    """A (1, 4+nc, N) tensor from (cx, cy, w, h, class scores) rows."""
    arr = np.array([[cx, cy, w, h, *scores] for cx, cy, w, h, scores in rows], dtype=np.float32)
    assert arr.shape[1] == 4 + nc
    return arr.T[None]


@check
def decode_is_class_wise_and_strict():
    out = _head([
        (50, 50, 20, 20, [0.90, 0.10]),   # class 0
        (51, 51, 20, 20, [0.05, 0.80]),   # overlaps the first, different class: kept
        (52, 50, 20, 20, [0.70, 0.00]),   # overlaps the first, same class: suppressed
        (200, 200, 10, 10, [0.25, 0.00]), # exactly at conf: dropped (strict >)
        (300, 300, 10, 10, [0.00, 0.2501]),
    ], nc=2)
    rows = decode_yolov8(out, nc=2, conf=0.25, iou=0.7)
    got = [(round(float(r[4]), 4), int(r[5])) for r in rows]
    assert got == [(0.9, 0), (0.8, 1), (0.2501, 1)], got
    assert np.allclose(rows[0, :4], [40, 40, 60, 60]), rows[0]
    # The transposed layout decodes the same.
    assert np.allclose(decode_yolov8(out.transpose(0, 2, 1), nc=2, conf=0.25), rows)
    # max_det caps the result.
    assert len(decode_yolov8(out, nc=2, conf=0.0, max_det=1)) == 1


@check
def load_rgb_replicates_greyscale_and_reduces_16_bit():
    from PIL import Image

    grey = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16), mode="L")
    rgb = load_rgb(grey)
    assert rgb.shape == (16, 16, 3) and (rgb[..., 0] == rgb[..., 2]).all()
    deep = Image.fromarray((np.arange(256, dtype=np.uint16) * 257).reshape(16, 16))
    reduced = load_rgb(deep)
    assert reduced.dtype == np.uint8 and reduced[..., 0].max() == 255 and reduced[0, 1, 0] == 1


# --- Against the real models --------------------------------------------------

@check
def interface_attributes_present():
    paths = onnx_models()
    det = OnnxDetector(paths, conf=0.3, providers=["CPUExecutionProvider"])
    assert det.paths == paths and det.path == paths[0]
    assert det.conf == 0.3 and det.imgsz == 640
    assert det.name == "known.onnx + anomaly.onnx", det.name
    assert {"aircraft", "human", "ship", "fish", "other", "shipwreck"} <= set(det.classes), det.classes
    assert det.provider == "CPUExecutionProvider" and set(det.providers) == {"known", "anomaly"}
    boxes = det(SAMPLE_TILES[0])
    assert isinstance(boxes, list)
    for box in det(ROOT / "survey_hazard_map" / "samples" / "tiles" / "sidescan-waterfall-strip_2560_1024.jpg"):
        assert set(box) == {"class", "confidence", "bbox", "model"}, box
        assert box["model"] in {"known", "anomaly"} and len(box["bbox"]) == 4
    try:
        OnnxDetector(paths, imgsz=1024, providers=["CPUExecutionProvider"])
    except DetectorError as exc:
        assert "exported at 640x640" in str(exc)
    else:
        raise AssertionError("a static 640 export accepted imgsz=1024")


@check
def onnx_matches_ultralytics_on_sample_tiles():
    if not have_ultralytics():
        raise Skip("ultralytics not installed; nothing to compare against")
    from survey_hazard_map.hazard_detect import UltralyticsDetector

    paths = onnx_models()
    pts = [p.with_suffix(".pt") if p.parent == MODELS else MODELS / p.with_suffix(".pt").name
           for p in paths]
    torch_det = UltralyticsDetector(pts, conf=0.05)
    onnx_det = OnnxDetector(paths, conf=0.05, providers=["CPUExecutionProvider"])
    compared = 0
    for tile in SAMPLE_TILES:
        a = sorted(torch_det(tile), key=lambda b: (b["model"], -b["confidence"]))
        b = sorted(onnx_det(tile), key=lambda b: (b["model"], -b["confidence"]))
        assert len(a) == len(b), (tile.name, a, b)
        for x, y in zip(a, b):
            assert x["model"] == y["model"] and x["class"] == y["class"], (tile.name, x, y)
            assert abs(x["confidence"] - y["confidence"]) <= 1e-3, (tile.name, x, y)
            assert iou(x["bbox"], y["bbox"]) >= 0.99, (tile.name, x, y)
            compared += 1
    assert compared >= 5, f"only {compared} boxes compared; the check proves too little"

    dynamic = [MODELS / "known.dynamic.onnx", MODELS / "anomaly.dynamic.onnx"]
    if all(p.is_file() for p in dynamic):
        dyn = OnnxDetector(dynamic, conf=0.25, providers=["CPUExecutionProvider"])
        a = {(b["model"], b["class"]): b for b in torch_det(WHOLE_STRIP) if b["confidence"] > 0.25}
        b = {(b["model"].replace(".dynamic", ""), b["class"]): b for b in dyn(WHOLE_STRIP)}
        assert a.keys() == b.keys(), (a, b)
        for key in a:
            assert abs(a[key]["confidence"] - b[key]["confidence"]) <= 0.03, (a[key], b[key])
            assert iou(a[key]["bbox"], b[key]["bbox"]) >= 0.95, (a[key], b[key])


@check
def missing_names_metadata_fails_loudly():
    try:
        import onnx
    except ImportError:
        raise Skip("the onnx package is needed to write a test file without names; "
                   "it is not an inference dependency")

    src = onnx_models()[1]
    model = onnx.load(str(src))
    kept = [p for p in model.metadata_props if p.key != "names"]
    del model.metadata_props[:]
    for prop in kept:
        model.metadata_props.add(key=prop.key, value=prop.value)
    bare = _tmp / "nameless.onnx"
    onnx.save(model, str(bare))
    try:
        OnnxDetector(bare, providers=["CPUExecutionProvider"])
    except DetectorError as exc:
        assert "names" in str(exc) and "nameless.onnx" in str(exc), exc
    else:
        raise AssertionError("an ONNX without class names loaded")


@check
def make_detector_dispatches_by_extension():
    paths = onnx_models()
    det = make_detector(paths)
    assert isinstance(det, OnnxDetector), type(det)
    if not have_ultralytics():
        raise Skip("ultralytics not installed; .pt dispatch not checked")
    from survey_hazard_map.hazard_detect import UltralyticsDetector

    assert isinstance(make_detector(MODELS / "known.pt"), UltralyticsDetector)
    mixed = make_detector([MODELS / "known.pt", paths[1]], conf=0.25)
    assert mixed.backend == "mixed" and len(mixed.parts) == 2
    assert [p.name for p in mixed.paths] == ["known.pt", paths[1].name]
    boxes = mixed(ROOT / "survey_hazard_map" / "samples" / "tiles" / "sidescan-waterfall-strip_2048_1024.jpg")
    assert {b["model"] for b in boxes} <= {"known", "anomaly"} and boxes


@check
def worker_resolves_backends():
    from survey_hazard_map import detector_worker as w

    d = _tmp / "resolve"
    d.mkdir(exist_ok=True)
    pt, ox = d / "m.pt", d / "m.onnx"
    pt.write_bytes(b"")
    assert w.resolve(pt, "torch") == ("torch", pt)
    assert w.resolve(pt, "auto") == ("torch", pt)          # no .onnx yet
    try:
        w.resolve(pt, "onnx")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("onnx backend accepted a missing .onnx")
    ox.write_bytes(b"")
    assert w.resolve(pt, "auto") == ("onnx", ox)
    assert w.resolve(pt, "onnx") == ("onnx", ox)
    assert w.resolve(ox, "auto") == ("onnx", ox)


@check
def worker_handshake_and_records_match_across_backends():
    if not have_ultralytics():
        raise Skip("ultralytics not installed")
    if not all((MODELS / f"{n}.onnx").is_file() for n in ("known", "anomaly")):
        raise Skip("models/*.onnx not exported; run tools/export_onnx.py")
    tile = ROOT / "survey_hazard_map" / "samples" / "tiles" / "sidescan-waterfall-strip_2560_1024.jpg"
    request = json.dumps({"image_path": str(tile)}) + "\n"
    replies = {}
    for backend in ("torch", "onnx"):
        proc = subprocess.run([sys.executable, "-m", "survey_hazard_map.detector_worker"], input=request,
                              capture_output=True, text=True, cwd=str(ROOT), timeout=300,
                              env={**__import__("os").environ, "DEEPECHO_DETECTOR_BACKEND": backend,
                                   "HAZARD_ONNX_PROVIDERS": "CPUExecutionProvider"})
        lines = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        assert len(lines) == 2, (backend, proc.stdout, proc.stderr[-800:])
        hello, reply = lines
        assert hello["ready"] and hello["backend"] == backend, hello
        assert set(hello["backends"].values()) == {backend}
        assert set(hello["models"]) == {"known", "anomaly"}
        replies[backend] = (hello, reply)
    (th, tr), (oh, orr) = replies["torch"], replies["onnx"]
    assert th["classes"] == oh["classes"], (th["classes"], oh["classes"])
    assert oh["providers"]["known"][0] == "CPUExecutionProvider", oh["providers"]
    assert tr["ok"] and orr["ok"] and tr["boxes"], (tr, orr)
    key = lambda b: (b["model"], b["cls"], -b["confidence"])
    for a, b in zip(sorted(tr["boxes"], key=key), sorted(orr["boxes"], key=key)):
        assert set(a) == set(b) == {"model", "cls", "confidence", "bbox"}
        assert a["model"] == b["model"] and a["cls"] == b["cls"], (a, b)
        assert abs(a["confidence"] - b["confidence"]) <= 1e-3, (a, b)
        assert all(abs(x - y) <= 0.2 for x, y in zip(a["bbox"], b["bbox"])), (a, b)
    assert len(tr["boxes"]) == len(orr["boxes"])


@check
def onnx_path_never_imports_torch():
    paths = onnx_models()
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from hazard_detect_onnx import OnnxDetector, make_detector\n"
        "d = make_detector(%r)\n"
        "d(%r)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in {'torch', 'ultralytics', 'torchvision'})\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    ) % (str(ROOT), [str(p) for p in paths], str(SAMPLE_TILES[0]))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                          env={**__import__("os").environ, "HAZARD_ONNX_PROVIDERS": "CPUExecutionProvider"})
    assert proc.returncode == 0, (proc.stdout, proc.stderr[-800:])


def main() -> int:
    shutil.rmtree(_tmp, ignore_errors=True)
    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        print(f"{status}  {name:<{width}}  {detail}")
    failed = sum(1 for s, _, _ in results if s == "FAIL")
    skipped = sum(1 for s, _, _ in results if s == "SKIP")
    print(f"\n{len(results) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
