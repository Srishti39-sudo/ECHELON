"""The detector without torch: the same YOLOv8 checkpoints, run by onnxruntime.

WHY THIS EXISTS
    SIH26057 asks for a solution that can run "on edge devices or onboard a
    marine drone without requiring heavy cloud computing dependencies". The
    torch path cannot honestly claim that. ultralytics pulls torch, which is
    several hundred megabytes on disk and a large resident footprint before a
    single tile is read, and on a Jetson or a Raspberry Pi installing a
    matching torch wheel is a project in itself.

    The trained weights are not the heavy part; the framework is. Exported to
    ONNX (tools/export_onnx.py), the same two checkpoints run under onnxruntime
    with numpy and pillow and nothing else. This file is the whole of the
    inference path for that case, and it imports no torch, directly or through
    ultralytics, so it runs inside the `edge` Docker profile and on any machine
    with `pip install -r requirements-edge.txt`.

WHAT "THE SAME" MEANS
    OnnxDetector is a drop-in for hazard_detect.UltralyticsDetector: same
    constructor shape, same `classes`, `name`, `conf`, `paths` attributes, and
    `__call__(image_path)` returns the same list of

        {"class": str, "confidence": float, "bbox": [x1, y1, x2, y2], "model": stem}

    in the original tile's pixel coordinates. To get the same boxes, not merely
    boxes of the same shape, every step ultralytics 8.4 performs around the
    network is reproduced here rather than approximated:

    * decode to 3-channel 8-bit (a greyscale sonar tile is replicated into
      R=G=B, which is what cv2.IMREAD_COLOR does on the torch path);
    * letterbox with ultralytics' own arithmetic: scale by min(640/h, 640/w),
      round the new size, bilinear resize with OpenCV's half-pixel convention,
      split the padding with its round(d -/+ 0.1) rule, pad with grey 114;
    * RGB, CHW, float32 / 255;
    * decode (1, 4+nc, N): centre-xywh to corners, best class per anchor,
      strict `> conf`;
    * class-wise greedy NMS at IoU 0.7 (ultralytics' predict default) with its
      max_wh class-offset trick, keep at most 300, suppress on IoU strictly
      greater than the threshold as torchvision does;
    * undo the padding and scale, clip to the tile.

    Measured by tools/export_onnx.py on all 80 real images in this repository
    (docs/edge_parity.json): on every 640x640 tile, fp32 ONNX on the CPU
    provider returns the same boxes as torch, with confidence identical to the
    4 decimals both round to and IoU 1.0 at 2-decimal box coordinates.

WHERE IT DIFFERS, AND WHY
    * Tiles that are not 640x640. This is the one material difference, and it
      is structural rather than numeric. ultralytics' predict() sets rect=True,
      so torch pads a 640x359 edge tile only to 640x384 (the next multiple of
      the stride). A static ONNX accepts exactly 1x3x640x640, so the same tile
      arrives with 256 more rows of grey border, and the convolutions near the
      image edge see different context. Measured on the 27 non-square images:
      confidence moves by up to 0.058 (known) and 0.120 (anomaly), with no box
      gained or lost at the 0.25 and 0.30 thresholds. A dynamic-shape export
      (`tools/export_onnx.py --dynamic`, <stem>.dynamic.onnx) lets this path
      pad exactly as torch does, and brings those deltas to 0.0015 and 0.019;
      it costs a TensorRT optimisation profile, which is why static stays the
      default export.
    * Resizing, only for inputs larger than 640 (the whole-strip samples).
      OpenCV's INTER_LINEAR uses 11-bit fixed-point weights; this uses float
      and rounds, and was measured to differ by at most one grey level. That
      residual is most of the 0.019 above.
    * JPEG decoding. The torch path decodes with OpenCV, this one with Pillow.
      On every image here the decoded pixels were identical, but that depends
      on both linking compatible libjpeg builds and is not guaranteed.
    * Execution providers other than CPU (CoreML, CUDA, TensorRT) may run parts
      of the graph in fp16. On CoreML on an M1 the raw class scores moved by at
      most 0.0004. The provider that was registered is recorded on the detector
      (`provider`, `providers`) and should be reported alongside any result.
    A box whose confidence sits right at the threshold can therefore appear on
    one backend and not the other. That is not a bug in either; it is what a
    hard threshold does to a continuous score.

PROVIDERS AND THREADS
    Providers are chosen in the order TensorRT, CUDA, CoreML, CPU, keeping only
    those this onnxruntime build actually offers, so the same code picks up a
    Jetson's GPU without a flag. HAZARD_ONNX_PROVIDERS (comma-separated names,
    e.g. "CPUExecutionProvider") overrides that, which is how a benchmark pins
    the CPU. HAZARD_ONNX_THREADS sets intra-op threads; unset or 0 leaves the
    onnxruntime default (one per physical core). On a vehicle whose CPU is also
    running navigation and sonar acquisition, capping this is the knob that
    keeps detection from starving them.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_detect import DetectorError, UltralyticsDetector

log = logging.getLogger("deepecho.hazard")

# ultralytics predict defaults, 8.4. Kept as named constants so a reader can
# check them against ultralytics/cfg/default.yaml rather than trust them.
NMS_IOU = 0.7
MAX_DET = 300
MAX_NMS = 30000
MAX_WH = 7680
PAD_VALUE = 114

# Preference order when HAZARD_ONNX_PROVIDERS is not set. Only providers the
# installed onnxruntime build reports as available are kept.
PROVIDER_PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


# --- Preprocessing -----------------------------------------------------------

def load_rgb(image: Any) -> np.ndarray:
    """An (H, W, 3) uint8 RGB array from a path, bytes, a PIL image or an array.

    Greyscale becomes three identical channels. 16-bit greyscale (common for raw
    sonar exports saved as PNG or TIFF) is reduced to 8 bits by dropping the low
    byte, which is what OpenCV's IMREAD_COLOR does on the torch path; Pillow's
    own convert("RGB") would clip it to white instead. EXIF orientation is
    applied because OpenCV applies it too.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise DetectorError(f"expected an HxW or HxWx3 array, got {arr.shape}")
        return np.ascontiguousarray(arr[..., :3].astype(np.uint8, copy=False))

    from PIL import Image, ImageOps

    if isinstance(image, (bytes, bytearray)):
        import io
        pil = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise DetectorError(f"cannot read an image from {type(image).__name__}")

    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:  # pragma: no cover - malformed EXIF is not worth failing a tile over
        pass

    if pil.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        grey = np.asarray(pil, dtype=np.int64)
        grey = np.clip(grey >> 8, 0, 255).astype(np.uint8)
        return np.repeat(grey[..., None], 3, axis=2)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return np.asarray(pil, dtype=np.uint8)


def resize_bilinear(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """OpenCV INTER_LINEAR in numpy: half-pixel centres, edge clamp, no antialias.

    Not Pillow's resize, which antialiases when shrinking and would therefore
    feed the network a visibly softer image than ultralytics does.
    """
    h, w = img.shape[:2]
    if (w, h) == (new_w, new_h):
        return img

    def axis(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
        src = np.clip(src, 0, n_in - 1)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, n_in - 1)
        return lo, hi, (src - lo).astype(np.float32)

    y0, y1, fy = axis(new_h, h)
    x0, x1, fx = axis(new_w, w)
    src = img.astype(np.float32)
    top = src[y0][:, x0] * (1 - fx)[None, :, None] + src[y0][:, x1] * fx[None, :, None]
    bot = src[y1][:, x0] * (1 - fx)[None, :, None] + src[y1][:, x1] * fx[None, :, None]
    out = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640), auto: bool = False,
              stride: int = 32) -> tuple[np.ndarray, float, tuple[int, int]]:
    """(padded image, gain, (pad_left, pad_top)) with ultralytics' arithmetic.

    Mirrors ultralytics.data.augment.LetterBox with scaleup=True, center=True.
    The rounding rules are copied exactly because an off-by-one pad shifts
    every box by a pixel.

    auto=False pads to the full square, which is all a static export accepts.
    auto=True pads only up to the next multiple of `stride`, which is what
    ultralytics' torch predict does (it sets rect=True for predict), and what a
    dynamic-shape export lets this path do too. For a 640x640 tile the two are
    identical; for a 640x359 edge tile they are not, and the network sees a
    different amount of grey border. See the module docstring.
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h, new_w / w)
    unpad_w, unpad_h = round(w * r), round(h * r)
    dw, dh = new_w - unpad_w, new_h - unpad_h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    img = resize_bilinear(img, unpad_w, unpad_h)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = np.full((unpad_h + top + bottom, unpad_w + left + right, 3), PAD_VALUE, dtype=np.uint8)
    out[top:top + unpad_h, left:left + unpad_w] = img
    return out, r, (left, top)


def unletterbox(boxes: np.ndarray, gain: float, pad: tuple[int, int],
                orig_hw: tuple[int, int], input_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Letterboxed xyxy back to original pixels, clipped.

    ultralytics' scale_boxes recomputes the padding from the two shapes rather
    than reusing the letterbox's own; the two agree for every shape, and this
    uses the letterbox's values so the round trip is exact by construction.
    """
    out = boxes.astype(np.float64, copy=True)
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out[:, :4] /= gain
    h, w = orig_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


# --- Postprocessing ----------------------------------------------------------

def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box with many, areas without the +1 torchvision omits too."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS: indices kept, highest score first.

    A box is suppressed when its IoU with a kept box is strictly greater than
    the threshold, which is torchvision.ops.nms's rule. Stable sort, so equal
    scores keep their anchor order.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for pos, idx in enumerate(order):
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        rest = order[pos + 1:]
        rest = rest[~suppressed[rest]]
        if len(rest):
            suppressed[rest[box_iou_one_to_many(boxes[idx], boxes[rest]) > iou_thres]] = True
    return np.asarray(keep, dtype=np.int64)


def decode_yolov8(output: np.ndarray, nc: int, conf: float, iou: float = NMS_IOU,
                  max_det: int = MAX_DET, end2end: bool = False) -> np.ndarray:
    """(K, 6) rows of [x1, y1, x2, y2, score, class] in letterbox pixels.

    Accepts the raw YOLOv8 head (1, 4+nc, N) or the same transposed
    (1, N, 4+nc). With end2end=True, an export that already ran its own NMS
    and emits (1, K, 6). That is taken from the file's metadata, never guessed
    from the shape: with two classes a transposed raw head is also (1, N, 6).
    """
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        raise DetectorError(f"unexpected detector output shape {output.shape}")

    if end2end:
        if pred.shape[-1] != 6:
            raise DetectorError(f"end-to-end output should be (K, 6), got {output.shape}")
        rows = pred[pred[:, 4] > conf][:max_det]
        return rows.astype(np.float32)

    if pred.shape[0] == 4 + nc:
        pred = pred.T
    elif pred.shape[1] != 4 + nc:
        raise DetectorError(
            f"detector output {output.shape} does not match {nc} classes "
            f"(expected 4+{nc}={4 + nc} on one axis)")

    cls_scores = pred[:, 4:4 + nc]
    best = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(pred)), best]
    mask = score > conf
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    xywh = pred[mask, :4]
    score, best = score[mask], best[mask]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    if len(score) > MAX_NMS:
        top = np.argsort(-score, kind="stable")[:MAX_NMS]
        xyxy, score, best = xyxy[top], score[top], best[top]

    # Class-wise NMS in one pass: shifting each class into its own far-apart
    # region of the plane means boxes of different classes can never overlap.
    offset = best[:, None].astype(xyxy.dtype) * MAX_WH
    keep = nms(xyxy + offset, score, iou)[:max_det]
    return np.concatenate([xyxy[keep], score[keep, None], best[keep, None].astype(xyxy.dtype)],
                          axis=1).astype(np.float32)


# --- Sessions ----------------------------------------------------------------

def _env_threads() -> int:
    raw = os.environ.get("HAZARD_ONNX_THREADS", "").strip()
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        log.warning("HAZARD_ONNX_THREADS=%r is not an integer; using the default", raw)
        return 0


def choose_providers(requested: Iterable[str] | None = None) -> list[str]:
    """Execution providers to register, most capable first, CPU always last."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if requested is None:
        env = os.environ.get("HAZARD_ONNX_PROVIDERS", "").strip()
        requested = [p.strip() for p in env.split(",") if p.strip()] if env else None
    if requested:
        unknown = [p for p in requested if p not in available]
        if unknown:
            raise DetectorError(
                f"onnxruntime provider(s) not available in this build: {', '.join(unknown)}. "
                f"Available: {', '.join(available)}")
        chosen = list(requested)
    else:
        chosen = [p for p in PROVIDER_PREFERENCE if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _provider_options(name: str, model_path: Path) -> dict:
    """Per-provider options. Only TensorRT needs any to be usable on a vehicle.

    Without an engine cache TensorRT rebuilds its engine on every start, which
    on a Jetson Orin takes minutes; with one it is paid once. FP16 is on
    because it is the reason to use TensorRT on Orin at all. Neither has been
    exercised on Jetson hardware in this repository; see docs/EDGE.md.
    """
    if name == "TensorrtExecutionProvider":
        cache = os.environ.get("HAZARD_TRT_CACHE", str(model_path.parent / "trt_cache"))
        return {
            "trt_fp16_enable": os.environ.get("HAZARD_TRT_FP16", "1") not in {"0", "false", "no"},
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
        }
    return {}


def parse_names(metadata: dict[str, str], path: Path) -> list[str]:
    """Class names, in index order, from ultralytics' ONNX metadata.

    ultralytics writes them as the repr of a dict, "{0: 'aircraft', ...}". A
    missing or unreadable entry is an error, not a fallback to numbered
    classes: the severity lookup keys on these names, and "class_2" would be
    silently scored as an unknown instead of as `other`.
    """
    raw = metadata.get("names")
    if not raw:
        raise DetectorError(
            f"{path.name} carries no 'names' metadata, so its class indices cannot be "
            f"named. Export it with tools/export_onnx.py (ultralytics writes the names "
            f"into the file), or add a 'names' metadata_props entry.")
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise DetectorError(f"{path.name}: unreadable 'names' metadata {raw[:80]!r}") from exc
    if isinstance(parsed, dict):
        return [str(parsed[k]) for k in sorted(parsed, key=int)]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    raise DetectorError(f"{path.name}: 'names' metadata is neither a dict nor a list")


class _Model:
    """One ONNX file, one session, its class names and input size."""

    def __init__(self, path: Path, providers: list[str], threads: int) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options,
                providers=[(p, _provider_options(p, path)) for p in providers])
        except Exception as exc:
            raise DetectorError(f"could not load {path.name}: {type(exc).__name__}: {exc}") from exc

        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = parse_names(meta, path)
        self.end2end = str(meta.get("end2end", "False")).strip().lower() == "true"
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_type = inp.type
        shape = inp.shape
        if len(shape) != 4:
            raise DetectorError(f"{path.name} has input shape {shape}; expected (N, 3, H, W)")
        # A static export fixes H and W; a dynamic one leaves them symbolic and
        # records the size it was exported at in metadata.
        self.dynamic = not all(isinstance(d, int) for d in shape[2:])
        if self.dynamic:
            try:
                size = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
                self.input_hw = (int(size[0]), int(size[1]))
            except (ValueError, SyntaxError, TypeError, IndexError) as exc:
                raise DetectorError(f"{path.name}: unreadable 'imgsz' metadata") from exc
        else:
            self.input_hw = (int(shape[2]), int(shape[3]))
        try:
            self.stride = int(meta.get("stride", 32))
        except ValueError:
            self.stride = 32
        self.providers = self.session.get_providers()


class OnnxDetector:
    """One or more YOLOv8 ONNX exports, each loaded once and reused per tile.

    Drop-in for hazard_detect.UltralyticsDetector; see the module docstring for
    what is reproduced and what can still differ. Several paths for the same
    reason as there: known and anomaly answer different questions, and every
    box carries the file stem of the model that produced it.

    `imgsz`, when given, must equal the size the file was exported at. A static
    export cannot be run at another size, and quietly running it at 640 while
    the configuration says 1024 would make the configuration lie.
    """

    backend = "onnx"

    def __init__(self, model_path: Any, conf: float | None = None,
                 imgsz: int | None = None, threads: int | None = None,
                 providers: Iterable[str] | None = None) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DetectorError(
                "onnxruntime is not installed. `pip install -r requirements-edge.txt`.") from exc

        given = ([Path(model_path)] if isinstance(model_path, (str, Path))
                 else [Path(p) for p in model_path])
        if not given:
            raise DetectorError("no ONNX model given")
        missing = [p for p in given if not p.is_file()]
        if missing:
            raise DetectorError("ONNX model not found: " + ", ".join(str(p) for p in missing))

        paths: list[Path] = []
        seen: set[Path] = set()
        for path in given:
            resolved = path.resolve()
            if resolved in seen:
                log.info("%s was given more than once; loading it once", path.name)
                continue
            seen.add(resolved)
            paths.append(path)

        self.paths = paths
        self.path = paths[0]
        self.conf = cfg.CONF_THRESH if conf is None else float(conf)
        self.threads = _env_threads() if threads is None else max(0, int(threads))
        self.requested_providers = choose_providers(providers)

        self.models: dict[str, _Model] = {}
        for path in paths:
            name = path.stem
            if name in self.models:
                name = f"{path.parent.name}/{path.stem}"
            self.models[name] = _Model(path, self.requested_providers, self.threads)

        sizes = {m.input_hw for m in self.models.values()}
        wanted = None if imgsz is None else int(imgsz)
        for name, model in self.models.items():
            if wanted is not None and model.input_hw != (wanted, wanted):
                raise DetectorError(
                    f"{name} was exported at {model.input_hw[1]}x{model.input_hw[0]} but "
                    f"imgsz={wanted} was requested. Re-export at that size.")
        self.imgsz = wanted if wanted is not None else max(max(s) for s in sizes)

        classes: list[str] = []
        for name, model in self.models.items():
            classes.extend(c for c in model.names if c not in classes)
            log.info("loaded ONNX detector %s (%s) with %d classes: %s", name,
                     model.providers[0], len(model.names), ", ".join(model.names))
        self.classes = classes
        self.name = " + ".join(p.name for p in paths)
        self.providers = {name: m.providers for name, m in self.models.items()}
        # The provider that leads each session. onnxruntime assigns every node
        # it can to the first provider and falls back down the list for the
        # rest, so this is "what ran most of the graph", not a guarantee that
        # every op ran there.
        self.provider = next(iter(self.models.values())).providers[0]

    def detect_raw(self, rgb: np.ndarray) -> list[tuple[str, str, float, list[float]]]:
        """(model, class, confidence, [x1, y1, x2, y2]) unrounded, for an (H, W, 3) RGB array.

        Unrounded so a caller with its own output format (the backend worker
        writes [x, y, w, h] to one decimal) rounds once, from the same floats
        the torch path gives it, rather than rounding an already-rounded value.
        """
        boxes: list[tuple[str, str, float, list[float]]] = []
        orig_hw = rgb.shape[:2]
        prepared: dict[tuple, tuple[np.ndarray, float, tuple[int, int]]] = {}
        for model_name, model in self.models.items():
            # known and anomaly share one letterboxed tensor when their input
            # geometry agrees, which it does for the shipped exports.
            key = (model.input_hw, model.dynamic, model.stride)
            if key not in prepared:
                padded, gain, pad = letterbox(rgb, model.input_hw, auto=model.dynamic,
                                              stride=model.stride)
                tensor = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                prepared[key] = (np.ascontiguousarray(tensor), gain, pad)
            tensor, gain, pad = prepared[key]
            feed = tensor.astype(np.float16) if model.input_type == "tensor(float16)" else tensor
            output = model.session.run(None, {model.input_name: feed})[0].astype(np.float32)
            rows = decode_yolov8(output, len(model.names), self.conf, end2end=model.end2end)
            if not len(rows):
                continue
            xyxy = unletterbox(rows[:, :4], gain, pad, orig_hw)
            for box, score, cls in zip(xyxy, rows[:, 4], rows[:, 5]):
                boxes.append((model_name, model.names[int(cls)], float(score),
                              [float(v) for v in box]))
        return boxes

    def detect_array(self, rgb: np.ndarray) -> list[dict]:
        """Boxes for an already-decoded (H, W, 3) uint8 RGB image, rounded as
        UltralyticsDetector rounds them."""
        return [{
            "class": cls,
            "confidence": round(score, 4),
            "bbox": [round(v, 2) for v in xyxy],
            "model": model_name,
        } for model_name, cls, score, xyxy in self.detect_raw(rgb)]

    def __call__(self, image_path: Any) -> list[dict]:
        return self.detect_array(load_rgb(image_path))


class _ChainedDetector:
    """Several detectors of different backends presented as one, in given order."""

    def __init__(self, parts: list[Any]) -> None:
        self.parts = parts
        self.paths = [p for part in parts for p in part.paths]
        self.path = self.paths[0]
        self.conf = parts[0].conf
        self.imgsz = parts[0].imgsz
        self.classes = []
        for part in parts:
            self.classes.extend(c for c in part.classes if c not in self.classes)
        self.name = " + ".join(p.name for p in self.paths)
        self.backend = "mixed"

    def __call__(self, image_path: Any) -> list[dict]:
        return [box for part in self.parts for box in part(image_path)]


BACKENDS = ("auto", "onnx", "torch")


def requested_backend() -> str:
    """DEEPECHO_DETECTOR_BACKEND: auto (default), onnx or torch."""
    value = os.environ.get("DEEPECHO_DETECTOR_BACKEND", "auto").strip().lower() or "auto"
    if value not in BACKENDS:
        raise DetectorError(f"DEEPECHO_DETECTOR_BACKEND={value!r}; expected one of {', '.join(BACKENDS)}")
    return value


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_paths(paths: Any, backend: str | None = None) -> list[Path]:
    """Which file actually runs for each configured checkpoint.

    The same rule the detector worker applies to /detect, so a survey and a
    single-tile upload never disagree about which runtime produced a box:

        auto   the .onnx beside a .pt when it exists and onnxruntime imports,
               else the .pt itself (the default, so a machine without torch
               runs the committed export and a training box runs the checkpoint)
        onnx   the .onnx beside each .pt; a missing export is an error
        torch  the .pt, always

    A path that already ends in .onnx is used as given.
    """
    backend = backend or requested_backend()
    given = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    out: list[Path] = []
    for path in given:
        if path.suffix.lower() == ".onnx":
            if backend == "torch":
                raise DetectorError(f"{path.name} is an ONNX file but DEEPECHO_DETECTOR_BACKEND=torch")
            out.append(path)
            continue
        sibling = path.with_suffix(".onnx")
        if backend == "onnx":
            if not sibling.is_file():
                raise DetectorError(f"DEEPECHO_DETECTOR_BACKEND=onnx but {sibling.name} does not exist; "
                                    "run tools/export_onnx.py")
            out.append(sibling)
        elif backend == "auto" and sibling.is_file() and _onnxruntime_available():
            out.append(sibling)
        else:
            out.append(path)
    return out


def make_detector(paths: Any, conf: float | None = None, imgsz: int | None = None,
                  backend: str | None = None) -> Any:
    """The right detector for the files given.

    Paths go through resolve_paths first, so a .pt whose .onnx export sits
    beside it runs under onnxruntime by default (DEEPECHO_DETECTOR_BACKEND
    chooses otherwise). Then by extension: .onnx -> OnnxDetector, .pt ->
    UltralyticsDetector. A mix is allowed, for a deployment where one model
    has been exported and the other has not yet; the models still run in the
    order given, so detection ids stay deterministic. The torch import only
    happens if a .pt is actually going to run.
    """
    given = resolve_paths(paths, backend)
    if not given:
        raise DetectorError("no model given")

    groups: list[tuple[bool, list[Path]]] = []
    for path in given:
        is_onnx = path.suffix.lower() == ".onnx"
        if groups and groups[-1][0] == is_onnx:
            groups[-1][1].append(path)
        else:
            groups.append((is_onnx, [path]))

    parts = [OnnxDetector(group, conf=conf, imgsz=imgsz) if is_onnx
             else UltralyticsDetector(group, conf=conf, imgsz=imgsz)
             for is_onnx, group in groups]
    return parts[0] if len(parts) == 1 else _ChainedDetector(parts)
