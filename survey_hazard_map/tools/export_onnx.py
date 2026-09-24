"""Export the YOLOv8 checkpoints to ONNX, then prove the export detects the same things.

    .venv/bin/python tools/export_onnx.py                    # models/known.pt, models/anomaly.pt
    .venv/bin/python tools/export_onnx.py --int8 --fp16      # plus quantised variants
    .venv/bin/python tools/export_onnx.py --dynamic          # plus a dynamic-shape fp32
    .venv/bin/python tools/export_onnx.py --no-export        # re-check files already present
    .venv/bin/python tools/export_onnx.py models/known.pt --no-parity

Exit status: 0 every variant kept parity; 1 the fp32 (or dynamic) export does
not match torch, or nothing could be checked; 3 fp32 matched but a quantised
variant did not.

WHAT IS WRITTEN
    models/<stem>.onnx         fp32, static 1x3x640x640, opset 17, simplified
    models/<stem>.int8.onnx    --int8: weights int8, activations quantised at run time
    models/<stem>.fp16.onnx    --fp16: fp16 weights and activations, fp32 inputs/outputs
    models/<stem>.dynamic.onnx --dynamic: fp32 with symbolic H and W
    docs/edge_parity.json      the parity measurements below

WHY THESE EXPORT SETTINGS
    * Static 640 by default. Interior survey tiles are 640x640, and TensorRT
      builds its fastest engine for one fixed shape. The cost is measured
      below: tiles at a strip's edge (640x359, 198x640 ...) reach a static
      graph padded to the full square, while ultralytics' torch predict pads
      them only to the next multiple of 32, and confidences on those tiles
      move. --dynamic writes a second export that removes that difference, at
      the price of an optimisation profile on TensorRT.
    * Opset 17. onnxruntime 1.30 supports far newer, but TensorRT 8.6 (the
      JetPack 5 line still flying on many Orin carriers) tops out at 17, and
      TensorRT 10 reads it as well. A YOLOv8 graph uses no op newer than 17
      would add.
    * Simplify (onnxslim). Folds constants and removes shape arithmetic, which
      is what lets TensorRT and the CPU provider fuse Conv+Act layers.
    * No NMS in the graph. The numpy NMS in hazard_detect_onnx.py is exactly
      the ultralytics rule, and keeping it outside keeps the ONNX a plain
      feed-forward graph that every runtime accepts.

WHY THE INT8 IS DYNAMIC, AND WHAT THAT COSTS
    Dynamic quantisation needs no calibration data, which matters because this
    repository has 80 real sonar images, not the few hundred representative
    frames a static calibration should be fit on (and calibrating on the same
    tiles the parity check then scores would grade its own homework). The cost
    is that activations are quantised per inference.

    Measured, and it is not good enough to ship as equivalent: on the 640x640
    tiles the int8 files moved confidences by up to 0.10 (known) and 0.19
    (anomaly), and boxes appeared and disappeared at the 0.25 threshold. On an
    M1 CPU it ran about 2x (4 threads) to 2.5x (1 thread) faster than fp32 and
    is a quarter of the size.
    Excluding the detection head from quantisation and per-channel weights
    were tried and did not bring the score deltas under 0.1. It is written
    for anyone who wants to evaluate the trade on their own data; the run
    exits 3 to say it is not a drop-in.

WHY THERE IS A PARITY CHECK AT ALL
    An export that loads and returns boxes is not the same as an export that
    returns the SAME boxes. Every image in samples/ and every tile under
    data/surveys/*/tiles/ is run through the torch checkpoint (via
    hazard_detect.UltralyticsDetector, exactly as the survey engine runs it)
    and through the ONNX file (via hazard_detect_onnx.OnnxDetector, CPU
    provider, so the comparison is of the export and not of a GPU's fp16).

    Both run once at a low threshold; the comparison is then made at several
    thresholds by filtering. That is exact rather than an approximation: a box
    above a threshold can only be suppressed by a higher-scoring box, which is
    also above it, so NMS-then-filter equals filter-then-NMS.

    Boxes are paired within one image and one model, same class, greedily by
    IoU (>= 0.5). Reported: the largest confidence difference among pairs, the
    smallest IoU among pairs, and every box one side found and the other did
    not. An unpaired box whose confidence is within --edge of the threshold is
    counted as a threshold-edge flip, which a continuous score crossing a hard
    threshold will always produce; any other unpaired box is a real
    disagreement.

    fp32 parity fails (exit 1) when the confidence delta exceeds --tol-conf,
    the IoU falls below --min-iou, or there is any real disagreement. A
    quantised variant is held to the same bar; if it misses, it is still
    written but the run exits 3 and the JSON says "parity": false, so nobody
    ships an int8 file on the assumption that it matched.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from survey_hazard_map.hazard_detect import TILE_SUFFIXES  # noqa: E402

THRESHOLDS = (0.10, 0.25, 0.30)


def parity_images() -> list[Path]:
    """Every real image available: samples/ top level, samples/tiles, survey tiles."""
    found: list[Path] = []
    for folder in [ROOT / "survey_hazard_map" / "samples", ROOT / "survey_hazard_map" / "samples" / "tiles",
                   *sorted((ROOT / "data" / "surveys").glob("*/tiles"))]:
        if folder.is_dir():
            found.extend(sorted(p for p in folder.iterdir()
                                if p.is_file() and p.suffix.lower() in TILE_SUFFIXES))
    return found


def export_fp32(pt: Path, imgsz: int, opset: int, dynamic: bool = False) -> Path:
    from ultralytics import YOLO

    out = YOLO(str(pt)).export(format="onnx", imgsz=imgsz, opset=opset, simplify=True,
                               dynamic=dynamic, device="cpu", verbose=False)
    out = Path(out)
    target = pt.with_name(pt.stem + ".dynamic.onnx") if dynamic else pt.with_suffix(".onnx")
    if out.resolve() != target.resolve():
        out.replace(target)
    return target


def export_int8(fp32: Path) -> Path | None:
    """Dynamic int8. Falls back to uint8 weights if this build lacks int8 ConvInteger."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    target = fp32.with_name(fp32.stem + ".int8.onnx")
    prepped = fp32.with_name(fp32.stem + ".preq.onnx")
    try:
        try:
            quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
            source = prepped
        except Exception as exc:
            print(f"  int8: pre-processing skipped ({type(exc).__name__}: {exc})")
            source = fp32
        last: Exception | None = None
        for weight_type in (QuantType.QInt8, QuantType.QUInt8):
            try:
                quantize_dynamic(str(source), str(target), weight_type=weight_type)
                _check_loads(target)
                print(f"  int8: dynamic quantisation, {weight_type.name} weights")
                _copy_metadata(fp32, target)
                return target
            except Exception as exc:
                last = exc
                target.unlink(missing_ok=True)
        print(f"  int8: SKIPPED, quantisation failed: {type(last).__name__}: {last}")
        return None
    finally:
        prepped.unlink(missing_ok=True)


def export_fp16(fp32: Path) -> Path | None:
    """fp16 weights and activations, fp32 at the edges so callers feed float32."""
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        print(f"  fp16: SKIPPED, converter unavailable ({exc})")
        return None
    target = fp32.with_name(fp32.stem + ".fp16.onnx")
    try:
        model = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
        onnx.save(model, str(target))
        _check_loads(target)
        _copy_metadata(fp32, target)
        print("  fp16: converted, float32 inputs and outputs kept")
        return target
    except Exception as exc:
        target.unlink(missing_ok=True)
        print(f"  fp16: SKIPPED, conversion or load failed: {type(exc).__name__}: {exc}")
        return None


def _check_loads(path: Path) -> None:
    import onnxruntime as ort

    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _copy_metadata(src: Path, dst: Path) -> None:
    """Quantisers drop metadata_props on some versions; the class names must survive."""
    import onnx

    source, target = onnx.load(str(src)), onnx.load(str(dst))
    have = {p.key for p in target.metadata_props}
    missing = [p for p in source.metadata_props if p.key not in have]
    if missing:
        for prop in missing:
            target.metadata_props.add(key=prop.key, value=prop.value)
        onnx.save(target, str(dst))


# --- Parity ------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference: dict[str, list[dict]], candidate: dict[str, list[dict]],
            threshold: float, edge: float, images: list[str] | None = None) -> dict:
    """Pair boxes per image, same class, greedy by IoU, at one threshold."""
    pairs = 0
    max_delta = 0.0
    min_iou = 1.0
    total_ref = total_cand = 0
    missing: list[dict] = []
    extra: list[dict] = []
    for image in (reference if images is None else images):
        ref = [b for b in reference[image] if b["confidence"] > threshold]
        cand = [b for b in candidate.get(image, []) if b["confidence"] > threshold]
        total_ref += len(ref)
        total_cand += len(cand)
        options = sorted(((_iou(r["bbox"], c["bbox"]), i, j)
                          for i, r in enumerate(ref) for j, c in enumerate(cand)
                          if r["class"] == c["class"]), reverse=True)
        used_r: set[int] = set()
        used_c: set[int] = set()
        for iou, i, j in options:
            if iou < 0.5 or i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            pairs += 1
            max_delta = max(max_delta, abs(ref[i]["confidence"] - cand[j]["confidence"]))
            min_iou = min(min_iou, iou)
        for i, box in enumerate(ref):
            if i not in used_r:
                missing.append({"image": image, **box,
                                "threshold_edge": box["confidence"] <= threshold + edge})
        for j, box in enumerate(cand):
            if j not in used_c:
                extra.append({"image": image, **box,
                              "threshold_edge": box["confidence"] <= threshold + edge})
    real = [b for b in missing + extra if not b["threshold_edge"]]
    return {
        "threshold": threshold,
        "images": len(reference if images is None else images),
        "reference_boxes": total_ref,
        "candidate_boxes": total_cand,
        "matched": pairs,
        "max_confidence_delta": round(max_delta, 5),
        "min_box_iou": round(min_iou, 5) if pairs else None,
        "missing": missing,
        "extra": extra,
        "real_disagreements": len(real),
    }


def run_all(detector, images: list[Path]) -> tuple[dict[str, list[dict]], float]:
    out: dict[str, list[dict]] = {}
    started = time.perf_counter()
    for image in images:
        out[str(image.relative_to(ROOT))] = detector(image)
    return out, time.perf_counter() - started


def _passes(level: dict, tol_conf: float, min_iou: float) -> bool:
    return (level["real_disagreements"] == 0
            and level["max_confidence_delta"] <= tol_conf
            and (level["min_box_iou"] is None or level["min_box_iou"] >= min_iou))


def _print_level(tag: str, level: dict) -> None:
    edge_flips = sum(b["threshold_edge"] for b in level["missing"] + level["extra"])
    iou = level["min_box_iou"]
    print(f"        {tag:<7} conf>{level['threshold']:.2f}: torch {level['reference_boxes']:>3} "
          f"onnx {level['candidate_boxes']:>3} matched {level['matched']:>3}  "
          f"max dconf {level['max_confidence_delta']:.4f}  "
          f"min IoU {iou if iou is not None else '-':<7}  "
          f"missing {len(level['missing'])} extra {len(level['extra'])} "
          f"(edge flips {edge_flips}, real {level['real_disagreements']})")


def parity_for(pt: Path, variants: dict[str, Path], images: list[Path], low_conf: float,
               imgsz: int, tol_conf: float, min_iou: float, edge: float) -> dict:
    """Parity of every variant against the torch checkpoint, split by input shape.

    The split exists because of one measured fact. ultralytics' torch predict
    sets rect=True, so a 640x359 edge tile is padded only to 640x384; a static
    ONNX can only be fed 640x640. For a tile that is already 640x640 the two
    canvases are identical and parity is held to the tolerances. For any other
    shape the network sees a different grey border and confidences move by a
    few hundredths, which no amount of care in the preprocessing can remove.
    Those inputs are measured and reported for a static export, and gated only
    for a dynamic export, which can reproduce the rect canvas exactly.
    """
    from PIL import Image

    from survey_hazard_map.hazard_detect import UltralyticsDetector
    from survey_hazard_map.hazard_detect_onnx import OnnxDetector

    torch_det = UltralyticsDetector(pt, conf=low_conf, imgsz=imgsz)
    reference, t_ref = run_all(torch_det, images)
    native, other = [], []
    for image in images:
        with Image.open(image) as im:
            (native if im.size == (imgsz, imgsz) else other).append(str(image.relative_to(ROOT)))

    result: dict = {"checkpoint": str(pt.relative_to(ROOT)), "variants": {},
                    "native_640_images": len(native), "other_shape_images": len(other)}
    print(f"\n  parity for {pt.name}: {len(native)} native {imgsz}x{imgsz} inputs, "
          f"{len(other)} other shapes; torch pass {t_ref:.1f}s")
    for label, path in variants.items():
        det = OnnxDetector(path, conf=low_conf, imgsz=imgsz, providers=["CPUExecutionProvider"])
        dynamic = any(m.dynamic for m in det.models.values())
        candidate, t_cand = run_all(det, images)
        # The model stem differs between files (known vs known.int8); compare
        # boxes, not provenance strings.
        levels_native = [compare(reference, candidate, thr, edge, native) for thr in THRESHOLDS]
        levels_other = [compare(reference, candidate, thr, edge, other) for thr in THRESHOLDS]
        ok_native = all(_passes(level, tol_conf, min_iou) for level in levels_native)
        ok_other = all(_passes(level, tol_conf, min_iou) for level in levels_other)
        gated_other = dynamic
        ok = ok_native and (ok_other or not gated_other)
        result["variants"][label] = {
            "file": str(path.relative_to(ROOT)), "dynamic_shape": dynamic,
            "parity": ok, "parity_native_640": ok_native, "parity_other_shapes": ok_other,
            "other_shapes_gated": gated_other, "seconds": round(t_cand, 2),
            "levels_native_640": levels_native, "levels_other_shapes": levels_other}
        verdict = "PARITY OK" if ok else "PARITY FAILED"
        print(f"  {label:<7} {verdict:<14} (onnx pass {t_cand:.1f}s, provider {det.provider}, "
              f"{'dynamic' if dynamic else 'static'} shape)")
        for level in levels_native:
            _print_level("640x640", level)
        for level in levels_other:
            _print_level("other", level)
        if not ok_other and not gated_other:
            print(f"        note: non-640 inputs differ (rect canvas in torch, square in a static "
                  f"export); reported, not gated. Use --dynamic to remove this.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", nargs="*", type=Path,
                        default=[ROOT / "models" / "known.pt", ROOT / "models" / "anomaly.pt"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--int8", action="store_true", help="also write <stem>.int8.onnx")
    parser.add_argument("--fp16", action="store_true", help="also write <stem>.fp16.onnx")
    parser.add_argument("--dynamic", action="store_true",
                        help="also write <stem>.dynamic.onnx (dynamic H/W, exact rect parity)")
    parser.add_argument("--no-export", action="store_true",
                        help="skip exporting; run parity on the .onnx files already present")
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument("--low-conf", type=float, default=0.05,
                        help="threshold both backends run at before filtering (default 0.05)")
    parser.add_argument("--tol-conf", type=float, default=0.02,
                        help="largest acceptable confidence difference on a paired box")
    parser.add_argument("--min-iou", type=float, default=0.95,
                        help="smallest acceptable IoU between paired boxes")
    parser.add_argument("--edge", type=float, default=0.03,
                        help="an unpaired box this close above the threshold is a threshold flip")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "edge_parity.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    checkpoints = [p if p.is_absolute() else (Path.cwd() / p) for p in args.checkpoints]
    for pt in checkpoints:
        if not pt.is_file():
            print(f"checkpoint not found: {pt}", file=sys.stderr)
            return 1

    exported: dict[Path, dict[str, Path]] = {}
    for pt in checkpoints:
        print(f"{pt.name}:")
        if args.no_export:
            variants = {"fp32": pt.with_suffix(".onnx"),
                        "int8": pt.with_name(pt.stem + ".int8.onnx"),
                        "fp16": pt.with_name(pt.stem + ".fp16.onnx"),
                        "dynamic": pt.with_name(pt.stem + ".dynamic.onnx")}
            variants = {k: v for k, v in variants.items() if v.is_file()}
            if "fp32" not in variants:
                print(f"  no {pt.stem}.onnx to check; run without --no-export", file=sys.stderr)
                return 1
        else:
            # The dynamic export goes first: ultralytics writes <stem>.onnx
            # either way, and it is renamed before the static one lands there.
            dynamic = export_fp32(pt, args.imgsz, args.opset, dynamic=True) if args.dynamic else None
            variants = {"fp32": export_fp32(pt, args.imgsz, args.opset)}
            print(f"  fp32: {variants['fp32'].relative_to(ROOT)} "
                  f"({variants['fp32'].stat().st_size / 1e6:.1f} MB)")
            if args.int8 and (path := export_int8(variants["fp32"])):
                variants["int8"] = path
            if args.fp16 and (path := export_fp16(variants["fp32"])):
                variants["fp16"] = path
            if dynamic is not None:
                variants["dynamic"] = dynamic
        for label, path in variants.items():
            print(f"  {label}: {path.name} {path.stat().st_size / 1e6:.1f} MB")
        exported[pt] = variants

    if args.no_parity:
        return 0

    images = parity_images()
    if not images:
        print("no images found for the parity check", file=sys.stderr)
        return 1

    import onnxruntime
    import ultralytics

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {"platform": platform.platform(), "machine": platform.machine(),
                   "python": platform.python_version()},
        "versions": {"onnxruntime": onnxruntime.__version__, "ultralytics": ultralytics.__version__},
        "method": ("both backends run at low_conf, filtered to each threshold; boxes paired "
                   "per image and model, same class, greedy by IoU >= 0.5; ONNX on "
                   "CPUExecutionProvider"),
        "low_conf": args.low_conf, "tol_conf": args.tol_conf, "min_iou": args.min_iou,
        "edge": args.edge, "images": len(images),
        "checkpoints": [],
    }
    for pt, variants in exported.items():
        report["checkpoints"].append(parity_for(pt, variants, images, args.low_conf, args.imgsz,
                                                args.tol_conf, args.min_iou, args.edge))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nparity report: {args.report.relative_to(ROOT) if args.report.is_relative_to(ROOT) else args.report}")

    fp32_bad = [c["checkpoint"] for c in report["checkpoints"]
                if not c["variants"]["fp32"]["parity"]]
    lossy_bad = [f"{c['checkpoint']} {label}" for c in report["checkpoints"]
                 for label, v in c["variants"].items()
                 if label not in ("fp32", "dynamic") and not v["parity"]]
    dynamic_bad = [c["checkpoint"] for c in report["checkpoints"]
                   if "dynamic" in c["variants"] and not c["variants"]["dynamic"]["parity"]]
    fp32_bad += [f"{name} (dynamic)" for name in dynamic_bad]
    if fp32_bad:
        print("\n!!! FP32 ONNX DOES NOT MATCH TORCH for " + ", ".join(fp32_bad)
              + ". Do not deploy it. See the report for the boxes that differ.", file=sys.stderr)
        return 1
    if lossy_bad:
        print("\n!!! Quantised variant(s) did not keep parity: " + ", ".join(lossy_bad)
              + ". They were written but must not be deployed as equivalent to fp32.",
              file=sys.stderr)
        return 3
    print("all variants kept parity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
