#!/usr/bin/env python3
"""Fit detector-score calibration on a labelled validation split.

    python training/calibrate.py --weights models/known.pt --data training/data.yaml
    python training/calibrate.py --weights runs/pipe/weights/best.pt --data data.yaml \\
        --model-name pipe --out models/calibration.json

WHY
    A YOLO confidence is a ranking score, not a probability. On a small sonar
    set it is usually overconfident, and hazard_verify starts its fused score
    from logit(p): an overconfident p puts a thumb on every scale that follows.
    This script measures how often a box at a given score is actually right,
    and fits the smallest correction that makes the scores mean that.

WHAT "RIGHT" MEANS HERE
    Each prediction at or above --conf (default: the pipeline's CONF_THRESH,
    so the fit covers exactly the boxes the pipeline would keep) is matched to
    ground truth of the SAME class, greedily in descending confidence, at
    IoU >= 0.5. A matched box is label 1; an unmatched box, a duplicate, or a
    box of the wrong class is label 0. So the calibrated value is

        P(box matches a real object of its class | score)

    Missed objects never produce a prediction and do not enter the fit. This
    is calibration of precision, not recall, and the file says so.

THE TWO FITS
    temperature   p' = sigmoid(logit(p) / T)        one parameter, keeps ranking
    platt         p' = sigmoid(a * logit(p) + b)    two parameters, can shift bias

    Both are fitted by minimising negative log-likelihood with scipy. The
    method written as `method` is whichever has the lower 5-fold
    cross-validated NLL; Platt must win by more than 0.002 nats per box to be
    chosen, because a second parameter that buys nothing on held-out data is
    overfitting. The other fit is kept in the file for audit.

    Per-class entries are fitted only for classes with at least
    --min-per-class matched-or-unmatched predictions and at least 10 of each
    label; the model-level fit covers the rest.

OUTPUT
    models/calibration.json in exactly the format hazard_verify.load_calibration
    reads, with ECE before and after and a reliability table per model. The
    file is re-read through load_calibration before this script exits, so a
    file this script writes is a file the pipeline accepts.

HONEST LIMITS
    Calibration fitted on one survey's validation tiles is only valid for
    surveys that look like it (same sonar, frequency, range, seabed type). A
    validation split drawn from the same survey lines as training will look
    better calibrated than it is: split by survey line (see README.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EPS = 1e-6


# --- the math (tested in tests_verify.py on synthetic scores) ---------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _nll(z: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of labels y under probabilities sigmoid(z)."""
    # log(1 + exp(-z)) for y=1, log(1 + exp(z)) for y=0, computed stably.
    return float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)))


def fit_temperature(scores: Any, labels: Any) -> float:
    """T minimising NLL of sigmoid(logit(score) / T). Searched on log T, bounded to [0.05, 20]."""
    from scipy.optimize import minimize_scalar

    z, y = _logit(scores), np.asarray(labels, float)
    result = minimize_scalar(lambda t: _nll(z / math.exp(t), y),
                             bounds=(math.log(0.05), math.log(20.0)), method="bounded",
                             options={"xatol": 1e-5})
    return float(math.exp(result.x))


def fit_platt(scores: Any, labels: Any) -> tuple[float, float]:
    """(a, b) minimising NLL of sigmoid(a * logit(score) + b), with the analytic gradient."""
    from scipy.optimize import minimize

    z, y = _logit(scores), np.asarray(labels, float)

    def objective(params):
        a, b = params
        s = a * z + b
        p = 1 / (1 + np.exp(-np.clip(s, -40, 40)))
        grad = p - y
        return _nll(s, y), np.array([np.mean(grad * z), np.mean(grad)])

    result = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def apply_temperature(scores: Any, T: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(scores) / T))


def apply_platt(scores: Any, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * _logit(scores) + b)))


def reliability_table(probs: Any, labels: Any, bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width bins over [0, 1]: count, mean predicted, observed accuracy, gap."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = []
    for i in range(bins):
        members = index == i
        count = int(members.sum())
        row = {"bin": [round(float(edges[i]), 2), round(float(edges[i + 1]), 2)], "count": count,
               "mean_predicted": None, "observed": None, "gap": None}
        if count:
            mean_p, obs = float(p[members].mean()), float(y[members].mean())
            row.update({"mean_predicted": round(mean_p, 4), "observed": round(obs, 4),
                        "gap": round(obs - mean_p, 4)})
        table.append(row)
    return table


def expected_calibration_error(probs: Any, labels: Any, bins: int = 10) -> float:
    """sum over bins of (bin count / total) * |observed - mean predicted|."""
    table = reliability_table(probs, labels, bins)
    total = sum(row["count"] for row in table)
    return float(sum(row["count"] / total * abs(row["gap"]) for row in table if row["count"]))


def _cv_nll(scores: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Held-out NLL of each method, k-fold. Used only to choose between them."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(scores.size)
    parts = np.array_split(order, folds)
    totals = {"identity": 0.0, "temperature": 0.0, "platt": 0.0}
    for k in range(folds):
        test = parts[k]
        train = np.concatenate([parts[j] for j in range(folds) if j != k])
        T = fit_temperature(scores[train], labels[train])
        a, b = fit_platt(scores[train], labels[train])
        zt, yt = _logit(scores[test]), labels[test]
        totals["identity"] += _nll(zt, yt) * test.size
        totals["temperature"] += _nll(zt / T, yt) * test.size
        totals["platt"] += _nll(a * zt + b, yt) * test.size
    return {key: value / scores.size for key, value in totals.items()}


def fit_entry(scores: Any, labels: Any, bins: int = 10) -> dict[str, Any]:
    """One calibration entry (model-level or class-level), with its evidence."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    T = fit_temperature(s, y)
    a, b = fit_platt(s, y)
    folds = min(5, max(2, int(s.size // 20)))
    cv = _cv_nll(s, y, folds=folds)
    method = "platt" if cv["platt"] < cv["temperature"] - 0.002 else "temperature"
    calibrated = apply_platt(s, a, b) if method == "platt" else apply_temperature(s, T)
    entry = {
        "method": method,
        "temperature": round(T, 5),
        "a": round(a, 5), "b": round(b, 5),
        "n": int(s.size), "positives": int(y.sum()),
        "cv_nll": {k: round(v, 5) for k, v in cv.items()},
        "ece_before": round(expected_calibration_error(s, y, bins), 5),
        "ece_after": round(expected_calibration_error(calibrated, y, bins), 5),
        "nll_before": round(_nll(_logit(s), y), 5),
        "nll_after": round(_nll(_logit(calibrated), y), 5),
    }
    if method == "temperature":
        # Keep both fits readable, but make it unambiguous which one applies:
        # hazard_verify applies `method`, and Platt parameters on a temperature
        # entry are there for audit only.
        entry["platt_for_audit"] = {"a": entry.pop("a"), "b": entry.pop("b")}
    else:
        entry["temperature_for_audit"] = entry.pop("temperature")
    return entry


def build_calibration(records: list[dict[str, Any]], min_per_class: int = 200,
                      bins: int = 10) -> dict[str, Any]:
    """records: [{model, cls, score, label}] -> the calibration document."""
    models: dict[str, Any] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(str(record["model"]), []).append(record)

    from survey_hazard_map.hazard_severity import normalize_class

    for model, items in sorted(by_model.items()):
        scores = np.array([r["score"] for r in items])
        labels = np.array([r["label"] for r in items], float)
        entry = fit_entry(scores, labels, bins)
        entry["reliability_before"] = reliability_table(scores, labels, bins)
        applied = (apply_platt(scores, entry["a"], entry["b"]) if entry["method"] == "platt"
                   else apply_temperature(scores, entry["temperature"]))
        entry["reliability_after"] = reliability_table(applied, labels, bins)
        classes = {}
        per_class: dict[str, list[dict[str, Any]]] = {}
        for r in items:
            per_class.setdefault(normalize_class(r["cls"]), []).append(r)
        skipped = {}
        for cls, rows in sorted(per_class.items()):
            y = np.array([r["label"] for r in rows], float)
            if len(rows) < min_per_class or y.sum() < 10 or (len(y) - y.sum()) < 10:
                skipped[cls] = {"n": len(rows), "positives": int(y.sum()),
                                "reason": f"needs >= {min_per_class} predictions and >= 10 of each label"}
                continue
            classes[cls] = fit_entry(np.array([r["score"] for r in rows]), y, bins)
        entry["classes"] = classes
        if skipped:
            entry["classes_using_model_fit"] = skipped
        models[model] = entry

    return {
        "format": "deepecho-calibration",
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fitted_by": "training/calibrate.py",
        "target": ("P(prediction matches a ground-truth object of the same class at IoU >= 0.5 "
                   "| detector score), on the validation split named in `data`. Calibrates "
                   "precision only; missed objects are not represented."),
        "models": models,
    }


def write_calibration(doc: dict[str, Any], path: Any) -> Path:
    """Write, then prove the pipeline can read what was written."""
    from survey_hazard_map import hazard_verify
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    loaded = hazard_verify.load_calibration(path)
    if loaded is None:
        raise RuntimeError(f"wrote {path} but load_calibration could not read it")
    return path


# --- matching -----------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds: list[dict[str, Any]], gts: list[dict[str, Any]],
                      iou: float = 0.5) -> list[int]:
    """Label each prediction 1 (true positive) or 0, in the order given.

    Greedy per class in descending confidence, each ground truth used at most
    once: the COCO/VOC convention, so a duplicate box on a matched object is a
    false positive, as it would be in mAP.
    """
    labels = [0] * len(preds)
    used = [False] * len(gts)
    for i in sorted(range(len(preds)), key=lambda k: -preds[k]["conf"]):
        best, best_iou = None, iou
        for j, gt in enumerate(gts):
            if used[j] or gt["cls"] != preds[i]["cls"]:
                continue
            value = _iou(preds[i]["box"], gt["box"])
            if value >= best_iou:
                best, best_iou = j, value
        if best is not None:
            used[best] = True
            labels[i] = 1
    return labels


# --- dataset I/O ----------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _split_images(data_yaml: Path, split: str) -> tuple[list[Path], dict[int, str]]:
    import yaml

    spec = yaml.safe_load(data_yaml.read_text())
    base = Path(spec.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entry = spec.get(split)
    if entry is None:
        raise SystemExit(f"{data_yaml} has no '{split}' split")
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for item in entries:
        p = Path(item)
        p = p if p.is_absolute() else base / p
        if p.is_dir():
            images += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)
        elif p.suffix == ".txt":
            images += [Path(line.strip()) if Path(line.strip()).is_absolute() else base / line.strip()
                       for line in p.read_text().splitlines() if line.strip()]
    names = spec.get("names") or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    return images, {int(k): str(v) for k, v in names.items()}


def _label_path(image: Path) -> Path:
    """YOLO convention: .../images/... -> .../labels/..., suffix .txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _ground_truth(image: Path, width: int, height: int) -> list[dict[str, Any]]:
    path = _label_path(image)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, cx, cy, w, h = int(values[0]), *map(float, values[1:5])
        out.append({"cls": cls, "box": [(cx - w / 2) * width, (cy - h / 2) * height,
                                        (cx + w / 2) * width, (cy + h / 2) * height]})
    return out


def collect(weights: Path, data_yaml: Path, split: str, conf: float, imgsz: int,
            model_name: str, iou: float, device: str | None) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images, _ = _split_images(data_yaml, split)
    if not images:
        raise SystemExit(f"no images found for split '{split}' in {data_yaml}")
    records = []
    for start in range(0, len(images), 32):
        batch = images[start:start + 32]
        results = model.predict([str(p) for p in batch], conf=conf, imgsz=imgsz, verbose=False,
                                device=device)
        for image, result in zip(batch, results):
            height, width = result.orig_shape
            preds = [{"cls": int(b.cls), "conf": float(b.conf), "box": [float(v) for v in b.xyxy[0]]}
                     for b in result.boxes]
            labels = match_predictions(preds, _ground_truth(image, width, height), iou)
            records += [{"model": model_name, "cls": result.names[p["cls"]], "score": p["conf"],
                         "label": label, "image": image.name} for p, label in zip(preds, labels)]
    return records


def _print_table(title: str, table: list[dict[str, Any]]) -> None:
    print(f"  {title}")
    print("    bin          count  predicted  observed    gap")
    for row in table:
        if row["count"]:
            print(f"    {row['bin'][0]:.1f}-{row['bin'][1]:.1f}   {row['count']:7d}  "
                  f"{row['mean_predicted']:9.3f}  {row['observed']:8.3f}  {row['gap']:+.3f}")


def main() -> int:
    from survey_hazard_map import hazard_config as cfg
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="ultralytics dataset yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model-name", help="key in calibration.json; default: weights file stem")
    parser.add_argument("--conf", type=float, default=cfg.CONF_THRESH,
                        help="lowest score calibrated; default is the pipeline's CONF_THRESH")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=cfg.DETECTOR_IMGSZ)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / cfg.CALIBRATION_PATH)
    parser.add_argument("--merge", action="store_true",
                        help="keep other models already in --out instead of replacing the file")
    args = parser.parse_args()

    name = (args.model_name or args.weights.stem).lower()
    records = collect(args.weights, args.data, args.split, args.conf, args.imgsz, name,
                      args.iou, args.device)
    if len(records) < 50:
        print(f"only {len(records)} predictions at conf >= {args.conf}; too few to calibrate honestly")
        return 1

    doc = build_calibration(records, args.min_per_class, args.bins)
    doc["data"] = {"yaml": args.data.name, "split": args.split, "conf": args.conf,
                   "iou": args.iou, "weights": args.weights.name}
    if args.merge and args.out.is_file():
        existing = json.loads(args.out.read_text())
        existing.setdefault("models", {}).update(doc["models"])
        existing.update({k: v for k, v in doc.items() if k != "models"})
        doc = existing

    entry = doc["models"][name]
    print(f"\n{name}: {entry['n']} predictions, {entry['positives']} true positives")
    print(f"  method {entry['method']}  cv_nll {entry['cv_nll']}")
    print(f"  ECE {entry['ece_before']:.4f} -> {entry['ece_after']:.4f}   "
          f"NLL {entry['nll_before']:.4f} -> {entry['nll_after']:.4f}")
    _print_table("reliability before", entry["reliability_before"])
    _print_table("reliability after", entry["reliability_after"])
    for cls, c in entry["classes"].items():
        print(f"  class {cls}: {c['method']}, n={c['n']}, ECE {c['ece_before']:.4f} -> {c['ece_after']:.4f}")
    path = write_calibration(doc, args.out)
    print(f"\nwrote {path} (re-read through hazard_verify.load_calibration)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
