#!/usr/bin/env python3
"""Train a YOLO detector on side-scan tiles with sonar-appropriate augmentation.

    python training/train.py --data training/pipe.yaml --model yolov8s.pt --name pipe-v1
    python training/train.py --data training/pipe.yaml --print-args      # no training

A thin wrapper over ultralytics. What it adds is the augmentation defaults
(reasoned in training/README.md section 5) and a metrics.json written beside
the weights, so a checkpoint never travels without the numbers it was judged on.

AUGMENTATION DEFAULTS, IN ONE LINE EACH
    hsv_h=0 hsv_s=0   sonar is single-channel backscatter; colour jitter is noise
    hsv_v=0.3         gain / TVG variation between systems
    flipud=0.5        along-track reversal is physically indistinguishable
    fliplr=0.5        port/starboard swap: fine for whole tiles because the shadow
                      stays pointing away from nadir; --no-lr-flip if your data is
                      single-sided or you paste crops without their shadows
    degrees=3         small yaw only; big rotations break range-aligned shadows
    scale=0.3         apparent size changes with range and altitude
    shear=0 perspective=0 mixup=0   no physical counterpart in a waterfall
    mosaic=1.0 close_mosaic=10      helps small sets, ends on whole tiles

imgsz defaults to the pipeline's DETECTOR_IMGSZ (640) so a trained checkpoint
sees tiles at the size it will see them in a survey.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SONAR_AUGMENT: dict[str, Any] = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.3,
    "degrees": 3.0,
    "translate": 0.1,
    "scale": 0.3,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
}


def train_args(args: argparse.Namespace) -> dict[str, Any]:
    """The exact keyword arguments passed to YOLO.train, for printing and for the record."""
    try:
        from survey_hazard_map import hazard_config as cfg
        default_imgsz = cfg.DETECTOR_IMGSZ
    except Exception:
        default_imgsz = 640
    kwargs = dict(SONAR_AUGMENT)
    if args.no_lr_flip:
        kwargs["fliplr"] = 0.0
    if args.no_mosaic:
        kwargs["mosaic"] = 0.0
    kwargs.update({
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz or default_imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": True,
    })
    if args.device is not None:
        kwargs["device"] = args.device
    if args.workers is not None:
        kwargs["workers"] = args.workers
    for override in args.set or []:
        key, _, value = override.partition("=")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    return kwargs


def metrics_from(results: Any, names: dict[int, str]) -> dict[str, Any]:
    """mAP50, mAP50-95, precision, recall and per-class AP from an ultralytics val result."""
    box = results.box
    per_class = {}
    maps = list(getattr(box, "maps", []) or [])
    ap50 = list(getattr(box, "ap50", []) or [])
    for position, class_id in enumerate(list(getattr(box, "ap_class_index", []) or [])):
        name = names.get(int(class_id), str(class_id))
        per_class[name] = {
            "ap50": round(float(ap50[position]), 5) if position < len(ap50) else None,
            "ap50_95": round(float(maps[int(class_id)]), 5) if int(class_id) < len(maps) else None,
        }
    return {
        "map50": round(float(box.map50), 5),
        "map50_95": round(float(box.map), 5),
        "precision": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="ultralytics dataset yaml")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="starting weights: a hub name, or models/known.pt to fine-tune")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=None, help="default: DETECTOR_IMGSZ (640)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    parser.add_argument("--name", default="sonar")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-lr-flip", action="store_true",
                        help="disable port/starboard mirroring (single-sided data, pasted crops)")
    parser.add_argument("--no-mosaic", action="store_true")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="any other ultralytics train argument, e.g. --set lr0=0.005")
    parser.add_argument("--print-args", action="store_true",
                        help="print the train arguments and exit without importing ultralytics")
    args = parser.parse_args()

    kwargs = train_args(args)
    if args.print_args:
        print(json.dumps({"model": args.model, **kwargs}, indent=2))
        return 0
    if not args.data.is_file():
        print(f"dataset yaml not found: {args.data}")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    evaluator = YOLO(str(best if best.is_file() else save_dir / "weights" / "last.pt"))
    results = evaluator.val(data=str(args.data), imgsz=kwargs["imgsz"], split="val",
                            plots=False, verbose=False)
    names = {int(k): str(v) for k, v in (evaluator.names or {}).items()}
    record = {
        "weights": best.name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_weights": str(args.model),
        "data": args.data.name,
        "split": "val",
        "classes": names,
        "metrics": metrics_from(results, names),
        "train_args": kwargs,
        "caveats": [
            "val must be split by survey line or site; tile-level splits inflate every number here",
            "metrics on pasted or synthetic val examples measure the augmentation, not the detector",
            "confidence scores are uncalibrated until training/calibrate.py has been run",
        ],
    }
    out = best.parent / "metrics.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    m = record["metrics"]
    print(f"\nmAP50 {m['map50']:.4f}  mAP50-95 {m['map50_95']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}")
    for name, ap in m["per_class"].items():
        print(f"  {name:12s} AP50 {ap['ap50']}  AP50-95 {ap['ap50_95']}")
    print(f"wrote {out}\nnext: python training/calibrate.py --weights {best} --data {args.data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
