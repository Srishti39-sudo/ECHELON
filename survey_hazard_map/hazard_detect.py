"""Running a detector over a tile set.

The engine does not care what produced a box. It asks for a callable that takes
an image path and returns boxes, which keeps three things possible at once:

* the normal case, an Ultralytics YOLOv8 checkpoint loaded here, once;
* testing without a checkpoint, by passing a stand-in;
* hosting inside DeepEcho, where torch has to live in a subprocess because
  faiss and torch each ship their own libomp and whichever starts second aborts
  the process on macOS. A thin adapter around that worker satisfies this same
  interface, so nothing in the engine changes.

THE INTERFACE
    detector(image_path: Path) -> list[dict]

    Each dict:
        class       str    the model's own class name, verbatim
        confidence  float  0..1
        bbox        list   [x1, y1, x2, y2] in TILE pixel coordinates

    Plus, optionally, a `classes` attribute listing every class the model can
    emit, and a `name` attribute for the export's model field.

Class names are never assumed. Whatever the checkpoint emits is carried through
to the severity lookup, which has its own documented fallback for a class it
has not been taught.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable

from survey_hazard_map import hazard_config as cfg
log = logging.getLogger("deepecho.hazard")

Detector = Callable[[Path], list[dict]]

TILE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class DetectorError(RuntimeError):
    """The detector could not be loaded or could not run."""


class UltralyticsDetector:
    """One or more YOLO checkpoints, each loaded once and reused per tile.

    DeepEcho ships one, models/marine/marine.pt, whose six classes cover the
    corpus. The class accepts several because it once ran two stand-in models
    and the cross-model merge is still the right behaviour if a second one is
    ever added.

    Pass a single path for one model, or several for all of them. Every box
    carries the checkpoint that produced it, so a disagreement between two
    models stays attributable rather than being averaged away.

    Loading happens at construction rather than on first call, so a missing or
    corrupt checkpoint fails before any tiling work is spent on it.
    """

    def __init__(self, model_path: Any, conf: float | None = None,
                 imgsz: int | None = None) -> None:
        given = ([Path(model_path)] if isinstance(model_path, (str, Path))
                 else [Path(p) for p in model_path])
        if not given:
            raise DetectorError("no model checkpoint given")
        missing = [p for p in given if not p.is_file()]
        if missing:
            raise DetectorError("model checkpoint not found: "
                                + ", ".join(str(p) for p in missing))

        # The same checkpoint named two ways -- an absolute path and a relative
        # one, or through a symlink -- is one checkpoint. Loading it twice would
        # double every box it finds and then leave the cross-model merge to
        # clean up a mess it did not need to make.
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
        self.imgsz = cfg.DETECTOR_IMGSZ if imgsz is None else int(imgsz)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DetectorError(
                "ultralytics is not installed. `pip install ultralytics`, or pass "
                "your own detector callable to build_hazard_map().") from exc

        self.models: dict[str, Any] = {}
        for path in paths:
            # Keyed by stem, because that name is written into every detection
            # as its provenance. Two DIFFERENT checkpoints that happen to share
            # a filename would otherwise overwrite each other in this dict and
            # one of them would silently never run.
            name = path.stem
            if name in self.models:
                name = f"{path.parent.name}/{path.stem}"
            try:
                self.models[name] = YOLO(str(path))
            except Exception as exc:
                raise DetectorError(f"could not load {path.name}: "
                                    f"{type(exc).__name__}: {exc}") from exc

        classes: list[str] = []
        for name, model in self.models.items():
            names = getattr(getattr(model, "model", None), "names", None) or {}
            emitted = [str(v) for v in (names.values() if isinstance(names, dict) else names)]
            classes.extend(c for c in emitted if c not in classes)
            log.info("loaded detector %s with %d classes: %s",
                     name, len(emitted), ", ".join(emitted) or "unknown")
        self.classes = classes
        self.name = " + ".join(p.name for p in paths)

    def __call__(self, image_path: Path) -> list[dict]:
        boxes: list[dict] = []
        for model_name, model in self.models.items():
            for result in model.predict(source=str(image_path), conf=self.conf,
                                        imgsz=self.imgsz, verbose=False):
                names = result.names
                for box in result.boxes:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    boxes.append({
                        "class": str(names[int(box.cls)]),
                        "confidence": round(float(box.conf), 4),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "model": model_name,
                    })
        return boxes


def list_tiles(tiles_dir: Any) -> list[Path]:
    """Every tile image, in a stable order.

    Sorted by name, which for the {strip}_{x}_{y}.jpg convention means a
    deterministic sweep and therefore a deterministic set of detection ids.
    """
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"tiles directory not found: {tiles_dir}")
    return sorted((p for p in tiles_dir.iterdir() if p.suffix.lower() in TILE_SUFFIXES),
                  key=lambda p: p.name)


def parse_tile_name(name: str) -> tuple[str, int, int] | None:
    """(strip, x, y) from a tile filename, or None if it is not one of ours.

    Split from the right, so a strip whose own name contains underscores still
    parses. Only used when no manifest is available; with a manifest the
    offsets are read from it, because the manifest is the contract.
    """
    stem = Path(name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    strip, x, y = parts
    try:
        return strip, int(x), int(y)
    except ValueError:
        return None


TileCallback = Callable[[int, int, str, list[dict]], Any]


def _notify_tile(on_tile: TileCallback | None, index: int, total: int, name: str,
                 boxes: list[dict]) -> None:
    """Tell an observer a tile is finished, without letting it break the sweep.

    The callback exists so a user interface can show a survey's progress while
    it happens. It is an observer, never a participant: whatever it does, the
    sweep carries on, because a progress bar that crashes a survey has turned a
    convenience into a failure of the thing it was reporting on. A raising
    callback is logged once per tile and ignored.
    """
    if on_tile is None:
        return
    try:
        on_tile(index, total, name, boxes)
    except Exception:  # an observer's bug is not the survey's failure
        log.exception("on_tile callback raised for %s; the survey continues", name)


def run_detector(detector: Detector, tiles: Iterable[Path],
                 offsets: dict[str, dict[str, Any]],
                 on_tile: TileCallback | None = None) -> tuple[list[dict], list[dict]]:
    """Every box in every tile, tagged with where the tile sits in the survey.

    Returns (raw detections, tiles that failed). The second list is carried into
    the export rather than logged and forgotten, because a survey that quietly
    skipped nine tiles is not the same survey as one that processed them all,
    and the difference has to be visible in the artefact.

    `offsets` maps a tile filename to at least {"x", "y", "strip"}, and usually
    to its whole manifest row. A tile missing from it is still processed; its
    position is recovered from its filename, and if even that fails the tile is
    skipped with a warning rather than being placed at the origin, which would
    invent a position.

    `on_tile`, when given, is called after every tile -- processed, failed or
    unplaced alike, so a progress count always reaches the total -- as
    on_tile(index, total, tile_name, boxes_for_tile), where index is 1-based
    and boxes_for_tile are the raw detections just appended for that tile
    (empty for a tile that yielded nothing or could not be run). Omitted, the
    sweep is exactly what it was before the hook existed.
    """
    raw: list[dict] = []
    failed: list[dict] = []
    processed = 0
    unplaced = 0

    # Materialised only to know the total for the observer. A list is what
    # every caller already passes, so this costs nothing in practice.
    tiles = list(tiles)
    total = len(tiles)

    for index, tile_path in enumerate(tiles, start=1):
        name = tile_path.name
        row = offsets.get(name)
        if row is None:
            parsed = parse_tile_name(name)
            if parsed is None:
                unplaced += 1
                log.warning("tile %s is in neither the manifest nor the "
                            "{strip}_{x}_{y} naming convention; skipped", name)
                _notify_tile(on_tile, index, total, name, [])
                continue
            strip, x, y = parsed
            row = {"strip": strip, "x": x, "y": y}

        # One unreadable tile is one unreadable tile, not a failed survey. It is
        # skipped with its reason logged and the sweep continues. A survey where
        # EVERY tile fails is a different problem and does raise, below, because
        # an empty result there would look like a clean survey with nothing in it.
        try:
            boxes = detector(tile_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failed.append({"tile": name, "error": reason})
            log.error("tile %s could not be processed and was skipped: %s", name, reason)
            _notify_tile(on_tile, index, total, name, [])
            continue

        processed += 1
        tile_x, tile_y = int(row["x"]), int(row["y"])
        first_for_tile = len(raw)
        for box_index, box in enumerate(boxes):
            x1, y1, x2, y2 = (float(v) for v in box["bbox"])
            raw.append({
                "id": f"{Path(name).stem}_d{box_index}",
                "class": str(box["class"]),
                "detector_model": box.get("model"),
                "confidence": float(box["confidence"]),
                "bbox_tile": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "tile": name,
                "strip": row.get("strip"),
                "tile_x": tile_x,
                "tile_y": tile_y,
            })
        _notify_tile(on_tile, index, total, name, raw[first_for_tile:])

    if unplaced:
        log.warning("%d tile(s) skipped for having no recoverable position", unplaced)
    if failed:
        log.warning("%d tile(s) failed in the detector and were skipped: %s",
                    len(failed), ", ".join(f["tile"] for f in failed[:5])
                    + (" ..." if len(failed) > 5 else ""))
    if failed and not processed:
        raise DetectorError(
            f"every one of the {len(failed)} tile(s) failed in the detector. "
            f"First: {failed[0]['tile']} ({failed[0]['error']})")
    log.info("detector ran over %d tiles and returned %d raw boxes", processed, len(raw))
    return raw, failed
