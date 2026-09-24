#!/usr/bin/env python3
"""Tests for hazard_verify (confidence scoring and noise filtering) and the
calibration fitting in training/calibrate.py.

    python3 tests_verify.py
    python3 tests_verify.py --verbose     # also prints the per-scene table
    python3 tests_verify.py --real        # also runs the public sample surveys

Exit code is 1 if any case fails. Plain functions and asserts, like
unit_tests.py, for the same reason: no test framework dependency.

WHAT THE SYNTHETIC SCENE IS, AND IS NOT
    A Rayleigh-speckle seabed with a dark water column, and six hand-built
    targets that each exercise one cue: a proud cylinder with a far-side
    shadow, a rock field, a soft-edged depression, a box on the water-column
    boundary, a pipe, and a faint mesh. The tests assert the ORDERING of the
    fused scores and which hard reasons fire. They prove the cues respond to
    the physics they were written for. They do not prove the weights are
    right for any real sonar, which only a labelled validation set can.

The --real mode runs the public records in samples/ against the detections
the pipeline already exported in data/surveys/. Nothing in hazard_verify
knows about those files; the assertions here are the expected outcome
written down in samples/README.md (every waterfall-strip detection is a false
positive on the nadir/shadow boundary, the S-7 wreck is a true positive).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "survey_hazard_map" / "training"))

import numpy as np

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map import hazard_verify as hv
CASES = []
VERBOSE = False


def case(name):
    def register(fn):
        CASES.append((name, fn))
        return fn
    return register


# --- synthetic scene -------------------------------------------------------

H, W = 640, 1400
NADIR = 700.0
WC_HALF = 90          # water column 610 .. 790
SEABED = 90.0


def rayleigh(rng, shape, mean):
    """Rayleigh-distributed amplitude with the given mean (fully developed speckle)."""
    return rng.rayleigh(1.0, shape) * (mean / math.sqrt(math.pi / 2))


def base_strip(rng):
    """Speckled seabed either side of a dark, low-noise water column."""
    level = np.full((H, W), SEABED, np.float64)
    level[:, int(NADIR - WC_HALF):int(NADIR + WC_HALF)] = 6.0
    return level


def draw_object(level, x0, x1, y0, y1, far, highlight=2.2, shadow_len=None, shadow=0.12):
    """A proud rectangle: bright return, then a hard shadow on the far side."""
    level[y0:y1, x0:x1] *= highlight
    length = (x1 - x0) if shadow_len is None else shadow_len
    if far > 0:
        level[y0:y1, x1:x1 + length] *= shadow
    else:
        level[y0:y1, max(0, x0 - length):x0] *= shadow


def scene(seed=7):
    """The six-target scene and the detections a detector might report on it."""
    rng = np.random.default_rng(seed)
    level = base_strip(rng)
    targets = {}

    # (a) proud cylinder, starboard, highlight 60x22 then an equal shadow.
    draw_object(level, 1000, 1060, 90, 112, far=+1)
    targets["cylinder"] = ("cylinder", [996, 86, 1064, 116])

    # (b) rock field, port side (far = smaller x): 30 irregular rocks of
    # similar size, each lit on its nadir side with a shadow beyond.
    yy, xx = np.mgrid[0:H, 0:W]
    placed = []
    while len(placed) < 30:
        cx, cy = rng.uniform(130, 430), rng.uniform(300, 600)
        if any(math.hypot(cx - px, cy - py) < 34 for px, py in placed):
            continue
        placed.append((cx, cy))
        a, b = rng.uniform(7, 11), rng.uniform(5, 9)
        t = rng.uniform(0, math.pi)
        u = (xx - cx) * math.cos(t) + (yy - cy) * math.sin(t)
        v = -(xx - cx) * math.sin(t) + (yy - cy) * math.cos(t)
        jag = 1.0 + 0.25 * np.sin(5 * np.arctan2(v, u) + rng.uniform(0, 6))
        rock = (u / a) ** 2 + (v / b) ** 2 <= jag
        shadow = np.roll(rock, -int(2 * a), axis=1) & ~rock
        level[shadow] *= 0.15
        level[rock] *= 2.0
    # The detection is on the rock nearest the middle of the field: a rock on
    # the field's edge has half the neighbours, and that is a different test.
    cx, cy = min(placed, key=lambda p: math.hypot(p[0] - 280, p[1] - 450))
    targets["rock"] = ("mine", [int(cx - 16), int(cy - 14), int(cx + 16), int(cy + 14)])

    # (c) soft-edged depression: dark, no highlight, grading over ~20 px.
    g = np.exp(-(((xx - 1250) / 22.0) ** 2 + ((yy - 470) / 16.0) ** 2))
    level *= 1.0 - 0.85 * g
    targets["shadow_blob"] = ("mine", [1222, 448, 1278, 492])

    # (d) a box on the starboard first-return boundary.
    targets["nadir"] = ("shipwreck", [760, 300, 830, 440])

    # (e) a pipe crossing the track diagonally, starboard, with a thin shadow.
    for y in range(160, 420):
        x = int(880 + 0.3 * (y - 160))
        level[y, x:x + 7] *= 2.0
        level[y, x + 7:x + 13] *= 0.2
    targets["pipe"] = ("pipe", [872, 156, 972, 424])

    # (f) a faint draped net, port side: 2 px mesh lines every 11 px, +45%,
    # no relief and so no shadow.
    net = np.zeros((90, 90), bool)
    net[::11, :] = True
    net[1::11, :] = True
    net[:, ::11] = True
    net[:, 1::11] = True
    level[60:150, 200:290][net] *= 1.45
    targets["net"] = ("net", [200, 60, 290, 150])

    image = np.clip(rayleigh(rng, (H, W), 1.0) * level, 0, 255).astype(np.float32)
    detections = [{"id": name, "class": cls, "confidence": 0.6, "strip": "synthetic",
                   "detector_model": "known", "bbox_global": box}
                  for name, (cls, box) in targets.items()]
    return image, detections


_SCENE = {}


def verified_scene():
    if "result" not in _SCENE:
        image, detections = scene()
        ctx = hv.strip_context(None, None, strip="synthetic", grey=image)
        summary = hv.verify_survey(detections, {"synthetic": ctx}, None)
        _SCENE["result"] = ({d["id"]: d for d in detections}, summary, ctx)
        if VERBOSE:
            print("\n  synthetic scene (detector confidence 0.60 for every box)")
            for d in detections:
                v = d["verification"]
                print(f"    {d['id']:12s} {d['class']:10s} {d['confidence_pct']:5.1f}%  "
                      f"suppressed={str(d['suppressed']):5s} hard={v['hard_reasons']}")
                print("                 " + ", ".join(
                    f"{t['name']}={t['contribution']:+.2f}" for t in v["terms"]))
            print()
    return _SCENE["result"]


# --- nadir -----------------------------------------------------------------

@case("nadir is estimated from a plain image, and the water column is traced")
def _():
    _, _, ctx = verified_scene()
    assert ctx.nadir_col is not None, ctx.nadir_basis
    assert abs(ctx.nadir_col - NADIR) < 15, ctx.nadir_col
    assert ctx.nadir_basis["source"] == "estimated"
    left = np.nanmedian(ctx.wc_left)
    right = np.nanmedian(ctx.wc_right)
    assert abs(left - (NADIR - WC_HALF)) < 12, left
    assert abs(right - (NADIR + WC_HALF)) < 12, right


@case("no nadir is claimed for an image with no water column")
def _():
    rng = np.random.default_rng(1)
    flat = rayleigh(rng, (300, 600), 90.0).astype(np.float32)
    nadir, info = hv.estimate_nadir(flat)
    assert nadir is None, info
    assert "reason" in info


@case("a sidecar nadir is used as given")
def _():
    rng = np.random.default_rng(2)
    image = rayleigh(rng, (200, 400), 90.0).astype(np.float32)
    ctx = hv.strip_context(None, {"nadir_col": 123.0}, strip="s", grey=image)
    assert ctx.nadir_col == 123.0 and ctx.nadir_basis["source"] == "sidecar"


# --- the six targets -------------------------------------------------------

@case("proud cylinder keeps its confidence and is not suppressed")
def _():
    d, _, _ = verified_scene()
    c = d["cylinder"]
    assert not c["suppressed"]
    assert c["confidence_pct"] > 60.0, c["confidence_pct"]
    assert c["verification"]["hard_reasons"] == [], c["verification"]["hard_reasons"]
    assert c["verification"]["evidence"]["acoustic_shadow"]["score"] >= 0.6


@case("rock field is flagged as clutter and suppressed")
def _():
    d, _, _ = verified_scene()
    r = d["rock"]
    assert "rock_clutter" in r["verification"]["hard_reasons"], r["verification"]["evidence"]["rock_clutter"]
    assert r["suppressed"], r["confidence_pct"]


@case("soft dark blob is a natural shadow and suppressed")
def _():
    d, _, _ = verified_scene()
    s = d["shadow_blob"]
    assert "natural_shadow" in s["verification"]["hard_reasons"], s["verification"]["evidence"]["natural_shadow"]
    assert s["suppressed"]


@case("box on the water-column boundary is flagged nadir_zone and suppressed")
def _():
    d, _, _ = verified_scene()
    n = d["nadir"]
    assert "nadir_zone" in n["verification"]["hard_reasons"]
    assert n["suppressed"]
    assert any("nadir" in reason for reason in n["verification"]["reasons"])


@case("pipe is supported by straight, coherent edges and not suppressed")
def _():
    d, _, _ = verified_scene()
    p = d["pipe"]
    assert not p["suppressed"]
    assert p["verification"]["evidence"]["man_made_regularity"]["score"] >= 0.5, \
        p["verification"]["evidence"]["man_made_regularity"]
    assert p["confidence_pct"] >= 55.0, p["confidence_pct"]


@case("faint net is not penalised for lacking a shadow")
def _():
    d, _, _ = verified_scene()
    n = d["net"]
    assert not n["suppressed"]
    mesh = n["verification"]["evidence"]["man_made_regularity"]["measurements"]["mesh"]
    assert mesh["score"] >= 0.5, mesh
    shadow = next(t for t in n["verification"]["terms"] if t["name"] == "acoustic_shadow")
    assert shadow["contribution"] > -0.1, shadow
    assert n["confidence_pct"] >= 55.0, n["confidence_pct"]


@case("confidence ordering: supported targets above every artefact")
def _():
    d, _, _ = verified_scene()
    good = [d[k]["confidence_pct"] for k in ("cylinder", "pipe", "net")]
    bad = [d[k]["confidence_pct"] for k in ("rock", "shadow_blob", "nadir")]
    assert min(good) > max(bad), (good, bad)
    assert max(bad) < cfg.SUPPRESS_BELOW_PCT
    assert d["cylinder"]["confidence_pct"] > d["rock"]["confidence_pct"]


@case("every fused score is recomputable from its own terms")
def _():
    d, _, _ = verified_scene()
    for det in d.values():
        v = det["verification"]
        total = sum(t["contribution"] for t in v["terms"])
        assert abs(100 / (1 + math.exp(-total)) - det["confidence_pct"]) < 0.2, det["id"]
        for t in v["terms"][1:]:
            expected = t["weight"] * t["applicability"] * (t["score"] - t["neutral"])
            assert abs(expected - t["contribution"]) < 2e-3, (det["id"], t)
        assert v["detector_confidence"] == det["confidence"] == 0.6


@case("nothing is deleted, the raw confidence is untouched, summary counts add up")
def _():
    d, summary, _ = verified_scene()
    assert len(d) == 6
    assert summary["checked"] == 6
    assert summary["suppressed"] == sum(x["suppressed"] for x in d.values())
    for det in d.values():
        assert det["confidence"] == 0.6
        assert "dimensions" in det and "height_m" in det["dimensions"]


@case("verify_detection does not mutate its input")
def _():
    image, detections = scene()
    ctx = hv.strip_context(None, None, strip="synthetic", grey=image)
    before = json.dumps(detections[0], sort_keys=True)
    hv.verify_detection(detections[0], ctx, None)
    assert json.dumps(detections[0], sort_keys=True) == before


@case("a detection on a strip with no image is kept, unchecked and unsuppressed")
def _():
    det = {"id": "x", "class": "mine", "confidence": 0.7, "strip": "missing",
           "bbox_global": [0, 0, 10, 10]}
    summary = hv.verify_survey([det], {}, None)
    assert det["verification"]["status"] == "not_checked"
    assert det["suppressed"] is False and det["confidence_pct"] == 70.0
    assert summary["not_checked"] == 1


@case("water-column classes are exempt from the water-column cue, not the nadir line")
def _():
    image, _ = scene()
    ctx = hv.strip_context(None, None, strip="synthetic", grey=image)
    in_column = {"class": "fish", "confidence": 0.6, "bbox_global": [640, 200, 670, 230]}
    on_line = {"class": "fish", "confidence": 0.6, "bbox_global": [690, 400, 712, 422]}
    shipwreck = dict(in_column, **{"class": "shipwreck"})
    a = hv.verify_detection(in_column, ctx, None)["evidence"]["nadir_zone"]
    b = hv.verify_detection(on_line, ctx, None)["evidence"]["nadir_zone"]
    c = hv.verify_detection(shipwreck, ctx, None)["evidence"]["nadir_zone"]
    assert a["measurements"]["wc_score"] == 0.0 and a["measurements"]["water_column_class_exempt"]
    assert c["measurements"]["wc_score"] == 1.0
    assert b["measurements"]["line_score"] > 0.5


@case("rows flagged degraded in the sidecar raise a dropout reason")
def _():
    image, detections = scene()
    cylinder = next(d for d in detections if d["id"] == "cylinder")
    sidecar = {"nadir_col": NADIR, "degraded_rows": [[80, 130, "dropout"]]}
    ctx = hv.strip_context(None, sidecar, strip="synthetic", grey=image)
    v = hv.verify_detection(cylinder, ctx, None)
    assert "dropout" in v["hard_reasons"], v["evidence"]["dropout"]
    assert v["evidence"]["dropout"]["measurements"]["degraded_reasons"] == ["dropout"]
    clean = hv.verify_detection(cylinder, hv.strip_context(None, {"nadir_col": NADIR},
                                                           strip="s", grey=image), None)
    assert v["confidence_pct"] < clean["confidence_pct"]


# --- height from shadow ----------------------------------------------------

def _height_case(slant: bool):
    """Flat seabed, towfish at 20 m, 0.1 m/px, a 2.0 m object whose far edge is at 20 m ground range."""
    altitude, res, true_h = 20.0, 0.1, 2.0
    nadir, rows = 300.0, (200, 240)
    ground_edge = 20.0
    ground_end = ground_edge + true_h * ground_edge / (altitude - true_h)   # similar triangles

    def to_px(ground):
        rng_m = math.hypot(ground, altitude) if slant else ground
        return nadir + rng_m / res

    edge_px, end_px = int(round(to_px(ground_edge))), int(round(to_px(ground_end)))
    rng = np.random.default_rng(3)
    level = np.full((400, 1000), SEABED)
    level[rows[0]:rows[1], edge_px - 20:edge_px] *= 2.2
    level[rows[0]:rows[1], edge_px:end_px] *= 0.1
    image = np.clip(rayleigh(rng, level.shape, 1.0) * level, 0, 255).astype(np.float32)
    sidecar = {"nadir_col": nadir, "m_per_px_across": res, "m_per_px_along": res,
               "port_is_left": True, "slant_range_corrected": not slant,
               "rows": [{"row": r, "altitude_m": altitude, "quality": "ok"} for r in range(0, 400, 10)]}
    ctx = hv.strip_context(None, sidecar, strip="h", grey=image)
    det = {"class": "cylinder", "confidence": 0.7, "strip": "h",
           "bbox_global": [edge_px - 22, rows[0] - 2, edge_px + 2, rows[1] + 2]}
    return hv.verify_detection(det, ctx, None)["height"], true_h


@case("height from shadow, ground-range strip, matches the constructed 2.0 m")
def _():
    height, true_h = _height_case(slant=False)
    assert height["height_m"] is not None, height
    assert abs(height["height_m"] - true_h) <= 0.2, height
    assert height["formula"] == "h = altitude * Ls / (R + Ls)"


@case("height from shadow, slant-range strip, converts range and matches 2.0 m")
def _():
    height, true_h = _height_case(slant=True)
    assert height["height_m"] is not None, height
    assert abs(height["height_m"] - true_h) <= 0.3, height
    assert "slant range converted" in height["basis"]


@case("height is null, with the missing inputs named, when geometry is unknown")
def _():
    d, _, _ = verified_scene()
    h = d["cylinder"]["verification"]["height"]
    assert h["height_m"] is None and "altitude_m" in h["basis"]


# --- calibration -----------------------------------------------------------

@case("no calibration file means identity and an explicit 'uncalibrated' basis")
def _():
    assert hv.load_calibration(Path(tempfile.gettempdir()) / "definitely-absent.json") is None
    p, basis = hv.calibrate(0.37, "known", "mine", None)
    assert p == 0.37 and basis["basis"].startswith("uncalibrated")


@case("temperature and Platt scaling apply with model and class precedence")
def _():
    data = {"format": "deepecho-calibration", "format_version": 1, "models": {
        "known": {"method": "temperature", "temperature": 2.0,
                  "classes": {"ship": {"method": "platt", "a": 1.0, "b": 1.0}}},
        "*": {"method": "platt", "a": 1.0, "b": 0.0}}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration.json"
        path.write_text(json.dumps(data))
        cal = hv.load_calibration(path)
    z = math.log(0.9 / 0.1)
    assert abs(cal.apply(0.9, "models/known.pt", "mine") - 1 / (1 + math.exp(-z / 2))) < 1e-9
    assert abs(cal.apply(0.9, "known", "Ship") - 1 / (1 + math.exp(-(z + 1)))) < 1e-9
    assert abs(cal.apply(0.9, "anomaly", "mine") - 0.9) < 1e-9          # "*" platt a=1 b=0
    assert cal.apply(0.5, "known", "mine") == 0.5                        # T leaves 0.5 fixed
    assert cal.params("known", "ship")[1] == "model:known/class:ship"
    # T > 1 pulls confidences toward 0.5 on both sides.
    assert cal.apply(0.2, "known", "mine") > 0.2 and cal.apply(0.8, "known", "mine") < 0.8


@case("a malformed calibration file raises rather than being ignored")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"format": "deepecho-calibration",
                                    "models": {"known": {"method": "isotonic"}}}))
        try:
            hv.load_calibration(path)
        except ValueError:
            return
        raise AssertionError("malformed calibration accepted")


@case("calibrated probability feeds the prior term")
def _():
    image, detections = scene()
    ctx = hv.strip_context(None, None, strip="synthetic", grey=image)
    cal = hv.Calibration(models={"known": {"method": "temperature", "temperature": 3.0}})
    det = next(d for d in detections if d["id"] == "cylinder")
    v = hv.verify_detection(det, ctx, cal)
    expected = 1 / (1 + math.exp(-math.log(0.6 / 0.4) / 3.0))
    assert abs(v["calibrated_probability"] - expected) < 1e-4
    assert v["terms"][0]["value"] == round(expected, 4)
    assert v["detector_confidence"] == 0.6


@case("calibrate.py recovers a known temperature and Platt fit from synthetic scores")
def _():
    import calibrate as cal

    rng = np.random.default_rng(11)
    z = rng.normal(0.5, 1.6, 20000)
    labels = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-z))).astype(float)
    scores = 1 / (1 + np.exp(-2.0 * z))              # overconfident by T = 2
    T = cal.fit_temperature(scores, labels)
    assert abs(T - 2.0) < 0.12, T
    a, b = cal.fit_platt(scores, labels)
    assert abs(a - 0.5) < 0.05 and abs(b) < 0.08, (a, b)
    before = cal.expected_calibration_error(scores, labels)
    after = cal.expected_calibration_error(1 / (1 + np.exp(-np.log(scores / (1 - scores)) / T)), labels)
    assert after < before / 3, (before, after)
    table = cal.reliability_table(scores, labels, bins=10)
    assert sum(row["count"] for row in table) == scores.size


@case("calibrate.py IoU matching: one ground truth absorbs one prediction per class")
def _():
    import calibrate as cal

    preds = [{"cls": 0, "conf": 0.9, "box": [0, 0, 10, 10]},
             {"cls": 0, "conf": 0.8, "box": [1, 1, 11, 11]},      # duplicate -> FP
             {"cls": 1, "conf": 0.7, "box": [0, 0, 10, 10]},      # wrong class -> FP
             {"cls": 0, "conf": 0.6, "box": [50, 50, 60, 60]}]    # nothing there -> FP
    gts = [{"cls": 0, "box": [0, 0, 10, 10]}]
    labels = cal.match_predictions(preds, gts, iou=0.5)
    assert labels == [1, 0, 0, 0], labels


@case("calibrate.py writes a file load_calibration reads back")
def _():
    import calibrate as cal

    rng = np.random.default_rng(5)
    records = []
    for cls in ("ship", "mine"):
        z = rng.normal(0, 1.5, 3000)
        y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-z))).astype(float)
        s = 1 / (1 + np.exp(-1.5 * z))
        records += [{"model": "known", "cls": cls, "score": float(a), "label": float(b)}
                    for a, b in zip(s, y)]
    doc = cal.build_calibration(records, min_per_class=500)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration.json"
        cal.write_calibration(doc, path)
        loaded = hv.load_calibration(path)
    assert loaded is not None and "known" in loaded.models
    entry = loaded.models["known"]
    assert entry["method"] in ("temperature", "platt")
    assert set(entry["classes"]) == {"ship", "mine"}
    assert entry["ece_after"] <= entry["ece_before"]
    assert loaded.apply(0.9, "known", "ship") < 0.9          # overconfident scores pulled in


# --- training scaffold -----------------------------------------------------

@case("augment_sonar paste puts the shadow on the far side, and hazard_verify sees it")
def _():
    import augment_sonar as aug

    rng = np.random.default_rng(9)
    seabed = np.clip(rayleigh(rng, (400, 800), 90.0), 0, 255).astype(np.uint8)
    seabed[:, 370:430] = 5
    crop = np.clip(rayleigh(rng, (20, 50), 200.0), 0, 255).astype(np.uint8)
    for x, far in ((600, +1), (150, -1)):
        image, row = aug.paste_with_shadow(seabed, crop, x, 180, cls=1, rng=rng, nadir_col=400,
                                           height_ratio=1.0, mask=np.ones(crop.shape, bool))
        cls, cx, cy, w, h = row
        assert cls == 1 and abs(cx * 800 - (x + 25)) < 1 and abs(w * 800 - 50) < 1
        beyond = image[182:198, x + 55:x + 90] if far > 0 else image[182:198, x - 40:x - 5]
        near = image[182:198, x - 40:x - 5] if far > 0 else image[182:198, x + 55:x + 90]
        assert beyond.mean() < 0.4 * near.mean(), (x, beyond.mean(), near.mean())
        ctx = hv.strip_context(None, {"nadir_col": 400.0}, strip="p", grey=image.astype(np.float32))
        box = [x, 180, x + 50, 200]
        v = hv.verify_detection({"class": "cylinder", "confidence": 0.6, "bbox_global": box}, ctx, None)
        assert v["evidence"]["acoustic_shadow"]["score"] >= 0.5, v["evidence"]["acoustic_shadow"]
    for fn in (aug.speckle, aug.gain_drift, aug.resolution_jitter):
        out = fn(seabed, rng)
        assert out.shape == seabed.shape and out.dtype == np.uint8


@case("train.py uses sonar augmentation: no hue/saturation jitter, imgsz matches the pipeline")
def _():
    import train

    parser_args = argparse.Namespace(
        data=Path("x.yaml"), epochs=1, imgsz=None, batch=2, patience=1, seed=0, device=None,
        workers=None, project=Path("runs"), name="t", exist_ok=False, no_lr_flip=True,
        no_mosaic=False, set=["lr0=0.005"])
    kwargs = train.train_args(parser_args)
    assert kwargs["hsv_h"] == 0 and kwargs["hsv_s"] == 0 and kwargs["mixup"] == 0
    assert kwargs["fliplr"] == 0.0 and kwargs["flipud"] == 0.5 and kwargs["mosaic"] == 1.0
    assert kwargs["imgsz"] == cfg.DETECTOR_IMGSZ and kwargs["lr0"] == 0.005


# --- real sample surveys ---------------------------------------------------

def real_surveys():
    """(survey, strip image, detections) for every sample survey whose files exist."""
    out = []
    for survey, image in (("waterfall-strip", ROOT / "survey_hazard_map/samples/sidescan-waterfall-strip.jpg"),
                          ("s7-submarine", ROOT / "survey_hazard_map/samples/sidescan-s7-submarine.jpg")):
        export = ROOT / "data/surveys" / survey / "export.json"
        if not export.is_file() or not image.is_file():
            continue
        data = json.loads(export.read_text())
        detections = [{"id": d["id"], "class": d["object_class"],
                       "class_withheld": d.get("class_withheld"),
                       "confidence": d["confidence"],
                       "detector_model": d["provenance"].get("detector_model"),
                       "strip": d["provenance"]["strip"], "bbox_global": d["bbox_global"]}
                      for d in data["detections"]]
        out.append((survey, image, detections))
    return out


def run_real() -> list[str]:
    failures = []
    print("\n  public sample surveys (detections as exported by the pipeline)")
    print(f"    {'detection':42s} {'claimed':10s} {'detector':>8s} {'fused':>7s}  suppressed  hard reasons")
    for survey, image, detections in real_surveys():
        ctx = hv.strip_context(image, None)
        hv.verify_survey(detections, {ctx.strip: ctx}, None)
        for d in detections:
            v = d["verification"]
            print(f"    {d['id']:42s} {v['class_evaluated']:10s} {d['confidence']:8.3f} "
                  f"{d['confidence_pct']:6.1f}%  {str(d['suppressed']):10s}  {', '.join(v['hard_reasons']) or '-'}")
            if VERBOSE:
                for reason in v["reasons"]:
                    print(f"        - {reason}")
            if survey == "waterfall-strip":
                # samples/README.md: every detection on this strip is a false
                # positive. Each must lose confidence and finish below 50%,
                # the point at which it would read as more likely real than
                # not. Suppression additionally needs a hard artefact reason,
                # which a box on a display annotation does not have.
                if d["confidence_pct"] >= min(50.0, 100 * d["confidence"]):
                    failures.append(f"{d['id']}: false positive not down-weighted below 50%")
            if survey == "s7-submarine" and (d["confidence_pct"] < 70 or d["suppressed"]):
                failures.append(f"{d['id']}: true positive S-7 wreck lost confidence")
    return failures


def main() -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--real", action="store_true",
                        help="also verify the public sample surveys in data/surveys")
    args = parser.parse_args()
    VERBOSE = args.verbose

    failures = []
    for name, fn in CASES:
        try:
            fn()
            if args.verbose:
                print(f"  ok    {name}")
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {str(exc)[:600]}")

    if args.real:
        for failure in run_real():
            failures.append(("real survey", failure))
            print(f"  FAIL  real survey: {failure}")

    print()
    total = len(CASES) + (1 if args.real else 0)
    if failures:
        print(f"FAILED  {len(failures)} failure(s) across {total} verification tests")
        return 1
    print(f"PASSED  {len(CASES)}/{len(CASES)} verification tests" + (" + real surveys" if args.real else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
