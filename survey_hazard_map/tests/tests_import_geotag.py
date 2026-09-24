#!/usr/bin/env python3
"""Tests for import_geotag.py: the teammate's geotag.py output as a DeepEcho survey.

    .venv/bin/python tests_import_geotag.py

Plain functions and asserts, with an exit code. No torch and no network. The
sonar is geotag.write_synthetic_xtf's, read back with geotag.read_xtf, and the
"detector" is the blob threshold geotag's own selftest uses, labelled with the
trained model's class names by nearest planted object. hazards.json is written
by geotag.geotag + geotag.write_report exactly as sonar_pipeline.py writes it.
Everything lands in a temporary directory; data/surveys is never touched, and
nothing is written into the teammate's models folder (geotag is imported with
bytecode writing off).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = Path(tempfile.mkdtemp(prefix="deepecho-import-geotag-"))
SURVEYS = WORK / "surveys"
INPUTS = WORK / "inputs"
os.environ["DEEPECHO_SURVEYS_DIR"] = str(SURVEYS)
os.environ.pop("DEEPECHO_ENABLE_GHOSTTRACE_RUN", None)

import numpy as np  # noqa: E402

from survey_hazard_map import import_geotag as ig  # noqa: E402
from survey_hazard_map.validate_output import Report, validate  # noqa: E402

gt = ig.load_geotag()

SPEED = 2.0   # write_synthetic_xtf's default towfish speed, m/s

# The three selftest objects plus one at 3 m ground range, near nadir. `conf`
# is the calibrated detector probability and `shadow` what shadow_check.py
# would report; sonar_pipeline.py fuses them as conf * (0.6 + 0.4 * score).
OBJECTS = [
    dict(ping=80, side="stbd", ground_range_m=22.0, len_pings=12, width_m=2.5,
         label="mine_like_object", conf=0.93, shadow=(0.82, "physics-consistent", 0.6)),
    dict(ping=200, side="port", ground_range_m=35.0, len_pings=40, width_m=6.0,
         label="shipwreck", conf=0.90, shadow=(0.70, "physics-consistent", 1.2)),
    dict(ping=330, side="stbd", ground_range_m=41.0, len_pings=8, width_m=1.5,
         label="fishing_gear", conf=0.80, shadow=(0.40, "weak-shadow", None)),
    # Low confidence and no shadow: sonar_pipeline.py vetoes this one.
    dict(ping=140, side="port", ground_range_m=3.0, len_pings=20, width_m=2.0,
         label="mine_like_object", conf=0.45, shadow=(0.10, "no-shadow", None)),
]
ANOMALY_BOX = [850, 250, 880, 270]


def _write_objects(objects):
    return [{k: v for k, v in o.items() if k in ("ping", "side", "ground_range_m", "len_pings",
                                                  "width_m")} for o in objects]


def blob_boxes(img):
    """geotag.selftest's blob detector: it tests geometry, not a model."""
    import cv2

    mask = (img > 235).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, a in stats[1:] if a > 20]


def label_boxes(boxes, nav, truth, objects, *, shadow=True):
    """Detections labelled by nearest planted object, as the pipeline would pass them."""
    probe = gt.geotag([dict(label="o", confidence_pct=1.0, box_xyxy_px=b) for b in boxes], nav, "x")
    dets = []
    for box, located in zip(boxes, probe):
        i = min(range(len(truth)), key=lambda k: gt.GEOD.inv(
            located["lon"], located["lat"], truth[k]["lon"], truth[k]["lat"])[2])
        obj = objects[i]
        det = dict(label=obj["label"], box_xyxy_px=box, _truth=i)
        if shadow:
            score, verdict, height = obj["shadow"]
            det.update(shadow_score=score, verdict=verdict, height_m=height,
                       confidence_pct=round(100 * obj["conf"] * (0.6 + 0.4 * score), 1),
                       vetoed=verdict == "no-shadow" and obj["conf"] < 0.6)
        else:
            det.update(confidence_pct=round(100 * obj["conf"], 1))
        dets.append(det)
    return dets


def pipeline_output(folder, name, img, nav, dets, *, keep_veto_flag=True):
    """hazards.json + <name>_waterfall.png exactly as sonar_pipeline.py writes them.

    geotag.geotag does not carry `vetoed` (and the pipeline drops vetoed boxes),
    so keep_veto_flag adds it back, simulating the requested upstream change.
    """
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{name}_waterfall.png"), img)
    clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in dets]
    hazards = gt.geotag(clean, nav, name)
    gt.write_report(hazards, nav, folder, img, name)
    path = folder / "hazards.json"
    if keep_veto_flag:
        data = json.loads(path.read_text())
        for record, det in zip(data["hazards"], clean):
            if "vetoed" in det:
                record["vetoed"] = det["vetoed"]
        path.write_text(json.dumps(data, indent=2))
    return path


def strict_valid(survey):
    report = Report()
    validate(Path(survey), report)
    assert not report.failures and not report.warnings, (report.failures, report.warnings)


# --- fixtures, built once ----------------------------------------------------------

_CACHE: dict = {}


def fixture_xtf():
    if "xtf" in _CACHE:
        return _CACHE["xtf"]
    folder = INPUTS / "line01"
    folder.mkdir(parents=True, exist_ok=True)
    xtf = folder / "line01.xtf"
    truth = gt.write_synthetic_xtf(xtf, n_pings=400, objects=_write_objects(OBJECTS))
    img, nav = gt.read_xtf(xtf)
    dets = label_boxes(blob_boxes(img), nav, truth, OBJECTS)
    assert len(dets) == len(OBJECTS), f"blob detector found {len(dets)} of {len(OBJECTS)}"
    dets.append(dict(label="unknown_anomaly", anomaly_score=0.93, confidence_pct=93.0,
                     box_xyxy_px=ANOMALY_BOX))
    hazards = pipeline_output(folder, "line01", img, nav, dets)
    out = SURVEYS / "geotag-line01"
    export = ig.import_geotag(hazards, out, xtf=xtf, verify="auto")
    _CACHE["xtf"] = dict(folder=folder, xtf=xtf, truth=truth, img=img, nav=nav, dets=dets,
                         hazards=hazards, out=out, export=export)
    return _CACHE["xtf"]


def _truth_for(fx, detection):
    record = next(r for r in json.loads(fx["hazards"].read_text())["hazards"]
                  if r["id"] == detection["provenance"]["geotag"]["geotag_id"])
    index = next(i for i, d in enumerate(fx["dets"]) if d["box_xyxy_px"] == record["box_xyxy_px"])
    return fx["dets"][index].get("_truth"), record


# --- vocabulary ----------------------------------------------------------------------


def test_hazard_vocabulary():
    from survey_hazard_map.hazard_severity import class_weight, recommended_action
    from survey_hazard_map.hazard_verify import class_expectation

    assert class_weight("mine_like_object") == (1.0, "exact")
    action, how = recommended_action("mine_like_object")
    assert action == "Keep clear; send for expert identification as possible ordnance", action
    assert how == "exact" and "EOD" not in action
    # The plain mine keeps its own action: the exact key is what separates them.
    assert recommended_action("mine")[0] == "Deploy EOD team"
    assert recommended_action("fishing_gear") == ("Schedule ghost-gear recovery", "exact")
    assert class_weight("fishing_gear")[0] == 0.5
    assert recommended_action("pipeline")[0] == "Check against charted pipelines; flag if uncharted"
    assert class_weight("pipeline")[0] == 0.5
    assert class_weight("unknown_anomaly") == (0.8, "exact")
    assert recommended_action("unknown_anomaly")[0] == "Send for expert identification"
    for name in ("shipwreck", "aircraft", "human"):
        assert class_weight(name)[1] == "exact", name
    assert class_expectation("fishing-gear")[0]["shadow"] == 0.2
    assert class_expectation("mine-like-object")[0]["shadow"] == 1.0


def test_ghosttrace_vocabulary():
    from ghosttrace import changes
    from ghosttrace.engine import select_targets

    assert changes.is_target_class("fishing_gear") and changes.is_target_class("fishing-gear")
    assert changes.class_family("fishing_gear") == "gear"
    assert not changes.is_target_class("mine_like_object")
    chosen, _ = select_targets([{"id": "a", "class_normalized": "fishing-gear"},
                                {"id": "b", "class_normalized": "shipwreck"}])
    assert [d["id"] for d in chosen] == ["a"]


def test_assistant_vocabulary():
    from rag_assistant import chat
    from backend import config

    for raw in ("mine_like_object", chat._normalise("mine_like_object")):
        assert config.DETECTOR_CLASS_MAP[raw] == "suspected mine-like object"
    label = config.DETECTOR_CLASS_MAP[chat._normalise("mine_like_object")]
    assert label not in ("mine", "naval mine", "sea mine")
    record = {"label": label, "confidence": 0.9}
    assert chat.severity_for(record, False) == "high"
    assert not chat.coverage_gap(record)
    assert "unidentified" in chat.class_synonyms(label) and "mine" in chat.class_synonyms(label)
    # Classified, so the explain brief applies, with the note that a classifier
    # output is not a confirmed identification.
    task = chat.build_task("explain", "Brief me on this contact", record, "", [])
    assert "not a confirmed identification" in task

    fishing = config.DETECTOR_CLASS_MAP[chat._normalise("fishing_gear")]
    assert fishing == "derelict fishing gear" and fishing in config.CLASS_SYNONYMS
    assert config.DETECTOR_CLASS_MAP["fishing_gear"] == fishing
    assert chat.severity_for({"label": fishing}, False) == "medium"
    assert not chat.coverage_gap({"label": fishing})

    unknown = config.DETECTOR_CLASS_MAP[chat._normalise("unknown_anomaly")]
    assert unknown == "unknown" and config.DETECTOR_CLASS_MAP["unknown_anomaly"] == "unknown"
    assert chat.route_intent("Brief me on this contact", {"label": unknown}) == "anomaly"
    assert chat.severity_for({"label": unknown}, True) == "unknown"
    for raw in ("pipeline", "shipwreck", "aircraft", "human"):
        assert raw in config.DETECTOR_CLASS_MAP, raw


def test_assistant_retrieval_routes_mine_like():
    from rag_assistant import chat
    from backend import config

    record = {"label": config.DETECTOR_CLASS_MAP["mine_like_object"], "confidence": 0.9}
    query = chat.condense("Brief me on this contact", [], record, "explain")
    hits = chat.get_retriever().search(query, k=config.TOP_K, per_doc=config.PER_DOC)
    docs = {chunk.id.split("#")[0] for chunk, _ in hits}
    assert {"naval-mine-identification", "unidentified-object-protocol"} <= docs, docs

    record = {"label": config.DETECTOR_CLASS_MAP["fishing_gear"], "confidence": 0.9}
    query = chat.condense("Brief me on this contact", [], record, "explain")
    hits = chat.get_retriever().search(query, k=config.TOP_K, per_doc=config.PER_DOC)
    assert "ghost-gear-reporting-india" in {c.id.split("#")[0] for c, _ in hits}


# --- the XTF path ----------------------------------------------------------------------


def test_xtf_import_is_a_valid_survey():
    fx = fixture_xtf()
    strict_valid(fx["out"])
    for name in ("export.json", "actions.csv", "report.csv", "report.geojson", "map.html",
                 "manifest.json", "manifest.csv", "ghosttrace.json", "ghosttrace.geojson"):
        assert (fx["out"] / name).is_file(), name
    export = fx["export"]
    assert export["metadata"]["coordinate_mode"] == "Geo-referenced"
    assert export["survey_summary"]["georeferenced"] is True
    assert export["metadata"]["demo"] is False and export["provenance"]["demo"] is False
    assert "/Users/" not in json.dumps(export) and "/home/" not in json.dumps(export)
    assert all(d["provenance"]["detector_model"] == "geotag:sonar_detector"
               for d in export["detections"])


def test_positions_are_geotags_unchanged():
    fx = fixture_xtf()
    records = {r["id"]: r for r in json.loads(fx["hazards"].read_text())["hazards"]}
    ids = set()
    for det in fx["export"]["detections"]:
        record = records[det["provenance"]["geotag"]["geotag_id"]]
        assert det["latitude"] == record["lat"] and det["longitude"] == record["lon"], det["id"]
        assert det["id"] == f"line01_waterfall_g{record['id']}"
        assert det["bbox_global"] == [float(v) for v in record["box_xyxy_px"]]
        assert det["provenance"]["geotag"]["ping_time"].endswith("Z")
        ids.add(det["id"])
    assert len(ids) == len(records)
    # Hotspot centroids are means of member positions, suppressed excluded.
    by_id = {d["id"]: d for d in fx["export"]["detections"]}
    for hotspot in fx["export"]["hotspots"]:
        members = [by_id[i] for i in hotspot["detection_ids"]]
        lat = sum(m["latitude"] for m in members) / len(members)
        assert abs(hotspot["centroid"]["latitude"] - lat) < 1e-7


def test_sizes_are_geotags_and_close_to_planted():
    fx = fixture_xtf()
    rate = 5.0
    near_nadir = None
    for det in fx["export"]["detections"]:
        truth_index, record = _truth_for(fx, det)
        dims = det["dimensions"]
        assert dims["width_m"] == record["width_m"] and dims["length_m"] == record["length_m"]
        if truth_index is None:
            continue
        assert dims["basis"] == ig.DIMENSION_BASIS
        obj = OBJECTS[truth_index]
        true_len = obj["len_pings"] * SPEED / rate
        # Near nadir a pixel spans a lot of ground: 25% or 0.5 m.
        assert abs(dims["width_m"] - obj["width_m"]) <= max(0.5, 0.25 * obj["width_m"]), (
            obj["label"], obj["ground_range_m"], dims["width_m"], obj["width_m"])
        assert dims["length_m"] > 0
        assert abs(dims["length_m"] - true_len) <= max(0.5, 0.25 * true_len), (
            dims["length_m"], true_len)
        if obj["ground_range_m"] == 3.0:
            near_nadir = dims
    assert near_nadir is not None
    cross = fx["export"]["provenance"]["import"]["dimension_cross_check"]
    assert cross["ran"] is True and cross["disagreements"] == [], cross["disagreements"]


def test_theirs_verification_and_veto():
    fx = fixture_xtf()
    export = fx["export"]
    verification = export["provenance"]["verification"]
    assert verification["path"] == "theirs" and verification["suppressed"] == 1
    vetoed = [d for d in export["detections"] if d["suppressed"]]
    assert len(vetoed) == 1
    det = vetoed[0]
    assert OBJECTS[_truth_for(fx, det)[0]]["ground_range_m"] == 3.0
    assert det["verification"]["vetoed"] is True
    assert det["verification"]["source"] == "shadow_check.py (teammate)"
    grouped = {i for h in export["hotspots"] for i in h["detection_ids"]}
    assert det["id"] not in grouped
    for d in export["detections"]:
        record = _truth_for(fx, d)[1]
        assert d["confidence_pct"] == record["confidence_pct"], d["id"]
    # Never both: nothing from hazard_verify's own fusion is present.
    assert all("terms" not in (d.get("verification") or {}) for d in export["detections"])


def test_verdict_alone_suppresses_without_veto_flag():
    fx = fixture_xtf()
    folder = INPUTS / "line01-noflag"
    folder.mkdir(parents=True, exist_ok=True)
    data = json.loads(fx["hazards"].read_text())
    for record in data["hazards"]:
        record.pop("vetoed", None)
    (folder / "hazards.json").write_text(json.dumps(data))
    shutil.copy(fx["folder"] / "line01_waterfall.png", folder / "line01_waterfall.png")
    export = ig.import_geotag(folder / "hazards.json", SURVEYS / "geotag-noflag", xtf=fx["xtf"],
                              ghosttrace=False)
    suppressed = [d for d in export["detections"] if d["suppressed"]]
    assert len(suppressed) == 1 and suppressed[0]["verification"]["verdict"] == "no-shadow"
    assert "no vetoed flag" in suppressed[0]["verification"]["suppression_basis"]
    strict_valid(SURVEYS / "geotag-noflag")


def test_confident_no_shadow_is_kept_without_veto_flag():
    """The brief's hard veto is no-shadow AND confidence < 0.6. A confident call
    with no shadow -- a flush pipe, a draped net -- must stay reported."""
    fx = fixture_xtf()
    folder = INPUTS / "line01-confident"
    folder.mkdir(parents=True, exist_ok=True)
    data = json.loads(fx["hazards"].read_text())
    for record in data["hazards"]:
        record.pop("vetoed", None)
        if record.get("verdict") == "no-shadow":
            record["confidence_pct"] = 85.0
    (folder / "hazards.json").write_text(json.dumps(data))
    shutil.copy(fx["folder"] / "line01_waterfall.png", folder / "line01_waterfall.png")
    export = ig.import_geotag(folder / "hazards.json", SURVEYS / "geotag-confident", xtf=fx["xtf"],
                              ghosttrace=False)
    no_shadow = [d for d in export["detections"]
                 if (d.get("verification") or {}).get("verdict") == "no-shadow"]
    assert no_shadow and not any(d["suppressed"] for d in no_shadow), no_shadow
    assert all("0.60" in d["verification"]["suppression_basis"] for d in no_shadow)
    strict_valid(SURVEYS / "geotag-confident")


def test_classes_actions_and_anomaly():
    fx = fixture_xtf()
    export = fx["export"]
    mine = [d for d in export["detections"] if d["object_class"] == "mine_like_object"]
    assert mine and all(d["recommended_action"] ==
                        "Keep clear; send for expert identification as possible ordnance"
                        for d in mine)
    anomaly = [d for d in export["detections"]
               if d["provenance"]["geotag"]["classification"] == "unknown_anomaly"]
    assert len(anomaly) == 1
    anomaly = anomaly[0]
    assert anomaly["object_class"] == "unknown" and anomaly["class_weight"] == 0.8
    assert anomaly["confidence_pct"] == 93.0 and "anomaly_score" in anomaly["confidence_pct_basis"]
    assert anomaly["verification"]["anomaly_score"] == 0.93
    with (fx["out"] / "actions.csv").open() as handle:
        actions = list(csv.DictReader(handle))
    assert all("EOD" not in row["recommended_action"] for row in actions)


def test_sidecar_is_slant_and_not_a_georeference():
    fx = fixture_xtf()
    out = fx["out"]
    manifest = json.loads((out / "manifest.json").read_text())
    navigation = manifest["survey"]["navigation"]
    assert navigation["mode"] == ig.NAV_MODE != "ping"
    sidecar = json.loads((out / navigation["sidecars"]["line01_waterfall"]).read_text())
    assert sidecar["format"] == "deepecho-strip-nav/1"
    assert sidecar["slant_range_corrected"] is False and sidecar["m_per_px_across"] is None
    assert abs(sidecar["slant_m_per_sample"] - 50.0 / 512) < 1e-6
    assert abs(sidecar["m_per_px_along"] - SPEED / 5.0) < 0.01
    assert sidecar["nadir_col"] == fx["nav"].nadir_col and sidecar["port_is_left"] is True
    assert len(sidecar["rows"]) == fx["img"].shape[0]
    row = sidecar["rows"][10]
    assert row["time"].endswith("Z") and row["quality"] == "ok" and row["depth_m"] is None
    for key in ("lat", "lon", "heading_deg", "altitude_m", "roll_deg", "pitch_deg", "heave_m",
                "seabed_depth_m"):
        assert key in row, key
    # Nothing downstream turns the sidecar into a pixel-locating transform.
    from survey_hazard_map.hazard_geo import references_for_survey
    references, _ = references_for_survey(manifest["tiles"], manifest["survey"], out)
    assert all(ref.mode != "ping" for ref in references.values())
    # Tile rows carry geotag geometry positions.
    assert manifest["tiles"] and all(t["lat"] is not None for t in manifest["tiles"])
    # Water column in watercolumn.py's format.
    from ghosttrace.watercolumn import locate_sidecars
    nav_path, wc_path = locate_sidecars(out, "line01_waterfall")
    assert nav_path is not None and wc_path is not None
    with np.load(wc_path) as npz:
        port, bottom = npz["port"], npz["bottom_range_m"]
        assert npz["starboard"].shape == port.shape and port.shape[0] == fx["img"].shape[0]
    m_per_bin = sidecar["water_column"]["m_per_bin"]
    assert np.allclose(bottom, 8.0)
    last_valid = (port.shape[1] - 1) * m_per_bin
    assert np.isnan(port[:, int(8.0 / m_per_bin) + 1:]).all() and last_valid < 8.0 + m_per_bin
    assert "AS IS" in sidecar["water_column"]["note"]


def test_ghosttrace_runs_on_the_import():
    fx = fixture_xtf()
    doc = json.loads((fx["out"] / "ghosttrace.json").read_text())
    classes = {t["object_class"] for t in doc["targets"]}
    assert "fishing_gear" in classes, classes
    for target in doc["targets"]:
        activity = target["activity"]
        assert activity["available"] or activity.get("reason"), activity
        assert target["latitude"] is not None
    assert doc["run"]["survey_time"].startswith("2026-01-01T08:00:00")
    assert "navigation sidecars" in doc["run"]["survey_time_basis"]
    assert doc["demo"] is False and not doc["run"]["failures"], doc["run"]["failures"]


def test_report_csv_has_positions_and_sizes():
    fx = fixture_xtf()
    with (fx["out"] / "report.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(fx["export"]["detections"])
    for row in rows:
        assert row["latitude"] and row["longitude"] and row["length_m"] and row["width_m"]
        assert "geotag.py" in row["dimension_basis"]
    geojson = json.loads((fx["out"] / "report.geojson").read_text())
    assert geojson["properties"]["detections_located"] == len(rows)


def test_map_does_not_contradict_the_export():
    from survey_hazard_map.hazard_theme import TEXT

    fx = fixture_xtf()
    html = (fx["out"] / "map.html").read_text()
    assert TEXT["relative_note"] not in html and TEXT["relative_mode"] not in html
    assert TEXT["geo_mode"] in html


def test_overwrite_rules():
    fx = fixture_xtf()
    try:
        ig.import_geotag(fx["hazards"], fx["out"], xtf=fx["xtf"], ghosttrace=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("a second import into a non-empty directory was not refused")
    assert ig.main([str(fx["hazards"]), str(fx["out"]), "--xtf", str(fx["xtf"]), "--quiet",
                    "--no-ghosttrace"]) == 1
    # Never deletes a folder holding the inputs.
    try:
        ig.import_geotag(fx["hazards"], fx["folder"], xtf=fx["xtf"], overwrite=True)
    except (ValueError, FileExistsError):
        pass
    else:
        raise AssertionError("overwrite deleted the input folder")
    assert fx["hazards"].is_file()
    again = SURVEYS / "geotag-line01-again"
    ig.import_geotag(fx["hazards"], again, xtf=fx["xtf"], ghosttrace=False)
    export = ig.import_geotag(fx["hazards"], again, xtf=fx["xtf"], overwrite=True)
    assert [d["id"] for d in export["detections"]] == [d["id"] for d in fx["export"]["detections"]]
    strict_valid(again)


def test_http_surveys_and_ghosttrace():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ghosttrace.routes import api as gt_routes
    from survey_hazard_map.routes import survey as survey_routes

    fx = fixture_xtf()
    assert Path(survey_routes.SURVEYS_DIR) == SURVEYS and Path(gt_routes.SURVEYS_DIR) == SURVEYS
    app = FastAPI()
    app.include_router(survey_routes.router)
    app.include_router(gt_routes.router)
    client = TestClient(app)
    listing = client.get("/survey").json()["surveys"]
    entry = next(s for s in listing if s["survey_id"] == fx["out"].name)
    assert entry["georeferenced"] and entry["has_map"] and entry["has_ghosttrace"]
    assert entry["demo"] is False
    response = client.get(f"/ghosttrace/{fx['out'].name}")
    assert response.status_code == 200 and response.json()["targets"]
    assert client.get(f"/survey/{fx['out'].name}/report.csv").status_code == 200
    assert client.get(f"/survey/{fx['out'].name}/export").json()["detections"]


# --- repeated fixes, heading 0, our verification --------------------------------------


def test_repeated_fixes_nav_csv_and_our_verification():
    folder = INPUTS / "line02"
    folder.mkdir(parents=True, exist_ok=True)
    objects = [
        dict(ping=100, side="port", ground_range_m=3.0, len_pings=20, width_m=2.0,
             label="mine_like_object", conf=0.7, shadow=None),
        dict(ping=220, side="stbd", ground_range_m=11.0, len_pings=30, width_m=2.0,
             label="fishing_gear", conf=0.8, shadow=None),
        dict(ping=340, side="stbd", ground_range_m=31.0, len_pings=10, width_m=2.0,
             label="shipwreck", conf=0.85, shadow=None),
    ]
    xtf = folder / "line02.xtf"
    truth = gt.write_synthetic_xtf(xtf, heading=0.0, rate=10.0, gps_hz=1.0,
                                   objects=_write_objects(objects))
    img, nav_xtf = gt.read_xtf(xtf)
    # A navigation CSV with the fixes repeated 10 pings at a time, as a 1 Hz GPS
    # logged under a 10 Hz sonar looks before any interpolation.
    csv_path = folder / "nav.csv"
    nav_xtf.to_csv(csv_path)
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        source = rows[(i // 10) * 10]
        row["lat"], row["lon"] = source["lat"], source["lon"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    nav = gt.NavTable.from_csv(csv_path, nav_xtf.nadir_col)
    dets = label_boxes(blob_boxes(img), nav, truth, objects, shadow=False)
    hazards = pipeline_output(folder, "line02", img, nav, dets)
    records = json.loads(hazards.read_text())["hazards"]
    assert all(r["length_m"] > 0 for r in records), [r["length_m"] for r in records]

    out = SURVEYS / "geotag-line02"
    export = ig.import_geotag(hazards, out, nav_csv=csv_path, sensor_depth_m=12.0)
    strict_valid(out)
    assert export["provenance"]["verification"]["path"] == "ours"
    assert "auto" in export["provenance"]["verification"]["choice"]
    for det in export["detections"]:
        assert det["dimensions"]["length_m"] > 0
        assert det["verification"]["source"] == "hazard_verify.py (DeepEcho)"
        assert det["dimensions"]["height_m"] is None
        assert "slant-range" in det["dimensions"]["height_basis"]
        assert "shadow_score" not in det["verification"]      # never both
        assert det["verification"]["status"] == "checked"
    # hazard_verify's own nadir cue: the 3 m object's box reaches into the water
    # column of this slant-range strip, so it is filtered, with the reason kept.
    suppressed = [d for d in export["detections"] if d["suppressed"]]
    assert all("nadir_zone" in d["verification"]["hard_reasons"] for d in suppressed)
    assert all(d["object_class"] == "mine_like_object" for d in suppressed)
    sidecar = json.loads((out / "nav" / "line02_waterfall.nav.json").read_text())
    assert abs(sidecar["m_per_px_along"] - SPEED / 10.0) < 0.02, sidecar["m_per_px_along"]
    assert sidecar["rows"][5]["depth_m"] == 12.0
    assert sidecar["rows"][5]["seabed_depth_m"] == 20.0
    assert sidecar["rows"][5]["roll_deg"] is None            # not in NavTable.to_csv
    assert any("interpolated" in n for n in sidecar["processing"]["notes"])
    doc = json.loads((out / "ghosttrace.json").read_text())
    gear = [t for t in doc["targets"] if t["object_class"] == "fishing_gear"]
    assert gear and gear[0]["seabed_depth_m"] == 20.0


def test_dropout_rows_become_degraded():
    nav = gt.simulated_nav(50, 100, 100)
    for i in (20, 21, 22, 40):
        nav.rows[i]["gap_before"] = True
    sidecar, _ = ig.build_sidecar(gt, nav, strip="s", width=200, height=50, source_file="x",
                                  source_format="nav_csv", simulated=True, sensor_depth_m=None)
    assert sidecar["degraded_rows"] == [[20, 22, "dropout"], [40, 40, "dropout"]]
    assert sidecar["rows"][21]["quality"] == "dropout" and sidecar["rows"][23]["quality"] == "ok"


def test_verification_choice_rule():
    shadow = [{"shadow_score": 0.5}]
    plain = [{"confidence_pct": 50}]
    assert ig.choose_verification("auto", shadow, True)[0] == "theirs"
    assert ig.choose_verification("auto", plain, True)[0] == "ours"
    assert ig.choose_verification("auto", plain, False)[0] == "none"
    assert ig.choose_verification("none", shadow, True)[0] == "none"
    try:
        ig.choose_verification("both", shadow, True)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown verification path was accepted")


# --- simulated navigation ------------------------------------------------------------


def test_simulated_navigation_is_demo():
    fx = fixture_xtf()
    folder = INPUTS / "simulated"
    img = fx["img"]
    nav = gt.simulated_nav(img.shape[0], img.shape[1] // 2, img.shape[1] // 2)
    dets = [dict(label="shipwreck", confidence_pct=80.0, box_xyxy_px=b) for b in blob_boxes(img)]
    hazards = pipeline_output(folder, "tile", img, nav, dets)
    meta = json.loads(hazards.read_text())["meta"]
    assert meta["simulated_navigation"] is True
    nav.to_csv(folder / "nav.csv")
    out = SURVEYS / "geotag-simulated"
    export = ig.import_geotag(hazards, out, nav_csv=folder / "nav.csv", verify="none",
                              ghosttrace=False)
    strict_valid(out)
    assert export["metadata"]["demo"] is True and export["provenance"]["demo"] is True
    assert "SIMULATED" in export["metadata"]["data_source"]
    assert "SIMULATED" in export["metadata"]["demo_warning"]
    assert any("SIMULATED" in w for w in export["metadata"]["warnings"])
    sidecar = json.loads((out / "nav" / "tile_waterfall.nav.json").read_text())
    assert sidecar["synthetic"] is True
    assert all(d["confidence_pct_basis"].endswith("not verified") for d in export["detections"])


# --- no navigation at all ---------------------------------------------------------------


def test_hazards_json_only():
    fx = fixture_xtf()
    folder = INPUTS / "hazards-only"
    folder.mkdir(parents=True, exist_ok=True)
    data = json.loads(fx["hazards"].read_text())
    for record in data["hazards"]:
        for key in ("shadow_score", "verdict", "vetoed"):
            record.pop(key, None)
    (folder / "hazards.json").write_text(json.dumps(data))
    out = SURVEYS / "geotag-hazards-only"
    export = ig.import_geotag(folder / "hazards.json", out)
    strict_valid(out)
    assert export["survey_summary"]["coordinate_mode"] == "Geo-referenced"
    assert export["provenance"]["verification"]["path"] == "none"
    assert not (out / "nav").exists() and not (out / "tiles").exists()
    records = {r["id"]: r for r in data["hazards"]}
    for det in export["detections"]:
        record = records[det["provenance"]["geotag"]["geotag_id"]]
        assert (det["latitude"], det["longitude"]) == (record["lat"], record["lon"])
        assert det["dimensions"]["basis"] == ig.DIMENSION_BASIS_UNCHECKED
        assert det["provenance"]["representative_tile"] is None
    assert "not written" in export["provenance"]["import"]["water_column"]
    assert export["provenance"]["import"]["dimension_cross_check"]["ran"] is False
    doc = json.loads((out / "ghosttrace.json").read_text())
    gear = [t for t in doc["targets"] if t["object_class"] == "fishing_gear"]
    assert gear and gear[0]["activity"]["available"] is False
    assert "no water-column record" in gear[0]["activity"]["reason"]
    from survey_hazard_map.hazard_theme import TEXT
    html = (out / "map.html").read_text()
    assert TEXT["relative_note"] not in html and TEXT["geo_mode"] in html


def test_pipeline_without_navigation():
    """sonar_pipeline.py with no nav writes lat/lon null and no sizes."""
    folder = INPUTS / "no-nav"
    folder.mkdir(parents=True, exist_ok=True)
    hazards = [dict(id=1, image="tile", classification="shipwreck", confidence_pct=77.0,
                    box_xyxy_px=[10, 10, 60, 40], shadow_score=None, verdict=None, height_m=None,
                    anomaly_score=None, lat=None, lon=None,
                    navigation="none — no .xtf / nav file supplied")]
    (folder / "hazards.json").write_text(json.dumps({"meta": {"source": "/Users/someone/tile.png",
                                                              "n_hazards": 1},
                                                     "hazards": hazards}))
    out = SURVEYS / "geotag-no-nav"
    export = ig.import_geotag(folder / "hazards.json", out, ghosttrace=False)
    strict_valid(out)
    assert export["survey_summary"]["coordinate_mode"] == "Relative Survey Coordinates"
    det = export["detections"][0]
    assert det["latitude"] is None and det["dimensions"]["basis"] == ig.DIMENSION_BASIS_NONE
    assert "/Users/" not in (out / "export.json").read_text()


# --- runner --------------------------------------------------------------------------------


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    try:
        for name, fn in tests:
            try:
                fn()
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
            else:
                print(f"ok    {name}")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
