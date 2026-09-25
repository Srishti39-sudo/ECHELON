# Final model — `final.pt` (marine_v2: YOLO11s, 7 classes)

`final.pt` = `marine.pt` (6-class YOLO11s, best epoch 51/76) fine-tuned 30 epochs with a synthetic **ghost_net** class
(procedurally rendered nets with acoustic shadows composited on real seabed tiles). 9.4 M params, 19 MB, 640-px tiles, 128-px overlap.
Classes (index order): shipwreck, aircraft, human, pipeline, fishing_gear, mine_like_object, ghost_net.
Ship with `calibration.json` in this folder (identity: raw confidence is already calibrated, ECE 0.038; Platt scaling made it worse, 0.043).

All numbers below are measured on held-out test tiles never used in training, confidence 0.25, IoU 0.5.
Old models can only be scored on the classes they know.

## 1. final.pt vs marine.pt — same 1,533 test tiles (2026-09-25)

| class | **final.pt** AP50 (P / R) | marine.pt AP50 (P / R) | change |
|---|---|---|---|
| shipwreck | **0.58** (0.69 / 0.54) | 0.53 (0.62 / 0.53) | +0.05 |
| aircraft | 0.88 (0.82 / 0.90) | **0.90** (1.00 / 0.90) | −0.02 (noise) |
| human (3 test objects) | 0.33 (1.00 / 0.33) | 0.33 (1.00 / 0.33) | 0 |
| pipeline | **0.99** (0.98 / 0.99) | 0.99 (0.94 / 0.99) | 0 |
| fishing_gear | 0.25 (0.37 / 0.27) | **0.33** (0.41 / 0.31) | −0.08 (see §5) |
| mine_like_object | **0.66** (0.69 / 0.62) | 0.60 (0.69 / 0.61) | +0.06 |
| ghost_net (synthetic held-out regime, 648 boxes) | **0.93** (0.89 / 0.90) | — no class | new |
| false alarms per empty seafloor tile | **0.26** | 0.30 | better |
| Ultralytics validator, all classes | **mAP50 0.655 · mAP50-95 0.466** (7 cls) | mAP50 0.605 · mAP50-95 0.408 (6 cls, 1,106-tile v1 split) | |

## 2. marine.pt vs the original models — 1,106-tile v1 test split (2026-09-15)

| class | marine.pt | known.pt (YOLOv8s, SCTD) | anomaly.pt (YOLOv8n, Roboflow) |
|---|---|---|---|
| pipeline | **0.99** | — no class | — no class |
| aircraft | **0.90** (1.00 / 0.90) | 0.74 (0.04 / 1.00) | 0.56 (0.04 / 0.80) |
| mine_like_object | **0.60** | — no class | — no class |
| shipwreck | **0.52** (0.62 / 0.48) | 0.20 (0.42 / 0.21) | 0.22 (0.55 / 0.22) |
| fishing_gear | **0.34** (0.41 / 0.31) | — no class | — no class |
| human | 0.33 | 1.00 | 0.00 |
| false alarms per empty tile | 0.30 | 0.06 | 0.31 |

The old models fire on almost nothing (aircraft precision 0.04 = ~25 false boxes per real plane), which is also why `known.pt` has few false alarms.

## 3. Architecture benchmark — crab pots only, 365 test tiles / 567 boxes (2026-09-25)

| weights | AP50 | recall | false alarms / empty tile | CPU ms / tile |
|---|---|---|---|---|
| yolo26s, 1-class specialist | 0.362 | 0.32 | 0.90 | 330 |
| yolo11s, 1-class specialist | **0.366** | 0.25 | **0.39** | **241** |
| marine.pt, 6-class | 0.337 | 0.31 | 0.58 | 241 |

Conclusion: architecture is not the bottleneck for fishing gear; YOLO26 and single-class training give no real gain.
Large-target recall stays 0.24–0.37 for every model → the source labels (pot clusters / strings boxed as one object) are the limit.

## 4. Calibration and speed

| | ECE raw | ECE Platt | shipped |
|---|---|---|---|
| marine.pt | 0.041 | 0.050 | identity |
| final.pt | **0.038** | 0.043 | identity |

Speed: 11.3 ms inference per tile on a Tesla T4; ~0.5 s per tile on Apple-silicon CPU; 145-Mpx survey mosaic (558 tiles) ≈ 11 min on CPU.

## 5. What to say honestly

* **ghost_net 0.93 is a synthetic-validation number, not a field number.** No public real ghost-net side-scan dataset exists (GhostNetZero's is private).
  The test nets were rendered with a different parameter regime and on different background surveys than the training nets, so it is not memorisation of identical renders.
* **fishing_gear dropped 0.33 → 0.25** after adding the net class: tangled nets and crab-pot strings overlap visually, and some pot clusters are now
  called ghost_net. This is the argument for the dedicated fishing-gear specialist head in the two-model design; the fallback is a gentler
  re-run (20 epochs, lr0 0.002).
* Every rendered net carries an acoustic shadow on the far side of nadir, so the shadow-consistency check applies to nets as it does to every other class.

## 6. Data (all credited in `DATA_ATTRIBUTION.md`)
SCTD (ship / aircraft / human) · figshare mine-like-object survey · PING derelict crab-pot dataset · SubPipe pipeline tiles and AI4Shipwrecks wrecks via the
DRISHTI SSS mirror (CC-BY-SA 4.0) · synthetic ghost nets (own renderer, the team's own renderer) on those seabed tiles.
Test splits are per-source with survey sequences kept contiguous, so neighbouring frames never straddle train/test.
