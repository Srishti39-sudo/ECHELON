# Training pipe, cylinder, wreck and net detectors for DeepEcho

This folder is a scaffold, not a trained model. It exists so that a new
checkpoint (for example one that knows `pipe` and `cylinder`) is trained,
evaluated and calibrated the same way every time, and drops into the survey
pipeline without any code change.

```
training/
  README.md              this file
  data.example.yaml      ultralytics dataset template, project class vocabulary
  train.py               YOLO training with sonar-appropriate augmentation; writes metrics.json
  calibrate.py           fits temperature / Platt scaling on a val split; writes models/calibration.json
  augment_sonar.py       offline speckle, gain drift, resolution jitter, shadow-consistent copy-paste
```

The end-to-end path is:

```
build dataset (tiles at 640, split BY SURVEY LINE)
  -> python training/train.py --data my.yaml --model yolov8s.pt --name pipe-v1
  -> python training/calibrate.py --weights runs/detect/pipe-v1/weights/best.pt --data my.yaml
  -> copy best.pt to models/, keep models/calibration.json beside it
```

`hazard_verify.load_calibration()` reads the calibration file automatically;
without one, every verification block says `uncalibrated`.

---

## 1. Candidate public datasets

Always read the record page yourself before use. Licences below marked
"verify" were not confirmed from the record at the time of writing, and "open
for academic use" is not a licence that permits redistribution or commercial
use.

| Dataset | What it offers | Link | Licence |
|---|---|---|---|
| **SubPipe** | AUV survey over a subsea pipeline, including side-scan sonar images with pipe annotations | https://zenodo.org/records/12666132 | CC-BY-4.0 |
| **Marine-PULSE** | Subsea pipeline / cable imagery from AUV sonar | https://zenodo.org/records/7922705 | verify on record |
| **Side-scan sonar imaging for mine detection** (Santos & Moura, 2024, *Data in Brief*) | 1170 real Gavia AUV side-scan images, 2010-2021, labelled MILCO (mine-like, largely cylindrical) and NOMBO (non-mine bottom objects). Good for `cylinder` and, critically, for hard negatives | https://figshare.com/articles/dataset/_i_Side-scan_sonar_imaging_for_Mine_detection_i_/24574879 | verify on record |
| **SeabedObjects-KLSG** (Huo et al., 2020) | ~1190 SSS crops: wreck, drowning victim, airplane, mine (mine subset not public), seafloor. Classification crops, so boxes must be drawn | https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset | "academic use"; verify |
| **AI4Shipwrecks** (Sethuraman et al., 2024) | 286 high-resolution SSS images of 28 Great Lakes wrecks with pixel masks (convert masks to boxes) | https://umfieldrobotics.github.io/ai4shipwrecks/ | verify on record |
| **DRISHTI-SSS** | Side-scan detection set with several debris classes | https://huggingface.co/datasets/Raamanjal/drishti-sss | verify on record. **Its ghost-net class is synthetic.** Do not report net performance measured on it as real-world performance |

Also useful as an index: https://github.com/remaro-network/OpenSonarDatasets

### What is missing, and matters

- **Real nets / ghost gear.** No public, real, labelled SSS net set was found.
  A net model trained on synthetic nets is a demo, and the README of any such
  checkpoint must say so.
- **Hard negatives.** Rock fields, sand waves, nadir boundaries and
  water-column returns without any boxes. A detector never shown them learns
  that every strong edge is an object. This repository's own sample strip
  (`samples/sidescan-waterfall-strip.jpg`) is exactly this failure: every
  detection on it is a false positive on the nadir boundary. Include
  background-only tiles (an image with an empty label file) at roughly 10-30%
  of the training set.

---

## 2. Class map to the project vocabulary

Map every source label onto the names the pipeline already understands, so
severity weights, actions and the verification class expectations apply
without edits. Matching downstream is by normalised name, then longest
contained key (see `hazard_severity.py`), so `subsea-pipeline` would still
land on `pipe`, but be explicit.

| Project class | Source labels to map onto it | Shadow expectation in `hazard_verify` |
|---|---|---|
| `pipe` | pipeline, pipe, cable (if you do not keep `cable` separate) | 0.6 (often part-buried) |
| `cable` | cable, umbilical | 0.3 |
| `cylinder` | MILCO, cylinder, mine-like cylinder, drum-like | 1.0 |
| `mine` | mine (only where ground truth says it is one) | 1.0 |
| `shipwreck` | wreck, ship, shipwreck, submarine | 1.0 |
| `aircraft` | airplane, plane, aircraft | 1.0 |
| `net` | net, ghost net, ghost gear, fishing net | 0.2 (draped) |
| `tyre`, `drum` | tire/tyre, barrel/drum | 0.9 / 1.0 |
| `human` | drowning victim, body | 0.7 |
| *(no box)* | seafloor, NOMBO rock, sand ripple | background tile |

NOMBO is ambiguous: "non-mine-like bottom object" includes rocks and man-made
clutter. Either keep it as its own class (`nombo`, which the severity table
treats as `unknown` at 0.8) or use those tiles as background; do not fold it
into `cylinder`.

If you add a class the tables do not know, add it to `SEVERITY`, `ACTIONS` and
`VERIFY_CLASS_EXPECTATIONS` in `hazard_config.py`. Until you do, it is treated
as `unknown`: weighted 0.8, shadow expectation 0.5. Nothing is silently dropped.

---

## 3. Tiling: match the pipeline

The survey pipeline cuts strips into **640 px tiles at stride 512** and runs the
detector at `imgsz=640` (`TILE`, `STRIDE`, `DETECTOR_IMGSZ` in
`hazard_config.py`). Train on the same thing:

- Tile source images at 640 with the same overlap. `survey_preparation.py`
  already does it; reuse its `_offsets` so train and inference tiles align.
- Clip boxes to each tile; drop a box that keeps less than ~40% of its area in
  a tile (it is whole in a neighbour, thanks to the overlap).
- Do **not** resize a 3000 px strip down to 640 for training. The detector then
  learns objects 5x smaller than it will see at inference.
- Keep the along-track / across-track orientation of the pipeline: across-track
  (range) is image x, nadir is a vertical band.

## 4. Split by survey line, never by tile

Adjacent tiles overlap by 128 px, and consecutive pings of one line see the
same object, seabed and gain. If tile 17 of a line is in train and tile 18 is
in val, validation measures memorisation and every number that follows,
mAP and calibration included, is inflated.

- Split at the level of the **survey line** (or whole source image / dive).
- Better still, hold out whole **sites**, so validation is a place the model
  has never seen.
- Record the split lists (`train.txt`, `val.txt`) in the dataset folder so a
  result can be reproduced.
- Calibrate on val (or a separate calibration split), never on train.

---

## 5. Augmentation that respects sonar physics

`train.py` sets these by default. The reasoning:

| Augmentation | Setting | Why |
|---|---|---|
| Hue, saturation | `hsv_h=0`, `hsv_s=0` | Sonar intensity is single-channel backscatter. Colour jitter only teaches the model about display palettes |
| Value (brightness) | `hsv_v=0.3` | Stands in for gain / TVG differences between systems |
| Vertical flip | `flipud=0.5` | Reverses the along-track direction; physically indistinguishable |
| Horizontal flip | `fliplr=0.5`, `--no-lr-flip` to disable | Swaps port and starboard. Legitimate ONLY because the whole tile is mirrored, so the shadow still points away from nadir. It becomes wrong if you mirror an object crop without its shadow, or your data is single-sided and never shows the mirrored geometry |
| Rotation | `degrees=3` | Small heading / yaw. Large rotations break the rule that shadows run along range |
| Scale | `scale=0.3` | Range and altitude change apparent size |
| Shear, perspective | `0` | No physical counterpart in a waterfall |
| Mosaic | `mosaic=1.0`, `close_mosaic=10` | Helps small sets; the last epochs train on unmosaicked tiles so the model sees whole-tile context |
| Mixup | `0` | Blending two sonar images creates impossible shadows |

Offline, `augment_sonar.py` adds what ultralytics cannot: multiplicative
speckle (Rayleigh / gamma), along-track gain drift and across-track TVG
ramps, resolution jitter (degrade and restore, boxes unchanged), and
**shadow-consistent copy-paste** of object crops onto real seabed, with the
shadow placed on the far side of the paste location relative to nadir.

**Synthetic augmentation is not data.** It widens coverage of nuisance
variation around examples you already have. It cannot teach the appearance of
an object class you have too few real examples of, and a validation score on
pasted objects measures the paste, not the detector. Keep pasted examples out
of val entirely.

---

## 6. Evaluate and calibrate

`train.py` writes `metrics.json` beside the weights: mAP50, mAP50-95, precision,
recall and per-class AP50-95 on the val split.

Then:

```
python training/calibrate.py --weights runs/detect/pipe-v1/weights/best.pt \
    --data my.yaml --model-name pipe --merge
```

It runs the checkpoint on val at the pipeline's `CONF_THRESH`, matches boxes to
ground truth (same class, IoU >= 0.5, greedy by confidence), fits temperature
and Platt scaling by minimising NLL, picks between them by 5-fold
cross-validated NLL, prints ECE and a reliability table before and after, and
writes `models/calibration.json`. `--merge` keeps entries for other models
already in the file. The model name must match the checkpoint's file stem as
the pipeline reports it (`detector_model` on each detection).

What the calibrated number means: the probability a box at that score matches
a real object of its class. It says nothing about objects the detector missed.
