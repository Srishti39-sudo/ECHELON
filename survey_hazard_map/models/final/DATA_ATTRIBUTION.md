# Data & software attribution

The sonar hazard detector was trained and evaluated on the following public datasets. Their licences require
this credit; two of them (CC-BY-SA) also require that any *redistributed derivative dataset* carries the same licence.
The trained model weights and code are ours; we do not redistribute the source images.

## Training / evaluation data

| Dataset | Used for | Licence | Citation / source |
|---|---|---|---|
| **SCTD 1.0 — Sonar Common Target Detection Dataset** | shipwreck, aircraft, human | research use (see repo) | P. Zhang, M. Ning et al., https://github.com/freepoet/SCTD |
| **Side-scan sonar imaging for mine detection** (Teledyne Gavia AUV, 2010–2021) | mine-like objects + empty seafloor negatives | CC-BY-SA-4.0 | *Side-scan sonar imaging data of underwater vehicles for mine detection*, Data in Brief (2024), https://doi.org/10.6084/m9.figshare.24574879 |
| **Ghost-pot side-scan sonar detection dataset** (`PINGEcosystem/sss-crab-pot-detection-ds`) | fishing gear (derelict crab pots) | CC-BY-SA-4.0 | GhostVision project, Delaware Inland Bays; https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds |
| **DRISHTI side-scan sonar splits** (`Raamanjal/drishti-sss`) — only the `pipe_`, `bg_`, `wreckA_` tiles | pipeline, empty seafloor negatives, shipwreck | CC-BY-SA-4.0 | https://huggingface.co/datasets/Raamanjal/drishti-sss — an assembled dataset that itself derives from: |
| ↳ **SubPipe / SubPipeMini2** | pipeline tiles | CC-BY-4.0 | O. Álvarez-Tuñón et al., *SubPipe: A Submarine Pipeline Inspection Dataset*, OCEANS 2024, https://zenodo.org/doi/10.5281/zenodo.10053564 (OceanScan-MST / REMARO) |
| ↳ **AI4Shipwrecks** | shipwreck tiles | CC-BY-4.0 | A. Sethuraman et al., *Machine Learning for Shipwreck Segmentation from Side Scan Sonar Imagery: Dataset and Benchmark*, IJRR 2025, https://umfieldrobotics.github.io/ai4shipwrecks/ (Univ. of Michigan / NOAA Thunder Bay NMS) |
| **Side Scan Sonar (Ship, Plane)** — Roboflow Universe, Dae Hyeok Lee | shipwreck, aircraft | CC-BY-4.0 | https://universe.roboflow.com/dae-hyeok-lee/side-scan-sonar |
| **SeabedObjects-KLSG (Ship & Airplane)** | evaluation only (image-level) | research use | G. Huo et al., *Underwater Object Classification in Sidescan Sonar Images Using Deep Transfer Learning*, IEEE Access 2020, https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset |

Not used (documented for completeness): SASSED (Mendeley, CC-BY-4.0) and the raw SubPipe Zenodo archive — both were unreachable at training time.

## Software

| Component | Licence |
|---|---|
| Ultralytics YOLO11 (training, inference, export) | AGPL-3.0 — note: a closed-source commercial deployment would need an Ultralytics Enterprise licence |
| OpenCV, NumPy, PyTorch | Apache-2.0 / BSD |
| pyxtf (XTF sonar file reader) | MIT |
| pyproj (WGS-84 geodesics) | MIT |
| folium / Leaflet (map output) | MIT / BSD |

## Known limitations of the data
- No public labelled dataset of **ghost nets** in side-scan sonar exists; the `fishing_gear` class is trained on derelict crab pots and should be read as "derelict fishing gear".
- `aircraft` (~120 boxes) and `human` (35 boxes) are thin classes; their scores are less reliable.
- DRISHTI tiles were Lee-filtered + CLAHE'd upstream; all other data is raw.
