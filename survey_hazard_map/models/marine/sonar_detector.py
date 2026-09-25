# Sonar hazard detector: python sonar_detector.py IMAGE [--weights best.pt] [--calib calibration.json] [--out result.json] [--draw out.png]
#
# Runtime: the .pt runs on Ultralytics/torch when both import; otherwise, or when
# DEEPECHO_DETECTOR_BACKEND=onnx, the .onnx export beside the weights runs on
# onnxruntime (survey_hazard_map/hazard_detect_onnx.OnnxDetector, no torch at
# all). Same crops, same calibration, same class-wise NMS either way, so a
# machine without torch produces the same hazards.json as one with it; the
# export's parity with the checkpoint is measured in docs/edge_parity.json.
import json, os, sys, argparse, numpy as np, cv2
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
    _HAVE_TORCH = True
except ImportError:  # a clone that installed only requirements-server.txt
    torch = None; YOLO = None; _HAVE_TORCH = False


def _onnx_detector_class():
    root = Path(__file__).resolve().parents[3]          # <repo>/survey_hazard_map/models/marine/
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    from survey_hazard_map.hazard_detect_onnx import OnnxDetector
    return OnnxDetector


def _pick_runtime(weights):
    """('torch', weights) or ('onnx', export). DEEPECHO_DETECTOR_BACKEND: auto (default) | onnx | torch."""
    weights = Path(weights); export = weights if weights.suffix.lower() == '.onnx' else weights.with_suffix('.onnx')
    backend = os.environ.get('DEEPECHO_DETECTOR_BACKEND', 'auto').strip().lower() or 'auto'
    if backend == 'torch' or (backend == 'auto' and _HAVE_TORCH and weights.suffix.lower() != '.onnx'):
        if not _HAVE_TORCH: raise RuntimeError('DEEPECHO_DETECTOR_BACKEND=torch but torch / ultralytics are not installed')
        return 'torch', weights
    if not export.is_file():
        raise RuntimeError(f'torch / ultralytics are not installed and {export.name} does not exist beside the weights; '
                           'run survey_hazard_map/tools/export_onnx.py on a machine with torch, or pip install -r requirements-detector.txt')
    return 'onnx', export

def batched_nms(boxes, scores, cls, iou_thr):
    # class-wise greedy NMS in numpy (no torchvision dependency); returns kept indices
    boxes, scores, cls = np.asarray(boxes, float), np.asarray(scores, float), np.asarray(cls)
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0][np.argsort(-scores[cls == c])]
        while len(idx):
            i = idx[0]; keep.append(int(i))
            if len(idx) == 1: break
            b, r = boxes[i], boxes[idx[1:]]
            ix1, iy1 = np.maximum(b[0], r[:, 0]), np.maximum(b[1], r[:, 1]); ix2, iy2 = np.minimum(b[2], r[:, 2]), np.minimum(b[3], r[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            iou = inter / ((b[2] - b[0]) * (b[3] - b[1]) + (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) - inter + 1e-9)
            idx = idx[1:][iou < iou_thr]
    return keep

class SonarDetector:
    def __init__(self, weights, calib_json, device=None):
        c = json.load(open(calib_json))
        self.A, self.B, self.classes, self.imgsz, self.tile, self.overlap = c['A'], c['B'], c['classes'], c['imgsz'], c['tile'], c['overlap']
        self.backend, path = _pick_runtime(weights)
        if self.backend == 'torch':
            self.model = YOLO(str(path)); self.device = device if device is not None else (0 if torch.cuda.is_available() else 'cpu')
        else:
            self.model = None; self.device = 'onnxruntime'; self._onnx = None; self._onnx_path = path
    def _predict(self, crops, raw_conf):
        """Per crop: list of (xyxy, score, class index into self.classes). One code path per runtime, same output."""
        if self.backend == 'torch':
            for r in self.model.predict(crops, imgsz=self.imgsz, conf=raw_conf, iou=0.6, device=self.device, verbose=False):
                yield [(b, s, int(c)) for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            return
        if self._onnx is None or self._onnx.conf != raw_conf:
            self._onnx = _onnx_detector_class()(self._onnx_path, conf=raw_conf, imgsz=self.imgsz)
        index = {name: i for i, name in enumerate(self.classes)}
        for crop in crops:
            rgb = np.ascontiguousarray(crop[:, :, ::-1])                 # cv2 gives BGR; the export was trained on RGB
            yield [(xyxy, float(s), index[name]) for _, name, s, xyxy in self._onnx.detect_raw(rgb) if name in index]
    def calibrate(self, p):
        p = np.clip(p, 1e-4, 1 - 1e-4); return float(1 / (1 + np.exp(-(self.A * np.log(p / (1 - p)) + self.B))))
    def _windows(self, h, w):
        if max(h, w) <= 1024: return [(0, 0, w, h)]
        step = self.tile - self.overlap
        ys = list(range(0, max(h - self.tile, 0) + 1, step)); xs = list(range(0, max(w - self.tile, 0) + 1, step))
        if ys[-1] + self.tile < h: ys.append(max(h - self.tile, 0))
        if xs[-1] + self.tile < w: xs.append(max(w - self.tile, 0))
        return [(x, y, min(x + self.tile, w), min(y + self.tile, h)) for y in ys for x in xs]
    def detect(self, image, raw_conf=0.10, min_calibrated=0.25):
        im = cv2.imread(image) if isinstance(image, str) else image
        h, w = im.shape[:2]; wins = self._windows(h, w); crops = [im[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]
        boxes, scores, cls = [], [], []
        for i in range(0, len(crops), 16):
            for (x1, y1, _, _), found in zip(wins[i:i + 16], self._predict(crops[i:i + 16], raw_conf)):
                for b, s, c in found:
                    boxes.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]); scores.append(s); cls.append(int(c))
        out = []
        if boxes:
            for k in batched_nms(boxes, scores, cls, 0.5):
                p = self.calibrate(scores[k])
                if p >= min_calibrated:
                    x1, y1, x2, y2 = boxes[k]
                    out.append(dict(label=self.classes[cls[k]], confidence_pct=round(100 * p, 1), raw_score=round(scores[k], 3),
                                    box_xyxy_px=[round(v, 1) for v in (x1, y1, x2, y2)], width_px=round(x2 - x1, 1), height_px=round(y2 - y1, 1),
                                    man_made=True))
        return dict(image=image if isinstance(image, str) else None, image_size=[w, h], tiles=len(wins), runtime=self.backend, detections=sorted(out, key=lambda d: -d['confidence_pct']))
    def draw(self, image, result, path):
        im = cv2.imread(image) if isinstance(image, str) else image.copy()
        for d in result['detections']:
            x1, y1, x2, y2 = map(int, d['box_xyxy_px']); cv2.rectangle(im, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(im, f"{d['label']} {d['confidence_pct']:.0f}%", (x1, max(y1 - 5, 12)), 0, 0.6, (0, 220, 0), 2)
        cv2.imwrite(path, im)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--weights', default='best.pt'); ap.add_argument('--calib', default='calibration.json')
    ap.add_argument('--out'); ap.add_argument('--draw'); a = ap.parse_args()
    det = SonarDetector(a.weights, a.calib); res = det.detect(a.image)
    print(json.dumps(res, indent=2))
    if a.out: json.dump(res, open(a.out, 'w'), indent=2)
    if a.draw: det.draw(a.image, res, a.draw)
