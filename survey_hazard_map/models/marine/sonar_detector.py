# Sonar hazard detector: python sonar_detector.py IMAGE [--weights best.pt] [--calib calibration.json] [--out result.json] [--draw out.png]
import json, sys, argparse, numpy as np, cv2, torch
from ultralytics import YOLO

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
        self.model = YOLO(weights); c = json.load(open(calib_json))
        self.A, self.B, self.classes, self.imgsz, self.tile, self.overlap = c['A'], c['B'], c['classes'], c['imgsz'], c['tile'], c['overlap']
        self.device = device if device is not None else (0 if torch.cuda.is_available() else 'cpu')
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
            for (x1, y1, _, _), r in zip(wins[i:i + 16], self.model.predict(crops[i:i + 16], imgsz=self.imgsz, conf=raw_conf, iou=0.6, device=self.device, verbose=False)):
                for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
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
        return dict(image=image if isinstance(image, str) else None, image_size=[w, h], tiles=len(wins), detections=sorted(out, key=lambda d: -d['confidence_pct']))
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
