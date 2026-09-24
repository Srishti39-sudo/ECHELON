# shadow_check.py — acoustic-shadow physics check for side-scan sonar detections (numpy + OpenCV only)
import numpy as np, cv2

def analyze(image, box, nadir_x=None, altitude_m=None, m_per_px=None, max_len_factor=3.0, dark_factor=0.5):
    # image: BGR or gray array; box: (x1,y1,x2,y2) px; nadir_x: column of the nadir line (None = try both sides)
    im = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32); H, W = im.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]; x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
    bw, bh = x2 - x1, y2 - y1
    # local seabed level from a ring around the box
    r = max(10, bw // 2); rx1, ry1, rx2, ry2 = max(0, x1 - r), max(0, y1 - r), min(W, x2 + r), min(H, y2 + r)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool); ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    bg = max(float(np.median(im[ry1:ry2, rx1:rx2][ring])) if ring.any() else float(np.median(im)), 1.0)
    dirs = [1 if (x1 + x2) / 2 > nadir_x else -1] if nadir_x is not None else [1, -1]
    best = None
    for d in dirs:
        L = int(max_len_factor * bw); cx = (x1 + x2) // 2
        # search strip starts at the box centre (annotators often include the shadow inside the box) and runs outward
        if d == 1: sx1, sx2 = cx, min(W, x2 + L); strip = im[y1:y2, sx1:sx2]
        else: sx1, sx2 = max(0, x1 - L), cx; strip = im[y1:y2, sx1:sx2][:, ::-1]
        if strip.shape[1] < 3: continue
        dark = strip < dark_factor * bg; lens, ends = [], []
        for row in dark:
            best_run, run, start, best_end = 0, 0, 0, 0
            for j, v in enumerate(row):
                run = run + 1 if v else 0
                if run > best_run and (j - run + 1) <= bw: best_run, best_end = run, j   # run must begin inside the box or right after it
            lens.append(best_run); ends.append(best_end)
        lens = np.array(lens, float); slen = float(np.median(lens))
        if slen >= 1:
            mask = np.zeros_like(dark)
            for i, (l, e) in enumerate(zip(lens, ends)): mask[i, max(0, int(e - l + 1)):int(e) + 1] = True
            contrast = float(strip[mask].mean() / bg) if mask.any() else 1.0
        else: contrast = 1.0
        if slen >= 3 and len(lens) >= 4:
            ys = np.arange(len(lens)); resid = np.sqrt(np.mean((np.array(ends) - np.polyval(np.polyfit(ys, ends, 1), ys)) ** 2)); straight = float(np.exp(-resid / 4))
        else: straight = 0.0
        res = dict(direction=int(d), shadow_len_px=int(slen), shadow_contrast=contrast, edge_straightness=straight, strip_x=(int(sx1), int(sx2)))
        if best is None or res['shadow_len_px'] > best['shadow_len_px']: best = res
    if best is None: best = dict(direction=0, shadow_len_px=0, shadow_contrast=1.0, edge_straightness=0.0, strip_x=(x2, x2))
    inside = im[y1:y2, x1:x2]; top = np.sort(inside.ravel())[-max(1, inside.size // 5):]
    sl, sc, st = best['shadow_len_px'], best['shadow_contrast'], best['edge_straightness']
    score = 0.5 * float(np.clip(sl / (0.5 * bw), 0, 1)) + 0.3 * float(np.clip((1 - sc) / 0.6, 0, 1)) + 0.2 * st
    out = dict(shadow_score=score, has_shadow=bool(sl >= 0.3 * bw and sc < 0.6), highlight_contrast=float(top.mean() / bg), seabed_level=bg,
               verdict='physics-consistent' if score >= 0.6 else ('weak-shadow' if score >= 0.35 else 'no-shadow'),
               height_m=None, width_m=None, length_m=None, **best)
    if altitude_m and m_per_px and nadir_x is not None:
        R = abs((x1 + x2) / 2 - nadir_x) * m_per_px; Lm = sl * m_per_px
        out.update(height_m=altitude_m * Lm / (R + Lm) if (R + Lm) > 0 else None, width_m=bw * m_per_px, length_m=bh * m_per_px)
    return out

def combine(conf, res, veto_below=0.35, veto_conf=0.6):
    # soft re-weighting + hard veto for weak detections without a shadow; returns (final_conf, vetoed)
    final = conf * (0.6 + 0.4 * res['shadow_score']); vetoed = res['shadow_score'] < veto_below and conf < veto_conf
    return (0.0 if vetoed else final), vetoed

def draw(image, box, res, label=''):
    im = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = [int(v) for v in box]; cv2.rectangle(im, (x1, y1), (x2, y2), (0, 220, 0), 2)
    sx1, sx2 = res['strip_x']; cv2.rectangle(im, (sx1, y1), (sx2, y2), (0, 200, 255), 1)
    txt = f"{label} shadow {res['shadow_score']:.2f} len {res['shadow_len_px']}px" + (f" h={res['height_m']:.1f}m" if res['height_m'] else '')
    cv2.putText(im, txt, (max(2, x1), max(12, y1 - 5)), 0, 0.45, (0, 220, 0), 1); return im
