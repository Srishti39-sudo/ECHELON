# sonar_pipeline.py — end-to-end: raw sonar (.xtf or image) → detections → physics check → open-set anomalies → geotagged hazard report
#
#   python sonar_pipeline.py survey.xtf --weights best.pt --calib calibration.json --out report/
#   python sonar_pipeline.py tile.png --weights best.pt --calib calibration.json --simulate-nav --out report/
#   optional:  --shadow (needs shadow_check.py)   --anomaly bg_embed.pkl (needs anomaly.py)   --altitude 8 --m-per-px 0.05 (for image input)
#
# Every stage is optional except the detector, so the pipeline degrades gracefully: no .xtf → no lat/lon (unless --simulate-nav, labelled),
# no shadow module → no height, no anomaly module → no unknown_anomaly channel.

import argparse, json, sys, time
from pathlib import Path
import numpy as np, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='.xtf survey file or a sonar image')
    ap.add_argument('--weights', required=True); ap.add_argument('--calib', required=True)
    ap.add_argument('--modules', default='.', help='folder containing sonar_detector.py / shadow_check.py / anomaly.py / geotag.py')
    ap.add_argument('--shadow', action='store_true', help='run the acoustic-shadow physics check'); ap.add_argument('--shadow-veto', type=float, default=0.35, help='veto when shadow_score is below this ...')
    ap.add_argument('--shadow-veto-conf', type=float, default=0.6, help='... and the detector confidence is below this (strong detections survive without a shadow)')
    ap.add_argument('--anomaly', help='bg_embed.pkl → run the open-set anomaly channel')
    ap.add_argument('--nav', help='navigation CSV for image input'); ap.add_argument('--simulate-nav', action='store_true'); ap.add_argument('--nadir-col', type=int)
    ap.add_argument('--altitude', type=float, help='vehicle altitude (m) for height estimation on image input'); ap.add_argument('--m-per-px', type=float)
    ap.add_argument('--raw-conf', type=float, default=0.10); ap.add_argument('--min-conf', type=float, default=0.25)
    ap.add_argument('--out', default='pipeline_out')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.modules).resolve())); sys.path.insert(0, str(Path(a.weights).resolve().parent))
    from sonar_detector import SonarDetector
    import geotag as gt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); src = Path(a.source); t0 = time.time(); timings = {}

    # 1. input → image (+ navigation)
    if src.suffix.lower() == '.xtf':
        img, nav = gt.read_xtf(src); name = src.stem; cv2.imwrite(str(out / f'{name}_waterfall.png'), img)
    elif src.suffix.lower() in ('.tif', '.tiff'):          # georeferenced mosaic (USGS / NOAA GeoTIFF): lat/lon from the raster transform
        nav = gt.GeoRaster(src); img = nav.image; name = src.stem; print(f'GeoTIFF {img.shape[1]}x{img.shape[0]} px, {nav.res[0]:.2f} m/px, {nav.crs}')
        cv2.imwrite(str(out / f'{name}_waterfall.png'), img)      # the exact raster the boxes are drawn on, for downstream importers
    else:
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE); name = src.stem; nadir = a.nadir_col if a.nadir_col is not None else img.shape[1] // 2
        nav = gt.NavTable.from_csv(a.nav, nadir) if a.nav else (gt.simulated_nav(img.shape[0], nadir, max(nadir, img.shape[1] - nadir)) if a.simulate_nav else None)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); timings['load'] = time.time() - t0

    # 2. detector (tiled, calibrated)
    t = time.time(); det = SonarDetector(a.weights, a.calib); res = det.detect(bgr, raw_conf=a.raw_conf, min_calibrated=a.min_conf); dets = res['detections']
    if hasattr(nav, 'valid'):        # mosaic: drop detections whose box lies mostly outside the surveyed area (no-data fill)
        keep = []
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d['box_xyxy_px']]; v = nav.valid[max(0, y1):y2, max(0, x1):x2]
            if v.size and v.mean() >= 0.6: keep.append(d)
        res['dropped_outside_survey'] = len(dets) - len(keep); dets = keep
    timings['detect'] = time.time() - t

    # 3. shadow physics check (optional)
    if a.shadow:
        t = time.time(); import shadow_check
        ping0 = nav.rows[0] if getattr(nav, 'rows', None) else None
        nadir_x = getattr(nav, 'nadir_col', None); alt = a.altitude or (ping0['altitude_m'] if ping0 else None)
        mpp = a.m_per_px or (ping0['slant_range_m'] / ping0['samples_per_side'] if ping0 else (nav.res[0] / nav.scale if hasattr(nav, 'res') else None))
        kept = []
        for d in dets:
            sh = shadow_check.analyze(bgr, d['box_xyxy_px'], nadir_x=nadir_x, altitude_m=alt, m_per_px=mpp)
            final, vetoed = shadow_check.combine(d['confidence_pct'] / 100, sh, a.shadow_veto, a.shadow_veto_conf)
            d.update(shadow_score=round(sh['shadow_score'], 3), verdict=sh['verdict'], height_m=None if sh['height_m'] is None else round(sh['height_m'], 2),
                     confidence_pct=round(100 * final, 1), vetoed=vetoed)
            if not vetoed: kept.append(d)
        res['vetoed_by_shadow_check'] = len(dets) - len(kept); dets = kept; timings['shadow'] = time.time() - t

    # 4. open-set anomaly channel (optional)
    anomalies = []
    if a.anomaly:
        t = time.time(); import anomaly
        AN = anomaly.SeafloorAnomaly.load(a.anomaly, weights=a.weights); anomalies, _, _ = AN.score_regions(bgr, dets); timings['anomaly'] = time.time() - t

    # 5. geotag + report
    everything = dets + [dict(x, label='unknown_anomaly', confidence_pct=round(100 * x['anomaly_score'], 1)) for x in anomalies]
    if nav is not None:
        hz = gt.geotag(everything, nav, name); gt.write_report(hz, nav, out, img, name)
    else:
        hz = [dict(id=i + 1, image=name, classification=d['label'], confidence_pct=d['confidence_pct'], box_xyxy_px=d['box_xyxy_px'],
                   shadow_score=d.get('shadow_score'), verdict=d.get('verdict'), height_m=d.get('height_m'), anomaly_score=d.get('anomaly_score'),
                   lat=None, lon=None, navigation='none — no .xtf / nav file supplied') for i, d in enumerate(everything)]
        json.dump(dict(meta=dict(source=str(src), n_hazards=len(hz)), hazards=hz), open(out / 'hazards.json', 'w'), indent=2, default=float)
    timings['total'] = time.time() - t0
    summary = dict(source=str(src), image_size=[int(img.shape[1]), int(img.shape[0])], tiles=res['tiles'], detections=len(dets),
                   unknown_anomalies=len(anomalies), vetoed_by_shadow_check=res.get('vetoed_by_shadow_check', 0), navigation=('simulated' if nav is not None and nav.simulated else ('xtf/csv' if nav is not None else 'none')),
                   timings_s={k: round(v, 2) for k, v in timings.items()})
    json.dump(summary, open(out / 'summary.json', 'w'), indent=2); print(json.dumps(summary, indent=2))
    for h in hz:
        pos = f"{h.get('lat_dms', '')}, {h.get('lon_dms', '')}" if h.get('lat') is not None else 'no position'
        print(f"  #{h['id']:>2} {h['classification']:18s} {h['confidence_pct']:>5}%  {pos}  {h.get('verdict') or ''}  h={h.get('height_m')}")

if __name__ == '__main__': main()
