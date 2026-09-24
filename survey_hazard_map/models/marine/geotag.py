# geotag.py — Anomalous reporting & geotagging engine for side-scan sonar detections.
#
#   raw .xtf survey file  ──►  waterfall image + per-ping navigation table
#   detections (pixel boxes) ──►  slant range → ground range → port/stbd offset → lat/lon (WGS-84)
#                          ──►  hazards.json / hazards.csv / hazards_map.html
#
# Usage
#   python geotag.py survey.xtf --detections dets.json --out report/          # geotag existing detections
#   python geotag.py survey.xtf --weights best.pt --calib calibration.json     # render + detect + geotag in one go
#   python geotag.py tile.jpg --nav nav.csv --detections dets.json             # image + CSV navigation (one row per image row / ping)
#   python geotag.py --selftest                                                # synthetic survey: proves the maths (< 1 m error)
#
# Dependencies: numpy, opencv-python, pyproj, pyxtf (for .xtf), folium (optional, for the map)

import argparse, csv, json, math, sys, datetime, ctypes
from pathlib import Path
import numpy as np, cv2
from pyproj import Geod

GEOD = Geod(ellps='WGS84')
SOUND_SPEED_DEFAULT = 1500.0


# ----------------------------------------------------------------------------------------------------------------------
# Navigation table: one entry per image row (= one sonar ping)
# ----------------------------------------------------------------------------------------------------------------------
class NavTable:
    """Per-ping navigation. Arrays indexed by image row. nadir_col splits port (left) from starboard (right)."""
    FIELDS = ['ping', 'time', 'lat', 'lon', 'heading_deg', 'altitude_m', 'slant_range_m', 'samples_per_side']

    def __init__(self, rows, nadir_col, source, simulated=False):
        self.rows, self.nadir_col, self.source, self.simulated = rows, int(nadir_col), source, simulated
        self.lat = np.array([r['lat'] for r in rows], float); self.lon = np.array([r['lon'] for r in rows], float)

    def __len__(self): return len(self.rows)

    def along_track_m_per_row(self, row):
        # mean ping spacing over a window around the row. A window (not neighbouring pings) is needed because real files repeat the
        # same GPS fix for several pings (1 Hz GPS vs 5-10 Hz sonar) — neighbouring pings would give 0 m.
        n = len(self.rows); i = min(max(int(row), 0), n - 1)
        for w in (10, 50, 250, n):
            a, b = max(0, i - w), min(n - 1, i + w)
            if b <= a: return 0.0
            _, _, d = GEOD.inv(self.lon[a], self.lat[a], self.lon[b], self.lat[b])
            if d > 0: return float(d) / (b - a)
        return 0.0

    def ground_range_at(self, col, row):
        g = self.pixel_to_latlon(col, row); return g['ground_range_m'], g['side']

    def pixel_to_latlon(self, col, row):
        """Core geometry. Returns dict with lat/lon, side, slant & ground range, resolution."""
        r = self.rows[min(max(int(row), 0), len(self.rows) - 1)]
        n = r['samples_per_side']; slant_max = r['slant_range_m']; alt = r['altitude_m']
        if col < self.nadir_col: side, idx = 'port', (self.nadir_col - 1 - col)          # port: far range at the left edge
        else: side, idx = 'stbd', (col - self.nadir_col)
        m_per_sample = slant_max / max(n, 1)
        slant = (idx + 0.5) * m_per_sample
        ground = math.sqrt(max(slant ** 2 - alt ** 2, 0.0))                           # flat-seabed slant-range correction
        bearing = (r['heading_deg'] + (-90 if side == 'port' else 90)) % 360             # perpendicular to the track
        lon, lat, _ = GEOD.fwd(r['lon'], r['lat'], bearing, ground)
        return dict(lat=lat, lon=lon, side=side, slant_range_m=slant, ground_range_m=ground, across_track_m_per_px=m_per_sample,
                    along_track_m_per_px=self.along_track_m_per_row(row), ping=r['ping'], time=r['time'], altitude_m=alt, heading_deg=r['heading_deg'])

    # ---- constructors ----
    @classmethod
    def from_csv(cls, path, nadir_col, simulated=False):
        rows = []
        with open(path) as f:
            for rec in csv.DictReader(f):
                rows.append(dict(ping=int(float(rec.get('ping', len(rows)))), time=rec.get('time', ''), lat=float(rec['lat']), lon=float(rec['lon']),
                                 heading_deg=float(rec['heading_deg']), altitude_m=float(rec['altitude_m']), slant_range_m=float(rec['slant_range_m']),
                                 samples_per_side=int(float(rec['samples_per_side']))))
        return cls(rows, nadir_col, str(path), simulated)

    def to_csv(self, path):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS); w.writeheader(); [w.writerow({k: r[k] for k in self.FIELDS}) for r in self.rows]


# ----------------------------------------------------------------------------------------------------------------------
# XTF reading: waterfall image + NavTable
# ----------------------------------------------------------------------------------------------------------------------
def _ping_dt(p):
    try: t = p.get_time()                      # pyxtf returns numpy.datetime64 (naive; XTF clocks are UTC)
    except Exception: t = datetime.datetime(p.Year, p.Month, p.Day, p.Hour, p.Minute, p.Second)
    if not hasattr(t, 'isoformat'): t = np.datetime64(t, 'us').astype(datetime.datetime)
    return t.replace(tzinfo=datetime.timezone.utc)

def _interpolate_repeated_fixes(rows):
    """Real files repeat one GPS fix for several pings (1 Hz GPS, 5-10 Hz sonar). Interpolate position by time between fix changes."""
    n = len(rows); lat = np.array([r['lat'] for r in rows]); lon = np.array([r['lon'] for r in rows])
    t = np.array([datetime.datetime.fromisoformat(r['time']).timestamp() for r in rows])
    change = [0] + [i for i in range(1, n) if lat[i] != lat[i - 1] or lon[i] != lon[i - 1]]
    if len(change) < 2 or len(change) > 0.9 * n: return False           # no repeats (or no fixes at all): nothing to do
    for a, b in zip(change[:-1], change[1:]):
        if b - a > 1 and t[b] > t[a]:
            f = (t[a:b] - t[a]) / (t[b] - t[a]); lat[a:b] = lat[a] + f * (lat[b] - lat[a]); lon[a:b] = lon[a] + f * (lon[b] - lon[a])
    for r, la, lo in zip(rows, lat, lon): r['lat'], r['lon'] = float(la), float(lo)
    return True

def read_xtf(path, port_chan=0, stbd_chan=1, gain='log', altitude_override=None):
    """Returns (image uint8 HxW, NavTable). Row 0 = first ping. Port channel is mirrored so the nadir is in the middle."""
    import pyxtf
    fh, packets = pyxtf.xtf_read(str(path))
    pings = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not pings: raise ValueError(f'{path}: no sonar pings found')
    types = {i: fh.ChanInfo[i].TypeOfChannel for i in range(int(fh.NumberOfSonarChannels))}
    port = next((i for i, t in types.items() if t == pyxtf.XTFChannelType.port.value), port_chan)
    stbd = next((i for i, t in types.items() if t == pyxtf.XTFChannelType.stbd.value), stbd_chan)
    pings = [p for p in pings if len(p.data) > max(port, stbd)]; pings.sort(key=_ping_dt)
    latlon = int(fh.NavUnits) == int(pyxtf.XTFNavUnits.latlon.value)
    if not latlon: raise ValueError('XTF navigation is in projected metres (NavUnits=meters) — projected-coordinate support not implemented')
    # file-level decisions (a single 0 is a valid heading/altitude, so decide on the whole file, not per ping)
    use_sensor_heading = any(float(p.SensorHeading) != 0 for p in pings)
    use_sensor_pos = any(float(p.SensorXcoordinate) != 0 or float(p.SensorYcoordinate) != 0 for p in pings)
    alt_field = 'SensorPrimaryAltitude' if any(float(p.SensorPrimaryAltitude) > 0 for p in pings) else ('SensorAuxAltitude' if any(float(p.SensorAuxAltitude) > 0 for p in pings) else None)
    notes = []
    if not use_sensor_pos: notes.append('no towfish position in file: using ship position' + (' minus cable layback' if any(float(p.Layback) > 0 for p in pings) else ' (no layback recorded → towfish assumed under the ship)'))
    if alt_field is None and altitude_override is None: notes.append('WARNING: altitude is 0 in every ping → no slant-range correction (positions biased outward near the track). Pass --altitude.')
    rows, port_lines, stbd_lines = [], [], []
    for p in pings:
        dp, ds = np.asarray(p.data[port], np.float32), np.asarray(p.data[stbd], np.float32); ch = p.ping_chan_headers[stbd]
        heading = float(p.SensorHeading) if use_sensor_heading else float(p.ShipGyro)
        if use_sensor_pos: x, y = float(p.SensorXcoordinate), float(p.SensorYcoordinate)
        else:
            x, y = float(p.ShipXcoordinate), float(p.ShipYcoordinate)
            if float(p.Layback) > 0: x, y, _ = GEOD.fwd(x, y, (heading + 180) % 360, float(p.Layback))     # towfish trails the ship along the cable
        alt = altitude_override if altitude_override is not None else (float(getattr(p, alt_field)) if alt_field else 0.0)
        rows.append(dict(ping=int(p.PingNumber), time=_ping_dt(p).isoformat(), lat=y, lon=x, heading_deg=heading, altitude_m=alt,
                         slant_range_m=float(ch.SlantRange), samples_per_side=int(ch.NumSamples) or len(ds),     # per ping: range scale may change mid-line
                         roll_deg=float(p.SensorRoll), pitch_deg=float(p.SensorPitch), heave_m=float(p.Heave)))
        port_lines.append(dp[::-1]); stbd_lines.append(ds)
    # ping drop-outs: time gaps much larger than the median ping interval (rows are NOT inserted; flagged for downstream)
    t = np.array([datetime.datetime.fromisoformat(r['time']).timestamp() for r in rows]); dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else 0.0; gaps = int(np.sum(dt > 2.5 * med)) if med > 0 else 0
    for i, r in enumerate(rows): r['gap_before'] = bool(i > 0 and med > 0 and dt[i - 1] > 2.5 * med)
    if gaps: notes.append(f'{gaps} ping drop-outs (time gaps > 2.5x median interval) flagged')
    if _interpolate_repeated_fixes(rows): notes.append('GPS fixes repeated across pings → positions interpolated by time')
    # image: keep the near-range n samples of every ping (index = sample index, so per-ping SlantRange/NumSamples still applies)
    n = min(min(len(a) for a in port_lines), min(len(a) for a in stbd_lines))
    img = np.hstack([np.vstack([a[-n:] for a in port_lines]), np.vstack([a[:n] for a in stbd_lines])])
    img = np.log1p(np.clip(img, 0, None)) if gain == 'log' else img
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    img8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    for m in notes: print('read_xtf:', m)
    nav = NavTable(rows, nadir_col=n, source=str(path)); nav.notes = notes; return img8, nav


# ----------------------------------------------------------------------------------------------------------------------
# Georeferenced mosaic (GeoTIFF, e.g. USGS / NOAA processed side-scan): lat/lon straight from the raster transform
# ----------------------------------------------------------------------------------------------------------------------
class GeoRaster:
    """Same interface as NavTable for a GeoTIFF mosaic. No pings/slant range — the mosaic is already ground-range corrected."""
    def __init__(self, path, max_side=None):
        import rasterio
        from rasterio.warp import transform as _tr
        self.source, self.simulated, self.rows = str(path), False, []
        with rasterio.open(path) as ds:
            self.crs, self.transform, self.res = ds.crs, ds.transform, ds.res; self.width, self.height = ds.width, ds.height
            self.scale = 1.0
            if max_side and max(ds.width, ds.height) > max_side:
                self.scale = max_side / max(ds.width, ds.height)
            img = ds.read(1, out_shape=(int(ds.height * self.scale), int(ds.width * self.scale)))
            img = img if img.dtype == np.uint8 else np.clip((img - np.percentile(img, 1)) / max(np.ptp(img), 1e-6) * 255, 0, 255).astype(np.uint8)
            # no-data outside the survey lines (white or black in most mosaics) → fill with the seabed median so the sharp
            # data/no-data edges are not mistaken for objects; tiles with no survey data are skipped by the detector wrapper
            nd = ds.nodata if ds.nodata is not None else (255 if (img == 255).mean() > 0.2 else (0 if (img == 0).mean() > 0.2 else None))
            self.valid = np.ones(img.shape, bool) if nd is None else (img != nd)
            if nd is not None and self.valid.any():
                img = img.copy(); img[~self.valid] = int(np.median(img[self.valid]))
                self.valid = cv2.erode(self.valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # trim the mosaic feathering at strip edges
                img[~self.valid] = int(np.median(img[self.valid]))
            self.image = img
            xs = [ds.bounds.left, ds.bounds.right, ds.bounds.right, ds.bounds.left, ds.bounds.left]; ys = [ds.bounds.top, ds.bounds.top, ds.bounds.bottom, ds.bounds.bottom, ds.bounds.top]
            lon, lat = _tr(ds.crs, 'EPSG:4326', xs, ys); self.lat, self.lon = np.array(lat), np.array(lon)   # footprint outline for the map
        self._tr = _tr
    def __len__(self): return self.height
    def pixel_to_latlon(self, col, row):
        c, r = col / self.scale, row / self.scale                      # back to full-resolution pixel coordinates
        x, y = self.transform * (c, r); lon, lat = self._tr(self.crs, 'EPSG:4326', [x], [y])
        return dict(lat=lat[0], lon=lon[0], side='mosaic', slant_range_m=0.0, ground_range_m=0.0, across_track_m_per_px=self.res[0] / self.scale,
                    along_track_m_per_px=self.res[1] / self.scale, ping=None, time=None, altitude_m=None, heading_deg=0.0)


# ----------------------------------------------------------------------------------------------------------------------
# Geotag detections
# ----------------------------------------------------------------------------------------------------------------------
def geotag(detections, nav, image_name='', shadow=None):
    """detections: [{label, confidence_pct, box_xyxy_px, ...}] → hazard records with lat/lon + size in metres."""
    out = []
    for k, d in enumerate(detections):
        x1, y1, x2, y2 = [float(v) for v in d['box_xyxy_px']]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        g = nav.pixel_to_latlon(cx, cy)
        # across-track width = difference of ground ranges at the two box edges (exact; a constant m/px would be wrong near nadir)
        if hasattr(nav, 'ground_range_at'):
            g1, s1 = nav.ground_range_at(x1, cy); g2, s2 = nav.ground_range_at(x2, cy)
            width_m = abs(g2 - g1) if s1 == s2 else g1 + g2          # box straddling the nadir line: ranges add up
        else: width_m = (x2 - x1) * g['across_track_m_per_px']
        length_m = (y2 - y1) * g['along_track_m_per_px']
        rec = dict(id=k + 1, image=image_name, classification=d.get('label'), confidence_pct=d.get('confidence_pct'),
                   lat=round(g['lat'], 6), lon=round(g['lon'], 6), lat_dms=_dms(g['lat'], 'NS'), lon_dms=_dms(g['lon'], 'EW'),
                   side=g['side'], ground_range_m=round(g['ground_range_m'], 1), slant_range_m=round(g['slant_range_m'], 1),
                   length_m=round(length_m, 2), width_m=round(width_m, 2), height_m=d.get('height_m'),
                   ping=g['ping'], ping_time=g['time'], vehicle_altitude_m=g['altitude_m'], heading_deg=round(g['heading_deg'], 1),
                   box_xyxy_px=[round(v, 1) for v in (x1, y1, x2, y2)], man_made=d.get('label') != 'unknown_anomaly' or None,
                   shadow_score=d.get('shadow_score'), verdict=d.get('verdict'), anomaly_score=d.get('anomaly_score'),
                   navigation=('SIMULATED — not real positions' if nav.simulated else nav.source))
        out.append(rec)
    return out

def _dms(v, hemi):
    h = hemi[0] if v >= 0 else hemi[1]; v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:05.2f}\" {h}"

def write_report(hazards, nav, out_dir, image=None, image_name='survey'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=nav.source, simulated_navigation=nav.simulated,
                n_pings=len(nav), n_hazards=len(hazards), datum='WGS-84', method='slant→ground range (flat seabed), perpendicular-to-heading offset, geodesic (pyproj)',
                notes=getattr(nav, 'notes', []), limitations=['flat-seabed assumption', 'heading (yaw) applied; roll/pitch/heave recorded per ping but not compensated',
                                                                'position accuracy inherits the vehicle navigation (typically 2-10 m)'])
    json.dump(dict(meta=meta, hazards=hazards), open(out / 'hazards.json', 'w'), indent=2, default=float)
    if hazards:
        with open(out / 'hazards.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(hazards[0].keys())); w.writeheader()
            for h in hazards: w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in h.items()})
    if image is not None:
        im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        for h in hazards:
            x1, y1, x2, y2 = [int(v) for v in h['box_xyxy_px']]; flagged = h.get('verdict') == 'no-shadow'
            col = (0, 200, 255) if flagged else (0, 220, 0)                      # amber = flagged by the shadow check (review), green = consistent
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            cv2.putText(im, f"#{h['id']} {h['classification']} {h['confidence_pct'] or ''}" + (' [no shadow - review]' if flagged else ''), (x1, max(12, y1 - 5)), 0, 0.5, col, 1)
        cv2.imwrite(str(out / f'{image_name}_annotated.png'), im)
    try:
        write_map(hazards, nav, out / 'hazards_map.html')
    except ImportError: print('folium not installed — map skipped (pip install folium)')
    return out

def write_map(hazards, nav, path):
    import folium
    colors = {'shipwreck': 'darkblue', 'aircraft': 'purple', 'human': 'red', 'pipeline': 'orange', 'fishing_gear': 'green', 'mine_like_object': 'black', 'unknown_anomaly': 'gray'}
    ok = ~np.isnan(nav.lat); centre = [float(np.nanmean(nav.lat)), float(np.nanmean(nav.lon))]
    m = folium.Map(location=centre, zoom_start=15, tiles='OpenStreetMap')
    step = max(1, len(nav) // 2000)
    folium.PolyLine([[float(a), float(b)] for a, b in zip(nav.lat[ok][::step], nav.lon[ok][::step])], color='blue', weight=2, opacity=0.6, tooltip='vehicle track').add_to(m)
    for h in hazards:
        popup = '<br>'.join(f'<b>{k}</b>: {v}' for k, v in h.items() if k in ('id', 'classification', 'confidence_pct', 'lat', 'lon', 'length_m', 'width_m', 'height_m', 'side', 'ground_range_m', 'ping_time', 'verdict', 'navigation'))
        flagged = h.get('verdict') == 'no-shadow'
        folium.Marker([h['lat'], h['lon']], popup=folium.Popup(popup, max_width=300),
                      tooltip=f"#{h['id']} {h['classification']} {h['confidence_pct']}%" + (' — no shadow: review' if flagged else ''),
                      icon=folium.Icon(color='lightgray' if flagged else colors.get(h['classification'], 'gray'), icon='question-sign' if flagged else 'warning-sign')).add_to(m)
    if nav.simulated:
        folium.map.Marker(centre, icon=folium.DivIcon(html='<div style="font-size:14px;color:red;font-weight:bold">SIMULATED NAVIGATION — positions are not real</div>')).add_to(m)
    m.save(str(path))


# ----------------------------------------------------------------------------------------------------------------------
# Simulated navigation for images that have no .xtf (demo / testing only — always labelled as simulated)
# Default start 18.85 N 72.60 E: open water in the Arabian Sea, ~20 km west of Mumbai, so a synthetic track never lands on a street map.
# ----------------------------------------------------------------------------------------------------------------------
def simulated_nav(n_rows, nadir_col, samples_per_side, lat0=18.8500, lon0=72.6000, heading_deg=90.0, speed_mps=2.0, ping_rate_hz=5.0,
                  altitude_m=8.0, slant_range_m=50.0, t0=None):
    t0 = t0 or datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc); rows = []
    for i in range(n_rows):
        lon, lat, _ = GEOD.fwd(lon0, lat0, heading_deg, i * speed_mps / ping_rate_hz)
        rows.append(dict(ping=i, time=(t0 + datetime.timedelta(seconds=i / ping_rate_hz)).isoformat(), lat=lat, lon=lon, heading_deg=heading_deg,
                         altitude_m=altitude_m, slant_range_m=slant_range_m, samples_per_side=samples_per_side))
    return NavTable(rows, nadir_col, 'simulated', simulated=True)


# ----------------------------------------------------------------------------------------------------------------------
# Self-test: write a synthetic .xtf with pyxtf, plant objects at known lat/lon, read it back, recover positions
# ----------------------------------------------------------------------------------------------------------------------
def write_synthetic_xtf(path, n_pings=400, n_samples=512, slant=50.0, alt=8.0, lat0=18.8500, lon0=72.6000, heading=90.0, speed=2.0, rate=5.0, objects=(), gps_hz=None):
    import pyxtf
    fh = pyxtf.XTFFileHeader(); fh.SonarName = b'SyntheticSSS'; fh.SonarType = pyxtf.XTFSonarType.unknown1; fh.NavUnits = pyxtf.XTFNavUnits.latlon.value
    fh.NumberOfSonarChannels = 2
    for i, t in ((0, pyxtf.XTFChannelType.port), (1, pyxtf.XTFChannelType.stbd)):
        fh.ChanInfo[i].TypeOfChannel = t.value; fh.ChanInfo[i].SubChannelNumber = i; fh.ChanInfo[i].BytesPerSample = 1; fh.ChanInfo[i].SampleFormat = pyxtf.XTFSampleFormat.byte.value
    rng = np.random.default_rng(0); t0 = datetime.datetime(2026, 1, 1, 8, 0, 0); truth = []
    with open(path, 'wb') as f:
        f.write(fh.to_bytes())
        for i in range(n_pings):
            t = t0 + datetime.timedelta(seconds=i / rate)
            i_fix = i if not gps_hz else int(i // (rate / gps_hz)) * int(rate / gps_hz)      # GPS slower than the sonar: fix repeats across pings
            lon, lat, _ = GEOD.fwd(lon0, lat0, heading, i_fix * speed / rate)
            p = pyxtf.XTFPingHeader(); p.HeaderType = pyxtf.XTFHeaderType.sonar.value; p.NumChansToFollow = 2
            p.Year, p.Month, p.Day, p.Hour, p.Minute, p.Second, p.HSeconds = t.year, t.month, t.day, t.hour, t.minute, t.second, int(t.microsecond / 1e4)
            p.PingNumber = i; p.SoundVelocity = 1500; p.SensorSpeed = speed; p.SensorXcoordinate = lon; p.SensorYcoordinate = lat
            p.SensorDepth = 20; p.SensorPrimaryAltitude = alt; p.SensorHeading = heading
            # seabed: speckle with a mild range-dependent decay, nadir gap for the water column
            samp = np.arange(n_samples); base = 90 * np.exp(-samp / (2.5 * n_samples)) * rng.gamma(4, 0.25, n_samples); base[samp * slant / n_samples < alt] = 8
            lines = [base.copy(), base.copy()]
            for obj in objects:                       # obj: dict(ping, side, ground_range_m, len_pings, width_m)
                if abs(i - obj['ping']) <= obj['len_pings'] // 2:
                    # object occupies ground range [g - w/2, g + w/2]; convert each edge to a slant-range sample index
                    g0, g1 = obj['ground_range_m'] - obj['width_m'] / 2, obj['ground_range_m'] + obj['width_m'] / 2
                    a = int(round(math.sqrt(g0 ** 2 + alt ** 2) / slant * n_samples)); b = int(round(math.sqrt(g1 ** 2 + alt ** 2) / slant * n_samples))
                    ch = 0 if obj['side'] == 'port' else 1; b = max(b, a + 1)
                    lines[ch][a:b] = 250; lines[ch][b:b + max(2, int(0.6 * (b - a)))] = 2      # highlight + acoustic shadow behind it
            c = (pyxtf.XTFPingChanHeader(), pyxtf.XTFPingChanHeader())
            for k in (0, 1): c[k].ChannelNumber = k; c[k].SlantRange = slant; c[k].Frequency = 400; c[k].NumSamples = n_samples; c[k].SampleFormat = 8
            p.ping_chan_headers = c; p.data = [np.clip(l, 0, 255).astype(np.uint8) for l in lines]
            p.NumBytesThisRecord = ctypes.sizeof(pyxtf.XTFPingHeader) + 2 * ctypes.sizeof(pyxtf.XTFPingChanHeader) + 2 * n_samples
            f.write(p.to_bytes())
    for obj in objects:
        lon, lat, _ = GEOD.fwd(lon0, lat0, heading, obj['ping'] * speed / rate)
        lon2, lat2, _ = GEOD.fwd(lon, lat, (heading + (-90 if obj['side'] == 'port' else 90)) % 360, obj['ground_range_m'])
        truth.append(dict(obj, lat=lat2, lon=lon2))
    return truth

def selftest(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); ok = True
    scenarios = [  # (name, kwargs, objects)  — B: heading 0 (north), 10 Hz sonar with 1 Hz GPS, objects right next to the track
        ('A_east_5Hz', dict(heading=90.0, rate=5.0), [dict(ping=80, side='stbd', ground_range_m=22.0, len_pings=12, width_m=2.5, label='mine_like_object'),
                                                      dict(ping=200, side='port', ground_range_m=35.0, len_pings=40, width_m=6.0, label='shipwreck'),
                                                      dict(ping=330, side='stbd', ground_range_m=41.0, len_pings=8, width_m=1.5, label='fishing_gear')]),
        ('B_north_10Hz_1HzGPS', dict(heading=0.0, rate=10.0, gps_hz=1.0), [dict(ping=100, side='port', ground_range_m=3.0, len_pings=20, width_m=2.0, label='near_nadir'),
                                                                            dict(ping=220, side='stbd', ground_range_m=11.0, len_pings=30, width_m=2.0, label='mid_range'),
                                                                            dict(ping=340, side='stbd', ground_range_m=31.0, len_pings=10, width_m=2.0, label='far_range')])]
    for name, kw, objects in scenarios:
        xtf = out / f'synthetic_{name}.xtf'; truth = write_synthetic_xtf(xtf, objects=objects, **kw)
        img, nav = read_xtf(xtf); cv2.imwrite(str(out / f'synthetic_{name}.png'), img)
        mask = (img > 235).astype(np.uint8); n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)   # blob "detector" — tests geometry, not the model
        dets = [dict(label='object', confidence_pct=99.0, box_xyxy_px=[x, y, x + w, y + h]) for x, y, w, h, a in stats[1:] if a > 20]
        hz = geotag(dets, nav, f'synthetic_{name}.png'); rate = kw['rate']
        print(f'[{name}] {len(nav)} pings, image {img.shape[1]}x{img.shape[0]}, {len(dets)} blobs, {len(truth)} planted')
        for t in truth:
            best = min(hz, key=lambda h: GEOD.inv(h['lon'], h['lat'], t['lon'], t['lat'])[2]); _, _, e = GEOD.inv(best['lon'], best['lat'], t['lon'], t['lat'])
            tl, tw = t['len_pings'] * 2.0 / rate, t['width_m']; lerr, werr = abs(best['length_m'] - tl), abs(best['width_m'] - tw)
            good = e < 1.0 and werr < 0.5 and lerr < max(0.5, 0.25 * tl); ok &= good
            print(f"  {'ok ' if good else 'BAD'} {t['label']:18s} pos err {e:.2f} m | width {best['width_m']:.2f} (true {tw}) | length {best['length_m']:.2f} (true {tl:.1f}) | {best['side']} {best['ground_range_m']} m")
        write_report(hz, nav, out / name, img, f'synthetic_{name}')
    print('SELFTEST', 'PASS' if ok else 'FAIL', f'(report in {out})'); return ok


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__ or 'sonar geotagging')
    ap.add_argument('source', nargs='?', help='.xtf survey file, or an image (.png/.jpg) with --nav')
    ap.add_argument('--detections', help='JSON from sonar_detector.py (list or {"detections": [...]})')
    ap.add_argument('--weights'); ap.add_argument('--calib', help='run sonar_detector on the rendered image')
    ap.add_argument('--nav', help='CSV navigation for an image input (columns: ping,time,lat,lon,heading_deg,altitude_m,slant_range_m,samples_per_side)')
    ap.add_argument('--nadir-col', type=int, help='column of the nadir line for image input (default: image centre)')
    ap.add_argument('--simulate-nav', action='store_true', help='image input without real navigation: generate a SIMULATED track (demo only, labelled as such)')
    ap.add_argument('--altitude', type=float, help='override towfish altitude (m) when the .xtf records 0')
    ap.add_argument('--out', default='geotag_out'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest(a.out) else 1)
    if not a.source: ap.error('source required')
    src = Path(a.source)
    if src.suffix.lower() == '.xtf':
        img, nav = read_xtf(src, altitude_override=a.altitude); name = src.stem
    elif src.suffix.lower() in ('.tif', '.tiff'):
        nav = GeoRaster(src); img = nav.image; name = src.stem; print(f'GeoTIFF {img.shape[1]}x{img.shape[0]} px, {nav.res[0]:.2f} m/px, CRS {nav.crs}')
    else:
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE); name = src.stem; nadir = a.nadir_col if a.nadir_col is not None else img.shape[1] // 2
        if a.nav: nav = NavTable.from_csv(a.nav, nadir)
        elif a.simulate_nav: nav = simulated_nav(img.shape[0], nadir, max(nadir, img.shape[1] - nadir))
        else: ap.error('image input needs --nav nav.csv or --simulate-nav')
    if a.detections:
        d = json.load(open(a.detections)); dets = d['detections'] if isinstance(d, dict) else d
    elif a.weights:
        sys.path.insert(0, str(Path(a.weights).parent)); from sonar_detector import SonarDetector
        det = SonarDetector(a.weights, a.calib or str(Path(a.weights).parent / 'calibration.json'))
        cv2.imwrite(str(Path(a.out) / f'{name}_waterfall.png'), img) if Path(a.out).mkdir(parents=True, exist_ok=True) is None else None
        dets = det.detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))['detections']
    else: ap.error('need --detections or --weights')
    hz = geotag(dets, nav, name); out = write_report(hz, nav, a.out, img, name)
    print(f'{len(hz)} hazards geotagged from {len(nav)} pings → {out}/hazards.json, hazards.csv, hazards_map.html')
    for h in hz: print(f"  #{h['id']:>2} {h['classification']:18s} {h['confidence_pct']}%  {h['lat_dms']}, {h['lon_dms']}  {h['length_m']}x{h['width_m']} m  ({h['side']}, {h['ground_range_m']} m)")

if __name__ == '__main__': main()
