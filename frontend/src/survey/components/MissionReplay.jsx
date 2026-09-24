import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import L from "leaflet"
import {
  CircleMarker,
  MapContainer,
  Marker,
  Polygon,
  Polyline,
  Popup,
  TileLayer,
} from "react-leaflet"
import "leaflet/dist/leaflet.css"
import { Pause, Play, RotateCcw } from "lucide-react"

import { fetchReplay } from "../api"
import { copy, mapColors, replaySpeeds, surveyConfig } from "../config"
import {
  boundsOf,
  clock,
  escapeHtml,
  km,
  km2,
  lastIndexAtOrBefore,
  pct,
  swathEdges,
} from "../format"
import { usePrefersReducedMotion, useSurveyDocument } from "../hooks"

const QUALITY_LABEL = {
  dropout: "dropout",
  attitude: "attitude (motion)",
  interpolated: "interpolated",
  nav_jump: "navigation jump",
}

function tierColor(tier) {
  return mapColors.tier[tier] || mapColors.tierFallback
}

function contactIcon(detection, animate) {
  const filtered = detection.suppressed === true
  const color = filtered ? mapColors.filtered : tierColor(detection.tier)
  const label = `${detection.class ?? "contact"} ${pct(detection.confidence_pct)}`
  return L.divIcon({
    className: `sv-rp-contact${filtered ? " is-filtered" : ""}${animate ? " is-animated" : ""}`,
    html:
      `<span class="sv-rp-dot" style="--c:${color}"></span>` +
      `<span class="sv-rp-label">${escapeHtml(label)}${filtered ? " · filtered" : ""}</span>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  })
}

/** Contiguous runs of the same strip and quality: [{ from, to, quality }]. */
function qualityRuns(track) {
  const runs = []
  const n = track.t.length
  let start = 0
  for (let i = 1; i <= n; i += 1) {
    if (i === n || track.quality[i] !== track.quality[start] || track.strip[i] !== track.strip[start]) {
      // The boundary point joins the run it ends, so the track has no breaks.
      const joins = i < n && track.strip[i] === track.strip[start]
      runs.push({ from: start, to: joins ? i : i - 1, quality: track.quality[start] })
      start = i
    }
  }
  return runs
}

/**
 * The replay itself, for one loaded document. Keyed by survey in the parent,
 * so every piece of state here belongs to exactly one survey.
 */
function ReplayView({ data }) {
  const reducedMotion = usePrefersReducedMotion()
  const track = data.track
  const duration = data.duration_s || 0
  const count = track.t.length

  // Reduced motion opens on the finished survey, so nothing has to animate
  // for the picture to be complete. Playback is still one click away.
  const [t, setT] = useState(() => (reducedMotion ? duration : 0))
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(10)
  const [basemap, setBasemap] = useState(true)
  const clockRef = useRef(t)

  const latlngs = useMemo(() => track.lat.map((lat, i) => [lat, track.lon[i]]), [track])
  const edges = useMemo(
    () =>
      track.lat.map((lat, i) =>
        swathEdges(lat, track.lon[i], track.heading[i], track.half_width_port_m[i], track.half_width_stbd_m[i]),
      ),
    [track],
  )
  const runs = useMemo(() => qualityRuns(track), [track])
  const bounds = useMemo(() => boundsOf(edges.flat()), [edges])
  const icons = useMemo(
    () => new Map(data.detections.map((d) => [d.id, contactIcon(d, !reducedMotion)])),
    [data.detections, reducedMotion],
  )

  useEffect(() => {
    if (!playing) return undefined
    let frame = 0
    let last = null
    const tick = (now) => {
      if (last !== null) {
        const next = Math.min(duration, clockRef.current + ((now - last) / 1000) * speed)
        clockRef.current = next
        setT(next)
        if (next >= duration) {
          setPlaying(false)
          return
        }
      }
      last = now
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, speed, duration])

  const seek = useCallback((value) => {
    clockRef.current = value
    setT(value)
  }, [])

  const togglePlay = () => {
    if (!playing && clockRef.current >= duration) seek(0)
    setPlaying((on) => !on)
  }

  // Where the towfish is now: between point i and i + 1.
  const i = Math.max(0, lastIndexAtOrBefore(track.t, t))
  const j = Math.min(count - 1, i + 1)
  const span = track.t[j] - track.t[i]
  const w = span > 0 && track.strip[i] === track.strip[j] ? Math.min(1, Math.max(0, (t - track.t[i]) / span)) : 0
  const lerp = (a, b) => a + (b - a) * w
  const head = [lerp(track.lat[i], track.lat[j]), lerp(track.lon[i], track.lon[j])]
  const headEdges = [
    [lerp(edges[i][0][0], edges[j][0][0]), lerp(edges[i][0][1], edges[j][0][1])],
    [lerp(edges[i][1][0], edges[j][1][0]), lerp(edges[i][1][1], edges[j][1][1])],
  ]
  const started = t > 0 || count === 1
  const quality = track.quality[i]

  const drawn = runs
    .filter((run) => run.from <= i && started)
    .map((run) => {
      const end = Math.min(run.to, i)
      const partial = end === i && run.to > i
      const line = latlngs.slice(run.from, end + 1)
      const port = edges.slice(run.from, end + 1).map((e) => e[0])
      const stbd = edges.slice(run.from, end + 1).map((e) => e[1])
      if (partial) {
        line.push(head)
        port.push(headEdges[0])
        stbd.push(headEdges[1])
      }
      return { ...run, line, swath: [...port, ...stbd.reverse()] }
    })

  const distance = lerp(track.distance_km[i], track.distance_km[j])
  const area = lerp(track.area_km2[i], track.area_km2[j])
  const shown = data.detections.filter((d) => d.t <= t && started)
  const found = shown.filter((d) => d.suppressed !== true).length
  const filtered = shown.length - found
  const latest = shown[shown.length - 1]

  return (
    <div className="sv-rp">
      <div className="sv-rp-map-wrap">
        <MapContainer
          bounds={bounds}
          boundsOptions={{ padding: [24, 24] }}
          scrollWheelZoom={false}
          className="sv-rp-map"
          maxZoom={21}
        >
          {basemap && (
            <TileLayer
              url={surveyConfig.basemap.url}
              attribution={surveyConfig.basemap.attribution}
              maxNativeZoom={surveyConfig.basemap.maxNativeZoom}
              maxZoom={21}
            />
          )}

          {/* The whole planned line, faint, so the replay has somewhere to go. */}
          <Polyline
            positions={latlngs}
            interactive={false}
            pathOptions={{ color: mapColors.trackFuture, weight: 1.5, dashArray: "3 5", opacity: 0.8 }}
          />

          {drawn.map((run) => {
            const bad = run.quality !== "ok"
            return (
              <Polygon
                key={`swath-${run.from}`}
                positions={run.swath}
                interactive={false}
                pathOptions={{
                  stroke: false,
                  fillColor: bad ? mapColors.swathDegraded : mapColors.swath,
                  fillOpacity: bad ? 0.28 : 0.16,
                  fillRule: "nonzero",
                }}
              />
            )
          })}
          {drawn.map((run) => {
            const bad = run.quality !== "ok"
            return (
              <Polyline
                key={`track-${run.from}`}
                positions={run.line}
                interactive={false}
                pathOptions={{
                  color: bad ? mapColors.degraded : mapColors.track,
                  weight: bad ? 5 : 3,
                  opacity: 0.95,
                }}
              />
            )
          })}

          {started && (
            <>
              <Polyline
                positions={headEdges}
                interactive={false}
                pathOptions={{
                  color: quality === "ok" ? mapColors.sweep : mapColors.degraded,
                  weight: 3,
                  opacity: 0.9,
                }}
              />
              <CircleMarker
                center={head}
                radius={6}
                interactive={false}
                pathOptions={{ color: "#fff", weight: 2, fillColor: mapColors.track, fillOpacity: 1 }}
              />
            </>
          )}

          {shown.map((d) =>
            d.lat == null || d.lon == null ? null : (
              <Marker key={d.id} position={[d.lat, d.lon]} icon={icons.get(d.id)}>
                <Popup>
                  <strong>{d.class}</strong>
                  {d.suppressed === true && <em> (filtered)</em>}
                  <br />
                  Confidence {pct(d.confidence_pct)}, {d.tier} tier
                  <br />
                  {d.recommended_action}
                  <br />
                  <small>
                    {d.id} at {clock(d.t)}
                  </small>
                </Popup>
              </Marker>
            ),
          )}
        </MapContainer>

        <label className="sv-rp-basemap">
          <input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} />
          Basemap
        </label>
      </div>

      <div className="sv-rp-side">
        <div className="sv-rp-counters" aria-live="off">
          <div>
            <span>Distance surveyed</span>
            <strong>{km(distance)}</strong>
          </div>
          <div>
            <span>Area imaged</span>
            <strong>{km2(area)}</strong>
          </div>
          <div>
            <span>Contacts found</span>
            <strong>{found}</strong>
          </div>
          <div>
            <span>Filtered</span>
            <strong>{filtered}</strong>
          </div>
        </div>

        <div className={`sv-rp-status${quality !== "ok" && started ? " is-degraded" : ""}`}>
          <span className="sv-rp-status-dot" />
          {quality !== "ok" && started
            ? `Degraded sonar: ${QUALITY_LABEL[quality] || quality}`
            : "Sonar rows good"}
        </div>

        <p className="sv-sr-only" aria-live="polite">
          {latest ? `Contact: ${latest.class}, ${pct(latest.confidence_pct)}` : ""}
        </p>

        <ol className="sv-rp-log">
          {shown.length === 0 && <li className="sv-muted">No contacts passed yet.</li>}
          {shown
            .slice()
            .reverse()
            .map((d) => (
              <li key={d.id} className={d.suppressed === true ? "is-filtered" : undefined}>
                <span className="sv-tier-dot" style={{ background: d.suppressed === true ? mapColors.filtered : tierColor(d.tier) }} />
                <span className="sv-rp-log-class">{d.class}</span>
                <span className="sv-rp-log-pct">{pct(d.confidence_pct)}</span>
                <span className="sv-rp-log-time">{clock(d.t)}</span>
                {d.suppressed === true && <span className="sv-flag sv-flag-filtered">{copy.filteredChip}</span>}
              </li>
            ))}
        </ol>
      </div>

      <div className="sv-rp-controls">
        <button
          type="button"
          className="sv-btn sv-btn-primary sv-rp-play"
          onClick={togglePlay}
          aria-label={playing ? "Pause replay" : "Play replay"}
        >
          {playing ? <Pause size={15} /> : <Play size={15} />}
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          className="sv-btn"
          onClick={() => {
            setPlaying(false)
            seek(0)
          }}
          aria-label="Back to the start"
        >
          <RotateCcw size={15} />
        </button>
        <div className="sv-rp-speeds" role="group" aria-label="Replay speed">
          {replaySpeeds.map((value) => (
            <button
              key={value}
              type="button"
              className={`sv-rp-speed${speed === value ? " is-active" : ""}`}
              aria-pressed={speed === value}
              onClick={() => setSpeed(value)}
            >
              {value}×
            </button>
          ))}
        </div>
        <span className="sv-rp-clock">
          {clock(t)} / {clock(duration)}
        </span>
      </div>

      <div className="sv-rp-timeline">
        <div className="sv-rp-bar" aria-hidden="true">
          {data.degraded.map((gap) => (
            <span
              key={`${gap.strip}-${gap.rows[0]}`}
              className="sv-rp-bar-gap"
              title={`${QUALITY_LABEL[gap.reason] || gap.reason}, rows ${gap.rows[0]}-${gap.rows[1]}`}
              style={{
                left: `${(100 * gap.t_start) / (duration || 1)}%`,
                width: `max(2px, ${(100 * (gap.t_end - gap.t_start)) / (duration || 1)}%)`,
              }}
            />
          ))}
          {data.detections.map((d) => (
            <span
              key={d.id}
              className={`sv-rp-bar-tick${d.suppressed === true ? " is-filtered" : ""}`}
              style={{
                left: `${(100 * d.t) / (duration || 1)}%`,
                background: d.suppressed === true ? mapColors.filtered : tierColor(d.tier),
              }}
            />
          ))}
          <span className="sv-rp-bar-fill" style={{ width: `${(100 * t) / (duration || 1)}%` }} />
        </div>
        <input
          type="range"
          className="sv-rp-scrub"
          min={0}
          max={duration}
          step={Math.max(duration / 1000, 0.01)}
          value={t}
          onChange={(event) => {
            setPlaying(false)
            seek(Number(event.target.value))
          }}
          aria-label="Replay time"
          aria-valuetext={`${clock(t)} of ${clock(duration)}`}
        />
        <div className="sv-rp-legend">
          <span>
            <i className="sv-lg-line" style={{ background: mapColors.track }} /> Track, good rows
          </span>
          <span>
            <i className="sv-lg-line" style={{ background: mapColors.degraded }} /> Degraded rows
          </span>
          <span>
            <i className="sv-lg-box" style={{ background: "rgba(8,127,156,0.2)" }} /> Swath swept
          </span>
          <span>
            <i className="sv-lg-dot" style={{ background: mapColors.tier.medium }} /> Contact (tier colour)
          </span>
        </div>
      </div>

      <p className="sv-rp-basis">
        {data.start_time && <>Survey start {data.start_time.replace("T", " ").replace("+00:00", " UTC")}. </>}
        {data.basis}
      </p>
    </div>
  )
}

/** The mission replay panel: fetches GET /survey/{id}/replay and plays it. */
function MissionReplay({ surveyId }) {
  const { data, error, loading } = useSurveyDocument(fetchReplay, surveyId)

  return (
    <section className="sv-panel" aria-labelledby="sv-replay-title">
      <div className="sv-panel-head">
        <div>
          <h2 id="sv-replay-title">
            {copy.replayTitle}
            {data?.synthetic && <span className="sv-flag sv-flag-synthetic">{copy.synthetic}</span>}
          </h2>
          <p>{copy.replayNote}</p>
        </div>
      </div>
      {loading && <p className="sv-muted">Loading replay…</p>}
      {error && <p className="sv-error">{error.message}</p>}
      {data && !data.available && (
        <p className="sv-notice">
          {copy.replayUnavailable}
          {data.reason && <span className="sv-sub">{data.reason}</span>}
        </p>
      )}
      {data?.available && data.track?.t?.length > 0 && <ReplayView key={surveyId} data={data} />}
    </section>
  )
}

export default MissionReplay
