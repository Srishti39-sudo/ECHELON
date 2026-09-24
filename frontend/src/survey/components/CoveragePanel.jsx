import { useMemo, useState } from "react"
import { MapContainer, TileLayer } from "react-leaflet"
import "leaflet/dist/leaflet.css"
import { Download } from "lucide-react"

import { fetchCoverage, relookUrl } from "../api"
import { copy, mapColors, surveyConfig } from "../config"
import { coverageBounds, km, km2, latDms, lonDms, m2, pct, titleCase } from "../format"
import { useSurveyDocument } from "../hooks"
import CoverageLayers, { HatchDefs } from "./CoverageLayers"

export function CoverageLegend() {
  return (
    <div className="sv-rp-legend sv-cov-legend">
      <span>
        <i className="sv-lg-box" style={{ background: "rgba(13,148,136,0.35)", borderColor: mapColors.imaged }} /> Imaged seabed
      </span>
      <span>
        <i className="sv-lg-box" style={{ background: "rgba(100,116,139,0.6)" }} /> Nadir blind strip
      </span>
      <span>
        <i className="sv-lg-box sv-lg-hatch" /> Coverage gap
      </span>
      <span>
        <i className="sv-lg-box sv-lg-dashed" /> Survey hull
      </span>
      <span>
        <i className="sv-lg-line" style={{ background: mapColors.relook }} /> Re-look line (numbered start)
      </span>
    </div>
  )
}

/** One row per re-look line. Shared with the printable report. */
export function RelookTable({ lines }) {
  if (!lines.length) return <p className="sv-muted">No re-look lines: no gap above the threshold and no contact that needs one.</p>
  return (
    <div className="sv-inner-wrap">
      <table className="sv-inner-table sv-relook-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Kind</th>
            <th>Reason</th>
            <th className="sv-col-num">Heading</th>
            <th className="sv-col-num">Length</th>
            <th>Start</th>
          </tr>
        </thead>
        <tbody>
          {lines.map((line) => {
            const p = line.properties
            return (
              <tr key={p.id}>
                <td>
                  <span className="sv-relook-num">{p.number}</span>
                </td>
                <td>{p.kind === "contact" ? "Contact, 2nd aspect" : "Coverage gap"}</td>
                <td>
                  {p.reason}
                  {p.target_class && <span className="sv-sub">{p.target_class} · {p.target_id}</span>}
                </td>
                <td className="sv-col-num">{p.heading_deg.toFixed(1)}°</td>
                <td className="sv-col-num">{Math.round(p.length_m)} m</td>
                <td className="sv-nowrap">
                  {latDms(p.start_lat)}
                  <span className="sv-sub">{lonDms(p.start_lon)}</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function CoverageView({ surveyId, coverage }) {
  const [basemap, setBasemap] = useState(true)
  const bounds = useMemo(() => coverageBounds(coverage), [coverage])
  const m = coverage.metrics
  const lines = coverage.relook_lines?.features || []
  const byReason = Object.entries(m.degraded_gap_by_reason_km2 || {})
    .map(([reason, value]) => `${titleCase(reason)} ${m2(value)}`)
    .join(" · ")

  return (
    <>
      <div className="sv-cov-metrics">
        <div className="sv-metric">
          <span>Seabed imaged</span>
          <strong>{km2(m.imaged_km2)}</strong>
          <small>{m2(m.imaged_km2)} over {km(m.track_length_km)} of track</small>
        </div>
        <div className="sv-metric">
          <span>Share of survey hull imaged</span>
          <strong>{pct(m.imaged_pct_of_hull)}</strong>
          <small>hull {km2(m.hull_km2)}; {pct(m.imaged_pct_of_footprint)} of the swath footprint</small>
        </div>
        <div className="sv-metric">
          <span>Nadir blind strip</span>
          <strong>{m2(m.nadir_blind_km2)}</strong>
          <small>mean width {m.mean_nadir_strip_width_m ?? "—"} m under the towfish</small>
        </div>
        <div className="sv-metric is-alert">
          <span>Degraded-row gaps</span>
          <strong>{m2(m.degraded_gap_km2)}</strong>
          <small>{byReason || "none"}</small>
        </div>
        <div className="sv-metric">
          <span>Re-look lines</span>
          <strong>{m.relook_lines}</strong>
          <small>
            {m.relook_lines_gaps} for gaps, {m.relook_lines_contacts} for contacts
          </small>
        </div>
      </div>

      <div className="sv-cov-body">
        <div className="sv-rp-map-wrap">
          <MapContainer bounds={bounds} boundsOptions={{ padding: [24, 24] }} scrollWheelZoom={false} className="sv-rp-map" maxZoom={21}>
            {basemap && (
              <TileLayer
                url={surveyConfig.basemap.url}
                attribution={surveyConfig.basemap.attribution}
                maxNativeZoom={surveyConfig.basemap.maxNativeZoom}
                maxZoom={21}
              />
            )}
            <CoverageLayers coverage={coverage} />
          </MapContainer>
          <label className="sv-rp-basemap">
            <input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} />
            Basemap
          </label>
        </div>

        <div className="sv-cov-side">
          <CoverageLegend />
          <div className="sv-cov-downloads">
            <a className="sv-btn" href={relookUrl(surveyId, "gpx")} download>
              <Download size={15} />
              Re-look lines (GPX)
            </a>
            <a className="sv-btn" href={relookUrl(surveyId, "csv")} download>
              <Download size={15} />
              Re-look lines (CSV)
            </a>
          </div>
          <details className="sv-reasons sv-cov-method">
            <summary>How this was computed</summary>
            <ul>
              <li>{coverage.method.footprint}</li>
              <li>{coverage.method.imaged}</li>
              {coverage.method.strips.map((strip) => (
                <li key={strip.strip}>
                  <strong>{strip.strip}</strong>: {strip.half_width} {strip.nadir}
                </li>
              ))}
              <li>{coverage.method.relook}</li>
              {coverage.limitations.map((text) => (
                <li key={text}>{text}</li>
              ))}
              {coverage.references.map((ref) => (
                <li key={ref.url}>
                  <a href={ref.url} target="_blank" rel="noreferrer">
                    {ref.title}
                  </a>
                  : {ref.used_for}
                </li>
              ))}
            </ul>
          </details>
        </div>
      </div>

      <h3 className="sv-detail-sub">Re-look lines</h3>
      <RelookTable lines={lines} />
    </>
  )
}

/** The coverage and blind-spot panel: GET /survey/{id}/coverage. */
function CoveragePanel({ surveyId }) {
  const { data, error, loading } = useSurveyDocument(fetchCoverage, surveyId)

  return (
    <section className="sv-panel" aria-labelledby="sv-coverage-title">
      <HatchDefs />
      <div className="sv-panel-head">
        <div>
          <h2 id="sv-coverage-title">
            {copy.coverageTitle}
            {data?.synthetic && <span className="sv-flag sv-flag-synthetic">{copy.synthetic}</span>}
          </h2>
          <p>{copy.coverageNote}</p>
        </div>
      </div>
      {loading && <p className="sv-muted">Computing coverage…</p>}
      {error && <p className="sv-error">{error.message}</p>}
      {data && !data.available && (
        <p className="sv-notice">
          {copy.coverageUnavailable}
          {data.reason && <span className="sv-sub">{data.reason}</span>}
        </p>
      )}
      {data?.available && <CoverageView key={surveyId} surveyId={surveyId} coverage={data} />}
    </section>
  )
}

export default CoveragePanel
