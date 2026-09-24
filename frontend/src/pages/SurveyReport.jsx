import { useMemo } from "react"
import { Link, useParams } from "react-router-dom"
import L from "leaflet"
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer } from "react-leaflet"
import "leaflet/dist/leaflet.css"
import { ArrowLeft, Printer } from "lucide-react"

import {
  fetchCoverage,
  fetchExport,
  fetchGhostTraceOrNull,
  fetchReplay,
  listSurveys,
} from "../survey/api"
import { copy, mapColors, styleForTier, surveyConfig } from "../survey/config"
import {
  confidencePct,
  dimensions,
  hardReasonLabels,
  isVerified,
  verificationReasons,
} from "../survey/detections"
import {
  boundsOf,
  coverageBounds,
  escapeHtml,
  geometryLatLngs,
  km,
  km2,
  latDms,
  lonDms,
  m2,
  pct,
  titleCase,
} from "../survey/format"
import { useSurveyDocument } from "../survey/hooks"
import CoverageLayers, { HatchDefs } from "../survey/components/CoverageLayers"
import { CoverageLegend, RelookTable } from "../survey/components/CoveragePanel"
import "../survey/survey.css"
import "./survey-report.css"

/**
 * Everything the report needs, fetched once through the existing survey APIs.
 *
 * Only export.json is required. Coverage and replay answer { available: false }
 * for a survey without navigation, and GhostTrace is asked for only when the
 * survey list says the survey has one, so an ordinary survey produces no 404.
 */
async function loadReport(surveyId) {
  const [exportDoc, surveys] = await Promise.all([
    fetchExport(surveyId),
    listSurveys().catch(() => []),
  ])
  const meta = surveys.find((s) => s.survey_id === surveyId) || null
  const unavailable = (error) => ({ available: false, reason: error.message })
  const [coverage, replay, ghosttrace] = await Promise.all([
    fetchCoverage(surveyId).catch(unavailable),
    fetchReplay(surveyId).catch(unavailable),
    meta?.has_ghosttrace ? fetchGhostTraceOrNull(surveyId).catch(() => null) : Promise.resolve(null),
  ])
  return { exportDoc, meta, coverage, replay, ghosttrace, generatedAt: new Date() }
}

function finite(value) {
  return typeof value === "number" && Number.isFinite(value)
}

function position(detection) {
  if (finite(detection.latitude) && finite(detection.longitude)) {
    return (
      <>
        {latDms(detection.latitude)}
        <span className="rp-sub">{lonDms(detection.longitude)}</span>
      </>
    )
  }
  return (
    <>
      x {Math.round(detection.global_x)}, y {Math.round(detection.global_y)} px
      <span className="rp-sub">relative, not a position</span>
    </>
  )
}

function sizeText(detection) {
  const size = dimensions(detection)
  if (!size) return "—"
  return size.label && !size.label.startsWith("L ×") ? `${size.text} (${size.label})` : size.text
}

function timestamp(value) {
  if (!value) return "—"
  const date = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`
}

function TierChip({ tier }) {
  const style = styleForTier(tier)
  return (
    <span className="rp-tier" style={{ background: style.tint, color: style.color }}>
      {style.label}
    </span>
  )
}

function numberIcon(number, tier) {
  const color = mapColors.tier[tier] || mapColors.tierFallback
  return L.divIcon({
    className: "rp-map-num",
    html: `<span style="background:${color}">${escapeHtml(number)}</span>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  })
}

/** A non-interactive snapshot of the track, coverage and numbered hazards. */
function ReportMap({ coverage, replay, hazards }) {
  const track = useMemo(
    () => (replay?.available ? replay.track.lat.map((lat, i) => [lat, replay.track.lon[i]]) : []),
    [replay],
  )
  const located = useMemo(
    () => hazards.filter((h) => finite(h.detection.latitude) && finite(h.detection.longitude)),
    [hazards],
  )
  const icons = useMemo(
    () => new Map(hazards.map((h) => [h.detection.id, numberIcon(h.number, h.detection.severity_tier)])),
    [hazards],
  )
  const bounds = useMemo(() => {
    const points = [...track, ...located.map((h) => [h.detection.latitude, h.detection.longitude])]
    if (coverage?.available) {
      const extra = coverageBounds(coverage)
      if (extra) points.push(...extra)
    }
    return boundsOf(points)
  }, [track, located, coverage])

  if (!bounds) {
    return (
      <p className="rp-note">
        No map: this survey has no geographic positions (relative survey coordinates), and
        nothing is placed on a chart without one.
      </p>
    )
  }

  const hull = coverage?.polygons?.features?.find((f) => f.properties?.kind === "hull")
  const hullPoints = hull ? geometryLatLngs(hull.geometry) : []

  return (
    <div className="rp-map-wrap">
      <MapContainer
        bounds={hullPoints.length ? boundsOf([...hullPoints, ...track]) : bounds}
        boundsOptions={{ padding: [18, 18] }}
        className="rp-map"
        dragging={false}
        zoomControl={false}
        scrollWheelZoom={false}
        doubleClickZoom={false}
        touchZoom={false}
        boxZoom={false}
        keyboard={false}
        zoomSnap={0.25}
        maxZoom={21}
      >
        <TileLayer
          url={surveyConfig.basemap.url}
          attribution={surveyConfig.basemap.attribution}
          maxNativeZoom={surveyConfig.basemap.maxNativeZoom}
          maxZoom={21}
        />
        {coverage?.available && <CoverageLayers coverage={coverage} interactive={false} />}
        {track.length > 1 && (
          <Polyline
            positions={track}
            interactive={false}
            pathOptions={{ color: mapColors.track, weight: 2.5, opacity: 0.95 }}
          />
        )}
        {track.length > 0 && (
          <CircleMarker
            center={track[0]}
            radius={4}
            interactive={false}
            pathOptions={{ color: "#fff", weight: 1.5, fillColor: mapColors.track, fillOpacity: 1 }}
          />
        )}
        {located.map((h) => (
          <Marker
            key={h.detection.id}
            position={[h.detection.latitude, h.detection.longitude]}
            icon={icons.get(h.detection.id)}
            interactive={false}
            keyboard={false}
          />
        ))}
      </MapContainer>
      <div className="rp-map-legend">
        <span>
          <i className="sv-lg-line" style={{ background: mapColors.track }} /> Towfish track (dot = start)
        </span>
        <span>
          <i className="rp-legend-num">1</i> Hazard, numbered as in the table, coloured by tier
        </span>
      </div>
      {coverage?.available && <CoverageLegend />}
    </div>
  )
}

function Section({ title, children, breakBefore = false, id }) {
  return (
    <section className={`rp-section${breakBefore ? " rp-break" : ""}`} aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      {children}
    </section>
  )
}

function GhostTraceSummary({ doc }) {
  const targets = [...(doc.targets || [])].sort(
    (a, b) => (a.priority?.rank ?? 1e9) - (b.priority?.rank ?? 1e9),
  )
  return (
    <>
      <p className="rp-note">
        GhostTrace ranks suspected ghost gear for recovery. Its priority terms are a
        configurable heuristic chosen for this project, not fitted to recovery data and not an
        official procedure. {doc.synthetic_inputs || doc.demo ? "This run used synthetic or demo inputs." : ""}
      </p>
      <table className="rp-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Target</th>
            <th>Priority</th>
            <th>Activity</th>
            <th>Nearest habitat</th>
            <th>Drift, top impact</th>
            <th>Authorities named</th>
          </tr>
        </thead>
        <tbody>
          {targets.map((target) => {
            const nearest = target.habitat?.available ? target.habitat.nearest?.[0] : null
            const drift = target.drift || {}
            const impacts = [...(drift.impacts || [])].sort(
              (a, b) => (b.probability ?? 0) - (a.probability ?? 0),
            )
            const top = impacts[0]
            return (
              <tr key={target.detection_id}>
                <td className="rp-num">{target.priority?.rank ?? "—"}</td>
                <td>
                  {target.object_class}
                  <span className="rp-sub">{target.detection_id}</span>
                </td>
                <td>
                  {titleCase(target.priority?.tier || "—")}
                  <span className="rp-sub">score {finite(target.priority?.score) ? target.priority.score.toFixed(3) : "—"}</span>
                </td>
                <td>
                  {target.activity?.available ? titleCase(target.activity.level) : "not assessed"}
                  {finite(target.activity?.score) && <span className="rp-sub">score {target.activity.score.toFixed(2)}</span>}
                </td>
                <td>
                  {nearest ? (
                    <>
                      {nearest.name || titleCase(nearest.kind)}
                      <span className="rp-sub">
                        {titleCase(nearest.kind)}, {finite(nearest.distance_m) ? km(nearest.distance_m / 1000) : "—"}
                      </span>
                    </>
                  ) : (
                    "none mapped nearby"
                  )}
                </td>
                <td>
                  {drift.available === false
                    ? "forecast not run"
                    : top
                      ? (
                        <>
                          {top.name} ({titleCase(top.kind)})
                          <span className="rp-sub">{pct(100 * (top.probability ?? 0), 0)} within {drift.horizon_hours} h</span>
                        </>
                      )
                      : `no mapped habitat reached in ${drift.horizon_hours ?? "—"} h`}
                </td>
                <td>{(target.alert?.authorities || []).map((a) => a.name).join("; ") || "—"}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {(doc.caveats || []).length > 0 && (
        <ul className="rp-list">
          {doc.caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      )}
    </>
  )
}

function Report({ data }) {
  const { exportDoc, coverage, replay, ghosttrace, generatedAt } = data
  const metadata = exportDoc.metadata || {}
  const summary = exportDoc.survey_summary || {}
  const provenance = exportDoc.provenance || {}
  const configuration = exportDoc.configuration || {}
  const detections = useMemo(() => exportDoc.detections || [], [exportDoc])
  const hazards = useMemo(
    () =>
      detections
        .filter((d) => d.suppressed !== true)
        .sort((a, b) => (b.severity ?? 0) - (a.severity ?? 0))
        .map((detection, index) => ({ detection, number: index + 1 })),
    [detections],
  )
  const filtered = detections.filter((d) => d.suppressed === true)
  const hotspots = exportDoc.hotspots || []
  const verified = isVerified(exportDoc)
  const demo = Boolean(metadata.demo)
  const cov = coverage?.available ? coverage : null
  const lines = cov?.relook_lines?.features || []
  const top = summary.highest_priority_hotspot
  const navigation = provenance.navigation || {}

  return (
    <article className="rp-doc">
      <HatchDefs />

      {/* COVER */}
      <header className="rp-cover">
        <div className="rp-brand">DeepEcho · Survey Hazard Report</div>
        {demo && (
          <div className="rp-synthetic" role="note">
            <strong>SYNTHETIC DATA</strong>
            <span>{metadata.demo_warning || copy.demoNote}</span>
          </div>
        )}
        <h1>{metadata.title || metadata.survey_id}</h1>
        <dl className="rp-cover-facts">
          <div>
            <dt>Survey ID</dt>
            <dd>{metadata.survey_id}</dd>
          </div>
          <div>
            <dt>Processed</dt>
            <dd>{timestamp(metadata.processed_at)}</dd>
          </div>
          <div>
            <dt>Survey start</dt>
            <dd>{replay?.available && replay.start_time ? timestamp(replay.start_time) : "not recorded"}</dd>
          </div>
          <div>
            <dt>Coordinate mode</dt>
            <dd>
              {summary.coordinate_mode || metadata.coordinate_mode}
              {navigation.mode && <span className="rp-sub">navigation: {navigation.mode}</span>}
            </dd>
          </div>
          <div>
            <dt>Pipeline</dt>
            <dd>
              {metadata.engine} {metadata.processing_version}
              <span className="rp-sub">severity policy {provenance.severity_policy_version || "—"}</span>
            </dd>
          </div>
          <div>
            <dt>Detector</dt>
            <dd>
              {metadata.detector || metadata.model_name}
              <span className="rp-sub">
                model {metadata.model_name}
                {finite(metadata.confidence_threshold) ? `, threshold ${metadata.confidence_threshold}` : ""}
              </span>
            </dd>
          </div>
          <div>
            <dt>Strips / tiles</dt>
            <dd>
              {summary.strips_processed ?? "—"} / {summary.tiles_processed ?? "—"}
            </dd>
          </div>
          <div>
            <dt>Report generated</dt>
            <dd>{timestamp(generatedAt)}</dd>
          </div>
        </dl>
      </header>

      {/* EXECUTIVE SUMMARY */}
      <Section title="Executive summary" id="rp-summary">
        <div className="rp-kpis">
          <div>
            <strong>{hazards.length}</strong>
            <span>hazards retained</span>
          </div>
          <div className={summary.detections_by_tier?.critical ? "is-critical" : undefined}>
            <strong>{summary.detections_by_tier?.critical ?? 0}</strong>
            <span>critical</span>
          </div>
          <div>
            <strong>{verified ? summary.suppressed_detections ?? filtered.length : "—"}</strong>
            <span>filtered false positives</span>
          </div>
          <div>
            <strong>{summary.total_hotspots ?? hotspots.length}</strong>
            <span>hotspots</span>
          </div>
          <div>
            <strong>{cov ? pct(cov.metrics.imaged_pct_of_hull) : "—"}</strong>
            <span>{cov ? "of survey hull imaged" : "coverage not computable"}</span>
          </div>
          <div>
            <strong>{cov ? cov.metrics.relook_lines : "—"}</strong>
            <span>re-look lines</span>
          </div>
        </div>
        <p className="rp-lead">
          {summary.total_deduplicated_detections ?? detections.length} detection(s) after
          deduplication ({summary.duplicates_removed ?? 0} duplicate(s) merged)
          {verified ? `, of which ${filtered.length} were filtered by verification as likely false positives` : ""}.
          {top
            ? ` The highest-priority hotspot is ${top.hotspot_id} (${top.dominant_class}, total severity ${top.total_severity}); recommended action: ${top.recommended_action}.`
            : " No hotspot was ranked."}
          {cov
            ? ` The sonar imaged ${km2(cov.metrics.imaged_km2)} of seabed along ${km(cov.metrics.track_length_km)} of track; ${m2(cov.metrics.degraded_gap_km2)} fell in degraded rows and ${m2(cov.metrics.nadir_blind_km2)} in the nadir blind strip.`
            : ` ${copy.coverageUnavailable}`}
        </p>
      </Section>

      {/* MAP */}
      <Section title="Survey map" id="rp-map">
        <ReportMap coverage={cov} replay={replay} hazards={hazards} />
      </Section>

      {/* HAZARDS */}
      <Section title="Hazards" id="rp-hazards" breakBefore>
        {hazards.length === 0 ? (
          <p className="rp-note">{copy.noHotspots}</p>
        ) : (
          <table className="rp-table">
            <thead>
              <tr>
                <th>#</th>
                <th>ID</th>
                <th>Class</th>
                <th className="rp-num">Conf.</th>
                <th>Position</th>
                <th>L × W × H</th>
                <th>Tier</th>
                <th>Recommended action</th>
              </tr>
            </thead>
            <tbody>
              {hazards.map(({ detection, number }) => (
                <tr key={detection.id}>
                  <td className="rp-num">{number}</td>
                  <td className="rp-id">{detection.id}</td>
                  <td>{detection.object_class}</td>
                  <td className="rp-num">
                    {confidencePct(detection) !== null ? pct(confidencePct(detection)) : pct(100 * detection.confidence)}
                  </td>
                  <td className="rp-nowrap">{position(detection)}</td>
                  <td className="rp-nowrap">{sizeText(detection)}</td>
                  <td>
                    <TierChip tier={detection.severity_tier} />
                  </td>
                  <td>{detection.recommended_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="rp-small">
          Confidence is the engine's verified 0-100 figure where verification ran, otherwise the
          detector's score. Sizes are in metres only where the strip's resolution is known.
        </p>
      </Section>

      <Section title="Filtered false positives" id="rp-filtered">
        {!verified ? (
          <p className="rp-note">This export predates verification; nothing was filtered or measured.</p>
        ) : filtered.length === 0 ? (
          <p className="rp-note">{copy.filteredNone}</p>
        ) : (
          <table className="rp-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Class</th>
                <th className="rp-num">Conf.</th>
                <th>Position</th>
                <th>Why it was filtered</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((detection) => (
                <tr key={detection.id}>
                  <td className="rp-id">{detection.id}</td>
                  <td>{detection.object_class}</td>
                  <td className="rp-num">{pct(confidencePct(detection))}</td>
                  <td className="rp-nowrap">{position(detection)}</td>
                  <td>
                    {hardReasonLabels(detection).join("; ") || "—"}
                    {verificationReasons(detection).slice(0, 2).map((reason) => (
                      <span key={reason} className="rp-sub">
                        {reason}
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {verified && filtered.length > 0 && <p className="rp-small">{copy.filteredNote}</p>}
      </Section>

      <Section title="Hotspots" id="rp-hotspots">
        {hotspots.length === 0 ? (
          <p className="rp-note">{copy.noHotspots}</p>
        ) : (
          <table className="rp-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Hotspot</th>
                <th>Dominant class</th>
                <th>Tier</th>
                <th className="rp-num">Detections</th>
                <th className="rp-num">Total severity</th>
                <th>Centroid</th>
                <th>Recommended action</th>
              </tr>
            </thead>
            <tbody>
              {hotspots.map((hotspot) => (
                <tr key={hotspot.hotspot_id}>
                  <td className="rp-num">{hotspot.priority_rank}</td>
                  <td>{hotspot.hotspot_id}</td>
                  <td>{hotspot.dominant_class}</td>
                  <td>
                    <TierChip tier={hotspot.severity_tier} />
                  </td>
                  <td className="rp-num">{hotspot.detection_count}</td>
                  <td className="rp-num">{finite(hotspot.total_severity) ? hotspot.total_severity.toFixed(3) : "—"}</td>
                  <td className="rp-nowrap">
                    {finite(hotspot.centroid?.latitude)
                      ? (
                        <>
                          {latDms(hotspot.centroid.latitude)}
                          <span className="rp-sub">{lonDms(hotspot.centroid.longitude)}</span>
                        </>
                      )
                      : `x ${Math.round(hotspot.centroid?.global_x ?? 0)}, y ${Math.round(hotspot.centroid?.global_y ?? 0)} px`}
                  </td>
                  <td>{hotspot.recommended_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>

      {/* COVERAGE */}
      <Section title="Coverage and re-look lines" id="rp-coverage" breakBefore>
        {!cov ? (
          <p className="rp-note">
            {copy.coverageUnavailable}
            {coverage?.reason && <span className="rp-sub">{coverage.reason}</span>}
          </p>
        ) : (
          <>
            <table className="rp-table rp-metrics">
              <tbody>
                <tr>
                  <th>Seabed imaged</th>
                  <td>{km2(cov.metrics.imaged_km2)} ({m2(cov.metrics.imaged_km2)})</td>
                  <th>Track length</th>
                  <td>{km(cov.metrics.track_length_km)}</td>
                </tr>
                <tr>
                  <th>Survey hull</th>
                  <td>
                    {km2(cov.metrics.hull_km2)}, {pct(cov.metrics.imaged_pct_of_hull)} imaged
                  </td>
                  <th>Swath footprint</th>
                  <td>
                    {km2(cov.metrics.swath_footprint_km2)}, {pct(cov.metrics.imaged_pct_of_footprint)} imaged
                  </td>
                </tr>
                <tr>
                  <th>Nadir blind strip</th>
                  <td>
                    {m2(cov.metrics.nadir_blind_km2)}, mean width {cov.metrics.mean_nadir_strip_width_m ?? "—"} m
                  </td>
                  <th>Degraded-row gaps</th>
                  <td>
                    {m2(cov.metrics.degraded_gap_km2)}
                    {Object.entries(cov.metrics.degraded_gap_by_reason_km2 || {}).map(([reason, value]) => (
                      <span key={reason} className="rp-sub">
                        {titleCase(reason)}: {m2(value)}
                      </span>
                    ))}
                  </td>
                </tr>
                <tr>
                  <th>Mean swath width</th>
                  <td>{cov.metrics.mean_swath_width_m ?? "—"} m</td>
                  <th>Altitude</th>
                  <td>
                    {cov.metrics.altitude_m
                      ? `${cov.metrics.altitude_m.min}–${cov.metrics.altitude_m.max} m (median ${cov.metrics.altitude_m.median} m)`
                      : "not recorded"}
                  </td>
                </tr>
              </tbody>
            </table>
            <h3 className="rp-h3">Re-look lines</h3>
            <RelookTable lines={lines} />
            <p className="rp-small">
              Gap lines keep the original heading and put the gap at mid-range, clear of the nadir
              strip. Contact lines run perpendicular to the original track so the contact's
              shadow is seen from a second aspect, passing it abeam at mid-range. {cov.method.relook}
            </p>
          </>
        )}
      </Section>

      {ghosttrace && (
        <Section title="GhostTrace summary" id="rp-ghosttrace">
          <GhostTraceSummary doc={ghosttrace} />
        </Section>
      )}

      {/* PROVENANCE */}
      <Section title="Provenance, method and disclaimers" id="rp-provenance" breakBefore={Boolean(ghosttrace)}>
        <dl className="rp-prov">
          <div>
            <dt>Navigation</dt>
            <dd>
              {navigation.source || navigation.note || "none"}
              {navigation.assumptions && <span className="rp-sub">{navigation.assumptions}</span>}
            </dd>
          </div>
          <div>
            <dt>Coordinates</dt>
            <dd>{provenance.coordinate_note || "—"}</dd>
          </div>
          <div>
            <dt>Severity</dt>
            <dd>
              {provenance.severity_formula || configuration.severity_policy?.formula || "—"}
              {provenance.ranking_rule && <span className="rp-sub">{provenance.ranking_rule}</span>}
            </dd>
          </div>
          {provenance.suppression_rule && (
            <div>
              <dt>Verification</dt>
              <dd>
                {String(provenance.suppression_rule)}
                {provenance.verification?.note && <span className="rp-sub">{provenance.verification.note}</span>}
              </dd>
            </div>
          )}
          {cov && (
            <div>
              <dt>Coverage</dt>
              <dd>
                {cov.method.footprint}. {cov.method.imaged}.
                {cov.method.strips.map((strip) => (
                  <span key={strip.strip} className="rp-sub">
                    {strip.strip}: {strip.half_width}. {strip.nadir}.
                  </span>
                ))}
              </dd>
            </div>
          )}
          {cov?.references?.length > 0 && (
            <div>
              <dt>References</dt>
              <dd>
                {cov.references.map((ref) => (
                  <span key={ref.url} className="rp-sub">
                    {ref.title}. {ref.url}. Used for {ref.used_for}.
                  </span>
                ))}
              </dd>
            </div>
          )}
        </dl>

        <div className="rp-disclaimers">
          {demo && <p><strong>Synthetic data.</strong> {metadata.demo_warning || copy.demoNote}</p>}
          <p>
            <strong>Severity and actions.</strong> {configuration.disclaimer || copy.disclaimer}
          </p>
          {cov && (
            <p>
              <strong>Coverage.</strong> {cov.limitations.join(" ")} Re-look lines are planning
              geometry, not a navigation procedure: check depth, traffic and turning circle
              before running one.
            </p>
          )}
          <p>
            <strong>Detections.</strong> A detection is a model prediction, not a confirmed
            object. Nothing in this report is evidence of an object until it is identified.
          </p>
        </div>
        <p className="rp-footer">
          Generated {timestamp(generatedAt)} by the DeepEcho dashboard from export.json
          {cov ? ", coverage.json" : ""}
          {ghosttrace ? " and ghosttrace.json" : ""} for survey {metadata.survey_id}.
        </p>
      </Section>
    </article>
  )
}

/** /report/:surveyId, a printable A4 report of one processed survey. */
function SurveyReport() {
  const { surveyId } = useParams()
  const { data, error, loading } = useSurveyDocument(loadReport, surveyId)

  return (
    <div className="rp-page">
      <div className="rp-toolbar">
        <Link className="sv-btn" to={`/map?survey=${encodeURIComponent(surveyId || "")}`}>
          <ArrowLeft size={15} />
          Back to the hazard map
        </Link>
        <span className="rp-toolbar-hint">Print, then choose “Save as PDF”. A4, portrait.</span>
        <button type="button" className="sv-btn sv-btn-primary" onClick={() => window.print()} disabled={!data}>
          <Printer size={15} />
          Print / Save PDF
        </button>
      </div>
      {loading && <p className="sv-empty">Preparing the report…</p>}
      {error && <p className="sv-error">{error.message}</p>}
      {data && <Report data={data} />}
    </div>
  )
}

export default SurveyReport
