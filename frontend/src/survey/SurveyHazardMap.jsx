import { useEffect, useMemo, useState } from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"
import { Download, FileText } from "lucide-react"

import StatCard from "../components/StatCard"
import { actionsUrl, fetchExport, listSurveys, mapUrl } from "./api"
import { copy } from "./config"
import { filteredDetections, isVerified } from "./detections"
import { handoffQuestion, hotspotContext } from "./handoff"
import DetectionTable from "./components/DetectionTable"
import Filters from "./components/Filters"
import HotspotDetail from "./components/HotspotDetail"
import HotspotTable from "./components/HotspotTable"
import MapFrame from "./components/MapFrame"
import CoveragePanel from "./components/CoveragePanel"
import MissionReplay from "./components/MissionReplay"
import "./survey.css"

const EMPTY_FILTERS = { objectClass: "all", tier: "all", priority: "all" }

/**
 * The Survey Hazard Map page.
 *
 * Everything on screen is read out of one export.json. No number is recomputed
 * here and no threshold is duplicated: the engine decided the severities, the
 * ranking and the actions, recorded why, and this page renders that decision.
 * When the two could disagree, the export wins, because it is the artefact that
 * gets archived and audited.
 */
function SurveyHazardMap() {
  const navigate = useNavigate()
  // A page that just finished a survey (Live survey) passes its id, and the
  // operator expects to land on that survey, not on whichever sorts first.
  // ?survey=<id> does the same for a link that has no router state to carry,
  // such as the printable report's "back to the map" or a bookmarked URL.
  const location = useLocation()
  const requestedId =
    location.state?.surveyId ?? new URLSearchParams(location.search).get("survey") ?? null

  const [surveys, setSurveys] = useState([])
  const [surveyId, setSurveyId] = useState(null)
  const [survey, setSurvey] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [filters, setFilters] = useState(EMPTY_FILTERS)
  const [showFiltered, setShowFiltered] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    listSurveys()
      .then((found) => {
        if (cancelled) return
        setSurveys(found)
        // Lead with a real survey where there is one. A demo lands first in
        // the list alphabetically and opening on synthetic data by default
        // would be the wrong first impression.
        const preferred =
          found.find((s) => s.survey_id === requestedId) ||
          found.find((s) => !s.demo) ||
          found[0]
        setSurveyId(preferred ? preferred.survey_id : null)
        setLoading(Boolean(preferred))
      })
      .catch((cause) => {
        if (!cancelled) {
          setError(cause.message)
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [requestedId])

  // Choosing a survey clears what belonged to the previous one. This lives in
  // the handler rather than in the effect below: resetting state synchronously
  // inside an effect body triggers a second render pass for no reason, and the
  // reset is caused by the choice, not by the fetch.
  const selectSurvey = (nextId) => {
    setSurveyId(nextId)
    setSurvey(null)
    setSelectedId(null)
    setFilters(EMPTY_FILTERS)
    setShowFiltered(false)
    setError(null)
    setLoading(true)
  }

  useEffect(() => {
    if (!surveyId) return undefined
    let cancelled = false

    fetchExport(surveyId)
      .then((loaded) => {
        if (cancelled) return
        setSurvey(loaded)
        setSelectedId(loaded.hotspots[0]?.hotspot_id ?? null)
        setLoading(false)
      })
      .catch((cause) => {
        if (cancelled) return
        setError(cause.message)
        setSurvey(null)
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [surveyId])

  const summary = survey?.survey_summary
  // Memoised so the empty-array fallbacks keep a stable identity between
  // renders; without it every derived useMemo below recomputes on every render.
  const hotspots = useMemo(() => survey?.hotspots ?? [], [survey])
  const detections = useMemo(() => survey?.detections ?? [], [survey])

  const detectionsById = useMemo(() => {
    const index = new Map()
    detections.forEach((detection) => index.set(detection.id, detection))
    return index
  }, [detections])

  // Filter options come from the loaded export, never from a fixed list, so a
  // survey full of classes this project has never seen still filters correctly.
  const classes = useMemo(
    () => [...new Set(detections.map((d) => d.object_class))].sort(),
    [detections],
  )
  const tiers = useMemo(
    () => [...new Set(hotspots.map((h) => h.severity_tier))],
    [hotspots],
  )

  const visible = useMemo(() => {
    let rows = hotspots

    if (filters.objectClass !== "all") {
      // A hotspot matches a class when any detection in it has that class, not
      // only when it is the dominant one. Filtering on the dominant class alone
      // would hide a mine sitting in a cell full of debris, which is exactly
      // the contact somebody filtering for mines is looking for.
      rows = rows.filter((hotspot) =>
        hotspot.detection_ids.some(
          (id) => detectionsById.get(id)?.object_class === filters.objectClass,
        ),
      )
    }
    if (filters.tier !== "all") {
      rows = rows.filter((hotspot) => hotspot.severity_tier === filters.tier)
    }
    if (filters.priority !== "all") {
      rows = rows.slice(0, Number(filters.priority))
    }
    return rows
  }, [hotspots, filters, detectionsById])

  const selected = hotspots.find((h) => h.hotspot_id === selectedId) ?? null
  const selectedDetections = selected
    ? selected.detection_ids.map((id) => detectionsById.get(id)).filter(Boolean)
    : []

  // Verification fields exist only in exports written since it was added. An
  // older export shows no filtered count at all rather than a zero it never
  // measured.
  const verified = isVerified(survey)
  const filtered = useMemo(() => filteredDetections(survey), [survey])
  const filteredCount = summary?.suppressed_detections ?? filtered.length

  const surveyMeta = surveys.find((s) => s.survey_id === surveyId)
  const isDemo = Boolean(survey?.metadata?.demo)
  const georeferenced = Boolean(summary?.georeferenced)

  /**
   * Hand a hotspot to the assistant and go there.
   *
   * The context object crosses as router state. This function builds it and
   * navigates; it does not retrieve, prompt, or explain anything, because that
   * is the assistant's half of the boundary and not this feature's.
   */
  const askAssistant = (hotspot) => {
    const context = hotspotContext(hotspot, survey)
    navigate("/assistant", {
      state: { surveyContext: context, question: handoffQuestion(context) },
    })
  }

  if (error) {
    return (
      <div className="sv-page">
        <header className="sv-header">
          <h1>{copy.title}</h1>
        </header>
        <p className="sv-error">{error}</p>
      </div>
    )
  }

  return (
    <div className="sv-page">
      <header className="sv-header">
        <div>
          <h1>{copy.title}</h1>
          <p>{copy.subtitle}</p>
        </div>

        <div className="sv-header-controls">
          <label className="sv-filter">
            <span>Survey</span>
            <select
              value={surveyId ?? ""}
              onChange={(event) => selectSurvey(event.target.value)}
              disabled={!surveys.length}
            >
              {surveys.map((item) => (
                <option key={item.survey_id} value={item.survey_id}>
                  {item.title}
                  {item.demo ? " (demo)" : ""}
                </option>
              ))}
            </select>
          </label>

          {surveyId && (
            <a className="sv-btn" href={actionsUrl(surveyId)} download>
              <Download size={15} />
              Action list
            </a>
          )}

          {surveyId && survey && (
            <Link className="sv-btn sv-btn-primary" to={`/report/${encodeURIComponent(surveyId)}`}>
              <FileText size={15} />
              {copy.reportButton}
            </Link>
          )}
        </div>
      </header>

      {!surveys.length && !loading && <p className="sv-empty">{copy.noSurveys}</p>}

      {isDemo && (
        <p className="sv-demo">
          <strong>{copy.demoBanner}</strong>
          {copy.demoNote}
        </p>
      )}

      {summary && (
        <>
          <div className="sv-stats">
            <StatCard title="Total Detections" value={summary.total_deduplicated_detections} />
            <StatCard title="Hotspots" value={summary.total_hotspots} />
            <StatCard title="Total Severity" value={summary.total_severity.toFixed(2)} />
            <StatCard
              title="Critical"
              value={summary.detections_by_tier?.critical ?? 0}
              type={summary.detections_by_tier?.critical ? "anomaly" : undefined}
            />
            <StatCard
              title="Filtered False Positives"
              value={verified ? filteredCount : "—"}
              subtitle={
                verified ? "kept in the export, not in any hotspot" : copy.notVerifiedNote
              }
            />
            <StatCard
              title="Coordinates"
              type="sv-stat-text"
              value={georeferenced ? copy.geoMode : copy.relativeMode}
              subtitle={summary.coordinate_mode}
            />
            <StatCard
              title="Survey Strips"
              value={summary.strips_processed}
              subtitle={`${summary.tiles_processed} tiles processed`}
            />
          </div>

          {/* Stated on the page, not only in the export. An operator reading a
              pixel offset needs to know it is not a position. */}
          <p className="sv-note">{georeferenced ? copy.geoNote : copy.relativeNote}</p>

          <section className="sv-map-section">
            <MapFrame
              src={mapUrl(surveyId)}
              focusId={selectedId}
              title={`${surveyMeta?.title ?? surveyId} hazard map`}
            />
          </section>

          <section className="sv-lower">
            <div className="sv-list">
              <h2>Priority Hotspots</h2>
              <Filters
                classes={classes}
                tiers={tiers}
                value={filters}
                onChange={setFilters}
                resultCount={visible.length}
                totalCount={hotspots.length}
              />
              <HotspotTable
                hotspots={visible}
                selectedId={selectedId}
                onSelect={setSelectedId}
                emptyMessage={hotspots.length ? copy.noMatch : copy.noHotspots}
              />
            </div>

            <aside className="sv-side">
              <h2>Selected Hotspot</h2>
              <HotspotDetail
                hotspot={selected}
                detections={selectedDetections}
                survey={{ mapHref: mapUrl(surveyId, selectedId) }}
                onAsk={askAssistant}
              />
            </aside>
          </section>

          {/* Replay and coverage read the survey's navigation sidecars through
              their own routes. Keyed by survey so nothing carries across. */}
          <MissionReplay key={`replay-${surveyId}`} surveyId={surveyId} />
          <CoveragePanel key={`coverage-${surveyId}`} surveyId={surveyId} />

          {/* Filtered detections belong to no hotspot, so the hotspot detail
              can never show them. They get their own section, closed until
              asked for, so a disagreement with the filter is one click away. */}
          {verified && (
            <section className="sv-filtered">
              <div className="sv-filtered-head">
                <div>
                  <h2>
                    {copy.filteredTitle} <span className="sv-count">{filteredCount}</span>
                  </h2>
                  <p>{filteredCount ? copy.filteredNote : copy.filteredNone}</p>
                </div>
                {filtered.length > 0 && (
                  <button
                    type="button"
                    className="sv-btn"
                    aria-expanded={showFiltered}
                    onClick={() => setShowFiltered((open) => !open)}
                  >
                    {showFiltered ? copy.filteredHide : copy.filteredShow(filtered.length)}
                  </button>
                )}
              </div>
              {showFiltered && <DetectionTable detections={filtered} />}
            </section>
          )}

          <p className="sv-disclaimer">{copy.disclaimer}</p>
        </>
      )}

      {loading && <p className="sv-empty">Loading survey…</p>}
    </div>
  )
}

export default SurveyHazardMap
