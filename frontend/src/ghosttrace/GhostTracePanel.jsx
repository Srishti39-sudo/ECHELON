import { useEffect, useMemo, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Loader2, Play, TerminalSquare } from "lucide-react"

import { Empty, Failed, Loading } from "../components/PageState"
import {
  alertTextUrl,
  fetchCapabilities,
  fetchGhostTrace,
  fetchLayer,
  geojsonUrl,
  runGhostTrace,
} from "./api"
import { copy, ghostConfig, layerKinds } from "./config"
import {
  canonicalKind,
  extentOf,
  featureKey,
  isApproximate,
  isNum,
  isUnavailable,
  padExtent,
  snapshotsOf,
  sortByRank,
} from "./format"
import { ghosttraceContext, ghosttraceQuestion } from "./handoff"
import { usePlayback, useReducedMotion } from "./hooks"
import ChangePanel from "./components/ChangePanel"
import DriftPlayback from "./components/DriftPlayback"
import GhostTraceMap from "./components/GhostTraceMap"
import MapLegend from "./components/MapLegend"
import RecoveryPlan from "./components/RecoveryPlan"
import RescueQueue from "./components/RescueQueue"
import SummaryHeader from "./components/SummaryHeader"
import TargetDetail from "./components/TargetDetail"
import "./ghosttrace.css"

const DEFAULT_VISIBLE = {
  ...Object.fromEntries(layerKinds.map((l) => [l.kind, true])),
  drift: true,
  particles: true,
  route: true,
  removed: true,
  basemap: true,
}

function layerEntry(data) {
  if (!data) return { status: "absent" }
  return {
    status: "ready",
    data,
    approximate: (data.features || []).some((f) => isApproximate(f?.properties?.geometry_quality)),
  }
}

/* ------------------------------------------------------------------ view */

function GhostTraceView({ doc, surveyId, offline, layers, canRun, running, onRun, title, isExample }) {
  const reducedMotion = useReducedMotion()
  const navigate = useNavigate()
  const detailHeading = useRef(null)
  const detailSection = useRef(null)

  const [selectedId, setSelectedId] = useState(null)
  const [tab, setTab] = useState("activity")
  const [visible, setVisible] = useState(DEFAULT_VISIBLE)
  const [showSuppressed, setShowSuppressed] = useState(false)

  const allTargets = useMemo(() => sortByRank(doc.targets || []), [doc])
  const suppressedCount = allTargets.filter((t) => t.suppressed).length
  const targets = useMemo(
    () => (showSuppressed ? allTargets : allTargets.filter((t) => !t.suppressed)),
    [allTargets, showSuppressed],
  )
  const targetsById = useMemo(() => new Map(allTargets.map((t) => [String(t.detection_id), t])), [allTargets])

  // Until the operator picks one, the top of the queue is selected.
  const effectiveId = targetsById.has(selectedId) ? selectedId : targets[0]?.detection_id ?? null
  const selected = effectiveId ? targetsById.get(effectiveId) : null

  // "forecast" is the drift for the mode the net was found in; "if_refloated"
  // is the labelled what-if the engine adds for a seabed net. Everything drawn
  // from drift -- particles, cones, reached habitat -- follows the choice.
  const [scenario, setScenario] = useState("forecast")
  const refloated = selected?.drift_scenarios?.if_refloated
  const activeScenario = scenario === "if_refloated" && refloated && !isUnavailable(refloated)
    ? "if_refloated" : "forecast"
  const activeDrift = activeScenario === "if_refloated" ? refloated : selected?.drift

  const snapshots = useMemo(() => (selected ? snapshotsOf(activeDrift, selected) : []), [selected, activeDrift])
  // Autoplay once on arrival, never when reduced motion is requested.
  const playback = usePlayback(snapshots.length, ghostConfig.playbackStepMs, !reducedMotion)
  const snapshot = snapshots[playback.index] || null
  const t = snapshot?.t_hours ?? 0

  const trail = useMemo(
    () =>
      snapshots
        .slice(0, playback.index + 1)
        .filter((s) => s.points.length)
        .map((s) => [
          s.points.reduce((a, p) => a + p[0], 0) / s.points.length,
          s.points.reduce((a, p) => a + p[1], 0) / s.points.length,
        ]),
    [snapshots, playback.index],
  )

  const highlights = useMemo(() => {
    const map = new Map()
    const drift = activeDrift
    if (!drift || drift.available === false) return map
    for (const impact of drift.impacts || []) {
      if (!isNum(impact.probability) || impact.probability <= 0) continue
      const first = impact.first_arrival_hours
      map.set(featureKey(canonicalKind(impact.kind), impact.name), {
        probability: impact.probability,
        first,
        state: isNum(first) && t >= first ? "reached" : "forecast",
      })
    }
    return map
  }, [activeDrift, t])

  const extent = useMemo(() => extentOf(doc, { includeRouteStart: false }) || extentOf(doc), [doc])

  const select = (id) => {
    if (id === effectiveId) return
    setSelectedId(id)
    playback.reset(false)
  }

  const openDetail = (id) => {
    select(id)
    // After React commits the new target, bring the detail into view and put
    // focus on its heading so keyboard and screen-reader users land there too.
    window.requestAnimationFrame(() => {
      detailSection.current?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" })
      detailHeading.current?.focus({ preventScroll: true })
    })
  }

  const toggle = (key, value) => setVisible((prev) => ({ ...prev, [key]: value }))

  // Hand one target to the grounded assistant. The survey's display title
  // travels with the document so the assistant can name the survey.
  const askAssistant = (target) => {
    const context = ghosttraceContext(target, { ...doc, title: doc.title ?? title ?? null })
    if (!context) return
    navigate("/assistant", { state: { ghosttraceContext: context, question: ghosttraceQuestion(context) } })
  }

  if (!allTargets.length) {
    return (
      <div className="gt-root">
        <SummaryHeader doc={doc} title={title} canRun={canRun} running={running} onRerun={onRun} isExample={isExample} geojsonHref={offline ? null : geojsonUrl(surveyId)} />
        <Empty title="No targets">{copy.noTargets}</Empty>
      </div>
    )
  }

  const located = allTargets.some((target) => isNum(target.latitude) && isNum(target.longitude))

  return (
    <div className="gt-root">
      <SummaryHeader
        doc={doc}
        title={title}
        canRun={canRun}
        running={running}
        onRerun={onRun}
        isExample={isExample}
        geojsonHref={offline ? null : geojsonUrl(surveyId)}
      />

      <div className="gt-grid">
        <div className="gt-map-col">
          <section className="gt-map-card" aria-label="GhostTrace map">
            {located ? (
              <GhostTraceMap
                doc={doc}
                targets={targets}
                layers={layers}
                visible={visible}
                selectedId={effectiveId}
                onSelect={select}
                snapshot={snapshot}
                trail={trail}
                highlights={highlights}
                extent={extent}
                reducedMotion={reducedMotion}
              />
            ) : (
              <p className="gt-callout gt-callout--muted gt-no-map">
                No target in this survey has a geographic position (the survey is not georeferenced), so there is no map,
                drift or habitat context to draw. The queue and details below still apply.
              </p>
            )}
            {located && <MapLegend layers={layers} visible={visible} onToggle={toggle} />}
          </section>
          <DriftPlayback
            target={selected}
            drift={activeDrift}
            scenario={activeScenario}
            hasRefloatScenario={Boolean(refloated) && !isUnavailable(refloated)}
            onScenario={(next) => {
              setScenario(next)
              playback.reset(false)
            }}
            snapshots={snapshots}
            playback={playback}
            reducedMotion={reducedMotion}
          />
        </div>

        <RescueQueue
          targets={targets}
          selectedId={effectiveId}
          onSelect={select}
          onOpenDetail={openDetail}
          onAsk={askAssistant}
          showSuppressed={showSuppressed}
          onToggleSuppressed={setShowSuppressed}
          suppressedCount={suppressedCount}
        />
      </div>

      <div className="gt-grid gt-grid--lower">
        <div ref={detailSection} className="gt-detail-anchor">
          <TargetDetail
            target={selected}
            tab={tab}
            onTab={setTab}
            surveyId={doc.survey_id || surveyId}
            dataSources={doc.data_sources}
            alertDownloadUrl={offline || !selected ? null : alertTextUrl(surveyId, selected.detection_id)}
            headingRef={detailHeading}
            onAsk={askAssistant}
          />
        </div>
        <div className="gt-side">
          <RecoveryPlan plan={doc.recovery_plan} targetsById={targetsById} selectedId={effectiveId} onSelect={select} />
          <ChangePanel summary={doc.change_summary} removed={doc.removed_since_previous} />
        </div>
      </div>

      <p className="gt-disclaimer">
        GhostTrace output is decision support from automated detections and configurable heuristics. It is not an official
        assessment, forecast or notice. Verify on site before any action.
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------- loader */

/**
 * Load one survey's ghosttrace.json and its habitat layers, or offer to run it.
 */
function GhostTraceLoader({ surveyId, title }) {
  const [state, setState] = useState({ status: "loading" })
  const [nonce, setNonce] = useState(0)
  const [caps, setCaps] = useState(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState(null)
  const [layers, setLayers] = useState({})

  useEffect(() => {
    let alive = true
    fetchGhostTrace(surveyId)
      .then((doc) => alive && setState({ status: "ready", doc }))
      .catch((error) => alive && setState({ status: error.status === 404 ? "missing" : "error", error }))
    return () => {
      alive = false
    }
  }, [surveyId, nonce])

  useEffect(() => {
    let alive = true
    fetchCapabilities()
      .then((body) => alive && setCaps(body))
      .catch(() => alive && setCaps({ run_enabled: false, unreachable: true }))
    return () => {
      alive = false
    }
  }, [])

  const doc = state.status === "ready" ? state.doc : null
  const bbox = useMemo(() => padExtent(doc ? extentOf(doc) : null, ghostConfig.layerPaddingDeg), [doc])
  const bboxKey = bbox ? bbox.map((n) => n.toFixed(3)).join(",") : null

  useEffect(() => {
    if (!bboxKey) return undefined
    let alive = true
    const box = bboxKey.split(",").map(Number)
    for (const { kind } of layerKinds) {
      fetchLayer(kind, box)
        .then((data) => alive && setLayers((prev) => ({ ...prev, [kind]: layerEntry(data) })))
        .catch((error) => alive && setLayers((prev) => ({ ...prev, [kind]: { status: "error", message: error.message } })))
    }
    return () => {
      alive = false
    }
  }, [bboxKey])

  const run = async () => {
    setRunning(true)
    setRunError(null)
    try {
      await runGhostTrace(surveyId)
      setNonce((n) => n + 1)
    } catch (error) {
      setRunError(error)
    } finally {
      setRunning(false)
    }
  }

  const canRun = Boolean(caps?.run_enabled)

  if (state.status === "loading") return <Loading label="Loading GhostTrace" />
  if (state.status === "error") return <Failed error={state.error} onRetry={() => setNonce((n) => n + 1)} />

  if (state.status === "missing") {
    return (
      <div className="gt-root">
        <section className="gt-missing" aria-labelledby="gt-missing-title">
          <p className="gt-eyebrow">{copy.title}</p>
          <h2 id="gt-missing-title">{copy.notGenerated}</h2>
          <p className="gt-muted">{state.error?.message}</p>
          {running ? (
            <p className="gt-running" role="status">
              <Loader2 size={16} className="gt-spin" aria-hidden="true" /> {copy.running}
            </p>
          ) : canRun ? (
            <button type="button" className="gt-btn gt-btn--primary" onClick={run}>
              <Play size={16} aria-hidden="true" /> {copy.runButton}
            </button>
          ) : (
            <div className="gt-command">
              <p>
                <TerminalSquare size={15} aria-hidden="true" /> {copy.runDisabled}
              </p>
              <code>.venv/bin/python run_ghosttrace.py --survey data/surveys/{surveyId}</code>
            </div>
          )}
          {runError && <p className="gt-error" role="alert">{runError.message}</p>}
        </section>
      </div>
    )
  }

  return (
    <>
      {runError && <p className="gt-error" role="alert">Re-run failed: {runError.message}</p>}
      <GhostTraceView
        doc={doc}
        surveyId={surveyId}
        layers={layers}
        canRun={canRun}
        running={running}
        onRun={run}
        title={title || doc.title || doc.survey_id || surveyId}
      />
    </>
  )
}

/* ----------------------------------------------------------------- panel */

/**
 * GhostTrace for one survey: the rescue decision for every ghost net.
 *
 * Mount as a route body or embed as a tab. With `data` (and optionally
 * `layers`, a {kind: FeatureCollection} map) it renders that document with no
 * backend at all, which is how the fixture preview works.
 *
 *   <GhostTracePanel surveyId="demo-synthetic" />
 *   <GhostTracePanel surveyId="example" data={doc} layers={layers} />
 */
function GhostTracePanel({ surveyId, data, layers, title }) {
  const fixtureLayers = useMemo(
    () => Object.fromEntries(layerKinds.map(({ kind }) => [kind, layerEntry(layers?.[kind] || null)])),
    [layers],
  )

  if (data) {
    return (
      <GhostTraceView
        key={`${data.survey_id}|${data.generated_at}`}
        doc={data}
        surveyId={surveyId || data.survey_id}
        offline
        layers={fixtureLayers}
        canRun={false}
        running={false}
        title={title || data.title || data.survey_id}
        isExample={Boolean(data.synthetic_example)}
      />
    )
  }

  if (!surveyId) {
    return <Empty title="No survey selected">Choose a processed survey to see its GhostTrace.</Empty>
  }

  // Keyed by survey so selection, playback and layers never leak between surveys.
  return <GhostTraceLoader key={surveyId} surveyId={surveyId} title={title} />
}

export default GhostTracePanel
