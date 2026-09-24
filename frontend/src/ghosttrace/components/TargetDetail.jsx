import { useRef } from "react"
import { Crosshair, MessageSquare } from "lucide-react"

import { styleForTier } from "../config"
import { confidence, isUnavailable, latLngOf, num, size, titleCase } from "../format"
import AlertDraft from "./AlertDraft"
import { TierChip } from "./Chips"
import {
  ActivitySection,
  ChangeSection,
  DriftSection,
  HabitatSection,
  PeopleSection,
} from "./TargetSections"

const DETAIL_TABS = [
  { id: "activity", label: "Activity", section: "activity" },
  { id: "habitat", label: "Habitat", section: "habitat" },
  { id: "drift", label: "Drift", section: "drift" },
  { id: "people", label: "People", section: "people" },
  { id: "change", label: "Change", section: "change" },
  { id: "alert", label: "Alert", section: "alert" },
]

const isNetClass = (name) =>
  String(name || "").toLowerCase().split(/[^a-z]+/).some((token) => ["net", "nets", "gear"].includes(token))

function tabUnavailable(target, tab) {
  const section = target?.[tab.section]
  if (tab.id === "alert") return !section?.draft_text
  if (tab.id === "activity") return !section || section.available === false
  if (tab.id === "change") return !section
  return isUnavailable(section)
}

/**
 * Everything about one target, one tab per question GhostTrace answers.
 * Tabs follow the WAI-ARIA tabs pattern: arrow keys move, Home/End jump.
 */
function TargetDetail({ target, tab, onTab, surveyId, dataSources, alertDownloadUrl, headingRef, onAsk }) {
  const tabRefs = useRef({})

  if (!target) {
    return (
      <section className="gt-detail gt-detail--empty" aria-label="Target detail">
        <p className="gt-muted">Select a target in the rescue queue or on the map.</p>
      </section>
    )
  }

  const priority = target.priority || {}
  const tier = styleForTier(priority.tier)
  const ll = latLngOf(target)

  const onKeyDown = (event) => {
    const index = DETAIL_TABS.findIndex((t) => t.id === tab)
    const map = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: DETAIL_TABS.length - 1 }
    if (!(event.key in map)) return
    event.preventDefault()
    const next = DETAIL_TABS[(map[event.key] + DETAIL_TABS.length) % DETAIL_TABS.length]
    onTab(next.id)
    tabRefs.current[next.id]?.focus()
  }

  let body
  switch (tab) {
    case "habitat":
      body = <HabitatSection habitat={target.habitat} dataSources={dataSources} />
      break
    case "drift":
      body = <DriftSection drift={target.drift} />
      break
    case "people":
      body = <PeopleSection people={target.people} target={target} />
      break
    case "change":
      body = <ChangeSection change={target.change} />
      break
    case "alert":
      body = <AlertDraft key={target.detection_id} surveyId={surveyId} target={target} downloadUrl={alertDownloadUrl} />
      break
    default:
      body = <ActivitySection activity={target.activity} />
  }

  return (
    <section className="gt-detail" aria-labelledby="gt-detail-title" style={{ "--gt-tier": tier.color }}>
      <header className="gt-detail-head">
        <div>
          <h2 id="gt-detail-title" ref={headingRef} tabIndex={-1}>
            <span className="gt-detail-rank">#{priority.rank ?? "–"}</span>
            {titleCase(target.object_class)}
            <span className="gt-detail-id">{target.detection_id}</span>
          </h2>
          <p className="gt-muted">
            <Crosshair size={13} aria-hidden="true" />{" "}
            {ll ? `${ll[0].toFixed(5)}, ${ll[1].toFixed(5)}` : "no position"} · {confidence(target.confidence_pct)} confidence ·{" "}
            {size(target.dimensions)}
            {target.dimensions && Number.isFinite(target.dimensions.height_m) ? ` · ${num(target.dimensions.height_m, 1)} m high` : ""}
          </p>
        </div>
        <div className="gt-detail-score">
          <TierChip tier={priority.tier} />
          <span>
            score <strong>{num(priority.score, 2)}</strong>
          </span>
          {onAsk && (
            <button type="button" className="gt-btn gt-btn--small" onClick={() => onAsk(target)}>
              <MessageSquare size={15} aria-hidden="true" />
              Ask the assistant about this {isNetClass(target.object_class) ? "net" : "object"}
            </button>
          )}
        </div>
      </header>
      {target.suppressed && (
        <p className="gt-callout gt-callout--muted">
          The engine suppressed this contact. It is listed for transparency; check the recovery plan for whether it is included.
        </p>
      )}

      {target.errors?.length > 0 && (
        <div className="gt-callout gt-callout--warn" role="note">
          <strong>Stage errors on this target:</strong> {target.errors.join("; ")}
        </div>
      )}

      <div className="gt-tabs" role="tablist" aria-label="Target detail sections" onKeyDown={onKeyDown}>
        {DETAIL_TABS.map((t) => {
          const unavailable = tabUnavailable(target, t)
          return (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[t.id] = el
              }}
              type="button"
              role="tab"
              id={`gt-tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={tab === t.id ? `gt-tabpanel-${t.id}` : undefined}
              tabIndex={tab === t.id ? 0 : -1}
              className={`gt-tab${tab === t.id ? " is-active" : ""}${unavailable ? " is-unavailable" : ""}`}
              onClick={() => onTab(t.id)}
            >
              {t.label}
              {unavailable && <span className="gt-tab-flag">n/a</span>}
            </button>
          )
        })}
      </div>
      <div role="tabpanel" id={`gt-tabpanel-${tab}`} aria-labelledby={`gt-tab-${tab}`} className="gt-tabpanel">
        {body}
      </div>
    </section>
  )
}

export default TargetDetail
