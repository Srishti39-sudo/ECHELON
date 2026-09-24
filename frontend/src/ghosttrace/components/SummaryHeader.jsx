import { AlertOctagon, Download, FlaskConical, Fish, Leaf, Loader2, RefreshCw, Ship, Waypoints } from "lucide-react"

import { copy } from "../config"
import { dateTime, isNum } from "../format"

function Stat({ icon: Icon, value, label, note, tone }) {
  return (
    <div className={`gt-stat${tone ? ` gt-stat--${tone}` : ""}`}>
      <span className="gt-stat-icon" aria-hidden="true">
        <Icon size={18} />
      </span>
      <span className="gt-stat-value">{isNum(value) ? value : "—"}</span>
      <span className="gt-stat-label">{label}</span>
      {note && <span className="gt-stat-note">{note}</span>}
    </div>
  )
}

/**
 * The top of the page: which survey, the four numbers that matter, what is
 * synthetic, and how it was computed. Every number is from `summary`.
 */
function SummaryHeader({ doc, title, onRerun, running, canRun, geojsonHref, isExample }) {
  const s = doc.summary || {}
  const synthetic = Boolean(doc.synthetic_inputs)
  const demo = Boolean(doc.demo)

  const headline = []
  if (isNum(s.urgent)) headline.push(`${s.urgent} ${s.urgent === 1 ? "target needs" : "targets need"} action now`)
  if (isNum(s.actively_fishing) && s.actively_fishing > 0)
    headline.push(`${s.actively_fishing} ${s.actively_fishing === 1 ? "shows" : "show"} high fish activity at the net`)
  if (isNum(s.near_sensitive_habitat) && s.near_sensitive_habitat > 0)
    headline.push(`${s.near_sensitive_habitat} near sensitive habitat`)

  return (
    <header className="gt-header">
      <div className="gt-header-top">
        <div className="gt-title-block">
          <p className="gt-eyebrow">
            <Waypoints size={14} aria-hidden="true" /> {copy.title} · rescue decisions
          </p>
          <h1>{title}</h1>
          <p className="gt-subtitle">{copy.subtitle}</p>
          <p className="gt-meta">
            Survey <code>{doc.survey_id}</code> · generated {dateTime(doc.generated_at)}
            {s.targets != null ? ` · ${s.targets} targets assessed` : ""}
            {isNum(s.suppressed_excluded) && s.suppressed_excluded > 0 ? ` · ${s.suppressed_excluded} suppressed detection${s.suppressed_excluded === 1 ? "" : "s"} excluded` : ""}
          </p>
        </div>

        <div className="gt-header-actions">
          <div className="gt-badges">
            {isExample && (
              <span className="gt-badge gt-badge--example">
                <FlaskConical size={13} aria-hidden="true" /> {copy.exampleBadge}
              </span>
            )}
            {synthetic && (
              <span className="gt-badge gt-badge--synthetic">
                <FlaskConical size={13} aria-hidden="true" /> {copy.syntheticBadge}
              </span>
            )}
            {demo && <span className="gt-badge gt-badge--demo">{copy.demoBadge}</span>}
          </div>
          <div className="gt-header-buttons">
            {geojsonHref && (
              <a className="gt-btn" href={geojsonHref} download>
                <Download size={15} aria-hidden="true" /> GeoJSON
              </a>
            )}
            {canRun && (
              <button type="button" className="gt-btn" onClick={onRerun} disabled={running}>
                {running ? <Loader2 size={15} className="gt-spin" aria-hidden="true" /> : <RefreshCw size={15} aria-hidden="true" />}
                {running ? "Running…" : copy.rerunButton}
              </button>
            )}
          </div>
        </div>
      </div>

      {isNum(s.stage_failures) && s.stage_failures > 0 && (
        <p className="gt-callout gt-callout--warn" role="note">
          {s.stage_failures} analysis stage{s.stage_failures === 1 ? "" : "s"} failed on individual targets. Those sections show the error instead of a result.
        </p>
      )}

      {headline.length > 0 && <p className="gt-headline">{headline.join(" · ")}.</p>}

      <div className="gt-stats">
        <Stat icon={AlertOctagon} value={s.urgent} label="Urgent targets" note={isNum(s.high) ? `+${s.high} high priority` : null} tone={s.urgent ? "urgent" : null} />
        <Stat icon={Fish} value={s.actively_fishing} label="Actively fishing" note="water-column echoes at the net" tone={s.actively_fishing ? "fishing" : null} />
        <Stat icon={Leaf} value={s.near_sensitive_habitat} label="Near sensitive habitat" note="reefs, MPAs, turtles, dugongs" tone={s.near_sensitive_habitat ? "habitat" : null} />
        <Stat icon={Ship} value={s.propeller_hazards} label="Propeller hazards" note="to fishing boats" tone={s.propeller_hazards ? "hazard" : null} />
      </div>

    </header>
  )
}

export default SummaryHeader
