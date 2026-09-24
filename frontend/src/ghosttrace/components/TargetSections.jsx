import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  HelpCircle,
  XCircle,
} from "lucide-react"

import { copy, labelForKind, singularLabel, styleForActivity, styleForChange, styleForHazard } from "../config"
import {
  DASH,
  arrival,
  compass,
  dateTime,
  distance,
  kindOf,
  isNum,
  isUnavailable,
  num,
  pct,
  qualityLabel,
  reasonOf,
  sourceName,
  sourceUrl,
  titleCase,
} from "../format"
import { ActivityChip, ChangeChip, HazardChip, Tag, Unavailable } from "./Chips"

/* ---------------------------------------------------------------- shared */

function TextList({ items, empty = "None recorded." }) {
  const list = Array.isArray(items) ? items : items ? [items] : []
  if (!list.length) return <p className="gt-muted">{empty}</p>
  return (
    <ul className="gt-bullets">
      {list.map((item, i) => (
        <li key={i}>{typeof item === "string" ? item : JSON.stringify(item)}</li>
      ))}
    </ul>
  )
}

function Facts({ rows }) {
  return (
    <dl className="gt-facts">
      {rows.map(([label, value, hint]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>
            {value}
            {hint && <small>{hint}</small>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function Terms({ terms }) {
  const entries = Object.entries(terms || {})
  if (!entries.length) return null
  return (
    <details className="gt-terms">
      <summary>Engine terms ({entries.length})</summary>
      <dl className="gt-facts gt-facts--dense">
        {entries.map(([name, value]) => (
          <div key={name}>
            <dt>{titleCase(name)}</dt>
            <dd>{typeof value === "object" && value !== null ? JSON.stringify(value) : String(value)}</dd>
          </div>
        ))}
      </dl>
    </details>
  )
}

function SourceLink({ source, dataSources }) {
  const name = sourceName(source)
  const href = sourceUrl(source, dataSources)
  if (!name && !href) return <span className="gt-muted">{DASH}</span>
  let label = name
  if (!label && href) {
    try {
      label = new URL(href).hostname
    } catch {
      label = href
    }
  }
  const licence = typeof source === "object" ? source?.licence : null
  if (!href) return <span title={licence ? `Licence: ${licence}` : undefined}>{label}</span>
  return (
    <a href={href} target="_blank" rel="noreferrer" className="gt-source-link" title={licence ? `Licence: ${licence}` : undefined}>
      {label} <ExternalLink size={11} aria-hidden="true" />
    </a>
  )
}

/* -------------------------------------------------------------- activity */

export function ActivitySection({ activity }) {
  if (!activity || (activity.available === false && !activity.evidence)) {
    return (
      <div className="gt-section">
        <Unavailable title="No activity estimate" reason={reasonOf(activity)} />
        {activity?.limitations && <p className="gt-limit">{activity.limitations}</p>}
      </div>
    )
  }
  const e = activity.evidence || {}
  const style = styleForActivity(activity.level)
  return (
    <div className="gt-section">
      <div className="gt-section-lead">
        <ActivityChip activity={activity} />
        <span className="gt-muted">
          score <strong className="gt-strong">{num(activity.score, 2)}</strong> · {style.label}
        </span>
        <Tag tone="info">Heuristic</Tag>
      </div>
      <p className="gt-callout">{copy.activityHeuristic}</p>
      {activity.available === false && <Unavailable title="Marked unavailable" reason={reasonOf(activity)} />}
      <Facts
        rows={[
          ["Echo clusters near object", num(e.echo_clusters_near, 0), isNum(e.window_m) ? `within a ${num(e.window_m, 0)} m window` : null],
          ["Echo area near object", num(e.echo_area_near_m2, 1, "m²")],
          ["Background clusters", num(e.background_clusters_per_window, 2), "per window of the same size"],
          ["Enrichment ratio", isNum(e.enrichment_ratio) ? `${num(e.enrichment_ratio, 2)}×` : DASH, "near ÷ background"],
          ["Sonar side", e.side ? titleCase(e.side) : DASH],
          ...(isNum(e.background_rows)
            ? [["Background measured over", `${num(e.background_rows, 0)} pings`,
                "same side, same line, clear of detections"]]
            : []),
          ...(isNum(e.control_windows) ? [["Control windows", num(e.control_windows, 0), "same side, same line"]] : []),
        ]}
      />
      {e.area_note && <p className="gt-limit">{e.area_note}</p>}
      {activity.formula && <code className="gt-formula">{activity.formula}</code>}
      {activity.basis && (
        <div className="gt-subsection">
          <h4>Basis</h4>
          <p>{activity.basis}</p>
        </div>
      )}
      <div className="gt-subsection">
        <h4>What this cannot tell you</h4>
        <TextList items={activity.limitations} empty="The engine recorded no limitations for this estimate." />
      </div>
    </div>
  )
}

/* --------------------------------------------------------------- habitat */

export function HabitatSection({ habitat, dataSources }) {
  if (isUnavailable(habitat)) {
    return (
      <div className="gt-section">
        <Unavailable title="Habitat not assessed" reason={reasonOf(habitat)} />
      </div>
    )
  }
  const inside = habitat.inside || []
  // Nearest feature of each kind, nearest first.
  const nearestByKind = new Map()
  for (const entry of habitat.nearest || []) {
    const kind = kindOf(entry)
    const current = nearestByKind.get(kind)
    if (!current || (isNum(entry.distance_m) && entry.distance_m < (current.distance_m ?? Infinity))) nearestByKind.set(kind, entry)
  }
  const rows = [...nearestByKind.values()].sort((a, b) => (a.distance_m ?? Infinity) - (b.distance_m ?? Infinity))

  return (
    <div className="gt-section">
      {habitat.covered === false && (
        <Unavailable
          title="Outside habitat layer coverage"
          reason="The bundled habitat layers do not cover this position, so 'nothing nearby' here means 'not known', not 'safe'."
        />
      )}
      {inside.length > 0 && (
        <div className="gt-inside">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>
            Inside:{" "}
            {inside.map((h, i) => (
              <strong key={i}>
                {i > 0 ? ", " : ""}
                {typeof h === "string" ? h : `${h.name} (${singularLabel(h.kind || h.layer)})`}
              </strong>
            ))}
          </span>
        </div>
      )}
      <div className="gt-section-lead">
        <span className="gt-muted">
          Habitat score <strong className="gt-strong">{num(habitat.score, 2)}</strong>
        </span>
      </div>
      {rows.length ? (
        <div className="gt-table-wrap">
          <table className="gt-table">
            <caption className="gt-sr-only">Nearest habitat feature of each kind</caption>
            <thead>
              <tr>
                <th scope="col">Kind</th>
                <th scope="col">Nearest feature</th>
                <th scope="col" className="gt-num">Distance</th>
                <th scope="col">Bearing</th>
                <th scope="col">Outline</th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${kindOf(row)}-${row.name}`}>
                  <td>{labelForKind(row.kind || row.layer)}</td>
                  <td>{row.name || DASH}</td>
                  <td className="gt-num">{distance(row.distance_m)}</td>
                  <td>{isNum(row.bearing_deg) ? `${compass(row.bearing_deg)} (${Math.round(row.bearing_deg)}°)` : DASH}</td>
                  <td>
                    <span className="gt-quality">{qualityLabel(row.geometry_quality)}</span>
                  </td>
                  <td>
                    <SourceLink source={row.source} dataSources={dataSources} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="gt-muted">No habitat features were recorded near this target.</p>
      )}
      <p className="gt-limit">
        Distances are measured to layer outlines. An approximate outline carries its own positional error, so a
        distance of a few hundred metres may mean "at the edge".
      </p>
      <Terms terms={habitat.terms} />
    </div>
  )
}

/* ----------------------------------------------------------------- drift */

export function DriftSection({ drift }) {
  if (isUnavailable(drift)) {
    return (
      <div className="gt-section">
        <Unavailable title="No drift forecast" reason={reasonOf(drift)} />
      </div>
    )
  }
  const impacts = (drift.impacts || []).slice().sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0))
  return (
    <div className="gt-section">
      <p className="gt-callout">{copy.driftModel}</p>
      <Facts
        rows={[
          ["Mode", titleCase(drift.mode || drift.requested_mode) || DASH, drift.mode_basis],
          ["Horizon", isNum(drift.horizon_hours) ? `${drift.horizon_hours} h (${arrival(drift.horizon_hours)})` : DASH],
          ["Particles", num(drift.n_particles, 0)],
          ["Current source", <SourceLink key="src" source={drift.current_source} />, isNum(drift.mean_current_ms) ? `mean ${drift.mean_current_ms} m/s` : null],
          ["Start time", drift.start_time ? dateTime(drift.start_time) : DASH, drift.start_time_basis],
          ["Stranding probability", pct(drift.stranding_probability)],
        ]}
      />
      <div className="gt-table-wrap">
        <table className="gt-table">
          <caption className="gt-sr-only">Forecast impacts</caption>
          <thead>
            <tr>
              <th scope="col">Feature</th>
              <th scope="col">Kind</th>
              <th scope="col" className="gt-num">Probability</th>
              <th scope="col" className="gt-num">First arrival</th>
              <th scope="col">Source</th>
            </tr>
          </thead>
          <tbody>
            {impacts.length ? (
              impacts.map((impact) => (
                <tr key={`${impact.kind}-${impact.name}`}>
                  <td>{impact.name}</td>
                  <td>{labelForKind(impact.kind)}</td>
                  <td className="gt-num gt-strong">{pct(impact.probability, 1)}</td>
                  <td className="gt-num">
                    {arrival(impact.first_arrival_hours)}
                    {isNum(impact.first_arrival_hours) && <small> ({impact.first_arrival_hours} h)</small>}
                  </td>
                  <td>
                    <SourceLink source={impact.source} />
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5} className="gt-muted">
                  No feature reached within the horizon.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="gt-two-col">
        <div className="gt-subsection">
          <h4>Assumptions</h4>
          <TextList items={drift.assumptions} />
        </div>
        <div className="gt-subsection">
          <h4>Limitations</h4>
          <TextList items={drift.limitations} />
        </div>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- people */

const STATUS = {
  ok: { Icon: CheckCircle2, label: "OK", className: "is-ok" },
  caution: { Icon: AlertTriangle, label: "Caution", className: "is-caution" },
  stop: { Icon: XCircle, label: "No-go", className: "is-stop" },
  unknown: { Icon: HelpCircle, label: "Unknown", className: "is-unknown" },
  info: { Icon: HelpCircle, label: "Info", className: "is-info" },
}

function CheckItem({ status, title, value, threshold, statusLabel }) {
  const s = STATUS[status] || STATUS.unknown
  return (
    <li className={`gt-check ${s.className}`}>
      <s.Icon size={18} aria-hidden="true" />
      <div>
        <div className="gt-check-head">
          <strong>{title}</strong>
          <span className="gt-check-status">{statusLabel || s.label}</span>
        </div>
        <div className="gt-check-value">{value}</div>
        {threshold && <small>{threshold}</small>}
      </div>
    </li>
  )
}

const boolStatus = (value) => (value === true ? "ok" : value === false ? "stop" : "unknown")

export function PeopleSection({ people, target }) {
  if (isUnavailable(people)) {
    return (
      <div className="gt-section">
        <Unavailable title="No people-at-risk assessment" reason={reasonOf(people)} />
      </div>
    )
  }
  const hazard = people.propeller_hazard
  const brief = people.diver_brief
  const thresholds = brief?.thresholds || {}
  // within_recreational_limit is checked against the advanced (30 m) limit;
  // the open-water figure is shown beside it so neither number is hidden.
  const depthLimit = thresholds.advanced_depth_m ?? thresholds.recreational_depth_limit_m ?? brief?.recreational_limit_m?.advanced
  const openWaterLimit = thresholds.recreational_depth_m
  const currentLimit = thresholds.max_current_mps ?? brief?.diver_current_limit_mps ?? brief?.max_current_mps
  const label = (value) => (value ? ` (${value})` : "")
  const depthThreshold = isNum(depthLimit)
    ? `Limit: ≤ ${depthLimit} m${isNum(openWaterLimit) ? ` (open water ≤ ${openWaterLimit} m)` : ""}${label(thresholds.depth_label)}`
    : null
  const currentThreshold = isNum(currentLimit)
    ? `Limit: ≤ ${currentLimit} m/s${isNum(thresholds.max_current_knots) ? ` (${thresholds.max_current_knots} kn)` : ""}${label(thresholds.current_label)}`
    : null
  const notStated = "threshold not stated in ghosttrace.json"
  const seabed = isNum(brief?.seabed_depth_m) ? brief.seabed_depth_m : target?.seabed_depth_m
  const has = (key) => brief && brief[key] !== undefined && brief[key] !== null
  const showChecklist = ["seabed_depth_m", "within_recreational_limit", "current_mps_at_depth", "current_ok_for_divers", "entanglement_risk", "net_size_m"].some(has)
  // The engine writes "high: large net; ..." — the level is the part before the colon.
  const entanglement = String(brief?.entanglement_risk || "").toLowerCase().split(":")[0].trim()
  const entStatus = entanglement === "high" ? "stop" : entanglement === "moderate" || entanglement === "medium" ? "caution" : entanglement === "low" ? "ok" : "unknown"

  return (
    <div className="gt-section">
      <div className="gt-subsection">
        <h4>Propeller hazard to boats</h4>
        {hazard ? (
          <>
            <div className="gt-section-lead">
              <HazardChip level={hazard.level} />
              <span className="gt-muted">{styleForHazard(hazard.level).label} hazard</span>
            </div>
            <TextList items={hazard.reasons} empty="No reasons recorded." />
            <Terms terms={hazard.terms} />
          </>
        ) : (
          <p className="gt-muted">Not assessed.</p>
        )}
      </div>

      <div className="gt-subsection">
        <h4>Diver safety brief</h4>
        {brief ? (
          <>
            {brief.summary && <p className="gt-callout">{brief.summary}</p>}
            {!showChecklist && (
              <p className="gt-muted">
                The brief carries no depth, current or entanglement fields, so no checklist is shown.
                {isNum(target?.seabed_depth_m) ? ` Seabed depth at the detection: ${num(target.seabed_depth_m, 1)} m (${target.seabed_depth_basis || "basis not stated"}).` : ""}
              </p>
            )}
            {showChecklist && <ul className="gt-checklist">
              <CheckItem
                status={boolStatus(brief.within_recreational_limit)}
                title="Depth within recreational limit"
                value={`Seabed ${num(seabed, 1, "m")}`}
                threshold={depthThreshold || notStated}
              />
              <CheckItem
                status={boolStatus(brief.current_ok_for_divers)}
                title="Current at depth"
                value={`${num(brief.current_mps_at_depth, 2, "m/s")}`}
                threshold={currentThreshold || notStated}
              />
              <CheckItem
                status={entStatus}
                title="Entanglement risk"
                value={brief.entanglement_risk ? titleCase(brief.entanglement_risk) : DASH}
                statusLabel={entStatus !== "unknown" ? `${titleCase(entanglement)} risk` : undefined}
                threshold="As assessed by the engine: high = red, moderate = amber, low = green"
              />
              <CheckItem status="info" title="Net size" value={num(brief.net_size_m, 1, "m")} />
            </ul>}
            {brief.recommended_method && (
              <p className="gt-method">
                <strong>Recommended method:</strong> {brief.recommended_method}
              </p>
            )}
            {brief.notes && <TextList items={brief.notes} />}
            {(thresholds.depth_basis || thresholds.current_basis) && (
              <details className="gt-terms">
                <summary>Where the limits come from</summary>
                <dl className="gt-facts gt-facts--dense">
                  {thresholds.depth_basis && (
                    <div>
                      <dt>Depth{label(thresholds.depth_label)}</dt>
                      <dd>{thresholds.depth_basis}</dd>
                    </div>
                  )}
                  {thresholds.current_basis && (
                    <div>
                      <dt>Current{label(thresholds.current_label)}</dt>
                      <dd>{thresholds.current_basis}</dd>
                    </div>
                  )}
                  {thresholds.basis && (
                    <div>
                      <dt>How they are applied</dt>
                      <dd>{thresholds.basis}</dd>
                    </div>
                  )}
                </dl>
              </details>
            )}
            <p className="gt-limit">Planning aid only. Confirm conditions on site; this is not a dive plan.</p>
          </>
        ) : (
          <p className="gt-muted">No diver brief recorded.</p>
        )}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- change */

export function ChangeSection({ change }) {
  if (!change) {
    return (
      <div className="gt-section">
        <Unavailable title="No change assessment" reason="The engine recorded no comparison for this target." />
      </div>
    )
  }
  return (
    <div className="gt-section">
      <div className="gt-section-lead">
        <ChangeChip status={change.status} />
        <span className="gt-muted">{styleForChange(change.status).label}</span>
      </div>
      <Facts
        rows={[
          ["Compared with survey", change.previous_survey_id || DASH],
          ["Previous detection", change.previous_detection_id || DASH],
          ["Moved", isNum(change.moved_m) ? distance(change.moved_m) : DASH],
          ...(isNum(change.previous_latitude) && isNum(change.previous_longitude)
            ? [["Previous position", `${change.previous_latitude.toFixed(5)}, ${change.previous_longitude.toFixed(5)}`]]
            : []),
        ]}
      />
      {change.basis && (
        <div className="gt-subsection">
          <h4>Basis</h4>
          <p>{change.basis}</p>
        </div>
      )}
      <p className="gt-limit">
        Matching across surveys is by position and class. Navigation error between surveys can make one object look
        moved, or two nearby objects look like one.
      </p>
    </div>
  )
}
