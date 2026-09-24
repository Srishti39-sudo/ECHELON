import { GitCompareArrows, XCircle } from "lucide-react"

import { changeStyle } from "../config"
import { isNum, latLngOf, titleCase } from "../format"

/**
 * What changed since the previous survey: counts, and the contacts that are
 * gone. A removed contact is listed with the engine's basis, because "not
 * seen again" and "recovered" are different claims and only the basis says which.
 */
function ChangePanel({ summary, removed }) {
  const counts = ["new", "moved", "persistent", "removed"]
  const compared = Array.isArray(summary?.compared_with) ? summary.compared_with.join(", ") : summary?.compared_with

  return (
    <section className="gt-panel-card" aria-labelledby="gt-change-title">
      <header className="gt-section-head">
        <h2 id="gt-change-title" className="gt-card-heading">
          <GitCompareArrows size={17} aria-hidden="true" /> Change since last survey
        </h2>
      </header>
      {summary ? (
        <>
          <p className="gt-muted">{compared ? <>Compared with <strong className="gt-strong">{compared}</strong></> : "No previous survey to compare with."}</p>
          <div className="gt-change-counts">
            {counts.map((key) => (
              <div key={key} className="gt-change-count" style={{ "--gt-c": changeStyle[key].color }}>
                <strong>{isNum(summary[key]) ? summary[key] : "—"}</strong>
                <span>{changeStyle[key].label}</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="gt-muted">The engine recorded no change summary.</p>
      )}

      {removed?.length > 0 && (
        <ul className="gt-removed-list">
          {removed.map((item, i) => {
            const ll = latLngOf(item)
            return (
              <li key={`${item.previous_detection_id}-${i}`}>
                <XCircle size={15} aria-hidden="true" />
                <div>
                  <strong>
                    {titleCase(item.object_class)} {item.previous_detection_id}
                  </strong>
                  <small>
                    {ll ? `${ll[0].toFixed(4)}, ${ll[1].toFixed(4)}` : "no position"} · from {item.previous_survey_id || "previous survey"}
                  </small>
                  {item.basis && <p>{item.basis}</p>}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

export default ChangePanel
