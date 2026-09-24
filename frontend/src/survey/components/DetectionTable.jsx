import { Fragment } from "react"

import { copy, styleForTier } from "../config"
import {
  confidencePct,
  detectorConfidence,
  dimensions,
  formatPct,
  hardReasonLabels,
  verificationNotes,
  verificationReasons,
} from "../detections"

/**
 * One row per detection: what it is, how sure the engine is, how big it is,
 * how severe it is and which tile it was seen in.
 *
 * Confidence leads with the engine's verified 0-100 figure and keeps the
 * detector's own score beneath it, so the two are never confused. The reasons
 * verification recorded open underneath the row on request; an export written
 * before verification has none, and its rows look as they always did.
 */
function DetectionTable({ detections, emptyMessage }) {
  if (!detections.length) {
    return emptyMessage ? <p className="sv-muted">{emptyMessage}</p> : null
  }

  return (
    <div className="sv-inner-wrap">
      <table className="sv-inner-table">
        <thead>
          <tr>
            <th>Class</th>
            <th className="sv-col-num">Confidence</th>
            <th>Size</th>
            <th className="sv-col-num">Severity</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((detection) => {
            const rowStyle = styleForTier(detection.severity_tier)
            const pct = confidencePct(detection)
            const detector = detectorConfidence(detection)
            const size = dimensions(detection)
            const reasons = verificationReasons(detection)
            const hard = hardReasonLabels(detection)
            const notes = [...verificationNotes(detection), ...(size?.notes || [])]
            const filtered = detection.suppressed === true
            const expandable = reasons.length > 0 || notes.length > 0

            return (
              <Fragment key={detection.id}>
                <tr className={filtered ? "sv-detection-filtered" : undefined}>
                  <td>
                    {detection.object_class}
                    <span className="sv-chips">
                      {filtered && <span className="sv-flag sv-flag-filtered">{copy.filteredChip}</span>}
                      {detection.class_withheld && (
                        <span className="sv-flag sv-flag-withheld">{copy.withheldChip}</span>
                      )}
                    </span>
                    {/* The detector's own call, where this system judged it too
                        uncertain to assert. Kept visible: the operator should be
                        able to see what was withheld and why. */}
                    {detection.class_withheld && (
                      <span className="sv-withheld">
                        {detection.class_withheld} withheld below {detection.class_floor}
                      </span>
                    )}
                  </td>
                  <td className="sv-col-num">
                    {pct !== null ? (
                      <>
                        <strong>{formatPct(pct)}</strong>
                        {detector !== null && (
                          <span className="sv-sub">detector {detector.toFixed(2)}</span>
                        )}
                      </>
                    ) : (
                      detector?.toFixed(2)
                    )}
                  </td>
                  <td className="sv-size">
                    {size ? (
                      <>
                        {size.text}
                        {size.label && <span className="sv-sub">{size.label}</span>}
                      </>
                    ) : (
                      <span className="sv-muted">not recorded</span>
                    )}
                  </td>
                  <td className="sv-col-num" style={{ color: filtered ? undefined : rowStyle.color }}>
                    {detection.severity.toFixed(3)}
                    <span className="sv-sub">weight {detection.class_weight}</span>
                  </td>
                  <td className="sv-evidence">
                    {detection.provenance?.representative_tile}
                    {detection.provenance?.merged_count > 1 && (
                      <span className="sv-sub">
                        {detection.provenance.merged_count} views merged
                      </span>
                    )}
                    {detection.provenance?.second_opinion?.map((opinion, index) => (
                      <span className="sv-sub" key={index}>
                        {opinion.model} called it {opinion.object_class} at{" "}
                        {opinion.confidence.toFixed(2)}
                      </span>
                    ))}
                  </td>
                </tr>
                {expandable && (
                  <tr className="sv-reason-row">
                    <td colSpan={5}>
                      <details className="sv-reasons" open={filtered}>
                        <summary>
                          {filtered
                            ? `Why it was filtered${hard.length ? `: ${hard.join(", ")}` : ""}`
                            : reasons.length
                              ? `Verification notes (${reasons.length})`
                              : "How these numbers were measured"}
                        </summary>
                        {reasons.length > 0 && (
                          <ul>
                            {reasons.map((reason, index) => (
                              <li key={index}>{reason}</li>
                            ))}
                          </ul>
                        )}
                        {notes.map((note, index) => (
                          <p className="sv-sub" key={index}>
                            {note}
                          </p>
                        ))}
                      </details>
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default DetectionTable
