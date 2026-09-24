import { ExternalLink, ShieldQuestion } from "lucide-react"

import { copy } from "../config"
import { DASH } from "../format"

/**
 * "How this is computed & what it can't tell you": the caveats the engine
 * wrote and every data source it used, with licence and snapshot. A native
 * <details> so it is keyboard- and screen-reader-operable with no script.
 */
function MethodNotes({ sources = [], caveats = [], run }) {
  const stages = Object.entries(run?.stages || {})
  return (
    <details className="gt-method-notes">
      <summary>
        <ShieldQuestion size={16} aria-hidden="true" />
        {copy.explainerToggle}
        <span className="gt-muted">
          {caveats.length} caveat{caveats.length === 1 ? "" : "s"} · {sources.length} data source{sources.length === 1 ? "" : "s"}
        </span>
      </summary>

      <div className="gt-method-body">
        <div>
          <h3>What it can't tell you</h3>
          {caveats.length ? (
            <ul className="gt-bullets">
              {caveats.map((caveat, i) => (
                <li key={i}>{caveat}</li>
              ))}
            </ul>
          ) : (
            <p className="gt-muted">The engine recorded no caveats. That is not the same as there being none.</p>
          )}
          <h3>How the pieces work</h3>
          <ul className="gt-bullets">
            <li><strong>Activity</strong> — fish-echo clusters in the sonar water column near the object, against the survey's background rate. {copy.activityHeuristic}</li>
            <li><strong>Habitat</strong> — distance to bundled layers of reefs, protected areas, turtle nesting beaches and dugong habitat. Dashed outlines are approximate.</li>
            <li><strong>Drift</strong> — {copy.driftModel}</li>
            <li><strong>Priority</strong> — {copy.priorityHeuristic} Open "Why?" on any card to see each term.</li>
            <li><strong>Alerts</strong> — drafts for a person to check and send. {copy.alertNotSent}</li>
          </ul>
        </div>

        <div>
          <h3>Data sources</h3>
          {sources.length ? (
            <div className="gt-table-wrap">
              <table className="gt-table">
                <caption className="gt-sr-only">Data sources used by GhostTrace</caption>
                <thead>
                  <tr>
                    <th scope="col">Source</th>
                    <th scope="col">Used for</th>
                    <th scope="col">Licence</th>
                    <th scope="col">Snapshot</th>
                  </tr>
                </thead>
                <tbody>
                  {sources.map((source, i) => (
                    <tr key={`${source.name}-${i}`}>
                      <td>
                        {source.url ? (
                          <a href={source.url} target="_blank" rel="noreferrer" className="gt-source-link">
                            {source.name} <ExternalLink size={11} aria-hidden="true" />
                          </a>
                        ) : (
                          source.name || DASH
                        )}
                      </td>
                      <td>{source.used_for || DASH}</td>
                      <td>{source.licence || DASH}</td>
                      <td>{source.snapshot || DASH}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="gt-muted">No data sources were listed in ghosttrace.json.</p>
          )}

          {stages.length > 0 && (
            <>
              <h3>Analysis stages in this run</h3>
              <ul className="gt-stage-list">
                {stages.map(([name, stage]) => (
                  <li key={name} className={stage?.available ? "is-on" : "is-off"}>
                    <span className="gt-stage-dot" aria-hidden="true" />
                    <strong>{name}</strong>
                    <span>{stage?.available ? "available" : "unavailable"}</span>
                    {stage?.reason && <small>{stage.reason}</small>}
                  </li>
                ))}
              </ul>
            </>
          )}
          {run?.heuristic && <p className="gt-muted gt-small">Weights and thresholds: {run.heuristic}.</p>}
        </div>
      </div>
    </details>
  )
}

export default MethodNotes
