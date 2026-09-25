import { Anchor, Route } from "lucide-react"

import { styleForTier } from "../config"
import { DASH, isNum, num, stopId, titleCase } from "../format"
import { TierChip } from "./Chips"

/**
 * The recovery plan as the engine ordered it: start, stops, leg distances and
 * the total. Selecting a stop selects that target everywhere else.
 */
function RecoveryPlan({ plan, targetsById, selectedId, onSelect }) {
  if (!plan || !(plan.order || []).length) {
    return (
      <section className="gt-panel-card" aria-labelledby="gt-plan-title">
        <h2 id="gt-plan-title" className="gt-card-heading">
          <Route size={17} aria-hidden="true" /> Recovery plan
        </h2>
        <p className="gt-muted">{plan?.notes ? String(plan.notes) : "The engine produced no recovery plan for this survey."}</p>
      </section>
    )
  }

  const legs = plan.legs || []
  const legTo = new Map(legs.map((leg) => [String(leg.to), leg]))
  const notes = Array.isArray(plan.notes) ? plan.notes : plan.notes ? [plan.notes] : []
  const start = plan.start

  return (
    <section className="gt-panel-card" aria-labelledby="gt-plan-title">
      <header className="gt-section-head">
        <h2 id="gt-plan-title" className="gt-card-heading">
          <Route size={17} aria-hidden="true" /> Recovery plan
        </h2>
        <span className="gt-plan-total">
          {isNum(plan.total_km) ? `${num(plan.total_km, 1)} km` : DASH}
          <small>total</small>
        </span>
      </header>

      <ol className="gt-plan">
        {start && (
          <li className="gt-plan-start">
            <span className="gt-plan-dot">
              <Anchor size={13} aria-hidden="true" />
            </span>
            <div>
              <strong>{start.name || "Start"}</strong>
              <small>Start</small>
            </div>
          </li>
        )}
        {plan.order.map((stop, i) => {
          const id = stopId(stop)
          const target = targetsById.get(id)
          const leg = legTo.get(id)
          const tier = styleForTier(target?.priority?.tier)
          return (
            <li key={`${id}-${i}`} className={id === selectedId ? "is-selected" : ""}>
              {leg && (
                <span className="gt-plan-leg" aria-label={`Leg ${num(leg.distance_km, 1)} kilometres`}>
                  {num(leg.distance_km, 1)} km
                </span>
              )}
              <span className="gt-plan-dot gt-plan-dot--stop" style={{ "--gt-c": tier.color }}>
                {i + 1}
              </span>
              <button type="button" className="gt-plan-stop" onClick={() => target && onSelect(id)} disabled={!target}>
                <strong>{target ? titleCase(target.object_class) : "Unknown target"}</strong>
                <small>
                  {id}
                  {target?.priority?.rank != null ? ` · rank #${target.priority.rank}` : ""}
                </small>
              </button>
              {target && <TierChip tier={target.priority?.tier} compact />}
            </li>
          )
        })}
      </ol>

      {plan.method && (
        <p className="gt-plan-method">
          <strong>Method:</strong> {plan.method}
        </p>
      )}
      <ul className="gt-bullets gt-bullets--small">
        {notes.map((note, i) => (
          <li key={i}>{String(note)}</li>
        ))}
      </ul>
    </section>
  )
}

export default RecoveryPlan
