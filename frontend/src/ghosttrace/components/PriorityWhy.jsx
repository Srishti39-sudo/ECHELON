import { termLabels } from "../config"
import { DASH, isNum, titleCase } from "../format"

const fmt = (value, digits = 2) => (isNum(value) ? value.toFixed(digits) : DASH)

/**
 * The priority score, opened up: each term's value × weight = contribution,
 * any multiplier term, and the formula the engine recorded. Every number is
 * the engine's own. The recomputation is shown beside the recorded score so a
 * mismatch is visible rather than papered over.
 */
function PriorityWhy({ priority, id }) {
  const entries = Object.entries(priority?.terms || {})
  const additive = entries.filter(([, t]) => t?.role !== "multiplier")
  const multipliers = entries.filter(([, t]) => t?.role === "multiplier")
  const sum = additive.reduce((acc, [, t]) => acc + (isNum(t?.contribution) ? t.contribution : 0), 0)
  const factor = multipliers.reduce((acc, [, t]) => acc * (isNum(t?.value) ? t.value : 1), 1)
  const recomputed = sum * factor
  const score = priority?.score
  const agrees = !isNum(score) || Math.abs(recomputed - score) < 0.002
  const maxContribution = Math.max(0.0001, ...additive.map(([, t]) => (isNum(t?.contribution) ? t.contribution : 0)))
  const tiers = priority?.tiers

  return (
    <div className="gt-why" id={id}>
      {additive.length ? (
        <table className="gt-why-table">
          <caption className="gt-sr-only">Priority score terms</caption>
          <thead>
            <tr>
              <th scope="col">Term</th>
              <th scope="col" className="gt-num">Value</th>
              <th scope="col" aria-hidden="true" />
              <th scope="col" className="gt-num">Weight</th>
              <th scope="col" aria-hidden="true" />
              <th scope="col" className="gt-num">Adds</th>
            </tr>
          </thead>
          <tbody>
            {additive.map(([name, term]) => (
              <tr key={name} title={term?.basis || undefined}>
                <th scope="row">
                  <span className="gt-why-name">
                    {termLabels[name] || titleCase(name)}
                    {term?.measured === false && (
                      <span className="gt-why-neutral" title="Input unavailable; the engine used its stated neutral value">
                        neutral
                      </span>
                    )}
                  </span>
                  <span className="gt-why-bar" aria-hidden="true">
                    <span style={{ width: `${Math.max(0, ((term?.contribution ?? 0) / maxContribution) * 100)}%` }} />
                  </span>
                  {term?.basis && <small className="gt-why-basis">{term.basis}</small>}
                </th>
                <td className="gt-num">{fmt(term?.value)}</td>
                <td className="gt-op" aria-hidden="true">×</td>
                <td className="gt-num">{fmt(term?.weight)}</td>
                <td className="gt-op" aria-hidden="true">=</td>
                <td className="gt-num gt-strong">{fmt(term?.contribution, 3)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            {multipliers.length > 0 && (
              <>
                <tr>
                  <th scope="row">Sum of terms</th>
                  <td colSpan={4} />
                  <td className="gt-num">{fmt(isNum(priority?.weighted_sum) ? priority.weighted_sum : sum, 3)}</td>
                </tr>
                {multipliers.map(([name, term]) => (
                  <tr key={name} title={term?.basis || undefined}>
                    <th scope="row">
                      × {termLabels[name] || titleCase(name)} <span className="gt-why-neutral">multiplier</span>
                      {term?.basis && <small className="gt-why-basis">{term.basis}</small>}
                    </th>
                    <td colSpan={4} />
                    <td className="gt-num">× {fmt(term?.value)}</td>
                  </tr>
                ))}
              </>
            )}
            <tr className="gt-why-total">
              <th scope="row">Score</th>
              <td colSpan={4} className="gt-muted">
                {tiers ? `urgent ≥ ${tiers.urgent} · high ≥ ${tiers.high}` : ""}
              </td>
              <td className="gt-num gt-strong">{fmt(score, 3)}</td>
            </tr>
          </tfoot>
        </table>
      ) : (
        <p className="gt-muted">The engine recorded no individual terms for this score.</p>
      )}
      {!agrees && additive.length > 0 && (
        <p className="gt-warn-text">
          Recomputed from the terms: {recomputed.toFixed(3)}; recorded score: {fmt(score, 3)}. Shown as written by the engine.
        </p>
      )}
      {priority?.formula && <code className="gt-formula">{priority.formula}</code>}
      {priority?.basis && <p className="gt-why-foot">{priority.basis}</p>}
    </div>
  )
}

export default PriorityWhy
