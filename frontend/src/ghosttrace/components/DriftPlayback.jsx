import { Pause, Play, RotateCcw, Waves } from "lucide-react"

import { colorForKind, copy, labelForKind } from "../config"
import { arrival, hoursLabel, isNum, isUnavailable, pct, sourceName, titleCase } from "../format"
import { InfoTip, Tag } from "./Chips"

/**
 * Time slider, play button and the impact list for the selected target.
 *
 * Playback never starts on its own when the operator prefers reduced motion;
 * the panel decides that and this only renders the controls. An impact is
 * marked "reached" once the slider passes its first arrival time.
 */
function DriftPlayback({
  target,
  drift: driftProp,
  scenario = "forecast",
  hasRefloatScenario = false,
  onScenario,
  snapshots,
  playback,
  reducedMotion,
}) {
  const drift = driftProp ?? target?.drift

  if (!target) return null

  // Offered only for a seabed net whose engine run produced the scenario.
  const scenarioSwitch = hasRefloatScenario && onScenario && (
    <div className="gt-scenario" role="radiogroup" aria-label="Drift scenario">
      <button
        type="button"
        role="radio"
        aria-checked={scenario === "forecast"}
        className={scenario === "forecast" ? "is-active" : ""}
        onClick={() => onScenario("forecast")}
      >
        Forecast: on the seabed
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={scenario === "if_refloated"}
        className={scenario === "if_refloated" ? "is-active" : ""}
        onClick={() => onScenario("if_refloated")}
      >
        Scenario: if refloated
      </button>
    </div>
  )
  const scenarioNote = scenario === "if_refloated" && (
    <p className="gt-callout gt-scenario-note">
      <Tag tone="warn">Scenario, not a forecast</Tag>{" "}
      {drift?.scenario_note ||
        "What this net would reach if lifted off the seabed. Not used in the priority score."}
    </p>
  )

  if (isUnavailable(drift)) {
    return (
      <section className="gt-playback is-unavailable" aria-label="Drift forecast playback">
        <Waves size={16} aria-hidden="true" />
        <p>
          <strong>No drift forecast for {target.detection_id}.</strong> {drift?.reason || "The engine did not provide one."}
        </p>
      </section>
    )
  }

  const { index, last, playing } = playback
  const snapshot = snapshots[index]
  const t = snapshot?.t_hours ?? 0
  const horizon = drift.horizon_hours ?? snapshots[last]?.t_hours
  const step = snapshots.length > 1 ? (snapshots[1].t_hours ?? 0) - (snapshots[0].t_hours ?? 0) : null
  const impacts = (drift.impacts || []).slice().sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0))
  const pointCount = snapshot?.points?.length ?? 0

  if (!snapshots.length) {
    return (
      <section className="gt-playback is-unavailable" aria-label="Drift forecast playback">
        <Waves size={16} aria-hidden="true" />
        <p>The forecast for {target.detection_id} has no snapshots to play.</p>
      </section>
    )
  }

  return (
    <section className="gt-playback" aria-label={`Drift forecast playback for ${target.detection_id}`}>
      {scenarioSwitch}
      {scenarioNote}
      <div className="gt-playback-bar">
        <button
          type="button"
          className="gt-play"
          onClick={playing ? playback.pause : playback.play}
          aria-label={playing ? "Pause drift playback" : "Play drift forecast"}
          disabled={last === 0}
        >
          {playing ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
        </button>
        <button
          type="button"
          className="gt-icon-btn"
          onClick={() => playback.seek(0)}
          aria-label="Back to t = 0"
          disabled={index === 0}
        >
          <RotateCcw size={15} aria-hidden="true" />
        </button>

        <div className="gt-slider">
          <input
            type="range"
            min={0}
            max={last}
            step={1}
            value={index}
            onChange={(e) => playback.seek(Number(e.target.value))}
            aria-label="Drift forecast time"
            aria-valuetext={`t = ${hoursLabel(t)}`}
          />
          <div className="gt-slider-scale" aria-hidden="true">
            <span>0 h</span>
            <span>{hoursLabel(horizon)}</span>
          </div>
        </div>

        <div className="gt-clock" aria-live={playing ? "off" : "polite"}>
          <span className="gt-clock-label">t =</span>
          <span className="gt-clock-value">{hoursLabel(t)}</span>
        </div>
      </div>

      <div className="gt-playback-meta">
        <span>
          <strong>{titleCase(target.object_class)} {target.detection_id}</strong> · {titleCase(drift.mode || "forecast")} ·{" "}
          {isNum(drift.n_particles) ? `${drift.n_particles} particle${drift.n_particles === 1 ? "" : "s"}` : `${pointCount} point${pointCount === 1 ? "" : "s"}`}
          {step ? ` · ${step} h steps` : ""} · horizon {hoursLabel(horizon)}
        </span>
        <span className="gt-playback-source">
          Current: {sourceName(drift.current_source) || "source not stated"}
          {isNum(drift.mean_current_ms) ? ` · mean ${drift.mean_current_ms} m/s` : ""}
          {/synthetic|fake|test/i.test(sourceName(drift.current_source) || "") && <Tag tone="warn">Synthetic current</Tag>}
        </span>
        <span className="gt-muted gt-playback-note">
          {copy.driftModel}
          {reducedMotion ? " Reduced motion is on: playback starts only when you press play." : ""}
        </span>
      </div>

      <div className="gt-impacts">
        <div className="gt-impacts-head">
          <h3>Where it may reach</h3>
          <span className="gt-muted">
            Stranding probability{" "}
            <strong className="gt-strong">{pct(drift.stranding_probability)}</strong>
            <InfoTip label="What stranding probability means" align="end">
              The engine's estimate that the object comes ashore within the forecast horizon, under the assumptions listed in the detail panel. A model output, not an observation.
            </InfoTip>
          </span>
        </div>
        {impacts.length ? (
          <ul className="gt-impact-list">
            {impacts.map((impact) => {
              const reached = isNum(impact.first_arrival_hours) && t >= impact.first_arrival_hours && impact.probability > 0
              const color = colorForKind(impact.kind)
              return (
                <li key={`${impact.kind}-${impact.name}`} className={reached ? "is-reached" : ""} style={{ "--gt-c": color }}>
                  <span className="gt-impact-name">
                    <span className="gt-swatch" aria-hidden="true" />
                    <span>
                      {impact.name}
                      <small>{labelForKind(impact.kind)}</small>
                    </span>
                  </span>
                  <span className="gt-impact-bar" aria-hidden="true">
                    <span style={{ width: `${Math.max(2, (impact.probability ?? 0) * 100)}%` }} />
                  </span>
                  <span className="gt-impact-prob">{pct(impact.probability)}</span>
                  <span className="gt-impact-when">
                    first {arrival(impact.first_arrival_hours)}
                    {reached && <span className="gt-reached">reached</span>}
                  </span>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="gt-muted">No habitat feature is reached by any simulated particle within the horizon.</p>
        )}
      </div>
    </section>
  )
}

export default DriftPlayback
