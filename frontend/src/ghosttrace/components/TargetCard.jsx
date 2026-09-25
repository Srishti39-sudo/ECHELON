import { useState } from "react"
import { ChevronDown, Leaf, MessageSquare, Waves } from "lucide-react"

import { singularLabel, styleForTier } from "../config"
import {
  arrival,
  compass,
  confidence,
  distance,
  headlineHabitat,
  isNum,
  isUnavailable,
  num,
  pct,
  size,
  titleCase,
  topImpact,
} from "../format"
import { ActivityChip, ChangeChip, HazardChip, InfoTip, Tag, TierChip } from "./Chips"
import PriorityWhy from "./PriorityWhy"

function ActivityEvidence({ activity }) {
  if (isUnavailable(activity)) return activity?.reason || "No activity estimate for this target."
  const e = activity.evidence || {}
  return (
    <>
      <strong>Water-column echo evidence</strong>
      <span>
        {num(e.echo_clusters_near, 0)} echo clusters within {num(e.window_m, 0, "m")} ({num(e.echo_area_near_m2, 1, "m²")})
      </span>
      <span>Background: {num(e.background_clusters_per_window, 1)} per window</span>
      <span>Enrichment: {num(e.enrichment_ratio, 1)}×{e.side ? ` · ${e.side} side` : ""}</span>
      <em>Heuristic: echoes may be fish, bubbles or sediment; not proof of catching.</em>
    </>
  )
}

/**
 * One target in the rescue queue.
 *
 * The main button selects the target (map, drift and detail follow). The
 * "why" toggle and the tooltip are separate controls, so nothing interactive
 * is nested inside another.
 */
function TargetCard({ target, selected, onSelect, onOpenDetail, onAsk, buttonRef, onKeyDown }) {
  const [whyOpen, setWhyOpen] = useState(false)
  const priority = target.priority || {}
  const tier = styleForTier(priority.tier)
  const habitat = headlineHabitat(target.habitat)
  const impact = topImpact(target.drift)
  const people = target.people
  const whyId = `gt-why-${target.detection_id}`

  return (
    <li
      className={`gt-card${selected ? " is-selected" : ""}${target.suppressed ? " is-suppressed" : ""}`}
      style={{ "--gt-tier": tier.color }}
    >
      <button
        type="button"
        ref={buttonRef}
        className="gt-card-main"
        aria-current={selected ? "true" : undefined}
        onClick={() => onSelect(target.detection_id)}
        onDoubleClick={() => onOpenDetail(target.detection_id)}
        onKeyDown={onKeyDown}
      >
        <span className="gt-card-rank" aria-label={`Rank ${isNum(priority.rank) ? priority.rank : "unranked"}`}>
          {isNum(priority.rank) ? `#${priority.rank}` : "—"}
        </span>
        <span className="gt-card-head">
          <span className="gt-card-title">
            {titleCase(target.object_class)}
            {priority.confidence_missing ? (
              <span
                className="gt-card-conf gt-card-conf--missing"
                title="No confidence reached the scorer; a neutral 0.5 multiplier was used. A data fault, not a low-risk result."
              >
                confidence missing
              </span>
            ) : (
              <span className="gt-card-conf">{confidence(target.confidence_pct)} conf.</span>
            )}
          </span>
          <span className="gt-card-sub">
            {size(target.dimensions)} · {target.detection_id}
          </span>
        </span>
        <span className="gt-card-score" aria-label={`Priority score ${num(priority.score, 2)}`}>
          {isNum(priority.score) ? priority.score.toFixed(2) : "—"}
        </span>
      </button>

      <div className="gt-card-body">
        <div className="gt-chips">
          <TierChip tier={priority.tier} compact />
          {target.suppressed && <Tag tone="muted" title="The engine suppressed this contact">Suppressed</Tag>}
          <span className="gt-chip-with-tip">
            <ActivityChip activity={target.activity} />
            <InfoTip label="Activity evidence">
              <ActivityEvidence activity={target.activity} />
            </InfoTip>
          </span>
          {target.change?.status && <ChangeChip status={target.change.status} />}
          {!isUnavailable(people) && people.propeller_hazard?.level && (
            <HazardChip level={people.propeller_hazard.level} />
          )}
        </div>

        <dl className="gt-card-facts">
          <div>
            <dt>
              <Leaf size={13} aria-hidden="true" /> Habitat
            </dt>
            <dd>
              {isUnavailable(target.habitat)
                ? "not assessed"
                : habitat
                  ? habitat.inside
                    ? `Inside ${habitat.name}`
                    : `${habitat.name} · ${distance(habitat.distance_m)} ${compass(habitat.bearing_deg)}`
                  : target.habitat.covered === false
                    ? "outside layer coverage"
                    : "none nearby in layers"}
              {habitat && (
                <span className="gt-muted"> ({singularLabel(habitat.kind || habitat.layer)})</span>
              )}
            </dd>
          </div>
          <div>
            <dt>
              <Waves size={13} aria-hidden="true" /> Drift
            </dt>
            <dd>
              {isUnavailable(target.drift)
                ? "no forecast"
                : impact
                  ? `${pct(impact.probability)} reach ${impact.name} · ${arrival(impact.first_arrival_hours)}`
                  : "no habitat reached in forecast"}
            </dd>
          </div>
        </dl>

        <div className="gt-card-actions">
          <button
            type="button"
            className="gt-link"
            aria-expanded={whyOpen}
            aria-controls={whyId}
            onClick={() => setWhyOpen((open) => !open)}
          >
            <ChevronDown size={14} className={whyOpen ? "gt-rot" : ""} aria-hidden="true" />
            Why {isNum(priority.score) ? priority.score.toFixed(2) : "this score"}?
          </button>
          <button type="button" className="gt-link" onClick={() => onOpenDetail(target.detection_id)}>
            Open details
          </button>
          {onAsk && (
            <button
              type="button"
              className="gt-link"
              onClick={() => onAsk(target)}
              aria-label={`Ask the assistant about ${target.object_class || "this target"} ${target.detection_id}`}
            >
              <MessageSquare size={14} aria-hidden="true" />
              Ask the assistant
            </button>
          )}
        </div>
        {whyOpen && <PriorityWhy priority={priority} id={whyId} />}
      </div>
    </li>
  )
}

export default TargetCard
