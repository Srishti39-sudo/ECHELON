import { copy } from '../config/copy'
import { languageFor } from '../config/languages'
import { theme } from '../config/theme'
import type { ChatResponse, GhostTraceContext, Severity, SurveyContext } from '../lib/types'
/**
 * The one-line verdict above an answer: what kind of answer it is, what the
 * classifier said, and the risk level.
 *
 * Severity comes from the backend, which looks it up rather than reading it out
 * of the generated text. This component only decides how it looks.
 */
export function StatusBadge({
  meta,
  survey,
  ghosttrace,
}: {
  meta: Partial<ChatResponse>
  survey?: SurveyContext | null
  ghosttrace?: GhostTraceContext | null
}) {
  // When a hotspot was handed over, the map's risk is the one displayed. The
  // assistant's own lookup is suppressed rather than shown beside it: one
  // hotspot carrying two different urgencies is how an operator stops trusting
  // both numbers.
  const severity = (survey
    ? (survey.severity_tier ?? 'unknown')
    : (meta.severity ?? 'unknown')) as Severity
  const tone = theme.severity[severity]
  const confidence = typeof meta.confidence === 'number' ? meta.confidence : null
  const labels = languageFor(meta.language).labels
  if (meta.mode === 'copilot') {
    // A copilot answer is about survey records, not one contact, so there is
    // no single risk level to show. The records carry their own tiers.
    return (
      <div className="badge-row">
        <span className="chip tone-steady" title={meta.route_reason ?? undefined}>
          {labels.copilot}
          {meta.route_reason?.startsWith('auto') && (
            <>
              <span className="chip-divider" aria-hidden="true" />
              {copy.copilot.autoRouted}
            </>
          )}
        </span>
        {meta.generated_by === 'data_only' && (
          <span className="chip tone-alert">{labels.dataOnly}</span>
        )}
        {meta.language && meta.language !== 'en' && (
          <span className="chip chip-plain">{languageFor(meta.language).native}</span>
        )}
      </div>
    )
  }
  return (
    <div className="badge-row">
      {meta.intent && <span className="chip chip-quiet">{copy.badge.intent[meta.intent]}</span>}
      {meta.object_class && !meta.is_anomaly && (
        <span className="chip chip-quiet">{meta.object_class}</span>
      )}
      {meta.is_anomaly && <span className="chip chip-quiet">{copy.badge.unclassified}</span>}
      {confidence !== null && (
        <span className="chip chip-plain">{copy.badge.confidence(confidence)}</span>
      )}
      {survey && <span className="chip chip-quiet">{copy.survey.hotspot} {survey.hotspot_id}</span>}
      {meta.generated_by === 'retrieval_only' && (
        <span className="chip tone-alert">{labels.offline}</span>
      )}
      {ghosttrace ? (
        // The rescue queue owns the priority, so its tier is what is shown and
        // the assistant's own class-table severity is not shown beside it.
        <>
          {ghosttrace.synthetic && (
            <span className="chip tone-caution">{copy.ghosttrace.synthetic}</span>
          )}
          <span className="chip chip-quiet">
            {copy.ghosttrace.priorityChip}
            <span className="chip-divider" aria-hidden="true" />
            {ghosttrace.priority?.tier ?? copy.ghosttrace.notAvailable}
          </span>
        </>
      ) : (
      <span className={`chip tone-${tone}`}>
        {survey ? copy.survey.severityFromMap : copy.badge.severityLabel}
        <span className="chip-divider" aria-hidden="true" />
        {copy.badge.severity[severity]}
      </span>
      )}
    </div>
  )
}
