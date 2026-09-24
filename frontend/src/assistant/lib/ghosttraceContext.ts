import type { GhostTraceContext, GhostTraceImpact } from './types'

/**
 * Build the assistant's GhostTrace handoff from one target of a ghosttrace.json.
 *
 * The rescue queue builds this object when its "ask the assistant" button is
 * pressed. This copy exists for the dev-only demo route on the assistant page
 * (/assistant?demoContext=ghosttrace) and mirrors eval/ghosttrace_context.py,
 * which the evaluation cases are generated with. Every value is copied from the
 * document; none is computed. A stage that did not run becomes null.
 */

type Loose = Record<string, unknown>

const obj = (value: unknown): Loose | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Loose) : null

const stage = (value: unknown): Loose | null => {
  const block = obj(value)
  return block && block.available !== false ? block : null
}

const num = (value: unknown): number | null => (typeof value === 'number' ? value : null)
const str = (value: unknown): string | null => (typeof value === 'string' ? value : null)

function topImpact(impacts: unknown): GhostTraceImpact | null {
  if (!Array.isArray(impacts)) return null
  const ranked = impacts
    .map(obj)
    .filter((i): i is Loose => i !== null)
    .sort((a, b) => (num(b.probability) ?? 0) - (num(a.probability) ?? 0))
  const top = ranked[0]
  if (!top) return null
  return {
    name: str(top.name),
    kind: str(top.kind),
    probability: num(top.probability),
    first_arrival_hours: num(top.first_arrival_hours),
  }
}

export function ghosttraceContextFromTarget(doc: Loose, target: Loose): GhostTraceContext {
  const priority = obj(target.priority)
  const activity = stage(target.activity)
  const evidence = obj(activity?.evidence) ?? {}
  const habitat = stage(target.habitat)
  const drift = stage(target.drift)
  const people = stage(target.people)
  const diver = obj(people?.diver_brief) ?? {}
  const change = obj(target.change) ?? {}
  const refloat = stage(obj(target.drift_scenarios)?.if_refloated)
  const alert = obj(target.alert) ?? {}
  const terms = obj(priority?.terms) ?? {}

  return {
    kind: 'ghosttrace_target',
    survey_id: str(doc.survey_id),
    survey_title: str(doc.title),
    synthetic: Boolean(doc.demo || doc.synthetic_inputs),
    detection_id: str(target.detection_id),
    object_class: str(target.object_class),
    latitude: num(target.latitude),
    longitude: num(target.longitude),
    confidence_pct: num(target.confidence_pct),
    priority: priority
      ? {
          score: num(priority.score),
          tier: str(priority.tier),
          rank: num(priority.rank),
          formula: str(priority.formula),
          terms: Object.fromEntries(
            Object.entries(terms).map(([name, raw]) => {
              const term = obj(raw) ?? {}
              return [
                name,
                { value: num(term.value), weight: num(term.weight), contribution: num(term.contribution) },
              ]
            }),
          ),
        }
      : null,
    activity: activity
      ? {
          level: str(activity.level),
          score: num(activity.score),
          enrichment_ratio: num(evidence.enrichment_ratio),
          echo_clusters_near: num(evidence.echo_clusters_near),
          background_clusters_per_window: num(evidence.background_clusters_per_window),
          limitations: str(activity.limitations),
        }
      : null,
    habitat_nearest: (Array.isArray(habitat?.nearest) ? habitat.nearest : [])
      .map(obj)
      .filter((h): h is Loose => h !== null)
      .slice(0, 3)
      .map((h) => ({
        name: str(h.name),
        kind: str(h.kind),
        distance_m: num(h.distance_m),
        source: typeof h.source === 'string' ? h.source : obj(h.source),
      })),
    drift: drift
      ? {
          mode: str(drift.mode) ?? str(drift.requested_mode),
          top_impact: topImpact(drift.impacts),
          stranding_probability: num(drift.stranding_probability),
        }
      : null,
    refloat_scenario: refloat ? { top_impact: topImpact(refloat.impacts) } : null,
    people: people
      ? {
          propeller_hazard_level: str(obj(people.propeller_hazard)?.level),
          diver_recommended_method: str(diver.recommended_method),
          seabed_depth_m: num(diver.seabed_depth_m) ?? num(target.seabed_depth_m),
          current_mps_at_depth: num(diver.current_mps_at_depth),
        }
      : null,
    change: { status: str(change.status), moved_m: num(change.moved_m) },
    authorities: (Array.isArray(alert.authorities) ? alert.authorities : [])
      .map(obj)
      .filter((a): a is Loose => a !== null)
      .map((a) => ({ name: str(a.name), role: str(a.role), situation: str(a.situation) })),
    caveats: Array.isArray(doc.caveats) ? doc.caveats.filter((c): c is string => typeof c === 'string') : [],
  }
}
