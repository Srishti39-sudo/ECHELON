/**
 * The seam between GhostTrace and the grounded assistant.
 *
 * GhostTrace owns the rescue decision: the priority, its terms, the tier, the
 * activity evidence, the drift and the named authorities. The assistant owns
 * grounded explanation over the knowledge base. So, as with the hazard map's
 * hotspot handoff (src/survey/handoff.js), this module builds one plain object
 * copied out of ghosttrace.json and hands it over through router state:
 *
 *   navigate("/assistant", { state: { ghosttraceContext, question } })
 *
 * Nothing here recomputes a score, a tier or a probability, and nothing is
 * filled in: a stage that was unavailable becomes null, never a plausible
 * default, so the assistant can say "not assessed" instead of inventing it.
 *
 * THE SHAPE IS A CONTRACT. The assistant side is written against exactly the
 * keys below; add keys only additively and never rename one.
 */

import { isUnavailable, sourceName, topImpact } from "./format"

const orNull = (value) => (value === undefined ? null : value)

function numOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function impactOf(drift) {
  const impact = topImpact(drift)
  if (!impact) return null
  return {
    name: orNull(impact.name),
    kind: orNull(impact.kind),
    probability: numOrNull(impact.probability),
    first_arrival_hours: numOrNull(impact.first_arrival_hours),
  }
}

function priorityOf(priority) {
  if (!priority) return null
  const terms = {}
  for (const [name, term] of Object.entries(priority.terms || {})) {
    if (!term || typeof term !== "object") continue
    terms[name] = {
      value: numOrNull(term.value),
      weight: numOrNull(term.weight),
      contribution: numOrNull(term.contribution),
    }
  }
  return {
    score: numOrNull(priority.score),
    tier: orNull(priority.tier),
    rank: numOrNull(priority.rank),
    formula: orNull(priority.formula),
    terms,
  }
}

function activityOf(activity) {
  if (isUnavailable(activity)) return null
  const evidence = activity.evidence || {}
  return {
    level: orNull(activity.level),
    score: numOrNull(activity.score),
    enrichment_ratio: numOrNull(evidence.enrichment_ratio),
    echo_clusters_near: numOrNull(evidence.echo_clusters_near),
    background_clusters_per_window: numOrNull(evidence.background_clusters_per_window),
    limitations: orNull(activity.limitations),
  }
}

function habitatNearestOf(habitat) {
  if (isUnavailable(habitat)) return null
  return (habitat.nearest || [])
    .filter((entry) => entry && typeof entry === "object")
    .slice()
    .sort((a, b) => (numOrNull(a.distance_m) ?? Infinity) - (numOrNull(b.distance_m) ?? Infinity))
    .slice(0, 3)
    .map((entry) => ({
      name: orNull(entry.name),
      kind: orNull(entry.kind ?? entry.layer),
      distance_m: numOrNull(entry.distance_m),
      source: sourceName(entry.source),
    }))
}

function driftOf(drift) {
  if (isUnavailable(drift)) return null
  return {
    mode: orNull(drift.mode ?? drift.requested_mode),
    top_impact: impactOf(drift),
    stranding_probability: numOrNull(drift.stranding_probability),
  }
}

function refloatOf(target) {
  const scenario = target?.drift_scenarios?.if_refloated
  if (isUnavailable(scenario)) return null
  return { top_impact: impactOf(scenario) }
}

function peopleOf(people) {
  if (isUnavailable(people)) return null
  const brief = people.diver_brief || {}
  return {
    propeller_hazard_level: orNull(people.propeller_hazard?.level),
    diver_recommended_method: orNull(brief.recommended_method),
    seabed_depth_m: numOrNull(brief.seabed_depth_m),
    current_mps_at_depth: numOrNull(brief.current_mps_at_depth),
  }
}

function changeOf(change) {
  if (!change) return null
  return { status: orNull(change.status), moved_m: numOrNull(change.moved_m) }
}

/**
 * The context object for one GhostTrace target.
 *
 * `doc` is the ghosttrace.json document; a `title` on it (the survey's display
 * title, added by the page) is used for survey_title, else the survey id.
 */
export function ghosttraceContext(target, doc) {
  if (!target) return null
  const surveyId = doc?.survey_id ?? null
  return {
    kind: "ghosttrace_target",
    survey_id: surveyId,
    survey_title: doc?.title ?? surveyId,
    synthetic: Boolean(doc?.synthetic_inputs || doc?.demo),
    detection_id: orNull(target.detection_id),
    object_class: orNull(target.object_class),
    latitude: numOrNull(target.latitude),
    longitude: numOrNull(target.longitude),
    confidence_pct: numOrNull(target.confidence_pct),
    priority: priorityOf(target.priority),
    activity: activityOf(target.activity),
    habitat_nearest: habitatNearestOf(target.habitat),
    drift: driftOf(target.drift),
    refloat_scenario: refloatOf(target),
    people: peopleOf(target.people),
    change: changeOf(target.change),
    authorities: (target.alert?.authorities || [])
      .filter((a) => a && typeof a === "object")
      .map((a) => ({ name: orNull(a.name), role: orNull(a.role), situation: orNull(a.situation) })),
    caveats: (doc?.caveats || []).filter((c) => typeof c === "string"),
  }
}

/** The question asked when a target is handed over. */
export function ghosttraceQuestion(context) {
  if (!context) return ""
  const tier = context.priority?.tier || "unranked"
  return `Why is this net ranked ${tier} priority, and who should be told?`
}
