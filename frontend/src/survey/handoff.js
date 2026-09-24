/**
 * The seam between the hazard map and the grounded assistant.
 *
 * THE BOUNDARY, AND WHY IT IS DRAWN HERE
 *
 *   The hazard map answers  WHERE something is and HOW URGENT it is.
 *   The assistant answers   WHAT it is and WHAT IS KNOWN about it.
 *
 * Those are different questions with different evidence and different ways of
 * being wrong. A severity is arithmetic over a detector's output and can be
 * recomputed by hand. A grounded answer is retrieval over a document corpus.
 * Merging them would let a confident sentence raise a priority, or a priority
 * imply a fact, and neither system can support that.
 *
 * So this module contains no retrieval, no prompt and no knowledge. It builds
 * one plain object and hands it over.
 *
 * WHY IT IS NOT A DETECTION RECORD
 *
 * The assistant already accepts a detection record, and a hotspot context
 * would appear to fit through that door: the schema allows extra keys. It must
 * not go that way. The assistant maps object_class onto its own label and then
 * looks severity up in its own table, so the same hotspot would carry one
 * urgency on the map and a different one beside the answer, with nobody able
 * to see both numbers at once.
 *
 * The map owns urgency. So the context travels as its own field, the
 * assistant displays the severity it is given rather than deriving one, and
 * there is exactly one number for a given hotspot anywhere in the product.
 */

/**
 * Build the context object for a hotspot.
 *
 * The first eight fields are the agreed contract with the assistant. The rest
 * are additive and safe to ignore; nothing downstream is required to read them.
 * Every value is copied straight out of export.json and none is recomputed.
 */
export function hotspotContext(hotspot, survey) {
  if (!hotspot) return null

  const centroid = hotspot.centroid || {}
  const top = hotspot.top_detection || {}
  const byId = new Map((survey?.detections || []).map((d) => [d.id, d]))
  const members = (hotspot.detection_ids || []).map((id) => byId.get(id)).filter(Boolean)
  const topRecord = byId.get(top.id) || null

  return {
    hotspot_id: hotspot.hotspot_id,
    dominant_class: hotspot.dominant_class,
    severity: hotspot.max_severity,
    confidence: hotspot.confidence_max,
    centroid: {
      global_x: centroid.global_x,
      global_y: centroid.global_y,
    },
    // Null rather than absent, and null wherever the survey has no navigation.
    // The assistant is expected to render "not georeferenced" rather than to
    // fill a gap, which is the same rule the engine applies upstream.
    lat: centroid.latitude ?? null,
    lon: centroid.longitude ?? null,
    recommended_action: hotspot.recommended_action,

    // Additive context. Useful for display and provenance, never required.
    severity_tier: hotspot.severity_tier,
    priority_rank: hotspot.priority_rank,
    detection_count: hotspot.detection_count,
    total_severity: hotspot.total_severity,
    coordinate_mode: survey?.survey_summary?.coordinate_mode ?? null,
    survey_id: survey?.metadata?.survey_id ?? null,
    // Carried so the assistant can mark synthetic data as synthetic. Set by
    // the engine; never inferred here.
    demo: Boolean(survey?.metadata?.demo),
    evidence_tile: top.tile ?? null,

    // Verification, copied from export.json. Null wherever the export predates
    // it, never filled in. confidence_pct and dimensions are those of the
    // detection that sets the hotspot's severity; `detections` lists every
    // detection in the cell with its own figures and reasons.
    confidence_pct: topRecord?.confidence_pct ?? null,
    confidence_pct_basis: topRecord?.confidence_pct_basis ?? null,
    dimensions: topRecord?.dimensions ?? null,
    suppressed: topRecord ? topRecord.suppressed ?? null : null,
    verification_reasons: topRecord?.verification?.reasons ?? [],
    detections: members.map(detectionSummary),
    survey_suppressed_detections: survey?.survey_summary?.suppressed_detections ?? null,
  }
}

/** One detection as the handoff carries it: identity, figures, and reasons. */
function detectionSummary(detection) {
  return {
    id: detection.id,
    object_class: detection.object_class,
    confidence: detection.confidence,
    confidence_pct: detection.confidence_pct ?? null,
    dimensions: detection.dimensions ?? null,
    suppressed: detection.suppressed ?? null,
    verification_reasons: detection.verification?.reasons ?? [],
    hard_reasons: detection.verification?.hard_reasons ?? [],
  }
}

/**
 * The question the assistant is asked when a hotspot is handed over.
 *
 * Deliberately about the class of object, not about the hotspot. The corpus
 * has documents about mines and wrecks; it has nothing about grid cell H003,
 * and asking it about one would invite an answer it cannot ground.
 */
export function handoffQuestion(context) {
  if (!context) return ""
  const name = context.dominant_class || "unidentified object"
  return `Explain this hazard and provide grounded information about ${name}.`
}
