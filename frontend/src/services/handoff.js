/**
 * The seam between a stored detection and the grounded assistant.
 *
 * Mirrors src/survey/handoff.js, which does the same job for hotspots. No
 * retrieval, no prompt and no knowledge here: it builds one plain object and
 * hands it over through router state.
 *
 * The stored row is flat, with corner coordinates, because that is what a
 * database query wants. The assistant wants the detector's own record, with
 * [x, y, w, h] and the provenance fields that say which model made the call and
 * whether a class was withheld. Both are kept: the full record travels when the
 * row has one, and the flat columns are used to rebuild a minimal record when
 * it does not.
 */

/** The detection record for one stored row, in the shape the engine expects. */
export function detectionContext(row) {
  if (!row) return null

  // The whole detector record, saved at detection time. Preferred whenever it
  // is there, because it carries detector_model, detector_class, the visual
  // description and any withheld class, none of which survive in the columns.
  const record = row.record && Object.keys(row.record).length ? { ...row.record } : null
  if (record) return record

  const { x1 = 0, y1 = 0, x2 = 0, y2 = 0 } = row
  return {
    object_class: row.object_class || "unknown",
    confidence: row.confidence ?? null,
    bbox: [x1, y1, x2 - x1, y2 - y1],
  }
}

/**
 * The question the assistant is asked when a detection is handed over.
 *
 * About the class of object, not about the row. The corpus has documents about
 * wrecks and ordnance; it has nothing about detection 061fd772, and asking it
 * about one would invite an answer it cannot ground.
 */
export function detectionQuestion(row) {
  const label = row?.object_class
  const named = label && label !== "unknown" && label !== ""

  return named
    ? `Explain this ${label} contact and what is known about objects of that class.`
    : "The detector saw a contact and could not classify it. What should I do?"
}

/**
 * Everything the assistant route needs, ready to pass as router state.
 *
 * `stub` is read off the scan the row came from, never guessed at from the
 * record. A synthetic detection that reads like a real one is worse than no
 * detection, so the assistant has to be told which it was handed.
 */
export function handoffState(row, scan) {
  return {
    detectionContext: detectionContext(row),
    detectionKey: row?.id ?? null,
    detectionIsStub: Boolean(scan?.stub ?? row?.stub),
    question: detectionQuestion(row),
  }
}
