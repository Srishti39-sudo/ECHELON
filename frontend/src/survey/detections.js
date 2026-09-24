/**
 * Reading the verification fields of an exported detection.
 *
 * Every value here is read out of export.json as the engine wrote it: the
 * 0-100 confidence, the size in metres, whether verification filtered the
 * detection and the reasons it gave. Nothing is recomputed, and an export
 * written before verification existed simply has none of these fields, which
 * every function below answers with null or an empty list rather than a guess.
 */

import { copy } from "./config"

function finite(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

/** True when the export carries verification at all. */
export function isVerified(survey) {
  if (!survey) return false
  if (survey.survey_summary && "suppressed_detections" in survey.survey_summary) return true
  return (survey.detections || []).some((d) => d && "suppressed" in d)
}

/** Detections verification filtered as likely false positives. */
export function filteredDetections(survey) {
  return (survey?.detections || []).filter((d) => d?.suppressed === true)
}

/** The engine's 0-100 confidence, or null for an older export. */
export function confidencePct(detection) {
  return finite(detection?.confidence_pct)
}

export function formatPct(value) {
  return `${value.toFixed(1)}%`
}

/** The detector's own 0-1 score, as the verification block kept it. */
export function detectorConfidence(detection) {
  return finite(detection?.verification?.detector_confidence) ?? finite(detection?.confidence)
}

/**
 * Size as the engine measured it: { text, label, notes }, or null.
 *
 * Metres only where the export has metres. A strip with no known resolution
 * still has a box in pixels, and that is shown as pixels, labelled as such.
 */
export function dimensions(detection) {
  const dims = detection?.dimensions
  if (!dims || typeof dims !== "object") return null
  const length = finite(dims.length_m)
  const width = finite(dims.width_m)
  const height = finite(dims.height_m)
  const notes = [dims.basis, dims.height_basis].filter(Boolean).map(String)

  if (length !== null && width !== null) {
    return height !== null
      ? { text: `${length.toFixed(1)} × ${width.toFixed(1)} × ${height.toFixed(1)} m`, label: "L × W × H", notes }
      : { text: `${length.toFixed(1)} × ${width.toFixed(1)} m`, label: "L × W", notes }
  }
  const lengthPx = finite(dims.length_px)
  const widthPx = finite(dims.width_px)
  if (lengthPx !== null && widthPx !== null) {
    return {
      text: `${lengthPx.toFixed(0)} × ${widthPx.toFixed(0)} px`,
      label: height !== null ? `not in metres; height ${height.toFixed(1)} m` : "not measured in metres",
      notes,
    }
  }
  if (height !== null) return { text: `height ${height.toFixed(1)} m`, label: "", notes }
  return notes.length ? { text: "not measured", label: "", notes } : null
}

/** Plain-English reasons verification recorded, filtered or not. */
export function verificationReasons(detection) {
  const reasons = detection?.verification?.reasons
  return Array.isArray(reasons) ? reasons.filter(Boolean).map(String) : []
}

/** The artefact categories that justified filtering, in words. */
export function hardReasonLabels(detection) {
  const hard = detection?.verification?.hard_reasons
  if (!Array.isArray(hard)) return []
  return hard.map((code) => copy.hardReasons[code] || String(code).replaceAll("_", " "))
}

/** Verification notes and the confidence basis, for the expanded view. */
export function verificationNotes(detection) {
  const notes = []
  const verification = detection?.verification
  if (verification?.status === "not_checked" && verification.reason) {
    notes.push(`Not checked: ${verification.reason}`)
  }
  if (detection?.confidence_pct_basis) notes.push(String(detection.confidence_pct_basis))
  if (Array.isArray(verification?.notes)) notes.push(...verification.notes.map(String))
  return notes
}
