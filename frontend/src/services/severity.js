/**
 * How a severity is shown, and how two of them compare.
 *
 * The backend decides which tier something is in; this only decides how it
 * looks. A tier that is not listed renders in the neutral fallback rather than
 * guessing, because picking a colour for an unknown risk level is itself a
 * claim about the risk.
 *
 * `unknown` is deliberately not grey-and-quiet. An unidentified object is the
 * case this system handles most carefully, and the backend's own severity table
 * says the interface is expected to render unknown with the same caution as
 * high. So it gets a colour of its own and sits above medium in every ordering.
 *
 * Constants and functions live here rather than beside the badge component so
 * that file exports only a component, which is what keeps fast refresh working.
 */

export const severityStyle = {
  high: { label: "High", color: "var(--danger)", tint: "#fdeaea" },
  unknown: { label: "Unidentified", color: "var(--purple)", tint: "#f2ecfd" },
  medium: { label: "Medium", color: "var(--warning)", tint: "#fdf3e7" },
  low: { label: "Low", color: "var(--success)", tint: "#eaf6ef" },
}

export const severityFallback = {
  label: "Unclassified",
  color: "var(--muted)",
  tint: "#eef2f6",
}

export function styleForSeverity(tier) {
  return severityStyle[String(tier || "").toLowerCase()] || severityFallback
}

/** Worst first, and unknown directly below high. Mirrors the backend's rank. */
export const severityRank = { high: 3, unknown: 2, medium: 1, low: 0 }

export function compareSeverity(a, b) {
  return (severityRank[b] ?? 2) - (severityRank[a] ?? 2)
}

// --- Verification -----------------------------------------------------------
//
// Every detection carries a 0-100 confidence_pct: the detector's score fused
// with evidence measured in the image (acoustic shadow, rock clutter, the nadir
// and water column, dropouts). A detection the evidence argues against is
// `suppressed`, kept and shown as filtered with its reasons, never deleted.
//
// These read either shape a detection arrives in: the upload response (the
// fields on the record itself) and a stored row (the fields flattened beside the
// columns, with the record under `record`). A row stored before verification
// existed has none of them, and is shown as unverified, never as filtered.

function verificationOf(d) {
  return d?.verification ?? d?.record?.verification ?? null
}

/** The 0-100 figure to show, or null when there is none at all. */
export function confidencePct(d) {
  const pct = d?.confidence_pct ?? d?.record?.confidence_pct
  if (typeof pct === "number") return pct
  return typeof d?.confidence === "number" ? d.confidence * 100 : null
}

export function formatPct(value) {
  return typeof value === "number" ? `${value.toFixed(1)}%` : "not reported"
}

/** True only when verification flagged it. Old rows are never filtered. */
export function isFiltered(d) {
  return Boolean(d?.suppressed ?? d?.record?.suppressed)
}

/** True when the percentage was checked against the image. */
export function isVerified(d) {
  if (typeof d?.verified === "boolean") return d.verified
  return verificationOf(d)?.status === "checked"
}

/** Plain-English reasons, for filtered and unfiltered detections alike. */
export function verificationReasons(d) {
  return d?.verification_reasons ?? verificationOf(d)?.reasons ?? []
}

/** What the percentage is made of, in the backend's own words. */
export function confidenceBasis(d) {
  return (
    d?.confidence_pct_basis ??
    d?.record?.confidence_pct_basis ??
    "detector confidence, not verified"
  )
}

/**
 * The record handed to the assistant, with the verification verdict kept and
 * the raw measurements left out. Every key of a record is written into the
 * prompt, and a block of pixel statistics would only crowd out the evidence the
 * assistant is meant to reason from.
 */
export function forAssistant(record) {
  if (!record || typeof record !== "object") return record
  const block = record.verification
  if (!block || typeof block !== "object") return record
  return {
    ...record,
    verification: {
      status: block.status,
      reasons: block.reasons ?? [],
      hard_reasons: block.hard_reasons ?? [],
      calibrated_probability: block.calibrated_probability,
      reason: block.reason,
      heuristic: block.heuristic,
    },
  }
}
