/**
 * Formatting and small read-only derivations over ghosttrace.json.
 *
 * Nothing here recomputes a score, a probability or a tier. It turns numbers
 * the engine wrote into words, picks which of several engine values to show
 * first, and computes map extents. Anything absent stays absent: a missing
 * number renders as an em dash, never as zero.
 */

import { kindAliases, layerByKind, preciseQualities } from "./config"

export const DASH = "—"

export const isNum = (value) => typeof value === "number" && Number.isFinite(value)

/** A section object that says it is unavailable, e.g. {"available": false, "reason"}. */
export function isUnavailable(section) {
  return !section || section.available === false
}

export function reasonOf(section, fallback = "Not provided in ghosttrace.json.") {
  if (!section) return fallback
  return section.reason || section.basis || fallback
}

export function pct(probability, digits = 0) {
  if (!isNum(probability)) return DASH
  const value = probability * 100
  if (value > 0 && value < 1 && digits === 0) return "<1%"
  return `${value.toFixed(digits)}%`
}

export function confidence(value) {
  return isNum(value) ? `${Math.round(value)}%` : DASH
}

export function num(value, digits = 1, unit = "") {
  if (!isNum(value)) return DASH
  return `${Number(value.toFixed(digits)).toLocaleString()}${unit ? ` ${unit}` : ""}`
}

export function distance(meters) {
  if (!isNum(meters)) return DASH
  if (meters === 0) return "inside"
  if (meters < 1000) return `${Math.round(meters)} m`
  return `${(meters / 1000).toFixed(meters < 10000 ? 1 : 0)} km`
}

/** 36 → "~36 h"; 216 → "~9 days"; 84 → "~3.5 days". */
export function arrival(hours) {
  if (!isNum(hours)) return DASH
  if (hours < 48) return `~${Math.round(hours)} h`
  const days = hours / 24
  const rounded = days < 10 ? Math.round(days * 2) / 2 : Math.round(days)
  return `~${rounded} days`
}

export function hoursLabel(hours) {
  if (!isNum(hours)) return DASH
  if (hours < 48) return `${Math.round(hours)} h`
  const days = hours / 24
  return `${Math.round(hours)} h (${Number(days.toFixed(1))} d)`
}

const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
export function compass(bearing) {
  if (!isNum(bearing)) return ""
  return COMPASS[Math.round((((bearing % 360) + 360) % 360) / 22.5) % 16]
}

export function size(dimensions) {
  if (!dimensions) return "size not estimated"
  const { length_m: l, width_m: w } = dimensions
  if (isNum(l) && isNum(w)) return `${num(l, 1)} × ${num(w, 1)} m`
  if (isNum(l)) return `${num(l, 1)} m long`
  return "size not estimated"
}

export function titleCase(text) {
  if (!text) return ""
  const s = String(text).replace(/_/g, " ")
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function dateTime(iso) {
  if (!iso) return DASH
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso)
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
}

/** A geometry whose quality is not explicitly precise is drawn dashed and labelled. */
export function isApproximate(quality) {
  if (!quality) return true
  const q = String(quality).toLowerCase()
  if (q.includes("approx")) return true
  return !preciseQualities.some((word) => q.includes(word))
}

export function qualityLabel(quality) {
  if (!quality) return "outline quality not stated"
  return isApproximate(quality) ? `approximate (${quality})` : String(quality)
}

/** "coral_reef", "reefs", "mpa" ... → the legend's kind ("reef", "protected_area" ...). */
export function canonicalKind(kind) {
  if (!kind) return null
  const key = String(kind).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "")
  if (layerByKind[key]) return key
  return kindAliases[key] || key
}

/** Key that matches a drift impact to a habitat feature: canonical kind + name. */
export const featureKey = (kind, name) => `${canonicalKind(kind)}::${String(name ?? "").trim().toLowerCase()}`

export const kindOf = (entry) => canonicalKind(entry?.kind || entry?.layer)

export const isSensitiveKind = (kind) => {
  const k = canonicalKind(kind)
  return layerByKind[k]?.sensitive ?? k !== "harbour"
}

/** A source may be a string or a {name, url, licence, snapshot, used_for} object. */
export function sourceName(source) {
  if (!source) return null
  if (typeof source === "string") return source
  return source.name || source.title || source.doc || null
}

export function sourceUrl(source, dataSources = []) {
  if (!source) return null
  if (typeof source === "object") return source.url || source.source_url || null
  if (/^https?:\/\//i.test(source)) return source
  return (dataSources || []).find((d) => d?.name === source)?.url || null
}

/** The habitat entry worth a card line: inside-first, then nearest sensitive. */
export function headlineHabitat(habitat) {
  if (isUnavailable(habitat)) return null
  const inside = (habitat.inside || []).filter((h) => h && typeof h === "object" && isSensitiveKind(h.kind || h.layer))
  if (inside.length) return { ...inside[0], inside: true }
  const nearest = (habitat.nearest || [])
    .filter((h) => isSensitiveKind(h.kind || h.layer) && isNum(h.distance_m))
    .sort((a, b) => a.distance_m - b.distance_m)
  return nearest[0] ? { ...nearest[0], inside: nearest[0].distance_m === 0 } : null
}

/** Highest-probability impact on sensitive habitat, else any impact. */
export function topImpact(drift) {
  if (isUnavailable(drift)) return null
  const impacts = (drift.impacts || []).filter((i) => isNum(i.probability) && i.probability > 0)
  const sensitive = impacts.filter((i) => isSensitiveKind(i.kind))
  const pool = sensitive.length ? sensitive : impacts
  return pool.slice().sort((a, b) => b.probability - a.probability)[0] || null
}

export function rankOf(target) {
  return isNum(target?.priority?.rank) ? target.priority.rank : Number.POSITIVE_INFINITY
}

export function sortByRank(targets) {
  return targets.slice().sort((a, b) => {
    if (Boolean(a.suppressed) !== Boolean(b.suppressed)) return a.suppressed ? 1 : -1
    const byRank = rankOf(a) - rankOf(b)
    if (byRank) return byRank
    return (b.priority?.score ?? 0) - (a.priority?.score ?? 0)
  })
}

const snapshotCache = new WeakMap()

const pairOk = (p) => Array.isArray(p) && isNum(p[0]) && isNum(p[1])

/**
 * Whether this drift stores points as [lon, lat] (GeoJSON order) rather than
 * [lat, lon]. The contract does not fix the order and producers differ, so
 * the reading that puts the earliest point nearer the target wins, exactly as
 * the engine does for cones. With no target position, GeoJSON order is assumed
 * only when the first value cannot be a latitude.
 */
function pointsAreLonLat(snapshots, target) {
  const first = snapshots.flatMap((s) => s.points || []).find(pairOk)
    || snapshots.map((s) => (Array.isArray(s.cone50) ? s.cone50[0] : null)).find(pairOk)
  if (!first) return false
  const [a, b] = first
  if (Math.abs(a) > 90) return true
  if (Math.abs(b) > 90) return false
  const lat = target?.latitude
  const lon = target?.longitude
  if (!isNum(lat) || !isNum(lon)) return false
  const asLatLon = Math.abs(a - lat) + Math.abs(b - lon)
  const asLonLat = Math.abs(b - lat) + Math.abs(a - lon)
  return asLonLat < asLatLon
}

/** A drift cone as a GeoJSON geometry, whatever shape it arrived in. */
function coneGeometry(cone, lonLat) {
  if (!cone) return null
  if (typeof cone === "object" && !Array.isArray(cone)) {
    if ((cone.type === "Polygon" || cone.type === "MultiPolygon") && cone.coordinates) return cone
    if (cone.type === "Feature" && cone.geometry) return coneGeometry(cone.geometry, lonLat)
    for (const key of ["geometry", "polygon", "coordinates", "ring", "points"]) {
      if (cone[key]) return coneGeometry(cone[key], lonLat)
    }
    return null
  }
  if (!Array.isArray(cone)) return null
  const ring = cone.filter(pairOk).map((p) => (lonLat ? [p[0], p[1]] : [p[1], p[0]]))
  if (ring.length < 3) return null
  const [f, l] = [ring[0], ring[ring.length - 1]]
  if (f[0] !== l[0] || f[1] !== l[1]) ring.push([...f])
  return { type: "Polygon", coordinates: [ring] }
}

/**
 * Snapshots in time order with points as [lat, lon] and cones as GeoJSON,
 * whatever order and shape the file used. Cached per drift object.
 */
export function snapshotsOf(drift, target) {
  if (isUnavailable(drift) || !Array.isArray(drift.snapshots)) return []
  const cached = snapshotCache.get(drift)
  if (cached) return cached
  const sorted = drift.snapshots.slice().sort((a, b) => (a.t_hours ?? 0) - (b.t_hours ?? 0))
  const lonLat = pointsAreLonLat(sorted, target)
  const out = sorted.map((snap) => ({
    t_hours: snap.t_hours,
    points: (snap.points || []).filter(pairOk).map((p) => (lonLat ? [p[1], p[0]] : [p[0], p[1]])),
    cone50: coneGeometry(snap.cone50, lonLat),
    cone90: coneGeometry(snap.cone90, lonLat),
  }))
  snapshotCache.set(drift, out)
  return out
}

export function latLngOf(entity) {
  if (!entity) return null
  const lat = entity.latitude ?? entity.lat
  const lon = entity.longitude ?? entity.lon ?? entity.lng
  return isNum(lat) && isNum(lon) ? [lat, lon] : null
}

export function previousLatLng(change) {
  if (!change) return null
  const lat = change.previous_latitude ?? change.previous_position?.latitude
  const lon = change.previous_longitude ?? change.previous_position?.longitude
  return isNum(lat) && isNum(lon) ? [lat, lon] : null
}

/** Move a lat/lng by distance_m along bearing_deg (flat earth; fine under ~50 km). */
export function offsetLatLng(ll, distanceM, bearingDeg) {
  if (!ll || !isNum(distanceM) || !isNum(bearingDeg)) return null
  const rad = (bearingDeg * Math.PI) / 180
  const dLat = (distanceM * Math.cos(rad)) / 111_320
  const dLon = (distanceM * Math.sin(rad)) / (111_320 * Math.cos((ll[0] * Math.PI) / 180))
  return [ll[0] + dLat, ll[1] + dLon]
}

/** Nearest sensitive habitat within reach of each target, as the map point the
 *  engine measured to. Nets sit in open water more often than not, and a view
 *  fitted to the nets alone is a rectangle of flat blue; the nearest reef or
 *  park edge is what tells a reader where they are. */
const HABITAT_CONTEXT_MAX_M = 15_000
function habitatContextPoints(target) {
  const here = latLngOf(target)
  const nearest = target?.habitat?.nearest || []
  return nearest
    .filter((h) => h && h.kind !== "harbour" && isNum(h.distance_m) && h.distance_m <= HABITAT_CONTEXT_MAX_M)
    .map((h) => offsetLatLng(here, h.distance_m, h.bearing_deg))
    .filter(Boolean)
}

/**
 * [minLon, minLat, maxLon, maxLat] over targets (and their previous positions),
 * removed contacts, drift, the nearest sensitive habitat of each target when
 * it is within 15 km and, unless excluded, the recovery route start. The
 * map fits without the route start: a harbour tens of km away would shrink a
 * survey line, and a 40 m move, to a few pixels.
 */
export function extentOf(doc, { includeDrift = true, includeRouteStart = true, includeHabitat = true } = {}) {
  let minLat = Infinity
  let minLon = Infinity
  let maxLat = -Infinity
  let maxLon = -Infinity
  const add = (ll) => {
    if (!ll) return
    minLat = Math.min(minLat, ll[0])
    maxLat = Math.max(maxLat, ll[0])
    minLon = Math.min(minLon, ll[1])
    maxLon = Math.max(maxLon, ll[1])
  }
  for (const target of doc?.targets || []) {
    add(latLngOf(target))
    add(previousLatLng(target.change))
    if (includeHabitat) for (const point of habitatContextPoints(target)) add(point)
    if (includeDrift) {
      for (const snap of snapshotsOf(target.drift, target)) {
        for (const point of snap.points) add(point)
      }
    }
  }
  for (const removed of doc?.removed_since_previous || []) add(latLngOf(removed))
  if (includeRouteStart) add(latLngOf(doc?.recovery_plan?.start))
  if (!Number.isFinite(minLat)) return null
  return [minLon, minLat, maxLon, maxLat]
}

export function padExtent(extent, degrees) {
  if (!extent) return null
  return [
    Math.max(-180, extent[0] - degrees),
    Math.max(-90, extent[1] - degrees),
    Math.min(180, extent[2] + degrees),
    Math.min(90, extent[3] + degrees),
  ]
}

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c])
}

/** Resolve a recovery-plan stop (a detection id string or an object) to its target. */
export function stopId(stop) {
  if (stop == null) return null
  if (typeof stop === "string" || typeof stop === "number") return String(stop)
  return stop.detection_id != null ? String(stop.detection_id) : stop.id != null ? String(stop.id) : null
}

/** The same text the backend's .txt route serves, for the offline fixture preview. */
export function alertPlainText(surveyId, target) {
  const alert = target.alert || {}
  const lines = [
    "DRAFT - verify every detail before sending. Not an official notice.",
    `Generated by: ${alert.generated_by || "not stated"}`,
    `Survey: ${surveyId}   Detection: ${target.detection_id}`,
  ]
  const authorities = (alert.authorities || []).filter(Boolean)
  if (authorities.length) {
    lines.push(`To: ${authorities.map((a) => `${a.name}${a.role ? ` (${a.role})` : ""}`).join("; ")}`)
  }
  if (alert.subject) lines.push(`Subject: ${alert.subject}`)
  lines.push("", String(alert.draft_text || "").trimEnd(), "")
  if (alert.citations?.length) {
    lines.push("Sources:", ...alert.citations.map((c) => `  - ${citationText(c)}`))
  }
  if (alert.basis) lines.push(`Basis: ${alert.basis}`)
  return `${lines.join("\n")}\n`
}

export function citationText(citation) {
  if (!citation) return ""
  if (typeof citation === "string") return citation
  if (citation.assistant_source) return `assistant source: ${citationText(citation.assistant_source)}`
  const parts = [citation.doc || citation.doc_id, citation.section, citation.title && citation.title !== citation.section ? citation.title : null]
    .filter(Boolean)
  return parts.length ? parts.join(" — ") : JSON.stringify(citation)
}
