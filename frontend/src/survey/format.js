/**
 * Small, pure formatting and geometry helpers for the replay, the coverage
 * panel and the printable report. Nothing here fetches or remembers anything.
 */

function finite(value) {
  return typeof value === "number" && Number.isFinite(value)
}

/** Degrees, minutes, seconds with hemisphere: 9°07′11.32″N. */
export function dms(value, positive, negative) {
  if (!finite(value)) return "—"
  const hemisphere = value >= 0 ? positive : negative
  let rest = Math.abs(value)
  let degrees = Math.floor(rest)
  rest = (rest - degrees) * 60
  let minutes = Math.floor(rest)
  let seconds = (rest - minutes) * 60
  // 59.995 rounds to 60.00; carry it rather than print it.
  if (Number(seconds.toFixed(2)) >= 60) {
    seconds = 0
    minutes += 1
  }
  if (minutes >= 60) {
    minutes = 0
    degrees += 1
  }
  return `${degrees}°${String(minutes).padStart(2, "0")}′${seconds.toFixed(2).padStart(5, "0")}″${hemisphere}`
}

export function latDms(lat) {
  return dms(lat, "N", "S")
}

export function lonDms(lon) {
  return dms(lon, "E", "W")
}

/** A small area in km², with enough decimals to be non-zero. */
export function km2(value) {
  if (!finite(value)) return "—"
  if (value === 0) return "0 km²"
  const digits = value >= 10 ? 1 : value >= 1 ? 2 : value >= 0.1 ? 3 : 4
  return `${value.toFixed(digits)} km²`
}

export function m2(valueKm2) {
  if (!finite(valueKm2)) return "—"
  return `${Math.round(valueKm2 * 1e6).toLocaleString("en-GB")} m²`
}

export function km(value) {
  if (!finite(value)) return "—"
  return `${value.toFixed(value >= 10 ? 1 : 3)} km`
}

export function clock(seconds) {
  if (!finite(seconds)) return "--:--"
  const whole = Math.max(0, Math.floor(seconds))
  const m = Math.floor(whole / 60)
  const s = whole % 60
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
}

export function pct(value, digits = 1) {
  return finite(value) ? `${value.toFixed(digits)}%` : "—"
}

export function titleCase(text) {
  return String(text ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

const EARTH_A = 6378137.0
const EARTH_E2 = 6.69437999014e-3

/**
 * Move a position by east/north metres on a local flat earth (WGS84 radii).
 * The same approximation hazard_geo.offset_latlon uses, at the same scale.
 */
export function offsetLatLon(lat, lon, east, north) {
  const s = Math.sin((lat * Math.PI) / 180)
  const w = 1 - EARTH_E2 * s * s
  const meridional = (EARTH_A * (1 - EARTH_E2)) / w ** 1.5
  const prime = EARTH_A / Math.sqrt(w)
  const dLat = (north / meridional) * (180 / Math.PI)
  const dLon = (east / (prime * Math.max(Math.cos((lat * Math.PI) / 180), 1e-12))) * (180 / Math.PI)
  return [lat + dLat, lon + dLon]
}

/** [port, starboard] swath edge positions for a towfish position and heading. */
export function swathEdges(lat, lon, headingDeg, portM, stbdM) {
  const h = (headingDeg * Math.PI) / 180
  const sx = Math.cos(h)
  const sy = -Math.sin(h)
  return [offsetLatLon(lat, lon, -portM * sx, -portM * sy), offsetLatLon(lat, lon, stbdM * sx, stbdM * sy)]
}

/** Index of the last element of a sorted array that is <= value, or -1. */
export function lastIndexAtOrBefore(sorted, value) {
  let lo = 0
  let hi = sorted.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] <= value) {
      found = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return found
}

export function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

/** Leaflet bounds [[south, west], [north, east]] around a list of [lat, lon], or null. */
export function boundsOf(points) {
  let south = Infinity
  let west = Infinity
  let north = -Infinity
  let east = -Infinity
  for (const [lat, lon] of points) {
    if (!finite(lat) || !finite(lon)) continue
    south = Math.min(south, lat)
    north = Math.max(north, lat)
    west = Math.min(west, lon)
    east = Math.max(east, lon)
  }
  if (!Number.isFinite(south)) return null
  return [
    [south, west],
    [north, east],
  ]
}

/** Every [lat, lon] in a GeoJSON geometry, for bounds. */
export function geometryLatLngs(geometry) {
  const out = []
  const walk = (coords) => {
    if (!Array.isArray(coords)) return
    if (coords.length >= 2 && typeof coords[0] === "number") {
      out.push([coords[1], coords[0]])
      return
    }
    coords.forEach(walk)
  }
  walk(geometry?.coordinates)
  return out
}

/** Bounds of a coverage document: every polygon and re-look line in it. */
export function coverageBounds(coverage) {
  const features = [
    ...(coverage?.polygons?.features || []),
    ...(coverage?.relook_lines?.features || []),
  ]
  return boundsOf(features.flatMap((f) => geometryLatLngs(f.geometry)))
}
