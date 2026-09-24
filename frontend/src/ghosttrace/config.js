/**
 * Where the GhostTrace API is, and every label, colour and word this feature uses.
 *
 * Same rule as src/survey/config.js: nothing else in this directory knows a URL
 * or hard-codes a phrase the operator reads. The engine decides tiers, levels
 * and statuses; this file only decides how they look. A value the engine adds
 * later that is not listed here renders with the neutral fallback rather than a
 * guessed colour, because choosing a colour for an unknown risk is a claim.
 */

const fromEnv = import.meta.env.VITE_API_BASE_URL

export const ghostConfig = {
  apiBaseUrl: (fromEnv && fromEnv.trim()) || "http://127.0.0.1:8000",
  endpoints: {
    capabilities: "/ghosttrace/capabilities",
    document: (id) => `/ghosttrace/${encodeURIComponent(id)}`,
    geojson: (id) => `/ghosttrace/${encodeURIComponent(id)}/geojson`,
    run: (id) => `/ghosttrace/${encodeURIComponent(id)}/run`,
    alert: (id, detectionId) =>
      `/ghosttrace/${encodeURIComponent(id)}/alerts/${encodeURIComponent(detectionId)}.txt`,
    layer: (kind) => `/ghosttrace/layers/${encodeURIComponent(kind)}`,
  },
  /** Degrees added around the survey's extent when asking for habitat layers. */
  layerPaddingDeg: 0.35,
  /** Milliseconds per drift snapshot during playback. */
  playbackStepMs: 650,
  /** Most particles drawn at once; more are sampled evenly and the UI says so. */
  maxParticlesDrawn: 2500,
  basemap: {
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
  },
}

// Priority tiers. `shape` is the marker outline, so tier is never colour alone.
export const tierStyle = {
  urgent: { label: "Urgent", color: "#c81e1e", tint: "#fdecec", shape: "diamond", glyph: "!" },
  high: { label: "High", color: "#c26a05", tint: "#fdf3e6", shape: "circle", glyph: "▲" },
  routine: { label: "Routine", color: "#2f6f86", tint: "#e8f3f7", shape: "square", glyph: "•" },
}
export const tierFallback = { label: "Unranked", color: "#64748b", tint: "#eef2f6", shape: "square", glyph: "?" }
export const styleForTier = (tier) => tierStyle[tier] || tierFallback

export const activityStyle = {
  high: { label: "Actively fishing (high)", short: "High", color: "#c81e1e", tint: "#fdecec" },
  moderate: { label: "Some fish activity (moderate)", short: "Moderate", color: "#c26a05", tint: "#fdf3e6" },
  low: { label: "Little fish activity (low)", short: "Low", color: "#2f6f86", tint: "#e8f3f7" },
  unknown: { label: "Activity unknown", short: "Unknown", color: "#64748b", tint: "#eef2f6" },
}
export const styleForActivity = (level) => activityStyle[level] || activityStyle.unknown

export const hazardStyle = {
  high: { label: "High", color: "#c81e1e", tint: "#fdecec" },
  moderate: { label: "Moderate", color: "#c26a05", tint: "#fdf3e6" },
  low: { label: "Low", color: "#15803d", tint: "#eaf6ef" },
}
export const styleForHazard = (level) =>
  hazardStyle[level] || { label: level ? String(level) : "Unknown", color: "#64748b", tint: "#eef2f6" }

export const changeStyle = {
  new: { label: "New", color: "#7c3aed", tint: "#f3eefe" },
  moved: { label: "Moved", color: "#c26a05", tint: "#fdf3e6" },
  persistent: { label: "Persistent", color: "#2f6f86", tint: "#e8f3f7" },
  removed: { label: "Removed", color: "#475569", tint: "#eef2f6" },
  first_survey: { label: "First survey", color: "#64748b", tint: "#eef2f6" },
  unmatched_no_prior: { label: "No prior match", color: "#64748b", tint: "#eef2f6" },
}
export const styleForChange = (status) =>
  changeStyle[status] || { label: status ? String(status) : "Unknown", color: "#64748b", tint: "#eef2f6" }

/** Habitat layers, in legend order. Harbours are context, not sensitive habitat. */
export const layerKinds = [
  { kind: "reef", label: "Coral reefs", color: "#d9467a", sensitive: true },
  { kind: "protected_area", label: "Marine protected areas", color: "#15803d", sensitive: true },
  { kind: "turtle_nesting", label: "Turtle nesting beaches", color: "#a16207", sensitive: true },
  { kind: "dugong", label: "Dugong habitat", color: "#4f46e5", sensitive: true },
  { kind: "harbour", label: "Harbours", color: "#334155", sensitive: false },
]
export const layerByKind = Object.fromEntries(layerKinds.map((l) => [l.kind, l]))

/** Other spellings producers use for the same kinds (mirrors the backend's LAYER_ALIASES). */
export const kindAliases = {
  reefs: "reef", coral_reef: "reef", coral_reefs: "reef", coral: "reef",
  protected_areas: "protected_area", mpa: "protected_area", mpas: "protected_area",
  marine_protected_area: "protected_area", marine_protected_areas: "protected_area", wdpa: "protected_area",
  turtle_nesting_beach: "turtle_nesting", turtle_nesting_beaches: "turtle_nesting", turtle_nesting_sites: "turtle_nesting",
  turtles: "turtle_nesting", turtle: "turtle_nesting", nesting_beach: "turtle_nesting", nesting_beaches: "turtle_nesting",
  dugongs: "dugong", dugong_habitat: "dugong", seagrass_dugong: "dugong",
  harbours: "harbour", harbor: "harbour", harbors: "harbour", port: "harbour", ports: "harbour",
  fishing_harbour: "harbour", fishing_harbours: "harbour", landing_centre: "harbour", landing_centres: "harbour",
}
const canonical = (kind) => {
  const key = String(kind ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "")
  return layerByKind[key] ? key : kindAliases[key] || key
}
export const labelForKind = (kind) => layerByKind[canonical(kind)]?.label ?? (kind ? String(kind).replace(/_/g, " ") : "Habitat")
export const colorForKind = (kind) => layerByKind[canonical(kind)]?.color ?? "#475569"
export const singularLabel = (kind) => singularForKind[canonical(kind)] || (kind ? String(kind).replace(/_/g, " ") : "Habitat")
const singularForKind = {
  reef: "Coral reef",
  protected_area: "Protected area",
  turtle_nesting: "Turtle nesting beach",
  dugong: "Dugong habitat",
  harbour: "Harbour",
}

/** Geometry quality words that mean an outline can be drawn as solid. */
export const preciseQualities = ["exact", "surveyed", "authoritative", "official", "precise", "point"]

export const driftColor = "#6d28d9"
export const routeColor = "#163047"

/** Readable names for priority terms. Unknown terms show their key. */
export const termLabels = {
  activity: "Fish activity",
  habitat: "Habitat sensitivity",
  drift_impact: "Drift impact",
  propeller_hazard: "Propeller hazard",
  people_risk: "People at risk",
  recoverability: "Recoverability",
  confidence: "Detector confidence",
  size: "Size",
  change: "Change status",
}

export const copy = {
  title: "GhostTrace",
  subtitle:
    "For every ghost net: is it catching now, what it threatens, where it will drift, who is at risk, and what to do first.",
  explainerToggle: "How this is computed & what it can't tell you",
  runButton: "Run GhostTrace",
  rerunButton: "Re-run",
  running: "Running GhostTrace over this survey. This can take a minute.",
  notGenerated: "GhostTrace has not been run for this survey yet.",
  runDisabled:
    "Running from the interface is off on this server. Run it from the repository root:",
  noTargets:
    "GhostTrace ran and found no targets to assess in this survey. It ran; there is simply nothing to rank.",
  syntheticBadge: "Synthetic inputs",
  demoBadge: "Demo data",
  exampleBadge: "Synthetic example",
  syntheticNote:
    "Some or all inputs to this analysis are synthetic. Treat every number here as a demonstration of the method, not as evidence about the sea.",
  activityHeuristic:
    "Heuristic signal: water-column echoes (fish, but possibly bubbles, sediment or turbulence) clustered near the object. Evidence of aggregation, not proof that the net is catching.",
  priorityHeuristic:
    "A transparent weighted score chosen for this project, not an official prioritisation standard.",
  driftModel: "Model forecast — shares of simulated particles under the stated current, not calibrated odds.",
  alertDraft: "Draft — verify before sending",
  alertNotSent: "Nothing is sent from this page.",
  routeNote: "Straight-line legs, not a navigable route.",
}
