/** The backend contract, mirrored. Field names match the API exactly. */

export type Role = 'user' | 'assistant'
export type Intent = 'question' | 'explain' | 'anomaly' | 'report'
export type Severity = 'low' | 'medium' | 'high' | 'unknown'

export interface Turn {
  role: Role
  content: string
}

export interface DetectionRecord {
  object_class?: string | null
  label?: string | null
  confidence?: number | null
  bbox?: number[] | null
  depth_m?: number | null
  timestamp?: string | null
  latitude?: string | null
  longitude?: string | null
  sensor?: string | null
  platform?: string | null
  notes?: string | null
  visual_description?: string | null
  embedding?: number[] | null
  // Provenance added by the detector. Which checkpoint made the call, what it
  // called the object before the class map translated it, and the other
  // model's opinion where the two disagreed on the same box.
  detector_model?: string | null
  detector_class?: string | null
  second_opinion?: string | null
  /** Set when the detector's class was below the floor this system requires
   *  for that class. The contact is kept, the claim is not. */
  downgraded_from?: string | null
}

export interface Source {
  n: number
  id: string
  title: string
  section?: string | null
  snippet: string
  authority?: string | null
  status?: string | null
  doc_id?: string | null
  path?: string | null
  score: number
  pdf_url?: string | null
}

export interface Match {
  rank: number
  id: string
  name: string
  object_class: string
  hazard: string
  similarity: number
  confirms: string
  rules_out: string
  source: string
  status: string
}

/** Answer path: the corpus alone, or survey data tools plus the corpus. */
export type AssistantMode = 'auto' | 'copilot' | 'reference'

/** One read-only data lookup the Mission Copilot made. */
export interface ToolCall {
  name: string
  args: Record<string, unknown>
  /** "Looked up GhostTrace targets across 2 surveys (4 targets)" */
  summary: string
  record_count: number
  /** The [Dn] numbers this call returned. */
  citations: number[]
  error?: string | null
  planned_by: string
}

/** A survey record cited as [Dn]. Survey data, never a reference source. */
export interface DataCitation {
  n: number
  kind: string
  survey_id?: string | null
  label: string
  source_file: string
  record_id?: string | null
  synthetic?: boolean | null
  /** /ghosttrace/<id> or /map?survey=<id> */
  link?: string | null
  summary: Record<string, unknown>
}

export interface ChatResponse {
  answer: string
  intent: Intent
  object_class?: string | null
  confidence?: number | null
  is_anomaly: boolean
  severity: Severity
  grounded: boolean
  sources: Source[]
  refusal: boolean
  /** The detector named a class the corpus has no document about. */
  coverage_gap: boolean
  matches: Match[]
  query: string
  provider: string
  model: string
  /**
   * Who wrote the answer. `retrieval_only` means every provider failed and the
   * answer is the retrieved passages quoted, with nothing generated. `none`
   * means retrieval found nothing and no model was asked.
   */
  generated_by?: 'model' | 'retrieval_only' | 'data_only' | 'none'
  /** Why no provider answered, one line each. Set on a retrieval-only answer. */
  provider_errors?: string[]
  /** Figures in the answer found in no source, message, record or context. */
  unsourced_numbers?: string[]
  /** "GhostTrace output for <survey>/<detection>", when a GhostTrace context rode along. */
  ghosttrace_citation?: string | null
  mode?: 'copilot' | 'reference'
  /** Why the turn took that path. */
  route_reason?: string
  language?: string
  /** The English text the corpus was searched with, for a non-English question. */
  query_translated?: string | null
  planned_by?: string | null
  tool_calls?: ToolCall[]
  data_citations?: DataCitation[]
}

export interface Health {
  status: 'ready' | 'degraded'
  corpus_loaded: boolean
  documents: number
  chunks: number
  embedder: string
  index: string
  catalog_entries: number
  catalog_space?: string | null
  provider: string
  model: string
  detector: 'stub' | 'loaded' | 'disabled'
  detector_models: string[]
  upload_enabled: boolean
  /** "connected", or the reason storage is unavailable. */
  storage?: string
}

export interface DetectResult {
  stub: boolean
  models: string[]
  filename: string
  bytes: number
  detections: DetectionRecord[]
  /** False when Supabase is unconfigured. The detection still ran. */
  stored?: boolean
  scan_id?: string
  image_url?: string | null
}

/** Frames on the streaming route. */
export type StreamFrame =
  | ({ type: 'meta'; matches: Match[] } & Omit<ChatResponse, 'answer' | 'sources' | 'grounded' | 'refusal' | 'matches'>)
  | { type: 'sources'; sources: Source[] }
  | { type: 'tools'; tool_calls: ToolCall[]; data_citations: DataCitation[]; planned_by?: string | null }
  | { type: 'delta'; text: string }
  | ({ type: 'done' } & ChatResponse)
  | { type: 'error'; detail: string }

/**
 * A hotspot handed over from the survey hazard map.
 *
 * The map owns where and how urgent; the assistant owns what and what to do.
 * So `severity` and `severity_tier` here are the map's numbers and are
 * displayed as given. The assistant does not recompute them, because the same
 * hotspot showing two different urgencies is worse than showing one.
 */
export interface SurveyContext {
  hotspot_id: string
  dominant_class: string
  /** max_severity across the cell. Always 0 to 1, so it maps onto the tiers. */
  severity: number
  confidence: number
  centroid: { global_x: number; global_y: number }
  lat: number | null
  lon: number | null
  recommended_action: string
  severity_tier?: 'critical' | 'medium' | 'low'
  priority_rank?: number
  detection_count?: number
  /** Summed over the grid cell and CAN EXCEED 1. Never render as a severity. */
  total_severity?: number
  coordinate_mode?: string
  survey_id?: string
  demo?: boolean
  evidence_tile?: string
}

export interface GhostTraceImpact {
  name?: string | null
  kind?: string | null
  probability?: number | null
  first_arrival_hours?: number | null
}

/**
 * A GhostTrace target handed over from the rescue queue.
 *
 * Survey DATA, not a source. The assistant quotes its numbers attributed to
 * GhostTrace and never as a corpus fact, and its priority is displayed as
 * given, the same way the hazard map's severity is. Null means unknown.
 */
export interface GhostTraceContext {
  kind: 'ghosttrace_target'
  survey_id: string | null
  survey_title?: string | null
  synthetic: boolean | null
  detection_id: string | null
  object_class: string | null
  latitude: number | null
  longitude: number | null
  confidence_pct: number | null
  priority: {
    score: number | null
    tier: string | null
    rank: number | null
    formula?: string | null
    terms?: Record<
      string,
      { value: number | null; weight: number | null; contribution: number | null } | null
    > | null
  } | null
  activity: {
    level: string | null
    score: number | null
    enrichment_ratio?: number | null
    echo_clusters_near?: number | null
    background_clusters_per_window?: number | null
    limitations?: string | null
  } | null
  habitat_nearest: Array<{
    name: string | null
    kind: string | null
    distance_m: number | null
    source?: string | Record<string, unknown> | null
  }> | null
  drift: {
    mode: string | null
    top_impact: GhostTraceImpact | null
    stranding_probability: number | null
  } | null
  refloat_scenario?: { top_impact: GhostTraceImpact | null } | null
  people: {
    propeller_hazard_level: string | null
    diver_recommended_method?: string | null
    seabed_depth_m?: number | null
    current_mps_at_depth?: number | null
  } | null
  change: { status: string | null; moved_m: number | null } | null
  authorities: Array<{ name: string | null; role: string | null; situation: string | null }> | null
  caveats: string[] | null
}

/** One rendered message. Assistant messages carry the answer metadata with them. */
export interface Message {
  id: string
  role: Role
  content: string
  streaming?: boolean
  failed?: string | null
  meta?: Partial<ChatResponse> | null
  detection?: DetectionRecord | null
  detectionIsStub?: boolean
  /** Set when this turn came from the hazard map. Its severity wins. */
  survey?: SurveyContext | null
  /** Set when this turn carried a GhostTrace target. Its priority is shown as given. */
  ghosttrace?: GhostTraceContext | null
}
