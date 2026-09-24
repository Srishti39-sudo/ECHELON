/**
 * The only module in GhostTrace that talks to the backend.
 *
 * Errors carry a status: 0 means the request never arrived (backend not
 * running), 404 on the document means "not generated yet", which the panel
 * treats as a state with a button rather than as a failure.
 */

import { ghostConfig } from "./config"

const url = (path) => `${ghostConfig.apiBaseUrl}${path}`

export class GhostTraceApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = "GhostTraceApiError"
    this.status = status
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(url(path), options)
  } catch (cause) {
    const failure = new GhostTraceApiError(
      "The backend is unreachable. Start it with `uvicorn backend.app.main:app`.",
      0,
    )
    failure.cause = cause
    throw failure
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
    } catch {
      // A non-JSON error body is not worth a second failure.
    }
    throw new GhostTraceApiError(detail, response.status)
  }
  return response.json()
}

/** ghosttrace.json for one survey, exactly as the engine wrote it. */
export function fetchGhostTrace(surveyId) {
  return request(ghostConfig.endpoints.document(surveyId))
}

/** Whether the run button may be offered and which habitat layers are bundled. */
export function fetchCapabilities() {
  return request(ghostConfig.endpoints.capabilities)
}

/** Run the engine over a processed survey. Resolves with its summary. */
export function runGhostTrace(surveyId) {
  return request(ghostConfig.endpoints.run(surveyId), { method: "POST" })
}

/**
 * One habitat layer, narrowed server-side to a bbox [minLon, minLat, maxLon, maxLat].
 * Resolves null when the server has no such layer bundled (404).
 */
export async function fetchLayer(kind, bbox) {
  const query = bbox ? `?bbox=${bbox.map((n) => n.toFixed(4)).join(",")}` : ""
  try {
    return await request(`${ghostConfig.endpoints.layer(kind)}${query}`)
  } catch (error) {
    if (error.status === 404) return null
    throw error
  }
}

export const geojsonUrl = (surveyId) => url(ghostConfig.endpoints.geojson(surveyId))
export const alertTextUrl = (surveyId, detectionId) => url(ghostConfig.endpoints.alert(surveyId, detectionId))
