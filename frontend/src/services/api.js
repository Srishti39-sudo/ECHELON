/**
 * The one module the dashboard pages use to talk to the backend.
 *
 * This file was empty, so every page that needed data either went without or
 * called fetch itself. Same rule the other two features already follow:
 * src/assistant/lib/api.ts and src/survey/api.js each own their own endpoints
 * and nothing outside them knows a URL. This owns the rest.
 *
 * Errors carry a status and a readable message, because "the backend is not
 * running" and "the backend said no" are different problems and an operator
 * should be told which one they have.
 */

import { settings } from "../assistant/config/settings"

const BASE = settings.apiBaseUrl

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

/** Status 0 is reserved for "the request never arrived". */
export function isOffline(error) {
  return error instanceof ApiError && error.status === 0
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, options)
  } catch (cause) {
    const failure = new ApiError(
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
      if (body?.detail) detail = body.detail
    } catch {
      // A non-JSON error body is not worth a second failure.
    }
    throw new ApiError(detail, response.status)
  }

  return response.status === 204 ? null : response.json()
}

// --- Health ----------------------------------------------------------------

/** What is loaded, which provider answers, and where scans are being kept. */
export function fetchHealth() {
  return request(settings.endpoints.health)
}

// --- Scans and history -----------------------------------------------------

/** Every scan, newest first, each carrying its own detection counts. */
export async function fetchHistory(limit) {
  const query = limit ? `?limit=${encodeURIComponent(limit)}` : ""
  const body = await request(`/history${query}`)
  return body.scans ?? []
}

/** One scan with every detection that came out of it. */
export function fetchScan(scanId) {
  return request(`/history/${encodeURIComponent(scanId)}`)
}

export function deleteScan(scanId) {
  return request(`/history/${encodeURIComponent(scanId)}`, { method: "DELETE" })
}

/** The tile a scan was run on. A URL, not a fetch: this goes in an <img src>. */
export function scanImageUrl(scanId) {
  return `${BASE}/scans/${encodeURIComponent(scanId)}/image`
}

// --- Detections ------------------------------------------------------------

/**
 * Every detection across every scan.
 *
 * The filters are sent to the backend rather than applied here, so a long
 * history is narrowed before it crosses the wire.
 */
export async function fetchDetections({ severity, anomalyOnly, limit } = {}) {
  const params = new URLSearchParams()
  if (severity?.length) params.set("severity", severity.join(","))
  if (anomalyOnly) params.set("anomaly_only", "true")
  if (limit) params.set("limit", String(limit))

  const query = params.toString()
  const body = await request(`/detections${query ? `?${query}` : ""}`)
  return body.detections ?? []
}

// --- Statistics ------------------------------------------------------------

export function fetchStats() {
  return request("/stats")
}

/** Uploaded scans that have a position, ranked worst severity first. */
export async function fetchHazardMap() {
  return request("/hazard/map")
}

// --- Upload ----------------------------------------------------------------

/**
 * Send one tile through the detector.
 *
 * Position is optional and is passed through untouched. A tile carries no
 * position of its own, so leaving it out means the row stores null rather than
 * a guess, and the hazard map will say the scan exists but cannot be placed.
 */
export function uploadTile(file, { latitude, longitude } = {}) {
  const form = new FormData()
  form.append("file", file)
  if (latitude !== undefined && latitude !== null && latitude !== "")
    form.append("latitude", String(latitude))
  if (longitude !== undefined && longitude !== null && longitude !== "")
    form.append("longitude", String(longitude))

  return request(settings.endpoints.detect, { method: "POST", body: form })
}

// --- Surveys ---------------------------------------------------------------
// Processed surveys read off disk. The hazard map page has its own client for
// these; this is here so the dashboard pages can reach them without importing
// across features.

export async function fetchSurveys() {
  const body = await request("/survey")
  return body.surveys ?? []
}

export function fetchSurveyExport(surveyId) {
  return request(`/survey/${encodeURIComponent(surveyId)}/export`)
}
