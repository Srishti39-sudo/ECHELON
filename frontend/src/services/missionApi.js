/**
 * The survey mission console's line to the backend.
 *
 * Same rule as services/api.js and survey/api.js: the page never builds a URL
 * itself. Everything the mission page needs -- starting a job, following it,
 * and fetching what it produced -- goes through here.
 *
 * The upload uses XMLHttpRequest rather than fetch for one reason: fetch
 * cannot report upload progress, and a raw sonar log can be hundreds of
 * megabytes. A progress bar that sits at zero while the browser is in fact
 * sending the file would be the one dishonest progress bar on the page.
 */

import { settings } from "../assistant/config/settings"

const BASE = settings.apiBaseUrl

export const ACCEPTED_STRIP_EXTENSIONS = [".xtf", ".jsf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
export const RAW_SONAR_EXTENSIONS = [".xtf", ".jsf"]
export const MAX_UPLOAD_BYTES = 512 * 1024 * 1024

export class MissionApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = "MissionApiError"
    this.status = status
  }
}

const OFFLINE_MESSAGE =
  "The backend is unreachable. Start it with `uvicorn backend.app.main:app`."

/** Status 0 means the request never arrived. */
export function isOffline(error) {
  return error instanceof MissionApiError && error.status === 0
}

function detailOf(status, statusText, body) {
  if (status === 404 && (!body || body.detail === "Not Found")) {
    return (
      "Survey jobs are not enabled on this backend. They run only where a " +
      "detector can really run (torch, ultralytics and a checkpoint), or when " +
      "DEEPECHO_ENABLE_SURVEY_JOBS=1 is set."
    )
  }
  const detail = body?.detail
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) return detail.map((d) => d.msg ?? String(d)).join("; ")
  return `${status} ${statusText}`
}

/**
 * Upload the survey and start the job.
 *
 * Resolves with the created job ({ job_id, events_url, ... }). Rejects with a
 * MissionApiError carrying the server's own detail, so a rejected upload tells
 * the operator exactly what was wrong with it.
 */
export function createJob({ files, nav, title, corners, onProgress }) {
  const form = new FormData()
  for (const file of files) form.append("files", file, file.name)
  if (nav) form.append("nav", nav, nav.name)
  if (title) form.append("title", title)
  if (corners) form.append("corners", JSON.stringify(corners))

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open("POST", `${BASE}/survey/jobs`)
    xhr.responseType = "json"

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total)
    }
    xhr.onerror = () => reject(new MissionApiError(OFFLINE_MESSAGE, 0))
    xhr.onload = () => {
      const body = xhr.response
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new MissionApiError(detailOf(xhr.status, xhr.statusText, body), xhr.status))
    }
    xhr.send(form)
  })
}

/**
 * Which pipeline the next upload will run, and why, before anything is sent.
 *
 * Resolves with null when the backend does not have survey jobs (a 404): the
 * page already says so from /health, and a missing capabilities answer is not
 * a second error worth showing.
 */
export async function fetchCapabilities() {
  let response
  try {
    response = await fetch(`${BASE}/survey/jobs/capabilities`)
  } catch {
    throw new MissionApiError(OFFLINE_MESSAGE, 0)
  }
  if (response.status === 404) return null
  let body = null
  try {
    body = await response.json()
  } catch {
    // handled below
  }
  if (!response.ok) {
    throw new MissionApiError(detailOf(response.status, response.statusText, body), response.status)
  }
  return body
}

/** The operator-facing name of a pipeline, e.g. "Pipeline: team detector (best.pt)". */
export function pipelineTitle(label, pipeline) {
  if (label) return `Pipeline: ${label}`
  if (pipeline === "teammate") return "Pipeline: team detector"
  if (pipeline === "echelon") return "Pipeline: DeepEcho engine"
  return "Pipeline: not reported"
}

/** Where a job is, without opening a stream. */
export async function fetchJobStatus(jobId) {
  let response
  try {
    response = await fetch(`${BASE}/survey/jobs/${encodeURIComponent(jobId)}`)
  } catch {
    throw new MissionApiError(OFFLINE_MESSAGE, 0)
  }
  let body = null
  try {
    body = await response.json()
  } catch {
    // A non-JSON body is not worth a second failure.
  }
  if (!response.ok) {
    throw new MissionApiError(
      response.status === 404 && body?.detail && body.detail !== "Not Found"
        ? body.detail
        : detailOf(response.status, response.statusText, body),
      response.status,
    )
  }
  return body
}

/** The Server-Sent Events URL, resuming after sequence number `after`. */
export function jobEventsUrl(jobId, after = 0) {
  const query = after > 0 ? `?after=${after}` : ""
  return `${BASE}/survey/jobs/${encodeURIComponent(jobId)}/events${query}`
}

/** A path the backend put in an event (a strip preview), made absolute. */
export function backendUrl(path) {
  return `${BASE}${path}`
}

/**
 * What each downloadable file is, in the operator's words. The backend decides
 * which of these exist for a given survey; this only labels them.
 */
export const DOWNLOAD_LABELS = {
  "report.csv": {
    label: "Anomaly report (CSV)",
    note: "One row per hazard: position, dimensions, class, confidence",
  },
  "report.geojson": {
    label: "Anomaly report (GeoJSON)",
    note: "The same hazards, for GIS tools",
  },
  "export.json": {
    label: "Full export (JSON)",
    note: "Every detection, hotspot and the provenance behind them",
  },
  "actions.csv": {
    label: "Action list (CSV)",
    note: "Hotspots in priority order with recommended actions",
  },
  "map.html": {
    label: "Offline map (HTML)",
    note: "Self-contained, opens with no server or network",
  },
  "ghosttrace.json": {
    label: "GhostTrace rescue analysis (JSON)",
    note: "Per net: activity, habitat, drift, safety, priority, alert draft",
  },
  "ghosttrace.geojson": {
    label: "GhostTrace (GeoJSON)",
    note: "Targets, drift cones and the recovery route, for GIS tools",
  },
}

// Every file a job lists, GhostTrace's included, downloads from the survey
// router under its own name.
export function downloadUrl(surveyId, name) {
  return `${BASE}/survey/${encodeURIComponent(surveyId)}/${name}`
}
