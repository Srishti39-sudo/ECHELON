/**
 * The only module that talks to the backend.
 *
 * Every URL, every fetch and every piece of stream parsing lives here. A
 * component that needs data calls one of these functions; none of them knows
 * where the backend is.
 */

import { settings } from '../config/settings'
import type {
  AssistantMode,
  ChatResponse,
  DetectResult,
  DetectionRecord,
  GhostTraceContext,
  Health,
  StreamFrame,
  Turn,
} from './types'

function url(path: string): string {
  return `${settings.apiBaseUrl}${path}`
}

export interface ChatRequest {
  message: string
  history: Turn[]
  detection_record?: DetectionRecord | null
  /** A GhostTrace target. Travels as its own field: it is data, not a detection record. */
  ghosttrace_context?: GhostTraceContext | null
  /** "auto" lets the backend route survey-data questions to the Mission Copilot. */
  mode?: AssistantMode
  /** Answer language code. Retrieval and source extracts stay English. */
  language?: string
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function failure(response: Response): Promise<ApiError> {
  let detail = response.statusText
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') detail = body.detail
    else if (Array.isArray(body?.detail) && body.detail[0]?.msg) detail = body.detail[0].msg
  } catch {
    /* a non-JSON error body is not worth a second failure */
  }
  return new ApiError(detail, response.status)
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  const response = await fetch(url(settings.endpoints.health), { signal })
  if (!response.ok) throw await failure(response)
  return response.json()
}

export async function detectTile(file: File, signal?: AbortSignal): Promise<DetectResult> {
  const form = new FormData()
  // The field is named `file` because that is what the upload route has always
  // called it; the assistant was the newcomer here.
  form.append('file', file)
  const response = await fetch(url(settings.endpoints.detect), {
    method: 'POST',
    body: form,
    signal,
  })
  if (!response.ok) throw await failure(response)
  return response.json()
}

export async function sendChat(request: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  const response = await fetch(url(settings.endpoints.chat), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) throw await failure(response)
  return response.json()
}

export interface StreamHandlers {
  onFrame: (frame: StreamFrame) => void
}

/**
 * Read the SSE route.
 *
 * Frames arrive as `data: {json}` separated by a blank line. A chunk from the
 * network can split a frame anywhere, so the tail of each read is held back
 * until its terminator shows up.
 */
export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(url(settings.endpoints.chatStream), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) throw await failure(response)
  if (!response.body) throw new ApiError('The server sent no response body.', 500)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let split = buffer.indexOf('\n\n')
    while (split !== -1) {
      const block = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      const line = block.split('\n').find((l) => l.startsWith('data: '))
      if (line) {
        try {
          handlers.onFrame(JSON.parse(line.slice(6)) as StreamFrame)
        } catch {
          /* a frame that will not parse is dropped rather than killing the stream */
        }
      }
      split = buffer.indexOf('\n\n')
    }
  }
}
