import { useCallback, useEffect, useRef, useState } from 'react'
import { copy } from '../config/copy'
import { settings } from '../config/settings'
import { defaultLanguage, languageFor, languages } from '../config/languages'
import { theme, themeCss } from '../config/theme'
import { getHealth, sendChat, streamChat } from '../lib/api'
import type { CitationTarget } from '../lib/citations'
import type {
  AssistantMode,
  DataCitation,
  DetectResult,
  DetectionRecord,
  GhostTraceContext,
  Health,
  Message,
  Source,
  SurveyContext,
  Turn,
} from '../lib/types'
import { CitationPanel } from './CitationPanel'
import { Composer } from './Composer'
import { MessageBubble } from './MessageBubble'
import { SystemStatus } from './SystemStatus'
import { UploadControl } from './UploadControl'
let counter = 0
const nextId = () => `m${++counter}`
const LANGUAGE_KEY = 'deepecho.assistant.language'
const MODES: AssistantMode[] = ['auto', 'copilot', 'reference']

/** The remembered answer language. Storage can be unavailable; that is not an error. */
function storedLanguage(): string {
  try {
    const value = window.localStorage.getItem(LANGUAGE_KEY)
    return value && languages.some((l) => l.code === value) ? value : defaultLanguage
  } catch {
    return defaultLanguage
  }
}
/**
 * The conversation.
 *
 * Holds the turns, the detection record currently attached, and the streaming
 * state. History is client-side: every request carries the turns it needs, so
 * the backend stores nothing and there is no session to lose.
 */
interface Props {
  /** A hotspot handed over from the survey hazard map, or null. */
  surveyContext?: SurveyContext | null
  /** The opening question to ask about it, sent automatically on arrival. */
  initialQuestion?: string | null
  /**
   * A stored detection handed over from the detections or history page.
   *
   * This one IS a detection record, unlike the survey context above, and the
   * difference is the point. A hotspot's severity is the map's arithmetic and
   * has to travel separately so the two views cannot disagree about it. A
   * stored detection came out of this same detector and carries no severity of
   * its own into the prompt, so it goes through the normal door and the
   * assistant looks severity up in its own table exactly as it would for a
   * fresh upload.
   */
  detectionContext?: DetectionRecord | null
  /** Identity of that detection, so the same handoff does not fire twice. */
  detectionKey?: string | null
  /**
   * Whether the run that produced it came from the stub detector.
   *
   * A property of the run, not of the record, which is why it is a separate
   * prop rather than a field on DetectionRecord. The caller reads it off the
   * scan; it is never inferred from the record's own contents.
   */
  detectionIsStub?: boolean
  /**
   * A GhostTrace target handed over from the rescue queue, or null.
   *
   * Like the survey context it travels as its own field and never as a
   * detection record: its priority is the queue's and is shown as given. Unlike
   * the survey context it stays attached for follow-up turns, because "why is
   * the habitat term 0.7?" means nothing to the backend without it.
   */
  ghosttraceContext?: GhostTraceContext | null
}

export function ChatWindow({
  surveyContext = null,
  initialQuestion = null,
  detectionContext = null,
  detectionKey = null,
  detectionIsStub: detectionFromStub = false,
  ghosttraceContext = null,
}: Props = {}) {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [detection, setDetection] = useState<DetectionRecord | null>(null)
  const [contacts, setContacts] = useState<DetectionRecord[]>([])
  const [detectionIsStub, setDetectionIsStub] = useState(false)
  const [ghosttrace, setGhosttrace] = useState<GhostTraceContext | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [panelSources, setPanelSources] = useState<Source[]>([])
  const [panelData, setPanelData] = useState<DataCitation[]>([])
  const [panelSelected, setPanelSelected] = useState<CitationTarget | null>(null)
  const [mode, setMode] = useState<AssistantMode>('auto')
  const [language, setLanguage] = useState<string>(storedLanguage)
  const abort = useRef<AbortController | null>(null)
  const handedOver = useRef<string | null>(null)
  const lastAttached = useRef<string | null>(null)
  const scroller = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let live = true
    const check = async () => {
      try {
        const result = await getHealth()
        if (!live) return
        setHealth(result)
        setHealthError(null)
      } catch (error) {
        if (!live) return
        setHealth(null)
        setHealthError(error instanceof Error ? error.message : copy.error.backendDown)
      }
    }
    check()
    const timer = setInterval(check, settings.chat.healthPollMs)
    return () => {
      live = false
      clearInterval(timer)
    }
  }, [])
  useEffect(() => {
    const node = scroller.current
    if (node) node.scrollTop = node.scrollHeight
  }, [messages])

  useEffect(() => {
    if (!surveyContext || !initialQuestion) return
    if (handedOver.current === surveyContext.hotspot_id) return
    handedOver.current = surveyContext.hotspot_id

    // A minimal record, so routing and the corpus synonyms work: without a class
    // the retrieval for "ship" misses the wreck document entirely. Severity is
    // NOT taken from it. The map's number travels on the survey context and is
    // what gets displayed, which is why the two can never disagree here.
    const record: DetectionRecord = {
      object_class: surveyContext.dominant_class,
      confidence: surveyContext.confidence,
    }
    setDetection(record)
    setDetectionIsStub(false)
    lastAttached.current = null
    void send({ text: initialQuestion, record, survey: surveyContext })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surveyContext, initialQuestion])

  useEffect(() => {
    if (!detectionContext || !initialQuestion) return
    const key = detectionKey ?? JSON.stringify(detectionContext)
    if (handedOver.current === key) return
    handedOver.current = key

    setDetection(detectionContext)
    // A stored detection is only a stub if the run that produced it was, and
    // that is recorded on the scan rather than guessed at here.
    setDetectionIsStub(detectionFromStub)
    lastAttached.current = null
    void send({ text: initialQuestion, record: detectionContext })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionContext, detectionKey, detectionFromStub, initialQuestion])
  useEffect(() => {
    if (!ghosttraceContext) return
    const key = `ghosttrace:${ghosttraceContext.survey_id ?? ''}/${ghosttraceContext.detection_id ?? ''}`
    if (handedOver.current === key) return
    handedOver.current = key

    // No detection record is attached: the backend derives the class it needs
    // for retrieval from the context itself, and the queue's priority is not
    // turned into the assistant's severity.
    setDetection(null)
    setContacts([])
    setDetectionIsStub(false)
    lastAttached.current = null
    setGhosttrace(ghosttraceContext)
    void send({
      text: initialQuestion ?? copy.ghosttrace.defaultQuestion,
      ghosttrace: ghosttraceContext,
      noRecord: true,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ghosttraceContext, initialQuestion])
  const patch = useCallback((id: string, change: Partial<Message>) => {
    setMessages((current) =>
      current.map((message) => (message.id === id ? { ...message, ...change } : message)),
    )
  }, [])
  const openCitation = useCallback(
    (target: CitationTarget, sources: Source[], data: DataCitation[]) => {
      if (sources.length === 0 && data.length === 0) return
      setPanelSources(sources)
      setPanelData(data)
      if (target.kind === 'data') {
        if (data.length === 0) return
        setPanelSelected({ kind: 'data', n: data.some((d) => d.n === target.n) ? target.n : data[0].n })
      } else {
        if (sources.length === 0) return
        setPanelSelected({
          kind: 'source',
          n: sources.some((s) => s.n === target.n) ? target.n : sources[0].n,
        })
      }
    },
    [],
  )
  const chooseLanguage = useCallback((code: string) => {
    setLanguage(code)
    try {
      window.localStorage.setItem(LANGUAGE_KEY, code)
    } catch {
      /* private mode or blocked storage: the choice still holds for this page */
    }
  }, [])
  const stop = useCallback(() => {
    abort.current?.abort()
    abort.current = null
    setBusy(false)
  }, [])
  const send = useCallback(async (override?: {
    text?: string
    record?: DetectionRecord
    survey?: SurveyContext | null
    ghosttrace?: GhostTraceContext | null
    noRecord?: boolean
  }) => {
    const text = (override?.text ?? draft).trim()
    if (!text || busy) return
    // An upload sends its own opening turn before React has committed the new
    // detection state, so the record travels with the call rather than being
    // read back from state that is one render behind.
    const record = override?.noRecord ? null : (override?.record ?? detection)
    // Same reason: an explicit null detaches the target for this very turn.
    const target =
      override && override.ghosttrace !== undefined ? override.ghosttrace : ghosttrace
    const snapshot = record ? JSON.stringify(record) : null
    const showRecord = snapshot !== null && snapshot !== lastAttached.current
    lastAttached.current = snapshot
    const history: Turn[] = messages
      .filter((message) => !message.failed || message.role === 'user')
      .slice(-settings.chat.maxHistoryTurns)
      .map((message) => ({ role: message.role, content: message.content }))
    const userMessage: Message = {
      id: nextId(),
      role: 'user',
      content: text,
      detection: showRecord ? record : null,
      detectionIsStub: showRecord ? detectionIsStub : false,
    }
    const replyId = nextId()
    const survey = override?.survey ?? null
    setMessages((current) => [
      ...current,
      userMessage,
      {
        id: replyId,
        role: 'assistant',
        content: '',
        streaming: true,
        meta: {},
        survey,
        ghosttrace: target,
      },
    ])
    setDraft('')
    setBusy(true)
    const controller = new AbortController()
    abort.current = controller
    const request = {
      message: text,
      history,
      detection_record: record,
      ghosttrace_context: target,
      mode,
      language,
    }
    try {
      if (settings.features.streaming) {
        await streamChat(
          request,
          {
            onFrame: (frame) => {
              if (frame.type === 'delta') {
                setMessages((current) =>
                  current.map((message) =>
                    message.id === replyId
                      ? { ...message, content: message.content + frame.text }
                      : message,
                  ),
                )
              } else if (frame.type === 'meta') {
                const { type, ...meta } = frame
                patch(replyId, { meta })
              } else if (frame.type === 'tools') {
                setMessages((current) =>
                  current.map((message) =>
                    message.id === replyId
                      ? {
                          ...message,
                          meta: {
                            ...message.meta,
                            tool_calls: frame.tool_calls,
                            data_citations: frame.data_citations,
                          },
                        }
                      : message,
                  ),
                )
              } else if (frame.type === 'sources') {
                setMessages((current) =>
                  current.map((message) =>
                    message.id === replyId
                      ? { ...message, meta: { ...message.meta, sources: frame.sources } }
                      : message,
                  ),
                )
              } else if (frame.type === 'done') {
                const { type, ...meta } = frame
                patch(replyId, { meta, content: meta.answer, streaming: false })
              } else if (frame.type === 'error') {
                patch(replyId, { streaming: false, failed: frame.detail })
              }
            },
          },
          controller.signal,
        )
      } else {
        const response = await sendChat(request, controller.signal)
        patch(replyId, { meta: response, content: response.answer, streaming: false })
      }
    } catch (error) {
      const aborted = error instanceof DOMException && error.name === 'AbortError'
      patch(replyId, {
        streaming: false,
        failed: aborted
          ? copy.error.streamInterrupted
          : error instanceof Error
            ? error.message
            : copy.error.generic,
      })
    } finally {
      patch(replyId, { streaming: false })
      abort.current = null
      setBusy(false)
    }
  }, [busy, detection, detectionIsStub, draft, ghosttrace, language, messages, mode, patch])
  const onDetections = (result: DetectResult) => {
    setUploadError(null)
    const first = result.detections[0]
    if (!first) {
      setUploadError(copy.upload.noDetections)
      return
    }
    setContacts(result.detections)
    setDetection(first)
    setDetectionIsStub(result.stub)
    setGhosttrace(null)
    lastAttached.current = null
    // The tile is the question. Nobody should have to type one to find out what
    // the detector just found.
    void send({ text: copy.upload.autoBrief, record: first, ghosttrace: null })
  }
  const uploadEnabled = Boolean(health?.upload_enabled)
  return (
    <div className="dq-assistant">
      {/* config/theme.ts stays the single source of every value. Next.js used
          to inject these in the root layout; here the page carries them. */}
      <style dangerouslySetInnerHTML={{ __html: themeCss(theme) }} />
      <SystemStatus health={health} error={healthError} />
      <div className="thread" ref={scroller}>
        <div className="thread-inner">
          {messages.length === 0 ? (
            <EmptyState
              onPick={(value, pickMode) => {
                setDraft(value)
                if (pickMode) setMode(pickMode)
              }}
              disabled={busy}
            />
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} onCitation={openCitation} />
            ))
          )}
          {healthError && messages.length === 0 && (
            <div className="notice tone-alert">
              <p className="notice-title">{copy.error.title}</p>
              <p className="notice-body">{copy.error.backendDown}</p>
            </div>
          )}
        </div>
      </div>
      <div className="dock">
        <div className="dock-inner">
          {ghosttrace && (
            <div className="attached">
              <span className="attached-label">{copy.ghosttrace.attachedLabel}</span>
              <span className="attached-value mono">
                {ghosttrace.detection_id ?? copy.ghosttrace.notAvailable}
                {ghosttrace.priority?.tier && ` · ${ghosttrace.priority.tier}`}
              </span>
              {ghosttrace.synthetic && (
                <span className="attached-stub">{copy.ghosttrace.synthetic}</span>
              )}
              <button
                type="button"
                className="button-quiet"
                onClick={() => setGhosttrace(null)}
              >
                {copy.upload.detach}
              </button>
            </div>
          )}
          {detection && (
            <div className="attached">
              <span className="attached-label">{copy.detection.title}</span>
              <span className="attached-value">
                {detection.object_class ?? detection.label ?? copy.badge.unclassified}
                {typeof detection.confidence === 'number' &&
                  ` · ${copy.badge.confidence(detection.confidence)}`}
              </span>
              {detectionIsStub && <span className="attached-stub">{copy.notice.stubDetectionTitle}</span>}
              {detection.downgraded_from && (
                <span className="attached-stub" title={detection.downgraded_from}>
                  {copy.upload.downgraded}
                </span>
              )}
              <button
                type="button"
                className="button-quiet"
                onClick={() => {
                  setDetection(null)
                  setContacts([])
                  setDetectionIsStub(false)
                  lastAttached.current = null
                }}
              >
                {copy.upload.detach}
              </button>
            </div>
          )}
          {contacts.length > 1 && (
            <div className="attached">
              <span className="attached-label">{copy.upload.contactsLabel}</span>
              {contacts.map((contact, index) => (
                <button
                  key={index}
                  type="button"
                  className={`source-pill ${contact === detection ? 'is-active' : ''}`}
                  onClick={() => {
                    setDetection(contact)
                    lastAttached.current = null
                  }}
                >
                  {copy.upload.contact(
                    index + 1,
                    contact.object_class ?? copy.badge.unclassified,
                    contact.confidence ?? 0,
                  )}
                </button>
              ))}
            </div>
          )}
          {uploadError && <p className="dock-error">{uploadError}</p>}
          <div className="dock-controls">
            <div className="mode-toggle" role="radiogroup" aria-label={copy.copilot.modeLabel}>
              <span className="dock-control-label" aria-hidden="true">
                {copy.copilot.modeLabel}
              </span>
              {MODES.map((option) => (
                <button
                  key={option}
                  type="button"
                  role="radio"
                  aria-checked={mode === option}
                  className={`mode-option ${mode === option ? 'is-active' : ''}`}
                  title={copy.copilot.modeHint[option]}
                  disabled={busy}
                  onClick={() => setMode(option)}
                >
                  {copy.copilot.modes[option]}
                </button>
              ))}
            </div>
            <label className="language-picker">
              <span className="dock-control-label">{copy.language.label}</span>
              <select
                value={language}
                disabled={busy}
                onChange={(event) => chooseLanguage(event.target.value)}
                aria-describedby="language-note"
              >
                {languages.map((option) => (
                  <option key={option.code} value={option.code} lang={option.code}>
                    {option.code === 'en' ? option.native : `${option.native} (${option.english})`}
                  </option>
                ))}
              </select>
              <span id="language-note" className="language-note">
                {language === defaultLanguage ? '' : `${languageFor(language).english}: `}
                {copy.language.note}
              </span>
            </label>
          </div>
          <Composer
            value={draft}
            onChange={setDraft}
            onSend={() => send()}
            onStop={stop}
            busy={busy}
            disabled={Boolean(healthError)}
          >
            <UploadControl
              enabled={uploadEnabled}
              onResult={onDetections}
              onError={setUploadError}
            />
          </Composer>
        </div>
      </div>
      <CitationPanel
        sources={panelSources}
        data={panelData}
        selected={panelSelected}
        onSelect={setPanelSelected}
        onClose={() => setPanelSelected(null)}
      />
    </div>
  )
}
function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (value: string, mode?: AssistantMode) => void
  disabled: boolean
}) {
  return (
    <section className="empty">
      <h1 className="empty-title">{copy.empty.title}</h1>
      <p className="empty-body">{copy.empty.body}</p>
      <p className="empty-label">{copy.copilot.examplesLabel}</p>
      <ul className="empty-examples">
        {copy.copilot.examples.map((example) => (
          <li key={example}>
            <button
              type="button"
              className="example example-copilot"
              disabled={disabled}
              onClick={() => onPick(example, 'auto')}
            >
              <span className="example-tag">{copy.copilot.modes.copilot}</span>
              {example}
            </button>
          </li>
        ))}
      </ul>
      <p className="empty-label">{copy.copilot.referenceExamplesLabel}</p>
      <ul className="empty-examples">
        {copy.empty.examples.map((example) => (
          <li key={example}>
            <button
              type="button"
              className="example"
              disabled={disabled}
              onClick={() => onPick(example)}
            >
              {example}
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
