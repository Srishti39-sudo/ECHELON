import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { copy } from '../config/copy'
import { languageFor } from '../config/languages'
import { citationTarget, linkCitations, type CitationTarget } from '../lib/citations'
import type { DataCitation, GhostTraceContext, Message, Source, SurveyContext, ToolCall } from '../lib/types'
import { StatusBadge } from './StatusBadge'
interface Props {
  message: Message
  onCitation: (target: CitationTarget, sources: Source[], data: DataCitation[]) => void
}
/**
 * One message.
 *
 * The assistant half carries the honesty machinery. An ungrounded answer, an
 * answer that declined to fill a gap, and an unidentified object each render
 * differently from a confident one, because an operator scanning quickly should
 * not have to read the prose to find out which they are looking at.
 */
export function MessageBubble({ message, onCitation }: Props) {
  if (message.role === 'user') {
    return (
      <article className="message message-user">
        <p className="message-role">{copy.roles.user}</p>
        <div className="bubble bubble-user">{message.content}</div>
        {message.detection && <DetectionSummary message={message} />}
      </article>
    )
  }
  const meta = message.meta ?? {}
  const sources = meta.sources ?? []
  const ungrounded = meta.grounded === false && !message.streaming
  const showRefusal = meta.refusal === true && meta.grounded !== false
  const offline = meta.generated_by === 'retrieval_only'
  const dataOnly = meta.generated_by === 'data_only'
  const data = meta.data_citations ?? []
  const toolCalls = meta.tool_calls ?? []
  const copilot = meta.mode === 'copilot'
  const labels = languageFor(meta.language).labels
  const unsourced = !message.streaming ? (meta.unsourced_numbers ?? []) : []
  return (
    <article className="message message-assistant">
      <p className="message-role">{copy.roles.assistant}</p>
      <div className={`bubble bubble-assistant ${ungrounded ? 'is-ungrounded' : ''}`}>
        {!message.streaming && meta.intent && (
          <StatusBadge meta={meta} survey={message.survey} ghosttrace={message.ghosttrace} />
        )}

        {message.survey && <SurveyHandover survey={message.survey} />}
        {message.ghosttrace && <GhostTraceHandover context={message.ghosttrace} />}
        {copilot && (
          <ToolChips
            calls={toolCalls}
            streaming={Boolean(message.streaming)}
            onOpen={(n) => onCitation({ kind: 'data', n }, sources, data)}
          />
        )}
        {(offline || dataOnly) && !message.streaming && (
          <div className="notice tone-alert offline-notice" role="status">
            <p className="notice-title">{dataOnly ? labels.dataOnly : labels.offline}</p>
            <p className="notice-body">{dataOnly ? copy.copilot.dataOnlyBody : copy.offline.body}</p>
            {meta.provider_errors && meta.provider_errors.length > 0 && (
              <p className="notice-body offline-why">
                {copy.offline.why}: {meta.provider_errors.join('; ')}
              </p>
            )}
          </div>
        )}
        {meta.coverage_gap && (
          <Notice
            tone="caution"
            title={copy.notice.coverageGapTitle}
            body={copy.notice.coverageGapBody}
          />
        )}
        {meta.is_anomaly && (
          <Notice tone="caution" title={copy.notice.anomalyTitle} body={copy.notice.anomalyBody} />
        )}
        {ungrounded && (
          <Notice tone="alert" title={copy.notice.ungroundedTitle} body={copy.notice.ungroundedBody} />
        )}
        {showRefusal && (
          <Notice tone="caution" title={copy.notice.refusalTitle} body={copy.notice.refusalBody} />
        )}
        {unsourced.length > 0 && (
          <Notice tone="caution" title={copy.numbers.title} body={copy.numbers.body(unsourced)} />
        )}
        <div className="prose">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a({ href, children, ...rest }) {
                const target = citationTarget(href)
                if (target === null) {
                  return (
                    <a href={href} target="_blank" rel="noreferrer" {...rest}>
                      {children}
                    </a>
                  )
                }
                return (
                  <button
                    type="button"
                    className={`cite ${target.kind === 'data' ? 'cite-data' : ''}`}
                    title={
                      target.kind === 'data'
                        ? copy.data.markerTitle(target.n)
                        : copy.citations.markerTitle(target.n)
                    }
                    onClick={() => onCitation(target, sources, data)}
                  >
                    {target.kind === 'data' ? `D${target.n}` : target.n}
                  </button>
                )
              },
            }}
          >
            {linkCitations(message.content)}
          </ReactMarkdown>
          {message.streaming && <span className="caret" aria-hidden="true" />}
        </div>
        {meta.matches && meta.matches.length > 0 && !message.streaming && (
          <section className="matches">
            <h4 className="matches-title">{copy.matches.title}</h4>
            <p className="matches-caveat">{copy.matches.caveat}</p>
            <ol className="matches-list">
              {meta.matches.map((match) => (
                <li key={match.id}>
                  <div className="matches-head">
                    <span className="matches-name">{match.name}</span>
                    <span className="matches-score mono">
                      {copy.matches.similarity} {match.similarity.toFixed(2)}
                    </span>
                  </div>
                  <p className="matches-line">
                    <span className="matches-label">{copy.matches.hazard}</span> {match.hazard}
                  </p>
                  <p className="matches-line">
                    <span className="matches-label">{copy.matches.confirms}</span> {match.confirms}
                  </p>
                  <p className="matches-line">
                    <span className="matches-label">{copy.matches.rulesOut}</span> {match.rules_out}
                  </p>
                </li>
              ))}
            </ol>
          </section>
        )}
        {sources.length > 0 && (
          <footer className="sources-strip">
            <span className="sources-count">{copy.citations.count(sources.length)}</span>
            {sources.map((source) => (
              <button
                key={source.id}
                type="button"
                className="source-pill"
                onClick={() => onCitation({ kind: 'source', n: source.n }, sources, data)}
              >
                <span className="source-pill-index">{source.n}</span>
                {source.title}
              </button>
            ))}
          </footer>
        )}
        {data.length > 0 && (
          <footer className="sources-strip data-strip">
            <span className="sources-count">{copy.copilot.dataCount(data.length)}</span>
            {data.map((record) => (
              <button
                key={record.n}
                type="button"
                className="source-pill data-pill"
                title={record.label}
                onClick={() => onCitation({ kind: 'data', n: record.n }, sources, data)}
              >
                <span className="source-pill-index">D{record.n}</span>
                {record.record_id ?? record.label}
              </button>
            ))}
          </footer>
        )}
      </div>
      {message.failed && (
        <div className="notice tone-alert">
          <p className="notice-title">{copy.error.title}</p>
          <p className="notice-body">{message.failed}</p>
        </div>
      )}
    </article>
  )
}
/**
 * Which survey data the copilot consulted, one chip per tool call. Each chip
 * opens the first record that call returned.
 */
function ToolChips({
  calls,
  streaming,
  onOpen,
}: {
  calls: ToolCall[]
  streaming: boolean
  onOpen: (n: number) => void
}) {
  if (calls.length === 0) {
    return streaming ? (
      <p className="tool-chips-label" role="status">
        {copy.copilot.lookingUp}
      </p>
    ) : null
  }
  const plannedBy = calls[0]?.planned_by
  return (
    <div className="tool-chips">
      <p className="tool-chips-label">
        {copy.copilot.consulted}
        {plannedBy && copy.copilot.plannedBy[plannedBy] && (
          <span className="tool-chips-planner"> · {copy.copilot.plannedBy[plannedBy]}</span>
        )}
      </p>
      <ul className="tool-chips-list">
        {calls.map((call, index) => (
          <li key={`${call.name}-${index}`}>
            <button
              type="button"
              className={`tool-chip ${call.error ? 'is-error' : ''}`}
              title={`${call.name} ${JSON.stringify(call.args)}`}
              disabled={call.citations.length === 0}
              onClick={() => call.citations[0] && onOpen(call.citations[0])}
            >
              <span className="tool-chip-name mono">{call.name}</span>
              {call.summary}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function SurveyHandover({ survey }: { survey: SurveyContext }) {
  const georeferenced = survey.lat !== null && survey.lon !== null
  return (
    <div className="survey-handover">
      <p className="survey-from">{copy.survey.from}</p>
      <dl className="detection-fields">
        <div>
          <dt>{copy.survey.action}</dt>
          <dd>{survey.recommended_action}</dd>
        </div>
        <div>
          <dt>{copy.survey.position}</dt>
          <dd className="mono">
            {georeferenced
              ? `${survey.lat}, ${survey.lon}`
              : `x ${survey.centroid.global_x}, y ${survey.centroid.global_y}`}
          </dd>
        </div>
        {typeof survey.priority_rank === 'number' && (
          <div>
            <dt>{copy.survey.rank}</dt>
            <dd>{survey.priority_rank}</dd>
          </div>
        )}
        {typeof survey.detection_count === 'number' && (
          <div>
            <dt>{copy.survey.detections}</dt>
            <dd>{survey.detection_count}</dd>
          </div>
        )}
      </dl>
      {!georeferenced && <p className="survey-note">{copy.survey.notGeoreferenced}</p>}
      {survey.demo && <p className="survey-demo">{copy.survey.demo}</p>}
    </div>
  )
}

const given = (value: unknown): string =>
  value === null || value === undefined || value === '' ? copy.ghosttrace.notAvailable : String(value)

/**
 * The GhostTrace target as it was handed over. Every value is displayed as
 * given; the card computes nothing. Synthetic runs say so before anything else.
 */
function GhostTraceHandover({ context }: { context: GhostTraceContext }) {
  const georeferenced = context.latitude !== null && context.longitude !== null
  const priority = context.priority
  const habitat = context.habitat_nearest?.[0]
  return (
    <div className="survey-handover ghosttrace-handover">
      <p className="survey-from">
        {copy.ghosttrace.from}
        {context.synthetic && (
          <span className="ghosttrace-synthetic">{copy.ghosttrace.synthetic}</span>
        )}
      </p>
      {context.synthetic && <p className="survey-demo">{copy.ghosttrace.syntheticBody}</p>}
      <dl className="detection-fields">
        <div>
          <dt>{copy.ghosttrace.target}</dt>
          <dd className="mono">{given(context.detection_id)}</dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.survey}</dt>
          <dd>{given(context.survey_title ?? context.survey_id)}</dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.objectClass}</dt>
          <dd>
            {given(context.object_class)}
            {typeof context.confidence_pct === 'number' && ` · ${context.confidence_pct}%`}
          </dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.position}</dt>
          <dd className="mono">
            {georeferenced
              ? `${context.latitude}, ${context.longitude}`
              : copy.ghosttrace.notGeoreferenced}
          </dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.priority}</dt>
          <dd>
            {priority
              ? copy.ghosttrace.priorityValue(
                  given(priority.tier),
                  given(priority.rank),
                  given(priority.score),
                )
              : copy.ghosttrace.notAvailable}
          </dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.activity}</dt>
          <dd>{given(context.activity?.level)}</dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.habitat}</dt>
          <dd>
            {habitat
              ? copy.ghosttrace.habitatValue(habitat.name ?? given(habitat.kind), given(habitat.distance_m))
              : copy.ghosttrace.notAvailable}
          </dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.propeller}</dt>
          <dd>{given(context.people?.propeller_hazard_level)}</dd>
        </div>
        <div>
          <dt>{copy.ghosttrace.change}</dt>
          <dd>{given(context.change?.status)}</dd>
        </div>
        {context.authorities && context.authorities.length > 0 && (
          <div>
            <dt>{copy.ghosttrace.authorities}</dt>
            <dd>{context.authorities.map((a) => given(a.name)).join('; ')}</dd>
          </div>
        )}
      </dl>
      <p className="survey-note">{copy.ghosttrace.dataNotSource}</p>
      {context.caveats && context.caveats.length > 0 && (
        <details className="ghosttrace-caveats">
          <summary>{copy.ghosttrace.caveats}</summary>
          <ul>
            {context.caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

function Notice({ tone, title, body }: { tone: string; title: string; body: string }) {
  return (
    <div className={`notice tone-${tone}`}>
      <p className="notice-title">{title}</p>
      <p className="notice-body">{body}</p>
    </div>
  )
}
function DetectionSummary({ message }: { message: Message }) {
  const record = message.detection
  if (!record) return null
  const entries = Object.entries(record).filter(
    ([, value]) => value !== null && value !== undefined && value !== '',
  )
  return (
    <div className="detection-attached">
      {message.detectionIsStub && (
        <p className="detection-stub">{copy.notice.stubDetectionTitle}</p>
      )}
      <dl className="detection-fields">
        {entries.map(([key, value]) => (
          <div key={key}>
            <dt>{copy.detection.fields[key] ?? key}</dt>
            <dd>{Array.isArray(value) ? value.join(', ') : String(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
