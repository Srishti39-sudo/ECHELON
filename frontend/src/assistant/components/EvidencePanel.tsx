import { Link } from 'react-router-dom'
import { copy } from '../config/copy'
import { languageFor } from '../config/languages'
import { settings } from '../config/settings'
import type { CitationTarget } from '../lib/citations'
import type { ChatResponse, DataCitation, Message, Source } from '../lib/types'

interface Props {
  /** The answer whose evidence is shown: the latest assistant turn, or none. */
  message: Message | null
  selected: CitationTarget | null
  onSelect: (target: CitationTarget | null) => void
}

/**
 * The evidence column. Always on screen beside the conversation.
 *
 * Three boxes, top to bottom: what produced the answer (model, mode, intent,
 * risk, grounding), the sources it cites with the passage behind each one,
 * and the survey records the copilot looked up. Clicking a citation marker in
 * the answer highlights and expands the matching card here.
 */
export function EvidencePanel({ message, selected, onSelect }: Props) {
  const meta = (message?.meta ?? null) as Partial<ChatResponse> | null
  const sources: Source[] = meta?.sources ?? []
  const data: DataCitation[] = meta?.data_citations ?? []
  const streaming = Boolean(message?.streaming)

  return (
    <aside className="evidence" aria-label={copy.evidence.title}>
      <header className="evidence-head">
        <h2 className="evidence-title">{copy.evidence.title}</h2>
        {message && !streaming && (
          <span className="evidence-count mono">
            {copy.citations.count(sources.length)}
            {data.length > 0 && ` · ${copy.copilot.dataCount(data.length)}`}
          </span>
        )}
      </header>

      {!message && (
        <p className="evidence-empty">{copy.evidence.empty}</p>
      )}

      {message && meta && !streaming && (
        <section className="evidence-box">
          <h3 className="evidence-box-title">{copy.evidence.details}</h3>
          <dl className="evidence-details">
            <Row label={copy.evidence.model} value={meta.model ? `${meta.provider} · ${meta.model}` : meta.provider} mono />
            <Row label={copy.evidence.mode} value={modeLabel(meta)} />
            {meta.intent && <Row label={copy.evidence.intent} value={copy.badge.intent[meta.intent]} />}
            {meta.severity && (
              <Row label={copy.badge.severityLabel} value={copy.badge.severity[meta.severity]} tone={meta.severity} />
            )}
            <Row
              label={copy.evidence.grounded}
              value={meta.grounded ? copy.evidence.yes : copy.evidence.no}
              tone={meta.grounded ? 'ok' : 'alert'}
            />
            {meta.generated_by && meta.generated_by !== 'model' && (
              <Row label={copy.evidence.generatedBy} value={copy.evidence.generated[meta.generated_by]} tone="alert" />
            )}
            {meta.language && meta.language !== 'en' && (
              <Row label={copy.language.label} value={languageFor(meta.language).english} />
            )}
            {meta.unsourced_numbers && meta.unsourced_numbers.length > 0 && (
              <Row label={copy.evidence.unsourced} value={meta.unsourced_numbers.join(', ')} tone="alert" mono />
            )}
          </dl>
        </section>
      )}

      {sources.length > 0 && (
        <section className="evidence-box">
          <h3 className="evidence-box-title">{copy.citations.listTitle}</h3>
          <ol className="evidence-list">
            {sources.map((source) => {
              const active = selected?.kind === 'source' && selected.n === source.n
              return (
                <li key={source.id} className={`evidence-card ${active ? 'is-active' : ''}`}>
                  <button
                    type="button"
                    className="evidence-card-head"
                    aria-expanded={active}
                    onClick={() => onSelect(active ? null : { kind: 'source', n: source.n })}
                  >
                    <span className="evidence-index">{source.n}</span>
                    <span className="evidence-card-text">
                      <span className="evidence-card-title">{source.title}</span>
                      {source.authority && <span className="evidence-card-sub">{source.authority}</span>}
                    </span>
                  </button>
                  {active && (
                    <div className="evidence-card-body">
                      {source.section && (
                        <p className="evidence-card-section">{source.section}</p>
                      )}
                      <blockquote className="evidence-snippet">{source.snippet}</blockquote>
                      <p className="evidence-card-section mono">
                        {scoreLabel(source.score)}: {source.score.toFixed(2)}
                      </p>
                      {source.pdf_url ? (
                        <a
                          className="button-link"
                          href={`${settings.apiBaseUrl}${source.pdf_url}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {copy.citations.openPdf}
                        </a>
                      ) : null}
                    </div>
                  )}
                </li>
              )
            })}
          </ol>
        </section>
      )}

      {data.length > 0 && (
        <section className="evidence-box">
          <h3 className="evidence-box-title">{copy.data.listTitle}</h3>
          <ol className="evidence-list">
            {data.map((record) => {
              const active = selected?.kind === 'data' && selected.n === record.n
              return (
                <li key={record.n} className={`evidence-card ${active ? 'is-active' : ''}`}>
                  <button
                    type="button"
                    className="evidence-card-head"
                    aria-expanded={active}
                    onClick={() => onSelect(active ? null : { kind: 'data', n: record.n })}
                  >
                    <span className="evidence-index evidence-index-data">D{record.n}</span>
                    <span className="evidence-card-text">
                      <span className="evidence-card-title">{record.label}</span>
                      <span className="evidence-card-sub">
                        {record.survey_id ?? record.source_file}
                        {record.synthetic ? ` · ${copy.ghosttrace.synthetic}` : ''}
                      </span>
                    </span>
                  </button>
                  {active && (
                    <div className="evidence-card-body">
                      <dl className="evidence-details">
                        {Object.entries(record.summary).map(([key, value]) => (
                          <Row key={key} label={key} value={formatValue(value)} mono />
                        ))}
                      </dl>
                      {record.link && (
                        <Link className="button-link" to={record.link}>
                          {record.link.startsWith('/ghosttrace') ? copy.data.openGhosttrace : copy.data.openMap}
                        </Link>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ol>
        </section>
      )}
    </aside>
  )
}

function Row({
  label,
  value,
  mono = false,
  tone,
}: {
  label: string
  value: string | undefined
  mono?: boolean
  tone?: string
}) {
  if (!value) return null
  return (
    <div className={`evidence-row ${tone ? `tone-${tone}` : ''}`}>
      <dt>{label}</dt>
      <dd className={mono ? 'mono' : ''}>{value}</dd>
    </div>
  )
}

/** A cosine similarity sits in 0..1; anything else is the reranker's logit. */
function scoreLabel(score: number): string {
  return score >= 0 && score <= 1 ? copy.citations.similarity : copy.citations.rerankScore
}

function modeLabel(meta: Partial<ChatResponse>): string {
  return meta.mode === 'copilot' ? copy.copilot.modes.copilot : copy.copilot.modes.reference
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return copy.ghosttrace.notAvailable
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
