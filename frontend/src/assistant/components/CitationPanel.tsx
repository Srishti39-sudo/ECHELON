import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { copy } from '../config/copy'
import { settings } from '../config/settings'
import type { CitationTarget } from '../lib/citations'
import type { DataCitation, Source } from '../lib/types'
interface Props {
  sources: Source[]
  data: DataCitation[]
  selected: CitationTarget | null
  onSelect: (target: CitationTarget) => void
  onClose: () => void
}
/**
 * The proof panel.
 *
 * Everything the assistant asserts is supposed to trace to a passage in a real
 * publication or, on a Mission Copilot answer, to a survey record. This is
 * where an operator checks that, so it shows the passage itself, or the record
 * exactly as the answer saw it, rather than a summary of either.
 */
export function CitationPanel({ sources, data, selected, onSelect, onClose }: Props) {
  const open = selected !== null && (sources.length > 0 || data.length > 0)
  const tab = selected?.kind ?? 'source'
  const activeSource = tab === 'source' ? (sources.find((s) => s.n === selected?.n) ?? null) : null
  const activeData = tab === 'data' ? (data.find((d) => d.n === selected?.n) ?? null) : null
  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  return (
    <>
      <div
        className={`panel-scrim ${open ? 'is-open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        className={`panel ${open ? 'is-open' : ''}`}
        aria-hidden={!open}
        aria-label={tab === 'data' ? copy.data.panelTitle : copy.citations.panelTitle}
      >
        <header className="panel-head">
          <h2 className="panel-title">
            {tab === 'data' ? copy.data.panelTitle : copy.citations.panelTitle}
          </h2>
          <button type="button" className="button-quiet" onClick={onClose}>
            {copy.citations.close}
          </button>
        </header>
        {sources.length > 0 && data.length > 0 && (
          <div className="panel-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'source'}
              className={`panel-tab ${tab === 'source' ? 'is-active' : ''}`}
              onClick={() => onSelect({ kind: 'source', n: sources[0].n })}
            >
              {copy.data.tabSources} <span className="mono">{sources.length}</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'data'}
              className={`panel-tab ${tab === 'data' ? 'is-active' : ''}`}
              onClick={() => onSelect({ kind: 'data', n: data[0].n })}
            >
              {copy.data.tabData} <span className="mono">{data.length}</span>
            </button>
          </div>
        )}
        {activeSource && (
          <SourceBody
            active={activeSource}
            sources={sources}
            onSelect={(n) => onSelect({ kind: 'source', n })}
          />
        )}
        {activeData && (
          <DataBody active={activeData} data={data} onSelect={(n) => onSelect({ kind: 'data', n })} />
        )}
      </aside>
    </>
  )
}

function SourceBody({
  active,
  sources,
  onSelect,
}: {
  active: Source
  sources: Source[]
  onSelect: (n: number) => void
}) {
  return (
    <div className="panel-body">
      <div className="source-card">
        <div className="source-index">{active.n}</div>
        <div className="source-headings">
          <h3 className="source-title">{active.title}</h3>
          {active.section && (
            <p className="source-sub">
              {copy.citations.section}: {active.section}
            </p>
          )}
        </div>
      </div>
      <dl className="source-meta">
        {active.authority && (
          <div>
            <dt>{copy.citations.authority}</dt>
            <dd>{active.authority}</dd>
          </div>
        )}
        {active.status && (
          <div>
            <dt>{copy.citations.status}</dt>
            <dd>{active.status}</dd>
          </div>
        )}
        {settings.features.showRetrievalScores && (
          <div>
            <dt>{copy.citations.similarity}</dt>
            <dd className="mono">{active.score.toFixed(3)}</dd>
          </div>
        )}
      </dl>
      <blockquote className="source-snippet">{active.snippet}</blockquote>
      {active.pdf_url ? (
        <a
          className="button-link"
          href={`${settings.apiBaseUrl}${active.pdf_url}`}
          target="_blank"
          rel="noreferrer"
        >
          {copy.citations.openPdf}
        </a>
      ) : (
        <p className="source-sub">{copy.citations.noPdf}</p>
      )}
      <h4 className="panel-subhead">{copy.citations.listTitle}</h4>
      <ul className="source-list">
        {sources.map((source) => (
          <li key={source.id}>
            <button
              type="button"
              className={`source-list-item ${source.n === active.n ? 'is-active' : ''}`}
              onClick={() => onSelect(source.n)}
            >
              <span className="source-list-index">{source.n}</span>
              <span className="source-list-title">{source.title}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** Where a record opens. The hazard map takes its survey through router state. */
function recordLink(link: string | null | undefined) {
  if (!link) return null
  const map = /^\/map\?survey=(.+)$/.exec(link)
  if (map) {
    return { to: '/map', state: { surveyId: decodeURIComponent(map[1]) }, label: copy.data.openMap }
  }
  if (link.startsWith('/ghosttrace/')) return { to: link, state: undefined, label: copy.data.openGhosttrace }
  return null
}

function fieldValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function DataBody({
  active,
  data,
  onSelect,
}: {
  active: DataCitation
  data: DataCitation[]
  onSelect: (n: number) => void
}) {
  const target = recordLink(active.link)
  return (
    <div className="panel-body">
      <div className="source-card">
        <div className="source-index data-index">D{active.n}</div>
        <div className="source-headings">
          <h3 className="source-title">{active.label}</h3>
          <p className="source-sub">{copy.data.note}</p>
        </div>
      </div>
      {active.synthetic && <p className="survey-demo">{copy.data.synthetic}</p>}
      <dl className="source-meta">
        {active.survey_id && (
          <div>
            <dt>{copy.data.survey}</dt>
            <dd className="mono">{active.survey_id}</dd>
          </div>
        )}
        <div>
          <dt>{copy.data.file}</dt>
          <dd>{active.source_file}</dd>
        </div>
        {active.record_id && (
          <div>
            <dt>{copy.data.record}</dt>
            <dd className="mono">{active.record_id}</dd>
          </div>
        )}
        <div>
          <dt>{copy.data.kind}</dt>
          <dd>{active.kind}</dd>
        </div>
      </dl>
      <h4 className="panel-subhead">{copy.data.fields}</h4>
      <dl className="data-fields">
        {Object.entries(active.summary).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd className="mono">{fieldValue(value)}</dd>
          </div>
        ))}
      </dl>
      {target && (
        <Link className="button-link" to={target.to} state={target.state}>
          {target.label}
        </Link>
      )}
      <h4 className="panel-subhead">{copy.data.listTitle}</h4>
      <ul className="source-list">
        {data.map((record) => (
          <li key={record.n}>
            <button
              type="button"
              className={`source-list-item ${record.n === active.n ? 'is-active' : ''}`}
              onClick={() => onSelect(record.n)}
            >
              <span className="source-list-index">D{record.n}</span>
              <span className="source-list-title">{record.label}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
