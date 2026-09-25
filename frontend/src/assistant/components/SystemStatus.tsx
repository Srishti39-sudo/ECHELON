import { copy } from '../config/copy'
import type { Health } from '../lib/types'
/** The header strip: is the assistant up, and what is behind it. */
export function SystemStatus({ health, error }: { health: Health | null; error: string | null }) {
  const state = error ? 'offline' : !health ? 'checking' : health.status === 'ready' ? 'ready' : 'degraded'
  return (
    <header className="app-head">
      <div className="app-head-inner">
        <div className="app-identity">
          <span className="app-name">{copy.app.name}</span>
          <span className="app-subtitle">{copy.app.subtitle}</span>
        </div>
        <div className="app-status">
          <span className={`status-dot status-${state}`} aria-hidden="true" />
          <span className="status-label">{copy.status[state]}</span>
        </div>
      </div>
      {health?.detector === 'stub' && (
        <p className="app-head-notice">{copy.status.detectorStub}</p>
      )}
    </header>
  )
}
