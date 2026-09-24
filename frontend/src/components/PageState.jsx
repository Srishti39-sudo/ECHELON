/**
 * The three things a data page shows when it has no data to show.
 *
 * Kept together because the distinction matters more than it looks. "Nothing
 * has been scanned yet" and "the backend is not running" look identical if you
 * render one empty state for both, and they call for completely different
 * actions from the operator.
 */

import { AlertTriangle, Inbox, Loader2, PlugZap } from "lucide-react"

export function Loading({ label = "Loading" }) {
  return (
    <div className="page-state">
      <Loader2 size={30} className="spin" />
      <p>{label}</p>
    </div>
  )
}

/**
 * A failed request, told apart by status.
 *
 * Status 0 comes from services/api.js and means the request never arrived, so
 * the message is about starting the backend rather than about what it said.
 */
export function Failed({ error, onRetry }) {
  const offline = error?.status === 0

  return (
    <div className="page-state error">
      {offline ? <PlugZap size={30} /> : <AlertTriangle size={30} />}

      <h3>{offline ? "Backend unreachable" : "That request failed"}</h3>
      <p>{error?.message ?? "No detail was given."}</p>

      {offline && (
        <code className="page-state-code">
          uvicorn backend.app.main:app --reload
        </code>
      )}

      {onRetry && (
        <button className="view-all-button" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

export function Empty({ title, children, action }) {
  return (
    <div className="page-state">
      <Inbox size={30} />
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}
