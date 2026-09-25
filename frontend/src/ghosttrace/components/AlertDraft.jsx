import { useEffect, useRef, useState } from "react"
import { Check, Copy, Download, FileWarning } from "lucide-react"

import { copy } from "../config"
import { DASH, alertPlainText, citationText } from "../format"
import { Unavailable } from "./Chips"

/**
 * The drafted alert for one target.
 *
 * Read-only on purpose: this page drafts, a person sends. Copy puts the text
 * on the clipboard; Download fetches the backend's .txt (or builds the same
 * text locally when previewing a fixture with no backend).
 */
function AlertDraft({ surveyId, target, downloadUrl }) {
  const [copied, setCopied] = useState(null)
  const textarea = useRef(null)
  const timer = useRef(null)
  const alert = target?.alert

  useEffect(() => () => window.clearTimeout(timer.current), [])

  if (!alert || !alert.draft_text) {
    return (
      <div className="gt-section">
        <Unavailable title="No alert drafted" reason={alert?.reason || alert?.basis || "The engine did not draft an alert for this target."} />
      </div>
    )
  }

  const flash = (state) => {
    setCopied(state)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setCopied(null), 2200)
  }

  const onCopy = async () => {
    const text = [alert.subject ? `Subject: ${alert.subject}` : null, alert.draft_text].filter(Boolean).join("\n\n")
    try {
      await navigator.clipboard.writeText(text)
      flash("ok")
    } catch {
      // Clipboard API refused (insecure context or permissions): select the
      // text so a manual copy is one keystroke away, and say so.
      textarea.current?.focus()
      textarea.current?.select()
      flash("manual")
    }
  }

  const onLocalDownload = () => {
    const blob = new Blob([alertPlainText(surveyId, target)], { type: "text/plain;charset=utf-8" })
    const href = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = href
    link.download = `${surveyId}-${target.detection_id}-alert.txt`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(href), 1000)
  }

  return (
    <div className="gt-section">
      <div className="gt-draft-banner" role="note">
        <FileWarning size={17} aria-hidden="true" />
        <div>
          <strong>{copy.alertDraft}</strong>
        </div>
      </div>

      <div className="gt-subsection">
        <h4>Suggested recipients</h4>
        {alert.authorities?.length ? (
          <div className="gt-table-wrap">
            <table className="gt-table">
              <caption className="gt-sr-only">Suggested authorities</caption>
              <thead>
                <tr>
                  <th scope="col">Authority</th>
                  <th scope="col">Role</th>
                  <th scope="col">Why suggested</th>
                </tr>
              </thead>
              <tbody>
                {alert.authorities.map((a, i) => (
                  <tr key={`${a.name}-${i}`}>
                    <td className="gt-strong">{a.name || DASH}</td>
                    <td>{a.role || DASH}</td>
                    <td className="gt-muted">{a.contact_basis || DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="gt-muted">No authorities suggested.</p>
        )}
      </div>

      <label className="gt-field">
        <span>Subject</span>
        <input type="text" readOnly value={alert.subject || ""} />
      </label>
      <label className="gt-field">
        <span>Draft text</span>
        <textarea ref={textarea} readOnly rows={11} value={alert.draft_text} />
      </label>

      <div className="gt-draft-actions">
        <button type="button" className="gt-btn" onClick={onCopy}>
          {copied === "ok" ? <Check size={15} aria-hidden="true" /> : <Copy size={15} aria-hidden="true" />}
          {copied === "ok" ? "Copied" : "Copy"}
        </button>
        {downloadUrl ? (
          <a className="gt-btn" href={downloadUrl} download>
            <Download size={15} aria-hidden="true" /> Download .txt
          </a>
        ) : (
          <button type="button" className="gt-btn" onClick={onLocalDownload}>
            <Download size={15} aria-hidden="true" /> Download .txt
          </button>
        )}
        <span className="gt-muted" role="status" aria-live="polite">
          {copied === "manual" ? "Clipboard unavailable — text selected, press Ctrl/Cmd+C." : copied === "ok" ? "Copied to clipboard." : ""}
        </span>
      </div>

      {(alert.citations?.length > 0 || alert.basis) && (
        <div className="gt-subsection">
          {alert.citations?.length > 0 && (
            <>
              <h4>Citations</h4>
              <ul className="gt-bullets">
                {alert.citations.map((c, i) => (
                  <li key={i}>
                    {citationText(c)}
                    {c?.status && <small className="gt-muted"> · {c.status}</small>}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default AlertDraft
