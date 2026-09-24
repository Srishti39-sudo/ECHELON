import { useRef } from "react"
import { ListOrdered } from "lucide-react"

import TargetCard from "./TargetCard"

/**
 * Targets in priority order. Arrow keys, Home and End move between cards;
 * Enter or Space selects; the "Open details" control jumps to the detail panel.
 */
function RescueQueue({ targets, selectedId, onSelect, onOpenDetail, onAsk, showSuppressed, onToggleSuppressed, suppressedCount }) {
  const buttons = useRef([])

  const move = (event, index) => {
    const keys = { ArrowDown: index + 1, ArrowUp: index - 1, Home: 0, End: targets.length - 1 }
    if (!(event.key in keys)) return
    event.preventDefault()
    const next = Math.max(0, Math.min(targets.length - 1, keys[event.key]))
    buttons.current[next]?.focus()
  }

  return (
    <section className="gt-queue" aria-labelledby="gt-queue-title">
      <header className="gt-section-head">
        <h2 id="gt-queue-title">
          <ListOrdered size={17} aria-hidden="true" /> Rescue queue
        </h2>
        <span className="gt-muted">{targets.length} shown · by priority rank</span>
      </header>

      {suppressedCount > 0 && (
        <label className="gt-toggle-line">
          <input type="checkbox" checked={showSuppressed} onChange={(e) => onToggleSuppressed(e.target.checked)} />
          Show {suppressedCount} suppressed contact{suppressedCount === 1 ? "" : "s"}
        </label>
      )}

      <ol className="gt-queue-list" aria-label="Rescue queue, highest priority first">
        {targets.map((target, index) => (
          <TargetCard
            key={target.detection_id}
            target={target}
            selected={target.detection_id === selectedId}
            onSelect={onSelect}
            onOpenDetail={onOpenDetail}
            onAsk={onAsk}
            buttonRef={(el) => {
              buttons.current[index] = el
            }}
            onKeyDown={(event) => move(event, index)}
          />
        ))}
      </ol>
    </section>
  )
}

export default RescueQueue
