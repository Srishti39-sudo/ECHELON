import { useId, useState } from "react"
import {
  AlertOctagon,
  AlertTriangle,
  CircleDot,
  Fish,
  HelpCircle,
  Info,
  MoveRight,
  Repeat,
  Ship,
  Sparkles,
  XCircle,
} from "lucide-react"

import {
  styleForActivity,
  styleForChange,
  styleForHazard,
  styleForTier,
} from "../config"

/**
 * Small labelled chips. Every chip carries an icon and a word as well as a
 * colour, so no state is signalled by colour alone.
 */

const TIER_ICON = { urgent: AlertOctagon, high: AlertTriangle, routine: CircleDot }

export function TierChip({ tier, compact = false }) {
  const style = styleForTier(tier)
  const Icon = TIER_ICON[tier] || HelpCircle
  return (
    <span className="gt-chip gt-chip--strong" style={{ color: style.color, background: style.tint }}>
      <Icon size={13} aria-hidden="true" />
      {compact ? style.label : `${style.label} priority`}
    </span>
  )
}

export function ActivityChip({ activity }) {
  const available = activity && activity.available !== false
  const level = available ? activity.level || "unknown" : "unknown"
  const style = styleForActivity(level)
  return (
    <span className="gt-chip" style={{ color: style.color, background: style.tint }}>
      <Fish size={13} aria-hidden="true" />
      {level === "high" ? "Actively fishing" : `Activity ${style.short.toLowerCase()}`}
    </span>
  )
}

const CHANGE_ICON = { new: Sparkles, moved: MoveRight, persistent: Repeat, removed: XCircle }

export function ChangeChip({ status }) {
  const style = styleForChange(status)
  const Icon = CHANGE_ICON[status] || HelpCircle
  return (
    <span className="gt-chip" style={{ color: style.color, background: style.tint }}>
      <Icon size={13} aria-hidden="true" />
      {style.label}
    </span>
  )
}

export function HazardChip({ level, prefix = "Propeller" }) {
  const style = styleForHazard(level)
  return (
    <span className="gt-chip" style={{ color: style.color, background: style.tint }}>
      <Ship size={13} aria-hidden="true" />
      {prefix} {style.label.toLowerCase()}
    </span>
  )
}

export function Tag({ children, tone = "neutral", title }) {
  return (
    <span className={`gt-tag gt-tag--${tone}`} title={title}>
      {children}
    </span>
  )
}

/**
 * A focusable "i" with a tooltip that opens on hover and on keyboard focus.
 * The tooltip text is also the button's accessible description.
 */
export function InfoTip({ label, children, align = "start" }) {
  const id = useId()
  // Fixed positioning so the tooltip is never clipped by a scrolling list.
  const [position, setPosition] = useState(null)
  const place = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const width = 290
    const left = align === "end" ? rect.right - width + 10 : rect.left - 10
    setPosition({
      top: Math.round(rect.bottom + 6),
      left: Math.round(Math.max(8, Math.min(left, window.innerWidth - width - 8))),
    })
  }
  return (
    <span className={`gt-infotip gt-infotip--${align}`} onMouseEnter={place} onFocus={place}>
      <button type="button" className="gt-infotip-trigger" aria-label={label} aria-describedby={id}>
        <Info size={13} aria-hidden="true" />
      </button>
      <span
        role="tooltip"
        id={id}
        className="gt-infotip-body"
        style={position ? { position: "fixed", top: position.top, left: position.left, right: "auto" } : undefined}
      >
        {children}
      </span>
    </span>
  )
}

/** What a section shows instead of a blank when the engine says it is unavailable. */
export function Unavailable({ title, reason }) {
  return (
    <div className="gt-unavailable" role="note">
      <HelpCircle size={16} aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        <p>{reason}</p>
      </div>
    </div>
  )
}
