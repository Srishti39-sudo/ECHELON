import { Loader2 } from "lucide-react"

import { driftColor, layerKinds, routeColor, styleForTier, tierStyle } from "../config"

/**
 * Layer toggles and the key to every mark on the map.
 *
 * A layer the server does not bundle is listed as unavailable rather than
 * hidden, so an empty map is never mistaken for "no reefs here".
 */
function MapLegend({ layers, visible, onToggle }) {
  const layerStatus = (kind) => {
    const entry = layers[kind]
    if (!entry || entry.status === "loading") return { text: "loading", loading: true }
    if (entry.status === "absent") return { text: "not bundled" }
    if (entry.status === "error") return { text: "failed to load", title: entry.message }
    const count = entry.data?.features?.length ?? 0
    const approximate = entry.approximate
    return { text: `${count} in view${approximate ? " · approximate" : ""}` }
  }

  return (
    <div className="gt-legend" aria-label="Map layers and legend">
      <fieldset className="gt-legend-group">
        <legend>Habitat layers</legend>
        {layerKinds.map((meta) => {
          const status = layerStatus(meta.kind)
          const disabled = !layers[meta.kind]?.data
          return (
            <label key={meta.kind} className={`gt-legend-item${disabled ? " is-disabled" : ""}`} title={status.title}>
              <input
                type="checkbox"
                checked={Boolean(visible[meta.kind]) && !disabled}
                disabled={disabled}
                onChange={(e) => onToggle(meta.kind, e.target.checked)}
              />
              <span
                className={`gt-swatch${meta.kind === "harbour" ? " gt-swatch--dot" : ""}`}
                style={{ "--gt-c": meta.color }}
                aria-hidden="true"
              />
              <span>{meta.label}</span>
              <span className="gt-legend-note">
                {status.loading && <Loader2 size={11} className="gt-spin" aria-hidden="true" />} {status.text}
              </span>
            </label>
          )
        })}
        <span className="gt-legend-key">
          <span className="gt-swatch gt-swatch--dashed" aria-hidden="true" /> dashed outline = approximate geometry
        </span>
      </fieldset>

      <fieldset className="gt-legend-group">
        <legend>Overlays</legend>
        <label className="gt-legend-item">
          <input type="checkbox" checked={visible.drift} onChange={(e) => onToggle("drift", e.target.checked)} />
          <span className="gt-swatch gt-swatch--cone" style={{ "--gt-c": driftColor }} aria-hidden="true" />
          <span>Drift cones (50% solid, 90% dashed)</span>
        </label>
        <label className="gt-legend-item">
          <input
            type="checkbox"
            checked={visible.particles}
            disabled={!visible.drift}
            onChange={(e) => onToggle("particles", e.target.checked)}
          />
          <span className="gt-swatch gt-swatch--dot gt-swatch--small" style={{ "--gt-c": driftColor }} aria-hidden="true" />
          <span>Simulated particles</span>
        </label>
        <label className="gt-legend-item">
          <input type="checkbox" checked={visible.route} onChange={(e) => onToggle("route", e.target.checked)} />
          <span className="gt-swatch gt-swatch--line" style={{ "--gt-c": routeColor }} aria-hidden="true" />
          <span>Recovery route (numbered stops)</span>
        </label>
        <label className="gt-legend-item">
          <input type="checkbox" checked={visible.removed} onChange={(e) => onToggle("removed", e.target.checked)} />
          <span className="gt-legend-x" aria-hidden="true">×</span>
          <span>Removed since previous survey</span>
        </label>
        <label className="gt-legend-item">
          <input type="checkbox" checked={visible.basemap} onChange={(e) => onToggle("basemap", e.target.checked)} />
          <span className="gt-swatch gt-swatch--base" aria-hidden="true" />
          <span>Basemap (OpenStreetMap, needs network)</span>
        </label>
      </fieldset>

      <div className="gt-legend-group gt-legend-tiers">
        <span className="gt-legend-title">Targets</span>
        {Object.keys(tierStyle).map((tier) => {
          const style = styleForTier(tier)
          return (
            <span key={tier} className="gt-legend-key">
              <span className={`gt-mini-marker gt-marker--${style.shape}`} style={{ "--gt-c": style.color }} aria-hidden="true" />
              {style.label}
            </span>
          )
        })}
        <span className="gt-legend-key">
          <span className="gt-mini-pulse" aria-hidden="true" /> ring = actively fishing (heuristic)
        </span>
        <span className="gt-legend-key">
          <span className="gt-prev gt-prev--inline" aria-hidden="true" /> previous position (moved)
        </span>
      </div>
    </div>
  )
}

export default MapLegend
