import { Fragment, useMemo } from "react"
import L from "leaflet"
import { CircleMarker, GeoJSON, Marker, Polyline, Tooltip } from "react-leaflet"

import { mapColors } from "../config"
import { escapeHtml } from "../format"

/**
 * The SVG hatch that marks a coverage gap.
 *
 * Leaflet cannot fill a path with a pattern, but a CSS fill of url(#id) can
 * reference a pattern defined anywhere in the same document, so the pattern
 * lives in this zero-sized SVG and the gap paths carry a class that uses it.
 * Rendered once per page that draws gaps.
 */
export function HatchDefs() {
  return (
    <svg className="sv-hatch-defs" width="0" height="0" aria-hidden="true" focusable="false">
      <defs>
        <pattern
          id="sv-gap-hatch"
          patternUnits="userSpaceOnUse"
          width="7"
          height="7"
          patternTransform="rotate(45)"
        >
          <rect width="7" height="7" fill={mapColors.gap} fillOpacity="0.12" />
          <line x1="0" y1="0" x2="0" y2="7" stroke={mapColors.gap} strokeWidth="2.4" />
        </pattern>
      </defs>
    </svg>
  )
}

function startIcon(number) {
  return L.divIcon({
    className: "sv-relook-start",
    html: `<span>${escapeHtml(number)}</span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  })
}

/**
 * Coverage polygons, gaps and re-look lines from GET /survey/{id}/coverage.
 *
 * Imaged seabed teal, the nadir strip grey, gaps red and hatched, the survey's
 * hull as a dashed outline, and every re-look line with its number at its
 * start. Used by the coverage panel and the printable report alike, so the two
 * cannot draw the same document differently.
 */
function CoverageLayers({ coverage, showLabels = true, interactive = true }) {
  const polygons = coverage?.polygons?.features || []
  const gaps = coverage?.gaps?.features || []
  const lines = useMemo(() => coverage?.relook_lines?.features || [], [coverage])
  const icons = useMemo(
    () => new Map(lines.map((line) => [line.properties.id, startIcon(line.properties.number)])),
    [lines],
  )

  const byKind = (kind) => polygons.find((f) => f.properties?.kind === kind)
  const hull = byKind("hull")
  const imaged = byKind("imaged")
  const nadir = byKind("nadir_blind")
  const key = `${coverage?.survey_id}|${coverage?.generated_at}`

  return (
    <>
      {hull && (
        <GeoJSON
          key={`hull-${key}`}
          data={hull}
          interactive={false}
          style={{ color: mapColors.hull, weight: 1.2, dashArray: "5 5", fill: false }}
        />
      )}
      {imaged && (
        <GeoJSON
          key={`imaged-${key}`}
          data={imaged}
          interactive={false}
          style={{ color: mapColors.imaged, weight: 1, fillColor: mapColors.imaged, fillOpacity: 0.3 }}
        />
      )}
      {nadir && (
        <GeoJSON
          key={`nadir-${key}`}
          data={nadir}
          interactive={false}
          style={{ color: mapColors.nadir, weight: 0.6, fillColor: mapColors.nadir, fillOpacity: 0.55 }}
        />
      )}
      {gaps.length > 0 && (
        <GeoJSON
          key={`gaps-${key}`}
          data={{ type: "FeatureCollection", features: gaps }}
          interactive={interactive}
          style={{ color: mapColors.gap, weight: 1.4, className: "sv-cov-gap" }}
          onEachFeature={(feature, layer) => {
            const p = feature.properties || {}
            layer.bindTooltip(
              `${escapeHtml(p.id)}: ${escapeHtml(p.reason)}, ${Math.round(p.area_m2)} m²`,
              { sticky: true },
            )
          }}
        />
      )}
      {lines.map((line) => {
        const p = line.properties
        const coords = line.geometry.coordinates.map(([lon, lat]) => [lat, lon])
        return (
          <Fragment key={p.id}>
            <Polyline
              positions={coords}
              interactive={interactive}
              pathOptions={{ color: mapColors.relook, weight: 2.6, opacity: 0.95 }}
            >
              {interactive && (
                <Tooltip sticky>
                  {p.id}: {p.reason}
                </Tooltip>
              )}
            </Polyline>
            <CircleMarker
              center={coords[coords.length - 1]}
              radius={3}
              interactive={false}
              pathOptions={{ color: mapColors.relook, weight: 1.5, fillColor: "#fff", fillOpacity: 1 }}
            />
            {showLabels && (
              <Marker
                position={coords[0]}
                icon={icons.get(p.id)}
                interactive={false}
                keyboard={false}
              />
            )}
          </Fragment>
        )
      })}
    </>
  )
}

export default CoverageLayers
