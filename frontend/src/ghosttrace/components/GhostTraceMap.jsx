import { Fragment, useEffect, useMemo, useRef } from "react"
import L from "leaflet"
import {
  GeoJSON,
  MapContainer,
  Marker,
  Pane,
  Polyline,
  ScaleControl,
  TileLayer,
  useMap,
} from "react-leaflet"
import "leaflet/dist/leaflet.css"

import {
  driftColor,
  ghostConfig,
  labelForKind,
  layerByKind,
  routeColor,
  styleForTier,
} from "../config"
import {
  arrival,
  escapeHtml,
  featureKey,
  isApproximate,
  latLngOf,
  pct,
  previousLatLng,
  qualityLabel,
  stopId,
  titleCase,
} from "../format"

/* ---------------------------------------------------------------- helpers */

/** Fit the map to the survey once per document, not on every render. */
function FitToExtent({ extent, fitKey, reducedMotion }) {
  const map = useMap()
  const done = useRef(null)
  useEffect(() => {
    if (!extent || done.current === fitKey) return
    done.current = fitKey
    const [minLon, minLat, maxLon, maxLat] = extent
    const bounds = L.latLngBounds([minLat, minLon], [maxLat, maxLon])
    if (bounds.getNorthEast().equals(bounds.getSouthWest())) {
      map.setView(bounds.getCenter(), 14, { animate: false })
    } else {
      // Capped at 13, not 18. Two nets 40 m apart fit at zoom 18 into 300 m
      // of open sea, which OpenStreetMap paints as one flat blue: the map
      // looked broken when it was only zoomed in. At 13 the view is about 12 km
      // wide, so the nearest coast and the park boundary are usually on screen.
      map.fitBounds(bounds.pad(0.15), { animate: !reducedMotion, maxZoom: 13 })
    }
  }, [map, extent, fitKey, reducedMotion])
  return null
}

/** Bring the selected target into view if it is off-screen. */
function FollowSelection({ latlng, reducedMotion }) {
  const map = useMap()
  const lat = latlng?.[0]
  const lon = latlng?.[1]
  useEffect(() => {
    if (lat == null || lon == null) return
    const point = L.latLng(lat, lon)
    if (!map.getBounds().pad(-0.1).contains(point)) map.panTo(point, { animate: !reducedMotion })
  }, [map, lat, lon, reducedMotion])
  return null
}

/**
 * The wheel zooms the map only after the map has been clicked or focused, so
 * scrolling the page past the map never gets trapped in it.
 */
function WheelGuard() {
  const map = useMap()
  useEffect(() => {
    const container = map.getContainer()
    const enable = () => map.scrollWheelZoom.enable()
    const disable = () => map.scrollWheelZoom.disable()
    map.on("click focus", enable)
    container.addEventListener("mouseleave", disable)
    map.on("blur", disable)
    return () => {
      map.off("click focus", enable)
      map.off("blur", disable)
      container.removeEventListener("mouseleave", disable)
    }
  }, [map])
  return null
}

/**
 * Drift particles on a canvas renderer. Hundreds of SVG circles re-rendered by
 * React on every playback tick would stutter; one canvas layer group does not.
 */
function ParticleLayer({ points, color }) {
  const map = useMap()
  const group = useRef(null)
  const renderer = useRef(null)

  useEffect(() => {
    renderer.current = L.canvas({ padding: 0.3, pane: "gt-drift" })
    group.current = L.layerGroup().addTo(map)
    return () => {
      group.current?.remove()
      group.current = null
    }
  }, [map])

  useEffect(() => {
    const layer = group.current
    if (!layer) return
    layer.clearLayers()
    for (const [lat, lon] of points) {
      L.circleMarker([lat, lon], {
        renderer: renderer.current,
        radius: 2.2,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.55,
        interactive: false,
      }).addTo(layer)
    }
  }, [points, color])

  return null
}

function sampleEvenly(points, max) {
  if (points.length <= max) return points
  const step = points.length / max
  const out = []
  for (let i = 0; i < max; i += 1) out.push(points[Math.floor(i * step)])
  return out
}


/* ----------------------------------------------------------------- layers */

function HabitatLayer({ kind, data, highlights }) {
  const meta = layerByKind[kind] || { color: "#475569", label: kind }
  // Re-mount when the highlight state changes: react-leaflet's GeoJSON applies
  // `style` only when created, and the change is rare (a threshold crossing).
  const signature = useMemo(
    () =>
      (data?.features || [])
        .map((f) => highlights.get(featureKey(kind, f.properties?.name))?.state ?? "")
        .join("|"),
    [data, highlights, kind],
  )

  const style = (feature) => {
    const props = feature?.properties || {}
    const hit = highlights.get(featureKey(kind, props.name))
    const approximate = isApproximate(props.geometry_quality)
    return {
      color: meta.color,
      weight: hit?.state === "reached" ? 3.5 : hit ? 2.5 : 1.5,
      opacity: hit ? 1 : 0.85,
      dashArray: approximate ? "6 5" : null,
      fillColor: meta.color,
      fillOpacity: hit?.state === "reached" ? 0.38 : hit ? 0.22 : 0.1,
      className: hit?.state === "reached" ? "gt-habitat-hit" : undefined,
    }
  }

  const pointToLayer = (feature, latlng) =>
    L.circleMarker(latlng, { ...style(feature), radius: kind === "harbour" ? 6 : 7, fillOpacity: 0.75 })

  const onEachFeature = (feature, layer) => {
    const props = feature?.properties || {}
    const hit = highlights.get(featureKey(kind, props.name))
    const lines = [
      `<strong>${escapeHtml(props.name || labelForKind(kind))}</strong>`,
      `<span>${escapeHtml(labelForKind(kind))} · outline ${escapeHtml(qualityLabel(props.geometry_quality))}</span>`,
    ]
    if (props.source) lines.push(`<span class="gt-tt-dim">Source: ${escapeHtml(props.source)}</span>`)
    if (hit) {
      lines.push(
        `<span class="gt-tt-hit">Drift: ${escapeHtml(pct(hit.probability))} of particles, first ${escapeHtml(arrival(hit.first))}</span>`,
      )
    }
    layer.bindTooltip(lines.join("<br/>"), { sticky: true, className: "gt-tooltip" })
  }

  if (!data?.features?.length) return null
  return (
    <GeoJSON
      key={`${kind}-${signature}`}
      data={data}
      style={style}
      pointToLayer={pointToLayer}
      onEachFeature={onEachFeature}
      pane="gt-habitat"
    />
  )
}

/* ---------------------------------------------------------------- markers */

const iconCache = new Map()

function targetIcon(tier, selected, pulsing, rank) {
  const key = `${tier}|${selected}|${pulsing}|${rank}`
  if (iconCache.has(key)) return iconCache.get(key)
  const style = styleForTier(tier)
  const label = Number.isFinite(rank) ? String(rank) : style.glyph
  const icon = L.divIcon({
    className: "gt-marker-wrap",
    html:
      `<span class="gt-marker gt-marker--${style.shape}${selected ? " is-selected" : ""}" style="--gt-c:${style.color}">` +
      (pulsing ? '<span class="gt-marker-pulse"></span>' : "") +
      `<span class="gt-marker-core"><span>${escapeHtml(label)}</span></span></span>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  })
  iconCache.set(key, icon)
  return icon
}

const removedIcon = L.divIcon({
  className: "gt-marker-wrap",
  html: '<span class="gt-removed" aria-hidden="true">×</span>',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
})

const previousIcon = L.divIcon({
  className: "gt-marker-wrap",
  html: '<span class="gt-prev" aria-hidden="true"></span>',
  iconSize: [14, 14],
  iconAnchor: [7, 7],
})

const startIcon = L.divIcon({
  className: "gt-marker-wrap",
  html: '<span class="gt-start">S</span>',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
})

function stopIcon(n) {
  return L.divIcon({
    className: "gt-marker-wrap",
    html: `<span class="gt-stop">${n}</span>`,
    iconSize: [18, 18],
    iconAnchor: [-6, 26],
  })
}

function arrowIcon(angleDeg) {
  return L.divIcon({
    className: "gt-marker-wrap",
    html: `<span class="gt-arrowhead" style="transform:rotate(${angleDeg}deg)"></span>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  })
}

/* -------------------------------------------------------------------- map */

function GhostTraceMap({
  doc,
  targets,
  layers,
  visible,
  selectedId,
  onSelect,
  snapshot,
  trail,
  highlights,
  extent,
  reducedMotion,
}) {
  const selected = targets.find((t) => t.detection_id === selectedId) || null
  const selectedLatLng = latLngOf(selected)

  const points = useMemo(
    () => sampleEvenly((snapshot?.points || []).filter((p) => Number.isFinite(p?.[0]) && Number.isFinite(p?.[1])), ghostConfig.maxParticlesDrawn),
    [snapshot],
  )

  const byId = useMemo(() => new Map((doc.targets || []).map((t) => [String(t.detection_id), t])), [doc])

  const plan = doc.recovery_plan
  const route = useMemo(() => {
    if (!plan) return { line: [], stops: [] }
    const line = []
    const start = latLngOf(plan.start)
    if (start) line.push(start)
    const stops = []
    ;(plan.order || []).forEach((stop, i) => {
      const id = stopId(stop)
      const ll = latLngOf(byId.get(id)) || latLngOf(typeof stop === "object" ? stop : null)
      if (ll) {
        line.push(ll)
        stops.push({ id, ll, n: i + 1 })
      }
    })
    return { line, stops, start }
  }, [plan, byId])

  const center = extent ? [(extent[1] + extent[3]) / 2, (extent[0] + extent[2]) / 2] : [0, 0]
  const fitKey = `${doc.survey_id}|${doc.generated_at}`

  return (
    <div className="gt-map-shell">
      <MapContainer
        center={center}
        zoom={extent ? 12 : 2}
        scrollWheelZoom={false}
        className="gt-leaflet"
        attributionControl
        keyboard
      >
        <Pane name="gt-habitat" style={{ zIndex: 380 }} />
        <Pane name="gt-drift" style={{ zIndex: 420 }} />
        <Pane name="gt-route" style={{ zIndex: 440 }} />

        {visible.basemap && <TileLayer url={ghostConfig.basemap.url} attribution={ghostConfig.basemap.attribution} />}
        <ScaleControl position="bottomleft" imperial={false} />

        <FitToExtent extent={extent} fitKey={fitKey} reducedMotion={reducedMotion} />
        <FollowSelection latlng={selectedLatLng} reducedMotion={reducedMotion} />
        <WheelGuard />

        {Object.entries(layers).map(([kind, entry]) =>
          visible[kind] && entry?.data ? (
            <HabitatLayer key={kind} kind={kind} data={entry.data} highlights={highlights} />
          ) : null,
        )}

        {visible.drift && snapshot?.cone90 && (
          <GeoJSON
            key={`c90-${selectedId}-${snapshot.t_hours}`}
            data={snapshot.cone90}
            pane="gt-drift"
            style={{ color: driftColor, weight: 1.2, dashArray: "4 4", fillColor: driftColor, fillOpacity: 0.1 }}
            interactive={false}
          />
        )}
        {visible.drift && snapshot?.cone50 && (
          <GeoJSON
            key={`c50-${selectedId}-${snapshot.t_hours}`}
            data={snapshot.cone50}
            pane="gt-drift"
            style={{ color: driftColor, weight: 1.6, fillColor: driftColor, fillOpacity: 0.22 }}
            interactive={false}
          />
        )}
        {visible.drift && trail.length > 1 && (
          <Polyline positions={trail} pathOptions={{ color: driftColor, weight: 2, opacity: 0.7, dashArray: "2 5" }} pane="gt-drift" interactive={false} />
        )}
        {visible.drift && visible.particles && <ParticleLayer points={points} color={driftColor} />}

        {visible.route && route.line.length > 1 && (
          <Polyline positions={route.line} pane="gt-route" pathOptions={{ color: routeColor, weight: 2.5, opacity: 0.75, dashArray: "8 6" }} interactive={false} />
        )}
        {visible.route && route.start && (
          <Marker position={route.start} icon={startIcon} title={`Recovery start: ${plan.start?.name || "start"}`} keyboard={false} />
        )}
        {visible.route &&
          route.stops.map((stop) => (
            <Marker key={`stop-${stop.id}`} position={stop.ll} icon={stopIcon(stop.n)} interactive={false} keyboard={false} zIndexOffset={900} />
          ))}

        {visible.removed &&
          (doc.removed_since_previous || []).map((removed, i) => {
            const ll = latLngOf(removed)
            if (!ll) return null
            return (
              <Marker
                key={`removed-${removed.previous_detection_id ?? i}`}
                position={ll}
                icon={removedIcon}
                title={`Removed since ${removed.previous_survey_id || "previous survey"}: ${titleCase(removed.object_class)} ${removed.previous_detection_id || ""}`}
              />
            )
          })}

        {targets.map((target) => {
          const ll = latLngOf(target)
          const prev = target.change?.status === "moved" ? previousLatLng(target.change) : null
          if (!ll) return null
          const angle = prev ? (Math.atan2(-(ll[0] - prev[0]), (ll[1] - prev[1]) * Math.cos((ll[0] * Math.PI) / 180)) * 180) / Math.PI : 0
          const mid = prev ? [prev[0] + (ll[0] - prev[0]) * 0.72, prev[1] + (ll[1] - prev[1]) * 0.72] : null
          return (
            <Fragment key={target.detection_id}>
              {prev && (
                <>
                  <Polyline positions={[prev, ll]} pathOptions={{ color: "#475569", weight: 2, dashArray: "3 4" }} interactive={false} />
                  <Marker position={prev} icon={previousIcon} title={`Previous position of ${target.detection_id}`} keyboard={false} />
                  <Marker position={mid} icon={arrowIcon(angle)} interactive={false} keyboard={false} />
                </>
              )}
              <Marker
                position={ll}
                icon={targetIcon(target.priority?.tier, target.detection_id === selectedId, target.activity?.level === "high" && target.activity?.available !== false, target.priority?.rank)}
                title={`Rank ${target.priority?.rank ?? "–"}: ${titleCase(target.object_class)}, ${styleForTier(target.priority?.tier).label} priority${target.activity?.level === "high" ? ", actively fishing" : ""}`}
                zIndexOffset={target.detection_id === selectedId ? 1000 : 500}
                eventHandlers={{ click: () => onSelect(target.detection_id) }}
              />
            </Fragment>
          )
        })}
      </MapContainer>
      <p className="gt-map-hint">Click the map to zoom with the scroll wheel. Markers are keyboard-focusable; press Enter to select.</p>
    </div>
  )
}

export default GhostTraceMap
