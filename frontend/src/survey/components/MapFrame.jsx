import { useEffect, useRef, useState } from "react"

/**
 * The generated map.html, embedded.
 *
 * The file is the same standalone artefact the engine writes: it carries its
 * own Leaflet and its own imagery and needs nothing from the network. Embedding
 * it rather than rebuilding the map in React means there is one map in this
 * project, not two that can disagree.
 *
 * Selecting a hotspot here posts a message into the frame, which is the only
 * channel between them. The message carries a hotspot identifier and nothing
 * else, and the map ignores any id that is not one of its own.
 *
 * The frame starts inert. A Leaflet map inside an iframe swallows the mouse
 * wheel, so the page stops scrolling the moment the pointer crosses the map and
 * appears frozen. One click activates it, clicking anywhere else releases it,
 * and the wheel belongs to whichever the operator last chose.
 */
function MapFrame({ src, focusId, title }) {
  const frame = useRef(null)
  const shell = useRef(null)
  const loaded = useRef(false)
  const [active, setActive] = useState(false)

  const focus = (id) => {
    frame.current?.contentWindow?.postMessage(
      { type: "deepecho:focus", hotspot_id: id },
      "*",
    )
  }

  useEffect(() => {
    if (!focusId || !loaded.current) return
    focus(focusId)
  }, [focusId, src])

  useEffect(() => {
    if (!active) return
    const release = (event) => {
      if (!shell.current?.contains(event.target)) setActive(false)
    }
    document.addEventListener("mousedown", release)
    return () => document.removeEventListener("mousedown", release)
  }, [active])

  return (
    <div className="sv-map-shell" ref={shell}>
      <iframe
        ref={frame}
        className="sv-map-frame"
        src={src}
        title={title}
        onLoad={() => {
          loaded.current = true
          if (focusId) focus(focusId)
        }}
      />

      {!active && (
        <button
          type="button"
          className="sv-map-veil"
          onClick={() => setActive(true)}
          aria-label="Activate the map"
        >
          <span>Click to use the map</span>
        </button>
      )}
    </div>
  )
}

export default MapFrame
