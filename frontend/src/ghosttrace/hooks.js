/**
 * The two pieces of GhostTrace state that are not plain data.
 */

import { useEffect, useReducer, useSyncExternalStore } from "react"

const QUERY = "(prefers-reduced-motion: reduce)"

function subscribe(callback) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {}
  const media = window.matchMedia(QUERY)
  media.addEventListener?.("change", callback)
  return () => media.removeEventListener?.("change", callback)
}

const readReduced = () =>
  typeof window !== "undefined" && Boolean(window.matchMedia?.(QUERY).matches)

/** True when the operator has asked the system for less motion. */
export function useReducedMotion() {
  return useSyncExternalStore(subscribe, readReduced, () => false)
}

function playbackReducer(state, action) {
  switch (action.type) {
    case "seek":
      return { index: Math.max(0, action.index), playing: false }
    case "play":
      // Pressing play at the end starts again from the beginning.
      return { index: state.index >= action.last ? 0 : state.index, playing: action.last > 0 }
    case "pause":
      return { ...state, playing: false }
    case "tick": {
      const next = Math.min(state.index + 1, action.last)
      return { index: next, playing: next < action.last }
    }
    case "reset":
      return { index: 0, playing: Boolean(action.autoplay) }
    default:
      return state
  }
}

/**
 * Drift playback over `count` snapshots.
 *
 * A reducer so the tick can stop itself at the last snapshot without a
 * set-state-in-effect: the interval dispatches, the reducer decides.
 */
export function usePlayback(count, stepMs, autoplay) {
  const [state, dispatch] = useReducer(playbackReducer, { index: 0, playing: Boolean(autoplay) })
  const last = Math.max(0, count - 1)
  const index = Math.min(state.index, last)
  const playing = state.playing && last > 0

  useEffect(() => {
    if (!playing) return undefined
    const timer = window.setInterval(() => dispatch({ type: "tick", last }), stepMs)
    return () => window.clearInterval(timer)
  }, [playing, last, stepMs])

  return {
    index,
    last,
    playing,
    play: () => dispatch({ type: "play", last }),
    pause: () => dispatch({ type: "pause" }),
    seek: (i) => dispatch({ type: "seek", index: i }),
    reset: (autoplayNext = false) => dispatch({ type: "reset", autoplay: autoplayNext }),
  }
}
