import { useEffect, useState, useSyncExternalStore } from "react"

const MOTION_QUERY = "(prefers-reduced-motion: reduce)"

function subscribeMotion(callback) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {}
  const query = window.matchMedia(MOTION_QUERY)
  query.addEventListener("change", callback)
  return () => query.removeEventListener("change", callback)
}

function motionSnapshot() {
  return typeof window !== "undefined" && window.matchMedia
    ? window.matchMedia(MOTION_QUERY).matches
    : false
}

/** True when the operator asked the system for reduced motion. Live. */
export function usePrefersReducedMotion() {
  return useSyncExternalStore(subscribeMotion, motionSnapshot, () => false)
}

/**
 * Load one document for one key, and forget it when the key changes.
 *
 * `load` must be a stable function (a module-level API call). While the key
 * has no answer yet, `loading` is true and `data` is null, so a component
 * never shows the previous survey's document under the next survey's name.
 */
export function useSurveyDocument(load, key) {
  const [state, setState] = useState({ key: null, data: null, error: null })

  useEffect(() => {
    if (!key) return undefined
    let cancelled = false
    load(key)
      .then((data) => {
        if (!cancelled) setState({ key, data, error: null })
      })
      .catch((error) => {
        if (!cancelled) setState({ key, data: null, error })
      })
    return () => {
      cancelled = true
    }
  }, [load, key])

  const current = state.key === key
  return {
    loading: Boolean(key) && !current,
    data: current ? state.data : null,
    error: current ? state.error : null,
  }
}
