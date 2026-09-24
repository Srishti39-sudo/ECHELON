/**
 * Loading, failing and refreshing, once instead of four times.
 *
 * Every page here has the same three states and the same race to avoid: a
 * request that resolves after the component is gone, or after a newer request
 * for the same thing has already answered. Both are handled here so no page has
 * to remember to.
 *
 * Every state change happens in a promise callback rather than in the effect
 * body. That is not a style choice: setting state synchronously inside an
 * effect schedules a second render before the browser paints the first, which
 * is what React's set-state-in-effect rule is about.
 */

import { useCallback, useEffect, useState } from "react"

/**
 * Run `loader` on mount, and again whenever its identity changes.
 *
 * The caller owns when that happens: wrap the loader in useCallback with the
 * values it depends on, and this re-fetches exactly when they change.
 *
 * `loading` is true only until the first answer arrives. A later re-fetch sets
 * `refreshing` instead, so the page keeps showing what it already has rather
 * than flashing back to a spinner.
 */
export function useApi(loader) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  // Bumped by reload(). Changing it re-runs the effect without the caller
  // having to give the loader a new identity.
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    // Guards against a response arriving after this effect has been cleaned up,
    // which is both unmount and a superseded request.
    let alive = true

    loader()
      .then((result) => {
        if (!alive) return
        setData(result)
        setError(null)
      })
      .catch((failure) => {
        if (!alive) return
        setError(failure)
      })
      .finally(() => {
        if (!alive) return
        setLoading(false)
        setRefreshing(false)
      })

    return () => {
      alive = false
    }
  }, [loader, nonce])

  const reload = useCallback(() => {
    // Safe to set synchronously: this is called from an event handler, not
    // from an effect.
    setRefreshing(true)
    setNonce((n) => n + 1)
  }, [])

  return { data, error, loading, refreshing, reload }
}
