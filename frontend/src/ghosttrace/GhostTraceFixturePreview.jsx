import { useState } from "react"

import GhostTracePanel from "./GhostTracePanel"
import engineExample from "./fixtures/engine_example.json"
import uiExample from "./fixtures/example.json"
import uiLayers from "./fixtures/example_layers.json"

/**
 * Dev harness: GhostTrace rendered from bundled fixtures, with no backend.
 *
 * Two documents, both synthetic and labelled so on screen:
 *   - "UI example": frontend/src/ghosttrace/fixtures/example.json, a rich
 *     invented survey with a particle drift and habitat layers, for exercising
 *     every part of the interface.
 *   - "Engine example": a copy of tests/fixtures/ghosttrace_example.json, the
 *     engine's own test fixture, to prove the panel reads what the engine
 *     writes (its drift and habitat stages are fakes).
 *
 * Mount it on a dev-only route; it has no place in a production build's nav.
 */
const EXAMPLES = {
  ui: { label: "UI example (rich synthetic)", data: uiExample, layers: uiLayers },
  engine: { label: "Engine test fixture", data: engineExample, layers: null },
}

function GhostTraceFixturePreview() {
  const [which, setWhich] = useState("ui")
  const example = EXAMPLES[which]

  return (
    <div className="gt-preview">
      <div className="gt-preview-bar" role="group" aria-label="Fixture">
        <strong>Fixture preview</strong>
        <span className="gt-muted">No backend. Everything below is synthetic.</span>
        {Object.entries(EXAMPLES).map(([key, value]) => (
          <button
            key={key}
            type="button"
            className={`gt-btn${which === key ? " gt-btn--primary" : ""}`}
            aria-pressed={which === key}
            onClick={() => setWhich(key)}
          >
            {value.label}
          </button>
        ))}
      </div>
      <GhostTracePanel key={which} data={example.data} layers={example.layers} />
    </div>
  )
}

export default GhostTraceFixturePreview
