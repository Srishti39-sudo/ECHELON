import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import { ChatWindow } from "../assistant/components/ChatWindow";
import { settings } from "../assistant/config/settings";
import "../assistant/assistant.css";

/**
 * The grounded assistant, as a dashboard page.
 *
 * Everything it needs lives under src/assistant: its own components, its own
 * theme and copy files, and a stylesheet scoped to .dq-assistant so it cannot
 * reach the dashboard around it.
 *
 * Three pages hand work over to it, through router state, and they do it
 * differently on purpose.
 *
 * The hazard map hands over a hotspot as a survey context rather than as a
 * detection record. A detection record would be remapped onto the assistant's
 * own severity table, and the same hotspot would then carry one urgency on the
 * map and a different one beside the answer. The map owns urgency, so its
 * severity travels with the context and is displayed as given.
 *
 * The GhostTrace rescue queue hands over a target as a GhostTrace context, for
 * the same reason: the queue owns the priority. Its numbers are survey data,
 * which the backend attributes to GhostTrace and never cites as a source.
 *
 * The detections and history pages hand over a stored detection as a real
 * detection record, because it came out of this same detector and has no
 * competing severity to protect.
 */
function Assistant() {
  const { state, search } = useLocation();
  const demoContext = useDevGhostTraceDemo(search);

  return (
    <ChatWindow
      surveyContext={state?.surveyContext ?? null}
      detectionContext={state?.detectionContext ?? null}
      detectionKey={state?.detectionKey ?? null}
      detectionIsStub={state?.detectionIsStub ?? false}
      ghosttraceContext={state?.ghosttraceContext ?? demoContext}
      initialQuestion={state?.question ?? null}
    />
  );
}

/**
 * Development only: /assistant?demoContext=ghosttrace[&survey=ID&target=N].
 *
 * Router state cannot be put in a URL, so without this the GhostTrace card can
 * only be seen by clicking through the rescue queue. It fetches a real
 * ghosttrace.json from the backend and builds the context with the same
 * builder the evaluation cases mirror. `import.meta.env.DEV` is false in a
 * production build, so the whole branch, and the builder with it, is dropped.
 */
function useDevGhostTraceDemo(search) {
  const [context, setContext] = useState(null);

  useEffect(() => {
    if (!import.meta.env.DEV) return undefined;
    const params = new URLSearchParams(search);
    if (params.get("demoContext") !== "ghosttrace") return undefined;
    const surveyId = params.get("survey") || "demo-ghosttrace-mannar";
    const index = Number(params.get("target") || 0);
    let live = true;
    (async () => {
      try {
        const [{ ghosttraceContextFromTarget }, response] = await Promise.all([
          import("../assistant/lib/ghosttraceContext"),
          fetch(`${settings.apiBaseUrl}/ghosttrace/${encodeURIComponent(surveyId)}`),
        ]);
        if (!response.ok) throw new Error(`ghosttrace ${surveyId}: HTTP ${response.status}`);
        const doc = await response.json();
        const target = doc?.targets?.[index];
        if (live && target) setContext(ghosttraceContextFromTarget(doc, target));
      } catch (error) {
        console.warn("GhostTrace demo context unavailable:", error);
      }
    })();
    return () => {
      live = false;
    };
  }, [search]);

  return context;
}

export default Assistant;
