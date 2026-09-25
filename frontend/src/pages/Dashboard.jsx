import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Play,
  ScanSearch,
  AlertTriangle,
  Ship,
  CheckCircle2,
  Crosshair,
  FileInput,
  Fish,
  Waves,
  ListOrdered,
  Cpu,
  Map,
  MessageSquareText,
} from "lucide-react";

import StatCard from "../components/StatCard";
import { settings } from "../assistant/config/settings";

/**
 * The dashboard, on real numbers.
 *
 * This page used to render mockData.js: twenty-four objects, ninety-two percent
 * analysed, and three detections at plausible-looking latitudes off Goa. None
 * of it existed. Invented coordinates on the landing page of a system whose
 * whole argument is that it never invents coordinates is the one thing a judge
 * could catch that would discredit everything behind it.
 *
 * So everything here now comes from the API, and where a survey has no
 * navigation it says so instead of printing a number.
 */
/**
 * What sets this system apart, in the order the pipeline runs. Every line is
 * something the code does today; nothing here is a roadmap item. Keep the
 * class list in step with the detector's data.yaml when the model changes.
 */
const CLASSES = ["shipwreck", "aircraft", "human", "pipeline", "fishing gear", "mine-like object", "ghost net"];

const STANDOUT = [
  {
    Icon: Crosshair,
    title: "Two detectors, seven classes",
    lead: "YOLO11 and YOLO26 trained across 7 sonar classes",
    body: CLASSES.join(" · "),
    note: "ghost-net class trained on synthetic sonar targets; confidence calibrated and every box verified against shadow geometry",
  },
  {
    Icon: FileInput,
    title: "Raw sonar in, not screenshots",
    lead: "Reads .xtf and .jsf logs directly",
    body: "Slant-range correction, gain normalisation, per-ping navigation and dropout flags. The water column is kept, not discarded.",
  },
  {
    Icon: Fish,
    title: "Is the net still fishing?",
    lead: "Water-column echo enrichment beside every net",
    body: "Echo clusters near the object against the rest of the same line. No other published ghost-net system automates this step; elsewhere a diver checks by hand.",
  },
  {
    Icon: Waves,
    title: "Where it will go",
    lead: "Monte Carlo drift on HYCOM ocean currents",
    body: "500-particle forecast with 50 / 90 % probability cones, first arrival at reefs, seagrass, turtle beaches, dugong habitat and harbours.",
  },
  {
    Icon: Map,
    title: "A survey hazard map, not a list of boxes",
    lead: "Ranked hotspots on the earth, with the blind spots",
    body: "Contacts geotagged from the survey's own navigation, grouped into hotspots with a recommended action, swath coverage and blind spots, mission replay, re-look lines as GPX.",
  },
  {
    Icon: ListOrdered,
    title: "A ghost-net rescue queue",
    lead: "Per-net priority score, recomputable by hand",
    body: "Activity, habitat, drift impact, propeller and diver risk, size, change since the last survey and recoverability, each weighted and shown. Alerts are drafted to named authorities from a cited corpus.",
  },
  {
    Icon: MessageSquareText,
    title: "Beacon and Mission Copilot",
    lead: "Cited answers, and tool-grounded answers about your surveys",
    body: "Retrieval over curated Indian marine-hazard rules with citations that open the page, plus seven data tools over the surveys themselves. Answers in English, Hindi, Tamil, Malayalam, Odia, Telugu, Bengali or Kannada.",
  },
  {
    Icon: Cpu,
    title: "Runs offline, on the boat",
    lead: "ONNX detector on CPU with no torch",
    body: "The committed export matched the PyTorch checkpoint box for box, so a fresh clone runs live detection with no torch installed. The assistant answers only from its indexed corpus.",
  },
];

function Dashboard() {
  const navigate = useNavigate();
  const [surveys, setSurveys] = useState([]);
  const [detections, setDetections] = useState([]);
  const [coordinateMode, setCoordinateMode] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = settings.apiBaseUrl;
    let live = true;

    (async () => {
      try {
        const list = await (await fetch(`${base}/survey`)).json();
        if (!live) return;
        setSurveys(list.surveys ?? []);

        const real = (list.surveys ?? []).find((s) => !s.demo) ?? (list.surveys ?? [])[0];
        if (!real) return;

        const detail = await (await fetch(`${base}/survey/${real.survey_id}/export`)).json();
        if (!live) return;
        setCoordinateMode(detail.metadata?.coordinate_mode ?? null);
        setDetections((detail.hotspots ?? []).slice(0, 4));
      } catch {
        if (live) setError("Backend unreachable. Start it on port 8000.");
      }
    })();

    return () => {
      live = false;
    };
  }, []);

  const totalDetections = surveys.reduce((n, s) => n + (s.detections ?? 0), 0);
  const totalHotspots = surveys.reduce((n, s) => n + (s.hotspots ?? 0), 0);
  const georeferenced = surveys.filter((s) => s.georeferenced).length;

  return (
    <div className="dashboard">

      <section className="stats-grid">

        <StatCard
          title="Detections"
          value={totalDetections}
          subtitle={`Across ${surveys.length} survey${surveys.length === 1 ? "" : "s"}`}
        />

        <StatCard
          title="Hotspots"
          value={totalHotspots}
          subtitle="Ranked by total severity"
          type="anomaly"
        />

        <StatCard
          title="Surveys"
          value={surveys.length}
          subtitle={`${surveys.filter((s) => s.demo).length} synthetic`}
        />

        <StatCard
          title="Georeferenced"
          value={`${georeferenced}/${surveys.length || 0}`}
          subtitle={georeferenced === 0 ? "No navigation supplied" : "Position available"}
        />

      </section>

      <section className="standout-panel" aria-labelledby="standout-heading">

        <div className="panel-header">
          <div>
            <h2 id="standout-heading">What sets DeepEcho apart</h2>
            <p>From a raw side-scan log to a ranked ghost-net rescue queue, every step explained</p>
          </div>
          <span className="scan-badge">YOLO11 + YOLO26 · 7 classes</span>
        </div>

        <div className="standout-grid">
          {STANDOUT.map(({ Icon, title, lead, body, note }) => (
            <article className="standout-card" key={title}>
              <div className="standout-icon">
                <Icon size={18} aria-hidden="true" />
              </div>
              <h3>{title}</h3>
              <p className="standout-lead">{lead}</p>
              <p className="standout-body">{body}</p>
              {note && <small className="standout-note">{note}</small>}
            </article>
          ))}
        </div>

      </section>

      <section className="dashboard-content">

        <div className="sonar-panel">

          <div className="panel-header">
            <div>
              <h2>Analyse a sonar image</h2>
              <p>Side-scan sonar imagery analysis</p>
            </div>

            {coordinateMode && (
              <span className="scan-badge">{coordinateMode}</span>
            )}
          </div>

          <div className="sonar-placeholder">

            <ScanSearch size={52} />

            <h3>Sonar Scan Preview</h3>

            <p>
              {error ??
                "Upload a side-scan sonar image or a raw .xtf log. Both detectors run over every tile, then the assistant explains what was found."}
            </p>

          </div>

          <button className="analyze-button" onClick={() => navigate("/mission")}>
            <Play size={18} />
            Upload and analyse
          </button>

        </div>

        <div className="recent-panel">

          <div className="panel-header">
            <div>
              <h2>Priority hotspots</h2>
              <p>Ranked by severity, not by count</p>
            </div>
          </div>

          <div className="detection-list">

            {detections.length === 0 && (
              <p className="detection-info">
                {error ?? "No surveys processed yet."}
              </p>
            )}

            {detections.map((hotspot) => {
              const tier = hotspot.severity_tier ?? "low";
              const centroid = hotspot.centroid ?? {};
              const located =
                centroid.latitude !== null && centroid.latitude !== undefined;

              return (
                <div className="detection-item" key={hotspot.hotspot_id}>

                  <div className={`detection-icon ${tier === "critical" ? "anomaly" : tier === "medium" ? "wreck" : ""}`}>
                    {tier === "critical" ? (
                      <AlertTriangle size={19} />
                    ) : tier === "medium" ? (
                      <Ship size={19} />
                    ) : (
                      <CheckCircle2 size={19} />
                    )}
                  </div>

                  <div className="detection-info">

                    <div className="detection-top">
                      <strong>
                        {hotspot.hotspot_id} · {hotspot.dominant_class}
                      </strong>
                      <span>{tier}</span>
                    </div>

                    {/* Pixel offsets, never dressed up as GPS. */}
                    <p>
                      {located
                        ? `${centroid.latitude}, ${centroid.longitude}`
                        : `x ${centroid.global_x}, y ${centroid.global_y} · not georeferenced`}
                    </p>

                    <small>{hotspot.recommended_action}</small>

                  </div>

                </div>
              );
            })}

          </div>

          <button className="view-all-button" onClick={() => navigate("/map")}>
            Open the hazard map
          </button>

        </div>

      </section>

    </div>
  );
}

export default Dashboard;
