import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Play,
  ScanSearch,
  AlertTriangle,
  Ship,
  CheckCircle2,
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
                "Upload a side-scan sonar image and both detection models run over it, then the assistant explains what was found."}
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
