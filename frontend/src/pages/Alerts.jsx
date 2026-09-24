import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, MapPin, RefreshCw, ScanSearch } from "lucide-react";

import SeverityBadge from "../components/SeverityBadge";
import {
  compareSeverity,
  confidenceBasis,
  confidencePct,
  forAssistant,
  formatPct,
  isFiltered,
  isVerified,
} from "../services/severity";
import { Empty, Failed, Loading } from "../components/PageState";
import { useApi } from "../hooks/useApi";
import { fetchDetections, fetchSurveyExport, fetchSurveys } from "../services/api";
import { handoffState } from "../services/handoff";
import { hotspotContext, handoffQuestion } from "../survey/handoff";

/**
 * The things that warrant a second look, from both engines at once.
 *
 * This is the only page that reads from both. Uploaded scans and processed
 * surveys are separate subsystems with separate severity models, and merging
 * their numbers would be wrong. So they are not merged: each alert keeps its
 * own severity, says which engine it came from, and links back to the page that
 * owns it. The only thing this page decides is what counts as worth raising.
 *
 * WHAT COUNTS
 *
 *   From detections: severity high, or the contact is unidentified, and not
 *                    filtered by verification as a likely false positive.
 *   From surveys:    hotspots the engine itself placed in the critical tier.
 *
 * Unidentified is on that list deliberately. The engine assigns it "unknown"
 * rather than "high", because asserting a risk level for an object nobody has
 * identified would itself be an unsourced claim, and the whole system is built
 * to refuse those. That is a statement about evidence, not about danger. An
 * unidentified contact is handled with more caution than a named one, so it
 * belongs here even though its severity is not "high".
 *
 * A contact verification suppressed (it sits on the water column, in a rock
 * field, on a dropout) is not raised. It is counted in a note instead, so the
 * operator knows it was seen and set aside, and can find it in detections.
 */
function Alerts() {
  const navigate = useNavigate();

  const loader = useCallback(async () => {
    // Both engines are asked at once and neither is allowed to sink the page:
    // a survey that fails to parse should not hide a high-severity contact.
    const [detections, surveys] = await Promise.all([
      fetchDetections({ limit: 500 }),
      fetchSurveys().catch(() => []),
    ]);

    const exports = await Promise.all(
      surveys.map((s) =>
        fetchSurveyExport(s.survey_id)
          .then((e) => ({ survey: s, export: e }))
          .catch(() => null),
      ),
    );

    return { detections, surveys: exports.filter(Boolean) };
  }, []);

  const { data, error, loading, refreshing, reload } = useApi(loader);

  // `unidentified` is read as well as `anomaly`, because a suppressed row is
  // stored with anomaly false and would otherwise not be counted as filtered.
  const raisable = (data?.detections ?? []).filter(
    (d) => d.anomaly || d.unidentified || (d.severity || "").toLowerCase() === "high",
  );
  const detectionAlerts = raisable.filter((d) => !isFiltered(d));
  const filteredCount = raisable.length - detectionAlerts.length;

  const hotspotAlerts = (data?.surveys ?? []).flatMap(({ survey, export: doc }) =>
    (doc.hotspots ?? [])
      .filter((h) => h.severity_tier === "critical")
      .map((h) => ({ hotspot: h, survey, doc })),
  );

  const total = detectionAlerts.length + hotspotAlerts.length;

  return (
    <div className="alerts-page">

      <header className="page-head">
        <div>
          <h1>
            <Bell size={22} /> Alerts
          </h1>
          <p>
            High-severity and unidentified contacts from uploaded scans, and
            critical hotspots from processed surveys. Each keeps the severity its
            own engine gave it.
          </p>
        </div>

        <button className="icon-button" onClick={reload} disabled={refreshing}>
          <RefreshCw size={16} className={refreshing ? "spin" : ""} />
          Refresh
        </button>
      </header>

      {loading && <Loading label="Checking both engines" />}

      {error && !loading && <Failed error={error} onRetry={reload} />}

      {!loading && !error && filteredCount > 0 && (
        <p className="muted-note filtered-note">
          {filteredCount} high-severity or unidentified contact
          {filteredCount === 1
            ? " was filtered as a likely false positive"
            : "s were filtered as likely false positives"}{" "}
          and not raised.{" "}
          <button className="link-button" onClick={() => navigate("/detections")}>
            Review in detections
          </button>
        </p>
      )}

      {!loading && !error && total === 0 && (
        <Empty title="Nothing is raised">
          No uploaded contact is high severity or unidentified, and no processed
          survey has a critical hotspot. This is a real result, not an empty
          page: both engines were asked.
        </Empty>
      )}

      {!loading && !error && detectionAlerts.length > 0 && (
        <section className="alert-group">
          <h2>
            <ScanSearch size={18} /> From uploaded scans
            <span className="count">{detectionAlerts.length}</span>
          </h2>

          {[...detectionAlerts]
            .sort((a, b) => compareSeverity(a.severity, b.severity))
            .map((row) => (
              <article className="alert-card" key={row.id}>
                <SeverityBadge tier={row.severity} />

                <div className="alert-main">
                  <strong>
                    {row.object_class && row.object_class !== "unknown"
                      ? row.object_class
                      : "Unidentified contact"}
                  </strong>

                  <p>
                    {row.anomaly
                      ? "The detector saw a contact and could not confidently name it."
                      : "Classified by the detector."}{" "}
                    Confidence {formatPct(confidencePct(row))}
                    {isVerified(row) ? ", checked against the tile." : ", not verified."}
                  </p>

                  <small>
                    {row.filename ?? row.scan_id?.slice(0, 8)}
                    {row.latitude !== null && row.latitude !== undefined
                      ? ` · ${Number(row.latitude).toFixed(4)}, ${Number(row.longitude).toFixed(4)}`
                      : " · no navigation"}
                  </small>
                </div>

                <div className="alert-actions">
                  <button
                    className="link-button"
                    title={confidenceBasis(row)}
                    onClick={() =>
                      navigate("/assistant", {
                        state: handoffState({ ...row, record: forAssistant(row.record) }),
                      })
                    }
                  >
                    Ask the assistant
                  </button>
                  <button className="link-button" onClick={() => navigate("/detections")}>
                    Open in detections
                  </button>
                </div>
              </article>
            ))}
        </section>
      )}

      {!loading && !error && hotspotAlerts.length > 0 && (
        <section className="alert-group">
          <h2>
            <MapPin size={18} /> From processed surveys
            <span className="count">{hotspotAlerts.length}</span>
          </h2>

          {hotspotAlerts.map(({ hotspot, survey, doc }) => {
            const centroid = hotspot.centroid ?? {};
            const located = centroid.latitude !== null && centroid.latitude !== undefined;

            return (
              <article className="alert-card" key={`${survey.survey_id}-${hotspot.hotspot_id}`}>
                <SeverityBadge tier="high">Critical</SeverityBadge>

                <div className="alert-main">
                  <strong>
                    {hotspot.hotspot_id} · {hotspot.dominant_class}
                  </strong>

                  <p>{hotspot.recommended_action}</p>

                  <small>
                    {survey.title ?? survey.survey_id}
                    {survey.demo && " · synthetic demo data"}
                    {located
                      ? ` · ${centroid.latitude}, ${centroid.longitude}`
                      : ` · x ${centroid.global_x}, y ${centroid.global_y}, not georeferenced`}
                  </small>
                </div>

                <div className="alert-actions">
                  <button
                    className="link-button"
                    onClick={() => {
                      const context = hotspotContext(hotspot, doc);
                      navigate("/assistant", {
                        state: { surveyContext: context, question: handoffQuestion(context) },
                      });
                    }}
                  >
                    Ask the assistant
                  </button>
                  <button className="link-button" onClick={() => navigate("/map")}>
                    Open on the map
                  </button>
                </div>
              </article>
            );
          })}
        </section>
      )}

    </div>
  );
}

export default Alerts;
