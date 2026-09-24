import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { Failed } from "../components/PageState";
import GhostTracePanel from "../ghosttrace/GhostTracePanel";
import { listSurveys } from "../survey/api";

/**
 * GhostTrace for one survey, with a picker for the others.
 *
 * Reached two ways: from the sidebar with no survey named, or from Live survey
 * with the survey that just finished. With none named it opens the most recent
 * survey that already has a GhostTrace result, then any real survey, and only
 * then a demo, for the same reason the hazard map avoids opening on synthetic
 * data by default.
 */
function GhostTracePage() {
  const { surveyId } = useParams();
  const navigate = useNavigate();
  const [surveys, setSurveys] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    listSurveys()
      .then((found) => {
        if (cancelled) return;
        setSurveys(found);
        if (!surveyId && found.length) {
          const newest = [...found].sort((a, b) =>
            String(b.processed_at ?? "").localeCompare(String(a.processed_at ?? "")));
          const pick =
            newest.find((s) => s.has_ghosttrace && !s.demo) ||
            newest.find((s) => s.has_ghosttrace) ||
            newest.find((s) => !s.demo) ||
            newest[0];
          navigate(`/ghosttrace/${encodeURIComponent(pick.survey_id)}`, { replace: true });
        }
      })
      .catch((cause) => !cancelled && setError(cause));
    return () => {
      cancelled = true;
    };
  }, [surveyId, navigate]);

  if (error) return <Failed error={error} />;

  return (
    <div className="ghosttrace-page">
      {surveys && surveys.length > 1 && (
        <label className="gt-survey-picker">
          <span>Survey</span>
          <select
            value={surveyId ?? ""}
            onChange={(e) => navigate(`/ghosttrace/${encodeURIComponent(e.target.value)}`)}
          >
            {surveys.map((s) => (
              <option key={s.survey_id} value={s.survey_id}>
                {s.title}
                {s.demo && !/synthetic|demo/i.test(s.title ?? "") ? " (synthetic demo)" : ""}
                {s.has_ghosttrace ? "" : " — not analysed yet"}
              </option>
            ))}
          </select>
        </label>
      )}

      {surveys && surveys.length === 0 && (
        <p className="muted-note">
          No survey has been processed yet. Upload a sonar log on Live survey first.
        </p>
      )}

      {surveyId && (
        <GhostTracePanel
          key={surveyId}
          surveyId={surveyId}
          title={surveys?.find((s) => s.survey_id === surveyId)?.title}
        />
      )}
    </div>
  );
}

export default GhostTracePage;
