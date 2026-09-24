import { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { HelpCircle, RefreshCw, ScanSearch } from "lucide-react";

import SeverityBadge from "../components/SeverityBadge";
import {
  compareSeverity,
  confidenceBasis,
  confidencePct,
  forAssistant,
  formatPct,
  isFiltered,
  isVerified,
  verificationReasons,
} from "../services/severity";
import { Empty, Failed, Loading } from "../components/PageState";
import { useApi } from "../hooks/useApi";
import { fetchDetections, scanImageUrl } from "../services/api";
import { handoffState } from "../services/handoff";

/**
 * Every contact the detector has ever reported, across every scan.
 *
 * The filters are sent to the backend rather than applied here, so a long
 * history is narrowed before it crosses the wire.
 *
 * Two display rules matter and both come from the engine's own policy. A
 * withheld class is shown, not hidden: the operator sees what the model said
 * and the floor that stopped it being reported. And an unidentified contact is
 * sorted and coloured above medium rather than at the bottom, because it is the
 * case this system handles most carefully.
 *
 * Contacts verification flagged as likely false positives are hidden by
 * default and counted in the filter bar. They are still in the store and one
 * tick away, each marked and with its reasons.
 */

const SEVERITIES = ["high", "unknown", "medium", "low"];

function Detections() {
  const navigate = useNavigate();
  const [severity, setSeverity] = useState([]);
  const [anomalyOnly, setAnomalyOnly] = useState(false);
  const [showFiltered, setShowFiltered] = useState(false);
  const [selected, setSelected] = useState(null);

  const loader = useCallback(
    () => fetchDetections({ severity, anomalyOnly }),
    [severity, anomalyOnly],
  );

  const { data, error, loading, refreshing, reload } = useApi(loader);

  const filteredCount = useMemo(() => (data ?? []).filter(isFiltered).length, [data]);

  const rows = useMemo(() => {
    const list = (data ?? []).filter((row) => showFiltered || !isFiltered(row));
    return [...list].sort(
      (a, b) =>
        compareSeverity(a.severity, b.severity) ||
        (confidencePct(b) ?? 0) - (confidencePct(a) ?? 0),
    );
  }, [data, showFiltered]);

  const toggle = (tier) =>
    setSeverity((current) =>
      current.includes(tier) ? current.filter((t) => t !== tier) : [...current, tier],
    );

  return (
    <div className="detections-page">

      <header className="page-head">
        <div>
          <h1>
            <ScanSearch size={22} /> Detections
          </h1>
          <p>
            Every contact reported by the detector, worst severity
            first. Severity is looked up from the class, never read out of the
            model's confidence.
          </p>
        </div>

        <button className="icon-button" onClick={reload} disabled={refreshing}>
          <RefreshCw size={16} className={refreshing ? "spin" : ""} />
          Refresh
        </button>
      </header>

      <div className="filter-bar">
        <span>Severity</span>

        {SEVERITIES.map((tier) => (
          <button
            key={tier}
            className={`filter-chip ${severity.includes(tier) ? "on" : ""}`}
            onClick={() => toggle(tier)}
          >
            <SeverityBadge tier={tier} />
          </button>
        ))}

        <label className="filter-toggle">
          <input
            type="checkbox"
            checked={anomalyOnly}
            onChange={(e) => setAnomalyOnly(e.target.checked)}
          />
          Unidentified only
        </label>

        <label className="filter-toggle">
          <input
            type="checkbox"
            checked={showFiltered}
            onChange={(e) => setShowFiltered(e.target.checked)}
          />
          Show filtered as likely false positives ({filteredCount})
        </label>

        {(severity.length > 0 || anomalyOnly) && (
          <button
            className="link-button"
            onClick={() => {
              setSeverity([]);
              setAnomalyOnly(false);
            }}
          >
            Clear
          </button>
        )}
      </div>

      {loading && <Loading label="Loading detections" />}

      {error && !loading && <Failed error={error} onRetry={reload} />}

      {!loading && !error && rows.length === 0 && (
        <Empty title="No detections yet">
          {severity.length > 0 || anomalyOnly
            ? "Nothing matches these filters. Clear them to see everything."
            : filteredCount > 0 && !showFiltered
            ? `Every stored detection was filtered as a likely false positive. Tick "Show filtered" to see all ${filteredCount}.`
            : "Analyse a tile on the feed page and the contacts appear here."}
        </Empty>
      )}

      {!loading && !error && rows.length > 0 && (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Confidence</th>
                <th>Severity</th>
                <th>Position</th>
                <th>Scan</th>
                <th />
              </tr>
            </thead>

            <tbody>
              {rows.map((row) => {
                const located = row.latitude !== null && row.latitude !== undefined;
                const withheld = row.record?.downgraded_from;
                const filtered = isFiltered(row);

                return (
                  <tr
                    key={row.id}
                    className={`${selected === row.id ? "selected" : ""} ${
                      filtered ? "row-filtered" : ""
                    }`}
                    onClick={() => setSelected(selected === row.id ? null : row.id)}
                  >
                    <td>
                      <strong>{row.object_class || "unidentified"}</strong>
                      {filtered && (
                        <small
                          className="filtered-chip"
                          title={verificationReasons(row).join("; ")}
                        >
                          filtered
                        </small>
                      )}
                      {withheld && (
                        <small className="withheld-tag" title={row.record.downgrade_note}>
                          <HelpCircle size={12} /> withheld: {withheld}
                        </small>
                      )}
                    </td>

                    <td className="numeric" title={confidenceBasis(row)}>
                      {formatPct(confidencePct(row))}
                      <small className="raw-confidence">
                        {typeof row.confidence === "number"
                          ? `detector ${row.confidence.toFixed(2)}`
                          : ""}
                        {isVerified(row) ? "" : " · not verified"}
                      </small>
                    </td>

                    <td>
                      <SeverityBadge tier={row.severity} />
                    </td>

                    {/* Never dressed up. A scan with no navigation says so
                        rather than showing a plausible-looking pair. */}
                    <td className="numeric">
                      {located
                        ? `${Number(row.latitude).toFixed(4)}, ${Number(row.longitude).toFixed(4)}`
                        : "no navigation"}
                    </td>

                    <td className="dim">{row.filename ?? row.scan_id?.slice(0, 8)}</td>

                    <td>
                      <button
                        className="link-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate("/assistant", {
                            state: handoffState({ ...row, record: forAssistant(row.record) }),
                          });
                        }}
                      >
                        Ask
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {selected && <DetectionDetail row={rows.find((r) => r.id === selected)} />}

    </div>
  );
}

/** The selected contact, shown on its tile with its box drawn over it. */
function DetectionDetail({ row }) {
  const [size, setSize] = useState(null);
  if (!row) return null;

  const record = row.record ?? {};
  const { x1 = 0, y1 = 0, x2 = 0, y2 = 0 } = row;
  const filtered = isFiltered(row);
  const reasons = verificationReasons(row);
  const unidentified =
    row.unidentified ||
    row.anomaly ||
    Boolean(record.downgraded_from) ||
    !row.object_class ||
    row.object_class === "unknown";

  return (
    <section className="detail-panel">

      <div className="panel-header">
        <div>
          <h2>{row.object_class || "unidentified"}</h2>
          <p>
            {record.detector_model
              ? `${record.detector_model} model, class ${record.detector_class}`
              : "provenance not recorded"}
          </p>
        </div>
        <SeverityBadge tier={row.severity} />
      </div>

      <div className="detail-body">

        <div className="tile-frame">
          <img
            src={scanImageUrl(row.scan_id)}
            alt=""
            onLoad={(e) =>
              setSize({
                w: e.currentTarget.naturalWidth,
                h: e.currentTarget.naturalHeight,
              })
            }
          />

          {size && (
            <span
              className={`bbox ${
                filtered ? "bbox-filtered" : row.anomaly ? "bbox-unknown" : ""
              }`}
              style={{
                left: `${(x1 / size.w) * 100}%`,
                top: `${(y1 / size.h) * 100}%`,
                width: `${((x2 - x1) / size.w) * 100}%`,
                height: `${((y2 - y1) / size.h) * 100}%`,
              }}
            />
          )}
        </div>

        <dl className="detail-facts">
          <dt>Confidence</dt>
          <dd>
            {formatPct(confidencePct(row))}
            <small className="raw-confidence">{confidenceBasis(row)}</small>
          </dd>

          <dt>Detector score</dt>
          <dd>
            {typeof row.confidence === "number"
              ? row.confidence.toFixed(4)
              : "not reported"}
          </dd>

          <dt>Verification</dt>
          <dd className={filtered ? "filtered-text" : ""}>
            {filtered
              ? "Filtered as a likely false positive. Kept, not deleted."
              : isVerified(row)
              ? "Checked against the tile; not filtered."
              : "Not verified."}
          </dd>

          {reasons.length > 0 && (
            <>
              <dt>Reasons</dt>
              <dd>
                <ul className="verification-reasons">
                  {reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </dd>
            </>
          )}

          <dt>Box</dt>
          <dd className="numeric">
            {x1.toFixed(0)}, {y1.toFixed(0)} to {x2.toFixed(0)}, {y2.toFixed(0)}
          </dd>

          <dt>Unidentified</dt>
          <dd>
            {unidentified ? "yes" : "no"}
            {unidentified && filtered ? " (not raised: filtered)" : ""}
          </dd>

          {record.downgraded_from && (
            <>
              <dt>Withheld class</dt>
              <dd className="withheld">
                {record.downgraded_from}
              </dd>
            </>
          )}

          {record.visual_description && (
            <>
              <dt>Description</dt>
              <dd>{record.visual_description}</dd>
            </>
          )}

          <dt>Sensor</dt>
          <dd>{record.sensor ?? "not recorded"}</dd>
        </dl>

      </div>
    </section>
  );
}

export default Detections;
