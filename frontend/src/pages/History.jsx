import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { History as HistoryIcon, RefreshCw, Trash2 } from "lucide-react";

import SeverityBadge from "../components/SeverityBadge";
import {
  compareSeverity,
  confidenceBasis,
  confidencePct,
  forAssistant,
  formatPct,
  isFiltered,
  verificationReasons,
} from "../services/severity";
import { Empty, Failed, Loading } from "../components/PageState";
import { useApi } from "../hooks/useApi";
import { deleteScan, fetchHistory, fetchScan, scanImageUrl } from "../services/api";
import { handoffState } from "../services/handoff";

/**
 * Every scan that has been run, and what came out of each.
 *
 * The feed page links here with a scan id in router state, so "see that one
 * again" lands on the right row already open rather than on a list to search.
 *
 * Contacts verification filtered as likely false positives are counted apart
 * from the rest, and listed after them with a chip and their reasons.
 */
function History() {
  const navigate = useNavigate();
  const { state } = useLocation();

  const loader = useCallback(() => fetchHistory(), []);
  const { data, error, loading, refreshing, reload } = useApi(loader);

  const [openId, setOpenId] = useState(state?.scanId ?? null);
  const scans = data ?? [];

  const remove = async (scanId) => {
    await deleteScan(scanId);
    if (openId === scanId) setOpenId(null);
    reload();
  };

  return (
    <div className="history-page">

      <header className="page-head">
        <div>
          <h1>
            <HistoryIcon size={22} /> Scan history
          </h1>
          <p>
            Every tile that has been analysed, newest first, with the contacts
            found in it.
          </p>
        </div>

        <button className="icon-button" onClick={reload} disabled={refreshing}>
          <RefreshCw size={16} className={refreshing ? "spin" : ""} />
          Refresh
        </button>
      </header>

      {loading && <Loading label="Loading scan history" />}

      {error && !loading && <Failed error={error} onRetry={reload} />}

      {!loading && !error && scans.length === 0 && (
        <Empty
          title="Nothing has been analysed yet"
          action={
            <button className="view-all-button" onClick={() => navigate("/live")}>
              Analyse a tile
            </button>
          }
        >
          Every scan you run is recorded here, with its image and its contacts.
        </Empty>
      )}

      {!loading && !error && scans.length > 0 && (
        <div className="scan-list">
          {scans.map((scan) => (
            <ScanRow
              key={scan.id}
              scan={scan}
              open={openId === scan.id}
              onToggle={() => setOpenId(openId === scan.id ? null : scan.id)}
              onDelete={() => remove(scan.id)}
              navigate={navigate}
            />
          ))}
        </div>
      )}

    </div>
  );
}

/**
 * One scan in the list, expanding to its detections.
 *
 * The detail request is made when the row opens rather than up front, so a page
 * of fifty scans is one request instead of fifty-one.
 */
function ScanRow({ scan, open, onToggle, onDelete, navigate }) {
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState(null);

  useEffect(() => {
    if (!open || detail) return;
    let live = true;
    fetchScan(scan.id)
      .then((d) => live && setDetail(d))
      .catch((e) => live && setDetailError(e));
    return () => {
      live = false;
    };
  }, [open, detail, scan.id]);

  const when = scan.created_at ? new Date(scan.created_at) : null;
  const located = scan.latitude !== null && scan.latitude !== undefined;

  const worst = (detail?.detections ?? [])
    .filter((d) => !isFiltered(d))
    .map((d) => d.severity)
    .sort(compareSeverity)[0];

  return (
    <article className={`scan-row ${open ? "open" : ""}`}>

      <button className="scan-row-head" onClick={onToggle}>
        <img src={scanImageUrl(scan.id)} alt="" loading="lazy" />

        <div className="scan-row-main">
          <strong>{scan.filename ?? scan.id.slice(0, 8)}</strong>
          <span>
            {scan.total_objects} contact{scan.total_objects === 1 ? "" : "s"}
            {scan.total_anomalies > 0 && ` · ${scan.total_anomalies} unidentified`}
            {scan.total_filtered > 0 && ` · ${scan.total_filtered} filtered as ${scan.total_filtered === 1 ? "a likely false positive" : "likely false positives"}`}
            {located
              ? ` · ${Number(scan.latitude).toFixed(4)}, ${Number(scan.longitude).toFixed(4)}`
              : " · no navigation"}
          </span>
        </div>

        <div className="scan-row-meta">
          {scan.stub && <span className="scan-badge warn">stub</span>}
          {worst && <SeverityBadge tier={worst} />}
          {when && <time dateTime={scan.created_at}>{when.toLocaleString()}</time>}
        </div>
      </button>

      {open && (
        <div className="scan-row-body">
          {detailError && <Failed error={detailError} />}

          {!detail && !detailError && <Loading label="Loading contacts" />}

          {detail && detail.detections.length === 0 && (
            <p className="muted-note">
              Both models ran over this tile and neither returned anything above
              its confidence floor.
            </p>
          )}

          {detail && detail.detections.length > 0 && (
            <table className="data-table compact">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Confidence</th>
                  <th>Severity</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {[...detail.detections]
                  .sort(
                    (a, b) =>
                      Number(isFiltered(a)) - Number(isFiltered(b)) ||
                      compareSeverity(a.severity, b.severity),
                  )
                  .map((d) => (
                    <tr key={d.id} className={isFiltered(d) ? "row-filtered" : ""}>
                      <td>
                        {d.object_class || "unidentified"}
                        {isFiltered(d) && (
                          <small
                            className="filtered-chip"
                            title={verificationReasons(d).join("; ")}
                          >
                            filtered
                          </small>
                        )}
                        {isFiltered(d) && verificationReasons(d).length > 0 && (
                          <ul className="verification-reasons">
                            {verificationReasons(d).map((reason) => (
                              <li key={reason}>{reason}</li>
                            ))}
                          </ul>
                        )}
                      </td>
                      <td className="numeric" title={confidenceBasis(d)}>
                        {formatPct(confidencePct(d))}
                        <small className="raw-confidence">
                          {typeof d.confidence === "number"
                            ? `detector ${d.confidence.toFixed(2)}`
                            : ""}
                        </small>
                      </td>
                      <td>
                        <SeverityBadge tier={d.severity} />
                      </td>
                      <td>
                        <button
                          className="link-button"
                          onClick={() =>
                            navigate("/assistant", {
                              state: handoffState(
                                { ...d, record: forAssistant(d.record), filename: scan.filename },
                                scan,
                              ),
                            })
                          }
                        >
                          Ask
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}

          <div className="scan-row-actions">
            <span className="dim">
              {scan.models?.length ? scan.models.join(" + ") : "models not recorded"}
              {scan.bytes ? ` · ${(scan.bytes / 1024).toFixed(0)} kB` : ""}
            </span>

            <button className="danger-button" onClick={onDelete}>
              <Trash2 size={15} /> Delete scan
            </button>
          </div>
        </div>
      )}

    </article>
  );
}

export default History;
