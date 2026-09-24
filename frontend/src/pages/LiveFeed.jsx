import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle2,
  CircleDashed,
  Crosshair,
  FileWarning,
  Radio,
  Upload,
} from "lucide-react";

import { Failed } from "../components/PageState";
import { useApi } from "../hooks/useApi";
import { fetchHealth, fetchHistory, scanImageUrl, uploadTile } from "../services/api";
import { handoffState } from "../services/handoff";
import {
  confidenceBasis,
  confidencePct,
  forAssistant,
  isFiltered,
  verificationReasons,
} from "../services/severity";

/**
 * The analysis console.
 *
 * This page said "Real-time sonar monitoring will appear here" and there is no
 * sonar to monitor: nothing in this system is connected to a towfish, and a
 * fake waterfall scrolling past would be the kind of invented evidence the rest
 * of the project goes out of its way to avoid. So the page is named for what it
 * honestly is, a feed of analyses as they are run, and it says so at the top.
 *
 * What it does do is the whole working loop in one place: drop a tile, both
 * checkpoints run over it, the boxes are drawn on the image, and the result is
 * recorded and appears in the feed underneath.
 */
function LiveFeed() {
  const navigate = useNavigate();

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [position, setPosition] = useState({ latitude: "", longitude: "" });

  const fileInput = useRef(null);
  const previewUrl = useRef(null);

  // Health and the recent strip are both secondary to the upload control, so
  // neither is allowed to fail the page: each resolves to a usable empty value
  // instead of throwing.
  const loadPage = useCallback(
    () =>
      Promise.all([
        fetchHealth().catch(() => null),
        fetchHistory(8).catch(() => []),
      ]).then(([health, recent]) => ({ health, recent })),
    [],
  );

  const { data, reload } = useApi(loadPage);
  const health = data?.health ?? null;
  const recent = data?.recent ?? [];

  // The preview is an object URL, and one held past its image is a leak that
  // only shows up after a long session of uploads.
  useEffect(
    () => () => {
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    },
    [],
  );

  const analyse = useCallback(
    async (file) => {
      if (!file) return;
      setBusy(true);
      setError(null);
      setResult(null);

      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
      previewUrl.current = URL.createObjectURL(file);
      setPreview(previewUrl.current);

      try {
        const response = await uploadTile(file, position);
        setResult(response);
        reload();
      } catch (failure) {
        setError(failure);
        setPreview(null);
      } finally {
        setBusy(false);
      }
    },
    [reload, position],
  );

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      setDragging(false);
      analyse(event.dataTransfer.files?.[0]);
    },
    [analyse],
  );

  const uploadDisabled = health ? !health.upload_enabled : false;

  return (
    <div className="feed-page">

      <header className="page-head">
        <div>
          <h1>
            <Radio size={22} /> Analysis feed
          </h1>
          <p>
            Nothing here is connected to a live sonar. This is the analysis
            console: a tile goes in, both detection models run over it, and the
            result is recorded below.
          </p>
        </div>

        {health && (
          <div className="head-meta">
            <span className="scan-badge">
              {health.detector === "loaded"
                ? `${health.detector_models.join(" + ")} loaded`
                : health.detector}
            </span>
            <span className="scan-badge muted">{health.storage}</span>
          </div>
        )}
      </header>

      {uploadDisabled && (
        <div className="notice warn">
          <FileWarning size={18} />
          <p>
            Uploads are disabled on this backend, so no tile can be analysed.
            That happens when the detector dependencies or the model weights are
            missing, or when the upload flag is explicitly off.
          </p>
        </div>
      )}

      <section className="upload-panel">

        <div
          className={`dropzone ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !busy && fileInput.current?.click()}
        >
          <input
            ref={fileInput}
            type="file"
            accept="image/png,image/jpeg,image/tiff,image/bmp,image/webp"
            hidden
            onChange={(e) => analyse(e.target.files?.[0])}
          />

          {busy ? (
            <>
              <CircleDashed size={40} className="spin" />
              <h3>Running the detector</h3>
              <p>The first run loads the weights, so it takes longer.</p>
            </>
          ) : (
            <>
              <Upload size={40} />
              <h3>Drop a side-scan tile, or click to choose one</h3>
              <p>PNG, JPEG, TIFF, BMP or WebP, up to 16 MB.</p>
            </>
          )}
        </div>

        <div className="position-fields">
          <Crosshair size={17} />
          <label>
            Latitude
            <input
              type="number"
              step="any"
              placeholder="optional"
              value={position.latitude}
              onChange={(e) => setPosition((p) => ({ ...p, latitude: e.target.value }))}
            />
          </label>
          <label>
            Longitude
            <input
              type="number"
              step="any"
              placeholder="optional"
              value={position.longitude}
              onChange={(e) => setPosition((p) => ({ ...p, longitude: e.target.value }))}
            />
          </label>
          <p>
            A tile carries no position of its own. Left blank, the scan is stored
            without one and will not appear on the hazard map.
          </p>
        </div>

      </section>

      {error && <Failed error={error} />}

      {result && <AnalysisResult result={result} preview={preview} navigate={navigate} />}

      <section className="recent-scans">
        <h2>Recent analyses</h2>

        {recent.length === 0 ? (
          <p className="muted-note">No tile has been analysed yet.</p>
        ) : (
          <div className="scan-strip">
            {recent.map((scan) => (
              <button
                key={scan.id}
                className="scan-chip"
                onClick={() => navigate("/history", { state: { scanId: scan.id } })}
              >
                <img src={scanImageUrl(scan.id)} alt="" loading="lazy" />
                <div>
                  <strong>{scan.filename ?? scan.id.slice(0, 8)}</strong>
                  <span>
                    {scan.total_objects} contact
                    {scan.total_objects === 1 ? "" : "s"}
                    {scan.total_anomalies > 0 && ` · ${scan.total_anomalies} unidentified`}
                    {scan.total_filtered > 0 && ` · ${scan.total_filtered} filtered`}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

    </div>
  );
}

/**
 * What came back from one analysis.
 *
 * The boxes are drawn over the tile in percentages of its natural size, so the
 * overlay stays aligned at whatever width the image ends up rendering at.
 *
 * Contacts verification flagged as likely false positives are not mixed in
 * with the rest. They are listed apart, collapsed, each with its reasons, and
 * their boxes are only drawn, dashed and grey, when asked for.
 */
function AnalysisResult({ result, preview, navigate }) {
  const [size, setSize] = useState(null);
  const [showFilteredBoxes, setShowFilteredBoxes] = useState(false);
  const detections = result.detections ?? [];

  const ranked = detections
    .map((d, index) => ({ d, index }))
    .sort((a, b) => (confidencePct(b.d) ?? 0) - (confidencePct(a.d) ?? 0));
  const reported = ranked.filter(({ d }) => !isFiltered(d));
  const filtered = ranked.filter(({ d }) => isFiltered(d));
  const drawn = showFilteredBoxes ? [...filtered, ...reported] : reported;

  const ask = (d, index) =>
    navigate("/assistant", {
      state: handoffState(
        {
          id: `${result.scan_id}-${index}`,
          object_class: d.object_class,
          confidence: d.confidence,
          record: forAssistant(d),
        },
        result,
      ),
    });

  return (
    <section className="result-panel">

      <div className="panel-header">
        <div>
          <h2>
            {reported.length} contact{reported.length === 1 ? "" : "s"} in{" "}
            {result.filename}
            {filtered.length > 0 && ` · ${filtered.length} filtered`}
          </h2>
          <p>
            {result.models.join(" + ")} ·{" "}
            {result.stored ? "recorded" : "not recorded"}
            {result.store_error ? ` (${result.store_error})` : ""}
          </p>
        </div>

        {result.stored ? (
          <span className="scan-badge ok">
            <CheckCircle2 size={14} /> Saved
          </span>
        ) : (
          <span className="scan-badge warn">Not saved</span>
        )}
      </div>

      {result.stub && (
        <div className="notice warn">
          <FileWarning size={18} />
          <p>
            This came from the stub detector, not a trained model. It is a
            placeholder and is not evidence of anything.
          </p>
        </div>
      )}

      <div className="result-body">

        <div className="tile-view">
          {preview && (
            <div className="tile-frame">
              <img
                src={preview}
                alt={result.filename}
                onLoad={(e) =>
                  setSize({
                    w: e.currentTarget.naturalWidth,
                    h: e.currentTarget.naturalHeight,
                  })
                }
              />

              {size &&
                drawn.map(({ d, index }) => {
                  const [x, y, w, h] = d.bbox ?? [0, 0, 0, 0];
                  const suppressed = isFiltered(d);
                  const unidentified =
                    Boolean(d.downgraded_from) ||
                    !d.object_class ||
                    d.object_class === "unknown";
                  const pct = confidencePct(d);

                  return (
                    <span
                      key={index}
                      className={`bbox ${
                        suppressed ? "bbox-filtered" : unidentified ? "bbox-unknown" : ""
                      }`}
                      style={{
                        left: `${(x / size.w) * 100}%`,
                        top: `${(y / size.h) * 100}%`,
                        width: `${(w / size.w) * 100}%`,
                        height: `${(h / size.h) * 100}%`,
                      }}
                    >
                      <b>
                        {d.object_class || "unidentified"}{" "}
                        {typeof pct === "number" && `${pct.toFixed(1)}%`}
                        {suppressed && " · filtered"}
                      </b>
                    </span>
                  );
                })}
            </div>
          )}

          {filtered.length > 0 && (
            <label className="filter-toggle filtered-boxes-toggle">
              <input
                type="checkbox"
                checked={showFilteredBoxes}
                onChange={(e) => setShowFilteredBoxes(e.target.checked)}
              />
              Show boxes filtered as likely false positives ({filtered.length})
            </label>
          )}
        </div>

        <div className="contact-list">
          {ranked.length === 0 && (
            <p className="muted-note">
              Both models ran and neither returned anything above its confidence
              floor. That is a result, not a failure.
            </p>
          )}

          {ranked.length > 0 && reported.length === 0 && (
            <p className="muted-note">
              Every contact in this tile was filtered as a likely false positive.
              They are listed below with the evidence against each.
            </p>
          )}

          {reported.map(({ d, index }) => (
            <ContactCard key={index} d={d} onAsk={() => ask(d, index)} />
          ))}

          {filtered.length > 0 && (
            <details className="filtered-list">
              <summary>Filtered as likely false positives ({filtered.length})</summary>
              {filtered.map(({ d, index }) => (
                <ContactCard key={index} d={d} onAsk={() => ask(d, index)} />
              ))}
            </details>
          )}
        </div>

      </div>

      {result.stored && (
        <div className="result-footer">
          <button className="view-all-button" onClick={() => navigate("/detections")}>
            See it in detections
          </button>
        </div>
      )}

    </section>
  );
}

/** One contact: the 0-100 confidence first, the detector's own score beneath it. */
function ContactCard({ d, onAsk }) {
  const pct = confidencePct(d);
  const suppressed = isFiltered(d);
  const reasons = verificationReasons(d);

  return (
    <article className={`contact-card ${suppressed ? "contact-filtered" : ""}`}>
      <header>
        <strong>{d.object_class || "unidentified"}</strong>
        {typeof pct === "number" && (
          <span className="confidence">Confidence {pct.toFixed(1)}%</span>
        )}
      </header>

      <p className="raw-confidence">
        {typeof d.confidence === "number" && `detector ${d.confidence.toFixed(2)} · `}
        {confidenceBasis(d)}
      </p>

      {d.detector_model && (
        <p className="provenance">
          {d.detector_model} model, class {d.detector_class}
        </p>
      )}

      {/* The class the floors withheld, kept visible rather than
          quietly dropped. The operator should be able to see what the
          model said and why it was not reported. */}
      {d.downgraded_from && (
        <p className="withheld">
          Withheld class: {d.downgraded_from}
        </p>
      )}

      {reasons.length > 0 && (
        <ul className="verification-reasons">
          {reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}

      {d.visual_description && <p>{d.visual_description}</p>}

      <button className="link-button" onClick={onAsk}>
        Ask the assistant about this contact
      </button>
    </article>
  );
}

export default LiveFeed;
