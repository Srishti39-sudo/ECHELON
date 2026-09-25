import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Cpu,
  Download,
  FileWarning,
  Layers,
  Loader2,
  MapPinOff,
  Navigation,
  PlugZap,
  Radar,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";
import {
  CircleMarker,
  MapContainer,
  Polygon,
  Polyline,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

import StatCard from "../components/StatCard";
import { Failed } from "../components/PageState";
import { useApi } from "../hooks/useApi";
import { fetchHealth } from "../services/api";
import {
  ACCEPTED_STRIP_EXTENSIONS,
  DOWNLOAD_LABELS,
  MAX_UPLOAD_BYTES,
  RAW_SONAR_EXTENSIONS,
  backendUrl,
  createJob,
  downloadUrl,
  fetchCapabilities,
  fetchJobStatus,
  jobEventsUrl,
  pipelineTitle,
} from "../services/missionApi";
import { styleForTier } from "../survey/config";
import "./survey-mission.css";

/**
 * The mission console: upload a sonar log, watch it being processed, take the
 * reports away.
 *
 * WHAT IS LIVE AND WHAT IS NOT
 * Nothing here is connected to a towfish, and the page does not pretend to
 * be. What is live is the processing of the files the operator uploads: every
 * step on the stepper, every tick of the progress bar and every marker that
 * appears is an event the backend worker wrote at the moment the real run
 * reached it (see backend/survey_job.py). There is no animation standing in
 * for work.
 *
 * PROVISIONAL, THEN FINAL
 * While the detector sweeps the tiles, each box it returns is drawn at once
 * with a dashed ring. That is a single model call on a single tile, before
 * duplicates across overlapping tiles are merged,
 * so the same object can briefly appear twice. When the engine writes its
 * export, those markers are replaced by the deduplicated, scored detections
 * the downloadable reports contain.
 *
 * POSITIONS
 * A latitude and longitude is shown only when the backend supplied one. A
 * detection without one is listed as "not georeferenced" with its pixel
 * position in the strip, and is never placed on the map. When nothing in the
 * survey is georeferenced, the map is replaced by the sonar strip itself with
 * the boxes drawn on it, which is where those detections really are.
 *
 * WHICH PIPELINE
 * The backend picks the DeepEcho engine or the team's detector per job (see
 * backend/survey_job.py select_pipeline). The page shows which one before the
 * upload, from /survey/jobs/capabilities, and which one actually ran, with the
 * reason, from the job's own events. It never infers either.
 */

/**
 * One plain sentence for the pipeline line. The backend's `reason` is an
 * audit string (which files it found, where, and why it fell back); it stays
 * available under the toggle for whoever needs it.
 */
function pipelineSentence(pipeline) {
  const weights = /\(([^)]+)\)/.exec(pipeline.label || "")?.[1] || "marine.pt";
  if (pipeline.pipeline === "teammate") {
    return `Ran ${weights} with geotag navigation, so every contact carries a position.`;
  }
  const fellBack = /image uploads need a navigation CSV/.test(pipeline.reason || "");
  return fellBack
    ? `Ran ${weights} on the image tiles. No navigation was uploaded, so positions are relative to the strip.`
    : `Ran ${weights} on the DeepEcho engine.`;
}

const STEPS = [
  { key: "ingest", label: "Ingest", note: "Decode logs and navigation" },
  { key: "tile", label: "Tile", note: "Overlapping 640 px tiles" },
  { key: "detect", label: "Detect", note: "marine.pt on every tile" },
  { key: "verify", label: "Verify", note: "Merge duplicates, shadow check" },
  { key: "geotag", label: "Geotag", note: "Latitude and longitude" },
  { key: "report", label: "Report", note: "Reports and offline map" },
  { key: "ghosttrace", label: "GhostTrace", note: "Which net to recover first" },
];

const STAGE_TO_STEP = {
  ingest: 0,
  tile: 1,
  detect: 2,
  dedup: 3,
  verify: 3,
  geo: 4,
  geotag: 4,
  hotspots: 4,
  export: 5,
  report: 5,
  ghosttrace: 6,
};

// SVG presentation attributes do not resolve CSS custom properties, so the
// map's colours are the dashboard palette's literal values.
const TIER_HEX = { critical: "#dc2626", medium: "#d97706", low: "#16a34a" };
const TIER_RANK = { critical: 3, medium: 2, low: 1 };
const PROVISIONAL_HEX = "#087f9c";
const SUPPRESSED_HEX = "#94a3b8";
const UNTIERED_HEX = "#64748b";

const MAX_LOG_LINES = 500;
const MAX_PROVISIONAL = 3000;

// -----------------------------------------------------------------------------
// helpers

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function clock(ts) {
  if (!ts) return "";
  const date = new Date(ts);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString([], { hour12: false });
}

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** 0-100 for display: the engine's confidence_pct when present, else raw x 100. */
function confidencePct(d) {
  if (isNumber(d.confidencePct)) return d.confidencePct;
  return isNumber(d.confidence) ? d.confidence * 100 : null;
}

function reasonText(reason) {
  if (typeof reason === "string") return reason;
  if (reason && typeof reason === "object") {
    return reason.reason ?? reason.message ?? reason.code ?? JSON.stringify(reason);
  }
  return String(reason);
}

/**
 * One detection, provisional or final, in the shape this page renders.
 *
 * Both sources are read defensively: fields the engine adds later
 * (confidence_pct, suppressed, verification, dimensions) are optional, and a
 * missing one is shown as missing rather than filled in.
 */
function normalize(d, provisional) {
  const lat = isNumber(d.latitude) ? d.latitude : null;
  const lon = isNumber(d.longitude) ? d.longitude : null;
  return {
    key: provisional ? `p:${d.seq ?? ""}:${d.id}` : `f:${d.id}`,
    id: d.id,
    provisional,
    cls: d.object_class ?? d.class ?? "unidentified",
    confidence: d.confidence,
    confidencePct: d.confidence_pct,
    tier: provisional ? null : d.severity_tier ?? null,
    severity: isNumber(d.severity) ? d.severity : null,
    strip: d.provenance?.strip ?? d.strip ?? null,
    bbox: Array.isArray(d.bbox_global) ? d.bbox_global : null,
    gx: d.global_x,
    gy: d.global_y,
    lat,
    lon,
    precision: d.position_precision ?? null,
    dims: d.dimensions ?? null,
    action: d.recommended_action ?? null,
    withheld: d.downgraded_from ?? null,
    suppressed: Boolean(d.suppressed),
    reasons: Array.isArray(d.verification?.reasons) ? d.verification.reasons : [],
    model: d.detector_model ?? d.provenance?.detector_model ?? null,
  };
}

function markerStyle(d, selected) {
  if (d.suppressed) {
    return {
      color: SUPPRESSED_HEX,
      fillColor: SUPPRESSED_HEX,
      fillOpacity: 0.2,
      weight: selected ? 3 : 1.5,
      dashArray: "2 4",
    };
  }
  const colour = d.provisional ? PROVISIONAL_HEX : TIER_HEX[d.tier] ?? UNTIERED_HEX;
  return {
    color: colour,
    fillColor: colour,
    fillOpacity: d.provisional ? 0.1 : 0.55,
    weight: selected ? 4 : 2,
    dashArray: d.provisional ? "5 4" : null,
  };
}

/** Rows flagged by ingest: a count, or [first, last, reason] ranges. */
function degradedRows(value) {
  if (isNumber(value)) return value;
  if (!Array.isArray(value)) return 0;
  return value.reduce(
    (n, range) => n + (Array.isArray(range) && isNumber(range[0]) && isNumber(range[1]) ? range[1] - range[0] + 1 : 0),
    0,
  );
}

function dimensionsText(dims) {
  if (!dims) return null;
  const parts = [dims.length_m, dims.width_m].filter(isNumber);
  if (parts.length < 2) return null;
  const base = `${dims.length_m.toFixed(1)} × ${dims.width_m.toFixed(1)}`;
  return isNumber(dims.height_m) ? `${base} × ${dims.height_m.toFixed(1)} m` : `${base} m`;
}

// -----------------------------------------------------------------------------
// the event stream, as state

const INITIAL_RUN = {
  lastSeq: 0,
  stage: null,
  // The furthest step reached. The engines do not emit stages in stepper
  // order (geotagging runs before verification), and a stepper that ticks a
  // step and then un-ticks it would misreport what has already happened.
  maxStep: -1,
  pipeline: null,
  stageMessage: null,
  progress: null,
  strips: {},
  provisional: [],
  finals: null,
  log: [],
  done: null,
  error: null,
};

function appendLog(log, line) {
  const next = [...log, line];
  return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next;
}

/**
 * Fold one backend event into the page's state.
 *
 * Events are numbered, and anything at or below the last number seen is
 * ignored. A reconnect that replays a few lines therefore cannot double a
 * marker or a log line.
 */
function runReducer(state, event) {
  if (!event || !isNumber(event.seq) || event.seq <= state.lastSeq) return state;
  const next = { ...state, lastSeq: event.seq };
  const line = (level, text) => ({ seq: event.seq, ts: event.ts, level, text });

  switch (event.type) {
    case "stage":
      next.stage = event.stage;
      next.stageMessage = event.message ?? null;
      next.maxStep = Math.max(state.maxStep, STAGE_TO_STEP[event.stage] ?? -1);
      if (event.pipeline && !state.pipeline) next.pipeline = { pipeline: event.pipeline };
      next.log = appendLog(state.log, line("stage", `${event.stage}: ${event.message ?? ""}`));
      break;
    case "log":
      next.log = appendLog(state.log, line(event.level ?? "info", event.message ?? ""));
      break;
    case "pipeline":
      next.pipeline = {
        pipeline: event.pipeline,
        label: event.label ?? null,
        reason: event.reason ?? null,
        requested: event.requested ?? null,
      };
      break;
    case "strip": {
      const previous = state.strips[event.strip] ?? {};
      next.strips = { ...state.strips, [event.strip]: { ...previous, ...event } };
      if (!state.strips[event.strip]) {
        next.log = appendLog(
          state.log,
          line("info", `strip ${event.strip}: ${event.width} × ${event.height} px`),
        );
      }
      break;
    }
    case "progress":
      next.progress = { done: event.tiles_done, total: event.tiles_total };
      break;
    case "detection":
      if (state.provisional.length < MAX_PROVISIONAL) {
        next.provisional = [...state.provisional, normalize(event, true)];
      }
      next.log = appendLog(
        state.log,
        line(
          "detection",
          `${event.class} ${(event.confidence * 100).toFixed(0)}% on ${event.tile ?? event.strip} (provisional)`,
        ),
      );
      break;
    case "final_detections":
      next.finals = (event.detections ?? []).map((d) => normalize(d, false));
      next.log = appendLog(
        state.log,
        line("stage", `${next.finals.length} final detection(s) after merging and scoring`),
      );
      break;
    case "done":
      next.done = event;
      if (event.pipeline) {
        next.pipeline = {
          ...(state.pipeline ?? {}),
          pipeline: event.pipeline,
          label: event.pipeline_label ?? state.pipeline?.label ?? null,
          reason: event.pipeline_reason ?? state.pipeline?.reason ?? null,
        };
      }
      next.log = appendLog(state.log, line("done", "survey complete"));
      break;
    case "error":
      next.error = event.message ?? "The survey failed without a message.";
      next.log = appendLog(state.log, line("error", next.error));
      break;
    default:
      break;
  }
  return next;
}

// -----------------------------------------------------------------------------
// page

function SurveyMission() {
  const [params, setParams] = useSearchParams();
  const jobId = params.get("job");

  return (
    <div className="msn-page">
      {jobId ? (
        <MissionRun key={jobId} jobId={jobId} onReset={() => setParams({})} />
      ) : (
        <UploadPanel onStarted={(id) => setParams({ job: id })} />
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// upload

const CORNER_KEYS = [
  ["top_left", "Top left"],
  ["top_right", "Top right"],
  ["bottom_left", "Bottom left"],
  ["bottom_right", "Bottom right"],
];

function UploadPanel({ onStarted }) {
  const [files, setFiles] = useState([]);
  const [nav, setNav] = useState(null);
  const [title, setTitle] = useState("");
  const [showCorners, setShowCorners] = useState(false);
  const [corners, setCorners] = useState({
    top_left: "",
    top_right: "",
    bottom_left: "",
    bottom_right: "",
  });
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(null);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const fileInput = useRef(null);
  const navInput = useRef(null);

  const { data: health } = useApi(useCallback(() => fetchHealth().catch(() => null), []));
  const { data: capabilities } = useApi(
    useCallback(() => fetchCapabilities().catch(() => null), []),
  );

  const addFiles = (list) => {
    const incoming = Array.from(list ?? []);
    const rejected = [];
    let csv = null;
    const accepted = [];
    for (const file of incoming) {
      const ext = extensionOf(file.name);
      if (ext === ".csv") csv = file;
      else if (ACCEPTED_STRIP_EXTENSIONS.includes(ext)) accepted.push(file);
      else rejected.push(file.name);
    }
    if (csv) setNav(csv);
    setFiles((current) => {
      const names = new Set(current.map((f) => f.name));
      return [...current, ...accepted.filter((f) => !names.has(f.name))];
    });
    setNotice(
      rejected.length
        ? `Not added (unsupported type): ${rejected.join(", ")}. Accepted: ${ACCEPTED_STRIP_EXTENSIONS.join(", ")} and a .csv navigation file.`
        : null,
    );
  };

  const rawCount = files.filter((f) => RAW_SONAR_EXTENSIONS.includes(extensionOf(f.name))).length;
  const imageCount = files.length - rawCount;
  const totalBytes = files.reduce((n, f) => n + f.size, 0) + (nav?.size ?? 0);
  const cornersFilled = CORNER_KEYS.filter(([key]) => corners[key].trim()).length;

  const parsedCorners = () => {
    if (!showCorners || cornersFilled === 0) return { value: null };
    if (cornersFilled < 4) return { error: "Give all four corners, or clear them." };
    const out = {};
    for (const [key, label] of CORNER_KEYS) {
      const parts = corners[key].split(",").map((p) => Number(p.trim()));
      if (parts.length !== 2 || parts.some((n) => !Number.isFinite(n))) {
        return { error: `${label} must be "latitude, longitude".` };
      }
      if (Math.abs(parts[0]) > 90 || Math.abs(parts[1]) > 180) {
        return { error: `${label} is out of range.` };
      }
      out[key] = parts;
    }
    return { value: out };
  };

  const willBeRelative = imageCount > 0 && !nav && !(showCorners && cornersFilled === 4);

  const submit = async () => {
    setError(null);
    const parsed = parsedCorners();
    if (parsed.error) {
      setError({ message: parsed.error, status: 400 });
      return;
    }
    if (nav && parsed.value) {
      setError({ message: "Give a navigation CSV or corners, not both.", status: 400 });
      return;
    }
    if (totalBytes > MAX_UPLOAD_BYTES) {
      setError({
        message: `The upload is ${formatBytes(totalBytes)}; the default limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`,
        status: 413,
      });
      return;
    }
    setUploading(0);
    try {
      const job = await createJob({
        files,
        nav,
        title: title.trim() || null,
        corners: parsed.value,
        onProgress: setUploading,
      });
      onStarted(job.job_id);
    } catch (failure) {
      setError(failure);
      setUploading(null);
    }
  };

  const disabledByBackend = health ? !health.upload_enabled : false;

  return (
    <>
      <header className="page-head">
        <div>
          <h1>
            <Radar size={22} /> Live survey
          </h1>
          <p>
            Upload a raw side-scan log or strip images. The backend tiles them,
            runs the seven-class YOLO11s detector (marine.pt: shipwreck, aircraft,
            human, pipeline, fishing gear, mine-like object, ghost net) over every
            tile, checks each contact
            against acoustic-shadow physics and geotags it from the log's
            navigation, and this page follows that run as it happens: stages,
            tile progress, and detections appearing on the map. When it
            finishes, the reports are ready to download.
          </p>
        </div>
        {health && (
          <div className="head-meta">
            <span className="scan-badge">
              {health.detector === "loaded"
                ? `${health.detector_models.join(" + ")} loaded · YOLO11s · 7 classes`
                : `detector ${health.detector}`}
            </span>
          </div>
        )}
      </header>

      {capabilities && <PipelineChoice capabilities={capabilities} />}

      {disabledByBackend && !capabilities?.enabled && (
        <div className="notice warn">
          <FileWarning size={18} />
          <p>
            This backend reports that it cannot run the detector, so survey jobs
            are likely disabled. A survey is never processed with placeholder
            detections.
          </p>
        </div>
      )}

      <section className="msn-upload">
        <div className="msn-upload-main">
          <div
            className={`dropzone msn-dropzone ${dragging ? "dragging" : ""} ${uploading !== null ? "busy" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (uploading === null) addFiles(e.dataTransfer.files);
            }}
            onClick={() => uploading === null && fileInput.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && uploading === null) {
                e.preventDefault();
                fileInput.current?.click();
              }
            }}
          >
            <input
              ref={fileInput}
              type="file"
              multiple
              accept={[...ACCEPTED_STRIP_EXTENSIONS, ".csv"].join(",")}
              hidden
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <Upload size={38} />
            <h3>Drop sonar logs or strip images, or click to choose</h3>
            <p>
              Raw <strong>.xtf</strong> / <strong>.jsf</strong> logs, or{" "}
              <strong>.png .jpg .tif</strong> strips. Several files make one survey.
            </p>
          </div>

          {notice && <p className="msn-inline-warn">{notice}</p>}

          {files.length > 0 && (
            <ul className="msn-file-list">
              {files.map((file) => {
                const raw = RAW_SONAR_EXTENSIONS.includes(extensionOf(file.name));
                return (
                  <li key={file.name}>
                    <span className={`msn-file-kind ${raw ? "raw" : ""}`}>
                      {raw ? "raw log" : "image"}
                    </span>
                    <span className="msn-file-name">{file.name}</span>
                    <span className="dim">{formatBytes(file.size)}</span>
                    <button
                      type="button"
                      className="msn-icon-x"
                      aria-label={`Remove ${file.name}`}
                      disabled={uploading !== null}
                      onClick={() => setFiles((list) => list.filter((f) => f !== file))}
                    >
                      <X size={14} />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <aside className="msn-upload-side">
          <label className="msn-field">
            <span>Survey title</span>
            <input
              type="text"
              maxLength={120}
              placeholder="optional, e.g. Harbour approach, line 3"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>

          <div className="msn-field">
            <span>Navigation for image strips</span>
            <div className="msn-nav-row">
              <button
                type="button"
                className="icon-button"
                onClick={() => navInput.current?.click()}
                disabled={uploading !== null}
              >
                <Navigation size={14} /> {nav ? "Replace CSV" : "Add nav CSV"}
              </button>
              {nav && (
                <span className="msn-nav-name">
                  {nav.name}
                  <button
                    type="button"
                    className="msn-icon-x"
                    aria-label="Remove navigation file"
                    onClick={() => setNav(null)}
                  >
                    <X size={13} />
                  </button>
                </span>
              )}
              <input
                ref={navInput}
                type="file"
                accept=".csv"
                hidden
                onChange={(e) => {
                  setNav(e.target.files?.[0] ?? null);
                  e.target.value = "";
                }}
              />
            </div>
            <button
              type="button"
              className="link-button"
              onClick={() => setShowCorners((v) => !v)}
            >
              {showCorners ? "Hide strip corners" : "Or give the four strip corners"}
            </button>
            {showCorners && (
              <div className="msn-corners">
                {CORNER_KEYS.map(([key, label]) => (
                  <label key={key}>
                    {label}
                    <input
                      type="text"
                      placeholder="lat, lon"
                      value={corners[key]}
                      onChange={(e) => setCorners((c) => ({ ...c, [key]: e.target.value }))}
                    />
                  </label>
                ))}
              </div>
            )}
          </div>

          <div className="msn-formats">
            <p>
              <strong>Raw XTF / JSF</strong> carry per-ping navigation, so every
              contact is geotagged from the log itself.
            </p>
            <p>
              <strong>Images</strong> carry no position. Add a navigation CSV of
              pixel-to-latitude/longitude fixes, or the four strip corners.
              Without either, results stay in relative pixel coordinates and are
              not placed on a map.
            </p>
          </div>

          {willBeRelative && files.length > 0 && (
            <p className="msn-inline-warn">
              <MapPinOff size={14} /> No navigation for {imageCount} image
              {imageCount === 1 ? "" : "s"}: those results will not be georeferenced.
            </p>
          )}

          <button
            type="button"
            className="analyze-button msn-submit"
            disabled={files.length === 0 || uploading !== null}
            onClick={submit}
          >
            {uploading !== null ? (
              <>
                <Loader2 size={17} className="spin" />
                Uploading {Math.round(uploading * 100)}%
              </>
            ) : (
              <>
                <Radar size={17} />
                Start survey
                {files.length > 0 && ` (${files.length} file${files.length === 1 ? "" : "s"}, ${formatBytes(totalBytes)})`}
              </>
            )}
          </button>
          {uploading !== null && (
            <div className="msn-bar" aria-label="Upload progress">
              <span style={{ width: `${Math.round(uploading * 100)}%` }} />
            </div>
          )}
        </aside>
      </section>

      {error &&
        (error.status === 0 ? (
          <Failed error={error} />
        ) : (
          <div className="notice msn-error">
            <AlertTriangle size={18} />
            <div>
              <strong>The upload was rejected</strong>
              <p>{error.message}</p>
            </div>
          </div>
        ))}
    </>
  );
}

// -----------------------------------------------------------------------------
// one running (or finished) job

function MissionRun({ jobId, onReset }) {
  const navigate = useNavigate();
  const [run, dispatch] = useReducer(runReducer, INITIAL_RUN);
  const [connection, setConnection] = useState({ state: "connecting", attempts: 0 });
  const [fatal, setFatal] = useState(null);
  const [status, setStatus] = useState(null);

  const [showSuppressed, setShowSuppressed] = useState(false);
  const [view, setView] = useState("auto");
  const [basemap, setBasemap] = useState(true);
  const [selectedKey, setSelectedKey] = useState(null);
  const [flyRequest, setFlyRequest] = useState(null);
  const [sort, setSort] = useState({ key: "severity", dir: "desc" });

  const lastSeq = useRef(0);

  // --- the stream -------------------------------------------------------------
  useEffect(() => {
    let source = null;
    let timer = null;
    let finished = false;
    let attempts = 0;

    const connect = () => {
      source = new EventSource(jobEventsUrl(jobId, lastSeq.current));

      source.onopen = () => {
        attempts = 0;
        setConnection({ state: "open", attempts: 0 });
      };

      source.onmessage = (message) => {
        let event;
        try {
          event = JSON.parse(message.data);
        } catch {
          return;
        }
        if (isNumber(event.seq)) lastSeq.current = Math.max(lastSeq.current, event.seq);
        dispatch(event);
        if (event.type === "done" || event.type === "error") {
          finished = true;
          source.close();
          setConnection({ state: "closed", attempts: 0 });
        }
      };

      // EventSource would retry by itself, but blindly: it cannot tell a
      // backend that is down from a job that does not exist. So the retry is
      // done here, after asking the status route which one it is, and it
      // resumes from the last event actually received.
      source.onerror = () => {
        source.close();
        if (finished) return;
        attempts += 1;
        setConnection({ state: "reconnecting", attempts });
        fetchJobStatus(jobId)
          .then(() => null)
          .catch((failure) => {
            if (failure.status && failure.status !== 0 && failure.status < 500) {
              finished = true;
              setFatal(failure);
            }
          })
          .finally(() => {
            if (finished) return;
            timer = setTimeout(connect, Math.min(15000, 1000 * 2 ** Math.min(attempts, 4)));
          });
      };
    };

    connect();
    return () => {
      finished = true;
      clearTimeout(timer);
      source?.close();
    };
  }, [jobId]);

  useEffect(() => {
    let alive = true;
    fetchJobStatus(jobId)
      .then((body) => alive && setStatus(body))
      .catch(() => null);
    return () => {
      alive = false;
    };
  }, [jobId, run.done, run.error]);

  // --- derived ------------------------------------------------------------------
  const allDetections = useMemo(() => run.finals ?? run.provisional, [run.finals, run.provisional]);
  const suppressedCount = allDetections.filter((d) => d.suppressed).length;
  const detections = useMemo(
    () => (showSuppressed ? allDetections : allDetections.filter((d) => !d.suppressed)),
    [allDetections, showSuppressed],
  );

  const sorted = useMemo(() => {
    const direction = sort.dir === "asc" ? 1 : -1;
    const value = (d) =>
      sort.key === "confidence"
        ? confidencePct(d) ?? -1
        : (TIER_RANK[d.tier] ?? 0) * 10 + (d.severity ?? (confidencePct(d) ?? 0) / 100);
    return [...detections].sort((a, b) => (value(a) - value(b)) * direction);
  }, [detections, sort]);

  const strips = useMemo(() => Object.values(run.strips), [run.strips]);
  const geoStrips = useMemo(
    () => strips.filter((s) => (Array.isArray(s.track) && s.track.length > 1) || s.footprint),
    [strips],
  );
  const geoDetections = useMemo(() => detections.filter((d) => d.lat !== null && d.lon !== null), [detections]);
  const anyGeoDetection = allDetections.some((d) => d.lat !== null);
  const hasGeo = geoStrips.length > 0 || anyGeoDetection;
  // The map is the default only when it can show the detections themselves.
  // A towfish track with every contact unplaced would be a map of the wrong
  // thing, so in that case the strip, where the contacts really are, leads.
  const autoView = anyGeoDetection || (geoStrips.length > 0 && allDetections.length === 0) ? "map" : "strip";
  const effectiveView = view === "auto" ? autoView : view;
  const unplacedCount = detections.filter((d) => d.lat === null || d.lon === null).length;
  const synthetic = strips.some((s) => s.synthetic);

  const finished = Boolean(run.done || run.error);
  const state = run.error ? "failed" : run.done ? "done" : run.lastSeq > 0 ? "running" : "queued";

  const stepIndex = run.done ? STEPS.length : run.maxStep;
  // What ran comes from the job's events; job.json carries the same record for
  // a page opened after the events it would have come from.
  const pipeline = run.pipeline?.pipeline
    ? run.pipeline
    : status?.pipeline
      ? { pipeline: status.pipeline, label: status.pipeline_label, reason: status.pipeline_reason }
      : null;
  const notProcessed = run.done?.inputs_not_processed ?? status?.inputs_not_processed ?? [];
  const percent = run.progress?.total ? Math.round((run.progress.done / run.progress.total) * 100) : 0;

  const summary = run.done?.summary;
  const title = run.done?.title ?? status?.title ?? jobId;

  const select = (d) => {
    setSelectedKey(d.key);
    if (d.lat !== null && d.lon !== null) {
      // A fresh object each click, so clicking the same row again flies again.
      setFlyRequest({ key: d.key, lat: d.lat, lon: d.lon });
      if (effectiveView !== "map") setView("map");
    } else if (strips.length) {
      setView("strip");
    }
  };

  const toggleSort = (key) =>
    setSort((current) =>
      current.key === key ? { key, dir: current.dir === "desc" ? "asc" : "desc" } : { key, dir: "desc" },
    );

  if (fatal) {
    return (
      <>
        <Failed error={fatal} />
        <div className="msn-center">
          <button type="button" className="view-all-button" onClick={onReset}>
            Start a new survey
          </button>
        </div>
      </>
    );
  }

  return (
    <>
      <header className="page-head">
        <div>
          <h1>
            <Radar size={22} /> {title}
          </h1>
          <p>
            Survey <code>{jobId}</code>
            {status?.inputs?.length ? ` · ${status.inputs.join(", ")}` : ""}
            {status?.created_at ? ` · started ${clock(status.created_at)}` : ""}
          </p>
        </div>
        <div className="head-meta">
          {pipeline && (
            <span className={`scan-badge msn-pipeline ${pipeline.pipeline}`} title={pipeline.reason ?? ""}>
              <Cpu size={13} /> {pipelineTitle(pipeline.label, pipeline.pipeline)}
            </span>
          )}
          <ConnectionBadge state={state} connection={connection} />
          <button type="button" className="icon-button" onClick={onReset}>
            <RotateCcw size={14} /> New survey
          </button>
        </div>
      </header>

      {pipeline?.label && (
        <details className="msn-pipeline-reason">
          <summary>{pipelineSentence(pipeline)}</summary>
          {pipeline.reason && <p>{pipeline.reason}</p>}
        </details>
      )}

      {notProcessed.length > 0 && (
        <div className="notice warn">
          <FileWarning size={18} />
          <div>
            <strong>
              {notProcessed.length} uploaded file{notProcessed.length === 1 ? " was" : "s were"} not processed
            </strong>
            <ul className="msn-reasons">
              {notProcessed.map((item) => (
                <li key={item.file}>
                  <code>{item.file}</code> {item.reason}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {synthetic && (
        <div className="notice warn">
          <FileWarning size={18} />
          <p>
            Synthetic sonar log. The ingest module found this file marked as
            generated, not recorded at sea: no sonar was deployed and nothing
            detected in it is evidence of a real object.
          </p>
        </div>
      )}

      {run.done?.demo && (
        <div className="notice warn">
          <FileWarning size={18} />
          <p>Synthetic demo data. Nothing in this survey is evidence of anything.</p>
        </div>
      )}

      {run.error && (
        <div className="notice msn-error">
          <AlertTriangle size={18} />
          <div>
            <strong>The survey failed</strong>
            <p>{run.error}</p>
          </div>
        </div>
      )}

      {/* --- stepper and progress --------------------------------------------- */}
      <section className="msn-panel msn-progress-panel">
        <ol className="msn-stepper">
          {STEPS.map((step, index) => {
            const stepState =
              index < stepIndex
                ? "complete"
                : index === stepIndex
                  ? run.error
                    ? "failed"
                    : "active"
                  : "pending";
            return (
              <li key={step.key} className={`msn-step ${stepState}`}>
                <span className="msn-step-dot">
                  {stepState === "complete" ? (
                    <CheckCircle2 size={16} />
                  ) : stepState === "active" ? (
                    <Loader2 size={16} className="spin" />
                  ) : stepState === "failed" ? (
                    <AlertTriangle size={16} />
                  ) : (
                    <CircleDashed size={16} />
                  )}
                </span>
                <span className="msn-step-text">
                  <strong>{step.label}</strong>
                  <small>{step.note}</small>
                </span>
              </li>
            );
          })}
        </ol>

        <div className="msn-progress-row">
          <div className="msn-bar" aria-label="Tile progress">
            <span
              className={finished ? "" : "live"}
              style={{ width: `${run.done ? 100 : percent}%` }}
            />
          </div>
          <span className="msn-progress-label numeric">
            {run.progress
              ? `${run.progress.done} / ${run.progress.total} tiles`
              : state === "queued"
                ? "waiting for the worker"
                : pipeline?.pipeline === "teammate"
                  ? "the team detector tiles the strip itself; no per-tile progress"
                  : "tiles not started"}
          </span>
        </div>
        <p className="msn-stage-message">
          {run.error
            ? "Stopped."
            : run.done
              ? "Complete. The reports below are the final record."
              : run.stageMessage ?? "Starting…"}
        </p>
      </section>

      {summary && (
        <section className="stats-grid msn-stats">
          <StatCard
            title="Hazards reported"
            value={summary.total_deduplicated_detections ?? 0}
            subtitle={`${summary.total_raw_detections ?? 0} raw boxes before merging`}
          />
          <StatCard
            title="Critical"
            value={summary.detections_by_tier?.critical ?? 0}
            type={summary.detections_by_tier?.critical ? "anomaly" : undefined}
            subtitle={`${summary.detections_by_tier?.medium ?? 0} medium · ${summary.detections_by_tier?.low ?? 0} low`}
          />
          <StatCard
            title="Coordinates"
            value={summary.georeferenced ? "Geo-referenced" : "Relative"}
            subtitle={summary.georeferenced ? "Latitude / longitude attached" : "No position on any detection: pixels only"}
          />
          <StatCard
            title="Filtered"
            value={summary.suppressed_detections ?? suppressedCount}
            subtitle="Likely false positives, kept and labelled"
          />
        </section>
      )}

      {/* --- map / strip and live log ----------------------------------------- */}
      <section className="msn-workspace">
        <div className="msn-panel msn-map-panel">
          <div className="msn-map-toolbar">
            <div className="msn-segmented" role="tablist">
              <button
                type="button"
                className={effectiveView === "map" ? "on" : ""}
                onClick={() => setView("map")}
              >
                Map
              </button>
              <button
                type="button"
                className={effectiveView === "strip" ? "on" : ""}
                onClick={() => setView("strip")}
                disabled={!strips.length}
              >
                Sonar strip
              </button>
            </div>
            <div className="msn-toolbar-right">
              {effectiveView === "map" && hasGeo && (
                <label className="filter-toggle">
                  <input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} />
                  <Layers size={13} /> Street basemap (needs network)
                </label>
              )}
              <label className="filter-toggle">
                <input
                  type="checkbox"
                  checked={showSuppressed}
                  onChange={(e) => setShowSuppressed(e.target.checked)}
                />
                Show filtered false positives{suppressedCount ? ` (${suppressedCount})` : ""}
              </label>
            </div>
          </div>

          {effectiveView === "map" ? (
            hasGeo ? (
              <>
              {unplacedCount > 0 && (
                <p className="msn-inline-warn">
                  <MapPinOff size={14} /> {unplacedCount} detection
                  {unplacedCount === 1 ? " has" : "s have"} no position and {unplacedCount === 1 ? "is" : "are"} not
                  drawn here. See the Sonar strip view and the table.
                </p>
              )}
              <SurveyMap
                strips={geoStrips}
                detections={geoDetections}
                finals={run.finals}
                basemap={basemap}
                selectedKey={selectedKey}
                flyRequest={flyRequest}
                onSelect={(d) => setSelectedKey(d.key)}
              />
              </>
            ) : (
              <div className="msn-map-empty">
                <MapPinOff size={30} />
                <h3>{finished || strips.length ? "Not georeferenced" : "Waiting for the first strip"}</h3>
                <p>
                  {finished || strips.length
                    ? "Nothing in this survey carries navigation, so there is no honest place to put it on a map. Detections are shown on the sonar strip instead, at their pixel positions."
                    : "Strip tracks and detections appear here as the worker reaches them."}
                </p>
              </div>
            )
          ) : (
            <StripView
              strips={strips}
              detections={detections}
              selectedKey={selectedKey}
              onSelect={(d) => setSelectedKey(d.key)}
              georeferenced={anyGeoDetection}
            />
          )}

          <MapLegend provisional={!run.finals} />
        </div>

        <LiveLog lines={run.log} />
      </section>

      {/* --- detections table --------------------------------------------------- */}
      <section className="msn-panel">
        <div className="panel-header">
          <div>
            <h2>
              {run.finals ? "Detections" : "Detections so far"} ({detections.length})
            </h2>
            <p>
              {run.finals
                ? "Deduplicated and scored. These are the rows in the downloadable reports."
                : "Provisional: single detector calls per tile, before merging. Replaced by the final list when the export is written."}
            </p>
          </div>
        </div>

        {sorted.length === 0 ? (
          <p className="muted-note">
            {finished
              ? "The detectors ran and returned nothing above their confidence thresholds. That is a result, not a failure."
              : "No detections yet."}
          </p>
        ) : (
          <div className="table-wrap msn-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Contact</th>
                  <th>
                    <button type="button" className="msn-sort" onClick={() => toggleSort("confidence")}>
                      Confidence {sort.key === "confidence" ? (sort.dir === "desc" ? "↓" : "↑") : ""}
                    </button>
                  </th>
                  <th>
                    <button type="button" className="msn-sort" onClick={() => toggleSort("severity")}>
                      Severity {sort.key === "severity" ? (sort.dir === "desc" ? "↓" : "↑") : ""}
                    </button>
                  </th>
                  <th>Position</th>
                  <th>Size</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((d) => (
                  <DetectionRow
                    key={d.key}
                    d={d}
                    selected={d.key === selectedKey}
                    onSelect={() => select(d)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* --- downloads ------------------------------------------------------------ */}
      {run.done && (
        <section className="msn-panel">
          <div className="panel-header">
            <div>
              <h2>Reports</h2>
              <p>Written by the engine for this survey. Only files that exist are offered.</p>
            </div>
            <button
              type="button"
              className="icon-button"
              onClick={() => navigate("/map", { state: { surveyId: run.done.survey_id ?? jobId } })}
            >
              Open in Survey Hazard Map
            </button>
            {(run.done.downloads ?? []).includes("ghosttrace.json") && (
              <button
                type="button"
                className="icon-button"
                onClick={() =>
                  navigate(`/ghosttrace/${encodeURIComponent(run.done.survey_id ?? jobId)}`)
                }
              >
                Open GhostTrace
              </button>
            )}
          </div>
          <div className="msn-downloads">
            {(run.done.downloads ?? []).map((name) => {
              const meta = DOWNLOAD_LABELS[name] ?? { label: name, note: "" };
              return (
                <a
                  key={name}
                  className="msn-download"
                  href={downloadUrl(run.done.survey_id ?? jobId, name)}
                  download
                >
                  <Download size={17} />
                  <span>
                    <strong>{meta.label}</strong>
                    <small>{meta.note}</small>
                  </span>
                </a>
              );
            })}
          </div>
        </section>
      )}
    </>
  );
}

// -----------------------------------------------------------------------------
// pieces

/** Which pipeline the next upload runs, and what the other one is missing. */
function PipelineChoice({ capabilities }) {
  const { pipelines = {}, selected, label, reason, error } = capabilities;
  const team = pipelines.teammate ?? {};
  const engine = pipelines.echelon ?? {};
  return (
    <section className={`msn-panel msn-pipeline-panel ${error ? "failed" : ""}`}>
      <div className="msn-pipeline-head">
        <span className={`scan-badge msn-pipeline ${selected ?? ""}`}>
          <Cpu size={13} /> {pipelineTitle(label, selected)}
        </span>
        <span className="dim">
          chosen by {capabilities.env ?? "DEEPECHO_PIPELINE"}={capabilities.requested}
        </span>
      </div>
      <p className="msn-pipeline-reason">{error ? `Uploads will fail: ${error}` : reason}</p>
      <ul className="msn-pipeline-list">
        <li>
          <strong>DeepEcho engine</strong>{" "}
          <span className={engine.available ? "ok" : "missing"}>
            {engine.available ? "available" : "unavailable"}
          </span>
          <small>{engine.reason}</small>
        </li>
        <li>
          <strong>Team detector</strong>{" "}
          <span className={team.available ? "ok" : "missing"}>
            {team.available ? "available" : "unavailable"}
          </span>
          <small>
            {team.available
              ? `${team.label}${team.shadow ? " + shadow check" : ""}${team.anomaly ? " + anomaly channel" : ""}`
              : team.missing?.length
                ? `waiting for ${team.missing.join(", ")} in ${team.modules}`
                : team.reason}
          </small>
        </li>
      </ul>
    </section>
  );
}

function ConnectionBadge({ state, connection }) {
  if (state === "done") {
    return (
      <span className="scan-badge ok">
        <CheckCircle2 size={14} /> Complete
      </span>
    );
  }
  if (state === "failed") return <span className="scan-badge msn-badge-failed">Failed</span>;
  if (connection.state === "reconnecting") {
    return (
      <span className="scan-badge warn">
        <PlugZap size={13} /> Connection lost, retrying ({connection.attempts})
      </span>
    );
  }
  if (connection.state === "open") {
    return (
      <span className="scan-badge msn-badge-live">
        <span className="msn-pulse" /> Processing live
      </span>
    );
  }
  return <span className="scan-badge muted">Connecting…</span>;
}

function DetectionFacts({ d }) {
  const pct = confidencePct(d);
  const dims = dimensionsText(d.dims);
  return (
    <dl className="msn-facts">
      <dt>Class</dt>
      <dd>
        {d.cls}
        {d.withheld ? ` (model said ${d.withheld}; below its floor)` : ""}
      </dd>
      <dt>Confidence</dt>
      <dd>{pct === null ? "not reported" : `${pct.toFixed(1)}%`}</dd>
      <dt>Position</dt>
      <dd>
        {d.lat !== null && d.lon !== null
          ? `${d.lat.toFixed(6)}, ${d.lon.toFixed(6)}${d.precision ? ` (${d.precision})` : ""}`
          : "not georeferenced"}
      </dd>
      {dims && (
        <>
          <dt>Dimensions</dt>
          <dd>
            {dims}
            {d.dims?.basis ? <small className="msn-basis">{d.dims.basis}</small> : null}
          </dd>
        </>
      )}
      {d.tier && (
        <>
          <dt>Severity</dt>
          <dd>
            {styleForTier(d.tier).label}
            {d.severity !== null ? ` (${d.severity.toFixed(2)})` : ""}
          </dd>
        </>
      )}
      {d.action && (
        <>
          <dt>Action</dt>
          <dd>{d.action}</dd>
        </>
      )}
      {d.provisional && (
        <>
          <dt>Status</dt>
          <dd>Provisional, before merging and scoring</dd>
        </>
      )}
      {d.suppressed && (
        <>
          <dt>Filtered</dt>
          <dd>
            {d.reasons.length ? (
              <ul className="msn-reasons">
                {d.reasons.map((reason, i) => (
                  <li key={i}>{reasonText(reason)}</li>
                ))}
              </ul>
            ) : (
              "Marked as a likely false positive"
            )}
          </dd>
        </>
      )}
    </dl>
  );
}

function DetectionRow({ d, selected, onSelect }) {
  const pct = confidencePct(d);
  const tierStyle = d.tier ? styleForTier(d.tier) : null;
  const dims = dimensionsText(d.dims);
  return (
    <tr className={`${selected ? "selected" : ""} ${d.suppressed ? "msn-row-suppressed" : ""}`} onClick={onSelect}>
      <td>
        <strong>{d.cls}</strong>
        <div className="dim">
          {d.id}
          {d.strip ? ` · ${d.strip}` : ""}
        </div>
        <div className="msn-tags">
          {d.provisional && <span className="msn-tag provisional">provisional</span>}
          {d.suppressed && <span className="msn-tag suppressed">filtered</span>}
          {d.withheld && <span className="msn-tag withheld">class withheld</span>}
        </div>
      </td>
      <td className="numeric">{pct === null ? "—" : `${pct.toFixed(1)}%`}</td>
      <td>
        {tierStyle ? (
          <span className="severity-badge" style={{ color: tierStyle.color, background: tierStyle.tint }}>
            {tierStyle.label}
          </span>
        ) : (
          <span className="dim">{d.provisional ? "after scoring" : "—"}</span>
        )}
      </td>
      <td className="numeric">
        {d.lat !== null && d.lon !== null ? (
          <>
            {d.lat.toFixed(6)}, {d.lon.toFixed(6)}
          </>
        ) : (
          <>
            <span className="msn-not-geo">not georeferenced</span>
            {isNumber(d.gx) && (
              <div className="dim">
                x {Math.round(d.gx)}, y {Math.round(d.gy)} px
              </div>
            )}
          </>
        )}
      </td>
      <td className="numeric">
        {dims ?? (d.bbox ? <span className="dim">{Math.round(d.bbox[2] - d.bbox[0])} × {Math.round(d.bbox[3] - d.bbox[1])} px</span> : "—")}
      </td>
      <td>{d.action ?? <span className="dim">—</span>}</td>
    </tr>
  );
}

function LiveLog({ lines }) {
  const box = useRef(null);
  const [follow, setFollow] = useState(true);

  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines, follow]);

  return (
    <div className="msn-panel msn-log-panel">
      <div className="msn-log-head">
        <h2>Worker log</h2>
        <label className="filter-toggle">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          Follow
        </label>
      </div>
      <ol className="msn-log" ref={box}>
        {lines.length === 0 && <li className="dim">Waiting for the first event…</li>}
        {lines.map((line) => (
          <li key={`${line.seq}-${line.level}`} className={`lvl-${line.level}`}>
            <time>{clock(line.ts)}</time>
            <span>{line.text}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function MapLegend({ provisional }) {
  return (
    <div className="msn-legend">
      {provisional && (
        <span>
          <i className="ring provisional" /> provisional
        </span>
      )}
      <span>
        <i style={{ background: TIER_HEX.critical }} /> critical
      </span>
      <span>
        <i style={{ background: TIER_HEX.medium }} /> medium
      </span>
      <span>
        <i style={{ background: TIER_HEX.low }} /> low
      </span>
      <span>
        <i className="ring suppressed" /> filtered
      </span>
      <span>
        <i className="line" /> towfish track / strip footprint
      </span>
    </div>
  );
}

// --- map ----------------------------------------------------------------------

function FitToData({ points, signature }) {
  const map = useMap();
  const fitted = useRef(null);

  useEffect(() => {
    if (!points.length || fitted.current === signature) return;
    fitted.current = signature;
    if (points.length === 1) map.setView(points[0], 17);
    else map.fitBounds(points, { padding: [36, 36], maxZoom: 18 });
  }, [map, points, signature]);

  return null;
}

function FlyTo({ request, markers }) {
  const map = useMap();

  useEffect(() => {
    if (!request) return undefined;
    map.flyTo([request.lat, request.lon], Math.max(map.getZoom(), 17), { duration: 0.6 });
    const timer = setTimeout(() => markers.current.get(request.key)?.openPopup(), 700);
    return () => clearTimeout(timer);
  }, [map, request, markers]);

  return null;
}

function SurveyMap({ strips, detections, finals, basemap, selectedKey, flyRequest, onSelect }) {
  const markers = useRef(new Map());

  const points = useMemo(() => {
    const out = [];
    for (const strip of strips) {
      if (strip.footprint) out.push(...strip.footprint);
      else if (strip.track) out.push(...strip.track);
    }
    if (!out.length) for (const d of detections) out.push([d.lat, d.lon]);
    return out;
  }, [strips, detections]);

  // Refit when a strip gains geometry, when the first positions arrive, and
  // once more when the final list replaces the provisional one. Not on every
  // detection, which would snatch the map away from an operator panning it.
  const signature = `${strips.length}|${points.length > 0}|${finals ? "final" : "provisional"}`;
  const center = points[0] ?? [0, 0];

  return (
    <div className="msn-map">
      <MapContainer center={center} zoom={15} scrollWheelZoom className="msn-leaflet">
        {basemap && (
          <TileLayer
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap contributors"
            maxZoom={19}
          />
        )}
        {strips.map((strip) => (
          <StripGeometry key={strip.strip} strip={strip} />
        ))}
        {detections.map((d) => (
          <CircleMarker
            key={d.key}
            center={[d.lat, d.lon]}
            radius={d.provisional ? 9 : 8}
            pathOptions={markerStyle(d, d.key === selectedKey)}
            eventHandlers={{ click: () => onSelect(d) }}
            ref={(layer) => {
              if (layer) markers.current.set(d.key, layer);
              else markers.current.delete(d.key);
            }}
          >
            <Popup>
              <DetectionFacts d={d} />
            </Popup>
          </CircleMarker>
        ))}
        <FitToData points={points} signature={signature} />
        <FlyTo request={flyRequest} markers={markers} />
      </MapContainer>
    </div>
  );
}

function StripGeometry({ strip }) {
  return (
    <>
      {strip.footprint && (
        <Polygon
          positions={strip.footprint}
          pathOptions={{ color: "#07677f", weight: 1, fillColor: "#087f9c", fillOpacity: 0.06 }}
        />
      )}
      {Array.isArray(strip.track) && strip.track.length > 1 && (
        <Polyline positions={strip.track} pathOptions={{ color: "#07677f", weight: 3, opacity: 0.85 }} />
      )}
    </>
  );
}

// --- strip view -------------------------------------------------------------------

function StripView({ strips, detections, selectedKey, onSelect, georeferenced }) {
  const [chosen, setChosen] = useState(null);
  const current = strips.find((s) => s.strip === chosen) ?? strips[0] ?? null;

  if (!current) {
    return (
      <div className="msn-map-empty">
        <CircleDashed size={30} className="spin" />
        <h3>Waiting for the first strip</h3>
        <p>The strip appears as soon as the worker has read it.</p>
      </div>
    );
  }

  const onStrip = detections.filter((d) => d.strip === current.strip && d.bbox);
  const canOverlay = isNumber(current.width) && isNumber(current.height) && current.width > 0;

  return (
    <div className="msn-strip">
      {!georeferenced && (
        <div className="notice warn msn-strip-notice">
          <MapPinOff size={17} />
          <p>
            Not georeferenced. No detection in this survey carries a latitude
            or longitude, so they are shown where they are in the sonar image,
            in pixels, and no position is given for them.
          </p>
        </div>
      )}
      {strips.length > 1 && (
        <div className="msn-strip-tabs">
          {strips.map((s) => (
            <button
              key={s.strip}
              type="button"
              className={s.strip === current.strip ? "on" : ""}
              onClick={() => setChosen(s.strip)}
            >
              {s.strip}
            </button>
          ))}
        </div>
      )}
      <div className="msn-strip-meta dim">
        {current.width} × {current.height} px
        {current.source ? ` · from ${current.source}` : ""}
        {isNumber(current.m_per_px_across) ? ` · ${current.m_per_px_across.toFixed(3)} m/px across` : ""}
        {degradedRows(current.degraded_rows) > 0 ? ` · ${degradedRows(current.degraded_rows)} degraded rows (kept, flagged)` : ""}
      </div>
      <div className="msn-strip-scroll">
        {current.image_url ? (
          <div className="tile-frame msn-strip-frame">
            <img src={backendUrl(current.image_url)} alt={`Sonar strip ${current.strip}`} />
            {canOverlay &&
              onStrip.map((d) => {
                const [x1, y1, x2, y2] = d.bbox;
                const style = markerStyle(d, d.key === selectedKey);
                return (
                  <button
                    type="button"
                    key={d.key}
                    className={`msn-box ${d.provisional ? "provisional" : ""} ${d.suppressed ? "suppressed" : ""} ${d.key === selectedKey ? "selected" : ""}`}
                    style={{
                      left: `${(x1 / current.width) * 100}%`,
                      top: `${(y1 / current.height) * 100}%`,
                      width: `${((x2 - x1) / current.width) * 100}%`,
                      height: `${((y2 - y1) / current.height) * 100}%`,
                      borderColor: style.color,
                    }}
                    title={`${d.cls} ${confidencePct(d)?.toFixed(1) ?? "?"}%`}
                    onClick={() => onSelect(d)}
                  >
                    <b style={{ background: style.color }}>
                      {d.cls} {confidencePct(d)?.toFixed(0) ?? "?"}%
                    </b>
                  </button>
                );
              })}
          </div>
        ) : (
          <p className="muted-note">No preview could be made of this strip.</p>
        )}
      </div>
      {selectedKey && onStrip.some((d) => d.key === selectedKey) && (
        <div className="msn-strip-detail">
          <DetectionFacts d={onStrip.find((d) => d.key === selectedKey)} />
        </div>
      )}
    </div>
  );
}

export default SurveyMission;
