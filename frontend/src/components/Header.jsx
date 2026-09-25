import { useLocation, useNavigate } from "react-router-dom";

import { Upload, Activity } from "lucide-react";

// The header used to say "Dashboard" on every page, including the ones that are
// not the dashboard. Titles follow the route instead; an unknown route falls
// back to the product name rather than to a page it is not.
const TITLES = {
  "/": ["Dashboard", "Monitor and analyze underwater sonar surveys"],
  "/mission": ["Live survey", "Upload a sonar log and follow its analysis as it runs"],
  "/ghosttrace": ["GhostTrace", "Which ghost net to recover first, and why"],
  "/live": ["Analysis feed", "Analyse a single sonar tile"],
  "/detections": ["Detections", "Every stored contact"],
  "/map": ["Survey hazard map", "Ranked hotspots for a processed survey"],
  "/alerts": ["Alerts", "Contacts that need attention"],
  "/history": ["History", "Stored scans"],
  "/assistant": ["Beacon", "Grounded answers, every claim cited"],
  "/report": ["Survey report", "A printable A4 report of one processed survey"],
};

function Header() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const route = pathname.startsWith("/ghosttrace")
    ? "/ghosttrace"
    : pathname.startsWith("/report/") ? "/report" : pathname;
  const [title, subtitle] = TITLES[route] ?? ["DeepEcho", "Side-scan sonar analysis"];

  return (
    <header className="header">

      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>

      <div className="header-actions">

        <div className="connection-status">
          <Activity size={16} />
          ML Service Ready
        </div>

        <button className="upload-button" onClick={() => navigate("/mission")}>
          <Upload size={18} />
          Upload Sonar Log
        </button>

      </div>

    </header>
  );
}

export default Header;
