import { Routes, Route } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Header from "./components/Header";

import Dashboard from "./pages/Dashboard";
import LiveFeed from "./pages/LiveFeed";
import Detections from "./pages/Detections";
import MapPage from "./pages/MapPage";
import Alerts from "./pages/Alerts";
import History from "./pages/History";
import Assistant from "./pages/Assistant";
import SurveyMission from "./pages/SurveyMission";
import GhostTracePage from "./pages/GhostTracePage";
import SurveyReport from "./pages/SurveyReport";
import GhostTraceFixturePreview from "./ghosttrace/GhostTraceFixturePreview";

function App() {
  return (
    <div className="app">

      <Sidebar />

      <main className="main">

        <Header />

        <div className="page-content">

          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/mission" element={<SurveyMission />} />
            <Route path="/ghosttrace" element={<GhostTracePage />} />
            <Route path="/ghosttrace/:surveyId" element={<GhostTracePage />} />
            {/* Renders the panel from a synthetic fixture with no backend. Dev only. */}
            {import.meta.env.DEV && (
              <Route path="/dev/ghosttrace" element={<GhostTraceFixturePreview />} />
            )}
            <Route path="/live" element={<LiveFeed />} />
            <Route path="/detections" element={<Detections />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/report/:surveyId" element={<SurveyReport />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/history" element={<History />} />
            <Route path="/assistant" element={<Assistant />} />
          </Routes>

        </div>

      </main>

    </div>
  );
}

export default App;