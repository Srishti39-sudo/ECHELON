import {
  LayoutDashboard,
  Radio,
  ScanSearch,
  Map,
  Bell,
  History,
  Waves,
  MessageSquareText,
  Radar,
  LifeBuoy,
} from "lucide-react";

import { NavLink } from "react-router-dom";

const menuItems = [
  // First and styled apart, because it is the one page that does the whole
  // job the problem statement asks for: upload a raw log, watch the models'
  // detections land on the map as the run happens, download the reports.
  {
    name: "Live survey",
    path: "/mission",
    icon: Radar,
    featured: true,
  },
  // Second, because it is what a detection is for: which ghost net is killing
  // now, what it will reach, and which one to recover first.
  {
    name: "GhostTrace",
    path: "/ghosttrace",
    icon: LifeBuoy,
    featured: true,
  },
  {
    name: "Dashboard",
    path: "/",
    icon: LayoutDashboard,
  },
  {
    name: "Live Feed",
    path: "/live",
    icon: Radio,
  },
  {
    name: "Detections",
    path: "/detections",
    icon: ScanSearch,
  },
  {
    name: "Survey Hazard Map",
    path: "/map",
    icon: Map,
  },
  {
    name: "Alerts",
    path: "/alerts",
    icon: Bell,
  },
  {
    name: "History",
    path: "/history",
    icon: History,
  },
  {
    name: "Beacon",
    path: "/assistant",
    icon: MessageSquareText,
  },
];

function Sidebar() {
  return (
    <aside className="sidebar">

      <div className="brand">
        <div className="brand-icon">
          <Waves size={25} />
        </div>

        <div>
          <h2>
            Deep<span>Echo</span>
          </h2>
          <p>Marine Intelligence</p>
        </div>
      </div>

      <nav className="sidebar-nav">
        {menuItems.map((item) => {
          const Icon = item.icon;

          return (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) =>
                `nav-item ${item.featured ? "nav-item-featured" : ""} ${isActive ? "active" : ""}`
              }
            >
              <Icon size={20} />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="system-status">
        <span className="status-dot"></span>

        <div>
          <strong>System Online</strong>
          <p>All services operational</p>
        </div>
      </div>

    </aside>
  );
}

export default Sidebar;