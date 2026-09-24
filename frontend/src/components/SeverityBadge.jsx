import { styleForSeverity } from "../services/severity";

/**
 * One severity, rendered.
 *
 * The colours and the ordering live in services/severity.js; this file exports
 * only the component so fast refresh keeps working.
 */
function SeverityBadge({ tier, children }) {
  const style = styleForSeverity(tier);

  return (
    <span
      className="severity-badge"
      style={{ color: style.color, background: style.tint }}
    >
      {children ?? style.label}
    </span>
  );
}

export default SeverityBadge;
