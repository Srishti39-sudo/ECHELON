"""GhostTrace from the command line: rescue decisions for one processed survey.

    .venv/bin/python run_ghosttrace.py --survey data/surveys/<survey>

Reads export.json (and manifest.json / navigation sidecars when present) from
the survey directory, writes ghosttrace.json and ghosttrace.geojson beside it,
and prints a human summary with urgent targets first. The JSON is the product;
this printout is a convenience and carries the same caveats.

Exit status: 0 on success, 2 when the survey directory has no readable
export.json. Per-target stage failures never change the exit status; they are
recorded in run.failures and counted in the summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghosttrace.engine import run_ghosttrace  # noqa: E402


def _fmt_pos(t: dict) -> str:
    if t.get("latitude") is None:
        return "position unavailable"
    return f"{t['latitude']:.5f}, {t['longitude']:.5f}"


def print_summary(result: dict) -> None:
    s = result["summary"]
    print(f"GhostTrace  survey {result['survey_id']}  ({result['format']})")
    if result.get("demo") or result.get("synthetic_inputs"):
        print("  ** SYNTHETIC / DEMO INPUTS - not evidence of any real object **")
    print(f"  targets {s['targets']}   urgent {s['urgent']}   high {s['high']}   "
          f"actively fishing {s['actively_fishing']}   near habitat {s['near_sensitive_habitat']}   "
          f"propeller hazards {s['propeller_hazards']}   suppressed excluded {s.get('suppressed_excluded', 0)}")
    cs = result["change_summary"]
    if cs["compared_with"]:
        print(f"  change vs {', '.join(cs['compared_with'])}: new {cs['new']}  moved {cs['moved']}  "
              f"persistent {cs['persistent']}  removed {cs['removed']}")
    else:
        print("  change: no earlier overlapping georeferenced survey")
    print()
    for tier in ("urgent", "high", "routine"):
        rows = [t for t in result["targets"] if t["priority"]["tier"] == tier]
        if not rows:
            continue
        print(f"  {tier.upper()}")
        for t in rows:
            p = t["priority"]
            conf = t.get("confidence_pct")
            conf_text = f"{conf:.0f}%" if isinstance(conf, (int, float)) else "n/a"
            print(f"    #{p['rank']:<3} {t['object_class']:<12} score {p['score']:.3f}  conf {conf_text:<5} "
                  f"activity {t['activity']['level']:<8} change {t['change']['status']:<18} "
                  f"{_fmt_pos(t)}  [{t['detection_id']}]")
            if t.get("errors"):
                print(f"         stage errors: {'; '.join(t['errors'])}")
    plan = result["recovery_plan"]
    if plan["order"]:
        start = (plan.get("start") or {}).get("name")
        print(f"\n  recovery route from {start}: {' -> '.join(plan['order'])}  ({plan['total_km']} km, straight-line)")
    print("\n  caveats:")
    for c in result["caveats"]:
        print(f"    - {c}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--survey", required=True, help="processed survey directory (contains export.json)")
    ap.add_argument("--surveys-root", default=None,
                    help="directory of processed surveys to compare against (default: the survey's parent)")
    ap.add_argument("--horizon-hours", type=float, default=240)
    ap.add_argument("--particles", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="no progress lines")
    args = ap.parse_args(argv)

    def progress(event: dict) -> None:
        if args.quiet:
            return
        if event.get("stage") == "target":
            errs = f"  errors: {len(event['errors'])}" if event.get("errors") else ""
            print(f"  [{event['index']}/{event['total']}] {event['detection_id']}  "
                  f"activity {event.get('activity_level')}{errs}", file=sys.stderr)
        else:
            print(f"  stage {event.get('stage')}: {event.get('status')}", file=sys.stderr)

    try:
        result = run_ghosttrace(args.survey, surveys_root=args.surveys_root,
                                horizon_hours=args.horizon_hours, n_particles=args.particles,
                                seed=args.seed, on_event=progress)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_summary(result)
    print(f"\n  wrote {Path(args.survey) / 'ghosttrace.json'} and ghosttrace.geojson")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
