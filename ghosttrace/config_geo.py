"""Every knob for the GhostTrace geographic stages, in one file.

The geographic stages answer three questions about a detected ghost net or
piece of debris: what habitat does it lie on or near (habitat.py), where could
it go (currents.py + drift.py), and who could it hurt (safety.py). Each of them
reads its numbers from here and nowhere else, the same rule hazard_config.py
applies to the survey hazard engine.

Every value is environment-overridable as GHOSTTRACE_<NAME>, so a demo, a test
or an operator can change a knob without editing code, and the value actually
used is written into the outputs beside the result it produced.

THREE KINDS OF NUMBER LIVE HERE, AND THEY ARE LABELLED
    cited       the value comes from a named, retrievable document, and the
                citation sits next to it
    heuristic   a configurable operational choice made for this project; it
                was not fitted to recovery data, because no labelled data on
                ghost-net drift or recovery exists in this repository
    uncalibrated assumption
                a physical parameter nobody has measured for this problem; the
                forecast still needs a number, so it gets one, and the output
                says it is an assumption

Nothing in this file is an official Navy, Coast Guard, INCOIS, NOAA or IMO
procedure.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Environment helpers ---------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"GHOSTTRACE_{name}", str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"GHOSTTRACE_{name}", str(default)))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"GHOSTTRACE_{name}", default)


# --- Identity ------------------------------------------------------------------------

GEO_VERSION = "1.0.0"

HEURISTIC_LABEL = ("configurable heuristic chosen for this project; not fitted to real "
                   "ghost-gear drift or recovery data and not an official procedure")

# --- Where the bundled data lives ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env_str("DATA_DIR", str(REPO_ROOT / "data" / "ghosttrace")))
LAYERS_DIR = DATA_DIR / "layers"
CURRENTS_DIR = DATA_DIR / "currents"
BATHYMETRY_DIR = DATA_DIR / "bathymetry"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# --- Regions with bundled offline data --------------------------------------------------
# (south, west, north, east) in decimal degrees, WGS84.
#
# GhostTrace is meant for the whole Indian EEZ, but the data bundled in the
# repository covers only these two boxes, to keep the checkout small. A point
# outside every box is answered with "no bundled data for this location",
# never with a distance of zero or an empty list that reads as "nothing near".
#
# gulf_of_mannar_palk_bay  Gulf of Mannar Marine National Park island chain
#                          (between Rameswaram and Thoothukudi), Palk Bay and
#                          the Sri Lankan side of both. The synthetic demo
#                          survey sits at 9.12 N 79.05 E inside this box.
# odisha_coast             the three olive ridley mass-nesting rookeries:
#                          Gahirmatha, Devi river mouth and Rushikulya.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "gulf_of_mannar_palk_bay": (8.4, 77.9, 10.4, 80.4),
    "odisha_coast": (19.0, 84.7, 21.0, 87.6),
}

# --- Habitat context (habitat.py) --------------------------------------------------------

# Features further than this from the target are not reported as "nearest".
# 50 km is about a day of drift at 0.5 m/s, the far end of what a single
# recovery sortie would plan around.
HABITAT_SEARCH_RADIUS_M = _env_float("HABITAT_SEARCH_RADIUS_M", 50_000.0)

# Sensitivity of each layer kind, 0..1. HEURISTIC. Consequence of a net
# reaching the feature, not likelihood:
#   reef            1.0  coral is killed in place by abrasion and smothering,
#                        and recovers over decades
#   turtle_nesting  0.9  olive ridley arribada beaches; nearshore breeding
#                        congregations are exactly where nets entangle turtles
#   dugong          0.9  dugong are Schedule I (Wild Life (Protection) Act,
#                        1972) with an estimated ~240 animals left in India
#                        (Tamil Nadu press release P.R. No. 1645, 21.09.2022)
#   seagrass        0.8  dugong forage habitat; nets smother meadows
#   protected_area  0.8  a legal designation, whatever habitat it contains
#   harbour         0.0  harbours are a PEOPLE exposure, not an ecological one;
#                        they are scored in safety.py, not in the habitat score
KIND_SENSITIVITY: dict[str, float] = {
    "reef": _env_float("SENS_REEF", 1.0),
    "turtle_nesting": _env_float("SENS_TURTLE_NESTING", 0.9),
    "dugong": _env_float("SENS_DUGONG", 0.9),
    "seagrass": _env_float("SENS_SEAGRASS", 0.8),
    "protected_area": _env_float("SENS_PROTECTED_AREA", 0.8),
    "harbour": _env_float("SENS_HARBOUR", 0.0),
}

# Habitat score. HEURISTIC.
#   term(kind)  = sensitivity(kind) * exp(-distance_m / HABITAT_DECAY_M)
#                 distance_m = 0 when the target lies inside the feature
#   score       = 1 - product over kinds of (1 - term(kind))
# The product form is a probabilistic OR: two moderately close sensitive
# layers score higher than either alone, and the score can never exceed 1.
# At the default decay length a reef 2 km away contributes 0.67, 5 km 0.37,
# 15 km 0.05. Only the nearest feature of each kind counts, so a field of 67
# reef polygons does not add up to certainty.
HABITAT_DECAY_M = _env_float("HABITAT_DECAY_M", 5_000.0)

# --- Currents (currents.py) ----------------------------------------------------------------

# Level names accepted by CurrentField.sample.
#   surface  the model's shallowest level (HYCOM z = 0 m)
#   bottom   the deepest model level with valid data at each grid cell. In a
#            z-level model that is the last level above the model bathymetry,
#            so it is a NEAR-bottom current, up to one level spacing above the
#            bed, not a boundary-layer velocity.
CURRENT_LEVELS = ("surface", "bottom")

# Bilinear interpolation next to the model's land mask: a sample is built from
# the valid corners only (weights renormalised) when at least this many of the
# four corners are valid. With 1 a thin strip of coast is not a dead zone; the
# alternative (all four required) would strand every particle one grid cell
# (~9 km) offshore of any coast.
CURRENT_MIN_VALID_CORNERS = _env_int("CURRENT_MIN_VALID_CORNERS", 1)

# --- Drift (drift.py) ------------------------------------------------------------------------

DRIFT_DEFAULT_HORIZON_H = _env_float("DRIFT_HORIZON_H", 240.0)
DRIFT_DEFAULT_PARTICLES = _env_int("DRIFT_PARTICLES", 500)
DRIFT_DEFAULT_DT_MIN = _env_float("DRIFT_DT_MIN", 30.0)

# Integration scheme: "rk4" or "rk2" (midpoint). RK4 costs four field samples a
# step; at 500 particles and 480 steps that is still well under a second of
# numpy work. RK2 is kept for speed comparisons.
DRIFT_INTEGRATOR = _env_str("DRIFT_INTEGRATOR", "rk4")

# Horizontal eddy diffusivity for the random walk, m^2/s. UNCALIBRATED
# ASSUMPTION. The random walk stands in for motion the 1/12 degree model cannot
# resolve (eddies, tides at sub-grid scale, Stokes drift). Values of order
# 1-10 m^2/s are common in coastal particle-tracking practice at this model
# resolution; 1.0 is the conservative end, so the cone widens slowly and is
# driven mostly by the model current. The step is sqrt(2 K dt) per axis.
DRIFT_K_SURFACE_M2S = _env_float("DRIFT_K_SURFACE_M2S", 1.0)
# Near the bed turbulence is damped by the bed itself; a tenth of the surface
# value. Also an uncalibrated assumption.
DRIFT_K_BOTTOM_M2S = _env_float("DRIFT_K_BOTTOM_M2S", 0.1)

# Seabed mobility. UNCALIBRATED ASSUMPTION.
# No published threshold current speed for the movement of a lost net on the
# seabed was found while building this module. The closest evidence is
# qualitative: Good et al. (2010), "Derelict fishing nets in Puget Sound and
# the Northwest Straits: patterns and threats to marine fauna", Marine
# Pollution Bulletin 60(1):39-50, doi:10.1016/j.marpolbul.2009.09.005, report
# derelict gillnets remaining in place for years, especially when entangled on
# high-relief rocky habitat. A snagged net may not move at all.
#
# The model therefore moves a seabed net only while the near-bottom current
# exceeds U_CRIT, and then at a fraction of that current. The 0.25 m/s default
# is of the order of the mean current that starts to move fine-to-medium sand
# on a flat bed, used purely as an order-of-magnitude analogy for "a current
# strong enough to shift loose material on the bottom"; a net is not sand. The
# seabed forecast is therefore POSSIBLE displacement, not expected displacement.
SEABED_U_CRIT_MPS = _env_float("SEABED_U_CRIT_MPS", 0.25)
SEABED_MOBILITY_FRACTION = _env_float("SEABED_MOBILITY_FRACTION", 0.3)
# Per-particle spread of U_CRIT, as a fraction of U_CRIT (uniform in
# [1-s, 1+s]), so the ensemble expresses that nobody knows the threshold.
# 0 makes every particle share the same threshold.
SEABED_U_CRIT_SPREAD = _env_float("SEABED_U_CRIT_SPREAD", 0.25)

# Floating mode: windage (a fraction of the 10 m wind added to the surface
# current) is only applied when wind data is bundled. None is bundled, so
# floating forecasts are current-only and say so.
WINDAGE_FRACTION = _env_float("WINDAGE_FRACTION", 0.0)

# Output shape.
DRIFT_SNAPSHOT_EVERY_H = _env_float("DRIFT_SNAPSHOT_EVERY_H", 12.0)
DRIFT_MAX_SNAPSHOT_POINTS = _env_int("DRIFT_MAX_SNAPSHOT_POINTS", 150)
# How often particle positions are tested against feature buffers and land.
# Every step: a particle cannot cross a 1 km buffer unseen at 30 min steps
# unless it moves faster than 0.55 m/s the whole way, which is then logged as
# a limitation of the step size.
DRIFT_IMPACT_CHECK_EVERY_STEPS = _env_int("DRIFT_IMPACT_CHECK_EVERY_STEPS", 1)

# A particle "reaches" a feature when it comes within this distance of it.
# HEURISTIC: a net within 1 km of a reef edge at the model's resolution is
# indistinguishable from a net on the reef.
IMPACT_BUFFER_M: dict[str, float] = {
    "reef": _env_float("IMPACT_BUFFER_REEF_M", 1000.0),
    "seagrass": _env_float("IMPACT_BUFFER_SEAGRASS_M", 1000.0),
    "protected_area": _env_float("IMPACT_BUFFER_PROTECTED_AREA_M", 0.0),
    "turtle_nesting": _env_float("IMPACT_BUFFER_TURTLE_NESTING_M", 2000.0),
    "dugong": _env_float("IMPACT_BUFFER_DUGONG_M", 2000.0),
    "harbour": _env_float("IMPACT_BUFFER_HARBOUR_M", 2000.0),
}
IMPACT_KINDS = ("reef", "seagrass", "protected_area", "turtle_nesting", "dugong", "harbour")

# Stranded particles are grouped on a grid of this size for `stranded_where`,
# and a group is named after the nearest named feature within this distance.
STRAND_GROUP_DEG = _env_float("STRAND_GROUP_DEG", 0.02)
STRAND_NAME_RADIUS_M = _env_float("STRAND_NAME_RADIUS_M", 10_000.0)

# Probability cones. The particle cloud is turned into a density on a grid
# (histogram smoothed with a Gaussian kernel whose width follows Scott's rule),
# and the cone is the smallest set of cells holding 50% / 90% of the particles.
CONE_GRID_CELLS = _env_int("CONE_GRID_CELLS", 96)
# Below this many particles with distinct positions, or when the cloud is
# smaller than CONE_MIN_EXTENT_M across, no density is estimated: the cone
# falls back to the convex hull of the nearest 50% / 90% of particles, or to
# null when everything is at one point.
CONE_MIN_DISTINCT = _env_int("CONE_MIN_DISTINCT", 10)
CONE_MIN_EXTENT_M = _env_float("CONE_MIN_EXTENT_M", 5.0)

# --- People safety (safety.py) -----------------------------------------------------------------

# Recreational diving depth limits. CITED, as training-agency limits, not law:
# PADI Open Water Diver is certified to 18 m (60 ft) and Advanced Open Water
# Diver to 30 m (100 ft), https://www.padi.com/courses/open-water-diver and
# https://www.padi.com/courses/advanced-open-water-diver . A recovery below
# these depths is a professional (commercial / military) diving job.
DIVE_LIMIT_OPEN_WATER_M = _env_float("DIVE_LIMIT_OPEN_WATER_M", 18.0)
DIVE_LIMIT_ADVANCED_M = _env_float("DIVE_LIMIT_ADVANCED_M", 30.0)

# Current above which a diver recovery is not planned: 1 knot (0.514 m/s).
# CITED, as a foreign regulation used as a planning analogue: the US OSHA
# Commercial Diving Operations standard, 29 CFR 1910.424(b)(3), "SCUBA diving
# shall not be conducted ... Against currents exceeding one (1) knot unless
# line-tended" (text retrieved 2026-09-13 from the eCFR,
# https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-T/section-1910.424 ).
# It is US workplace law, not Indian law and not a recreational rule; the US
# Navy Diving Manual could not be retrieved to cross-check (HTTP 403). An
# overridden value is reported as a heuristic, not as this citation.
DIVER_CURRENT_LIMIT_MPS = _env_float("DIVER_CURRENT_LIMIT_MPS", 0.514)
DIVER_CURRENT_LIMIT_DEFAULT_MPS = 0.514
DIVER_CURRENT_LIMIT_CITATION = (
    "US OSHA 29 CFR 1910.424(b)(3) (Commercial Diving Operations, SCUBA diving limits): SCUBA diving "
    "shall not be conducted against currents exceeding one (1) knot unless line-tended. "
    "https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-T/section-1910.424 "
    "(retrieved 2026-09-13). US workplace regulation used as a planning analogue; not Indian law.")
DIVE_DEPTH_LIMIT_CITATION = (
    "PADI course limits (training-agency limits, not law): Open Water Diver 18 m "
    "(https://www.padi.com/courses/open-water-diver), Advanced Open Water Diver 30 m "
    "(https://www.padi.com/courses/advanced-open-water-diver).")

# Window over which the near-bottom current is maximised for the diver brief.
DIVER_CURRENT_WINDOW_H = _env_float("DIVER_CURRENT_WINDOW_H", 12.0)

# Propeller hazard. HEURISTIC. Points add up; the level is the first tier
# whose floor the total reaches.
#   shallow         seabed depth < PROP_SHALLOW_M: a net lifted by a propeller
#                   wash or snagging a keel reaches the surface in shallow water
#   floating        floating / midwater mode: the net is in the propeller's
#                   depth band already
#   harbour_near    a harbour or fishing landing within PROP_HARBOUR_NEAR_M of
#                   the target, where vessel traffic concentrates
#   harbour_drift   the drift forecast reaches a harbour buffer with
#                   probability >= PROP_HARBOUR_DRIFT_P
#   large_net       net size >= PROP_LARGE_NET_M across
PROP_SHALLOW_M = _env_float("PROP_SHALLOW_M", 10.0)
PROP_HARBOUR_NEAR_M = _env_float("PROP_HARBOUR_NEAR_M", 5_000.0)
PROP_HARBOUR_DRIFT_P = _env_float("PROP_HARBOUR_DRIFT_P", 0.1)
PROP_LARGE_NET_M = _env_float("PROP_LARGE_NET_M", 20.0)
PROP_POINTS: dict[str, float] = {
    "shallow": _env_float("PROP_PTS_SHALLOW", 2.0),
    "floating": _env_float("PROP_PTS_FLOATING", 3.0),
    "harbour_near": _env_float("PROP_PTS_HARBOUR_NEAR", 1.0),
    "harbour_drift": _env_float("PROP_PTS_HARBOUR_DRIFT", 1.0),
    "large_net": _env_float("PROP_PTS_LARGE_NET", 1.0),
}
# (level, minimum points), checked in order.
PROP_TIERS: tuple[tuple[str, float], ...] = (
    ("high", _env_float("PROP_TIER_HIGH", 3.0)),
    ("moderate", _env_float("PROP_TIER_MODERATE", 1.5)),
    ("low", 0.0),
)

# --- Fishing activity source ---------------------------------------------------------------------
# Global Fishing Watch needs an API token. When GFW_API_TOKEN is set the
# fetcher may use it; otherwise fishing exposure falls back to OSM harbours and
# landing points. The fetcher records which one it used in the manifest.
GFW_API_TOKEN_ENV = "GFW_API_TOKEN"


def load_data_sources() -> list[dict]:
    """The data sources bundled in data/ghosttrace, from its manifest.

    Returned as {name, url, licence, snapshot, used_for} records, the shape the
    GhostTrace output contract expects. Empty when the data has not been
    fetched, so a caller never sees a source that is not on disk.
    """
    import json

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for entry in manifest.get("sources", []):
        out.append({
            "name": entry.get("name"),
            "url": entry.get("url"),
            "licence": entry.get("licence"),
            "snapshot": entry.get("accessed") or entry.get("snapshot"),
            "used_for": entry.get("used_for"),
        })
    return out


# Evaluated at import so the engine can read it as a plain value.
DATA_SOURCES: list[dict] = load_data_sources()
