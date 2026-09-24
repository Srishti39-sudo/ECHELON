"""Every knob for the GhostTrace decision core, in one file.

GhostTrace turns a detected ghost net or debris object into a rescue decision:
is it still killing (water-column activity), what would it hit (habitat and
drift, from the geo stages), who could it hurt (propeller and diver hazard),
has it moved since the last survey, how urgent is it, who should be told, and
in what order a recovery vessel should visit.

The same rule as hazard_config applies here: a number that decides anything
lives in this file and nowhere else, with the reason it has the value it has.

NOTHING HERE IS A PUBLISHED STANDARD. Every weight, threshold and neutral
value below is a heuristic chosen for this project. None was fitted against
real ghost-net recoveries, because no labelled data of that kind exists in this
repository. They are written into every ghosttrace.json beside the scores they
produce (`priority.terms`, `priority.formula`) so a changed knob is visible in
the output and any score can be recomputed by hand from the file alone.
"""

from __future__ import annotations

import os

# --- Identity ----------------------------------------------------------------

FORMAT = "deepecho-ghosttrace/1"
CORE_VERSION = "1.0.0"
OUTPUT_JSON = "ghosttrace.json"
OUTPUT_GEOJSON = "ghosttrace.geojson"

HEURISTIC_LABEL = ("configurable heuristic chosen for this project; not fitted to "
                   "real ghost-gear recovery data and not an official procedure")


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --- Which detections become targets -------------------------------------------
# Matched on hyphen-delimited tokens of the normalised class name (see
# engine.is_target_class), so "fishing-net", "ghost_net" and "Nets" all reach
# "net" while "cabinet" does not.
#
# `unknown` is excluded by default. An unidentified object is handled under the
# unidentified-object protocol in the corpus (treat as potentially explosive,
# do not approach), which is the opposite of sending a recovery diver to it.
# Turning it on is a deliberate operator choice, not a default.
GHOSTTRACE_CLASSES: tuple[str, ...] = (
    "net", "ghost-gear", "fishing-gear", "rope", "debris",
    "tyre", "tire", "drum", "container",
)
INCLUDE_UNKNOWN = _flag("GHOSTTRACE_INCLUDE_UNKNOWN")

# Verification may mark a detection suppressed (probable artefact). Excluded
# from targets by default and counted in summary.suppressed_excluded, never
# silently dropped from the count.
EXCLUDE_SUPPRESSED = not _flag("GHOSTTRACE_KEEP_SUPPRESSED")

# Class families for change matching. A net seen last month is allowed to be
# called "ghost-gear" this month; a net is never matched to a tyre.
CLASS_FAMILIES: dict[str, tuple[str, ...]] = {
    "gear": ("net", "ghost-gear", "fishing-gear", "rope", "line", "trap", "pot",
             "longline", "gillnet", "trawl"),
    "debris": ("debris", "tyre", "tire", "drum", "container", "bottle", "can",
               "barrel"),
    "unknown": ("unknown", "anomaly"),
}

# Gear-family classes are the ones that "keep fishing". Used for the change
# term and for the fisheries situation in alerts.
ACTIVE_GEAR_FAMILY = "gear"

# --- Drift mode -------------------------------------------------------------------
# A net found by side-scan is on the seabed: side-scan images the seabed, and a
# snagged net is the common case. So `seabed` is the default. `floating` is
# chosen only when something positively says the object is up in the water:
#   * the class name carries a midwater/floating hint, or
#   * dimensions.height_m is at least MIDWATER_HEIGHT_M (a net standing that
#     tall is being held up by its floats and will move like a floating one),
#   * or the verification block states midwater: true.
# Height is a weak proxy and is labelled as such in drift.mode_basis.
FLOATING_CLASS_HINTS: tuple[str, ...] = ("float", "buoy", "midwater", "surface", "fad",
                                         "drifting")
MIDWATER_HEIGHT_M = float(os.environ.get("GHOSTTRACE_MIDWATER_HEIGHT_M", "5.0"))
DRIFT_DT_MINUTES = float(os.environ.get("GHOSTTRACE_DRIFT_DT_MINUTES", "30"))
# For a seabed net, also run the floating forecast as a labelled "if refloated"
# scenario (target.drift_scenarios.if_refloated). Display only: priority uses
# the forecast for the mode the net was found in. Doubles drift run time.
DRIFT_REFLOAT_SCENARIO = not _flag("GHOSTTRACE_NO_REFLOAT_SCENARIO")

# --- Water-column activity (watercolumn.py) --------------------------------------
# Along-track half-window around a detection, in metres. 25 m either side is
# about the length of a gillnet panel plus a margin: fish attracted to a net
# hold station within tens of metres of it, not hundreds.
WC_WINDOW_M = float(os.environ.get("GHOSTTRACE_WC_WINDOW_M", "25"))
# Range bins nearest the transducer are dominated by ringdown and are excluded.
WC_RINGDOWN_BINS = int(os.environ.get("GHOSTTRACE_WC_RINGDOWN_BINS", "5"))
# Bins just above the bottom return are dominated by the bottom's own sidelobe
# and are excluded, per ping, relative to bottom_range_m.
WC_BOTTOM_GUARD_BINS = int(os.environ.get("GHOSTTRACE_WC_BOTTOM_GUARD_BINS", "3"))
# Robust z-score a cell must exceed to count as echo. 3.0 on a median/MAD
# background is roughly "clearly above the speckle" for Rayleigh-like noise.
WC_Z_THRESHOLD = float(os.environ.get("GHOSTTRACE_WC_Z", "3.0"))
# A connected component smaller than this many cells is speckle, not a school.
WC_MIN_CLUSTER_CELLS = int(os.environ.get("GHOSTTRACE_WC_MIN_CELLS", "4"))
# The background is a cluster rate over every ping row on the same side of the
# same line that is clear of detections, scaled to the near window's length.
# Other detections are excluded over their box plus this margin, so their own
# fish do not raise the background; the target's own window is excluded whole.
WC_EXCLUDE_MARGIN_M = float(os.environ.get("GHOSTTRACE_WC_EXCLUDE_MARGIN_M", "5"))
# Below this many near-window lengths of clear rows the rate rests on too little
# of the line to be a background, and activity is reported unavailable.
WC_MIN_BACKGROUND_WINDOWS = float(os.environ.get("GHOSTTRACE_WC_MIN_BACKGROUND_WINDOWS", "1"))
# Added to the background mean before dividing, in clusters per window, so a
# quiet line (background 0) does not turn one cluster into an infinite ratio.
WC_EPSILON = float(os.environ.get("GHOSTTRACE_WC_EPSILON", "0.5"))
# score = 1 / (1 + exp(-WC_LOGISTIC_K * (ln(enrichment) - ln(WC_LOGISTIC_MID))))
# i.e. 0.5 at twice the background cluster rate, ~0.83 at 5x, ~0.2 at 0.8x.
WC_LOGISTIC_MID = 2.0
WC_LOGISTIC_K = 1.5
# enrichment is clamped below by this before the log, so zero near-clusters
# scores ~0 instead of raising.
WC_ENRICHMENT_FLOOR = 1e-3
WC_LEVELS: tuple[tuple[str, float], ...] = (("high", 0.70), ("moderate", 0.40), ("low", 0.0))
WC_LIMITATIONS = (
    "Water-column echoes near the net may be fish, bubbles, suspended sediment "
    "or turbulence; the method cannot tell them apart. It has not been validated "
    "on real ghost-net data. A high score is evidence of biological aggregation "
    "(or other scatterers) near the object, not proof of entanglement or that "
    "the net is still catching.")

# --- Change detection (changes.py) ------------------------------------------------
# Gate for matching the same object across surveys. Two independently navigated
# side-scan surveys routinely disagree by 10-30 m (layback, GNSS, heading), so
# the gate must sit well above that or every persistent net reads as new.
MATCH_RADIUS_M = float(os.environ.get("GHOSTTRACE_MATCH_RADIUS_M", "75"))
# Below this displacement a match is "persistent": inside the combined
# positional uncertainty of two surveys, movement cannot be claimed.
STATIONARY_M = float(os.environ.get("GHOSTTRACE_STATIONARY_M", "20"))
# When both surveys measured dimensions, a plan-area ratio above this rejects
# the match; below it, ln(ratio) * SIZE_PENALTY_M is added to the cost.
SIZE_RATIO_MAX = 4.0
SIZE_PENALTY_M = 15.0
# Survey extents are compared as lat/lon bounding boxes grown by this margin.
EXTENT_MARGIN_M = MATCH_RADIUS_M
# Track rows sampled for the coverage hull (every Nth row). The hull is built
# from tile centres, located detections and sampled track points, all of which
# lie INSIDE the swath, so it under-states coverage: a prior object near the
# swath edge is left undecided rather than wrongly called removed.
COVERAGE_TRACK_STEP = 25

# --- Priority (priority.py) ---------------------------------------------------------
# score = confidence_factor * sum(weight_i * value_i), value_i in 0..1.
# Weights sum to 1.0, so the sum is 0..1 and the multiplier keeps a weak
# detection from ever becoming urgent on context alone.
PRIORITY_WEIGHTS: dict[str, float] = {
    # Still killing is the whole reason to hurry. Largest single weight.
    "activity": 0.25,
    # What it lies on or next to now (reef, turtle nesting approach, MPA).
    "habitat": 0.20,
    # What it will reach if left: max impact probability on a sensitive layer.
    "drift_impact": 0.15,
    # Propeller fouling and diver entanglement: human safety, not only ecology.
    "people_risk": 0.15,
    # A bigger net traps more; saturating so one huge net does not dominate.
    "size": 0.10,
    # Mobile gear keeps killing along its path; a moved/new net gets a bump.
    "change": 0.10,
    # Small bonus for quick wins: shallow and slack water is cheap to recover.
    "recoverability": 0.05,
}
# The value a term takes when its input is unavailable. Stated in each term's
# `basis`. Not zero: absence of a measurement is not evidence of absence, and a
# plain image survey must not rank below a water-column survey that measured
# "low". Not high either: an unmeasured term must not make something urgent.
PRIORITY_NEUTRAL: dict[str, float] = {
    "activity": 0.40,
    "habitat": 0.30,
    "drift_impact": 0.20,
    "people_risk": 0.30,
    "size": 0.30,
    "change": 0.50,
    "recoverability": 0.50,
}
# Impact kinds counted as sensitive for drift_impact (substring match).
SENSITIVE_IMPACT_KINDS: tuple[str, ...] = ("reef", "coral", "turtle", "dugong", "protected",
                                           "mpa", "marine-park", "marine_park", "seagrass",
                                           "mangrove", "sanctuary", "national-park")
PROPELLER_LEVEL_VALUE: dict[str, float] = {
    "high": 1.0, "severe": 1.0, "moderate": 0.6, "medium": 0.6, "low": 0.2, "none": 0.0,
    "negligible": 0.0,
}
# size = 1 - exp(-area_m2 / SIZE_SCALE_M2). 0.63 at 50 m2 (a 10 x 5 m panel).
SIZE_SCALE_M2 = 50.0
CHANGE_VALUE: dict[str, float] = {
    "moved": 1.0,          # proven mobile: still sweeping new seabed
    "new": 0.7,            # arrived since the last look over this spot
    "persistent": 0.4,     # stationary; still harmful, but not spreading
}
# recoverability parts, linear between easy and hard.
RECOVER_DEPTH_EASY_M = 10.0
RECOVER_DEPTH_HARD_M = 40.0
RECOVER_CURRENT_EASY_MS = 0.25
RECOVER_CURRENT_HARD_MS = 1.0
PRIORITY_TIERS: tuple[tuple[str, float], ...] = (("urgent", 0.55), ("high", 0.35),
                                                 ("routine", 0.0))

# --- Summary counters --------------------------------------------------------------
NEAR_HABITAT_M = float(os.environ.get("GHOSTTRACE_NEAR_HABITAT_M", "2000"))
NEAR_HABITAT_SCORE = 0.5
# The "if refloated" scenario names a protected area's manager in an alert only
# when at least this share of simulated particles reaches the area. Higher than
# the forecast's bar (> 0) because the scenario assumes the net is lifted.
SCENARIO_NOTIFY_PROBABILITY = float(os.environ.get("GHOSTTRACE_SCENARIO_NOTIFY_P", "0.5"))
PROPELLER_HAZARD_LEVELS: tuple[str, ...] = ("high", "severe", "moderate", "medium")

# --- Alerts (alerts.py) -----------------------------------------------------------------
KB_DIR = os.environ.get("GHOSTTRACE_KB_DIR", "rag_assistant/kb")
USE_ASSISTANT = _flag("GHOSTTRACE_USE_ASSISTANT")
ASSISTANT_TIMEOUT_S = float(os.environ.get("GHOSTTRACE_ASSISTANT_TIMEOUT_S", "45"))
ALERT_DISCLAIMER = ("Automated detection - verify before action. Positions, sizes and "
                    "scores come from an unvalidated sonar pipeline and heuristic "
                    "models.")
NOT_IN_CORPUS = "authority not in corpus"
