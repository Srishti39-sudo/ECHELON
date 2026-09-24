"""Every colour, label and piece of text the map shows.

Nothing visual is hard-coded in the map builder. Change a value here and the
whole rendering follows, which is the same rule hazard_config.py applies to
thresholds.

The palette is DeepEcho's, ported from the application's own design tokens
(frontend/config/theme.ts) so the map reads as part of the product rather than
as a separate tool bolted on. The direction stated there is a hydrographic
survey console, not a consumer product: deep navy, one teal accent, neutral
greys, flat surfaces, real borders instead of shadows.

The status colours are desaturated on purpose, and that is worth keeping. This
is a warning system, and a warning that shouts on every row stops being read.
A survey with two hundred low-severity contacts must not look like an
emergency, or the one that is will not stand out.

The tone for every string here, also DeepEcho's: plain, direct, no exclamation
marks, no emoji, nothing that sounds pleased with itself. This is read on a
working deck.
"""

from __future__ import annotations

# --- Palette ---------------------------------------------------------------
# Keys match the token names in frontend/config/theme.ts so the two can be
# compared by eye when either changes.

COLOR = {
    "navy": "#1F3864",
    "navy_deep": "#152845",
    "navy_tint": "#EAEEF5",
    "teal": "#0E6E7A",
    "teal_deep": "#0A5560",
    "teal_tint": "#E4F0F1",

    "background": "#F6F7F9",
    "surface": "#FFFFFF",
    "surface_muted": "#F0F2F5",
    "border": "#DCE0E6",
    "border_strong": "#BFC6D0",

    "text": "#1A1D21",
    "text_muted": "#59616D",
    "text_faint": "#868E9A",
    "text_inverse": "#FFFFFF",

    "alert_text": "#8A2E2E",
    "alert_surface": "#F9EDED",
    "alert_border": "#E0BDBD",
    "caution_text": "#7A5200",
    "caution_surface": "#FCF5E6",
    "caution_border": "#E3D0A4",
    "steady_text": "#2F5D3A",
    "steady_surface": "#EEF4EF",
    "steady_border": "#C3D6C9",
}

FONT_SANS = ("system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', "
             "Arial, sans-serif")
FONT_MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

# --- Severity tiers --------------------------------------------------------
# One entry per tier in hazard_config.SEVERITY_TIERS. `marker` is what a
# detection is drawn in, `fill` what a hotspot circle is filled with, and
# `surface`/`border` dress the panel rows.

TIER_STYLE = {
    "critical": {
        "label": "Critical",
        "marker": "#8A2E2E",
        "fill": "#C2453F",
        "surface": COLOR["alert_surface"],
        "border": COLOR["alert_border"],
        "text": COLOR["alert_text"],
    },
    "medium": {
        "label": "Medium",
        "marker": "#7A5200",
        "fill": "#C08A1E",
        "surface": COLOR["caution_surface"],
        "border": COLOR["caution_border"],
        "text": COLOR["caution_text"],
    },
    "low": {
        "label": "Low",
        "marker": "#2F5D3A",
        "fill": "#5C8A68",
        "surface": COLOR["steady_surface"],
        "border": COLOR["steady_border"],
        "text": COLOR["steady_text"],
    },
}

# A tier the style table has not been taught. Neutral rather than alarming:
# guessing a colour for an unknown tier would assert a risk level.
TIER_FALLBACK = {
    "label": "Unclassified",
    "marker": COLOR["text_muted"],
    "fill": COLOR["border_strong"],
    "surface": COLOR["surface_muted"],
    "border": COLOR["border"],
    "text": COLOR["text_muted"],
}

# Heat runs cool-to-warm through the same tier colours, so the heatmap and the
# markers agree about what red means.
HEAT_GRADIENT = {0.0: "#5C8A68", 0.45: "#C08A1E", 0.8: "#C2453F", 1.0: "#8A2E2E"}

# --- Marker geometry -------------------------------------------------------

DETECTION_RADIUS_PX = 6
DETECTION_WEIGHT = 2
DETECTION_FILL_OPACITY = 0.75

# A merged detection is drawn slightly larger, so a contact seen from four
# tiles is visibly more corroborated than one seen once.
MERGED_RADIUS_BONUS_PX = 1.5
MERGED_RADIUS_CAP_PX = 12

HOTSPOT_FILL_OPACITY = 0.18
HOTSPOT_WEIGHT = 2
# A hotspot ring is drawn at this fraction of the grid cell, so the ring sits
# inside its own cell rather than bleeding into the neighbours.
HOTSPOT_RADIUS_FRACTION = 0.42

TILE_OUTLINE_WEIGHT = 1
TILE_OUTLINE_OPACITY = 0.45

# Heat radius and blur, as a fraction of the hotspot grid. Tied to GRID so the
# heat layer stays meaningful if the grid is retuned.
HEAT_RADIUS_FRACTION = 0.10
HEAT_BLUR_FRACTION = 0.075
HEAT_MIN_OPACITY = 0.25

# --- Strip preview ---------------------------------------------------------
# The sonar image is embedded in the HTML as a data URI so the file works with
# nothing beside it. A full survey strip would make that file enormous, so the
# preview is downscaled. Coordinates are unaffected: the overlay is placed in
# original pixel units regardless of how many pixels the preview itself has.

PREVIEW_MAX_EDGE = 2200
PREVIEW_JPEG_QUALITY = 78

# The colour outside the survey. Leaflet's own default is #ddd, which reads as
# a broken page rather than as canvas. A tall strip in a wide window leaves
# large margins no matter how well it is fitted, so those margins have to look
# deliberate.
MAP_SURROUND = COLOR["background"]

# A thin outline around each strip's footprint, so the survey's extent is
# legible against that surround instead of dissolving into it.
STRIP_OUTLINE = COLOR["border_strong"]
STRIP_OUTLINE_WEIGHT = 1

# Horizontal gap between strips when several are laid out side by side in a
# relative-coordinate survey. See the note in hazard_mapview about what that
# layout does and does not mean.
STRIP_GUTTER_PX = 220

# --- Layer names -----------------------------------------------------------
# These are what the operator reads in the layer control, so they are wording,
# not identifiers.

LAYERS = {
    "strip": "Sonar imagery",
    "tiles": "Survey tiles",
    "heat": "Severity heatmap",
    "detections": "All detections",
    "critical": "Critical hazards",
    "medium": "Medium hazards",
    "low": "Low hazards",
    "hotspots": "Hotspots",
}

# Which layers are on when the map opens. Everything is available; only the
# ones that answer "where do I go first" are lit.
LAYERS_ON_BY_DEFAULT = ("strip", "hotspots", "detections")

# --- Text ------------------------------------------------------------------

TEXT = {
    "app": "DeepEcho",
    "subtitle": "Survey hazard intelligence",
    "relative_mode": "Relative Survey Coordinates (px)",
    "relative_note": (
        "This survey has no navigation. Positions are pixel offsets within the "
        "sonar strip, not geographic coordinates. Nothing on this map is a GPS "
        "position."),
    "geo_mode": "Geo-referenced Survey",
    "geo_note": (
        "Positions are interpolated from the survey's navigation. Relative pixel "
        "coordinates are retained alongside them."),
    "multi_strip_note": (
        "Strips are laid out side by side so they can be viewed together. Their "
        "separation on screen is a display convenience and carries no "
        "geographic meaning."),
    "heat_note": "Heat is weighted by severity, never by how many contacts are present.",
    "priority_heading": "Priority order",
    "start_here": "Start here",
    "legend_heading": "Severity",
    "risk_label": "Risk index",
    "risk_note": ("The worst detection in the cell, plus a quarter of everything "
                  "else in it. An index, not a percentage: it has no upper bound "
                  "and a crowded cell exceeds 1.0."),
    "total_severity_label": "total severity",
    "max_severity_label": "worst single",
    "no_detections": (
        "The detector returned nothing above the confidence threshold. The "
        "survey ran; there is simply nothing to rank."),
    "disclaimer_heading": "Not an official procedure",
    "demo_banner": "SYNTHETIC DEMO DATA",
    "demo_note": (
        "Every contact on this map was generated by the demo script. No sonar "
        "was recorded, no real object was detected, and nothing here is "
        "evidence of anything."),
}


def tier_style(tier: str) -> dict:
    """Styling for a severity tier, with a neutral fallback for an unknown one."""
    return TIER_STYLE.get(str(tier), TIER_FALLBACK)
