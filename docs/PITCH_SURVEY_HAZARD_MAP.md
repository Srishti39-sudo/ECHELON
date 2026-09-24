# Survey Hazard Map: what every element means and why it exists

Pitch preparation for the `/map` page. Everything below is read from the code
(`frontend/src/survey/`, `hazard_hotspots.py`, `hazard_severity.py`,
`hazard_coverage.py`, `hazard_mapview.py`) and the export of one processed
survey. Numbers quoted are from "Synthetic line A (geotag selftest)".

---

## 1. Why a map at all

A detector produces boxes on tiles. A tile is a 640 pixel square cut out of a
sonar strip. On its own a box tells a survey lead nothing they can act on:
they cannot send a diver to "tile 512_0, pixel 845". The map is where boxes
become **places**.

The page answers four operational questions, in order:

1. **Where are the hazards?** Every detection placed on the earth at a real
   latitude and longitude, from the survey's own navigation.
2. **Which one first?** Detections grouped into hotspots and ranked by
   consequence, with a recommended action per hotspot.
3. **What did the sonar actually see?** The mission replay and the coverage
   panel show which seabed was imaged, and where the blind spots are.
4. **Where do we go back?** Re-look lines, exportable as GPX, for the gaps
   and for the contacts worth a second pass.

Without the map, the product is a classifier. With it, the product is a
survey planning tool. That is the sentence for a judge.

---

## 2. Where the data comes from

The page reads one folder: `data/surveys/<survey-id>/`.

| File | Written by | Read by |
|---|---|---|
| `export.json` | survey engine (`hazard_map.py`) | stats, hotspot table, selected hotspot |
| `map.html` | `hazard_mapview.py` | the embedded dark map (an iframe) |
| `actions.csv` | `hazard_export.py` | the Action list button |
| `coverage.json` | `hazard_coverage.py`, cached on first request | Coverage and blind spots |
| navigation sidecars (`*.nav.json`) | `sonar_ingest.py` at ingest | mission replay, coverage |

Nothing on this page is computed in the browser except sorting and filtering.
The backend serves files; the engine that wrote them is the source of truth.
This is deliberate: the interface and the archived export can never disagree.

---

## 3. Top of the page

**Survey dropdown.** Every folder under `data/surveys` with a readable
`export.json`. Synthetic surveys are flagged by the engine in the export's
metadata, never inferred from the name.

**Action list.** Downloads `actions.csv`: one row per hotspot with rank,
position, dominant class, tier and recommended action. This is the deliverable
a survey lead hands to the vessel or the dive team.

**Survey report.** Opens the printable report page for the same survey.

### The seven stat cards

| Card | Value | Meaning |
|---|---|---|
| Total detections | 3 | Objects after deduplication. Overlapping tiles see the same object twice; boxes of the same class within 60 px on the same strip are merged. |
| Hotspots | 2 | 512 by 512 pixel grid cells that contain at least one detection. |
| Total severity | 0.90 | Sum of every detection's severity across the survey. |
| Critical | 0 | Hotspots in the critical tier. |
| Filtered false positives | 0 | Detections the verifier suppressed. Kept in the export with reasons, excluded from every hotspot. |
| Coordinates | Geo-referenced | The survey's coordinate mode. The alternative is "Relative Survey Coordinates", meaning no navigation and pixel positions only. |
| Survey strips | 1, 2 tiles | One strip image came out of the XTF and was cut into two tiles. |

**The blue note** states position provenance: interpolated from the survey's
navigation, with relative pixel coordinates kept alongside. On a survey with
no navigation this note changes and the map is not drawn.

---

## 4. Severity, in one paragraph

Every detection carries three numbers side by side:

```
severity = class_weight * confidence
```

`class_weight` is what it costs to be wrong about that class. A mine is 1.0,
a shipwreck 0.75, fishing gear 0.5, fish near zero. `confidence` is the
detector's calibrated score. So a 58 percent fishing gear detection has
severity 0.29, and a 58 percent mine would have 0.58.

Tiers are fixed thresholds on a single detection's severity:

| Tier | Severity |
|---|---|
| Critical | 0.75 and up |
| Medium | 0.40 to 0.75 |
| Low | below 0.40 |

Some classes have a floor. A shipwreck is never reported below a certain tier
regardless of score, because a wreck is a navigation hazard by definition.

The one rule to repeat: **severity is never read out of the model's
confidence alone.** Confidence says how sure the model is. Class weight says
what it means if the model is right. Both are shown, always.

---

## 5. The embedded map (dark header panel)

This is `map.html`, a self-contained Leaflet page the engine wrote. It has no
dependency on the backend and opens from a USB stick on a vessel with no
network. The page embeds it in an iframe. The frame starts inert ("Click to
use the map") because a Leaflet map inside an iframe swallows scroll events.

**Header badges.** Coordinate mode, filtered count, tiles processed.

**Left column.** The same survey stats, then the severity legend with the
thresholds above, then the note "Heat is weighted by severity, never by how
many contacts are present."

**Layer checkboxes, right side.**

| Layer | What it draws |
|---|---|
| Sonar imagery | The strip image itself, warped onto its georeferenced footprint. Only offered when navigation can place every pixel. |
| Survey tiles | Outlines of the 640 px tiles. |
| Severity heatmap | A density layer where each detection contributes its severity, not a count of one. |
| All detections | Every box as a marker, coloured by tier. |
| Critical, Medium, Low hazards | The same markers split by tier. |
| Filtered false positives | Suppressed detections, drawn differently, off by default. |
| Hotspots | The ranked cells with their popups. |

**Scale bar** bottom left. 20 m on this survey, so the whole line is a few
hundred metres.

### The hotspot popup (H001)

| Field | Value | Meaning |
|---|---|---|
| Priority | Rank 1 | Position in the sort by total severity, then worst single. |
| Dominant hazard | fishing_gear | The class contributing the most severity in the cell, not the most members. |
| Detections | 2 | Members in the cell. |
| Total severity | 0.5265 | Sum of members' severities. Can exceed 1. |
| Worst single | 0.2885 | Highest member severity. This sets the tier. |
| Risk index | 0.3480 | Worst single plus 0.25 times the rest. Density raises risk without letting many small contacts outrank one dangerous one. An index, not a probability. |
| Tier | Low | 0.2885 is below 0.40. |
| Centroid | x 845.3, y 205.9 px | Mean member position in strip pixels. |
| Latitude, Longitude | 19.075715, 72.878481 | The centroid interpolated through the navigation. |
| Action | Schedule ghost-gear recovery | Looked up from the dominant class. |

**The rationale sentence** underneath is generated from those numbers by the
engine, not by a language model. It says why this cell ranks where it does:
"Ranked 1 on a total severity of 0.5265 across 2 detections. The worst is
fishing_gear at confidence 0.577 and class weight 0.5, giving severity
0.2885, and 1 other detection adding 0.238 more severity."

**Why rank 1 can be Low.** Rank is relative within this survey. Tier is
absolute against fixed thresholds. In a survey with nothing dangerous, the
top hotspot is still Low, and the page says so rather than inflating it.

---

## 6. Priority hotspots table

The same hotspots as a table, with filters by class, severity tier and
priority. Columns: rank, hazard class with hotspot id, total severity with a
tier dot, member count, recommended action. Click a row to select it.

Recommended actions come from a fixed table keyed on class:

| Class | Action |
|---|---|
| mine, ordnance, uxo | Deploy EOD team |
| human, victim | Initiate SAR |
| shipwreck, wreck, ship, aircraft | Flag navigation hazard |
| fishing_gear, net | Schedule ghost-gear recovery |
| debris, bottle, can | Schedule cleanup |
| anything unknown | Send for expert identification |

---

## 7. Selected hotspot panel

The full record for the selected cell.

**Top row.** Id, dominant class, tier and rank badge, recommended action.

**Three metric tiles.** Total severity, worst single, risk index, each with
its one-line definition printed under it so nobody has to remember.

**Confidence.** The top detection's confidence and the detector's maximum raw
score in the cell.

**Detections in this hotspot.** One row per member:

| Column | Meaning |
|---|---|
| Class | The detector's class after mapping to the corpus vocabulary. |
| Confidence | Verified percentage, with the raw detector score under it. |
| Size | Length, width and height in metres, from pixel size times the navigation's metres per pixel, and height from the acoustic shadow length. |
| Severity | `weight * confidence`, with the weight shown. |
| Evidence | The tile the box was found on. |

**Verification notes.** Expand to see what the verifier measured: shadow
direction, nadir proximity, clutter, dropouts. Every cue is stored with its
raw measurement.

**Source tiles.** The tile filenames, so a contact can be traced back to the
exact image.

**Ask the assistant about this hazard.** Hands this hotspot to the Assistant
page as survey context. The map's severity travels with it and the assistant
displays it as given, because the map owns urgency.

**Open map full screen.** The same `map.html` in its own tab.

---

## 8. Mission replay: why the map moves

A processed survey is a static result. The replay shows **how it was
collected**, in ping-time order, from the navigation sidecar. It exists for
three reasons:

1. **Trust.** A judge or a survey lead can watch the towfish track, see the
   swath sweep across the seabed, and watch each contact appear at the moment
   the sonar passed over it. Nothing appears where the sonar did not go.
2. **Data quality.** Rows the ingest marked degraded (dropout, attitude
   excursion, interpolated navigation, a navigation jump) are drawn in red.
   A contact found on a red row is worth less trust, and the replay makes
   that visible without reading a log.
3. **Briefing.** Playing a survey back at 10x or 60x is the fastest way to
   explain to a crew what was done and where.

### Elements

| Element | Meaning |
|---|---|
| Basemap checkbox | OpenStreetMap tiles under the survey. Off for a clean view. |
| Blue line | The towfish track, good rows. |
| Red line | The track through degraded rows. |
| Grey swath | The seabed footprint swept so far, port and starboard edges perpendicular to heading. |
| Green dots with labels | Contacts, appearing at the ping time of the row they were found on, coloured by tier. |
| Distance surveyed | Length along the projected track up to the current time. |
| Area imaged | Union of the swept footprint, good rows only, nadir strip excluded. Computed exactly at 60 checkpoints and interpolated between. |
| Contacts found, Filtered | Running counts. |
| Sonar rows good | Quality of the row at the current playhead. |
| Contact list | Each contact with confidence and the survey clock when it was found. |
| Play, reset, 1x 10x 60x | Playback controls. |
| Timeline | Green ticks mark contact times. Red segments would mark degraded rows. |
| Footer | The exact provenance: survey start time, sampling, how swath and area were computed. |

**Honest note for judges.** This synthetic survey's coordinates fall on a
residential area in Mumbai on the basemap. That is because the synthetic XTF
generator was given a test origin, and the engine placed the data exactly
where the navigation said. It is a demonstration that positions come from
the file and nowhere else. Say it before they notice.

---

## 9. Coverage and blind spots

The question a survey lead asks after any line: **did we actually see the
whole area, and where do we have to go back?**

| Card | Value | Meaning |
|---|---|---|
| Seabed imaged | 0.0154 km² | Union of the swept footprint, good rows only, nadir excluded. |
| Share of survey hull imaged | 97.5% | Imaged area divided by the convex hull of the whole swath. The remainder is gaps and nadir. |
| Nadir blind strip | 400 m² | The band directly under the towfish where pixels exist but are interpolated, not resolved. Mean width 2.51 m here. A lower bound; the transducer's beam pattern usually widens it and is not recorded. |
| Degraded-row gaps | 0 m² | Seabed under rows marked dropout, attitude, interpolated or nav_jump, unless another row imaged the same seabed. Red border when nonzero. |
| Re-look lines | 2 | Planned passes: 0 for gaps, 2 for contacts worth a second look. |

**Legend.** Imaged seabed (green), nadir blind strip (grey), coverage gap (red
hatch), survey hull (dashed), re-look line (purple, numbered at its start).

**Re-look lines (GPX, CSV).** Downloadable waypoints. GPX loads directly into
a chart plotter or handheld GPS. The page says, in its own subtitle, that
these are a planning heuristic and not a navigation procedure: check depth,
traffic and turning circle first.

**How this was computed.** Expands to the method: local azimuthal equidistant
projection on WGS84, rows sampled along track, consecutive samples joined
into quads, footprints unioned with shapely. Consecutive samples far apart
are not joined, so a navigation jump never paints seabed that was not passed
over.

**Physics in one line each, if asked:**

- Swath half-width per row: `sqrt(F² - altitude²)` where F is the far slant
  range. Higher altitude means narrower ground swath.
- Nadir gap: ground ranges between 0 and `sqrt((altitude + ds)² - altitude²)`
  are served by one slant-range sample interval, where ds is the native sample
  spacing. They are not independently resolved.

---

## 10. What a survey without navigation looks like

Upload a plain image with no CSV and no corners, and:

- Coordinate mode becomes "Relative Survey Coordinates".
- Latitude and longitude are null everywhere; positions are pixels.
- The embedded map is not drawn; hotspots are still ranked and the action
  list is still written, in pixel coordinates.
- Mission replay and coverage both answer `available: false` with a reason.

This is a complete, usable result. The system says what it does not know
instead of guessing.

---

## 11. Questions to expect

**"Why hotspots instead of listing every detection?"** A crew is dispatched to
a place, not a box. Grouping by cell gives one place per decision. Ranking by
summed severity, not count, puts one mine above ten cans.

**"Why is rank 1 only Low?"** Rank is relative within the survey; tier is
absolute. Both are shown so neither misleads.

**"Where do the coordinates come from?"** For a raw XTF, from the per-ping
navigation in the file, interpolated to each pixel. For images, from a
control-point CSV or four corners the operator supplies. Never from anywhere
else.

**"Why is the synthetic survey on land in Mumbai?"** The synthetic generator's
test origin. The engine plotted exactly what the navigation said. That is the
point.

**"What is the risk index?"** Worst single severity plus a quarter of the rest.
It rewards density without letting it dominate. It is an index, not a
probability, and the page labels it that way.

**"How do you know the nadir strip is 2.5 m?"** From altitude and the native
slant-range sample spacing recorded at ingest. It is a lower bound and the
panel says so.

**"Can I use the re-look lines for navigation?"** No. They are straight
planning geometry. The page tells you to check depth, traffic and turning
circle. Sea routes around land are not computed.

**"Does the map need internet?"** The embedded `map.html` does not; it is
self-contained. The replay's basemap does, and can be switched off.

**"What is synthetic here?"** The sonar data in this survey. The physics,
the engine, the coverage geometry and the hotspot ranking are the same code
that runs on real data (validated on a USGS GeoTIFF mosaic).
