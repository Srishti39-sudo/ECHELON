"""The severity engine, and the action policy that hangs off it.

One formula, and it is the whole of it:

    severity = class_weight * confidence

class_weight is what it costs to be wrong about this class of object.
confidence is what the detector reported. Neither is hidden inside the result:
every record carries class_weight, confidence and severity side by side, along
with how the weight was chosen, so any score in the export can be recomputed by
hand from the record itself.

This is a configurable operational heuristic. It is not an official standard
and does not encode anyone's published doctrine.

HOW A CLASS IS MATCHED
    1. Normalise: lowercase, trim, and collapse spaces and underscores to
       single hyphens. "Ghost_Gear" and "ghost gear" both become "ghost-gear".
    2. Exact match against the table.
    3. Otherwise the longest table key that appears inside the normalised name.
       "naval-mine" contains "mine"; longest-first stops "mine" from being
       picked when a more specific key also matches.
    4. Otherwise UNKNOWN_CLASS_SEVERITY. Never 0.0: a detector trained on
       classes this policy has not been taught must not have its findings
       weighted out of existence.

Step 3 is why the tables can stay short. A checkpoint emitting "moored_mine",
"sea mine" or "Mine" lands on the same weight without any of them being listed.
"""

from __future__ import annotations

import re
from typing import Any

from survey_hazard_map import hazard_config as cfg
_SEPARATORS = re.compile(r"[\s_/]+")
_TRIM = re.compile(r"^[-\s]+|[-\s]+$")


def normalize_class(name: Any) -> str:
    """The canonical spelling used for every table lookup."""
    text = _SEPARATORS.sub("-", str(name or "").strip().lower())
    return _TRIM.sub("", text)


def _lookup(table: dict[str, Any], name: str) -> tuple[Any, str]:
    """(value, how it was matched). The `how` is kept for the audit trail."""
    if name in table:
        return table[name], "exact"
    candidates = [key for key in table if key and key in name]
    if candidates:
        key = max(candidates, key=len)
        return table[key], f"substring:{key}"
    return None, "default"


def class_weight(object_class: Any) -> tuple[float, str]:
    """Consequence weight for a class, and the basis for it."""
    name = normalize_class(object_class)
    value, how = _lookup(cfg.SEVERITY, name)
    if value is None:
        return cfg.UNKNOWN_CLASS_SEVERITY, "default:unknown-class"
    return float(value), how


def recommended_action(object_class: Any) -> tuple[str, str]:
    """Advisory next step for a class, and the basis for it."""
    name = normalize_class(object_class)
    value, how = _lookup(cfg.ACTIONS, name)
    if value is None:
        return cfg.DEFAULT_ACTION, "default:unknown-class"
    return str(value), how


def severity_tier(score: float) -> str:
    """The word for a score. Boundaries live in one table in hazard_config."""
    for tier, floor in cfg.SEVERITY_TIERS:
        if score >= floor:
            return tier
    return cfg.SEVERITY_TIERS[-1][0]


_TIER_RANK = {tier: rank for rank, (tier, _) in enumerate(cfg.SEVERITY_TIERS)}


def highest_tier(tiers: list[str]) -> str:
    """The most severe of several tier words. Lowest tier when given none."""
    known = [t for t in tiers if t in _TIER_RANK]
    if not known:
        return cfg.SEVERITY_TIERS[-1][0]
    return min(known, key=_TIER_RANK.__getitem__)


def class_tier_floor(object_class: Any) -> tuple[str | None, str]:
    """The tier a class is never reported below, and how it was matched."""
    name = normalize_class(object_class)
    value, how = _lookup(cfg.CLASS_TIER_FLOOR, name)
    return (str(value) if value else None), how


def tier_for(object_class: Any, severity: float) -> tuple[str, str]:
    """(tier, basis): the score's word, raised to the class floor if one applies.

    `basis` is "score" when the tiers table alone chose the word and
    "class-floor:<class-key>" when the floor did, so the audit trail says
    which. A class the detector could not assert has no floor; it was renamed
    to unknown before scoring and is not in the table.
    """
    tier = severity_tier(severity)
    floor, how = class_tier_floor(object_class)
    if floor and _TIER_RANK.get(floor, 99) < _TIER_RANK.get(tier, 99):
        return floor, f"class-floor:{how.split(':', 1)[-1] if ':' in how else normalize_class(object_class)}"
    return tier, "score"


def apply_confidence_floor(detection: dict[str, Any]) -> dict[str, Any]:
    """Withhold a class the detector is not confident enough to assert.

    Applied before scoring, because it changes what the object is called and
    therefore what it weighs. The box is never dropped: it is relabelled to
    DOWNGRADE_LABEL and the original call is kept in `downgraded_from`, which
    is the field name DeepEcho's detector already uses.

    This is not a de-escalation. See the note beside CLASS_CONFIDENCE_FLOOR in
    hazard_config: withholding `human` makes it quieter, withholding `aircraft`
    makes it louder, and both are the policy working.
    """
    floor = cfg.CLASS_CONFIDENCE_FLOOR.get(normalize_class(detection.get("class")))
    if floor is None:
        return detection
    confidence = float(detection.get("confidence", 0.0))
    if confidence >= floor:
        return detection

    original = str(detection.get("class"))
    detection["class"] = cfg.DOWNGRADE_LABEL
    detection["class_withheld"] = original
    detection["class_floor"] = floor
    detection["downgraded_from"] = cfg.DOWNGRADE_NOTE.format(
        cls=original, confidence=confidence, floor=floor)
    return detection


def score_detection(detection: dict[str, Any]) -> dict[str, Any]:
    """Attach weight, severity, tier and action to one detection, in place."""
    weight, basis = class_weight(detection.get("class"))
    confidence = float(detection.get("confidence", 0.0))
    severity = weight * confidence

    action, action_basis = recommended_action(detection.get("class"))
    tier, tier_basis = tier_for(detection.get("class"), severity)
    detection.update({
        "class_normalized": normalize_class(detection.get("class")),
        "class_weight": round(weight, 4),
        "confidence": round(confidence, 4),
        "severity": round(severity, 4),
        "severity_tier": tier,
        "severity_tier_basis": tier_basis,
        "severity_basis": basis,
        "recommended_action": action,
        "action_basis": action_basis,
    })
    return detection


def tier_counts(severities: list[float]) -> dict[str, int]:
    """How many scores fall in each tier. Every tier is present, even at zero."""
    counts = {tier: 0 for tier, _ in cfg.SEVERITY_TIERS}
    for score in severities:
        counts[severity_tier(score)] += 1
    return counts


def policy() -> dict[str, Any]:
    """The whole policy, as written into the export's configuration block."""
    return {
        "version": cfg.SEVERITY_POLICY_VERSION,
        "formula": "severity = class_weight * confidence",
        "class_weights": dict(cfg.SEVERITY),
        "unknown_class_severity": cfg.UNKNOWN_CLASS_SEVERITY,
        "tiers": {tier: floor for tier, floor in cfg.SEVERITY_TIERS},
        "class_tier_floors": dict(cfg.CLASS_TIER_FLOOR),
        "tier_applies_to": (
            "a detection's own severity, raised to its class's floor tier when "
            "one applies (severity_tier_basis records which); a hotspot takes "
            "the highest tier among its detections, so a hotspot is never "
            "called critical unless it contains a detection that is"),
        "matching": (
            "exact match on the normalised class name, else the longest table "
            "key contained in it, else unknown_class_severity"),
        "actions": dict(cfg.ACTIONS),
        "default_action": cfg.DEFAULT_ACTION,
        "disclaimer": cfg.DISCLAIMER,
    }
