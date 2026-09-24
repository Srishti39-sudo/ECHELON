"""The seam: one call in, structured answer out.

rag.py already does the hard parts -- retrieval, the grounding prompt, the
refusal behaviour -- but its entry point prints to stdout and takes an argparse
namespace, so a server cannot use it. This module builds a parallel path over
the same clean pieces:

    Retriever.search()  ->  the hits, unchanged
    rag.generate()      ->  the answer text, unchanged
    Chunk.meta          ->  the citations that generate() throws away

Nothing here re-implements retrieval or edits the system prompt. What it adds
is what a conversation needs and a one-shot CLI did not: history that survives
a follow-up, intent chosen for the operator instead of by them, and metadata
the interface can render.
"""

from __future__ import annotations

import json
import re
import logging
import time
import unicodedata
from functools import lru_cache
from typing import Any, Iterator

from rag_assistant import rag
from backend import config
from rag_assistant import copilot_tools


log = logging.getLogger("deepecho")


class EngineError(RuntimeError):
    """A failure inside rag.py, converted from SystemExit.

    rag.py is a CLI and exits the process on a bad state. Under uvicorn that
    would take the worker down mid-request, so every call into it is funnelled
    through here.
    """


def _transient(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in config.TRANSIENT_PROVIDER_ERRORS)


def provider_order(preferred: str) -> list[str]:
    """The preferred provider, then the others, as failover candidates."""
    if not config.PROVIDER_FAILOVER:
        return [preferred]
    return [preferred] + [p for p in sorted(rag.PROVIDERS) if p != preferred]


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        raise EngineError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Loaded once, not per request
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_retriever() -> "rag.Retriever":
    return _guard(rag.Retriever.load, ef_search=config.EF_SEARCH)


@lru_cache(maxsize=1)
def get_catalog() -> "rag.Catalog | None":
    return _guard(rag.Catalog.load)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _content_terms(text: str) -> list[str]:
    """Unigrams worth carrying. rag.tokenize also emits bigrams; not wanted here."""
    return [w for w in WORD_RE.findall(text.lower())
            if w not in rag.STOP and len(w) > 2]


def is_follow_up(message: str, history: list[dict]) -> bool:
    """Does this turn only make sense against the previous one?

    Three signals, any of which is enough: it is short, it opens like a
    continuation, or it refers to something by pronoun. All three are cheap and
    wrong in the same harmless direction -- carrying a few extra terms into a
    TF-IDF query costs nothing, while failing to carry them means "what if it's
    at 40 metres?" searches the corpus for the word "metres".
    """
    if not history:
        return False
    text = message.strip().lower()
    if len(text.split()) <= config.FOLLOW_UP_MAX_WORDS:
        return True
    if text.startswith(config.FOLLOW_UP_OPENERS):
        return True
    return any(re.search(rf"\b{re.escape(ref)}\b", text) for ref in config.FOLLOW_UP_REFERENTS)


def carried_terms(history: list[dict]) -> list[str]:
    """Topic words from earlier user turns, most recent first, deduplicated.

    Only user turns. Assistant answers are long, and folding them in lets the
    assistant's own wording steer the next retrieval instead of the operator's.
    """
    seen: list[str] = []
    for turn in reversed(history[-config.HISTORY_TURNS:]):
        if turn.get("role") != "user":
            continue
        for term in _content_terms(turn.get("content", "")):
            if term not in seen:
                seen.append(term)
            if len(seen) >= config.CARRY_TERMS:
                return seen
    return seen


def condense(message: str, history: list[dict], record: dict, mode: str) -> str:
    """The standalone query actually used for retrieval.

    The operator's words, plus the topic they are still talking about, plus the
    same task expansion the CLI uses. Deterministic: no model call, so a
    follow-up costs exactly one round trip like any other turn.
    """
    parts = [message.strip(), term_expansions(message)]
    if is_follow_up(message, history):
        parts.append(" ".join(carried_terms(history)))
    if record:
        parts.append(" ".join(str(record[f]) for f in ("label", "visual_description", "notes")
                              if record.get(f)))
        parts.append(class_synonyms(record.get("label")))
    parts.append(rag.QUERY_EXPANSION.get(mode, ""))
    return " ".join(p for p in parts if p.strip()).strip()


def term_expansions(message: str) -> str:
    """Corpus words for the operator's words. Deduplicated, order preserved."""
    out: list[str] = []
    for word in _content_terms(message):
        for extra in config.TERM_EXPANSIONS.get(word, "").split():
            if extra not in out:
                out.append(extra)
    return " ".join(out)


def class_synonyms(label: str | None) -> str:
    """Corpus words for a detector's class name. Longest match wins."""
    if not label:
        return ""
    name = _normalise(label)
    for phrase in sorted(config.CLASS_SYNONYMS, key=len, reverse=True):
        if phrase in name:
            return config.CLASS_SYNONYMS[phrase]
    return ""


def history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["CONVERSATION SO FAR (oldest first). Use it to resolve what the "
             "operator means by \"it\" or \"that\". It is context, not a source: "
             "it is never citable and never overrides the SOURCES block."]
    for turn in history[-config.HISTORY_TURNS:]:
        role = "OPERATOR" if turn.get("role") == "user" else "ASSISTANT"
        content = (turn.get("content") or "").strip()
        if len(content) > config.HISTORY_CHARS:
            content = content[:config.HISTORY_CHARS].rstrip() + " [...]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

def _unclassified(record: dict) -> bool:
    """Did the classifier actually identify this, or not?

    Low confidence counts as not. The classifier's uncertainty is the operator's
    uncertainty, and rounding it away is how a maybe becomes a fact.
    """
    if not record:
        return False
    label = str(record.get("label") or "").strip().lower()
    if label in config.UNKNOWN_LABELS:
        return True
    confidence = record.get("confidence")
    return isinstance(confidence, (int, float)) and confidence < config.ANOMALY_CONFIDENCE_FLOOR


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def route_intent(message: str, record: dict) -> str:
    """Pick the mode so the operator never has to.

    Order is deliberate and is a safety property, not a style choice. An
    explicit request for a report wins outright. Anything that says the object
    is unknown routes to the anomaly path, and so does an unclassified record,
    both of them ahead of the explain keywords -- otherwise "what is this?"
    asked about an unidentified object would route to explain, and explain is
    the one mode that is allowed to name what something is.
    """
    text = message.strip().lower()

    if _matches_any(text, config.INTENT_KEYWORDS["report"]):
        return "report"
    if _matches_any(text, config.INTENT_KEYWORDS["anomaly"]):
        return "anomaly"
    if _unclassified(record):
        return "anomaly"
    if _matches_any(text, config.INTENT_KEYWORDS["explain"]):
        return "explain"
    if config.INTENT_LLM_FALLBACK:
        guessed = _llm_intent(message, bool(record))
        if guessed:
            return guessed
    return config.INTENT_WITH_RECORD if record else config.INTENT_WITHOUT_RECORD


_INTENT_SYSTEM = (
    "Classify the operator's message into exactly one of: question, explain, "
    "anomaly, report. Reply with that single word and nothing else."
)


def _llm_intent(message: str, has_record: bool) -> str | None:
    """Optional tiebreak. Any failure falls back to the rules, silently."""
    try:
        backend = rag.PROVIDERS[config.PROVIDER]()
        context = "A detection record is on screen." if has_record else "No detection record."
        reply = backend.complete(_INTENT_SYSTEM, f"{context}\n\nMESSAGE: {message}",
                                 config.MODEL or backend.default_model)
    except Exception:
        return None
    word = reply.strip().split()[0].lower().strip(".,\"'") if reply.strip() else ""
    return word if word in config.INTENT_TO_MODE else None


# ---------------------------------------------------------------------------
# Severity: looked up, never inferred
# ---------------------------------------------------------------------------

def severity_for(record: dict, anomalous: bool) -> str:
    """Severity from the catalog's hazard field, then the class table, else unknown.

    Never read out of the generated text. A risk level lifted from prose is an
    unsourced number wearing a label, which is the one thing this system exists
    not to produce.
    """
    if anomalous:
        return config.ANOMALY_SEVERITY

    name = _normalise(record.get("label") or "")
    if not name:
        return config.DEFAULT_SEVERITY

    catalog = get_catalog()
    if catalog is not None:
        for entry in catalog.entries:
            if name in (_normalise(entry.get("id", "")), _normalise(entry.get("name", ""))):
                hazard = _normalise(entry.get("hazard", ""))
                if hazard in config.HAZARD_SEVERITY:
                    return config.HAZARD_SEVERITY[hazard]

    for phrase in sorted(config.CLASS_SEVERITY, key=len, reverse=True):
        if phrase in name:
            return config.CLASS_SEVERITY[phrase]
    return config.DEFAULT_SEVERITY


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def _pdf_url(meta: dict) -> str | None:
    """Link a citation to the publication it was written from, when there is one."""
    raw = meta.get("source_file")
    if not raw:
        return None
    first = str(raw).split(";")[0].strip()
    if not first:
        return None
    name = first.rsplit("/", 1)[-1]
    if not (config.SOURCES_DIR / name).exists():
        return None
    return f"{config.SOURCES_BASE_URL}/{name}"


def sources_from(hits: list[tuple[Any, float]]) -> list[dict]:
    """Turn the hits into citation records.

    Numbering is the whole point: `n` is assigned by enumerating this exact list
    in this exact order, which is the order rag.format_sources() used when it
    built the [S1], [S2] tags in the prompt. Reorder or filter between the two
    and every marker in the answer silently points at the wrong document.
    """
    out = []
    for n, (chunk, score) in enumerate(hits, start=1):
        meta = chunk.meta
        text = chunk.text
        out.append({
            "n": n,
            "id": chunk.id,
            "title": meta.get("title") or meta.get("doc_id") or chunk.id,
            "section": meta.get("section") or None,
            "snippet": text if len(text) <= config.SNIPPET_CHARS
                       else text[:config.SNIPPET_CHARS].rstrip() + " [...]",
            "authority": meta.get("authority"),
            "status": meta.get("status"),
            "doc_id": meta.get("doc_id"),
            "path": meta.get("path"),
            "score": round(float(score), 4),
            "pdf_url": _pdf_url(meta),
        })
    return out


def matches_from(matches: list[tuple[dict, float]]) -> list[dict]:
    return [{
        "rank": rank,
        "id": entry.get("id", ""),
        "name": entry.get("name", ""),
        "object_class": entry.get("class", ""),
        "hazard": entry.get("hazard", ""),
        "similarity": round(float(score), 4),
        "confirms": entry.get("confirms", ""),
        "rules_out": entry.get("rules_out", ""),
        "source": entry.get("source", ""),
        "status": entry.get("status", ""),
    } for rank, (entry, score) in enumerate(matches, start=1)]


def cited_markers(text: str) -> tuple[set[int], set[int]]:
    """([Sn] numbers, [Dn] numbers) cited in the text, in any bracket form."""
    sources: set[int] = set()
    data: set[int] = set()
    for group in re.findall(config.CITATION_GROUP_PATTERN, text):
        sources.update(int(n) for n in re.findall(config.CITATION_REF_PATTERN, group))
        data.update(int(n) for n in re.findall(config.DATA_CITATION_REF_PATTERN, group))
    return sources, data


def check_grounding(text: str, source_count: int,
                    data_count: int | None = None) -> tuple[bool, bool]:
    """(grounded, refusal).

    Grounded is mechanical: at least one [Sn] marker, and every marker resolves
    to a source that was actually retrieved. A marker pointing past the end of
    the list means the model numbered something that was never given to it.

    Refusal is reported separately because it is the designed behaviour, not a
    failure. An answer is routinely both: cited throughout and still saying the
    standoff distance is not specified in the sources. The flag means "part of
    this answer is the assistant declining to fill a gap", which is exactly what
    an operator should see rendered differently from a confident procedure.
    """
    cited, data = cited_markers(text)
    if data_count is None:
        grounded = bool(cited) and all(1 <= n <= source_count for n in cited)
    else:
        # A copilot answer is grounded by survey records as much as by passages:
        # "no mine-like object was found [D1]" needs no corpus citation. Every
        # marker of either kind must still resolve.
        grounded = (bool(cited or data)
                    and all(1 <= n <= source_count for n in cited)
                    and all(1 <= n <= data_count for n in data))
    plain = re.sub(r"\s+", " ", re.sub(config.MARKDOWN_NOISE, "", text)).lower()
    refusal = any(re.search(p, plain) for p in config.REFUSAL_PATTERNS)
    return grounded, refusal


# ---------------------------------------------------------------------------
# GhostTrace context: survey data, never a source
# ---------------------------------------------------------------------------

def normalise_ghosttrace(context: dict | None) -> dict | None:
    """Validate a GhostTrace handoff into the agreed shape, nulls kept.

    The HTTP layer has already validated it, but the evaluation suite and the
    tests call the engine directly, and both paths must see the same shape.
    """
    if not context:
        return None
    from backend.schemas import GhostTraceContext
    try:
        return GhostTraceContext.model_validate(context).to_engine_context()
    except Exception as exc:  # pydantic.ValidationError, kept out of the import
        raise EngineError(f"Invalid GhostTrace context: {exc}") from exc


def ghosttrace_citation(context: dict) -> str:
    return config.GHOSTTRACE_CITATION.format(
        survey=context.get("survey_id") or "unknown survey",
        detection=context.get("detection_id") or "unknown detection")


def _num(value: Any) -> str:
    """A number exactly as the context carries it. No rounding, no reformatting."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return f"{value:.1f}"
    return str(value)


def _given(value: Any, unit: str = "") -> str:
    if value is None or value == "":
        return "not available"
    return f"{_num(value)}{unit}" if isinstance(value, (int, float)) else str(value)


def priority_lines(priority: dict | None) -> list[str]:
    """The priority as its terms, value x weight = contribution, line by line.

    The sum and the recomputed score are the only figures computed here, and
    they are computed so the model never has to: a model doing arithmetic in
    prose is how a 0.7106 becomes a 0.71 becomes a 0.72.
    """
    if not priority:
        return ["PRIORITY: not available"]
    lines = [f"PRIORITY: rank {_given(priority.get('rank'))}, tier "
             f"{_given(priority.get('tier'))}, score {_given(priority.get('score'))}"]
    terms = priority.get("terms") or {}
    if not terms:
        lines.append("  terms: not available")
        return lines
    lines.append("  terms (value x weight = contribution):")
    contributions: list[float] = []
    multipliers: list[tuple[str, Any]] = []
    complete = True
    for name, term in terms.items():
        term = term or {}
        value, weight, contribution = term.get("value"), term.get("weight"), term.get("contribution")
        if weight is None and contribution is None:
            multipliers.append((name, value))
            continue
        if isinstance(contribution, (int, float)):
            contributions.append(float(contribution))
        else:
            complete = False
        lines.append(f"  - {name}: {_given(value)} x {_given(weight)} = {_given(contribution)}")
    for name, value in multipliers:
        lines.append(f"  - {name}: value {_given(value)} (a multiplier: it has no weight "
                     "and no contribution of its own)")
    if contributions and complete:
        total = round(sum(contributions), 4)
        lines.append(f"  sum of the contributions above: {_num(total)}")
        score = priority.get("score")
        if len(multipliers) == 1 and isinstance(multipliers[0][1], (int, float)):
            product = round(float(multipliers[0][1]) * total, 4)
            agrees = isinstance(score, (int, float)) and abs(product - float(score)) <= 0.001
            lines.append(
                f"  {multipliers[0][0]} {_num(multipliers[0][1])} x {_num(total)} = {_num(product)}"
                + (" (agrees with the stated score)" if agrees
                   else " (does NOT agree with the stated score; report the stated score and say so)"))
    elif not complete:
        lines.append("  sum of the contributions: not available, at least one contribution is missing")
    return lines


def render_ghosttrace(context: dict) -> str:
    """The context as plain labelled lines. Every figure is copied, none derived
    except the priority sum and product above."""
    c = context
    synthetic = c.get("synthetic")
    lines = [
        f"CITE AS: {ghosttrace_citation(c)}",
        f"SURVEY: {_given(c.get('survey_id'))}"
        + (f" ({c['survey_title']})" if c.get("survey_title") else ""),
        f"SYNTHETIC DATA: {'YES' if synthetic else 'no' if synthetic is False else 'not stated'}",
        f"DETECTION: {_given(c.get('detection_id'))}",
        f"OBJECT CLASS (detector output, not a confirmed identification): {_given(c.get('object_class'))}",
        ("POSITION: not georeferenced (no latitude/longitude)"
         if c.get("latitude") is None or c.get("longitude") is None
         else f"POSITION: latitude {_num(c['latitude'])}, longitude {_num(c['longitude'])}"),
        f"CONFIDENCE: {_given(c.get('confidence_pct'), '%')}",
    ]
    lines += priority_lines(c.get("priority"))

    activity = c.get("activity")
    if activity:
        lines.append(
            f"WATER-COLUMN ACTIVITY: level {_given(activity.get('level'))}, score "
            f"{_given(activity.get('score'))}, enrichment ratio {_given(activity.get('enrichment_ratio'))}, "
            f"echo clusters near {_given(activity.get('echo_clusters_near'))}, background clusters per "
            f"window {_given(activity.get('background_clusters_per_window'))}")
        if activity.get("limitations"):
            lines.append(f"  limitations: {activity['limitations']}")
    else:
        lines.append("WATER-COLUMN ACTIVITY: not available")

    habitats = c.get("habitat_nearest") or []
    if habitats:
        lines.append("NEAREST HABITAT (from map layers):")
        for h in habitats:
            source = h.get("source")
            if isinstance(source, dict):
                source = source.get("name")
            lines.append(f"  - {h.get('name') or 'unnamed'} ({_given(h.get('kind'))}), "
                         f"{_given(h.get('distance_m'), ' m')} away, layer source: {_given(source)}")
    else:
        lines.append("NEAREST HABITAT: not available")

    def impact(label: str, block: dict | None) -> str:
        top = (block or {}).get("top_impact")
        if not top:
            return f"{label}: none reached"
        return (f"{label}: {top.get('name') or 'unnamed'} ({_given(top.get('kind'))}), probability "
                f"{_given(top.get('probability'))}, first arrival {_given(top.get('first_arrival_hours'), ' hours')}")

    drift = c.get("drift")
    if drift:
        lines.append(f"DRIFT FORECAST (a model forecast, not an observation): mode {_given(drift.get('mode'))}; "
                     f"stranding probability {_given(drift.get('stranding_probability'))}")
        lines.append("  " + impact("top sensitive impact", drift))
    else:
        lines.append("DRIFT FORECAST: not available")
    refloat = c.get("refloat_scenario")
    lines.append("  " + (impact("if refloated, top impact", refloat) if refloat
                         else "if refloated: not available"))

    people = c.get("people")
    if people:
        lines.append(
            f"PEOPLE: propeller hazard level {_given(people.get('propeller_hazard_level'))}; diver "
            f"recommended method {_given(people.get('diver_recommended_method'))}; seabed depth "
            f"{_given(people.get('seabed_depth_m'), ' m')}; current at depth "
            f"{_given(people.get('current_mps_at_depth'), ' m/s')}")
    else:
        lines.append("PEOPLE: not available")

    change = c.get("change")
    lines.append(f"CHANGE SINCE AN EARLIER SURVEY: status {_given((change or {}).get('status'))}, "
                 f"moved {_given((change or {}).get('moved_m'), ' m')}")

    authorities = c.get("authorities") or []
    if authorities:
        lines.append("AUTHORITIES GHOSTTRACE LISTS (pointers only; name one only where SOURCES name it):")
        for a in authorities:
            lines.append(f"  - {_given(a.get('name'))} (situation: {_given(a.get('situation'))})")
    else:
        lines.append("AUTHORITIES GHOSTTRACE LISTS: none")

    # A caveat that only repeats the activity limitations already printed above
    # is dropped here; the interface still shows every caveat.
    limitations = (activity or {}).get("limitations") or ""
    caveats = [caveat for caveat in c.get("caveats") or []
               if not (limitations and limitations in caveat)]
    if caveats:
        lines.append("CAVEATS (show these):")
        lines += [f"  - {caveat}" for caveat in caveats]
    return "\n".join(lines)


def ghosttrace_block(context: dict) -> str:
    return config.GHOSTTRACE_NOTE.format(
        citation=ghosttrace_citation(context),
        synthetic=config.GHOSTTRACE_SYNTHETIC_RULE if context.get("synthetic") else "",
        body=render_ghosttrace(context))


def ghosttrace_record(context: dict) -> dict:
    """The minimum detection record, so routing and class synonyms work.

    Severity is NOT taken from GhostTrace here and GhostTrace's priority is not
    turned into a severity: the rescue queue owns the priority and the interface
    shows it as given, the same rule the survey hazard map follows.
    """
    record: dict[str, Any] = {}
    if context.get("object_class"):
        record["label"] = context["object_class"]
    pct = context.get("confidence_pct")
    if isinstance(pct, (int, float)):
        record["confidence"] = round(float(pct) / 100.0, 4)
    return record


def ghosttrace_query(context: dict) -> str:
    """Words that pull the documents a GhostTrace answer must cite."""
    parts = [config.GHOSTTRACE_QUERY_TERMS]
    for authority in context.get("authorities") or []:
        situation = (authority or {}).get("situation")
        parts.append(config.GHOSTTRACE_SITUATION_TERMS.get(situation or "", ""))
        name = (authority or {}).get("name")
        if name and name != "authority not in corpus":
            parts.append(name)
    for habitat in context.get("habitat_nearest") or []:
        if (habitat or {}).get("name"):
            parts.append(habitat["name"])
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Numbers guard
# ---------------------------------------------------------------------------

def normalise_digits(text: str) -> str:
    """Every Unicode decimal digit as its ASCII digit: "४०" -> "40", "௧௨" -> "12".

    An answer written in Hindi or Tamil may write a figure in its own script.
    The guard compares figures as ASCII, so they are normalised first; otherwise
    an invented "४० मीटर" would pass unseen.
    """
    if not text or text.isascii():
        return text
    out = []
    for ch in text:
        if not ch.isascii() and ch.isdecimal():
            out.append(str(unicodedata.decimal(ch)))
        else:
            out.append(ch)
    return "".join(out)


_NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)*)"
    r"(?:\s*(%|m/s|km|nm|metres?|meters?|m|hours?|hrs?|h|minutes?|mins?|days?|kg|tonnes?))?"
    r"(?![\w])", re.I)


def _allowed_numbers(text: str) -> list[tuple[float, int]]:
    lowered = text.lower()
    for word, digits in config.NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            lowered += f" {digits}"
    out = []
    for token, _unit in _NUMBER.findall(lowered):
        plain = token.replace(",", "")
        try:
            value = float(plain)
        except ValueError:
            continue
        decimals = len(plain.split(".", 1)[1]) if "." in plain else 0
        out.append((value, decimals))
    return out


def unsourced_numbers(answer: str, allowed_text: str) -> list[str]:
    """Figures in the answer that appear in none of the text the model was shown.

    A figure is accepted when it appears verbatim, or as a rounding of a more
    precise figure that does (0.2222 for 0.222225), or as a percentage of a
    fraction that does (35% for 0.35). A figure more precise than anything it
    could have come from is not accepted, because that precision was invented.
    """
    answer = normalise_digits(answer)
    allowed_text = normalise_digits(allowed_text)
    text = re.sub(config.CITATION_GROUP_PATTERN,
                  lambda m: " " if (re.search(config.CITATION_REF_PATTERN, m.group(1))
                                    or re.search(config.DATA_CITATION_REF_PATTERN, m.group(1)))
                  else m.group(0),
                  re.sub(config.MARKDOWN_NOISE, "", answer))
    # Indic scripts join words to numbers ("40மீட்டர்"). A figure glued to a
    # letter would slip past the word boundaries below, so it is separated first.
    text = re.sub(r"(\d)(?=[^\x00-\x7f\s\d])", r"\1 ", text)
    text = re.sub(r"(?<=[^\x00-\x7f\s\d])(\d)", r" \1", text)
    text = re.sub(r"(?m)^\s*(?:[-*]\s*)?\d+[.)]\s", " ", text)
    # "6 391 m" and "6\u202f391 m" are 6391 m written with a digit-group space.
    text = re.sub(r"(?<![\d.])(\d{1,3})(?:[ \u00a0\u2009\u202f](\d{3}))(?![\d])", r"\1\2", text)
    allowed_plain = re.sub(r"\s+", " ", allowed_text.lower()).replace(",", "")
    allowed = _allowed_numbers(allowed_text)

    flagged: list[str] = []
    for token, unit in _NUMBER.findall(text):
        plain = token.replace(",", "")
        if "." not in plain and not unit:
            try:
                if int(plain) < config.NUMBER_GUARD_MIN_BARE_INTEGER:
                    continue
            except ValueError:
                pass
        if re.search(rf"(?<![\d.]){re.escape(plain)}(?![\d])", allowed_plain):
            continue
        try:
            value = float(plain)
        except ValueError:
            flagged.append(f"{token}{(' ' + unit) if unit else ''}")
            continue
        # Trailing zeros are formatting, not precision: 0.20 is as precise as 0.2.
        decimals = len(plain.split(".", 1)[1].rstrip("0")) if "." in plain else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9

        def near(candidate: float, candidate_decimals: int) -> bool:
            return candidate_decimals >= decimals and abs(candidate - value) <= tolerance

        ok = any(near(a, d) for a, d in allowed)
        if not ok and unit == "%":
            ok = any(near(a * 100, d - 2) for a, d in allowed if d >= 2)
        if not ok:
            label = f"{token}{(' ' + unit) if unit and unit != '%' else unit or ''}"
            if label not in flagged:
                flagged.append(label)
    return flagged


def allowed_text_for(work: dict) -> str:
    """Everything a figure in the answer may legitimately have come from."""
    return "\n".join(work["allowed"])


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

def coverage_gap(record: dict) -> bool:
    """Does the corpus have a document about this class of object at all?

    The detector recognises aircraft, human remains, fish and ships. The corpus
    is about ordnance, wrecks, debris and unidentified objects. Those two lists
    only partly overlap, and where they do not the honest answer is that no
    reference covers it. Without this the model reaches for the nearest
    document that sounds close, which is how a body becomes a mine.
    """
    label = record.get("label")
    return bool(label) and config.CLASS_COVERAGE.get(label, True) is False


def build_task(mode: str, message: str, record: dict, matches_block: str,
               history: list[dict], ghosttrace: dict | None = None,
               language: str = "en") -> str:
    """Assemble the TASK half of the prompt. rag.TASKS is used as written."""
    blocks = []
    if coverage_gap(record):
        blocks.append(config.COVERAGE_GAP_NOTE.format(label=record["label"]))
    elif (mode in ("explain", "report") and record.get("label") and not _unclassified(record)
          and not ghosttrace):
        # A GhostTrace block already labels the class as detector output that
        # is not a confirmed identification; saying it twice costs tokens.
        blocks.append(config.CLASSIFIED_NOTE.format(
            label=record["label"],
            confidence=record.get("confidence", "not stated")))
    conversation = history_block(history)
    if conversation:
        blocks.append(conversation)
    if ghosttrace:
        blocks.append(ghosttrace_block(ghosttrace))

    if ghosttrace and mode in ("ask", "explain"):
        # The GhostTrace task replaces both: its operator is looking at a ranked
        # target, and the question is why it ranks there and who is responsible,
        # not the generic detection brief.
        blocks.append(config.GHOSTTRACE_TASK.format(
            question=message,
            citation=ghosttrace_citation(ghosttrace),
            opening=config.GHOSTTRACE_SYNTHETIC_OPENING if ghosttrace.get("synthetic") else ""))

    elif mode == "ask":
        if record:
            blocks.append("DETECTION CURRENTLY ON SCREEN:\n" + rag.render_detection(record))
        blocks.append(rag.TASKS["ask"].format(question=message))

    elif history and mode in config.FOLLOW_UP_MODES:
        # Second turn onward on the same contact. The brief has already been
        # given; what is wanted now is an answer to what was just asked.
        context = ["OBSERVATION:\n" + rag.render_detection(record)]
        if matches_block:
            context.append(matches_block)
        blocks.append(config.FOLLOW_UP_TASK.format(
            question=message, context="\n\n".join(context)))

    elif mode == "anomaly":
        blocks.append(rag.TASKS["anomaly"].format(
            detection=rag.render_detection(record), matches=matches_block))
        blocks.append(config.ANOMALY_QUESTION_NOTE.format(question=message))

    else:
        # report always produces the whole document, follow-up or not.
        blocks.append(rag.TASKS[mode].format(detection=rag.render_detection(record)))
        blocks.append(f"The operator also asked: {message}")

    if language_instruction(language):
        blocks.append(language_instruction(language))
    return "\n\n".join(blocks)


def _prepare_turn(message: str, history: list[dict], detection: dict | None,
                  intent: str | None, k: int | None, per_doc: int | None,
                  provider: str | None, model: str | None,
                  ghosttrace_context: dict | None = None, *,
                  language: str = "en", search_text: str | None = None,
                  route_reason: str = "", translation_errors: list[str] | None = None) -> dict:
    """Everything that happens before the model is called.

    Routing, condensing, retrieval and catalog matching are identical whether
    the answer is returned whole or streamed, so they live here and both entry
    points use them. It runs eagerly rather than inside a generator, so a
    missing index fails as a proper HTTP error instead of as a frame arriving
    after the response has already started.
    """
    history = history or []
    ghosttrace = normalise_ghosttrace(ghosttrace_context)
    record = {key: value for key, value in (detection or {}).items() if value is not None}
    if ghosttrace and not record:
        record = ghosttrace_record(ghosttrace)
    # The API speaks `object_class`, rag.py speaks `label`. schemas.py maps them
    # on the way in, but the engine is also called directly by the evaluation
    # suite and by anything else that skips the HTTP layer, and the two paths
    # disagreeing on a field name is how a classified contact silently becomes
    # an anomaly. Normalise here so the engine behaves the same either way.
    if record.get("object_class") and not record.get("label"):
        record["label"] = record.pop("object_class")
    record.pop("object_class", None)
    # And a raw detector class becomes the corpus's word for it. /detect already
    # maps these, but the survey handoff sends the checkpoint's own class name
    # and so does anything calling /chat directly. Without this, "ship" matches
    # no synonym, no severity and no coverage entry, and the answer comes back
    # saying the sources contain no information on ships while the wreck
    # document sits unretrieved.
    mapped = config.DETECTOR_CLASS_MAP.get(_normalise(record.get("label", "")))
    if mapped:
        record["label"] = mapped
    provider = provider or config.PROVIDER
    model = model if model is not None else config.MODEL

    # Retrieval, routing and the corpus synonyms read English. A question typed
    # in another script is searched in its translation when one was obtained.
    search = search_text or message
    intent = intent or route_intent(search, record)
    mode = config.INTENT_TO_MODE[intent]
    anomalous = intent == "anomaly" or _unclassified(record)

    if ghosttrace:
        # The mode's own expansion ("standoff distance", "do not approach") is
        # written for the detection brief and pulls mine documents in. A
        # GhostTrace turn expands towards the ghost-gear document instead.
        query = f"{condense(search, history, record, 'ask')} {ghosttrace_query(ghosttrace)}".strip()
        default_k, default_per_doc = config.GHOSTTRACE_TOP_K, config.GHOSTTRACE_PER_DOC
    else:
        query = condense(search, history, record, mode)
        default_k, default_per_doc = config.TOP_K, config.PER_DOC
    hits = _guard(get_retriever().search, query,
                  k=k or default_k, per_doc=per_doc or default_per_doc)

    matches: list[dict] = []
    matches_block = ""
    if mode == "anomaly" and hits:
        catalog = get_catalog()
        if catalog is None:
            matches_block = ("NEAREST KNOWN OBJECTS: unavailable. No catalog is indexed, "
                             "so no comparison was made. Do not speculate about identity.")
        else:
            ranked = _guard(catalog.match, record, top=config.CATALOG_TOP)
            matches_block = rag.format_matches(ranked, catalog.space)
            matches = matches_from(ranked)

    # What a figure in the answer may have come from. The history's assistant
    # turns are deliberately absent: a number the model invented last turn does
    # not become sourced by being repeated.
    allowed = [chunk.text for chunk, _ in hits]
    allowed.append(message)
    if search_text:
        allowed.append(search_text)
    allowed += [turn.get("content") or "" for turn in history if turn.get("role") == "user"]
    allowed.append(rag.render_detection(record) if record else "")
    allowed.append(matches_block)
    if ghosttrace:
        allowed.append(render_ghosttrace(ghosttrace))

    return {
        "meta": {
            "intent": intent,
            "object_class": record.get("label"),
            "confidence": record.get("confidence"),
            "is_anomaly": anomalous,
            "severity": severity_for(record, anomalous),
            "coverage_gap": coverage_gap(record),
            "query": query,
            "provider": provider,
            "model": model,
            "ghosttrace_citation": ghosttrace_citation(ghosttrace) if ghosttrace else None,
            "mode": "reference",
            "route_reason": route_reason,
            "language": language,
            "query_translated": search_text,
        },
        "language": language,
        "translation_errors": translation_errors or [],
        "hits": hits,
        "sources": sources_from(hits),
        "matches": matches,
        "matches_block": matches_block,
        "record": record,
        "ghosttrace": ghosttrace,
        "message": message,
        "allowed": allowed,
        "filled": (build_task(mode, message, record, matches_block, history, ghosttrace, language)
                   if hits else ""),
    }


def _finalise(work: dict, text: str) -> dict:
    copilot = work["meta"].get("mode") == "copilot"
    records = work.get("records") or []
    grounded, refusal = check_grounding(text, len(work["sources"]),
                                        len(records) if copilot else None)
    return {**work["meta"], "answer": text, "sources": work["sources"],
            "matches": work["matches"], "grounded": grounded, "refusal": refusal,
            "generated_by": "model", "provider_errors": list(work.get("planning_errors") or []),
            "unsourced_numbers": unsourced_numbers(text, allowed_text_for(work)),
            **_data_fields(work)}


def _data_fields(work: dict) -> dict:
    return {"tool_calls": work.get("tool_calls") or [],
            "data_citations": copilot_tools.data_citations(work.get("records") or [])}


# Nothing retrieved means nothing to ground an answer in, so no model is called
# at all. Inventing a protocol here is the exact failure this system is built to
# avoid, and the cheapest way not to do it is not to ask.
def _no_sources(work: dict) -> dict:
    return {**work["meta"], "answer": config.NO_SOURCE_ANSWER, "sources": [],
            "matches": [], "grounded": False, "refusal": True,
            "generated_by": "none", "provider_errors": [], "unsourced_numbers": [],
            **_data_fields(work)}


# ---------------------------------------------------------------------------
# Offline fallback: the retrieved passages, quoted, and nothing generated
# ---------------------------------------------------------------------------

_SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z]{20,})")


def _provider_error_line(provider: str, exc: BaseException) -> str:
    """One short line per failed provider. Never echoes a credential."""
    first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
    first = _SECRET.sub("[redacted]", first)
    if len(first) > 160:
        first = first[:157].rstrip() + "..."
    return f"{provider}: {first}"


def extract(text: str, query: str, section: str | None = None,
            limit: int | None = None) -> str:
    """A short verbatim extract of a passage: its most query-relevant sentence
    and what follows it, cut at a word boundary. Nothing is paraphrased."""
    limit = limit or config.OFFLINE_EXTRACT_CHARS
    body = text
    if section and body.startswith(section):
        body = body[len(section):]
    body = re.sub(r"\s+", " ", body).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s]
    # Chunks overlap, so a passage can open mid-sentence. That fragment is the
    # tail of the previous chunk's last sentence and reads as nonsense alone.
    if len(sentences) > 1 and not re.match(r"[A-Z0-9\"'(\[*-]", sentences[0]):
        sentences = sentences[1:]
    if not sentences:
        return ""
    terms = set(_content_terms(query))
    scores = [len(terms & set(_content_terms(s))) for s in sentences]
    start = max(range(len(sentences)), key=lambda i: (scores[i], -i))
    out = ""
    for sentence in sentences[start:]:
        candidate = f"{out} {sentence}".strip()
        if len(candidate) > limit:
            if not out:
                cut = sentence[:limit].rsplit(" ", 1)[0]
                out = cut.rstrip(",;:") + " [...]"
            else:
                out += " [...]"
            break
        out = candidate
    if start > 0:
        out = "[...] " + out
    return out


def context_facts(work: dict) -> list[str]:
    """The key handoff facts for the fallback, copied and never computed anew."""
    facts: list[str] = []
    gt = work.get("ghosttrace")
    if gt:
        cite = ghosttrace_citation(gt)
        facts.append(f"**From the GhostTrace context** ({cite}). Survey data, not a reference source.")
        if gt.get("synthetic"):
            facts.append("- SYNTHETIC survey data: nothing here is evidence of a real net.")
        facts.append(f"- Object class (detector output): {_given(gt.get('object_class'))}; "
                     f"confidence {_given(gt.get('confidence_pct'), '%')}")
        for line in priority_lines(gt.get("priority")):
            if line.lstrip().startswith("formula as stated"):
                continue
            # Term lines nest under the priority line, the way they are read.
            facts.append(("  - " if line.startswith("  ") else "- ") + line.strip().lstrip("- "))
        activity = gt.get("activity") or {}
        if activity:
            facts.append(f"- Water-column activity: level {_given(activity.get('level'))}, "
                         f"score {_given(activity.get('score'))}")
        for habitat in (gt.get("habitat_nearest") or [])[:2]:
            facts.append(f"- Nearest habitat: {habitat.get('name') or 'unnamed'} "
                         f"({_given(habitat.get('kind'))}), {_given(habitat.get('distance_m'), ' m')}")
        people = gt.get("people") or {}
        if people:
            facts.append(f"- Propeller hazard level: {_given(people.get('propeller_hazard_level'))}")
        change = gt.get("change") or {}
        if change:
            facts.append(f"- Change: {_given(change.get('status'))}")
        for authority in gt.get("authorities") or []:
            facts.append(f"- GhostTrace lists: {_given(authority.get('name'))} "
                         f"({_given(authority.get('situation'))}). Check it against the passages below.")
        for caveat in gt.get("caveats") or []:
            facts.append(f"- Caveat: {caveat}")
    elif work.get("record"):
        record = work["record"]
        label = record.get("label")
        if label:
            confidence = record.get("confidence")
            facts.append(f"**Detection on screen:** {label}"
                         + (f", classifier confidence {_num(confidence)}"
                            if isinstance(confidence, (int, float)) else ""))
    return facts


def retrieval_only(work: dict, errors: list[str]) -> dict:
    """What the operator gets when no provider answers.

    Clearly labelled, and deliberately not an answer: the top passages as short
    verbatim extracts with their [Sn] markers, so every line still opens its
    source in the citation panel, and the key context facts copied from the
    handoff. No sentence in it is generated.
    """
    labels = offline_labels(work.get("language"))
    stop = "." if normalise_language(work.get("language")) == "en" else ""
    lines = [f"**{labels['retrieval_title']}{stop}**", "", labels["retrieval_intro"], ""]
    facts = context_facts(work)
    if facts:
        lines += facts + [""]
    lines.append(f"**{labels['passages_heading']}**")
    lines.append("")
    query = f"{work['message']} {work['meta']['query']}"
    for source, (chunk, _score) in list(zip(work["sources"], work["hits"]))[:config.OFFLINE_PASSAGES]:
        heading = source["title"] + (f" — {source['section']}" if source.get("section") else "")
        lines.append(f"[S{source['n']}] **{heading}**")
        lines.append("> " + extract(chunk.text, query, source.get("section")))
        lines.append("")
    if work["meta"].get("coverage_gap"):
        lines.append("No reference document covers this class of object.")
    text = "\n".join(lines).rstrip()
    grounded, _ = check_grounding(text, len(work["sources"]))
    return {**work["meta"], "answer": text, "sources": work["sources"],
            "matches": work["matches"], "grounded": grounded,
            # Nothing declined to answer; nothing answered at all.
            "refusal": False, "provider": "", "model": "",
            "generated_by": "retrieval_only", "provider_errors": errors,
            "unsourced_numbers": unsourced_numbers(text, allowed_text_for(work)),
            **_data_fields(work)}


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

_INDIC = re.compile(r"[ऀ-෿]")


def normalise_language(language: str | None) -> str:
    code = (language or "").strip().lower()
    return code if code in config.LANGUAGES else config.DEFAULT_LANGUAGE


def language_instruction(language: str | None) -> str:
    code = normalise_language(language)
    if code == "en":
        return ""
    return config.LANGUAGE_INSTRUCTION.format(**config.LANGUAGES[code])


def offline_labels(language: str | None) -> dict:
    return config.OFFLINE_LABELS.get(normalise_language(language), config.OFFLINE_LABELS["en"])


def needs_translation(message: str) -> bool:
    """Is the question written in an Indic script? English typed with the Hindi
    answer language selected needs no translation for retrieval."""
    return bool(_INDIC.search(message or ""))


def translate_for_retrieval(message: str, provider: str | None = None,
                            model: str | None = None) -> tuple[str | None, list[str]]:
    """(English rendering for search, or None; one error line per provider tried).

    Retrieval stays English because the corpus is English. With no provider
    the question is searched as typed, which finds little in an English index,
    and the offline answer says what it found.
    """
    if not needs_translation(message):
        return None, []
    errors: list[str] = []
    preferred = provider or config.PROVIDER
    for name in provider_order(preferred):
        try:
            text = _guard(rag.quick_complete, config.TRANSLATE_SYSTEM, message, name,
                          (model or "") if name == preferred else "",
                          max_tokens=300, timeout_s=config.TRANSLATE_TIMEOUT_S)
        except Exception as exc:  # EngineError, or a raw SDK/transport failure
            errors.append(_provider_error_line(name, exc))
            continue
        text = (text or "").strip().strip('"').strip()
        if text and not needs_translation(text):
            return text, errors
        errors.append(f"{name}: translation came back empty or untranslated")
    return None, errors


def native_keywords(message: str) -> str:
    """English keywords for the native words in a message, for offline routing."""
    found = [english for native, english in config.COPILOT_NATIVE_KEYWORDS.items()
             if native in (message or "")]
    for token, aliases in config.COPILOT_SURVEY_ALIASES.items():
        if any(alias in (message or "") for alias in aliases):
            found.append(token)
    return " ".join(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# Mission Copilot: routing
# ---------------------------------------------------------------------------

def survey_tokens() -> dict[str, list[str]]:
    """Words that name a survey -> the survey ids they name."""
    out: dict[str, list[str]] = {}
    for sid in copilot_tools.survey_ids():
        out.setdefault(sid.lower(), []).append(sid)
        for token in re.split(r"[-_.]+", sid.lower()):
            if len(token) >= 2 and token not in config.COPILOT_GENERIC_ID_TOKENS:
                out.setdefault(token, []).append(sid)
    return out


def mentioned_surveys(text: str) -> list[str]:
    lowered = (text or "").lower()
    for token, aliases in config.COPILOT_SURVEY_ALIASES.items():
        if any(alias in (text or "") for alias in aliases):
            lowered += f" {token}"
    hits: list[str] = []
    tokens = survey_tokens()
    # Full ids first, so "demo-ghosttrace-mannar-repeat" is not also read as its prefix.
    for token in sorted(tokens, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", lowered):
            for sid in tokens[token]:
                if sid not in hits:
                    hits.append(sid)
            if "-" in token:
                lowered = re.sub(rf"(?<![\w-]){re.escape(token)}(?![\w-])", " ", lowered)
    if len(hits) > 1:
        wants_repeat = re.search(r"\b(repeat|second|later|latest|follow-?up|newer)\b", lowered)
        wants_all = re.search(r"\b(two|both|all|surveys|between|compare|changed?)\b", lowered)
        if wants_repeat and not wants_all:
            hits = [sid for sid in hits if "repeat" in sid] or hits
    return hits


def route_mode(message: str, route_text: str, has_attachment: bool,
               mode: str | None) -> tuple[bool, str]:
    """(copilot?, why). The rule is in config.COPILOT_* and docs/ASSISTANT.md."""
    requested = (mode or "auto").strip().lower()
    if requested == "copilot":
        return True, "copilot: requested"
    if requested == "reference":
        return False, "reference: requested"
    if has_attachment:
        return False, "auto: a detection or GhostTrace target is attached"
    text = f" {(route_text or message or '').lower()} "
    for phrase in config.COPILOT_STRONG_PHRASES:
        if phrase in text:
            return True, f"auto: \"{phrase}\" refers to survey data"
    for phrase in config.COPILOT_MEDIUM_PHRASES:
        if phrase in text:
            word = next((w for w in config.COPILOT_RECORD_WORDS if w in text), None)
            if word:
                return True, f"auto: \"{phrase}\" with \"{word.strip()}\" asks for survey records"
    surveys = mentioned_surveys(route_text or message)
    if surveys:
        noun = next((n for n in config.COPILOT_DATA_NOUNS if n in text), None)
        if noun:
            return True, f"auto: names survey {', '.join(surveys)} and asks about its data (\"{noun}\")"
    return False, "auto: no survey data referenced"


# ---------------------------------------------------------------------------
# Mission Copilot: planning
# ---------------------------------------------------------------------------

def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def _class_word(text: str) -> str | None:
    for word in sorted(config.COPILOT_CLASS_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}(?:e?s)?\b", text.replace("-", " ")):
            return word
    return None


def keyword_plan(text: str) -> list[tuple[str, dict]]:
    """The deterministic plan: question words -> tool calls.

    It runs when no model can plan, and it is what the offline data-only
    answer is built from, so it must be useful alone. Ordered by how specific
    the signal is; capped like any plan.
    """
    t = f" {(text or '').lower()} "
    surveys = mentioned_surveys(text)
    calls: list[tuple[str, dict]] = []

    def each(name: str, extra: dict | None = None) -> None:
        if surveys:
            for sid in surveys:
                calls.append((name, {"survey_id": sid, **(extra or {})}))
        else:
            calls.append((name, dict(extra or {})))

    changed = _has(t, "what changed", "changed", " change", "moved", "difference", "differ",
                   "compare", "removed", "new since", "since the last")
    filtered = _has(t, "false positive", "false-positive", "filtered", "suppressed",
                    "rejected", "discarded")
    word = _class_word(t)
    gear = _has(t, " net", "nets", "ghost gear", "fishing gear", "ghosttrace", "targets")
    recover = gear and _has(t, "recover", "retriev", "rescue", "priority", "prioriti", "first",
                            "rank", "urgent", "ghosttrace", "notify", "remove")
    hotspot = _has(t, "hotspot", "highest risk", "most dangerous", "riskiest", "worst area")
    countish = _has(t, "how many", "where", "list", "which", "find", "show", "were found",
                    "found in", "detected", "any ", "count", "number of")
    summary = _has(t, "summar", "overview", "brief ", "tell me about", "report on", "status of")
    listing = _has(t, "which surveys", "what surveys", "list surveys", "list the surveys",
                   "available surveys", "surveys do we have", "surveys are there")

    if changed and not filtered:
        if surveys:
            compared = [sid for sid in surveys
                        if ((copilot_tools.load_ghosttrace(sid) or {}).get("change_summary") or {})
                        .get("compared_with")]
            for sid in compared or surveys[:1]:
                calls.append(("change_report", {"survey_id": sid}))
        else:
            calls.append(("change_report", {}))
    if filtered:
        each("filtered_detections")
    if recover:
        calls.append(("ghosttrace_targets",
                      {"survey_id": surveys[0]} if len(surveys) == 1 else {}))
    if hotspot:
        each("top_hotspots")
    if word and countish and not (recover and config.COPILOT_CLASS_WORDS.get(word) == "net"):
        each("find_detections", {"object_class": word})
    if summary:
        if surveys:
            each("survey_summary")
        else:
            calls.append(("list_surveys", {}))
    if listing:
        calls.append(("list_surveys", {}))
    if not calls:
        if surveys:
            each("survey_summary")
        else:
            calls.append(("list_surveys", {}))
    return _dedupe(calls)


def _dedupe(calls: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen: set[str] = set()
    out = []
    for name, args in calls:
        key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if key in seen:
            continue
        seen.add(key)
        out.append((name, args))
    return out[:config.COPILOT_MAX_TOOL_CALLS]


def survey_catalogue() -> str:
    lines = []
    for sid in copilot_tools.survey_ids():
        export = copilot_tools.load_export(sid) or {}
        gt = copilot_tools.load_ghosttrace(sid)
        synthetic = copilot_tools.is_synthetic(export, gt)
        title = (export.get("metadata") or {}).get("title") or sid
        lines.append(f"- {sid}: {title} [{'synthetic' if synthetic else 'real data'}; "
                     f"GhostTrace: {'yes' if gt else 'no'}]")
    return "\n".join(lines) or "(no processed surveys)"


def _planner_user(message: str, search_text: str | None, history: list[dict],
                  done: list[copilot_tools.ToolOutput]) -> str:
    parts = []
    earlier = [t.get("content", "") for t in history[-config.HISTORY_TURNS:] if t.get("role") == "user"]
    if earlier:
        parts.append("EARLIER OPERATOR TURNS (context only):\n" + "\n".join(f"- {e[:300]}" for e in earlier[-3:]))
    parts.append(f"QUESTION: {message}")
    if search_text:
        parts.append(f"QUESTION IN ENGLISH: {search_text}")
    if done:
        parts.append("ALREADY LOOKED UP (do not repeat these):\n" + "\n".join(
            f"- {o.name} {json.dumps(o.args, default=str)}: {o.summary}" for o in done))
    return "\n\n".join(parts)


def parse_json_plan(reply: str) -> list[dict]:
    """The calls out of a JSON plan reply, tolerating a code fence or prose around it."""
    match = re.search(r"\{.*\}|\[.*\]", reply or "", re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []
    calls = parsed.get("calls") if isinstance(parsed, dict) else parsed
    out = []
    for call in calls or []:
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            out.append({"name": call["name"], "args": args})
    return out


_DEAD_PROVIDER = ("rate limit", "429", "api key", "api_key", "no valid credentials",
                  "not installed", "could not create")


def plan_copilot(message: str, search_text: str | None, route_text: str, history: list[dict],
                 provider: str, model: str) -> dict:
    """Pick the tools, run them, number the records.

    Native function calling on the preferred provider, then the other; a JSON
    plan when a provider rejects tools outright; the keyword plan when no
    provider plans at all, or plans nothing. Capped in calls and in time.
    """
    specs = copilot_tools.tool_specs()
    system = config.COPILOT_PLANNER_SYSTEM.format(max_calls=config.COPILOT_MAX_TOOL_CALLS,
                                                  catalogue=survey_catalogue())
    deadline = time.monotonic() + config.COPILOT_PLAN_TIMEOUT_S
    errors: list[str] = []
    dead: set[str] = set()
    ledger = copilot_tools.Ledger()
    planned_by = "keywords"

    def run(calls: list[tuple[str, dict]], by: str) -> int:
        added = 0
        for name, args in calls:
            if len(ledger.outputs) >= config.COPILOT_MAX_TOOL_CALLS:
                break
            if any(o.name == name and o.args == copilot_tools._coerce(name, args)
                   for o in ledger.outputs if name in copilot_tools.TOOLS):
                continue
            ledger.add(copilot_tools.run_tool(name, args, planned_by=by))
            added += 1
        return added

    for _round in range(max(1, config.COPILOT_PLAN_ROUNDS)):
        if len(ledger.outputs) >= config.COPILOT_MAX_TOOL_CALLS:
            break
        user = _planner_user(message, search_text, history, ledger.outputs)
        raw: list[dict] | None = None
        by = "model"
        for name in provider_order(provider):
            remaining = deadline - time.monotonic()
            if name in dead or remaining < 1:
                continue
            chosen_model = model if name == provider else ""
            try:
                raw = _guard(rag.plan_tools, system, user, specs, name, chosen_model,
                             timeout_s=remaining)
                by = "model"
            except Exception as exc:
                detail = str(exc)
                errors.append(_provider_error_line(name, exc))
                if any(marker in detail.lower() for marker in _DEAD_PROVIDER):
                    dead.add(name)
                if isinstance(exc, EngineError) and not _transient(detail):
                    # The provider is up but would not take tools: ask for JSON instead.
                    try:
                        reply = _guard(rag.quick_complete, system,
                                       user + "\n\n" + config.COPILOT_JSON_PLAN.format(
                                           tools=json.dumps(specs, ensure_ascii=False)),
                                       name, chosen_model, max_tokens=600,
                                       timeout_s=max(1.0, deadline - time.monotonic()))
                        raw = parse_json_plan(reply)
                        by = "json_plan"
                    except Exception as exc2:
                        errors.append(_provider_error_line(name, exc2))
                        continue
                else:
                    continue
            break
        if raw is None:
            break
        calls = [(c["name"], c.get("args") or {}) for c in raw if c.get("name") in copilot_tools.TOOLS]
        if not calls or not run(_dedupe(calls), by):
            break
        planned_by = by

    if not ledger.outputs:
        run(keyword_plan(route_text), "keywords")
        planned_by = "keywords"
    return {"ledger": ledger, "planned_by": planned_by, "errors": errors, "dead": dead}


# ---------------------------------------------------------------------------
# Mission Copilot: the turn
# ---------------------------------------------------------------------------

def _prepare_copilot(message: str, history: list[dict], provider: str | None, model: str | None,
                     *, language: str, search_text: str | None, route_text: str,
                     route_reason: str, translation_errors: list[str]) -> dict:
    """The eager part: nothing slow, but a missing index still fails before any frame."""
    get_retriever()
    return {
        "meta": {
            "intent": "question", "object_class": None, "confidence": None,
            "is_anomaly": False, "severity": "unknown", "coverage_gap": False,
            "query": "", "provider": provider or config.PROVIDER,
            "model": model if model is not None else config.MODEL,
            "ghosttrace_citation": None, "mode": "copilot", "route_reason": route_reason,
            "language": language, "query_translated": search_text,
        },
        "message": message, "history": history, "language": language,
        "search_text": search_text, "route_text": route_text,
        "translation_errors": translation_errors,
        "hits": [], "sources": [], "matches": [], "records": [], "tool_calls": [],
        "planning_errors": [], "dead": set(), "allowed": [], "filled": "", "planned": False,
    }


def _run_copilot(work: dict) -> dict:
    """Plan, run the tools, retrieve, and build the answer prompt. Mutates work."""
    meta = work["meta"]
    plan = plan_copilot(work["message"], work["search_text"], work["route_text"],
                        work["history"], meta["provider"], meta["model"])
    ledger: copilot_tools.Ledger = plan["ledger"]
    records = ledger.records
    work["records"] = records
    work["tool_calls"] = [o.as_call() for o in ledger.outputs]
    work["planning_errors"] = list(work["translation_errors"]) + plan["errors"]
    work["dead"] = plan["dead"]
    meta["planned_by"] = plan["planned_by"]

    search = work["search_text"] or work["message"]
    history = work["history"]
    parts = [search, term_expansions(search)]
    if is_follow_up(search, history):
        parts.append(" ".join(carried_terms(history)))
    for output in ledger.outputs:
        parts.append(config.COPILOT_TOOL_QUERY_TERMS.get(output.name, ""))
    query = " ".join(p for p in parts if p and p.strip())
    meta["query"] = query
    hits = _guard(get_retriever().search, query, k=config.COPILOT_TOP_K, per_doc=config.COPILOT_PER_DOC)
    work["hits"] = hits
    work["sources"] = sources_from(hits)

    data_block = copilot_tools.render_data_block(records)
    blocks = []
    conversation = history_block(history)
    if conversation:
        blocks.append(conversation)
    blocks.append(config.COPILOT_NOTE.format(data=data_block))
    blocks.append(config.COPILOT_TASK.format(question=work["message"]))
    if language_instruction(work["language"]):
        blocks.append(language_instruction(work["language"]))
    work["filled"] = "\n\n".join(blocks)

    allowed = [chunk.text for chunk, _ in hits]
    allowed += [work["message"], work["search_text"] or "", data_block]
    allowed += [t.get("content") or "" for t in history if t.get("role") == "user"]
    work["allowed"] = allowed
    work["planned"] = True
    return work


def _cell(value: Any, limit: int = 90) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).replace("|", "/")
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _where(data: dict) -> str:
    if data.get("latitude") is not None and data.get("longitude") is not None:
        return f"{data['latitude']}, {data['longitude']}"
    if data.get("pixel_x") is not None:
        return f"pixel {data['pixel_x']}, {data['pixel_y']} (not georeferenced)"
    return "not georeferenced"


_TABLES: dict[str, tuple[list[str], Any]] = {
    "ghosttrace_target": (["Survey", "Target", "Class", "Tier", "Score", "Change", "Latest",
                           "Authorities GhostTrace lists"],
                          lambda d: [d.get("survey_id"), d.get("detection_id"), d.get("object_class"),
                                     d.get("priority_tier"), d.get("priority_score"),
                                     d.get("change_status"), d.get("latest_observation"),
                                     "; ".join(str(a.get("name")) for a in d.get("authorities_ghosttrace_lists") or [])]),
    "detection": (["Survey", "Detection", "Class", "Confidence", "Tier", "Position", "Recommended action"],
                  lambda d: [d.get("survey_id"), d.get("detection_id"), d.get("object_class"),
                             d.get("detector_confidence"), d.get("severity_tier"), _where(d),
                             d.get("recommended_action")]),
    "hotspot": (["Survey", "Hotspot", "Class", "Tier", "Risk score", "Detections", "Recommended action"],
                lambda d: [d.get("survey_id"), d.get("hotspot_id"), d.get("dominant_class"),
                           d.get("severity_tier"), d.get("risk_score"), d.get("detection_count"),
                           d.get("recommended_action")]),
    "change": (["Survey", "Target", "Status", "Moved (m)", "Previous"],
               lambda d: [d.get("survey_id"), d.get("detection_id"), d.get("status"), d.get("moved_m"),
                          d.get("previous_detection_id")]),
    "removed": (["Survey", "Target", "Status", "Moved (m)", "Previous"],
                lambda d: [d.get("survey_id"), "—", d.get("status"), None, d.get("previous_detection_id")]),
    "change_summary": (["Survey", "Compared with", "New", "Moved", "Persistent", "Removed"],
                       lambda d: [d.get("survey_id"), d.get("compared_with"), d.get("new"), d.get("moved"),
                                  d.get("persistent"), d.get("removed")]),
    "filtered_detection": (["Survey", "Detection", "Class", "Verified confidence %", "Why it was filtered"],
                           lambda d: [d.get("survey_id"), d.get("detection_id"), d.get("object_class"),
                                      d.get("verified_confidence_pct"),
                                      "; ".join(d.get("reasons") or []) or d.get("hard_reasons")]),
}


def _facts(data: dict, limit: int = 14) -> str:
    skip = {"surveys_searched", "status_meanings", "ranking", "order", "survey_ids", "surveys_dir"}
    items = [f"{key}: {_cell(value, 160)}" for key, value in data.items()
             if key not in skip and value not in (None, "", [], {})]
    return "; ".join(items[:limit])


def data_only(work: dict, errors: list[str]) -> dict:
    """The copilot's answer when no model can write one: the records, tabled.

    Nothing in it is generated. Every row carries its [Dn] marker and every
    extract its [Sn], so the citation panels resolve exactly as they would for
    a written answer. Headings follow the chosen language; values stay as the
    files write them and extracts stay English.
    """
    labels = offline_labels(work.get("language"))
    records = work.get("records") or []
    stop = "." if normalise_language(work.get("language")) == "en" else ""
    lines = [f"**{labels['data_title']}{stop}**", "", labels["data_intro"], ""]
    if any(r.get("synthetic") for r in records):
        lines += [f"**{labels['synthetic']}**", ""]
    lines.append(f"**{labels['data_heading']}**")
    lines.append("")
    by_call: dict[int, list[dict]] = {}
    outputs = work.get("tool_calls") or []
    for index, call in enumerate(outputs):
        by_call[index] = [r for r in records if r.get("n") in set(call.get("citations") or [])]
    for index, call in enumerate(outputs):
        lines.append(f"_{call['summary']}_")
        lines.append("")
        group = by_call.get(index) or []
        rows = [r for r in group if r["kind"] != "query_result"]
        for record in group:
            if record["kind"] == "query_result":
                lines.append(f"- [D{record['n']}] {_facts(record['data'])}")
        if not rows:
            lines.append(f"- {labels['no_records']}")
        tabled = [r for r in rows if r["kind"] in _TABLES]
        table_of = {"removed": "change"}
        for table in dict.fromkeys(table_of.get(r["kind"], r["kind"]) for r in tabled):
            headers, _ = _TABLES[table]
            lines.append("")
            lines.append("| | " + " | ".join(headers) + " |")
            lines.append("|---|" + "---|" * len(headers))
            for record in (r for r in tabled if table_of.get(r["kind"], r["kind"]) == table):
                row_of = _TABLES[record["kind"]][1]
                lines.append(f"| [D{record['n']}] | " + " | ".join(_cell(v) for v in row_of(record["data"])) + " |")
        for record in (r for r in rows if r["kind"] not in _TABLES):
            lines.append(f"- [D{record['n']}] {record['label']}: {_facts(record['data'], 18)}")
        lines.append("")
    if work["hits"]:
        lines.append(f"**{labels['sources_heading']}**")
        lines.append("")
        query = f"{work['search_text'] or work['message']} {work['meta']['query']}"
        for source, (chunk, _score) in list(zip(work["sources"], work["hits"]))[:config.OFFLINE_PASSAGES]:
            heading = source["title"] + (f" — {source['section']}" if source.get("section") else "")
            lines.append(f"[S{source['n']}] **{heading}**")
            lines.append("> " + extract(chunk.text, query, source.get("section")))
            lines.append("")
    text = "\n".join(lines).rstrip()
    grounded, _ = check_grounding(text, len(work["sources"]), len(records))
    return {**work["meta"], "answer": text, "sources": work["sources"], "matches": [],
            "grounded": grounded, "refusal": False, "provider": "", "model": "",
            "generated_by": "data_only", "provider_errors": errors,
            "unsourced_numbers": unsourced_numbers(text, allowed_text_for(work)),
            **_data_fields(work)}


def _copilot_answer(work: dict, offline_fallback: bool) -> dict:
    _run_copilot(work)
    errors = list(work["planning_errors"])
    last: BaseException | None = None
    preferred = work["meta"]["provider"]
    for provider in provider_order(preferred):
        if provider in work["dead"]:
            continue
        try:
            text = _guard(rag.generate, work["filled"], work["hits"], provider,
                          work["meta"]["model"] if provider == preferred else "")
        except EngineError as exc:
            if not _transient(str(exc)):
                raise
            last = exc
            errors.append(_provider_error_line(provider, exc))
            continue
        except Exception as exc:
            last = exc
            errors.append(_provider_error_line(provider, exc))
            continue
        work["meta"]["provider"] = provider
        result = _finalise(work, text)
        result["provider_errors"] = []
        return result
    if not offline_fallback:
        if isinstance(last, EngineError):
            raise last
        raise EngineError("No provider produced an answer. " + "; ".join(errors))
    log.warning("no provider answered the copilot; returning data-only records")
    return data_only(work, errors)


def _emit_copilot(work: dict, offline_fallback: bool) -> Iterator[dict]:
    yield {"type": "meta", **work["meta"], "matches": []}
    try:
        _run_copilot(work)
    except EngineError as exc:
        yield {"type": "error", "detail": str(exc)}
        return
    yield {"type": "tools", "tool_calls": work["tool_calls"],
           "data_citations": copilot_tools.data_citations(work["records"]),
           "planned_by": work["meta"].get("planned_by")}
    if config.STREAM_SOURCES_EARLY:
        yield {"type": "sources", "sources": work["sources"]}

    pieces: list[str] = []
    errors = list(work["planning_errors"])
    preferred = work["meta"]["provider"]
    for provider in provider_order(preferred):
        if provider in work["dead"]:
            continue
        try:
            model = work["meta"]["model"] if provider == preferred else ""
            for piece in rag.generate_stream(work["filled"], work["hits"], provider, model):
                pieces.append(piece)
                yield {"type": "delta", "text": piece}
        except (SystemExit, Exception) as exc:
            detail = str(exc)
            if pieces:
                yield {"type": "error", "detail": detail}
                yield {"type": "done", **_finalise(work, "".join(pieces))}
                return
            if isinstance(exc, SystemExit) and not _transient(detail):
                yield {"type": "error", "detail": detail}
                return
            errors.append(_provider_error_line(provider, exc))
            continue
        work["meta"]["provider"] = provider
        final = _finalise(work, "".join(pieces))
        final["provider_errors"] = []
        yield {"type": "done", **final}
        return
    if not offline_fallback:
        yield {"type": "error", "detail": "No provider produced an answer. " + "; ".join(errors)}
        return
    final = data_only(work, errors)
    yield {"type": "delta", "text": final["answer"]}
    yield {"type": "done", **final}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _front_door(message: str, history: list[dict], detection: dict | None,
                ghosttrace_context: dict | None, provider: str | None, model: str | None,
                mode: str | None, language: str | None) -> dict:
    """Language, translation and routing: everything decided before either path runs."""
    language = normalise_language(language)
    requested = (mode or "auto").strip().lower()
    if requested not in config.ASSISTANT_MODES:
        raise EngineError(f"Unknown mode {mode!r}; choose from {', '.join(config.ASSISTANT_MODES)}")
    search_text, translation_errors = translate_for_retrieval(message, provider, model)
    route_text = search_text or message
    if needs_translation(message) and not search_text:
        route_text = f"{message} {native_keywords(message)}"
    if is_follow_up(route_text, history):
        route_text = f"{route_text} {' '.join(carried_terms(history))}"
    has_attachment = bool(detection) or bool(ghosttrace_context)
    copilot, reason = route_mode(message, route_text, has_attachment, requested)
    return {"language": language, "search_text": search_text, "route_text": route_text,
            "translation_errors": translation_errors, "copilot": copilot, "reason": reason}


def answer(message: str,
           history: list[dict] | None = None,
           detection: dict | None = None,
           *,
           intent: str | None = None,
           k: int | None = None,
           per_doc: int | None = None,
           provider: str | None = None,
           model: str | None = None,
           ghosttrace_context: dict | None = None,
           offline_fallback: bool = False,
           mode: str | None = None,
           language: str | None = None) -> dict:
    """Answer one turn, whole. Returns the response contract as a plain dict.

    `offline_fallback` decides what happens when every provider is unavailable.
    The API routes turn it on, so an operator gets the labelled retrieved
    passages instead of an error. It is off for direct callers, because a
    program that asked for a generated draft -- GhostTrace's alert writer is one
    -- must get the EngineError it already handles, not quoted passages that
    look like an answer.

    `mode` is "auto" (default), "copilot" or "reference"; `language` one of
    config.LANGUAGES. See route_mode for the auto rule.
    """
    history = history or []
    door = _front_door(message, history, detection, ghosttrace_context, provider, model,
                       mode, language)
    if door["copilot"]:
        work = _prepare_copilot(message, history, provider, model, language=door["language"],
                                search_text=door["search_text"], route_text=door["route_text"],
                                route_reason=door["reason"],
                                translation_errors=door["translation_errors"])
        return _copilot_answer(work, offline_fallback)

    work = _prepare_turn(message, history, detection, intent, k, per_doc,
                         provider, model, ghosttrace_context, language=door["language"],
                         search_text=door["search_text"], route_reason=door["reason"],
                         translation_errors=door["translation_errors"])
    if not work["hits"]:
        return _no_sources(work)

    # Try the preferred provider, then the other one if this one is simply
    # unavailable. A refusal or a bad request is not retried anywhere. When
    # every provider is unavailable the passages are returned, labelled.
    errors: list[str] = list(work["translation_errors"])
    last: BaseException | None = None
    for provider in provider_order(work["meta"]["provider"]):
        try:
            text = _guard(rag.generate, work["filled"], work["hits"],
                          provider, work["meta"]["model"] if provider == work["meta"]["provider"] else "")
        except EngineError as exc:
            if not _transient(str(exc)):
                raise
            last = exc
            errors.append(_provider_error_line(provider, exc))
            log.warning("provider unavailable, trying the next: %s", errors[-1])
            continue
        except Exception as exc:
            # Not a SystemExit from rag.py: a raw transport failure from an SDK.
            # No provider rejected anything, so it is unavailability, not a bug
            # in the request.
            last = exc
            errors.append(_provider_error_line(provider, exc))
            log.warning("provider failed outside the engine: %s", errors[-1])
            continue
        work["meta"]["provider"] = provider
        return _finalise(work, text)
    if not offline_fallback:
        if isinstance(last, EngineError):
            raise last
        raise EngineError("No provider produced an answer. " + "; ".join(errors))
    log.warning("no provider answered; returning retrieval-only passages")
    return retrieval_only(work, errors)


def answer_stream(message: str,
                  history: list[dict] | None = None,
                  detection: dict | None = None,
                  *,
                  intent: str | None = None,
                  k: int | None = None,
                  per_doc: int | None = None,
                  provider: str | None = None,
                  model: str | None = None,
                  ghosttrace_context: dict | None = None,
                  offline_fallback: bool = False,
                  mode: str | None = None,
                  language: str | None = None) -> Iterator[dict]:
    """Answer one turn, in frames. Same routing, same sources, same rules.

    Frame order is meta, sources, then deltas, then done. The citations go out
    before the first word because retrieval has already finished by then, so the
    panel is populated while the answer is still being written. `grounded` can
    only be computed once the whole answer exists, so it rides in the done
    frame and nowhere earlier.

    A copilot turn adds one frame: meta, tools (the calls made and the records
    they returned), sources, deltas, done. Planning runs after the meta frame,
    so the interface can show that it is looking things up.
    """
    history = history or []
    door = _front_door(message, history, detection, ghosttrace_context, provider, model,
                       mode, language)
    if door["copilot"]:
        work = _prepare_copilot(message, history, provider, model, language=door["language"],
                                search_text=door["search_text"], route_text=door["route_text"],
                                route_reason=door["reason"],
                                translation_errors=door["translation_errors"])
        return _emit_copilot(work, offline_fallback)
    work = _prepare_turn(message, history, detection, intent, k, per_doc,
                         provider, model, ghosttrace_context, language=door["language"],
                         search_text=door["search_text"], route_reason=door["reason"],
                         translation_errors=door["translation_errors"])
    return _emit(work, offline_fallback)


def _emit(work: dict, offline_fallback: bool = False) -> Iterator[dict]:
    yield {"type": "meta", **work["meta"], "matches": work["matches"]}
    if config.STREAM_SOURCES_EARLY:
        yield {"type": "sources", "sources": work["sources"]}

    if not work["hits"]:
        final = _no_sources(work)
        yield {"type": "delta", "text": final["answer"]}
        yield {"type": "done", **final}
        return

    pieces: list[str] = []
    errors: list[str] = list(work.get("translation_errors") or [])
    for provider in provider_order(work["meta"]["provider"]):
        try:
            model = work["meta"]["model"] if provider == work["meta"]["provider"] else ""
            for piece in rag.generate_stream(work["filled"], work["hits"], provider, model):
                pieces.append(piece)
                yield {"type": "delta", "text": piece}
        except (SystemExit, Exception) as exc:
            detail = str(exc)
            if pieces:
                # Failover is only possible before the first word. Once text has
                # gone out, switching provider mid-answer would splice two
                # different completions together. The response is already 200,
                # so the failure is reported in the stream, and whatever text
                # arrived is still grounded text and is finalised, not discarded.
                yield {"type": "error", "detail": detail}
                yield {"type": "done", **_finalise(work, "".join(pieces))}
                return
            if isinstance(exc, SystemExit) and not _transient(detail):
                yield {"type": "error", "detail": detail}
                return
            errors.append(_provider_error_line(provider, exc))
            log.warning("provider unavailable, trying the next: %s", errors[-1])
            continue
        work["meta"]["provider"] = provider
        yield {"type": "done", **_finalise(work, "".join(pieces))}
        return

    # Every provider was unavailable before writing a word. The passages go out
    # as one delta, labelled, and the done frame says what they are.
    if not offline_fallback:
        yield {"type": "error", "detail": "No provider produced an answer. " + "; ".join(errors)}
        return
    log.warning("no provider answered the stream; returning retrieval-only passages")
    final = retrieval_only(work, errors)
    yield {"type": "delta", "text": final["answer"]}
    yield {"type": "done", **final}
