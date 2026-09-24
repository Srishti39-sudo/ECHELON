"""Every knob in one file.

The rule for this module: if it is something you might want to change later --
a path, a threshold, a model name, a severity mapping, a phrase the operator
reads -- it lives here and nowhere else. Routes and engine code import from
here; they never hard-code a value of their own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# rag.py sits at the repository root and is imported as a top-level module.
# Running `uvicorn backend.main:app` from the root already puts it on the path;
# this insert makes the backend importable from anywhere else too.

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KB_DIR = ROOT / "rag_assistant" / "kb"
SOURCES_DIR = ROOT / "rag_assistant" / "sources"

# Original publications are served read-only so a citation can open the PDF it
# was written from. Mount point, then the URL a client should use.
SOURCES_MOUNT = "/sources"
SOURCES_BASE_URL = os.environ.get("DEEPECHO_SOURCES_BASE_URL", SOURCES_MOUNT)

# --- Server ----------------------------------------------------------------

HOST = os.environ.get("DEEPECHO_HOST", "127.0.0.1")
PORT = int(os.environ.get("DEEPECHO_PORT", "8000"))

# The Vite dev server, on both spellings of localhost, plus the Vite preview
# port. Local only: nothing here is exposed to the internet.
#
# The fallback ports are not padding. Vite takes the next free port when 5173 is
# busy, which happens the moment a second dev server is started or a previous
# one has not released the socket. The frontend then loads perfectly and every
# request to the backend fails CORS, which reads like a broken backend rather
# than like a port that moved.
_DEV_PORTS = (5173, 5174, 5175, 5176, 4173, 4174)
_DEV_ORIGINS = [f"http://{host}:{port}"
                for port in _DEV_PORTS
                for host in ("localhost", "127.0.0.1")]

CORS_ORIGINS = [
    o.strip() for o in
    os.environ.get("DEEPECHO_CORS_ORIGINS", ",".join(_DEV_ORIGINS)).split(",")
    if o.strip()
]

# --- Features --------------------------------------------------------------
# Which feature groups this process serves. One codebase and one image, run
# once per feature behind the gateway (compose.yaml), so the assistant, the
# hazard map and GhostTrace can each be restarted, redeployed or left switched
# off without touching the other two:
#
#   assistant    /chat, /chat/stream, /rag/query, /sources
#   hazard       /survey, /survey/jobs, /hazard/map, /history, /stats,
#                /detections, /detect, /scans
#   ghosttrace   /ghosttrace
#
# Unset, or "all", serves everything from one process, which is what
# `uvicorn backend.app.main:app` on a laptop has always done.
#
# An unknown name raises instead of being dropped. A typo that silently mounts
# no routes gives a container that is healthy and answers 404 to everything.
ALL_FEATURES = ("assistant", "hazard", "ghosttrace")

_features = {f.strip().lower() for f in
             os.environ.get("DEEPECHO_FEATURES", "all").split(",") if f.strip()}
if not _features or "all" in _features:
    _features = set(ALL_FEATURES)
if _features - set(ALL_FEATURES):
    raise ValueError(
        f"DEEPECHO_FEATURES names unknown feature(s) {sorted(_features - set(ALL_FEATURES))}; "
        f"expected a comma-separated subset of {list(ALL_FEATURES)}, or 'all'")
FEATURES = frozenset(_features)

# --- Generation ------------------------------------------------------------
# Reuse whatever the CLI is configured for. DEEPECHO_PROVIDER in .env selects
# gemini, groq or nvidia; an empty model string means the provider's own default.

PROVIDER = os.environ.get("DEEPECHO_PROVIDER", "gemini")
MODEL = os.environ.get("DEEPECHO_MODEL", "")

# Both providers are free tiers and both fail in normal use: Groq runs out of
# tokens, Gemini returns 503 under load. They fail independently, so trying the
# other one costs a second and roughly halves the chance of a dead answer in
# front of an audience.
#
# Failover happens only for a provider-availability error, never for a refusal
# or a bad request, and never once a streamed answer has begun. The response
# always reports which provider actually answered, so this is transparent
# rather than hidden.
PROVIDER_FAILOVER = os.environ.get("DEEPECHO_FAILOVER", "1").lower() in {"1", "true", "yes"}

# Substrings that mean "this provider is unavailable, try another", as opposed
# to "this request was wrong", which no other provider would fix either.
#
# A missing key or a missing SDK is here too. It is not a bad request: the other
# provider may well be configured, and the operator should get its answer rather
# than a configuration error from the one that is not.
TRANSIENT_PROVIDER_ERRORS = (
    "rate limit", "429", "503", "unavailable", "overload",
    "still failing after retries", "timeout", "timed out", "network error",
    "connection", "could not create", "no valid credentials", "api key",
    "api_key", "not installed", "returned no text", "streamed no text",
    # Groq refuses a request larger than the tier's tokens-per-minute outright.
    # Another provider with a larger window can still answer it.
    "413", "request too large",
)

# --- Retrieval -------------------------------------------------------------
# Mirrors the CLI defaults. Raise EF_SEARCH before anything else if recall ever
# drops below 1.000 in `python3 rag.py bench`.

# Raised from the CLI's 6 and 2 when the corpus grew from 7 documents to 9.
# The governing document for a question now has up to seven sections, and a cap
# of two meant the section that actually answered "who do I notify in India"
# lost to two broader sections of the same document. At 8 and 3 the answer is
# retrieved and the results still span five documents, so diversity holds.
TOP_K = int(os.environ.get("DEEPECHO_TOP_K", "8"))
PER_DOC = int(os.environ.get("DEEPECHO_PER_DOC", "3"))
EF_SEARCH = int(os.environ.get("DEEPECHO_EF_SEARCH", "64"))
CATALOG_TOP = int(os.environ.get("DEEPECHO_CATALOG_TOP", "3"))

# Second-stage reranking (rag_assistant/rerank.py). Retrieval fetches
# RERANK_FETCH candidates, the reranker keeps TOP_K. "auto" turns it on when an
# NVIDIA key is present and off otherwise, so a laptop with no key behaves as
# before. Measured on eval/retrieval_bench.py before it was switched on.
_rerank_flag = os.environ.get("DEEPECHO_RERANK", "auto").strip().lower()
RERANK = (True if _rerank_flag in {"1", "true", "yes", "on"} else
          False if _rerank_flag in {"0", "false", "no", "off"} else
          bool(os.environ.get("NVIDIA_API_KEY")))
RERANK_FETCH = int(os.environ.get("DEEPECHO_RERANK_FETCH", "20"))

# How much of a chunk the citation panel receives. Chunks are ~900 characters,
# so the default sends the whole thing and the panel shows real source text
# rather than a teaser.
SNIPPET_CHARS = int(os.environ.get("DEEPECHO_SNIPPET_CHARS", "900"))

# --- Conversation ----------------------------------------------------------

# Turns of history folded into the prompt. Older turns still shape retrieval
# through the condensed query, they just stop being quoted verbatim.
HISTORY_TURNS = 6

# A follow-up is condensed against earlier turns before it is used to search.
# Without this, "what if it's at 40 metres?" retrieves on the word "metres".
FOLLOW_UP_MAX_WORDS = 12
FOLLOW_UP_OPENERS = (
    "and", "but", "so", "then", "what if", "what about", "how about", "why",
    "why not", "ok", "okay", "also", "does that", "is that", "can it", "would it",
)
FOLLOW_UP_REFERENTS = (
    "it", "its", "it's", "that", "this", "they", "them", "those", "these",
    "the object", "the same", "the contact", "the target",
)
# Content words carried forward from earlier turns into the retrieval query.
CARRY_TERMS = 12

# Each quoted turn is clipped to this many characters before it goes into the
# prompt. An assistant answer can run several hundred words and six of them
# would crowd out the sources, which are the part that must not be crowded out.
HISTORY_CHARS = 700

# --- Intent routing --------------------------------------------------------
# Rules first: they are deterministic, free, and instant. Checked in this
# order, first hit wins. The LLM tiebreak below is off by default.

INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Deliberately phrases, not the bare word "report". "who do I report a
    # mine to?" is a question about the reporting channel, not a request to
    # generate a document, and a bare keyword gets that backwards.
    "report": (
        "write a report", "write me a report", "write up", "write-up", "write it up",
        "incident report", "generate a report", "draft a report", "draft the report",
        "make a report", "make me a report", "give me a report", "need a report",
        "prepare a report", "produce a report", "file a report", "log this",
        "paperwork", "formal submission",
    ),
    "anomaly": (
        "unidentified", "unknown object", "unclassified", "anomaly", "anomalous",
        "cannot identify", "can't identify", "couldn't identify", "could not identify",
        "unable to identify", "not sure what", "no idea what", "no classification",
        "didn't classify", "did not classify", "failed to classify",
    ),
    "explain": (
        "explain", "what is this", "what is it", "what am i looking at",
        "brief me", "walk me through", "tell me about this", "what does this mean",
        "is it dangerous", "is this a hazard", "how risky",
    ),
}

# Fallback when no keyword matches: with a detection record in hand the operator
# is asking about that object; without one it is a general corpus question.
INTENT_WITH_RECORD = "explain"
INTENT_WITHOUT_RECORD = "question"

# One cheap LLM classification when the rules are ambiguous. Off by default:
# it adds a round trip to every turn and the rules cover the demo vocabulary.
INTENT_LLM_FALLBACK = os.environ.get("DEEPECHO_INTENT_LLM", "").lower() in {"1", "true", "yes"}

# The API speaks the product's vocabulary; rag.py speaks its own.
INTENT_TO_MODE = {
    "question": "ask",
    "explain": "explain",
    "anomaly": "anomaly",
    "report": "report",
}

# --- Follow-up turns -------------------------------------------------------
# The first turn on a detection gets the full brief from rag.TASKS. A follow-up
# must not: asked "could it be ordnance?", an assistant that re-runs the whole
# four-step anomaly template has not answered the question, it has repeated
# itself. This reframes the task so the operator's actual question leads.
#
# It relaxes nothing. The grounding and unidentified-object rules live in the
# system prompt and apply to every turn regardless of which template is used.
FOLLOW_UP_MODES = ("explain", "anomaly")

FOLLOW_UP_TASK = """The operator is continuing the same conversation. Answer THIS \
question, and only this question.

QUESTION: {question}

{context}

Answer directly. Do not restate the full brief, and do not repeat protocol steps \
already given earlier in the conversation unless the question asks for them. \
Every standing rule still applies: cite every claim, say plainly where the \
sources do not cover something, and for an unclassified object never state or \
imply an identity."""

# --- Anomaly detection -----------------------------------------------------

# The operator's own question, on the unidentified-object path. Models like to
# echo a question back as a heading, and "Percentage chance this is a mine?"
# printed in bold over the answer is an identity claim on screen, whoever wrote
# the words first.
ANOMALY_QUESTION_NOTE = (
    "The operator also asked: {question}\n"
    "Answer that inside the four steps above. Do not repeat, quote or paraphrase "
    "the operator's wording anywhere in the answer, not even as a heading: it may "
    "name an identity this unidentified object does not have."
)

# A classified detection below this confidence is treated as unclassified. The
# classifier's own uncertainty is the operator's uncertainty.
ANOMALY_CONFIDENCE_FLOOR = float(os.environ.get("DEEPECHO_ANOMALY_FLOOR", "0.45"))

# Labels that mean "the classifier did not know".
UNKNOWN_LABELS = {
    "", "unknown", "unidentified", "unclassified", "anomaly", "other", "none", "null",
}

# --- Detector vocabulary ---------------------------------------------------
# A detector's class names are not the corpus's words. The corpus says "wreck";
# a YOLO class says "shipwreck", and TF-IDF scores that at exactly 0.000 against
# the wreck document while a mine document happens to score 0.095. So a detected
# class is expanded into the vocabulary the documents actually use.
#
# Query-side only. The corpus, the chunking and the index are untouched, and a
# term the index has never seen carries no weight, so a wrong guess here is
# inert rather than harmful.

CLASS_SYNONYMS: dict[str, str] = {
    "aircraft wreck": "aircraft wreckage notification custody",
    "human remains": "human remains respect disturbance grave",
    "fish or marine life": "natural",
    "shipwreck": "wreck vessel",
    "wreck": "wreck vessel",
    "naval mine": "mine ordnance munition",
    "sea mine": "mine ordnance munition",
    "moored mine": "mine moored ordnance munition",
    "bottom mine": "mine seabed ordnance munition",
    "mine": "mine ordnance munition",
    "unexploded ordnance": "unexploded ordnance munition",
    "uxo": "unexploded ordnance munition",
    "torpedo": "ordnance munition",
    "structural debris": "debris",
    "debris": "debris",
    # The ghost-gear document speaks of "ghost gear", "fishing gear" and ALDFG,
    # never of a "ghost net", so a net is expanded into those words as well or
    # the one document that names an authority for it is never retrieved.
    "derelict fishing gear": "net entanglement debris ghost gear aldfg abandoned lost fishing gear",
    "fishing net": "net entanglement debris ghost gear aldfg abandoned lost fishing gear",
    "ghost net": "net entanglement debris ghost gear aldfg abandoned lost fishing gear",
    "net": "net entanglement ghost gear fishing gear",
    "pressure vessel": "cylinder drum unknown contents",
    "gas cylinder": "cylinder drum unknown contents",
    "cylinder": "cylinder drum unknown contents",
    "barrel": "drum cylinder unknown contents",
    "drum": "drum cylinder unknown contents",
    "pipeline": "pipeline",
    "boulder": "rock natural",
    "geological": "rock natural",
    "rock": "rock natural",
    # The trained detector's mine_like_object, mapped below to "suspected
    # mine-like object". It must reach both the mine identification document
    # and the unidentified-object protocol, because a shape that resembles a
    # mine is exactly an object nobody has identified yet.
    "mine like object": "mine ordnance munition unidentified object potentially explosive identified",
}

# --- Detector classes ------------------------------------------------------
# The checkpoint in use emits these classes:
#
#   marine.pt   (yolo11s, team dataset)  shipwreck, aircraft, human, pipeline,
#                                         fishing_gear, mine_like_object
#
# The earlier stand-ins (known.pt: aircraft, human, ship; anomaly.pt: aircraft,
# fish, other, shipwreck) are retired but their class names stay in the map so
# a stored detection from that era still resolves. The map below translates a
# detector class into the label the knowledge base and the catalog use. Where
# no honest translation exists the class is marked as uncovered further down
# rather than being forced onto a document that does not describe it.

DETECTOR_CLASS_MAP: dict[str, str] = {
    "ship": "shipwreck",
    "shipwreck": "shipwreck",
    "aircraft": "aircraft wreck",
    "human": "human remains",
    "fish": "fish or marine life",
    # The detector's own word for "I saw something and cannot name it". It maps
    # to unknown so the assistant routes it down the unidentified-object path.
    "other": "unknown",
    # The SIH26057 target classes, for the pipe / cylinder / net checkpoints
    # being trained. Mapped onto the corpus's existing words so a new model's
    # detections reach the right synonyms and severity with no other change.
    "pipe": "pipeline",
    "pipeline": "pipeline",
    "cylinder": "cylinder",
    "net": "ghost net",
    "ghost_net": "ghost net",
    "ghost-net": "ghost net",
    "entangled_net": "ghost net",
    "fishing_net": "ghost net",
    # The trained detector (models/sonar_pipeline.py, via import_geotag.py):
    # shipwreck, aircraft, human, pipeline, fishing_gear, mine_like_object and
    # unknown_anomaly. The first four are above. Both spellings are listed
    # because chat.py looks the label up after _normalise (underscores become
    # spaces) while detect.py looks up the raw class name.
    "fishing_gear": "derelict fishing gear",
    "fishing gear": "derelict fishing gear",
    # Never "mine" or "naval mine": the detector recognised a shape, and a label
    # that names a confirmed mine would let the answer assert one. "suspected
    # mine-like object" keeps high severity (it contains "mine") and routes to
    # both the mine identification document and the unidentified-object
    # protocol through CLASS_SYNONYMS["mine like object"].
    "mine_like_object": "suspected mine-like object",
    "mine like object": "suspected mine-like object",
    # The open-set anomaly channel: something that is not seabed and not a known
    # class. Unknown routes it down the unidentified-object path.
    "unknown_anomaly": "unknown",
    "unknown anomaly": "unknown",
}

# Per-class confidence floors, above the detector's own global threshold.
#
# Measured across seven public side-scan tiles, six of them wrecks and none
# containing a human:
#
#   class      n   min    max    note
#   ship       8   0.318  0.829  the workhorse class, behaves well
#   other      3   0.359  0.545  the anomaly model's "I cannot name it"
#   shipwreck  1   0.843
#   human      1   0.463  fired on debris beside a wreck. A false positive.
#   aircraft   1   0.322  fired on a wreck. A false positive.
#
# Both false positives came from known.pt and both sat below 0.5, while the
# correct ship calls clustered higher. The floors below are set from that, and
# from consequence: "human remains" is the highest-consequence claim in the
# whole vocabulary, it is a legal and humanitarian assertion, and the corpus has
# no document supporting any procedure for it. It gets the strictest floor.
#
# One observation per false-positive class is not a fitted threshold. These are
# precautionary and are meant to be retuned against a labelled validation set.
CLASS_CONFIDENCE_FLOOR: dict[str, float] = {
    "human": 0.75,
    "aircraft": 0.60,
    "fish": 0.60,
    "ship": 0.25,
    "shipwreck": 0.25,
    "other": 0.25,
    # marine.pt's own classes. Its raw confidence is calibrated (test ECE
    # 0.041), so the global DETECTOR_CONFIDENCE is the floor for the three
    # classes below; human and aircraft keep the stricter floors above because
    # the consequence of the claim, not the model, sets them.
    "pipeline": 0.25,
    "fishing_gear": 0.25,
    "mine_like_object": 0.25,
}

# A class below its floor is downgraded to this, never deleted. Dropping the box
# would hide a contact from the operator, which is worse than reporting it
# without a name. Downgrading keeps the contact and routes it to the
# unidentified-object protocol, which is the correct handling for something the
# detector saw and cannot confidently name.
#
# "Downgrade" names what happens to the CLAIM, not to the risk. An unidentified
# object is treated as more serious than a confirmed one, so withholding a class
# lowers the assessed risk only where that class outranked unidentified. A
# withheld "human" gets quieter; a withheld "aircraft wreck" gets louder. Both
# are the policy working, and the word misleads if read as de-escalation.
DOWNGRADE_LABEL = "unknown"
DOWNGRADE_NOTE = (
    "the {model} model called this '{cls}' at {confidence:.2f}, below the {floor:.2f} "
    "this system requires before reporting that class. Reported as unidentified "
    "instead. The original call is retained here and nothing was discarded."
)

# Which classes the corpus actually has a document for. A class marked False is
# not a failure of the detector; it is a gap in the references, and the answer
# must say so instead of reaching for the nearest document that sounds close.
CLASS_COVERAGE: dict[str, bool] = {
    "shipwreck": True,
    # Covered since kb/aircraft-wreck-underwater.md was added: the Indian AAIB
    # rules for notification and custody, plus the heritage and military cases.
    "aircraft wreck": True,
    # Covered since kb/human-remains-underwater.md was added. The document is
    # about obligations at such a site, not about identifying remains from a
    # sonar return, and it says so.
    "human remains": True,
    "fish or marine life": False,
    "unknown": True,
    # kb/naval-mine-identification.md and kb/unidentified-object-protocol.md.
    "suspected mine-like object": True,
    # kb/ghost-gear-reporting-india.md and kb/marine-debris.md.
    "derelict fishing gear": True,
}

# Injected when the detector did classify the object and the corpus does cover
# that class. Without it the retrieved unidentified-object protocol dominates
# and the assistant opens with "unidentified object" for a contact its own
# detector called a shipwreck at 0.76, which reads as the system ignoring its
# own model. It adds no confidence the record does not already carry.
CLASSIFIED_NOTE = (
    "CLASSIFICATION: the detector classified this contact as '{label}' at a "
    "confidence of {confidence}. It is a classified detection, not an "
    "unclassified one, so answer about that class specifically and do not open "
    "by calling it unidentified. Say clearly that a classifier output is not a "
    "confirmed identification and state what would confirm or overturn it, "
    "using only the sources."
)

# Injected into the prompt when a detected class has no governing document.
COVERAGE_GAP_NOTE = (
    "CORPUS COVERAGE: the SOURCES below contain no document written about "
    "'{label}'. Say that plainly before anything else. Do not substitute a "
    "document about a different class of object, and do not give a procedure "
    "for one. Give only what the sources genuinely support for an object of "
    "unknown character, and name the gap."
)

# --- Operator vocabulary ---------------------------------------------------
# The index tokenises without stemming, so "notify" and "notification" are
# unrelated terms. An operator asking "who do I notify in India?" scored 0.000
# against the section that answers it, because that section says "notice",
# "notification" and names the authorities, and never says "notify".
#
# Same shape of fix as CLASS_SYNONYMS, and same limits: query-side only, and a
# term the index has never seen carries no weight, so a wrong entry is inert.
TERM_EXPANSIONS: dict[str, str] = {
    "notify": "notification notice authority",
    "notifying": "notification notice authority",
    "notified": "notification notice authority",
    "inform": "notification notice authority",
    "contact": "notification notice authority",
    "call": "notification notice",
    "tell": "notification notice",
    "standoff": "separation distance",
    "disarm": "render safe disposal",
    "defuse": "render safe disposal",
    "salvage": "recovery removal",
    "lift": "recovery removal",
    "raise": "recovery removal",
    "custody": "custody evidence preservation",
    "grave": "human remains venerated",
    "body": "human remains",
    "bodies": "human remains",
    "remains": "human remains venerated",
}

# --- Severity --------------------------------------------------------------
# Severity is looked up, never inferred from the generated text. Reading a risk
# level out of prose would be exactly the unsourced number this system refuses
# to produce.
#
# First table: the free-text `hazard` field already carried by every entry in
# catalog/objects.json, mapped onto the four API values. Keeping the catalog as
# the source of truth stops severity drifting away from the corpus.

HAZARD_SEVERITY: dict[str, str] = {
    "high": "high",
    # A wreck site is the canonical navigation and diving hazard. High, matching
    # the class tier floor the survey engine applies to wrecks
    # (hazard_config.CLASS_TIER_FLOOR), so a contact is never "medium" on one
    # page and "critical" on the next.
    "site hazard": "high",
    "low to moderate": "low",
    "entanglement": "medium",
    "treat as unidentified": "unknown",
    "navigational only": "low",
}

# Second table: object classes a detector may emit that are not catalog ids.
# Matched on normalised substrings, longest first, so "naval mine" beats "mine".
CLASS_SEVERITY: dict[str, str] = {
    # marine.pt's classes, in the corpus vocabulary. A wreck is the canonical
    # navigation hazard and is reported as high, matching the class tier floor
    # the survey engine applies (hazard_config.CLASS_TIER_FLOOR).
    "shipwreck": "high",
    "aircraft wreck": "high",
    # Not a hazard to the vessel. It is a legal and humanitarian obligation, and
    # no source states a risk level, so "unknown" keeps both the caution and the
    # honesty. The interface renders unknown with the same weight as high.
    "human remains": "unknown",
    "fish or marine life": "low",
    "unexploded ordnance": "high",
    "naval mine": "high",
    "sea mine": "high",
    "moored mine": "high",
    "bottom mine": "high",
    "drifting mine": "high",
    "ordnance": "high",
    "torpedo": "high",
    "uxo": "high",
    "mine": "high",
    "pressure vessel": "unknown",
    "gas cylinder": "unknown",
    "drum": "unknown",
    "barrel": "unknown",
    "container": "unknown",
    "derelict fishing gear": "medium",
    "fishing net": "medium",
    "ghost net": "medium",
    "net": "medium",
    "cable": "medium",
    "pipeline": "medium",
    "wreck": "high",
    "aircraft": "high",
    "structural debris": "low",
    "debris": "low",
    "tyre": "low",
    "rock": "low",
    "geological": "low",
    "boulder": "low",
    # "suspected mine-like object" already reaches "mine" (high); stated
    # explicitly so the trained detector's class does not rest on a substring.
    "mine like object": "high",
}

# An unidentified object gets "unknown", never "high". Asserting a risk level
# for an object nobody has identified is itself an unsourced claim. The UI is
# expected to render "unknown" with the same caution as "high".
ANOMALY_SEVERITY = "unknown"
DEFAULT_SEVERITY = "unknown"

# --- Grounding -------------------------------------------------------------

# The prompt asks for [S1], but models render citations their own way. Groq
# emits fullwidth brackets, and both providers collapse runs into [S1, S3, S6].
# Matching only the literal [S1] scored a fully cited answer as ungrounded, so
# bracket groups are found first and source numbers read out of them. A group
# with no S-number in it, such as [Detection], is simply not a citation.
CITATION_GROUP_PATTERN = r"[\[\(\u3010]([^\[\]\(\)\u3010\u3011]{0,80})[\]\)\u3011]"
CITATION_REF_PATTERN = r"S\s*(\d+)"

# Saying "the sources do not cover this" is the correct, designed behaviour, so
# it is reported separately rather than as a grounding failure. An answer can be
# fully cited and still carry one of these, and usually does: this corpus leaves
# most numeric standoff distances unstated on purpose.
#
# Regex rather than fixed strings, because the phrasing varies with the provider
# and the wording. "The provided sources do not contain" and "not specified in
# the available sources" both mean the same thing and neither matches a literal.
REFUSAL_PATTERNS = (
    r"sources?\b[^.]{0,40}?\b(?:do|does)\s+not\s+(?:contain|cover|specify|state|include|provide|give|name|list)",
    # "The provided sources contain no information on ..." says the same thing
    # without a "do not".
    r"sources?\s+(?:contain|provide|include|give|offer|have)\s+no\b",
    r"not\s+(?:specified|stated|named|given|published|provided|defined)\s+in\s+(?:the\s+)?(?:\w+\s+){0,2}sources?",
    r"\bis\s+not\s+(?:specified|stated|named|given|published)\b",
    r"\bnot\s+specified\b",
    r"\bno\s+(?:\w+[\s-]+){0,3}(?:procedure|distance|figure|steps?|authority|source|guidance)\s+is\s+(?:published|specified|given|stated|named|provided)",
    r"\bno\s+authoritative\s+source",
    r"\bcontains?\s+no\s+(?:\w+[\s-]+){0,3}(?:procedure|steps?|guidance)",
    r"\bnot\s+(?:covered|addressed)\s+(?:by|in)\s+the\s+sources?",
)

# Markdown emphasis is stripped before the patterns run, so bold inside a phrase
# ("not **specified** in the sources") does not hide it.
MARKDOWN_NOISE = r"[*_`]+"

# Returned verbatim when retrieval finds nothing. No model call is made, because
# there is nothing to ground an answer in.
NO_SOURCE_ANSWER = (
    "The knowledge base has no document covering that, so I will not give you a "
    "procedure for it.\n\n"
    "Universal fallback, which applies regardless: do not approach, do not touch, "
    "do not attempt recovery, maintain separation, and report the contact to the "
    "responsible maritime authority for expert assessment."
)

# --- Numbers guard ---------------------------------------------------------
# Every figure in an answer must be traceable to text the model was shown: a
# retrieved passage, the operator's own words, the detection record, or a
# GhostTrace context. Anything else is reported in `unsourced_numbers`.
#
# Bare integers below this are not checked. List numbering, a priority rank and
# "paragraph 41" are bare integers, and flagging them would bury the signal. A
# decimal, a number with a unit, and any larger integer (a year, a phone number,
# a count) are checked.
NUMBER_GUARD_MIN_BARE_INTEGER = 100

# Written-out numbers in the corpus, so "24 hours" quoting "twenty-four hours"
# is recognised as the same figure.
NUMBER_WORDS: dict[str, str] = {
    "twenty-four": "24", "twenty four": "24", "forty-eight": "48",
    "one hundred": "100", "seventy-two": "72", "twelve": "12", "ten": "10",
}

# --- GhostTrace handoff ----------------------------------------------------
# A GhostTrace target handed over from the rescue queue. Its numbers are
# computed survey data, not reference text, so they travel in their own block
# with their own citation label and never as a [Sn] source. Authority and
# procedure statements still have to come from retrieved passages.

GHOSTTRACE_CITATION = "GhostTrace output for {survey}/{detection}"

# Added to the retrieval query for every GhostTrace turn, so the ghost-gear
# document is in the SOURCES block whatever the operator typed.
GHOSTTRACE_QUERY_TERMS = "ghost gear abandoned lost fishing gear aldfg authority retrieval report"

# The ghost-gear document is the one that governs a GhostTrace target, and the
# answer needs several of its sections at once: the national position, the
# fisheries authorities and the protected-area offices. A cap of three per
# document dropped the section saying no reporting channel exists.
GHOSTTRACE_PER_DOC = int(os.environ.get("DEEPECHO_GHOSTTRACE_PER_DOC", "5"))
# Six passages rather than eight: the two dropped are the weakest matches, never
# the ghost-gear sections, and they are what keeps a GhostTrace prompt inside the
# free tier's per-request token ceiling described above.
GHOSTTRACE_TOP_K = int(os.environ.get("DEEPECHO_GHOSTTRACE_TOP_K", "6"))

# Per situation named on a GhostTrace authority, the corpus's words for it.
GHOSTTRACE_SITUATION_TERMS: dict[str, str] = {
    "fisheries": "fisheries department fishery survey lost fishing gear",
    "protected_habitat": "marine national park wildlife warden protected area",
    "navigation_hazard": "danger to navigation hydrographic note navarea warning",
    "unknown_contents": "drum cylinder unknown contents",
    "possible_pollution": "pollution coast guard spill",
    "unidentified_object": "unidentified underwater object protocol",
}

# Kept short on purpose. A GhostTrace turn carries this block on top of the
# sources, and Groq's free tier counts prompt plus max_tokens against an 8000
# tokens-per-minute ceiling for a single request: past it the request is refused
# outright (413), not throttled.
GHOSTTRACE_NOTE = """GHOSTTRACE CONTEXT -- DATA FROM THE SURVEY, NOT A SOURCE
Computed survey data (detector output, heuristic scores, model forecasts), not \
reference text. Rules, in addition to every standing rule:
A. Quote its numbers exactly as written, attributed "({citation})". Never put an \
[Sn] marker on a GhostTrace number, and never call one something the sources state.
B. Do not calculate any number that is not already written in this block: no new \
sums, no tier thresholds, no percentages, no unit conversions.
C. Authorities, reporting channels, recovery and diving guidance must come from \
the SOURCES block with [Sn] citations. The authorities listed here are pointers: \
name one only where a SOURCES passage names it for that situation.
D. Null or missing means not available. Never fill it in.
E. GhostTrace scores are configurable heuristics, not an official procedure.{synthetic}

{body}"""

GHOSTTRACE_SYNTHETIC_RULE = """
F. SYNTHETIC: this survey is synthetic demonstration data. Say so in your first \
sentence. Nothing in it is evidence of a real net."""

GHOSTTRACE_TASK = """The operator is asking about the GhostTrace target above.

QUESTION: {question}

{opening}Answer the question. Unless the question is only about one narrow \
point, cover, in this order: why GhostTrace ranked this target where it did, \
explained term by term from the priority lines (value x weight = contribution, \
figures as written), the multiplier and the score, attributed by writing \
"({citation})" immediately after the score (this exact attribution, at least \
once), and stating that the weights and tiers are a configurable heuristic, not \
an official procedure; what the activity, habitat, drift, people and change fields \
say, with their limitations; and who the SOURCES say is responsible for this \
situation, cited [Sn], including where the sources say no reporting channel \
exists. Keep it under 300 words."""

# The first line of every answer about a synthetic target. Given verbatim,
# because an instruction to "mention" it loses to the standing instruction to
# lead with the risk.
GHOSTTRACE_SYNTHETIC_OPENING = (
    'Begin your answer with exactly this sentence: "SYNTHETIC DATA: this '
    'GhostTrace target comes from synthetic demonstration data and is not '
    'evidence of a real net." '
)

# Asked when a GhostTrace target arrives without a question of its own.
GHOSTTRACE_DEFAULT_QUESTION = (
    "Why is this net ranked where it is, and who do the sources say should be told?"
)

# --- Offline fallback ------------------------------------------------------
# When no provider can answer -- no key, a rate limit, no network -- the operator
# still gets what retrieval found, clearly labelled as such, rather than an
# error. Nothing in the fallback is generated: the extracts are cut from the
# passages verbatim and the context facts are copied from the handoff.

OFFLINE_PASSAGES = int(os.environ.get("DEEPECHO_OFFLINE_PASSAGES", "4"))
OFFLINE_EXTRACT_CHARS = int(os.environ.get("DEEPECHO_OFFLINE_EXTRACT_CHARS", "320"))
OFFLINE_TITLE = "Offline — sources only, no generated answer"
OFFLINE_INTRO = (
    "No language model could be reached, so nothing below was written by a model. "
    "These are the passages retrieval found for your question, quoted as short "
    "extracts. Read the cited source before acting on any of them."
)

# --- Mission Copilot ---------------------------------------------------------
# Questions about the processed surveys themselves ("which net first?", "what
# changed?") are answered from read-only data tools over data/surveys
# (backend/copilot_tools.py) plus the corpus. Tool results travel as DATA
# records cited [D1], [D2]; corpus passages stay [S1], [S2]. The model never
# reads the survey files itself: it picks tools, the tools run here, and only
# their compact results reach the answer prompt.

SURVEYS_DIR = Path(os.environ.get("DEEPECHO_SURVEYS_DIR", str(ROOT / "data" / "surveys")))

# The request's `mode`. "auto" routes by the rule in chat.route_mode; "copilot"
# and "reference" force one path.
ASSISTANT_MODES = ("auto", "copilot", "reference")

# Hard caps. A planner that asks for ten lookups is answered with four.
COPILOT_MAX_TOOL_CALLS = int(os.environ.get("DEEPECHO_COPILOT_MAX_TOOLS", "4"))
# Planning round trips. One is enough because the survey catalogue is in the
# planning prompt; two lets the model look something up and then refine.
COPILOT_PLAN_ROUNDS = int(os.environ.get("DEEPECHO_COPILOT_PLAN_ROUNDS", "1"))
# Wall-clock budget for planning across every provider tried. Past it the
# deterministic keyword plan is used instead.
COPILOT_PLAN_TIMEOUT_S = float(os.environ.get("DEEPECHO_COPILOT_PLAN_TIMEOUT", "20"))
# Records rendered into the prompt, across all tool calls. Keeps a copilot
# prompt inside a free tier's per-request token ceiling.
COPILOT_MAX_DATA_RECORDS = int(os.environ.get("DEEPECHO_COPILOT_MAX_RECORDS", "24"))
COPILOT_TOP_K = int(os.environ.get("DEEPECHO_COPILOT_TOP_K", "6"))
COPILOT_PER_DOC = int(os.environ.get("DEEPECHO_COPILOT_PER_DOC", "3"))

DATA_CITATION_REF_PATTERN = r"(?<![A-Za-z0-9_])D\s*(\d+)"

# Auto-routing. A turn goes to the copilot only when nothing is attached (a
# detection record or a GhostTrace target means the operator is asking about
# that one contact) and the words point at the survey data:
#   * any STRONG phrase, or
#   * a MEDIUM phrase together with a word that asks for records, or
#   * a named survey (an id, or a token of one) together with a DATA noun.
# "I found a ghost net near the Gulf of Mannar, who do I tell?" names Mannar
# but asks a reference question, which is why a survey name alone is not enough,
# and "how do I tell a false positive from a rock?" is a reference question too,
# which is why "false positive" alone is not enough.
COPILOT_STRONG_PHRASES = (
    "all surveys", "across surveys", "across the surveys", "both surveys", "two surveys",
    "our surveys", "which surveys", "what surveys", "processed survey", "survey data",
    "hotspot", "ghosttrace", "filtered out as", "filtered as", "were filtered",
    "was filtered", "were suppressed", "what changed", "changed between",
    "recover first", "first to recover", "recovery order", "this survey",
    "survey summary", "summarise survey", "summarize survey", "summarise the survey",
    "summarize the survey",
)
COPILOT_MEDIUM_PHRASES = (
    "false positive", "false-positive", "detections", "surveys", "suppressed",
    "targets", "nets",
)
COPILOT_RECORD_WORDS = (
    "which", "were", "how many", "list", "show me", "found", "our ", "the survey",
    "across", "ranked", "rank", "first",
)
COPILOT_DATA_NOUNS = (
    "survey", "how many", "were found", "found in", "detected in", "targets",
    "nets", "objects", "contacts", "summar", "overview", "changed",
    "moved", "list", "ranked", "priority",
)
# Id tokens too generic to name a survey on their own.
COPILOT_GENERIC_ID_TOKENS = {
    "demo", "synthetic", "strip", "repeat", "ghosttrace", "survey", "line", "data",
}
# Other spellings of survey tokens: the scripts an operator may type them in.
COPILOT_SURVEY_ALIASES: dict[str, tuple[str, ...]] = {
    "mannar": ("मन्नार", "மன்னார்", "മാന്നാർ", "മന്നാർ", "మన్నార్", "মান্নার", "ಮನ್ನಾರ್",
               "મન્નાર", "ମାନ୍ନାର"),
    "waterfall": ("वॉटरफॉल",),
    "submarine": ("पनडुब्बी", "நீர்மூழ்கி"),
}
# Native-script keywords, mapped to the English word the keyword planner reads.
# Used only when no translation was obtained, so a provider outage does not make
# an Indic question unanswerable. A wrong entry fails to match; it cannot add a
# tool the router would not otherwise allow.
COPILOT_NATIVE_KEYWORDS: dict[str, str] = {
    # Hindi / Marathi (Devanagari)
    "सर्वे": "survey", "सर्वेक्षण": "survey", "जाल": "net", "जाळे": "net",
    "बदला": "what changed", "बदल": "what changed", "सारांश": "summarise survey",
    "कितने": "how many", "किती": "how many", "बारूदी सुरंग": "mine", "सुरंग": "mine",
    "निकाल": "recover first", "प्राथमिकता": "priority", "गलत पहचान": "false positive",
    "फ़िल्टर": "filtered as", "हॉटस्पॉट": "hotspot",
    # Tamil
    "ஆய்வு": "survey", "சர்வே": "survey", "வலை": "net", "மாற்ற": "what changed",
    "சுருக்க": "summarise survey", "எத்தனை": "how many", "கண்ணிவெடி": "mine",
    # Malayalam
    "സർവേ": "survey", "വല": "net", "മാറ്റ": "what changed", "സംഗ്രഹ": "summarise survey",
    "എത്ര": "how many", "കുഴിബോംബ്": "mine",
    # Telugu
    "సర్వే": "survey", "వల": "net", "మార్పు": "what changed", "సారాంశ": "summarise survey",
    "ఎన్ని": "how many", "మందుపాతర": "mine",
    # Bengali
    "জরিপ": "survey", "সার্ভে": "survey", "জাল": "net", "পরিবর্তন": "what changed",
    "সারাংশ": "summarise survey", "কতগুলো": "how many", "মাইন": "mine",
    # Kannada
    "ಸಮೀಕ್ಷೆ": "survey", "ಸರ್ವೆ": "survey", "ಬಲೆ": "net", "ಬದಲಾ": "what changed",
    "ಸಾರಾಂಶ": "summarise survey", "ಎಷ್ಟು": "how many",
    # Gujarati
    "સર્વે": "survey", "જાળ": "net", "ફેરફાર": "what changed", "સારાંશ": "summarise survey",
    "કેટલા": "how many",
    # Odia
    "ସର୍ଭେ": "survey", "ଜାଲ": "net", "ପରିବର୍ତ୍ତନ": "what changed", "ସାରାଂଶ": "summarise survey",
    "କେତେ": "how many",
}

# Detector classes grouped the way an operator asks for them. "Mine-like" never
# includes a cylinder: the export's own recommended action is quoted for those
# instead, so the answer can say what the data says without promoting a class.
COPILOT_CLASS_FAMILIES: dict[str, tuple[str, ...]] = {
    "mine": ("mine", "naval mine", "sea mine", "moored mine", "bottom mine", "uxo",
             "unexploded ordnance", "ordnance", "torpedo", "mine like object",
             "suspected mine like object"),
    "net": ("net", "ghost net", "ghost gear", "fishing gear", "fishing net",
            "derelict fishing gear", "entangled net", "rope"),
    "wreck": ("shipwreck", "wreck", "ship", "aircraft", "aircraft wreck"),
    "debris": ("debris", "drum", "barrel", "tire", "tyre", "bottle", "container",
               "structural debris"),
    "cylinder": ("cylinder", "gas cylinder", "pressure vessel"),
    "pipe": ("pipe", "pipeline", "cable", "chain"),
    "unknown": ("unknown", "unknown anomaly", "other", "anomaly", "unidentified"),
}
# Words an operator uses for a family. Plurals are stripped before lookup.
COPILOT_CLASS_WORDS: dict[str, str] = {
    "mine": "mine", "mine like": "mine", "mine like object": "mine", "uxo": "mine",
    "ordnance": "mine", "munition": "mine", "torpedo": "mine",
    "net": "net", "ghost net": "net", "ghost gear": "net", "fishing gear": "net",
    "wreck": "wreck", "shipwreck": "wreck", "ship": "wreck", "aircraft": "wreck",
    "debris": "debris", "drum": "debris", "barrel": "debris", "tyre": "debris",
    "tire": "debris", "bottle": "debris",
    "cylinder": "cylinder", "pipe": "pipe", "pipeline": "pipe", "cable": "pipe",
    "chain": "pipe", "unknown": "unknown", "anomaly": "unknown", "unidentified": "unknown",
}

# Retrieval terms added per tool, so the corpus passages an answer needs are in
# SOURCES whatever the planner chose.
COPILOT_TOOL_QUERY_TERMS: dict[str, str] = {
    "ghosttrace_targets": "ghost gear abandoned lost fishing gear aldfg authority retrieval report",
    "change_report": "ghost gear abandoned lost fishing gear report position",
    "filtered_detections": "false contact side scan shadow nadir natural rock identification",
    "find_detections": "notification notice authority report position",
    "top_hotspots": "danger to navigation notification notice authority",
    "survey_summary": "report contents position notification notice authority coast guard",
    "list_surveys": "report contents position",
}

COPILOT_NOTE = """MISSION COPILOT -- SURVEY DATA AND SOURCES
The DATA block below holds the results of read-only lookups over the processed \
surveys. It is admissible evidence for survey facts, alongside the SOURCES block \
for procedures. Rules, in addition to every standing rule:
A. Survey facts (counts, classes, positions, scores, tiers, changes, reasons a \
contact was filtered) come only from DATA records. Cite the record right after \
the fact as [D1], [D2]. Procedures, obligations and authorities come only from \
SOURCES, cited [S1], [S2]. Never put an [S#] on a survey fact or a [D#] on a procedure.
B. Copy every number exactly as a DATA record or a SOURCES passage writes it. Do \
not calculate anything new: no sums, differences, averages, percentages or unit \
conversions. Use the counts the query_result records give; do not count records yourself.
C. Where the DATA does not answer the question (no matching records, an unknown \
survey, a null field), say so plainly. Null means not available.
D. If any cited record says "synthetic": true, your first sentence says the \
survey data is synthetic demonstration data and is not evidence of real objects.
E. A class is detector output, not a confirmed identification. Never call a \
contact a mine; say mine-like object or give the class as recorded. Severity, \
risk scores and GhostTrace priorities are configurable heuristics, not official procedure.
F. Name an authority only where a SOURCES passage names it for that situation. \
Authorities listed in DATA are pointers to check against SOURCES.
G. Lead with the direct answer. Under 250 words. A short table is fine for \
comparing records.

{data}"""

COPILOT_TASK = """QUESTION: {question}

Answer from the DATA and SOURCES above, following rules A to G."""

COPILOT_NO_DATA = ("No survey data lookups ran for this question, so there is no "
                   "survey record to cite.")

COPILOT_PLANNER_SYSTEM = """You are the planning step of the DeepEcho Mission \
Copilot. Pick read-only data tools that fetch what is needed to answer the \
operator's question about processed side-scan sonar surveys. Do not answer the \
question. Call at most {max_calls} tools. Use survey ids exactly as listed below; \
leave survey_id empty to search every survey. For "which net first" questions use \
ghosttrace_targets without a survey_id. For "what changed" use change_report. For \
false positives use filtered_detections. Prefer one precise call to several vague ones.

PROCESSED SURVEYS:
{catalogue}"""

COPILOT_JSON_PLAN = """Reply with JSON only, no prose, in exactly this shape:
{{"calls": [{{"name": "<tool name>", "args": {{...}}}}]}}

TOOLS:
{tools}"""

# --- Languages -------------------------------------------------------------
# Answers can be written in any of these. Retrieval stays English because the
# corpus is English: a non-English question is translated for search when a
# provider is available, and searched as typed when not. Source extracts are
# never translated.

LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"name": "English", "native": "English"},
    "hi": {"name": "Hindi", "native": "हिन्दी"},
    "ta": {"name": "Tamil", "native": "தமிழ்"},
    "ml": {"name": "Malayalam", "native": "മലയാളം"},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ"},
    "te": {"name": "Telugu", "native": "తెలుగు"},
    "bn": {"name": "Bengali", "native": "বাংলা"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ"},
    "mr": {"name": "Marathi", "native": "मराठी"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી"},
}
DEFAULT_LANGUAGE = "en"

LANGUAGE_INSTRUCTION = """LANGUAGE: Write the whole answer in {name} ({native}), \
for fishermen, coast guard and field teams. Keep these exactly as they are, in \
Latin letters and ASCII digits: every citation marker ([S1], [D1]), every number, \
survey id and detection id. Give names of authorities, organisations, places and \
documents in {name}, followed by the original English form in parentheses where \
that helps, for example the first time a name appears. Any sentence you were told \
to write verbatim stays exactly as given."""

TRANSLATE_SYSTEM = ("Translate the operator's message into plain English for a "
                    "document search. Keep names, survey ids, numbers and units "
                    "exactly. Reply with the translation only.")
TRANSLATE_TIMEOUT_S = float(os.environ.get("DEEPECHO_TRANSLATE_TIMEOUT", "12"))

# Offline answer shells, per language. Only the headings and the framing are
# translated; the extracts and data values stay as written.
OFFLINE_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "data_title": "Offline — survey data and sources only, no generated answer",
        "data_intro": ("No language model could be reached, so nothing below was written by "
                       "a model. These are the survey records the copilot looked up for your "
                       "question, and short extracts of the reference passages retrieval "
                       "found. Read the cited record or source before acting."),
        "data_heading": "Survey data consulted",
        "sources_heading": "Reference extracts (English)",
        "synthetic": ("Synthetic demonstration data: nothing marked synthetic is evidence of "
                      "a real object."),
        "no_records": "No records matched.",
        "retrieval_title": OFFLINE_TITLE,
        "retrieval_intro": OFFLINE_INTRO,
        "passages_heading": "Retrieved passages",
    },
    "hi": {
        "data_title": "ऑफ़लाइन — केवल सर्वे डेटा और स्रोत, कोई जनरेट किया गया उत्तर नहीं",
        "data_intro": ("किसी भी भाषा मॉडल से संपर्क नहीं हो सका, इसलिए नीचे कुछ भी मॉडल द्वारा "
                       "नहीं लिखा गया है। ये वे सर्वे रिकॉर्ड हैं जो आपके प्रश्न के लिए देखे गए, और "
                       "संदर्भ दस्तावेज़ों के छोटे अंश (अंग्रेज़ी में)। कार्रवाई से पहले उद्धृत रिकॉर्ड "
                       "या स्रोत पढ़ें।"),
        "data_heading": "देखा गया सर्वे डेटा",
        "sources_heading": "संदर्भ अंश (अंग्रेज़ी)",
        "synthetic": "सिंथेटिक प्रदर्शन डेटा: सिंथेटिक चिह्नित कुछ भी किसी वास्तविक वस्तु का प्रमाण नहीं है।",
        "no_records": "कोई रिकॉर्ड मेल नहीं खाया।",
        "retrieval_title": "ऑफ़लाइन — केवल स्रोत, कोई जनरेट किया गया उत्तर नहीं",
        "retrieval_intro": ("किसी भी भाषा मॉडल से संपर्क नहीं हो सका, इसलिए नीचे कुछ भी मॉडल द्वारा "
                            "नहीं लिखा गया है। ये आपके प्रश्न के लिए खोजे गए अंश हैं (अंग्रेज़ी में)। "
                            "कार्रवाई से पहले उद्धृत स्रोत पढ़ें।"),
        "passages_heading": "खोजे गए अंश",
    },
    "ta": {
        "data_title": "ஆஃப்லைன் — ஆய்வுத் தரவும் மூலங்களும் மட்டும், உருவாக்கப்பட்ட பதில் இல்லை",
        "data_intro": ("எந்த மொழி மாதிரியையும் அணுக முடியவில்லை, எனவே கீழே உள்ள எதுவும் மாதிரியால் "
                       "எழுதப்படவில்லை. உங்கள் கேள்விக்காகப் பார்க்கப்பட்ட ஆய்வுப் பதிவுகளும், குறிப்பு "
                       "ஆவணங்களின் சிறு பகுதிகளும் (ஆங்கிலத்தில்) இவை. செயல்படுவதற்கு முன் மேற்கோள் "
                       "காட்டப்பட்ட பதிவையோ மூலத்தையோ படிக்கவும்."),
        "data_heading": "பார்க்கப்பட்ட ஆய்வுத் தரவு",
        "sources_heading": "குறிப்புப் பகுதிகள் (ஆங்கிலம்)",
        "synthetic": "செயற்கை செயல்விளக்கத் தரவு: செயற்கை எனக் குறிக்கப்பட்ட எதுவும் உண்மையான பொருளுக்கான சான்று அல்ல.",
        "no_records": "பொருந்தும் பதிவுகள் இல்லை.",
        "retrieval_title": "ஆஃப்லைன் — மூலங்கள் மட்டும், உருவாக்கப்பட்ட பதில் இல்லை",
        "retrieval_intro": ("எந்த மொழி மாதிரியையும் அணுக முடியவில்லை, எனவே கீழே உள்ள எதுவும் மாதிரியால் "
                            "எழுதப்படவில்லை. உங்கள் கேள்விக்காகக் கண்டெடுக்கப்பட்ட பகுதிகள் இவை "
                            "(ஆங்கிலத்தில்). செயல்படுவதற்கு முன் மேற்கோள் காட்டப்பட்ட மூலத்தைப் படிக்கவும்."),
        "passages_heading": "கண்டெடுக்கப்பட்ட பகுதிகள்",
    },
    "ml": {
        "data_title": "ഓഫ്‌ലൈൻ — സർവേ ഡാറ്റയും ഉറവിടങ്ങളും മാത്രം, സൃഷ്ടിച്ച ഉത്തരമില്ല",
        "data_intro": ("ഒരു ഭാഷാ മോഡലിലേക്കും എത്താനായില്ല, അതിനാൽ താഴെയുള്ളതൊന്നും മോഡൽ എഴുതിയതല്ല. "
                       "നിങ്ങളുടെ ചോദ്യത്തിനായി പരിശോധിച്ച സർവേ രേഖകളും റഫറൻസ് രേഖകളിൽ നിന്നുള്ള ചെറിയ "
                       "ഭാഗങ്ങളും (ഇംഗ്ലീഷിൽ) ആണ് ഇവ. നടപടിയെടുക്കുന്നതിന് മുമ്പ് ഉദ്ധരിച്ച രേഖയോ ഉറവിടമോ വായിക്കുക."),
        "data_heading": "പരിശോധിച്ച സർവേ ഡാറ്റ",
        "sources_heading": "റഫറൻസ് ഭാഗങ്ങൾ (ഇംഗ്ലീഷ്)",
        "synthetic": "കൃത്രിമ പ്രദർശന ഡാറ്റ: കൃത്രിമം എന്ന് അടയാളപ്പെടുത്തിയ ഒന്നും യഥാർത്ഥ വസ്തുവിന്റെ തെളിവല്ല.",
        "no_records": "പൊരുത്തപ്പെടുന്ന രേഖകളില്ല.",
        "retrieval_title": "ഓഫ്‌ലൈൻ — ഉറവിടങ്ങൾ മാത്രം, സൃഷ്ടിച്ച ഉത്തരമില്ല",
        "retrieval_intro": ("ഒരു ഭാഷാ മോഡലിലേക്കും എത്താനായില്ല, അതിനാൽ താഴെയുള്ളതൊന്നും മോഡൽ എഴുതിയതല്ല. "
                            "നിങ്ങളുടെ ചോദ്യത്തിനായി കണ്ടെത്തിയ ഭാഗങ്ങളാണ് ഇവ (ഇംഗ്ലീഷിൽ). "
                            "നടപടിയെടുക്കുന്നതിന് മുമ്പ് ഉദ്ധരിച്ച ഉറവിടം വായിക്കുക."),
        "passages_heading": "കണ്ടെത്തിയ ഭാഗങ്ങൾ",
    },
    "or": {
        "data_title": "ଅଫଲାଇନ୍ — କେବଳ ସର୍ଭେ ତଥ୍ୟ ଓ ଉତ୍ସ, କୌଣସି ସୃଷ୍ଟ ଉତ୍ତର ନାହିଁ",
        "data_intro": ("କୌଣସି ଭାଷା ମଡେଲ୍ ସହ ଯୋଗାଯୋଗ ହୋଇପାରିଲା ନାହିଁ, ତେଣୁ ତଳେ ଥିବା କିଛି ବି ମଡେଲ୍ "
                       "ଦ୍ୱାରା ଲେଖାଯାଇନାହିଁ। ଏଗୁଡ଼ିକ ଆପଣଙ୍କ ପ୍ରଶ୍ନ ପାଇଁ ଦେଖାଯାଇଥିବା ସର୍ଭେ ରେକର୍ଡ ଓ ସନ୍ଦର୍ଭ "
                       "ଦଲିଲର ଛୋଟ ଅଂଶ (ଇଂରାଜୀରେ)। କାର୍ଯ୍ୟ କରିବା ପୂର୍ବରୁ ଉଦ୍ଧୃତ ରେକର୍ଡ ବା ଉତ୍ସ ପଢ଼ନ୍ତୁ।"),
        "data_heading": "ଦେଖାଯାଇଥିବା ସର୍ଭେ ତଥ୍ୟ",
        "sources_heading": "ସନ୍ଦର୍ଭ ଅଂଶ (ଇଂରାଜୀ)",
        "synthetic": "କୃତ୍ରିମ ପ୍ରଦର୍ଶନ ତଥ୍ୟ: କୃତ୍ରିମ ଚିହ୍ନିତ କିଛି ବି ପ୍ରକୃତ ବସ୍ତୁର ପ୍ରମାଣ ନୁହେଁ।",
        "no_records": "କୌଣସି ମେଳ ଖାଉଥିବା ରେକର୍ଡ ନାହିଁ।",
        "retrieval_title": "ଅଫଲାଇନ୍ — କେବଳ ଉତ୍ସ, କୌଣସି ସୃଷ୍ଟ ଉତ୍ତର ନାହିଁ",
        "retrieval_intro": ("କୌଣସି ଭାଷା ମଡେଲ୍ ସହ ଯୋଗାଯୋଗ ହୋଇପାରିଲା ନାହିଁ, ତେଣୁ ତଳେ ଥିବା କିଛି ବି ମଡେଲ୍ "
                            "ଦ୍ୱାରା ଲେଖାଯାଇନାହିଁ। ଏଗୁଡ଼ିକ ଆପଣଙ୍କ ପ୍ରଶ୍ନ ପାଇଁ ମିଳିଥିବା ଅଂଶ (ଇଂରାଜୀରେ)। "
                            "କାର୍ଯ୍ୟ କରିବା ପୂର୍ବରୁ ଉଦ୍ଧୃତ ଉତ୍ସ ପଢ଼ନ୍ତୁ।"),
        "passages_heading": "ମିଳିଥିବା ଅଂଶ",
    },
    "te": {
        "data_title": "ఆఫ్‌లైన్ — సర్వే డేటా మరియు మూలాలు మాత్రమే, రూపొందించిన సమాధానం లేదు",
        "data_intro": ("ఏ భాషా మోడల్‌ను చేరుకోలేకపోయాం, కాబట్టి క్రింద ఉన్నదేదీ మోడల్ రాసినది కాదు. "
                       "మీ ప్రశ్న కోసం చూసిన సర్వే రికార్డులు, రిఫరెన్స్ పత్రాల చిన్న భాగాలు (ఆంగ్లంలో) "
                       "ఇవి. చర్య తీసుకునే ముందు ఉదహరించిన రికార్డు లేదా మూలాన్ని చదవండి."),
        "data_heading": "చూసిన సర్వే డేటా",
        "sources_heading": "రిఫరెన్స్ భాగాలు (ఆంగ్లం)",
        "synthetic": "కృత్రిమ ప్రదర్శన డేటా: కృత్రిమం అని గుర్తించినదేదీ నిజమైన వస్తువుకు రుజువు కాదు.",
        "no_records": "సరిపోలే రికార్డులు లేవు.",
        "retrieval_title": "ఆఫ్‌లైన్ — మూలాలు మాత్రమే, రూపొందించిన సమాధానం లేదు",
        "retrieval_intro": ("ఏ భాషా మోడల్‌ను చేరుకోలేకపోయాం, కాబట్టి క్రింద ఉన్నదేదీ మోడల్ రాసినది కాదు. "
                            "మీ ప్రశ్న కోసం కనుగొన్న భాగాలు ఇవి (ఆంగ్లంలో). చర్య తీసుకునే ముందు "
                            "ఉదహరించిన మూలాన్ని చదవండి."),
        "passages_heading": "కనుగొన్న భాగాలు",
    },
    "bn": {
        "data_title": "অফলাইন — শুধু জরিপের তথ্য ও উৎস, কোনো তৈরি উত্তর নেই",
        "data_intro": ("কোনো ভাষা মডেলের সঙ্গে যোগাযোগ করা যায়নি, তাই নিচের কিছুই মডেলের লেখা নয়। "
                       "এগুলো আপনার প্রশ্নের জন্য দেখা জরিপ রেকর্ড এবং রেফারেন্স নথির ছোট অংশ "
                       "(ইংরেজিতে)। পদক্ষেপ নেওয়ার আগে উদ্ধৃত রেকর্ড বা উৎস পড়ুন।"),
        "data_heading": "দেখা জরিপের তথ্য",
        "sources_heading": "রেফারেন্স অংশ (ইংরেজি)",
        "synthetic": "কৃত্রিম প্রদর্শনী তথ্য: কৃত্রিম চিহ্নিত কিছুই বাস্তব বস্তুর প্রমাণ নয়।",
        "no_records": "কোনো মিলে যাওয়া রেকর্ড নেই।",
        "retrieval_title": "অফলাইন — শুধু উৎস, কোনো তৈরি উত্তর নেই",
        "retrieval_intro": ("কোনো ভাষা মডেলের সঙ্গে যোগাযোগ করা যায়নি, তাই নিচের কিছুই মডেলের লেখা নয়। "
                            "এগুলো আপনার প্রশ্নের জন্য পাওয়া অংশ (ইংরেজিতে)। পদক্ষেপ নেওয়ার আগে "
                            "উদ্ধৃত উৎস পড়ুন।"),
        "passages_heading": "পাওয়া অংশ",
    },
    "kn": {
        "data_title": "ಆಫ್‌ಲೈನ್ — ಸಮೀಕ್ಷೆ ದತ್ತಾಂಶ ಮತ್ತು ಮೂಲಗಳು ಮಾತ್ರ, ರಚಿಸಿದ ಉತ್ತರವಿಲ್ಲ",
        "data_intro": ("ಯಾವುದೇ ಭಾಷಾ ಮಾದರಿಯನ್ನು ತಲುಪಲಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಕೆಳಗಿನ ಯಾವುದನ್ನೂ ಮಾದರಿ "
                       "ಬರೆದಿಲ್ಲ. ನಿಮ್ಮ ಪ್ರಶ್ನೆಗಾಗಿ ನೋಡಿದ ಸಮೀಕ್ಷೆ ದಾಖಲೆಗಳು ಮತ್ತು ಉಲ್ಲೇಖ ದಾಖಲೆಗಳ ಸಣ್ಣ "
                       "ಭಾಗಗಳು (ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ) ಇವು. ಕ್ರಮ ಕೈಗೊಳ್ಳುವ ಮೊದಲು ಉಲ್ಲೇಖಿಸಿದ ದಾಖಲೆ ಅಥವಾ ಮೂಲವನ್ನು ಓದಿ."),
        "data_heading": "ನೋಡಿದ ಸಮೀಕ್ಷೆ ದತ್ತಾಂಶ",
        "sources_heading": "ಉಲ್ಲೇಖ ಭಾಗಗಳು (ಇಂಗ್ಲಿಷ್)",
        "synthetic": "ಕೃತಕ ಪ್ರದರ್ಶನ ದತ್ತಾಂಶ: ಕೃತಕ ಎಂದು ಗುರುತಿಸಿದ ಯಾವುದೂ ನಿಜವಾದ ವಸ್ತುವಿನ ಪುರಾವೆಯಲ್ಲ.",
        "no_records": "ಹೊಂದುವ ದಾಖಲೆಗಳಿಲ್ಲ.",
        "retrieval_title": "ಆಫ್‌ಲೈನ್ — ಮೂಲಗಳು ಮಾತ್ರ, ರಚಿಸಿದ ಉತ್ತರವಿಲ್ಲ",
        "retrieval_intro": ("ಯಾವುದೇ ಭಾಷಾ ಮಾದರಿಯನ್ನು ತಲುಪಲಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಕೆಳಗಿನ ಯಾವುದನ್ನೂ ಮಾದರಿ "
                            "ಬರೆದಿಲ್ಲ. ನಿಮ್ಮ ಪ್ರಶ್ನೆಗಾಗಿ ಸಿಕ್ಕ ಭಾಗಗಳು ಇವು (ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ). ಕ್ರಮ ಕೈಗೊಳ್ಳುವ "
                            "ಮೊದಲು ಉಲ್ಲೇಖಿಸಿದ ಮೂಲವನ್ನು ಓದಿ."),
        "passages_heading": "ಸಿಕ್ಕ ಭಾಗಗಳು",
    },
    "mr": {
        "data_title": "ऑफलाइन — फक्त सर्वेक्षण डेटा आणि स्रोत, तयार केलेले उत्तर नाही",
        "data_intro": ("कोणत्याही भाषा मॉडेलशी संपर्क होऊ शकला नाही, त्यामुळे खालील काहीही मॉडेलने "
                       "लिहिलेले नाही. तुमच्या प्रश्नासाठी पाहिलेले सर्वेक्षण रेकॉर्ड आणि संदर्भ "
                       "दस्तऐवजांचे छोटे उतारे (इंग्रजीत) हे आहेत. कृती करण्यापूर्वी उद्धृत रेकॉर्ड किंवा स्रोत वाचा."),
        "data_heading": "पाहिलेला सर्वेक्षण डेटा",
        "sources_heading": "संदर्भ उतारे (इंग्रजी)",
        "synthetic": "कृत्रिम प्रात्यक्षिक डेटा: कृत्रिम म्हणून चिन्हांकित काहीही खऱ्या वस्तूचा पुरावा नाही.",
        "no_records": "जुळणारे रेकॉर्ड नाहीत.",
        "retrieval_title": "ऑफलाइन — फक्त स्रोत, तयार केलेले उत्तर नाही",
        "retrieval_intro": ("कोणत्याही भाषा मॉडेलशी संपर्क होऊ शकला नाही, त्यामुळे खालील काहीही मॉडेलने "
                            "लिहिलेले नाही. तुमच्या प्रश्नासाठी सापडलेले उतारे हे आहेत (इंग्रजीत). "
                            "कृती करण्यापूर्वी उद्धृत स्रोत वाचा."),
        "passages_heading": "सापडलेले उतारे",
    },
    "gu": {
        "data_title": "ઑફલાઇન — માત્ર સર્વે ડેટા અને સ્રોતો, કોઈ બનાવેલો જવાબ નહીં",
        "data_intro": ("કોઈ પણ ભાષા મૉડલ સુધી પહોંચી શકાયું નહીં, તેથી નીચેનું કંઈ પણ મૉડલે લખ્યું નથી. "
                       "તમારા પ્રશ્ન માટે જોયેલા સર્વે રેકોર્ડ અને સંદર્ભ દસ્તાવેજોના ટૂંકા અંશો "
                       "(અંગ્રેજીમાં) આ છે. પગલાં લેતા પહેલાં ટાંકેલો રેકોર્ડ અથવા સ્રોત વાંચો."),
        "data_heading": "જોયેલો સર્વે ડેટા",
        "sources_heading": "સંદર્ભ અંશો (અંગ્રેજી)",
        "synthetic": "કૃત્રિમ નિદર્શન ડેટા: કૃત્રિમ તરીકે ચિહ્નિત કંઈ પણ વાસ્તવિક વસ્તુનો પુરાવો નથી.",
        "no_records": "કોઈ મેળ ખાતા રેકોર્ડ નથી.",
        "retrieval_title": "ઑફલાઇન — માત્ર સ્રોતો, કોઈ બનાવેલો જવાબ નહીં",
        "retrieval_intro": ("કોઈ પણ ભાષા મૉડલ સુધી પહોંચી શકાયું નહીં, તેથી નીચેનું કંઈ પણ મૉડલે લખ્યું નથી. "
                            "તમારા પ્રશ્ન માટે મળેલા અંશો આ છે (અંગ્રેજીમાં). પગલાં લેતા પહેલાં ટાંકેલો સ્રોત વાંચો."),
        "passages_heading": "મળેલા અંશો",
    },
}

# --- Streaming -------------------------------------------------------------
# Server-sent events. Every frame is `data: {json}` with a `type` field, so a
# plain fetch reader handles it and no EventSource-only GET route is needed.

STREAM_MEDIA_TYPE = "text/event-stream"

# Retrieval finishes before generation starts, so the citations are already
# known when the first word is sent. Emitting them up front fills the panel
# while the answer is still arriving. Turn this off to hold everything back
# until the final frame instead.
STREAM_SOURCES_EARLY = True

# Frame names, in the order a well-behaved stream emits them:
#   meta    intent, severity, is_anomaly and the catalog matches
#   sources the citations (early, unless STREAM_SOURCES_EARLY is off)
#   delta   one piece of answer text
#   done    the complete response, including grounded, which can only be
#           computed once the whole answer exists
#   error   the stream failed partway; whatever text arrived still stands
STREAM_FRAMES = ("meta", "sources", "delta", "done", "error")

# --- Detector (wired in a later step) --------------------------------------


# One checkpoint: marine.pt, the team's trained YOLO11s. It emits the six
# classes the corpus is about (shipwreck, aircraft, human, pipeline,
# fishing_gear, mine_like_object), so the earlier pair of stand-in models
# (known.pt + anomaly.pt) is retired. The dict shape stays so a second model
# can be added again without touching the worker or the merge.
#
# The kit lives in models/marine/: the weights, the calibration the team fitted
# (identity; test ECE 0.041), and the scripts the survey pipeline runs
# (sonar_detector.py, geotag.py, sonar_pipeline.py, shadow_check.py).
MARINE_KIT_DIR = ROOT / os.environ.get("DEEPECHO_MARINE_KIT", "survey_hazard_map/models/marine")
DETECTOR_MODELS: dict[str, Path] = {
    "marine": ROOT / os.environ.get("DEEPECHO_MODEL_MARINE", "survey_hazard_map/models/marine/marine.pt"),
}
def _upload_default() -> bool:
    """On when the detector can actually run, off when it cannot.

    The flag has to gate the route rather than only the model loading, because
    the serve container ships no torch and would otherwise advertise /detect and
    answer it from the stub: a synthetic detection, correctly labelled, from a
    deployment that cannot detect anything.

    It used to default to off, which meant a full local checkout with both
    checkpoints on disk still had uploads disabled until you found the
    environment variable. So the default is now the honest answer to "can this
    process detect anything", and DEEPECHO_ENABLE_UPLOAD still overrides it in
    either direction.
    """
    try:
        import importlib.util
        for module in ("torch", "ultralytics"):
            if importlib.util.find_spec(module) is None:
                return False
    except Exception:
        return False
    return any(path.is_file() for path in DETECTOR_MODELS.values())


_upload_flag = os.environ.get("DEEPECHO_ENABLE_UPLOAD", "").strip().lower()
if _upload_flag in {"1", "true", "yes"}:
    ENABLE_UPLOAD = True
elif _upload_flag in {"0", "false", "no"}:
    ENABLE_UPLOAD = False
else:
    ENABLE_UPLOAD = _upload_default()

DETECTOR_CONFIDENCE = float(os.environ.get("DEEPECHO_DETECTOR_CONF", "0.25"))
DETECTOR_IMGSZ = int(os.environ.get("DEEPECHO_DETECTOR_IMGSZ", "640"))

# Two boxes overlapping by more than this are the same contact seen by both
# models. The higher-confidence one wins and the other is kept beside it as a
# second opinion rather than thrown away.
DETECTOR_MERGE_IOU = float(os.environ.get("DEEPECHO_DETECTOR_IOU", "0.5"))
DETECTOR_SENSOR = os.environ.get("DEEPECHO_DETECTOR_SENSOR", "side scan sonar, tile upload")
DETECTOR_PLATFORM = os.environ.get("DEEPECHO_DETECTOR_PLATFORM", "operator upload")

MAX_UPLOAD_BYTES = int(os.environ.get("DEEPECHO_MAX_UPLOAD_BYTES", str(16 * 1024 * 1024)))
ACCEPTED_UPLOAD_TYPES = ("image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp")

# What the stub returns until a trained model exists. Every field is marked, and
# `notes` says so in words, because a synthetic detection that reads like a real
# one is worse than no detection at all: the whole system downstream treats a
# record as evidence.
STUB_DETECTIONS: list[dict] = [
    {
        "object_class": "unknown",
        "confidence": 0.31,
        "bbox": [412.0, 233.0, 190.0, 64.0],
        "visual_description": "regular cylindrical return, roughly 2 m long, "
                              "partially buried, hard acoustic shadow",
        "notes": "SYNTHETIC placeholder detection from the stub detector. "
                 "Not produced by a trained model and not evidence of anything.",
    },
]
