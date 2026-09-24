"""Request and response models.

This is the one place the loose dict the CLI passes around becomes a validated
object. Two things matter here and are easy to get wrong:

* `object_class` is the API's name for what rag.py calls `label`. The mapping
  happens on the way in, so the engine keeps seeing the vocabulary it was
  written for.
* Absent fields are dropped, never sent as null. rag.render_detection() writes
  every key of the record straight into the prompt, and report mode is under
  instruction to write "NOT PROVIDED" for anything missing. A key arriving as
  None would satisfy that instruction with the word "None" instead.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config  # noqa: F401  puts the repository root on sys.path

from rag_assistant import rag
Role = Literal["user", "assistant"]
Intent = Literal["question", "explain", "anomaly", "report"]
Severity = Literal["low", "medium", "high", "unknown"]
# An int stays an int, so a count of 8 is rendered as 8 and not as 8.0.
Number = int | float


class Turn(BaseModel):
    """One message of conversation history, as the client replays it."""

    role: Role
    content: str


class DetectionRecord(BaseModel):
    """What the detector saw. Every field optional; nothing is invented.

    Extra keys are allowed and passed through to the prompt untouched, so a
    detector that emits a field this schema has never heard of does not lose it.
    """

    model_config = ConfigDict(extra="allow")

    object_class: str | None = None
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: list[float] | None = None
    depth_m: float | None = None
    timestamp: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    sensor: str | None = None
    platform: str | None = None
    notes: str | None = None
    visual_description: str | None = None
    embedding: list[float] | None = None
    detector_model: str | None = None
    detector_class: str | None = None
    second_opinion: str | None = None
    # Set when the detector's class was below the confidence this system
    # requires for that class. The original call is kept, never erased.
    downgraded_from: str | None = None
    # Verification against the image (hazard_verify). confidence_pct is the
    # 0-100 figure an operator sees; confidence_pct_basis says whether it is
    # fused with image evidence or only the detector's own score. `suppressed`
    # marks a likely false positive, kept and flagged, never deleted.
    confidence_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    confidence_pct_basis: str | None = None
    suppressed: bool | None = None
    verification: dict[str, Any] | None = None
    dimensions: dict[str, Any] | None = None

    @field_validator("bbox")
    @classmethod
    def _four_numbers(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != 4:
            raise ValueError("bbox must be [x, y, w, h]")
        return v

    @property
    def resolved_class(self) -> str | None:
        return self.object_class or self.label

    def to_engine_record(self) -> dict[str, Any]:
        """The dict shape rag.py expects: `label`, no nulls, extras preserved."""
        record = self.model_dump(exclude_none=True)
        label = record.pop("object_class", None) or record.get("label")
        if label:
            record["label"] = label
        return record


class Source(BaseModel):
    """One retrieved chunk, rendered for the citation panel.

    `n` is the number the model cited as [Sn]. It is assigned by enumerating the
    exact hit list the prompt was built from, so the marker in the answer text
    always resolves to the right entry here.
    """

    n: int
    id: str
    title: str
    section: str | None = None
    snippet: str
    authority: str | None = None
    status: str | None = None
    doc_id: str | None = None
    path: str | None = None
    score: float
    pdf_url: str | None = None


class Match(BaseModel):
    """A nearest known object. A ranking hint, never an identification."""

    rank: int
    id: str
    name: str
    object_class: str
    hazard: str
    similarity: float
    confirms: str
    rules_out: str
    source: str
    status: str


class _GhostTraceModel(BaseModel):
    """Every GhostTrace handoff field is optional and nullable.

    Unknown is null in the GhostTrace contract, never a plausible default, so a
    null here is carried through as "not available" and never filled in. Keys
    this schema has not heard of are ignored rather than rejected: the
    GhostTrace format is additive, and a newer producer must not break the chat.
    """

    model_config = ConfigDict(extra="ignore")


class GhostTraceTerm(_GhostTraceModel):
    value: Number | None = None
    weight: Number | None = None
    contribution: Number | None = None


class GhostTracePriority(_GhostTraceModel):
    score: Number | None = None
    tier: str | None = None
    rank: int | None = None
    formula: str | None = None
    terms: dict[str, GhostTraceTerm | None] | None = None


class GhostTraceActivity(_GhostTraceModel):
    level: str | None = None
    score: Number | None = None
    enrichment_ratio: Number | None = None
    echo_clusters_near: Number | None = None
    background_clusters_per_window: Number | None = None
    limitations: str | None = None


class GhostTraceHabitat(_GhostTraceModel):
    name: str | None = None
    kind: str | None = None
    distance_m: Number | None = None
    # A layer's source is a plain name in some producers and a
    # {name, url, licence, ...} object in others. Both are accepted.
    source: str | dict[str, Any] | None = None


class GhostTraceImpact(_GhostTraceModel):
    name: str | None = None
    kind: str | None = None
    probability: Number | None = None
    first_arrival_hours: Number | None = None


class GhostTraceDrift(_GhostTraceModel):
    mode: str | None = None
    top_impact: GhostTraceImpact | None = None
    stranding_probability: Number | None = None


class GhostTraceRefloat(_GhostTraceModel):
    top_impact: GhostTraceImpact | None = None


class GhostTracePeople(_GhostTraceModel):
    propeller_hazard_level: str | None = None
    diver_recommended_method: str | None = None
    seabed_depth_m: Number | None = None
    current_mps_at_depth: Number | None = None


class GhostTraceChange(_GhostTraceModel):
    status: str | None = None
    moved_m: Number | None = None


class GhostTraceAuthority(_GhostTraceModel):
    name: str | None = None
    role: str | None = None
    situation: str | None = None


class GhostTraceContext(_GhostTraceModel):
    """One GhostTrace target, handed to the assistant from the rescue queue.

    It is survey DATA. The assistant quotes its numbers with a GhostTrace
    attribution and never as a corpus source, and it still takes every
    authority and procedure statement from the retrieved passages.
    """

    kind: Literal["ghosttrace_target"] = "ghosttrace_target"
    survey_id: str | None = None
    survey_title: str | None = None
    synthetic: bool | None = None
    detection_id: str | None = None
    object_class: str | None = None
    latitude: Number | None = None
    longitude: Number | None = None
    confidence_pct: Number | None = Field(default=None, ge=0.0, le=100.0)
    priority: GhostTracePriority | None = None
    activity: GhostTraceActivity | None = None
    habitat_nearest: list[GhostTraceHabitat] | None = None
    drift: GhostTraceDrift | None = None
    refloat_scenario: GhostTraceRefloat | None = None
    people: GhostTracePeople | None = None
    change: GhostTraceChange | None = None
    authorities: list[GhostTraceAuthority] | None = None
    caveats: list[str] | None = None

    def to_engine_context(self) -> dict[str, Any]:
        """A plain dict with the nulls kept: null means unknown, and says so."""
        return self.model_dump()


class ToolCall(BaseModel):
    """One read-only data lookup the Mission Copilot made for this answer."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    # One line for the interface: "Looked up GhostTrace targets across 2 surveys".
    summary: str = ""
    record_count: int = 0
    # The [Dn] numbers of the records this call returned.
    citations: list[int] = Field(default_factory=list)
    error: str | None = None
    # "model" (native function calling), "json_plan", or "keywords" (the
    # deterministic plan used when no model planned).
    planned_by: str = "keywords"


class DataCitation(BaseModel):
    """A survey record cited as [Dn]. Data from the survey files, never a source."""

    n: int
    kind: str
    survey_id: str | None = None
    label: str
    source_file: str
    record_id: str | None = None
    synthetic: bool | None = None
    # Where the interface can open it: /ghosttrace/<id> or /map?survey=<id>.
    link: str | None = None
    # The compact record exactly as the answer prompt showed it.
    summary: dict[str, Any] = Field(default_factory=dict)


AssistantMode = Literal["auto", "copilot", "reference"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list)
    detection_record: DetectionRecord | None = None
    # A GhostTrace target from the rescue queue. Data, not a source.
    ghosttrace_context: GhostTraceContext | None = None
    # Escape hatches for testing. Omitted, everything comes from config.py.
    intent: Intent | None = None
    provider: str | None = None
    model: str | None = None
    # "auto" routes survey-data questions to the Mission Copilot (see
    # chat.route_mode); "copilot" and "reference" force a path.
    mode: AssistantMode | None = None
    # Answer language, one of config.LANGUAGES. Retrieval stays English.
    language: str | None = None

    @field_validator("language")
    @classmethod
    def _known_language(cls, v: str | None) -> str | None:
        if v is not None and v not in config.LANGUAGES:
            raise ValueError(f"unknown language; choose from {', '.join(config.LANGUAGES)}")
        return v

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, v: str | None) -> str | None:
        """Reject an unknown provider here rather than mid-stream.

        Once a streaming response has opened, the status line is already sent
        and a bad provider can only be reported as an error frame. Catching it
        at validation keeps it a plain 422.
        """
        if v is not None and v not in rag.PROVIDERS:
            raise ValueError(f"unknown provider; choose from {', '.join(sorted(rag.PROVIDERS))}")
        return v


class ChatResponse(BaseModel):
    answer: str
    intent: Intent
    object_class: str | None = None
    confidence: float | None = None
    is_anomaly: bool = False
    severity: Severity = "unknown"
    grounded: bool = False
    sources: list[Source] = Field(default_factory=list)

    # Additive, beyond the core contract.
    # `refusal` separates the two reasons an answer can be ungrounded: the
    # corpus genuinely has no authority for this (correct behaviour, worth
    # wording differently), versus retrieval simply missing.
    refusal: bool = False
    # True when the detector named a class the corpus has no document about.
    # Different from `refusal`: this is known before the model is called.
    coverage_gap: bool = False
    matches: list[Match] = Field(default_factory=list)
    query: str = ""
    provider: str = ""
    model: str = ""
    # "model" when a provider wrote the answer. "retrieval_only" when every
    # provider failed and the answer is the retrieved passages, quoted, with
    # nothing generated. "none" when retrieval found nothing and no model was
    # asked.
    # "data_only" is the Mission Copilot's equivalent of retrieval_only: the
    # survey records it looked up, tabled, plus source extracts, nothing generated.
    generated_by: Literal["model", "retrieval_only", "data_only", "none"] = "model"
    # Why no provider answered, one line per provider tried. Only set on a
    # retrieval-only answer.
    provider_errors: list[str] = Field(default_factory=list)
    # Figures in the answer found in none of: the retrieved passages, the
    # operator's message, the detection record, the GhostTrace context.
    unsourced_numbers: list[str] = Field(default_factory=list)
    # Set when the turn carried a GhostTrace context, so the interface can label
    # its numbers as GhostTrace's.
    ghosttrace_citation: str | None = None
    # "copilot" when survey data tools answered, "reference" for the corpus path.
    mode: Literal["copilot", "reference"] = "reference"
    # Why the turn took that path, e.g. 'auto: "all surveys" refers to survey data'.
    route_reason: str = ""
    language: str = "en"
    # The English rendering the corpus was searched with, when the question
    # was not English and a provider translated it.
    query_translated: str | None = None
    planned_by: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    data_citations: list[DataCitation] = Field(default_factory=list)


class DetectResponse(BaseModel):
    """What came back from a tile.

    `stub` is not decoration. While it is true the records are synthetic, and
    every surface that shows them is expected to say so.
    """

    stub: bool
    models: list[str]
    filename: str
    bytes: int
    detections: list[DetectionRecord] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    corpus_loaded: bool
    documents: int
    chunks: int
    embedder: str
    index: str
    catalog_entries: int
    catalog_space: str | None = None
    provider: str
    model: str
    detector: Literal["stub", "loaded", "disabled"]
    detector_models: list[str] = Field(default_factory=list)
    # "connected", or the reason it is not. Absent storage costs history, not
    # the assistant, so this is reported rather than fatal.
    storage: str = "unconfigured"
    upload_enabled: bool
    # The feature groups this process serves (config.FEATURES). Behind the
    # gateway each container reports its own, so "degraded" can be read against
    # what the process was meant to do.
    features: list[str] = Field(default_factory=list)
