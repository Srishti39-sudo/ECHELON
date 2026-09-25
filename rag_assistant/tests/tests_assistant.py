#!/usr/bin/env python3
"""Tests for the grounded assistant's GhostTrace handoff, offline fallback and
numbers guard.

    .venv/bin/python tests_assistant.py

Plain functions and asserts, with an exit code so it can gate a commit. No test
here reaches a network: every provider is replaced before it can be called,
and retrieval runs over the local index (`rag.py index` builds it).
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag_assistant" / "eval"))

# The failover order must not depend on the developer's environment.
os.environ.setdefault("DEEPECHO_FAILOVER", "1")

from rag_assistant import rag  # noqa: E402
from rag_assistant import chat
from backend import config  # noqa: E402
from backend.schemas import ChatRequest, ChatResponse, GhostTraceContext  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ghosttrace_example.json"

QUESTION = "Why is this net ranked where it is, and who do the sources say should be told?"

# A complete context in the agreed shape. Fixed here rather than read from a
# survey on disk, so the expected figures cannot drift under the test.
CONTEXT: dict = {
    "kind": "ghosttrace_target",
    "survey_id": "gt-test-s2",
    "survey_title": "GhostTrace test survey",
    "synthetic": True,
    "detection_id": "S2_D1",
    "object_class": "net",
    "latitude": 9.11936621,
    "longitude": 79.05214306,
    "confidence_pct": 88.0,
    "priority": {
        "score": 0.6253, "tier": "urgent", "rank": 1,
        "formula": "score = confidence.value * sum(weight * value for every other term)",
        "terms": {
            "activity": {"value": 0.8889, "weight": 0.25, "contribution": 0.222225},
            "habitat": {"value": 0.7, "weight": 0.2, "contribution": 0.14},
            "drift_impact": {"value": 0.35, "weight": 0.15, "contribution": 0.0525},
            "people_risk": {"value": 1.0, "weight": 0.15, "contribution": 0.15},
            "size": {"value": 0.6171, "weight": 0.1, "contribution": 0.06171},
            "change": {"value": 0.4, "weight": 0.1, "contribution": 0.04},
            "recoverability": {"value": 0.8833, "weight": 0.05, "contribution": 0.044165},
            "confidence": {"value": 0.88, "weight": None, "contribution": None},
        },
    },
    "activity": {"level": "high", "score": 0.8889, "enrichment_ratio": 8.0,
                 "echo_clusters_near": 8, "background_clusters_per_window": 0.5,
                 "limitations": "Echoes may be fish, bubbles or sediment."},
    "habitat_nearest": [{"name": "Gulf of Mannar Marine National Park", "kind": "protected_area",
                         "distance_m": 6391.1, "source": "OpenStreetMap"}],
    "drift": {"mode": "seabed", "top_impact": None, "stranding_probability": 0.0},
    "refloat_scenario": {"top_impact": {"name": "Gulf of Mannar Marine National Park",
                                        "kind": "protected_area", "probability": 1.0,
                                        "first_arrival_hours": 27.0}},
    "people": {"propeller_hazard_level": "low",
               "diver_recommended_method": "diver recovery (trained team, cutting tools, surface support)",
               "seabed_depth_m": 17.9, "current_mps_at_depth": 0.045},
    "change": {"status": "first_survey", "moved_m": None},
    "authorities": [
        {"name": "State Fisheries Departments", "role": "fisheries is a state subject",
         "situation": "fisheries"},
        {"name": "Wildlife Warden, Ramanathapuram", "role": "manager of the protected area",
         "situation": "protected_habitat"},
    ],
    "caveats": ["SYNTHETIC OR DEMO INPUTS: nothing in this file is evidence of a real object."],
}


# ---------------------------------------------------------------------------
# Providers that never reach a network
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def providers(generate=None, stream=None):
    """Replace rag.generate / rag.generate_stream for the duration."""
    saved = rag.generate, rag.generate_stream
    if generate is not None:
        rag.generate = generate
    if stream is not None:
        rag.generate_stream = stream
    try:
        yield
    finally:
        rag.generate, rag.generate_stream = saved


def failing(message: str):
    def fail(*_args, **_kwargs):
        raise SystemExit(message)
    return fail


def raw_network_failure(*_args, **_kwargs):
    raise ConnectionError("[Errno 8] nodename nor servname provided, or not known")


def offline():
    """Both providers down: one rate-limited, one without a key."""
    calls = []

    def fail(_filled, _hits, provider="", _model=""):
        calls.append(provider)
        if provider == "gemini":
            raise SystemExit("Gemini free-tier rate limit reached. Wait and retry.")
        raise SystemExit("Could not create the Groq client: api_key must be set\n"
                         "Set GROQ_API_KEY. key=gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    return providers(fail, fail), calls


# ---------------------------------------------------------------------------
# 1. Context schema
# ---------------------------------------------------------------------------

def test_schema_accepts_full_context():
    ctx = GhostTraceContext.model_validate(CONTEXT)
    dumped = ctx.to_engine_context()
    assert dumped["kind"] == "ghosttrace_target"
    assert dumped["priority"]["terms"]["activity"]["contribution"] == 0.222225
    assert dumped["priority"]["terms"]["confidence"]["weight"] is None
    # An int count stays an int; a float stays the float it was.
    assert dumped["activity"]["echo_clusters_near"] == 8
    assert isinstance(dumped["activity"]["echo_clusters_near"], int)
    request = ChatRequest.model_validate({"message": QUESTION, "ghosttrace_context": CONTEXT})
    assert request.ghosttrace_context is not None
    assert request.detection_record is None


def test_schema_nulls_everywhere():
    nulls = {key: None for key in CONTEXT}
    nulls["kind"] = "ghosttrace_target"
    ctx = GhostTraceContext.model_validate(nulls).to_engine_context()
    assert ctx["priority"] is None and ctx["authorities"] is None
    partial = copy.deepcopy(CONTEXT)
    partial["priority"]["terms"]["habitat"] = None
    partial["drift"] = {"mode": None, "top_impact": None, "stranding_probability": None}
    partial["habitat_nearest"] = [{"name": None, "kind": None, "distance_m": None, "source": None}]
    GhostTraceContext.model_validate(partial)


def test_schema_source_may_be_object():
    ctx = copy.deepcopy(CONTEXT)
    ctx["habitat_nearest"][0]["source"] = {"name": "Synthetic reef layer", "url": None}
    dumped = GhostTraceContext.model_validate(ctx).to_engine_context()
    assert dumped["habitat_nearest"][0]["source"]["name"] == "Synthetic reef layer"


def test_schema_rejects_wrong_kind_and_bad_confidence():
    for bad in ({**CONTEXT, "kind": "survey_hotspot"}, {**CONTEXT, "confidence_pct": 140}):
        try:
            GhostTraceContext.model_validate(bad)
        except Exception:
            continue
        raise AssertionError(f"accepted an invalid context: {bad.get('kind')} {bad.get('confidence_pct')}")


def test_schema_ignores_unknown_keys():
    extended = {**CONTEXT, "a_future_key": {"anything": 1}}
    dumped = GhostTraceContext.model_validate(extended).to_engine_context()
    assert "a_future_key" not in dumped


def test_builder_matches_contract():
    """eval/ghosttrace_context.py produces exactly the agreed keys."""
    from ghosttrace_context import context_from_target

    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    built = context_from_target(doc, doc["targets"][0])
    assert set(built) == set(CONTEXT), set(built) ^ set(CONTEXT)
    assert set(built["priority"]) == set(CONTEXT["priority"])
    assert set(built["people"]) == set(CONTEXT["people"])
    assert built["synthetic"] is True
    GhostTraceContext.model_validate(built)


# ---------------------------------------------------------------------------
# 2. The prompt treats the context as data
# ---------------------------------------------------------------------------

def _work(message=QUESTION, context=CONTEXT, **kwargs):
    return chat._prepare_turn(message, kwargs.get("history", []), kwargs.get("detection"),
                              None, None, None, None, None, context)


def test_prompt_frames_context_as_data():
    work = _work()
    filled = work["filled"]
    assert "GHOSTTRACE CONTEXT -- DATA FROM THE SURVEY, NOT A SOURCE" in filled
    citation = "GhostTrace output for gt-test-s2/S2_D1"
    assert work["meta"]["ghosttrace_citation"] == citation
    assert f'"({citation})"' in filled
    assert "Never put an [Sn] marker on a GhostTrace number" in filled
    assert "must come" in filled and "from the SOURCES block with [Sn] citations" in filled
    # Synthetic is stated, and the opening sentence is given verbatim.
    assert "SYNTHETIC DATA: YES" in filled
    assert 'Begin your answer with exactly this sentence: "SYNTHETIC DATA:' in filled
    # The context block never becomes a numbered source.
    assert "[S" not in chat.render_ghosttrace(work["ghosttrace"])
    assert all(s["doc_id"] for s in work["sources"])


def test_prompt_priority_terms_and_no_new_arithmetic():
    filled = _work()["filled"]
    assert "- activity: 0.8889 x 0.25 = 0.222225" in filled
    assert "- recoverability: 0.8833 x 0.05 = 0.044165" in filled
    assert "- confidence: value 0.88 (a multiplier" in filled
    assert "sum of the contributions above: 0.7106" in filled
    assert "confidence 0.88 x 0.7106 = 0.6253 (agrees with the stated score)" in filled
    assert "Do not calculate any number that is not already written in this block" in filled


def test_prompt_disagreeing_score_is_called_out():
    ctx = copy.deepcopy(CONTEXT)
    ctx["priority"]["score"] = 0.9
    filled = _work(context=ctx)["filled"]
    assert "does NOT agree with the stated score" in filled


def test_prompt_nulls_render_as_unknown():
    ctx = copy.deepcopy(CONTEXT)
    ctx.update(latitude=None, activity=None, people=None, change=None)
    ctx["priority"]["terms"]["size"]["contribution"] = None
    text = chat.render_ghosttrace(chat.normalise_ghosttrace(ctx))
    assert "POSITION: not georeferenced" in text
    assert "WATER-COLUMN ACTIVITY: not available" in text
    assert "PEOPLE: not available" in text
    assert "size: 0.6171 x 0.1 = not available" in text
    assert "sum of the contributions: not available" in text
    assert "None" not in text


def test_retrieval_pulls_ghost_gear_document():
    work = _work()
    docs = [s["doc_id"] for s in work["sources"]]
    assert "ghost-gear-reporting-india" in docs[:3], docs
    sections = {s["section"] for s in work["sources"] if s["doc_id"] == "ghost-gear-reporting-india"}
    assert any("Gulf of Mannar" in (sec or "") for sec in sections), sections


def test_context_derives_class_but_not_severity_from_priority():
    meta = _work()["meta"]
    assert meta["object_class"] == "ghost net"          # net -> corpus vocabulary
    assert meta["confidence"] == 0.88
    assert meta["is_anomaly"] is False and meta["coverage_gap"] is False
    # Severity is the class table's, never GhostTrace's tier turned into one.
    assert meta["severity"] == config.CLASS_SEVERITY["ghost net"]


def test_detection_record_wins_over_context_class():
    work = _work(detection={"object_class": "shipwreck", "confidence": 0.76})
    assert work["meta"]["object_class"] == "shipwreck"
    assert "GHOSTTRACE CONTEXT" in work["filled"]


def test_non_synthetic_has_no_synthetic_opening():
    ctx = {**CONTEXT, "synthetic": False}
    filled = _work(context=ctx)["filled"]
    assert "SYNTHETIC DATA: no" in filled
    assert "Begin your answer with exactly this sentence" not in filled


def test_report_intent_still_carries_context():
    work = _work(message="write me an incident report for this net")
    assert work["meta"]["intent"] == "report"
    assert "GHOSTTRACE CONTEXT" in work["filled"]
    assert "Generate an incident report" in work["filled"]


def test_prompt_forbids_computed_figures_and_echoed_identity():
    assert "Never calculate a new figure from the sources" in rag.SYSTEM
    work = chat._prepare_turn("what is the percentage chance this is a mine?", [],
                              {"object_class": "unknown", "confidence": 0.31,
                               "visual_description": "cylindrical, 2 m"},
                              None, None, None, None, None)
    assert "Do not repeat, quote or paraphrase the operator's wording" in work["filled"]


def test_invalid_context_is_an_engine_error():
    try:
        chat.answer(QUESTION, ghosttrace_context={"kind": "nope"})
    except chat.EngineError as exc:
        assert "Invalid GhostTrace context" in str(exc)
        return
    raise AssertionError("an invalid context was accepted")


# ---------------------------------------------------------------------------
# 3. A model answer through the full seam
# ---------------------------------------------------------------------------

def test_model_answer_uses_context_and_flags_invented_numbers():
    seen = {}

    def fake(filled, hits, provider="", model=""):
        seen["filled"] = filled
        n = next(i for i, (c, _) in enumerate(hits, 1)
                 if c.meta["doc_id"] == "ghost-gear-reporting-india")
        return ("SYNTHETIC DATA: this GhostTrace target comes from synthetic demonstration data. "
                "Activity 0.8889 x 0.25 = 0.2222 and the score is 0.6253 "
                "(GhostTrace output for gt-test-s2/S2_D1). The habitat is 6 391 m away, "
                "or 88% confident. The net will drift 12.5 km in 37 hours. "
                f"Fisheries is a state subject [S{n}].")

    with providers(generate=fake):
        result = chat.answer(QUESTION, ghosttrace_context=CONTEXT, provider="groq")
    ChatResponse(**result)
    assert result["generated_by"] == "model"
    assert result["grounded"] is True
    assert result["ghosttrace_citation"] == "GhostTrace output for gt-test-s2/S2_D1"
    assert result["unsourced_numbers"] == ["12.5 km", "37 hours"], result["unsourced_numbers"]


# ---------------------------------------------------------------------------
# 4. Numbers guard
# ---------------------------------------------------------------------------

def test_guard_accepts_context_numbers():
    allowed = chat.render_ghosttrace(chat.normalise_ghosttrace(CONTEXT))
    answer = ("Score 0.6253, activity 0.8889 x 0.25 = 0.222225, rounded 0.2222, sum 0.7106, "
              "0.20 weight, depth 17.9 m, current 0.045 m/s, 6391.1 m, 6 391 m, 27.0 hours, "
              "27 hours, 88% confidence, 35% drift, rank 1.")
    assert chat.unsourced_numbers(answer, allowed) == []


def test_guard_flags_invented_numbers():
    allowed = chat.render_ghosttrace(chat.normalise_ghosttrace(CONTEXT))
    answer = ("The standoff is 300 m [S1]. Activity 0.8895 is not 0.8889, the score is 0.63, "
              "and 0.62531 is over-precise. Report within 48 hours; call 18002700.")
    flagged = chat.unsourced_numbers(answer, allowed)
    assert "300 m" in flagged
    assert "48 hours" in flagged
    assert "18002700" in flagged
    assert "0.62531" in flagged
    assert "0.8895" in flagged
    # 0.63 is a rounding of 0.6253, a figure that is in the context.
    assert "0.63" not in flagged


def test_guard_ignores_citations_and_list_numbering():
    answer = "1. Stop [S1].\n2) Log it [S2, S3].\n- 3. Report (S4)."
    assert chat.unsourced_numbers(answer, "") == []


def test_guard_reads_sources_and_word_numbers():
    allowed = "Notify within twenty-four hours. The 2024 plan covers 560 sq km."
    assert chat.unsourced_numbers("Within 24 hours, per the 2024 plan, 560 sq km.", allowed) == []
    assert chat.unsourced_numbers("Within 72 hours.", allowed) == ["72 hours"]


def test_guard_does_not_launder_assistant_history():
    """A number the assistant said last turn is not a source this turn."""
    history = [{"role": "user", "content": "the net is 40 m long"},
               {"role": "assistant", "content": "Keep 500 m away."}]
    work = _work(message="and how far away should I stay?", history=history)
    text = chat.allowed_text_for(work)
    assert "40 m" in text
    assert chat.unsourced_numbers("Keep 500 m away.", text) == ["500 m"]


# ---------------------------------------------------------------------------
# 5. Offline fallback
# ---------------------------------------------------------------------------

def test_offline_answer_shape():
    patch, calls = offline()
    with patch:
        result = chat.answer(QUESTION, ghosttrace_context=CONTEXT, provider=config.PROVIDER,
                             offline_fallback=True)
    ChatResponse(**result)                     # the response contract still holds
    assert calls == chat.provider_order(config.PROVIDER), calls  # every provider was tried, preferred first
    assert result["generated_by"] == "retrieval_only"
    assert result["provider"] == "" and result["model"] == ""
    assert result["refusal"] is False
    assert result["grounded"] is True           # every extract carries its [Sn]
    answer = result["answer"]
    assert answer.startswith(f"**{config.OFFLINE_TITLE}.**")
    assert "no generated answer" in answer
    # Key context facts, copied, with the GhostTrace attribution.
    assert "GhostTrace output for gt-test-s2/S2_D1" in answer
    assert "SYNTHETIC survey data" in answer
    assert "score 0.6253" in answer and "activity: 0.8889 x 0.25 = 0.222225" in answer
    # Top passages as short extracts with their citations.
    markers = [line for line in answer.splitlines() if line.startswith("[S")]
    assert len(markers) == min(config.OFFLINE_PASSAGES, len(result["sources"]))
    extracts = [line for line in answer.splitlines() if line.startswith("> ")]
    assert extracts and all(len(e) <= config.OFFLINE_EXTRACT_CHARS + 20 for e in extracts)
    assert "ghost-gear-reporting-india" in {s["doc_id"] for s in result["sources"][:config.OFFLINE_PASSAGES]}
    # Nothing in it is unsourced, and no credential leaked into the errors.
    assert result["unsourced_numbers"] == []
    assert len(result["provider_errors"]) == len(rag.PROVIDERS)
    assert "gsk_" not in json.dumps(result)
    assert result["provider_errors"][0].startswith(f"{config.PROVIDER}: ")


def test_offline_extracts_are_verbatim():
    patch, _ = offline()
    with patch:
        result = chat.answer("who do I report a suspected mine to?", offline_fallback=True)
    by_n = {s["n"]: s["snippet"] for s in result["sources"]}
    lines = result["answer"].splitlines()
    for i, line in enumerate(lines):
        if line.startswith("[S"):
            n = int(line[2:line.index("]")])
            quote = lines[i + 1][2:].replace("[...]", "").strip()
            for piece in [p.strip() for p in quote.split("  ") if p.strip()]:
                flat = " ".join(by_n[n].split())
                assert piece in flat, (n, piece[:80])


def test_offline_detection_facts():
    patch, _ = offline()
    with patch:
        result = chat.answer("what is this and what do I do?",
                             detection={"object_class": "shipwreck", "confidence": 0.76},
                             offline_fallback=True)
    assert result["generated_by"] == "retrieval_only"
    assert "**Detection on screen:** shipwreck, classifier confidence 0.76" in result["answer"]
    # Looked up, fallback or not: the catalog's hazard field when a catalog index
    # exists on this machine, else the class table. Never read out of the text.
    assert result["severity"] == chat.severity_for({"label": "shipwreck"}, False)
    assert result["severity"] in ("high", "medium"), result["severity"]


def test_offline_raw_network_error():
    """A transport exception that is not a SystemExit still degrades, not 500s."""
    with providers(raw_network_failure, raw_network_failure):
        result = chat.answer(QUESTION, ghosttrace_context=CONTEXT, offline_fallback=True)
    assert result["generated_by"] == "retrieval_only"
    assert all("nodename" in e for e in result["provider_errors"])


def test_direct_callers_get_the_error_not_the_fallback():
    """Off by default: a program that wanted a generated draft (GhostTrace's
    alert writer) must get the EngineError it handles, not quoted passages."""
    patch, _ = offline()
    with patch:
        try:
            chat.answer(QUESTION, ghosttrace_context=CONTEXT)
        except chat.EngineError:
            pass
        else:
            raise AssertionError("fallback returned without offline_fallback=True")
        frames = list(chat.answer_stream("who do I report a suspected mine to?"))
    assert [f["type"] for f in frames] == ["meta", "sources", "error"], frames


def test_failover_still_prefers_a_working_provider():
    def half(filled, hits, provider="", model=""):
        if provider == "gemini":
            raise SystemExit("Could not create the Gemini client: missing key")
        return "Report to the responsible authority [S1]."

    with providers(generate=half):
        result = chat.answer("who do I report a suspected mine to?", provider="gemini")
    assert result["generated_by"] == "model" and result["provider"] == "groq"


def test_bad_request_is_not_hidden_by_fallback():
    with providers(generate=failing("Gemini rejected the request: 400 invalid model name")):
        try:
            chat.answer("who do I report a suspected mine to?", provider="gemini")
        except chat.EngineError:
            return
    raise AssertionError("a non-transient provider error was turned into an offline answer")


def test_no_sources_is_not_offline():
    saved = chat.get_retriever().search
    try:
        chat.get_retriever().search = lambda *a, **k: []
        result = chat.answer("anything at all")
    finally:
        chat.get_retriever().search = saved
    assert result["generated_by"] == "none" and result["refusal"] is True


# ---------------------------------------------------------------------------
# 6. Streaming fallback
# ---------------------------------------------------------------------------

def test_stream_offline_fallback():
    patch, calls = offline()
    with patch:
        frames = list(chat.answer_stream(QUESTION, ghosttrace_context=CONTEXT, offline_fallback=True))
    types = [f["type"] for f in frames]
    assert types == ["meta", "sources", "delta", "done"], types
    assert calls == chat.provider_order(config.PROVIDER), calls
    done = {k: v for k, v in frames[-1].items() if k != "type"}
    ChatResponse(**done)
    assert done["generated_by"] == "retrieval_only"
    assert frames[2]["text"] == done["answer"]
    assert frames[0]["ghosttrace_citation"] == "GhostTrace output for gt-test-s2/S2_D1"


def test_stream_generator_error_mid_iteration_before_first_word():
    """A provider whose stream raises on first iteration (Gemini does) falls through."""
    def lazy_fail(*_args, **_kwargs):
        def gen():
            raise SystemExit("Gemini free-tier rate limit reached. Wait and retry.")
            yield  # pragma: no cover
        return gen()

    with providers(stream=lazy_fail):
        frames = list(chat.answer_stream("who do I report a suspected mine to?", offline_fallback=True))
    assert [f["type"] for f in frames] == ["meta", "sources", "delta", "done"]
    assert frames[-1]["generated_by"] == "retrieval_only"


def test_stream_mid_answer_failure_keeps_partial_text():
    def partial(*_args, **_kwargs):
        def gen():
            yield "Do not approach [S1]. "
            raise SystemExit("Groq free-tier rate limit reached. Wait and retry.")
        return gen()

    with providers(stream=partial):
        frames = list(chat.answer_stream("who do I report a suspected mine to?"))
    types = [f["type"] for f in frames]
    assert types == ["meta", "sources", "delta", "error", "done"], types
    assert frames[-1]["generated_by"] == "model"
    assert frames[-1]["answer"] == "Do not approach [S1]. "


def test_stream_model_answer_carries_guard():
    def ok(*_args, **_kwargs):
        return iter(["Score 0.6253 (GhostTrace output for gt-test-s2/S2_D1). ",
                     "Keep 250 m away [S1]."])

    with providers(stream=ok):
        frames = list(chat.answer_stream(QUESTION, ghosttrace_context=CONTEXT))
    done = frames[-1]
    assert done["generated_by"] == "model"
    assert done["unsourced_numbers"] == ["250 m"]


def test_http_routes_offline():
    """/chat and /chat/stream degrade the same way, through the real app."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)  # no lifespan: nothing to warm, no detector loaded
    patch, _ = offline()
    with patch:
        whole = client.post("/chat", json={"message": QUESTION, "ghosttrace_context": CONTEXT})
        assert whole.status_code == 200, whole.text
        body = whole.json()
        assert body["generated_by"] == "retrieval_only"
        assert body["ghosttrace_citation"] == "GhostTrace output for gt-test-s2/S2_D1"

        streamed = client.post("/chat/stream", json={"message": QUESTION, "ghosttrace_context": CONTEXT})
        assert streamed.status_code == 200
        frames = [json.loads(line[6:]) for line in streamed.text.split("\n\n") if line.startswith("data: ")]
        assert [f["type"] for f in frames] == ["meta", "sources", "delta", "done"]
        assert frames[-1]["generated_by"] == "retrieval_only"

    bad = client.post("/chat", json={"message": QUESTION,
                                     "ghosttrace_context": {**CONTEXT, "kind": "other"}})
    assert bad.status_code == 422


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
