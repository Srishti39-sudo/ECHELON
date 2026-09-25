#!/usr/bin/env python3
"""Tests for the Mission Copilot: data tools, planning, grounding, languages.

    .venv/bin/python tests_copilot.py

No network. Tools run on the real data/surveys; every provider seam
(rag.generate, rag.generate_stream, rag.plan_tools, rag.quick_complete) is
replaced before it can be called.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPECHO_FAILOVER", "1")

from rag_assistant import rag  # noqa: E402
from rag_assistant import chat, copilot_tools as ct
from backend import config  # noqa: E402
from backend.schemas import ChatRequest, ChatResponse  # noqa: E402

SURVEYS = ROOT / "data" / "surveys"
MANNAR, REPEAT = "demo-ghosttrace-mannar", "demo-ghosttrace-mannar-repeat"


def gt(survey_id: str) -> dict:
    return json.loads((SURVEYS / survey_id / "ghosttrace.json").read_text())


def export(survey_id: str) -> dict:
    return json.loads((SURVEYS / survey_id / "export.json").read_text())


@contextlib.contextmanager
def seams(generate=None, stream=None, plan=None, quick=None):
    saved = rag.generate, rag.generate_stream, rag.plan_tools, rag.quick_complete

    def down(*_a, **_k):
        raise SystemExit("Forced offline for tests: rate limit reached (429).")

    rag.generate = generate or down
    rag.generate_stream = stream or down
    rag.plan_tools = plan or down
    rag.quick_complete = quick or down
    try:
        yield
    finally:
        rag.generate, rag.generate_stream, rag.plan_tools, rag.quick_complete = saved


def numbered(output: ct.ToolOutput) -> ct.ToolOutput:
    ct.Ledger(200).add(output)
    return output


# ---------------------------------------------------------------------------
# 1. Tools on the real surveys
# ---------------------------------------------------------------------------

def test_list_surveys():
    out = numbered(ct.run_tool("list_surveys"))
    ids = [r["data"]["survey_id"] for r in out.records if r["kind"] == "survey"]
    for sid in (MANNAR, REPEAT, "demo-synthetic", "s7-submarine", "waterfall-strip"):
        assert sid in ids, sid
    assert out.records[0]["kind"] == "query_result" and out.records[0]["n"] == 1
    s7 = next(r for r in out.records if r.get("record_id") == "s7-submarine")
    assert s7["synthetic"] is False and s7["data"]["ghosttrace"] is False


def test_survey_summary_matches_export():
    out = numbered(ct.run_tool("survey_summary", {"survey_id": "waterfall-strip"}))
    data = out.records[1]["data"]
    summary = export("waterfall-strip")["survey_summary"]
    assert data["filtered_as_false_positive"] == summary["suppressed_detections"] == 2
    assert data["class_distribution"] == summary["class_distribution"]
    assert out.records[1]["link"] == "/map?survey=waterfall-strip"
    bad = ct.run_tool("survey_summary", {"survey_id": "no-such-survey"})
    assert bad.error == "unknown survey id" and "known_surveys" in bad.records[0]["data"]
    # Path traversal is an unknown survey, not a file read.
    assert ct.run_tool("survey_summary", {"survey_id": "../kb"}).error == "unknown survey id"


def test_find_detections_mine_family_in_mannar():
    out = numbered(ct.run_tool("find_detections", {"survey_id": MANNAR, "object_class": "mine-like objects"}))
    query = out.records[0]["data"]
    assert query["matched"] == 0 and query["filters"]["class_family"] == "mine"
    related = query["other_detections_whose_recommended_action_mentions_ordnance"]
    assert {r["object_class"] for r in related} == {"cylinder"}
    assert out.records[0]["synthetic"] is True


def test_find_detections_filters():
    out = ct.run_tool("find_detections", {"survey_id": "demo-synthetic", "object_class": "mine"})
    classes = {r["data"]["object_class"] for r in out.records[1:]}
    assert classes <= {"mine", "uxo"} and "mine" in classes
    tier = ct.run_tool("find_detections", {"tier": "critical"})
    assert all(r["data"]["severity_tier"] == "critical" for r in tier.records[1:])
    hidden = ct.run_tool("find_detections", {"survey_id": "waterfall-strip"})
    shown = ct.run_tool("find_detections", {"survey_id": "waterfall-strip", "include_filtered": True})
    assert hidden.records[0]["data"]["filtered_false_positives_excluded"] == 2
    assert shown.records[0]["data"]["matched"] == hidden.records[0]["data"]["matched"] + 2
    near = ct.run_tool("find_detections", {"near": {"latitude": 9.11981242, "longitude": 79.05029841,
                                                    "radius_m": 5}})
    assert near.records[0]["data"]["matched"] >= 1
    assert all(r["data"]["distance_m_computed_by_tool"] <= 5 for r in near.records[1:])
    conf = ct.run_tool("find_detections", {"min_confidence": 0.85})
    assert all(r["data"]["detector_confidence"] >= 0.85 for r in conf.records[1:])


def test_top_hotspots():
    out = ct.run_tool("top_hotspots", {"survey_id": "demo-synthetic", "n": 2})
    rows = [r["data"] for r in out.records[1:]]
    assert len(rows) == 2 and rows[0]["hotspot_id"] == "H001" and rows[0]["dominant_class"] == "mine"
    allsurveys = ct.run_tool("top_hotspots", {"n": 3})
    scores = [r["data"]["risk_score"] for r in allsurveys.records[1:]]
    assert scores == sorted(scores, reverse=True)


def test_ghosttrace_cross_survey_ranking():
    out = numbered(ct.run_tool("ghosttrace_targets"))
    # Every survey with a GhostTrace output is listed; this test reasons about
    # the synthetic pair only, so other surveys (the USGS mosaic) are set aside.
    rows = [r["data"] for r in out.records[1:] if r["data"]["survey_id"] in (MANNAR, REPEAT)]
    assert len(rows) == 4, [r["survey_id"] for r in rows]
    current = [r for r in rows if r["latest_observation"]]
    # The moved net was matched by the repeat survey, and the other first-survey
    # net was not seen again, so only the repeat survey's two nets are current.
    assert {r["survey_id"] for r in current} == {REPEAT}
    assert current[0]["detection_id"] == "SYNTHETIC_mannar_line02_512_1536_d0"
    assert current[0]["cross_survey_rank"] == 1
    assert current[0]["priority_score"] >= current[1]["priority_score"]
    old = next(r for r in rows if r["detection_id"] == "SYNTHETIC_mannar_line01_512_2048_d0")
    assert old["superseded_by"]["status"] == "moved" and old["cross_survey_rank"] is None
    removed = next(r for r in rows if r["detection_id"] == "SYNTHETIC_mannar_line01_512_512_d0")
    assert "removed" in removed["superseded_by"]["status"]
    assert out.records[1]["link"] == f"/ghosttrace/{REPEAT}"
    one = ct.run_tool("ghosttrace_targets", {"survey_id": MANNAR})
    scores = [r["data"]["priority_score"] for r in one.records[1:]]
    assert scores == [t["priority"]["score"] for t in gt(MANNAR)["targets"]]


def test_change_report_matches_ghosttrace():
    expected = gt(REPEAT)["change_summary"]
    for asked in (REPEAT, MANNAR, None):
        out = ct.run_tool("change_report", {"survey_id": asked} if asked else {})
        summary = next(r["data"] for r in out.records if r["kind"] == "change_summary")
        for key in ("new", "moved", "persistent", "removed", "compared_with"):
            assert summary[key] == expected[key], (asked, key)
        kinds = [r["kind"] for r in out.records]
        assert kinds.count("removed") == expected["removed"]
        assert sum(1 for r in out.records if r["kind"] == "change" and r["data"]["status"] == "moved") == expected["moved"]
    assert "earlier survey" in ct.run_tool("change_report", {"survey_id": MANNAR}).records[0]["data"]["notes"][0]


def test_filtered_detections():
    out = ct.run_tool("filtered_detections")
    total = sum(export(s)["survey_summary"]["suppressed_detections"] for s in ct.survey_ids())
    assert out.records[0]["data"]["filtered_total"] == total == len(out.records) - 1
    row = next(r["data"] for r in out.records[1:] if r["survey_id"] == "waterfall-strip")
    assert "nadir_zone" in row["hard_reasons"] and row["reasons"]


def test_surveys_dir_env_override():
    saved = os.environ.get("DEEPECHO_SURVEYS_DIR")
    os.environ["DEEPECHO_SURVEYS_DIR"] = str(ROOT / "no-such-dir")
    try:
        assert ct.survey_ids() == []
        assert ct.run_tool("list_surveys").records[0]["data"]["survey_count"] == 0
    finally:
        if saved is None:
            os.environ.pop("DEEPECHO_SURVEYS_DIR")
        else:
            os.environ["DEEPECHO_SURVEYS_DIR"] = saved


def test_ledger_numbers_across_calls_and_caps():
    ledger = ct.Ledger(max_records=3)
    ledger.add(ct.run_tool("ghosttrace_targets"))
    ledger.add(ct.run_tool("filtered_detections"))
    assert [r["n"] for r in ledger.records] == [1, 2, 3, 4]  # the second query_result is always kept
    assert ledger.outputs[1].records[0]["kind"] == "query_result"


def test_bad_tool_and_arguments_never_raise():
    assert ct.run_tool("rm_rf").error == "unknown tool"
    out = ct.run_tool("find_detections", {"limit": "lots", "bogus": 1, "near": "here"})
    assert out.error is None


# ---------------------------------------------------------------------------
# 2. Routing and the keyword plan
# ---------------------------------------------------------------------------

def test_existing_eval_cases_never_route_to_copilot():
    for line in (ROOT / "rag_assistant" / "eval" / "cases.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case.get("mode") or case["category"] in ("copilot", "multilingual"):
            continue
        copilot, why = chat.route_mode(case["message"], case["message"], False, None)
        assert not copilot, (case["id"], why)
    for message in ("how do I tell a false positive from a rock on side scan?",
                    "I found a ghost net near the Gulf of Mannar Marine National Park. Who do I tell?",
                    "where is the Gulf of Mannar marine national park?"):
        assert not chat.route_mode(message, message, False, None)[0], message


def test_route_rules():
    q = "Which net should we recover first across all surveys, and who do we notify?"
    assert chat.route_mode(q, q, False, None)[0]
    assert not chat.route_mode(q, q, True, None)[0]  # attachment wins in auto
    assert chat.route_mode("hello", "hello", True, "copilot") == (True, "copilot: requested")
    assert chat.route_mode(q, q, False, "reference")[0] is False


PATTERNS = {
    "Which net should we recover first across all surveys, and who do we notify?": "ghosttrace_targets",
    "How many mine-like objects were found in the Mannar survey and where?": "find_detections",
    "What changed between the two Mannar surveys?": "change_report",
    "Summarise survey waterfall-strip for the Coast Guard": "survey_summary",
    "Which contacts were filtered as false positives and why?": "filtered_detections",
}


def test_keyword_plan_patterns():
    for question, tool in PATTERNS.items():
        assert tool in [name for name, _ in chat.keyword_plan(question)], question
    assert chat.keyword_plan("What changed between the two Mannar surveys?") == [
        ("change_report", {"survey_id": REPEAT})]


# ---------------------------------------------------------------------------
# 3. Offline: data-only answers
# ---------------------------------------------------------------------------

def test_offline_data_only_five_patterns():
    with seams():
        for question, tool in PATTERNS.items():
            result = chat.answer(question, offline_fallback=True)
            ChatResponse(**result)
            assert result["mode"] == "copilot", question
            assert result["generated_by"] == "data_only", question
            assert tool in [c["name"] for c in result["tool_calls"]], question
            assert all(c["planned_by"] == "keywords" for c in result["tool_calls"])
            assert result["grounded"] and result["unsourced_numbers"] == [], (question, result["unsourced_numbers"])
            assert "[D1]" in result["answer"] and "Offline" in result["answer"]
            numbers = {d["n"] for d in result["data_citations"]}
            _, cited = chat.cited_markers(result["answer"])
            assert cited and cited <= numbers
            assert result["provider_errors"], question
    with seams():
        first = chat.answer("Which net should we recover first across all surveys, and who do we notify?",
                            offline_fallback=True)
    assert "SYNTHETIC_mannar_line02_512_1536_d0" in first["answer"]
    assert "Synthetic demonstration data" in first["answer"]


def test_offline_without_fallback_raises():
    with seams():
        try:
            chat.answer("What changed between the two Mannar surveys?")
        except chat.EngineError:
            pass
        else:
            raise AssertionError("a direct caller must get the error")


def test_offline_hindi_labels_and_native_routing():
    with seams():
        result = chat.answer("मन्नार सर्वे में क्या बदला?", language="hi", offline_fallback=True)
    assert result["mode"] == "copilot" and result["generated_by"] == "data_only"
    assert [c["name"] for c in result["tool_calls"]] == ["change_report"]
    assert config.OFFLINE_LABELS["hi"]["data_title"] in result["answer"]
    assert config.OFFLINE_LABELS["hi"]["sources_heading"] in result["answer"]
    assert result["query_translated"] is None and result["language"] == "hi"


def test_offline_tamil_reference_labels():
    with seams():
        result = chat.answer("சந்தேகத்திற்குரிய கண்ணிவெடியை யாரிடம் தெரிவிக்க வேண்டும்?",
                             language="ta", offline_fallback=True, mode="reference")
    assert result["generated_by"] in ("retrieval_only", "none")
    if result["generated_by"] == "retrieval_only":
        assert config.OFFLINE_LABELS["ta"]["retrieval_title"] in result["answer"]


# ---------------------------------------------------------------------------
# 4. Orchestration with a mocked provider
# ---------------------------------------------------------------------------

def test_planner_tool_calls_reach_final_prompt():
    seen: dict = {}

    def plan(system, user, tools, provider="", model="", **kwargs):
        seen["tools"] = [t["name"] for t in tools]
        seen["system"] = system
        return [{"name": "ghosttrace_targets", "args": {}},
                {"name": "change_report", "args": {"survey_id": REPEAT}},
                {"name": "not_a_tool", "args": {}}]

    def generate(filled, hits, provider="", model=""):
        seen["filled"] = filled
        return ("SYNTHETIC DATA. The first net to recover is SYNTHETIC_mannar_line02_512_1536_d0, "
                "priority score 0.3162 [D2]; it moved 40.0 m since the earlier survey [D7]. "
                "Report lost gear to the fisheries authorities [S1].")

    with seams(generate=generate, plan=plan):
        result = chat.answer("Which net should we recover first across all surveys?", provider="groq")
    assert set(seen["tools"]) == set(ct.TOOLS)
    assert MANNAR in seen["system"]
    assert "[D1]" in seen["filled"] and "DATA\n====" in seen["filled"]
    assert "0.3162" in seen["filled"] and "MISSION COPILOT" in seen["filled"]
    assert [c["name"] for c in result["tool_calls"]] == ["ghosttrace_targets", "change_report"]
    assert all(c["planned_by"] == "model" for c in result["tool_calls"])
    assert result["generated_by"] == "model" and result["mode"] == "copilot"
    assert result["grounded"] is True
    assert result["unsourced_numbers"] == []  # 0.3162 and 40.0 m come from the tools
    assert len(result["data_citations"]) == sum(c["record_count"] for c in result["tool_calls"])
    ChatResponse(**result)


def test_numbers_guard_flags_numbers_not_in_tools():
    def plan(*_a, **_k):
        return [{"name": "change_report", "args": {"survey_id": REPEAT}}]

    def generate(*_a, **_k):
        return "The net moved 40.0 m [D3] and 55 m in total [D2]; score 0.9999 [D1]."

    with seams(generate=generate, plan=plan):
        result = chat.answer("What changed between the two Mannar surveys?")
    assert "55 m" in result["unsourced_numbers"] and "0.9999" in result["unsourced_numbers"]
    assert not any("40" in n for n in result["unsourced_numbers"])


def test_stray_data_marker_is_ungrounded():
    def plan(*_a, **_k):
        return [{"name": "list_surveys", "args": {}}]

    with seams(generate=lambda *a, **k: "There are 5 surveys [D1] and more [D99].", plan=plan):
        result = chat.answer("which surveys do we have?")
    assert result["grounded"] is False


def test_json_plan_when_provider_rejects_tools():
    def plan(*_a, **_k):
        raise SystemExit("Groq API error 400: tool_use_failed")

    def quick(system, user, provider="", model="", **kwargs):
        assert "Reply with JSON only" in user
        return '```json\n{"calls": [{"name": "filtered_detections", "args": {}}]}\n```'

    with seams(generate=lambda *a, **k: "Three were filtered [D1].", plan=plan, quick=quick):
        result = chat.answer("Which contacts were filtered as false positives and why?", provider="groq")
    assert [c["planned_by"] for c in result["tool_calls"]] == ["json_plan"]


def test_empty_model_plan_falls_back_to_keywords():
    with seams(generate=lambda *a, **k: "Changed [D1].", plan=lambda *a, **k: []):
        result = chat.answer("What changed between the two Mannar surveys?")
    assert [(c["name"], c["planned_by"]) for c in result["tool_calls"]] == [("change_report", "keywords")]


def test_tool_call_cap():
    many = [{"name": "survey_summary", "args": {"survey_id": sid}} for sid in ct.survey_ids()]
    with seams(generate=lambda *a, **k: "ok [D1]", plan=lambda *a, **k: many):
        result = chat.answer("x", mode="copilot")
    assert len(result["tool_calls"]) == config.COPILOT_MAX_TOOL_CALLS


def test_parse_json_plan():
    assert chat.parse_json_plan('noise {"calls":[{"name":"list_surveys","args":{}}]} tail') == [
        {"name": "list_surveys", "args": {}}]
    assert chat.parse_json_plan("no json here") == []


# ---------------------------------------------------------------------------
# 5. Languages
# ---------------------------------------------------------------------------

def test_indic_digit_normalisation():
    assert chat.normalise_digits("४०.५ मीटर") == "40.5 मीटर"
    assert chat.normalise_digits("௧௨ மீ") == "12 மீ"
    assert chat.normalise_digits("০.৩৫") == "0.35"
    allowed = "the net moved 40.0 m"
    assert chat.unsourced_numbers("जाल ४०.० m खिसका [D1]", allowed) == []
    # A bare integer below the guard's floor is not checked, as in English.
    assert chat.unsourced_numbers("जाल ७५ मीटर खिसका", allowed) == []
    assert chat.unsourced_numbers("जाल ७५.५ मीटर खिसका", allowed) == ["75.5"]
    assert chat.unsourced_numbers("जाल ७५ m खिसका", allowed) == ["75 m"]
    assert chat.unsourced_numbers("வலை 55.5மீட்டர் நகர்ந்தது", allowed) == ["55.5"]


def test_language_instruction_in_prompts():
    captured: dict = {}

    def generate(filled, hits, provider="", model=""):
        captured.setdefault("prompts", []).append(filled)
        return "उत्तर [S1] [D1]"

    def quick(system, user, *a, **k):
        assert system == config.TRANSLATE_SYSTEM
        return "What changed between the two Mannar surveys?"

    with seams(generate=generate, plan=lambda *a, **k: [{"name": "change_report", "args": {}}], quick=quick):
        copilot = chat.answer("मन्नार के दोनों सर्वे के बीच क्या बदला?", language="hi")
        reference = chat.answer("who do I report a suspected mine to?", language="ta")
    assert "Hindi (हिन्दी)" in captured["prompts"][0] and "[S1], [D1]" in captured["prompts"][0]
    assert "Tamil (தமிழ்)" in captured["prompts"][1]
    assert copilot["query_translated"] == "What changed between the two Mannar surveys?"
    assert copilot["mode"] == "copilot" and reference["mode"] == "reference"
    assert chat.language_instruction("en") == ""


def test_translation_failure_uses_original_text():
    text, errors = None, None
    with seams():
        text, errors = chat.translate_for_retrieval("सुरंग की सूचना किसे दें?")
    assert text is None and len(errors) == len(rag.PROVIDERS)  # one per provider tried
    assert chat.translate_for_retrieval("plain english") == (None, [])


def test_request_schema_language_and_mode():
    ChatRequest.model_validate({"message": "x", "mode": "copilot", "language": "ml"})
    for bad in ({"message": "x", "language": "xx"}, {"message": "x", "mode": "other"}):
        try:
            ChatRequest.model_validate(bad)
        except Exception:
            continue
        raise AssertionError(bad)


# ---------------------------------------------------------------------------
# 6. Streaming and HTTP
# ---------------------------------------------------------------------------

def test_stream_copilot_model():
    def stream(filled, hits, provider="", model=""):
        assert "[D1]" in filled
        yield "Moved 40.0 m "
        yield "[D3]."

    with seams(stream=stream, plan=lambda *a, **k: [{"name": "change_report", "args": {"survey_id": REPEAT}}]):
        frames = list(chat.answer_stream("What changed between the two Mannar surveys?"))
    types = [f["type"] for f in frames]
    assert types == ["meta", "tools", "sources", "delta", "delta", "done"], types
    assert frames[0]["mode"] == "copilot"
    assert frames[1]["tool_calls"][0]["name"] == "change_report" and frames[1]["data_citations"]
    done = frames[-1]
    assert done["grounded"] and done["unsourced_numbers"] == [] and done["generated_by"] == "model"


def test_stream_copilot_offline():
    with seams():
        frames = list(chat.answer_stream("Which contacts were filtered as false positives and why?",
                                         offline_fallback=True))
    assert [f["type"] for f in frames] == ["meta", "tools", "sources", "delta", "done"]
    assert frames[-1]["generated_by"] == "data_only" and frames[-1]["grounded"]


def test_http_copilot_routes():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    with seams():
        body = client.post("/chat", json={"message": "What changed between the two Mannar surveys?",
                                          "language": "ta"}).json()
        assert body["generated_by"] == "data_only" and body["mode"] == "copilot"
        assert body["data_citations"][0]["n"] == 1 and body["tool_calls"][0]["name"] == "change_report"
        assert config.OFFLINE_LABELS["ta"]["data_title"] in body["answer"]
        streamed = client.post("/chat/stream", json={"message": "list surveys", "mode": "copilot"})
        frames = [json.loads(chunk[6:]) for chunk in streamed.text.split("\n\n") if chunk.startswith("data: ")]
        assert [f["type"] for f in frames] == ["meta", "tools", "sources", "delta", "done"]
        assert frames[-1]["tool_calls"][0]["name"] == "list_surveys"
    assert client.post("/chat", json={"message": "x", "language": "fr"}).status_code == 422


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
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
