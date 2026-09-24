# DeepEcho Assistant and Mission Copilot

The assistant answers operators from two kinds of evidence and never from a
model's memory:

- **Reference passages** from the curated corpus in `rag_assistant/kb/`, cited `[S1]`, `[S2]`.
- **Survey records** from the processed surveys in `data/surveys/`, cited
  `[D1]`, `[D2]`. Only Mission Copilot turns use these.

## Architecture

```
POST /chat or /chat/stream  {message, history, mode, language, ...}
  └─ chat._front_door        language, translation for retrieval, routing
       ├─ reference path      retrieve kb -> rag.generate -> grounding + numbers guard
       │                      (offline: retrieval_only, quoted passages)
       └─ copilot path        plan -> run tools -> retrieve kb -> generate
                              (offline: data_only, tabled records + extracts)
```

| File | Role |
|---|---|
| `backend/copilot_tools.py` | Read-only data tools, the `[D#]` ledger, DATA block rendering |
| `rag_assistant/chat.py` | Routing, planning, orchestration, grounding, numbers guard, offline answers |
| `rag.py` | Providers: `generate`, `generate_stream`, `plan_tools` (native tool calling), `quick_complete` |
| `backend/config.py` | Routing phrases, prompts, caps, languages, offline labels |
| `backend/schemas.py` | `ChatRequest.mode/language`; `ChatResponse.tool_calls/data_citations/mode/route_reason/language` |

### Copilot turn

1. **Plan.** Native function calling: Gemini `function_declarations` (mode ANY),
   Groq OpenAI-style `tools` on `openai/gpt-oss-120b` (`tool_choice=required`).
   The planner sees the survey catalogue and never answers. If a provider rejects
   tools outright it is asked for a JSON plan instead. If no provider plans, or a
   plan is empty, the deterministic **keyword plan** is used. Caps: 4 tool calls
   (`DEEPECHO_COPILOT_MAX_TOOLS`), 20 s planning (`DEEPECHO_COPILOT_PLAN_TIMEOUT`),
   24 records in the prompt. A provider that failed planning with a rate limit or
   missing key is skipped for generation in the same turn.
2. **Run tools.** Deterministic, local, never raise. Each call's first record is a
   `query_result` (filters, counts), so "nothing matched" is citable.
3. **Retrieve.** The (translated) question plus per-tool corpus terms.
4. **Generate.** SOURCES block + DATA block + rules A–G (`config.COPILOT_NOTE`).
5. **Check.** `grounded` = at least one marker and every `[S#]`/`[D#]` resolves.
   The numbers guard accepts figures present in the passages, the question, or
   the tool results.

Stream frames for a copilot turn: `meta`, `tools`, `sources`, `delta`…, `done`.

## Tools

| Tool | Reads | Returns |
|---|---|---|
| `list_surveys()` | export.json, ghosttrace.json | every survey: title, synthetic, counts, GhostTrace, comparison |
| `survey_summary(survey_id)` | export.json (+ coverage.json if present) | classes, tiers, hotspots, filtered, GhostTrace counts |
| `find_detections(survey_id, object_class, min_confidence, include_filtered, tier, near{latitude,longitude,radius_m}, limit)` | export.json | matching detections; class families (`mine` = mine/uxo/ordnance/mine-like…); for `mine`, also lists detections whose own recommended action mentions ordnance, without promoting their class |
| `top_hotspots(survey_id, n)` | export.json | hotspots by risk score |
| `ghosttrace_targets(survey_id, tier, limit)` | ghosttrace.json | targets; without a survey, a cross-survey ranking: latest observation of each net first (a target a later survey matched or no longer saw is superseded), then priority score |
| `change_report(survey_id)` | ghosttrace.json | `change_summary` plus new/moved/persistent/removed records; asking about the earlier survey finds the later one that compared with it |
| `filtered_detections(survey_id)` | export.json | suppressed detections with verification reasons and rule |

Every record carries a label (`ghosttrace.json of demo-ghosttrace-mannar-repeat,
target …`), the synthetic flag and a link (`/ghosttrace/<id>` or
`/map?survey=<id>`). `DEEPECHO_SURVEYS_DIR` overrides the data directory.

## Routing (`mode: "auto"`)

A turn goes to the copilot only when nothing is attached (no detection record,
no GhostTrace target) and one of these holds:

- a strong phrase: "all surveys", "hotspot", "ghosttrace", "filtered as",
  "what changed", "recover first", "summarise survey", …
- a medium phrase ("false positive", "detections", "surveys", "nets", "targets")
  together with a record word ("which", "were", "how many", "list", …);
- a survey named by id or id token ("mannar", "s7", "waterfall") together with a
  data noun ("survey", "how many", "targets", "changed", …).

"I found a ghost net near the Gulf of Mannar, who do I tell?" names Mannar but
stays a reference question. `mode: "copilot"` / `"reference"` force a path. The
reason is returned as `route_reason`. No existing evaluation case routes to the
copilot (tested).

## Grounding rules (copilot)

- Survey facts only from DATA, cited `[D#]`; procedures and authorities only
  from SOURCES, cited `[S#]`.
- Numbers copied exactly; no new arithmetic; counts come from `query_result`.
- Missing or null data is said plainly.
- Synthetic surveys are declared in the first sentence.
- A class is detector output: never "a mine"; scores and priorities are heuristics.
- An authority is named only where a SOURCES passage names it for that situation.

## Offline behaviour

With no provider (no key, rate limit, network): the copilot runs the keyword
plan and returns `generated_by: "data_only"`: a labelled answer with one table
per record kind, every row carrying its `[D#]`, plus short verbatim `[S#]`
extracts. No sentence is generated. The reference path keeps its
`retrieval_only` answer. Direct callers (`offline_fallback=False`) get the
`EngineError` instead.

## Languages

`language`: `en`, `hi` हिन्दी, `ta` தமிழ், `ml` മലയാളം, `or` ଓଡ଼ିଆ, `te` తెలుగు,
`bn` বাংলা, `kn` ಕನ್ನಡ, `mr` मराठी, `gu` ગુજરાતી.

- The prompt instructs the answer language, keeps `[S#]`/`[D#]`, ASCII numbers
  and ids, and English names in parentheses where helpful.
- Retrieval stays English: an Indic-script question is translated for search by
  `rag.quick_complete` (short timeout, failover). Without a provider it is
  searched as typed, and routing uses a small native-keyword table
  (`COPILOT_NATIVE_KEYWORDS`, survey-name aliases).
- The numbers guard normalises Indic digits (`४०` → `40`) and separates digits
  glued to letters before checking.
- Offline answers translate only their headings (`config.OFFLINE_LABELS`); data
  values and source extracts stay English. The UI picker notes this.

## Interface

`/assistant`: mode toggle (Auto / Mission Copilot / References only), copilot
example prompts, answer-language picker (remembered per browser), tool-call
chips ("Looked up GhostTrace targets across 2 surveys"), `D#` markers and a data
strip, and a citation panel with Sources and Data tabs. A data record shows its
fields as the answer saw them and opens `/ghosttrace/<id>` or the hazard map.

## Tests and evaluation

- `tests_copilot.py` (no network): every tool on the real surveys, cross-survey
  ranking, change report vs `ghosttrace.json`, mocked planner → tools → prompt,
  JSON-plan and keyword fallbacks, call cap, numbers guard with tool numbers,
  five offline data-only patterns, Hindi/Tamil labels and routing, Indic digits,
  language instruction, streaming and HTTP.
- `tests_assistant.py` keeps covering the reference path.
- `eval/cases.jsonl`: K1–K5 (copilot), M1–M3 (multilingual). New checks:
  `mode`, `tool_called`, `cites_data`, `data_cites_resolve` (always run), `script`.

## Limits

- Refusal detection (`refusal`) uses English patterns; a refusal written in
  another language is not flagged.
- The numbers guard reads digits, not number words in Indic languages.
- Offline routing of non-English questions depends on the small keyword table.
- "The Mannar survey" matches both Mannar surveys; both are looked up.
- Survey facts are only as good as the files: synthetic demo surveys are marked,
  and severity, risk and priority scores are configurable heuristics.
- Three model calls for a non-English copilot turn (translate, plan, answer), two
  for English; free tiers may rate-limit, in which case the data-only answer is
  returned.
