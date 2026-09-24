#!/usr/bin/env python3
"""Run the evaluation suite and print a number you can defend.

    python3 eval/run.py                     # in-process, no server needed
    python3 eval/run.py --url http://127.0.0.1:8000
    python3 eval/run.py --category refusal --verbose
    python3 eval/run.py --repeat 3          # same cases N times, for variance
    python3 eval/run.py --delay 30          # pace a free tier's tokens-per-minute

Cases can carry `mode` ("auto", "copilot", "reference") and `language` (an
answer language code), a `ghosttrace_context` (a GhostTrace handoff, sent as its own
field), `offline: true` (every provider is forced to fail, in-process only, to
hold the retrieval-only fallback to its contract) and `stream: true` (the
streaming path is consumed and its done frame checked).

Every check is mechanical. Nothing here asks a model to grade another model,
because a suite whose purpose is evidence cannot rest on the same machinery it
is testing.

Exit code is 1 if any case fails, so this can gate a commit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag_assistant" / "eval"))

import checks  # noqa: E402

CASES = Path(__file__).resolve().parent / "cases.jsonl"


def load_cases(path: Path, category: str | None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [c for c in cases if not category or c["category"] == category]


class ProviderUnavailable(RuntimeError):
    """No provider answered, so the engine fell back to retrieval only.

    For an ordinary case that is the provider's rate limit talking, not the
    assistant, and it is retried rather than scored.
    """


def _offline_providers():
    """Every provider raises the error a rate-limited free tier raises."""
    import contextlib

    from rag_assistant import rag
    def unavailable(*_args, **_kwargs):
        raise SystemExit("Forced offline for evaluation: rate limit reached (429).")

    @contextlib.contextmanager
    def patched():
        # Every seam a turn can reach: generation, the copilot's tool planner
        # and the query translator.
        saved = rag.generate, rag.generate_stream, rag.plan_tools, rag.quick_complete
        rag.generate = rag.generate_stream = rag.plan_tools = rag.quick_complete = unavailable
        try:
            yield
        finally:
            rag.generate, rag.generate_stream, rag.plan_tools, rag.quick_complete = saved

    return patched()


def _consume_stream(frames) -> dict:
    seen, done = [], None
    for frame in frames:
        seen.append(frame["type"])
        if frame["type"] == "done":
            done = {k: v for k, v in frame.items() if k != "type"}
    if done is None:
        raise RuntimeError(f"stream ended without a done frame: {seen}")
    done["_frames"] = seen
    return done


MODEL: str | None = None  # --model, when set; None means the provider's default


def ask_direct(case: dict, provider: str) -> dict:
    import contextlib

    from rag_assistant import chat

    kwargs = {"detection": case.get("record"), "provider": provider, "model": MODEL,
              "ghosttrace_context": case.get("ghosttrace_context"),
              "mode": case.get("mode"), "language": case.get("language"),
              # As the API routes do: a provider outage returns the labelled
              # passages, which run_case retries for any case not about it.
              "offline_fallback": True}
    guard = _offline_providers() if case.get("offline") else contextlib.nullcontext()
    with guard:
        if case.get("stream"):
            return _consume_stream(chat.answer_stream(case["message"], **kwargs))
        return chat.answer(case["message"], **kwargs)


def ask_http(case: dict, provider: str, url: str) -> dict:
    if case.get("offline") or case.get("stream"):
        raise RuntimeError("offline and stream cases run in-process only")
    body = {"message": case["message"], "provider": provider}
    if MODEL:
        body["model"] = MODEL
    if case.get("record"):
        body["detection_record"] = case["record"]
    if case.get("ghosttrace_context"):
        body["ghosttrace_context"] = case["ghosttrace_context"]
    for key in ("mode", "language"):
        if case.get(key):
            body[key] = case[key]
    request = urllib.request.Request(url.rstrip("/") + "/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def _ghosttrace_text(context: dict | None) -> str:
    """The context as the model saw it, plus the raw values, for the checks."""
    if not context:
        return ""
    try:
        from rag_assistant import chat
        rendered = chat.render_ghosttrace(chat.normalise_ghosttrace(context))
    except Exception:
        rendered = ""
    return rendered + "\n" + json.dumps(context)


def run_case(case: dict, provider: str, url: str | None, attempts: int = 8,
             max_wait: float = 60.0) -> dict:
    # Free tiers rate-limit, and a suite that reports a limit as a failure is
    # measuring the provider's billing plan rather than the assistant.
    last = ""
    for attempt in range(attempts):
        try:
            result = ask_http(case, provider, url) if url else ask_direct(case, provider)
            if result.get("generated_by") in ("retrieval_only", "data_only") and not case.get("offline"):
                raise ProviderUnavailable("; ".join(result.get("provider_errors") or [])
                                          or "rate limit: no provider answered")
            break
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            transient = isinstance(exc, ProviderUnavailable) or any(
                marker in last.lower() for marker in ("rate limit", "429", "503", "unavailable"))
            if not transient:
                return {"case": case, "error": last, "results": []}
            # Free-tier windows are per minute, so back off into the next one.
            time.sleep(min(10 * (attempt + 1), max_wait) + random.random() * 3)
    else:
        return {"case": case, "error": last, "results": []}

    # The checks need to know what the model was allowed to see, so that a
    # number quoted from the operator's own message is not counted as invented.
    result["_record"] = case.get("record") or {}
    result["_message"] = case["message"]
    result["_ghosttrace"] = _ghosttrace_text(case.get("ghosttrace_context"))

    outcomes = []
    for name in checks.ALWAYS:
        ok, detail = checks.CHECKS[name](result, None)
        outcomes.append({"check": name, "ok": ok, "detail": detail})
    for name, expected in case["expect"].items():
        if name in checks.ALWAYS:
            continue
        ok, detail = checks.CHECKS[name](result, expected)
        outcomes.append({"check": name, "ok": ok, "detail": detail})

    return {"case": case, "error": None, "results": outcomes, "answer": result.get("answer", ""),
            "provider": result.get("provider", ""), "model": result.get("model", ""),
            "generated_by": result.get("generated_by", "model"),
            "unsourced_numbers": result.get("unsourced_numbers", []),
            "frames": result.get("_frames")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="run against a live backend instead of in-process")
    parser.add_argument("--provider", default="groq", help="LLM backend (default groq)")
    parser.add_argument("--model", help="override the provider's default model, e.g. gemini-2.5-flash")
    parser.add_argument("--category", help="run only one category")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the suite N times; generation is not deterministic")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel requests; a free tier will not take more than 1")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds between requests, to stay inside a free tier's window")
    parser.add_argument("--attempts", type=int, default=8,
                        help="tries per case while providers are rate-limited")
    parser.add_argument("--max-wait", type=float, default=60.0,
                        help="longest back-off between tries, in seconds")
    parser.add_argument("--only", help="comma-separated case ids to run")
    parser.add_argument("--report", default=str(Path(__file__).resolve().parent / "last-run.json"),
                        help="where the full result is written")
    parser.add_argument("--verbose", action="store_true", help="print failing answers")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args()
    global MODEL
    MODEL = args.model

    cases = load_cases(CASES, args.category)
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 1

    runs = cases * args.repeat

    def one(case: dict) -> dict:
        started = time.time()
        outcome = run_case(case, args.provider, args.url, args.attempts, args.max_wait)
        outcome["seconds"] = round(time.time() - started, 1)
        status = ("ERROR" if outcome["error"]
                  else "pass" if all(r["ok"] for r in outcome["results"]) else "FAIL")
        print(f"  {case['id']:4} {status:5} {outcome.get('provider') or '-':7} "
              f"{outcome['seconds']:>6}s", file=sys.stderr, flush=True)
        if args.delay:
            time.sleep(args.delay)
        return outcome

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            outcomes = list(pool.map(one, runs))
    else:
        outcomes = [one(case) for case in runs]

    # Always written, whole. A summary that reaches the terminal through `tail`
    # loses exactly the detail you need, and a run that costs this much of a
    # rate limit should not have to be repeated to find out what failed.
    Path(args.report).write_text(json.dumps(outcomes, indent=2, default=str))

    if args.as_json:
        print(json.dumps(outcomes, indent=2, default=str))

    passed = [o for o in outcomes if not o["error"] and all(r["ok"] for r in o["results"])]
    failed = [o for o in outcomes if o not in passed]

    by_category: dict[str, Counter] = defaultdict(Counter)
    for outcome in outcomes:
        by_category[outcome["case"]["category"]]["total"] += 1
        if outcome in passed:
            by_category[outcome["case"]["category"]]["pass"] += 1

    check_failures: Counter = Counter()
    for outcome in failed:
        for result in outcome["results"]:
            if not result["ok"]:
                check_failures[result["check"]] += 1

    if not args.as_json:
        print(f"{'category':12} {'pass':>8}")
        print(f"{'-' * 12} {'-' * 8}")
        for category in sorted(by_category):
            counts = by_category[category]
            print(f"{category:12} {counts['pass']:>4}/{counts['total']:<4}")
        print(f"{'-' * 12} {'-' * 8}")
        errored = [o for o in outcomes if o["error"]]
        print(f"{'-' * 12} {'-' * 8}")
        print(f"{'TOTAL':12} {len(passed):>4}/{len(outcomes):<4}")
        providers = Counter(o.get("provider") or o.get("generated_by") or "-"
                            for o in outcomes if not o["error"])
        print("answered by: " + ", ".join(f"{k} {v}" for k, v in providers.most_common()))
        if errored:
            # A provider limit is not an assistant failure. Reported apart from
            # the checks so a throttled run is never mistaken for a bad score.
            print(f"{'not run':12} {len(errored):>4}      (provider errors, see below)")
        print(f"\nfull report: {args.report}")

        if failed:
            print("\nFAILURES")
            for outcome in failed:
                case = outcome["case"]
                if outcome["error"]:
                    print(f"  {case['id']:4} {case['category']:10} ERROR {outcome['error']}")
                    continue
                bad = [r for r in outcome["results"] if not r["ok"]]
                print(f"  {case['id']:4} {case['category']:10} {case['message'][:52]}")
                for result in bad:
                    print(f"       {result['check']}: {result['detail']}")
                if args.verbose:
                    print("       ---")
                    for line in outcome["answer"].splitlines()[:12]:
                        print(f"       | {line}")
            print("\nby check: " + ", ".join(f"{k} {v}" for k, v in check_failures.most_common()))

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
