#!/usr/bin/env python3
"""The reranker's contract: advisory, never load-bearing.

    .venv/bin/python rag_assistant/tests/tests_rerank.py

No network. The scoring call is stubbed, so what is tested is the wrapper:
that a failure returns retrieval's order untouched, that a success reorders
and trims to k, that no passage the index did not return can appear, and that
chat.retrieve() is a plain search when reranking is off.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_assistant import rerank as rr  # noqa: E402


class _Chunk:
    def __init__(self, id_, text):
        self.id, self.text, self.meta = id_, text, {"doc_id": id_.split("#")[0]}


HITS = [(_Chunk(f"doc{i}#0", f"passage {i}"), 1.0 - i * 0.1) for i in range(6)]


def test_no_key_returns_retrieval_order_trimmed():
    os.environ.pop("NVIDIA_API_KEY", None)
    out = rr.NvidiaReranker().rerank("q", HITS, keep=4)
    assert [c.id for c, _ in out] == [c.id for c, _ in HITS[:4]]
    assert [s for _, s in out] == [s for _, s in HITS[:4]], "scores untouched on fallback"


def test_failure_mid_call_falls_back():
    r = rr.NvidiaReranker()
    r.scores = lambda q, p: (_ for _ in ()).throw(RuntimeError("HTTP 429"))
    out = r.rerank("q", HITS, keep=3)
    assert [c.id for c, _ in out] == [c.id for c, _ in HITS[:3]]


def test_success_reorders_and_trims():
    r = rr.NvidiaReranker()
    # the last candidate is the best, the first the worst
    r.scores = lambda q, p: [float(i) for i in range(len(p))]
    out = r.rerank("q", HITS, keep=3)
    assert [c.id for c, _ in out] == ["doc5#0", "doc4#0", "doc3#0"], [c.id for c, _ in out]
    assert [s for _, s in out] == [5.0, 4.0, 3.0], "returned score is the reranker logit"


def test_never_adds_a_passage():
    r = rr.NvidiaReranker()
    r.scores = lambda q, p: [0.0] * len(p)
    out = r.rerank("q", HITS[:2], keep=8)
    assert len(out) == 2


def test_empty_input():
    assert rr.NvidiaReranker().rerank("q", [], keep=8) == []


def test_chat_retrieve_is_plain_search_when_off():
    # The key stays: the index on disk may be NVIDIA-embedded, and encoding the
    # query then needs it. What is under test is that RERANK=off skips the
    # second stage, not which embedder built the index.
    os.environ["DEEPECHO_RERANK"] = "off"
    for m in [m for m in sys.modules if m.startswith(("backend.config", "rag_assistant.chat"))]:
        del sys.modules[m]
    from backend import config
    from rag_assistant import chat
    assert config.RERANK is False
    embedder = getattr(chat.get_retriever(), "embedder", None)
    if getattr(embedder, "kind", "") == "nvidia" and not os.environ.get("NVIDIA_API_KEY"):
        print("     (skipped: the index on disk is NVIDIA-embedded and no key is set)")
        return
    hits = chat.retrieve("who do I notify about a ghost net", k=4, per_doc=2)
    assert 0 < len(hits) <= 4
    assert all(hasattr(c, "text") for c, _ in hits)


def test_flag_parsing():
    import importlib
    from backend import config
    cases = {"on": True, "1": True, "off": False, "0": False}
    for value, expected in cases.items():
        os.environ["DEEPECHO_RERANK"] = value
        importlib.reload(config)
        assert config.RERANK is expected, (value, config.RERANK)
    os.environ["DEEPECHO_RERANK"] = "auto"
    os.environ.pop("NVIDIA_API_KEY", None)
    importlib.reload(config)
    assert config.RERANK is False, "auto without a key is off"
    os.environ["NVIDIA_API_KEY"] = "nvapi-test"
    importlib.reload(config)
    assert config.RERANK is True, "auto with a key is on"
    os.environ.pop("NVIDIA_API_KEY")
    os.environ["DEEPECHO_RERANK"] = "off"
    importlib.reload(config)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
