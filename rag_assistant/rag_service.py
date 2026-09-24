"""The grounded assistant, reached from the API shell.

This was a stub that raised NotImplementedError. The engine it was waiting for
is backend/chat.py: retrieval over the corpus in kb/, generation held to the
grounding rules in rag.py, and citations returned as structured data.

Everything here is a thin call into that engine. No retrieval, no prompting and
no severity logic lives in this file, because duplicating any of it is how the
API and the engine start disagreeing about what the assistant may say.
"""

from __future__ import annotations

from typing import Any

from rag_assistant import chat


def answer_question(message: str,
                    history: list[dict] | None = None,
                    detection: dict | None = None,
                    **options: Any) -> dict:
    """One grounded turn. Returns the full response contract as a dict."""
    return chat.answer(message, history=history, detection=detection, **options)


def stream_question(message: str,
                    history: list[dict] | None = None,
                    detection: dict | None = None,
                    **options: Any):
    """The same turn as frames: meta, sources, deltas, done."""
    return chat.answer_stream(message, history=history, detection=detection, **options)


def retrieve_context(query: str) -> dict:
    """Retrieval only, no generation and no model call.

    Kept under its original name because it is the function this service was
    created for. It now does what it said: returns the passages the assistant
    would answer from, with their provenance, so a caller can show the evidence
    without paying for a completion.
    """
    retriever = chat.get_retriever()
    hits = retriever.search(query, k=chat.config.TOP_K, per_doc=chat.config.PER_DOC)
    return {"query": query, "sources": chat.sources_from(hits)}
