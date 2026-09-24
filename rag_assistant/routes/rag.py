"""Assistant routes.

Three surfaces over one engine. /rag/query is the original contract, kept so
anything already calling it keeps working. /chat and /chat/stream are the full
one, used by the dashboard: the same answer with its citations, severity and
honesty flags attached, streamed where the interface wants it.

Both chat routes take `mode` ("auto", "copilot", "reference") and `language`.
A copilot turn answers from read-only survey data tools as well as the corpus,
and its response carries `tool_calls` and `data_citations`; its stream adds a
`tools` frame between meta and sources. docs/ASSISTANT.md has the details.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import config
from rag_assistant import chat
from backend.schemas import ChatRequest, ChatResponse
from rag_assistant import rag_service

router = APIRouter()
log = logging.getLogger("deepecho")


class RAGQuery(BaseModel):
    query: str = Field(min_length=1)
    retrieve_only: bool = False


@router.post("/rag/query")
def rag_query(request: RAGQuery):
    """The original route, now answering for real.

    `retrieve_only` returns the passages without calling a model, which is the
    cheapest way to prove the corpus is doing the work.
    """
    try:
        if request.retrieve_only:
            return rag_service.retrieve_context(request.query)
        result = rag_service.answer_question(request.query)
    except chat.EngineError as exc:
        log.warning("engine error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "query": request.query,
        "status": "ok",
        "answer": result["answer"],
        "grounded": result["grounded"],
        "refusal": result["refusal"],
        "intent": result["intent"],
        "severity": result["severity"],
        "sources": result["sources"],
    }


def _ghosttrace(request: ChatRequest) -> dict | None:
    """The GhostTrace handoff as the engine takes it: nulls kept, since null
    means unknown in that contract and must reach the prompt as unknown."""
    context = request.ghosttrace_context
    return context.to_engine_context() if context else None


@router.post("/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest) -> ChatResponse:
    """One conversational turn, answered from the corpus and nothing else."""
    record = request.detection_record.to_engine_record() if request.detection_record else None
    try:
        result = rag_service.answer_question(
            request.message,
            history=[turn.model_dump() for turn in request.history],
            detection=record,
            intent=request.intent,
            provider=request.provider,
            model=request.model,
            ghosttrace_context=_ghosttrace(request),
            # A person is reading this: when no model can answer, show them
            # the retrieved passages (or, for the copilot, the survey records),
            # labelled, rather than an error.
            offline_fallback=True,
            mode=request.mode,
            language=request.language,
        )
    except chat.EngineError as exc:
        log.warning("engine error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(**result)


def _sse(frames: Iterator[dict]) -> Iterator[str]:
    """Serialise frames as server-sent events.

    The done frame goes out through ChatResponse so the streamed payload is the
    same validated object /chat returns and the two cannot drift apart.
    """
    for frame in frames:
        if frame.get("type") == "done":
            payload = {"type": "done",
                       **ChatResponse(**{k: v for k, v in frame.items()
                                         if k != "type"}).model_dump()}
        else:
            payload = frame
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def post_chat_stream(request: ChatRequest) -> StreamingResponse:
    """The same turn, delivered as it is written.

    Frames are meta, then sources, then one delta per piece of text, then done.
    Retrieval finishes before generation starts, so the citations go out before
    the first word. `grounded` rides in the done frame alone, because it cannot
    be known until the answer is complete.
    """
    record = request.detection_record.to_engine_record() if request.detection_record else None
    try:
        frames = rag_service.stream_question(
            request.message,
            history=[turn.model_dump() for turn in request.history],
            detection=record,
            intent=request.intent,
            provider=request.provider,
            model=request.model,
            ghosttrace_context=_ghosttrace(request),
            # A person is reading this: when no model can answer, show them
            # the retrieved passages (or, for the copilot, the survey records),
            # labelled, rather than an error.
            offline_fallback=True,
            mode=request.mode,
            language=request.language,
        )
    except chat.EngineError as exc:
        log.warning("engine error before stream: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        _sse(frames),
        media_type=config.STREAM_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )
