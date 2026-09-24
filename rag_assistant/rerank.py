"""Second-stage ranking of retrieved passages, over NVIDIA NIM.

The index answers "which chunks are near this query" cheaply, for every chunk.
A reranker answers a more expensive question for a handful of candidates: does
THIS passage answer THIS query, with both read together. Retrieve wide, rerank,
keep few -- retrieve 20, keep 8 -- is the shape every production pipeline has
settled on, because the candidate set is where the signal is and the reranker
is where the precision is.

The NeMo Retriever reranker is a cross-encoder trained for retrieval in the
same languages as the embedder, so a Tamil question is scored against an
English passage directly. It runs as a NIM container on-premises too.

The reranker is advisory: it reorders and trims what retrieval returned, it
never adds a passage the index did not find, and any failure (no key, network,
rate limit) returns the retrieval order untouched with the reason logged. An
answer must never depend on a second network call succeeding.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("deepecho")

# The hosted catalogue retires models every few months (llama-3.2-nv-rerankqa
# went end-of-life in 2026-05, llama-nemotron-rerank-1b-v2 in 2026-08), so the
# model and its endpoint are both overridable from .env. The "vl" model scores
# a text query against text passages exactly like the text-only one did.
DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
# The NIM reranking endpoint is not OpenAI-shaped; it has its own path.
DEFAULT_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"


class NvidiaReranker:
    model = os.environ.get("DEEPECHO_RERANK_MODEL", DEFAULT_MODEL)
    url = os.environ.get("NVIDIA_RERANK_URL", DEFAULT_URL)

    def __init__(self, timeout_s: float = 20.0):
        self.timeout_s = timeout_s

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("NVIDIA_API_KEY"))

    def scores(self, query: str, passages: list[str]) -> list[float]:
        """One relevance logit per passage, in passage order. Raises on failure."""
        import requests

        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        body = {"model": self.model, "query": {"text": query},
                "passages": [{"text": p} for p in passages], "truncate": "END"}
        response = requests.post(
            self.url, json=body, timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        if response.status_code != 200:
            raise RuntimeError(f"reranker HTTP {response.status_code}: {response.text[:200]}")
        rankings = response.json().get("rankings") or []
        out = [float("-inf")] * len(passages)
        for r in rankings:
            out[int(r["index"])] = float(r["logit"])
        return out

    def rerank(self, query: str, hits: list[tuple[Any, float]], keep: int) -> list[tuple[Any, float]]:
        """Reorder (chunk, score) pairs by the reranker and keep the top `keep`.

        The returned score is the reranker's logit, so a caller can tell the two
        stages apart. On any failure the input is returned unchanged, trimmed.
        """
        if not hits:
            return hits
        try:
            logits = self.scores(query, [chunk.text for chunk, _ in hits])
        except Exception as exc:  # advisory: never let a second call break an answer
            log.warning("reranker unavailable, keeping retrieval order: %s", exc)
            return hits[:keep]
        order = sorted(range(len(hits)), key=lambda i: -logits[i])
        return [(hits[i][0], logits[i]) for i in order[:keep]]
