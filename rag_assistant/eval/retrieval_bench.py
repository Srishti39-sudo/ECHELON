#!/usr/bin/env python3
"""Does retrieval find the document that answers the question?

    python rag_assistant/eval/retrieval_bench.py                 # the index as built
    python rag_assistant/eval/retrieval_bench.py --rerank        # + NVIDIA reranker
    python rag_assistant/eval/retrieval_bench.py --k 8 --verbose

`rag.py bench` measures the approximate index against exact search over the
SAME vectors. This measures something different: whether a question phrased in
an operator's words, not the document's, reaches the document that answers it.
Cases are in retrieval_cases.jsonl, each a query and the kb doc_id that holds
the answer. Several are deliberately paraphrased ("dark stripe" for shadow,
"notify" for notification) and two are in Hindi and Tamil, because that is
where a lexical index fails and a semantic one should not.

Three numbers, all mechanical:

    hit@1     the answering document is the top chunk's document
    hit@k     it is among the first k chunks (k = what the prompt receives)
    MRR       1 / rank of its first chunk, averaged; 1.0 means always first

Exit code 1 if hit@k is below --min-hit, so this can gate a change to the
index or the embedder the way eval/run.py gates a change to the assistant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag_assistant import rag  # noqa: E402

CASES = Path(__file__).resolve().parent / "retrieval_cases.jsonl"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(retriever, cases: list[dict], k: int, per_doc: int, rerank, verbose: bool) -> dict:
    hit1 = hitk = 0
    rr_sum = 0.0
    rows = []
    for case in cases:
        hits = retriever.search(case["query"], k=k, per_doc=per_doc)
        if rerank is not None:
            hits = rerank(case["query"], hits, k)
        docs = [c.meta.get("doc_id", c.id) for c, _ in hits]
        rank = next((i + 1 for i, d in enumerate(docs) if d == case["doc"]), None)
        hit1 += rank == 1
        hitk += rank is not None
        rr_sum += (1.0 / rank) if rank else 0.0
        rows.append((case["id"], rank, case["doc"], docs[0] if docs else "-"))
        if verbose or rank is None or rank > 1:
            print(f"  {case['id']:<4} rank={rank if rank else '-':<3} want={case['doc']:<32} got={docs[0] if docs else '-'}")
    n = len(cases)
    return {"n": n, "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr_sum / n, "rows": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=8, help="chunks the prompt receives (config.TOP_K)")
    p.add_argument("--per-doc", type=int, default=3)
    p.add_argument("--rerank", action="store_true", help="rerank with NVIDIA NIM (needs NVIDIA_API_KEY)")
    p.add_argument("--min-hit", type=float, default=0.0, help="exit 1 if hit@k is below this")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", dest="as_json", action="store_true")
    args = p.parse_args()

    retriever = rag.Retriever.load()
    embedder = getattr(retriever, "embedder", None)
    print(f"index: {getattr(getattr(retriever, 'index', None), 'kind', retriever.mode)}  "
          f"embedder: {getattr(embedder, 'kind', retriever.mode)}  chunks: {len(retriever.chunks)}")

    rerank = None
    if args.rerank:
        from rag_assistant.rerank import NvidiaReranker
        if not NvidiaReranker.available():
            print("--rerank needs NVIDIA_API_KEY", file=sys.stderr)
            return 2
        reranker = NvidiaReranker()
        fetch = max(args.k, 20)
        def rerank(query, hits, k, _r=reranker, _fetch=fetch):
            wide = retriever.search(query, k=_fetch, per_doc=max(args.per_doc, _fetch // 4))
            return _r.rerank(query, wide, k)
        print(f"reranker: {NvidiaReranker.model} (fetch {fetch}, keep {args.k})")

    cases = load_cases()
    print(f"{len(cases)} queries, k={args.k}, per_doc={args.per_doc}\n")
    result = run(retriever, cases, args.k, args.per_doc, rerank, args.verbose)
    k_key = f"hit@{args.k}"
    print(f"\nhit@1 {result['hit@1']:.3f}   {k_key} {result[k_key]:.3f}   MRR {result['mrr']:.3f}   (n={result['n']})")
    if args.as_json:
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return 1 if result[k_key] < args.min_hit else 0


if __name__ == "__main__":
    raise SystemExit(main())
