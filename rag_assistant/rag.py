#!/usr/bin/env python3
"""DeepEcho RAG assistant.

The detection model answers "what's there?". This answers "what does it mean,
and what do I do?" -- grounded strictly in a curated corpus of maritime
documents under kb/, never in the model's own memory.

Four functions:
    explain   a detection            -> what it is, risk, standoff distance
    ask       an operator question   -> grounded answer with citations
    anomaly   an unclassified object -> unknown-object protocol + near matches
    report    an incident            -> formatted report for an authority

Retrieval is dense vector search over a FAISS HNSW index, with an exact
brute-force path for verification and a pure-stdlib fallback when numpy is not
installed. Generation runs on Gemini Flash by default, with Groq behind the same seam.

    python3 rag.py index
    python3 rag.py bench
    python3 rag.py ask "what is the disposal procedure for a moored mine?"
    python3 rag.py explain --detection detection.json
    python3 rag.py anomaly --describe "cylinder, hard shadow" --provider groq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "kb"
CATALOG_PATH = ROOT / "catalog" / "objects.json"
MANIFEST_PATH = ROOT / "index.json"
VECTORS_PATH = ROOT / "vectors.npy"
FAISS_PATH = ROOT / "index.faiss"
CATALOG_INDEX_PATH = ROOT / "catalog.faiss"
CATALOG_VECTORS_PATH = ROOT / "catalog_vectors.npy"

ENV_PATH = ROOT / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    """Read KEY=value lines from .env into the environment.

    A variable already set in the real environment always wins, so an exported
    key overrides the file rather than the other way round. No dependency: this
    is a handful of lines and python-dotenv would be another install.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env()
DEFAULT_PROVIDER = os.environ.get("DEEPECHO_PROVIDER", "gemini")
DEFAULT_K = 6
DEFAULT_DIM = 512

# HNSW build and search parameters. M is the graph degree, efConstruction the
# build-time candidate list, efSearch the query-time candidate list. efSearch is
# the recall/latency dial and is the one worth tuning per corpus -- see `bench`.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ModuleNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Knowledge base loading
# ---------------------------------------------------------------------------

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Read the leading `---` block as flat key: value metadata."""
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, raw[match.end():]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown on headings so a chunk never straddles two topics."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = line.lstrip("#").strip()
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, t) for h, t in sections if t]


def pack(paragraphs: list[str], target: int = 900, overlap: int = 150) -> list[str]:
    """Pack paragraphs into ~target-char chunks with a little overlap."""
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        if cur and len(cur) + len(para) + 2 > target:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + "\n\n" + para if overlap else para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def load_chunks(kb_dir: Path) -> list[Chunk]:
    docs = sorted(p for p in kb_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    docs = [p for p in docs
            if p.name.upper() != "README.MD" and "sources" not in p.parts]
    if not docs:
        raise SystemExit(f"No .md/.txt documents found in {kb_dir}")

    chunks: list[Chunk] = []
    for path in docs:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        # Front matter is what carries provenance. A file without it is a raw
        # capture or a stray note, not a corpus document, and indexing it would
        # make untraceable text citable. Skip it loudly.
        if not meta:
            print(f"SKIPPED {path.name}: no front matter, so no provenance.",
                  file=sys.stderr)
            continue
        meta.setdefault("title", path.stem.replace("-", " ").title())
        meta.setdefault("status", "UNVERIFIED")
        meta["doc_id"] = path.stem
        meta["path"] = str(path.relative_to(kb_dir))
        for section_heading, section_text in split_sections(body):
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
            for piece in pack(paragraphs):
                cid = f"{path.stem}#{len(chunks)}"
                text = f"{section_heading}\n{piece}" if section_heading else piece
                chunks.append(Chunk(cid, text, {**meta, "section": section_heading}))
    return chunks


# ---------------------------------------------------------------------------
# Term weighting
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "as", "at", "by", "be", "it", "this", "that", "from", "if", "not",
}


def tokenize(text: str) -> list[str]:
    words = [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP and len(w) > 1]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def build_idf(texts: list[str]) -> dict[str, float]:
    n = len(texts)
    df = Counter()
    for text in texts:
        df.update(set(tokenize(text)))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def tfidf(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Sparse TF-IDF, L2-normalised. Unseen terms carry no signal."""
    counts = Counter(tokenize(text))
    vec = {}
    for term, count in counts.items():
        weight = idf.get(term)
        if weight is not None:
            vec[term] = (1.0 + math.log(count)) * weight
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {t: v / norm for t, v in vec.items()} if norm else {}


# ---------------------------------------------------------------------------
# Embedders: sparse TF-IDF in, dense unit vectors out
# ---------------------------------------------------------------------------

class TfidfDenseEmbedder:
    """TF-IDF as a dense vector over the fitted vocabulary. No fidelity loss.

    The default. Dimension is the vocabulary size, so cosine here is exactly the
    cosine of the sparse representation -- the index returns what exhaustive
    term matching would return, and nothing is traded away for speed.

    It grows with the vocabulary rather than staying fixed, so past roughly tens
    of thousands of terms, move to sentence-transformers (better matching, fixed
    384 dimensions) or random projection (fixed dimensions, measured loss).
    """

    kind = "tfidf-dense"

    def __init__(self, idf: dict[str, float], center: bool = False):
        import numpy as np
        self.np = np
        self.idf = idf
        self.terms = sorted(idf)
        self.pos = {t: i for i, t in enumerate(self.terms)}
        self.dim = len(self.terms)
        self.center = center

    def encode(self, text: str):
        np = self.np
        vec = np.zeros(self.dim, dtype="float32")
        for term, weight in tfidf(text, self.idf).items():
            vec[self.pos[term]] = weight
        if self.center:
            vec -= vec.mean()
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode_many(self, texts: list[str]):
        return self.np.vstack([self.encode(t) for t in texts]).astype("float32")

    def state(self) -> dict:
        return {"kind": self.kind, "center": self.center, "idf": self.idf}


class RandomProjectionEmbedder:
    """TF-IDF projected into a fixed dense space by sparse random projection.

    Each term gets a deterministic pseudo-random unit-variance vector, seeded by
    a hash of the term itself, so no projection matrix is stored and the
    vocabulary can grow without a rebuild. Johnson-Lindenstrauss says inner
    products survive the projection, which is what the index needs.

    Opt in with `--embedder random-projection` when the vocabulary is too large
    for TfidfDenseEmbedder. It is lossy, and the loss is measurable: on this
    corpus, agreement with the exact ranking is 0.57 at 512 dimensions and 0.70
    at 2048. Measure before trusting it.
    """

    kind = "random-projection"

    def __init__(self, idf: dict[str, float], dim: int = DEFAULT_DIM, seed: int = 0,
                 center: bool = False):
        import numpy as np
        self.np = np
        self.idf = idf
        self.dim = dim
        self.seed = seed
        self.center = center
        self._cache: dict[str, "np.ndarray"] = {}

    def _term_vector(self, term: str):
        cached = self._cache.get(term)
        if cached is None:
            digest = hashlib.sha1(f"{self.seed}:{term}".encode()).digest()[:8]
            rng = self.np.random.default_rng(int.from_bytes(digest, "big"))
            cached = rng.standard_normal(self.dim).astype("float32")
            self._cache[term] = cached
        return cached

    def encode(self, text: str):
        np = self.np
        vec = np.zeros(self.dim, dtype="float32")
        for term, weight in tfidf(text, self.idf).items():
            vec += weight * self._term_vector(term)
        # Pearson is cosine on mean-centred vectors, so centring here makes the
        # whole index a Pearson index. Metric is a property of the index.
        if self.center:
            vec -= vec.mean()
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode_many(self, texts: list[str]):
        return self.np.vstack([self.encode(t) for t in texts]).astype("float32")

    def state(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "seed": self.seed,
                "center": self.center, "idf": self.idf}


class SentenceTransformerEmbedder:
    """Real sentence embeddings, when sentence-transformers is installed."""

    kind = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", center: bool = False):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.center = center

    def encode_many(self, texts: list[str]):
        vecs = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        if self.center:
            vecs -= vecs.mean(axis=1, keepdims=True)
        norms = self.np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode(self, text: str):
        return self.encode_many([text])[0]

    def state(self) -> dict:
        return {"kind": self.kind, "model_name": self.model_name,
                "dim": self.dim, "center": self.center}


class GeminiEmbedder:
    """Gemini embeddings. Free tier, and no local model to download.

    Document and query text are embedded with different task types, which is
    what the model is trained to expect for retrieval and is worth more than it
    looks. Every query costs a network call, unlike the local embedders.
    """

    kind = "gemini"
    BATCH = 100

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768,
                 center: bool = False):
        import numpy as np
        self.np = np
        self.client = gemini_client()
        self.model = model
        self.dim = dim
        self.center = center

    def _embed(self, texts: list[str], task_type: str):
        from google.genai import types, errors
        import numpy as np
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start:start + self.BATCH]
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=self.dim),
                )
            except errors.ClientError as exc:
                raise SystemExit(f"Gemini embedding request rejected: {exc}")
            except errors.ServerError as exc:
                raise SystemExit(f"Gemini embedding service error: {exc}")
            out.extend(e.values for e in response.embeddings)
        vecs = np.asarray(out, dtype="float32")
        if self.center:
            vecs -= vecs.mean(axis=1, keepdims=True)
        # Truncated Gemini embeddings are not unit length, so always renormalise.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode_many(self, texts: list[str]):
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def encode(self, text: str):
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def state(self) -> dict:
        return {"kind": self.kind, "model": self.model, "dim": self.dim,
                "center": self.center}


def make_embedder(state: dict):
    kind = state["kind"]
    if kind == GeminiEmbedder.kind:
        return GeminiEmbedder(state["model"], state["dim"], state.get("center", False))
    if kind == SentenceTransformerEmbedder.kind:
        return SentenceTransformerEmbedder(state["model_name"], state.get("center", False))
    if kind == RandomProjectionEmbedder.kind:
        return RandomProjectionEmbedder(state["idf"], state["dim"], state["seed"],
                                        state.get("center", False))
    return TfidfDenseEmbedder(state["idf"], state.get("center", False))


# ---------------------------------------------------------------------------
# Vector indexes
# ---------------------------------------------------------------------------

class FaissHNSW:
    """FAISS HNSW over unit vectors. Inner product on unit vectors is cosine.

    HNSW is a navigable small-world graph: search descends coarse layers and
    refines, so query cost grows with log(n) rather than n. The same code path
    serves 31 chunks and tens of millions.
    """

    kind = "faiss-hnsw"

    def __init__(self, index, ef_search: int = HNSW_EF_SEARCH):
        self.index = index
        self.ef_search = ef_search
        self.index.hnsw.efSearch = ef_search

    @classmethod
    def build(cls, vectors, m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION,
              ef_search: int = HNSW_EF_SEARCH) -> "FaissHNSW":
        import faiss
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(vectors)
        return cls(index, ef_search)

    def save(self, path: Path = FAISS_PATH) -> None:
        import faiss
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path = FAISS_PATH, ef_search: int = HNSW_EF_SEARCH) -> "FaissHNSW":
        import faiss
        return cls(faiss.read_index(str(path)), ef_search)

    def search(self, query, k: int) -> list[tuple[int, float]]:
        scores, ids = self.index.search(query.reshape(1, -1), min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


class ExactFlat:
    """Brute-force dense search. The ground truth `bench` measures HNSW against."""

    kind = "exact"

    def __init__(self, vectors):
        import numpy as np
        self.np = np
        self.vectors = vectors

    def search(self, query, k: int) -> list[tuple[int, float]]:
        scores = self.vectors @ query
        order = self.np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


class SparseExact:
    """Zero-dependency fallback: exact cosine over sparse TF-IDF dicts."""

    kind = "sparse-exact"

    def __init__(self, vectors: list[dict]):
        self.vectors = vectors

    @staticmethod
    def cosine(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def search(self, query: dict, k: int) -> list[tuple[int, float]]:
        scored = [(i, self.cosine(query, v)) for i, v in enumerate(self.vectors)]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# The retriever ties chunks, embedder and index together
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, chunks: list[Chunk], embedder, index, mode: str,
                 sparse_vectors: list[dict] | None = None, idf: dict | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.index = index
        self.mode = mode
        self.sparse_vectors = sparse_vectors
        self.idf = idf or {}

    # -- build -------------------------------------------------------------

    @classmethod
    def build(cls, kb_dir: Path = KB_DIR, dim: int = DEFAULT_DIM, center: bool = False,
              embedder_kind: str = "auto", ann: str = "auto",
              m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION) -> "Retriever":
        chunks = load_chunks(kb_dir)
        texts = [c.text for c in chunks]
        idf = build_idf(texts)

        if not have("numpy"):
            if embedder_kind != "auto" or ann != "auto":
                raise SystemExit("numpy is required for the dense path. pip install -r requirements.txt")
            print("numpy not installed -- falling back to exact sparse retrieval.", file=sys.stderr)
            vectors = [tfidf(t, idf) for t in texts]
            return cls(chunks, None, SparseExact(vectors), "sparse", vectors, idf)

        if embedder_kind == "sentence-transformers" or (
                embedder_kind == "auto" and have("sentence_transformers")):
            embedder = SentenceTransformerEmbedder(center=center)
        elif embedder_kind == "gemini":
            embedder = GeminiEmbedder(dim=dim if dim != DEFAULT_DIM else 768, center=center)
        elif embedder_kind == "random-projection":
            embedder = RandomProjectionEmbedder(idf, dim=dim, center=center)
        else:
            embedder = TfidfDenseEmbedder(idf, center=center)

        vectors = embedder.encode_many(texts)
        if ann == "exact" or (ann == "auto" and not have("faiss")):
            if ann == "auto" and not have("faiss"):
                print("faiss not installed -- using exact dense search.", file=sys.stderr)
            index = ExactFlat(vectors)
        else:
            index = FaissHNSW.build(vectors, m=m, ef_construction=ef_construction)
        return cls(chunks, embedder, index, "dense", None, idf)

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        manifest = {
            "mode": self.mode,
            "index": self.index.kind,
            "chunks": [{"id": c.id, "text": c.text, "meta": c.meta} for c in self.chunks],
        }
        if self.mode == "sparse":
            manifest["idf"] = self.idf
            manifest["vectors"] = self.sparse_vectors
        else:
            manifest["embedder"] = self.embedder.state()
            manifest["hnsw"] = {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}
            import numpy as np
            vectors = (self.index.vectors if isinstance(self.index, ExactFlat)
                       else self.index.index.reconstruct_n(0, self.index.index.ntotal))
            np.save(VECTORS_PATH, vectors)
            if isinstance(self.index, FaissHNSW):
                self.index.save()
            elif FAISS_PATH.exists():
                FAISS_PATH.unlink()
        MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")

    @classmethod
    def load(cls, ef_search: int = HNSW_EF_SEARCH) -> "Retriever":
        if not MANIFEST_PATH.exists():
            raise SystemExit("No index found. Run: python3 rag.py index")
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        chunks = [Chunk(c["id"], c["text"], c["meta"]) for c in data["chunks"]]

        if data["mode"] == "sparse":
            return cls(chunks, None, SparseExact(data["vectors"]), "sparse",
                       data["vectors"], data["idf"])

        embedder = make_embedder(data["embedder"])
        if data["index"] == FaissHNSW.kind:
            index = FaissHNSW.load(ef_search=ef_search)
        else:
            import numpy as np
            index = ExactFlat(np.load(VECTORS_PATH))
        return cls(chunks, embedder, index, "dense")

    def exact_twin(self) -> "Retriever":
        """The same corpus and embedder behind exact search, for benchmarking."""
        if self.mode != "dense":
            return self
        import numpy as np
        return Retriever(self.chunks, self.embedder, ExactFlat(np.load(VECTORS_PATH)), "dense")

    # -- query -------------------------------------------------------------

    def encode_query(self, query: str):
        if self.mode == "sparse":
            return tfidf(query, self.idf)
        return self.embedder.encode(query)

    def search(self, query: str, k: int = DEFAULT_K, per_doc: int = 2,
               overfetch: int = 4) -> list[tuple[Chunk, float]]:
        qv = self.encode_query(query)
        if self.mode == "sparse" and not qv:
            return []
        # Over-fetch before the per-document cap, so capping never returns fewer
        # than k. With HNSW this also raises the odds the true neighbour is in
        # the candidate set at all.
        raw = self.index.search(qv, min(k * overfetch, len(self.chunks)))

        kept: list[tuple[Chunk, float]] = []
        spare: list[tuple[Chunk, float]] = []
        seen = Counter()
        for idx, score in raw:
            chunk = self.chunks[idx]
            doc = chunk.meta.get("doc_id", chunk.id)
            if seen[doc] < per_doc:
                seen[doc] += 1
                kept.append((chunk, score))
            else:
                spare.append((chunk, score))
        return (kept + spare)[:k]

    def rank_ids(self, query: str, k: int) -> list[int]:
        """Ranked chunk ids. Encodes the query, so a remote embedder bills here."""
        return [i for i, _ in self.index.search(self.encode_query(query), k)]


# ---------------------------------------------------------------------------
# Known-object catalog: similarity matching for unclassified detections
# ---------------------------------------------------------------------------

class Catalog:
    """Nearest neighbours among known object classes, for anomalies only.

    Two matching spaces, and they are never mixed:

    * embedding -- when the detection record carries an `embedding` array and
      every catalog entry carries one too, matching happens in the detection
      model's own space. This is the real path.
    * descriptor -- otherwise, the entry's text descriptor is matched against
      the operator's description of the object. Weaker, but it runs today
      without a trained embedding head.

    A result here is a hint. It is not an identification, and the scores are
    cosine similarities, not probabilities and not confidences.
    """

    def __init__(self, entries: list[dict], vectors, space: str, embedder=None):
        self.entries = entries
        self.vectors = vectors
        self.space = space
        self.embedder = embedder
        self.index = None

    @staticmethod
    def _unit(vectors):
        import numpy as np
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    @classmethod
    def build(cls, path: Path = CATALOG_PATH) -> "Catalog | None":
        if not path.exists():
            return None
        entries = json.loads(path.read_text(encoding="utf-8"))["objects"]
        if not entries:
            return None
        if not have("numpy"):
            return None

        embedded = [e for e in entries if e.get("embedding")]
        if embedded and len(embedded) != len(entries):
            raise SystemExit(
                f"{len(embedded)} of {len(entries)} catalog entries carry an embedding. "
                "Matching in embedding space needs all of them or none.")

        if embedded:
            dims = {len(e["embedding"]) for e in entries}
            if len(dims) != 1:
                raise SystemExit(f"Catalog embeddings have inconsistent dimensions: {sorted(dims)}")
            return cls(entries, cls._unit([e["embedding"] for e in entries]), "embedding")

        texts = [cls.entry_text(e) for e in entries]
        embedder = TfidfDenseEmbedder(build_idf(texts))
        return cls(entries, embedder.encode_many(texts), "descriptor", embedder)

    @staticmethod
    def entry_text(entry: dict) -> str:
        return " ".join(str(entry.get(f, "")) for f in ("name", "class", "descriptor"))

    def attach_index(self) -> None:
        if self.index is not None:
            return
        if have("faiss") and len(self.entries) > 1:
            self.index = FaissHNSW.build(self.vectors)
        else:
            self.index = ExactFlat(self.vectors)

    def save(self) -> None:
        import numpy as np
        np.save(CATALOG_VECTORS_PATH, self.vectors)
        state = {"space": self.space, "entries": self.entries}
        if self.embedder is not None:
            state["embedder"] = self.embedder.state()
        (ROOT / "catalog_index.json").write_text(json.dumps(state), encoding="utf-8")

    @classmethod
    def load(cls) -> "Catalog | None":
        state_path = ROOT / "catalog_index.json"
        if not state_path.exists() or not have("numpy"):
            return None
        import numpy as np
        state = json.loads(state_path.read_text(encoding="utf-8"))
        embedder = make_embedder(state["embedder"]) if "embedder" in state else None
        catalog = cls(state["entries"], np.load(CATALOG_VECTORS_PATH), state["space"], embedder)
        catalog.attach_index()
        return catalog

    def match(self, record: dict, top: int = 3) -> list[tuple[dict, float]]:
        """Rank catalog entries against the detection. Empty when we cannot."""
        import numpy as np
        detection_embedding = record.get("embedding")

        if self.space == "embedding":
            if not detection_embedding:
                return []
            if len(detection_embedding) != self.vectors.shape[1]:
                raise SystemExit(
                    f"Detection embedding is {len(detection_embedding)}-dimensional, "
                    f"catalog is {self.vectors.shape[1]}-dimensional.")
            query = self._unit([detection_embedding])[0]
        else:
            if detection_embedding:
                raise SystemExit(
                    "The detection carries an embedding but the catalog does not. "
                    "Add an `embedding` array to every entry in catalog/objects.json, "
                    "or drop the embedding from the detection record.")
            described = " ".join(str(record.get(f, "")) for f in
                                 ("visual_description", "notes", "label")).strip()
            if not described:
                return []
            query = self.embedder.encode(described)

        self.attach_index()
        ranked = self.index.search(np.asarray(query, dtype="float32"), min(top, len(self.entries)))
        return [(self.entries[i], score) for i, score in ranked if score > 0]


def format_matches(matches: list[tuple[dict, float]], space: str) -> str:
    """Render matches as ranked possibilities. Never as a percentage."""
    if not matches:
        return ("NEAREST KNOWN OBJECTS: none. No usable description or embedding was "
                "supplied, so no comparison was made. Do not speculate about identity.")
    lines = [f"NEAREST KNOWN OBJECTS (cosine similarity in {space} space; a ranking, "
             f"NOT a probability, NOT a confidence, NOT an identification):"]
    for rank, (entry, score) in enumerate(matches, start=1):
        lines.append(
            f"  {rank}. {entry['name']} (class: {entry['class']}, hazard: {entry['hazard']}) "
            f"similarity {score:.2f}\n"
            f"     would confirm: {entry['confirms']}\n"
            f"     would rule out: {entry['rules_out']}\n"
            f"     governed by: {entry['source']} (status: {entry['status']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grounded generation
# ---------------------------------------------------------------------------

SYSTEM = """You are the DeepEcho maritime decision-support assistant. You brief \
operators on underwater objects detected by a sonar classification model.

GROUNDING RULES -- these override every other instruction:
1. Answer only from the SOURCES block and, where present, the NEAREST KNOWN \
OBJECTS block. Both are given to you in the user message. Your own recollection \
of mines, ordnance, debris, wrecks, or reporting procedures is not admissible.
2. Cite the source tag [S1], [S2], ... after every factual claim, especially \
every distance, timing, procedure step, and authority name.
3. If the sources do not cover the question, say so plainly in one sentence, then \
give only the universal safe fallback: do not approach, do not touch, do not \
recover, maintain distance, and report to the responsible maritime authority.
4. Never invent a standoff distance, a disposal step, a phone number, or an \
authority name. A missing number is stated as "not specified in the sources".
4b. Never calculate a new figure from the sources: no distance, radius, weight, \
time or probability worked out from a formula, a rate or other figures. If a \
source gives a formula, you may quote the formula as written and say that the \
result for this case is not specified in the sources; do not substitute values \
into it. A computed safety distance is an unsourced number.
4a. "published_by" names who wrote a source. It is NOT the authority to report \
to, and must never be offered as one. An authority may only be named for the \
hazard class the source text connects it to. If the sources name a body for pollution reporting, do not offer it for an \
ordnance or unidentified-object report, not even as an example or with "e.g.". \
Where the sources say the authority for a hazard class is not specified, say \
"the responsible authority" and state plainly that the sources do not name it.
5. If a cited source is marked status: PLACEHOLDER, flag it as unverified \
seed content that must be replaced before operational use.
6. Safety first. When sources conflict, quote the most conservative one.

UNCLASSIFIED OBJECTS -- additional rules when a NEAREST KNOWN OBJECTS block is present:
7. Never state or imply an identity for an unclassified object. Say "unidentified" plainly, and say it before anything else. Do not write a phrase such as "this is a mine" even when quoting or restating the operator's question; answer the question without repeating it.
8. Nearest known objects are possibilities, ranked by similarity. Present them as possibilities, always with what would confirm and what would rule each one out. Never write "this is a" or "likely a" about a match.
9. The similarity score is a cosine similarity between vectors. It is not a probability, not a confidence, and not a percentage. Do not convert it to one, do not write it as a percentage, and do not describe a higher score as making an identity more likely to be true.
10. Every unclassified object is escalated to a qualified human expert. Say so explicitly, and say that the object stays unidentified until that expert rules.

Write for an operator on deck: short sentences, plain words, action before \
explanation. Lead with the risk and the required standoff."""

TASKS = {
    "explain": """Explain this detection to the operator.

DETECTION:
{detection}

Cover, in this order: what the object most likely is; the immediate risk level; \
the required standoff distance; the immediate actions; who to notify. Keep it \
under 200 words.""",

    "ask": """Answer the operator's question. End every sentence that states a \
fact, a prohibition or a procedure step with its [Sn] citation.

QUESTION: {question}""",

    "anomaly": """The classifier could not identify this object. It is UNCLASSIFIED.

OBSERVATION:
{detection}

{matches}

Do four things, in this order:
1. State plainly that the object is unidentified, and that it is treated as \
potentially hazardous until a qualified expert identifies it.
2. Give the protocol for unidentified underwater objects exactly as the sources \
state it, cited.
3. Walk the nearest known objects above in rank order. For each, give the \
similarity score as a score, then what would confirm it and what would rule it \
out. Do not identify the object. Do not pick a winner.
4. State the escalation: who this goes to for expert review, per the sources, and \
that the object remains unidentified until they rule.""",

    "report": """Generate an incident report an authority can act on.

DETECTION RECORD:
{detection}

Use this structure with markdown headings: Summary; Detection Data (echo every \
field from the record verbatim, and write NOT PROVIDED for any field that is \
absent -- never fill a gap); Assessment (grounded, cited); Required Actions \
(cited); Notification (who, per the sources); Sources. Do not add fields that \
are not in the record or the sources.""",
}


def format_sources(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, score) in enumerate(hits, start=1):
        meta = chunk.meta
        header = (
            f"[S{i}] title: {meta.get('title')} | published_by: {meta.get('authority', 'unknown')} "
            f"| status: {meta.get('status')} | doc: {meta.get('path')} | similarity: {score:.3f}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


# --- LLM providers ---------------------------------------------------------
# One seam, three backends. The grounding rules live in SYSTEM and are identical
# across all of them, so switching provider does not change what the assistant
# is allowed to say.

GEMINI_KEY_HINT = ("GEMINI_API_KEY (or GOOGLE_API_KEY) in .env. "
                   "Free key: https://aistudio.google.com/apikey")


def gemini_client(attempts: int = 5, timeout_ms: int = 120_000):
    """A Gemini client that retries the free tier's overload responses.

    Free-tier Flash returns 503 under load often enough that a demo will hit it.
    The SDK has retry built in, so configure it rather than hand-rolling a loop:
    five attempts, exponential backoff with jitter, on the throttling and
    server-side status codes only. The short calls (translation, tool planning)
    ask for fewer attempts and a tighter timeout, because each has a fallback
    that is better than a minute of waiting.
    """
    from google import genai
    from google.genai import types

    try:
        return genai.Client(http_options=types.HttpOptions(
            timeout=timeout_ms,  # milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=attempts, initial_delay=1.0, max_delay=30.0,
                exp_base=2.0, jitter=0.5,
                http_status_codes=[408, 429, 500, 502, 503, 504]),
        ))
    except Exception as exc:
        raise SystemExit(f"Could not create the Gemini client: {exc}\nSet {GEMINI_KEY_HINT}")


class GeminiProvider:
    """Gemini Flash. Free tier, and the default."""

    name = "gemini"
    default_model = "gemini-3.8-flash"
    key_hint = GEMINI_KEY_HINT

    def complete(self, system: str, user: str, model: str, *, max_tokens: int = 4096,
                 attempts: int = 5, timeout_s: float = 120.0) -> str:
        from google.genai import types, errors

        client = gemini_client(attempts, int(timeout_s * 1000))

        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,      # grounded extraction, not composition
                    max_output_tokens=max_tokens,
                    # No tools are in play; without this the SDK warns per call.
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True),
                ),
            )
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise SystemExit("Gemini free-tier rate limit reached. Wait and retry, "
                                 "or check your limits at https://aistudio.google.com/rate-limit")
            raise SystemExit(f"Gemini rejected the request: {exc}\nCheck {self.key_hint}")
        except errors.ServerError as exc:
            raise SystemExit(
                f"Gemini is still failing after retries: {exc}\n"
                "Free-tier Flash overloads under demand. Try --provider groq, "
                "or --model gemini-2.5-flash.")
        except errors.APIError as exc:
            raise SystemExit(f"Gemini API error: {exc}")

        text = (response.text or "").strip()
        if text:
            return text

        # An empty response is usually a safety block. On a corpus about mines
        # and ordnance that is a live possibility, so name it rather than
        # returning nothing and letting it look like a retrieval failure.
        reason = None
        if response.candidates:
            reason = getattr(response.candidates[0], "finish_reason", None)
        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        raise SystemExit(
            f"Gemini returned no text (finish_reason={reason}, block_reason={blocked}). "
            "A safety filter on ordnance content is the likely cause. Retry, or run "
            "with --provider groq.")


    def stream(self, system: str, user: str, model: str):
        """The same call, yielded in pieces. Identical config to complete().

        Streaming changes when the operator sees the words, never which words
        they are: same system prompt, same temperature, same model. Nothing a
        streamed answer is allowed to say differs from a completed one.
        """
        from google.genai import types, errors

        client = gemini_client()
        produced = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=4096,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True),
                ),
            ):
                piece = chunk.text or ""
                if piece:
                    produced = True
                    yield piece
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise SystemExit("Gemini free-tier rate limit reached. Wait and retry, "
                                 "or check your limits at https://aistudio.google.com/rate-limit")
            raise SystemExit(f"Gemini rejected the request: {exc}\nCheck {self.key_hint}")
        except errors.ServerError as exc:
            raise SystemExit(
                f"Gemini is still failing after retries: {exc}\n"
                "Free-tier Flash overloads under demand. Try --provider groq, "
                "or --model gemini-2.5-flash.")
        except errors.APIError as exc:
            raise SystemExit(f"Gemini API error: {exc}")

        if not produced:
            raise SystemExit(
                "Gemini streamed no text. A safety filter on ordnance content is the "
                "likely cause. Retry, or run with --provider groq.")


    def plan(self, system: str, user: str, tools: list[dict], model: str, *,
             timeout_s: float = 20.0) -> list[dict]:
        """Native function calling: which tools to call, never an answer.

        Automatic function calling stays disabled. The calls come back as data
        and run in backend/copilot_tools.py, so nothing the model names is
        executed by the SDK.
        """
        from google.genai import types, errors

        client = gemini_client(2, int(timeout_s * 1000))
        declarations = [types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters_json_schema=t["parameters"]) for t in tools]
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=1024,
                    tools=[types.Tool(function_declarations=declarations)],
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=[t["name"] for t in tools])),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise SystemExit("Gemini free-tier rate limit reached. Wait and retry.")
            raise SystemExit(f"Gemini rejected the tool plan request: {exc}")
        except errors.ServerError as exc:
            raise SystemExit(f"Gemini is still failing after retries: {exc}")
        except errors.APIError as exc:
            raise SystemExit(f"Gemini API error: {exc}")
        return [{"name": call.name, "args": dict(call.args or {})}
                for call in (response.function_calls or []) if call.name]


class GroqProvider:
    """Open models on Groq. Fast, free tier, and the on-prem story."""

    name = "groq"
    default_model = "openai/gpt-oss-120b"
    key_hint = "GROQ_API_KEY. Free key: https://console.groq.com/keys"

    def complete(self, system: str, user: str, model: str, *, max_tokens: int = 4096,
                 attempts: int = 3, timeout_s: float = 120.0) -> str:
        import groq

        try:
            client = groq.Groq(max_retries=max(0, attempts - 1), timeout=timeout_s)
        except Exception as exc:
            raise SystemExit(f"Could not create the Groq client: {exc}\nSet {self.key_hint}")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
        except groq.AuthenticationError:
            raise SystemExit(f"No valid credentials. Set {self.key_hint}")
        except groq.RateLimitError:
            raise SystemExit("Groq free-tier rate limit reached. Wait and retry.")
        except groq.APIStatusError as exc:
            raise SystemExit(f"Groq API error {exc.status_code}: {exc.message}")
        except groq.APIConnectionError:
            raise SystemExit("Network error reaching the Groq API.")

        return (response.choices[0].message.content or "").strip()


    def stream(self, system: str, user: str, model: str):
        """Same call, same config, delivered incrementally."""
        import groq

        try:
            client = groq.Groq()
        except Exception as exc:
            raise SystemExit(f"Could not create the Groq client: {exc}\nSet {self.key_hint}")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=4096,
                stream=True,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            for chunk in response:
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
        except groq.AuthenticationError:
            raise SystemExit(f"No valid credentials. Set {self.key_hint}")
        except groq.RateLimitError:
            raise SystemExit("Groq free-tier rate limit reached. Wait and retry.")
        except groq.APIStatusError as exc:
            raise SystemExit(f"Groq API error {exc.status_code}: {exc.message}")
        except groq.APIConnectionError:
            raise SystemExit("Network error reaching the Groq API.")


    def plan(self, system: str, user: str, tools: list[dict], model: str, *,
             timeout_s: float = 20.0) -> list[dict]:
        """OpenAI-style tool calls on Groq. The calls are returned, not run."""
        import groq

        try:
            client = groq.Groq(max_retries=1, timeout=timeout_s)
        except Exception as exc:
            raise SystemExit(f"Could not create the Groq client: {exc}\nSet {self.key_hint}")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=1024,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice="required",
            )
        except groq.AuthenticationError:
            raise SystemExit(f"No valid credentials. Set {self.key_hint}")
        except groq.RateLimitError:
            raise SystemExit("Groq free-tier rate limit reached. Wait and retry.")
        except groq.APIStatusError as exc:
            raise SystemExit(f"Groq API error {exc.status_code}: {exc.message}")
        except groq.APIConnectionError:
            raise SystemExit("Network error reaching the Groq API.")
        calls = []
        for call in response.choices[0].message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = {}
            calls.append({"name": call.function.name, "args": args if isinstance(args, dict) else {}})
        return calls


class NvidiaProvider:
    """NVIDIA-hosted models (NIM endpoints on build.nvidia.com).

    The endpoint speaks the OpenAI chat-completions protocol, so this is the
    Groq provider with a different base URL and the openai SDK. The same NIM
    containers run on an NVIDIA GPU on-premises with only NVIDIA_BASE_URL
    changed, which is the offline, at-sea deployment story for the assistant.
    """

    name = "nvidia"
    # Override with DEEPECHO_NVIDIA_MODEL. The catalogue at
    # https://build.nvidia.com moves; whatever it lists under "nemotron" is the
    # current family. The literal below is a long-lived one.
    default_model = os.environ.get("DEEPECHO_NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    key_hint = "NVIDIA_API_KEY. Free key: https://build.nvidia.com (API Keys)"
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    def _client(self, **kwargs):
        import openai

        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise SystemExit(f"Could not create the NVIDIA client: api_key must be set\nSet {self.key_hint}")
        return openai.OpenAI(api_key=key, base_url=self.base_url, **kwargs)

    @staticmethod
    def _translate(exc: Exception) -> SystemExit:
        import openai

        if isinstance(exc, openai.AuthenticationError):
            return SystemExit(f"No valid credentials. Set {NvidiaProvider.key_hint}")
        if isinstance(exc, openai.RateLimitError):
            return SystemExit("NVIDIA API rate limit reached. Wait and retry.")
        if isinstance(exc, openai.APIStatusError):
            return SystemExit(f"NVIDIA API error {exc.status_code}: {exc.message}")
        if isinstance(exc, openai.APIConnectionError):
            return SystemExit("Network error reaching the NVIDIA API.")
        return SystemExit(f"NVIDIA API error: {exc}")

    def complete(self, system: str, user: str, model: str, *, max_tokens: int = 4096,
                 attempts: int = 3, timeout_s: float = 120.0) -> str:
        client = self._client(max_retries=max(0, attempts - 1), timeout=timeout_s)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        return (response.choices[0].message.content or "").strip()

    def stream(self, system: str, user: str, model: str):
        """Same call, same config, delivered incrementally."""
        client = self._client()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=4096,
                stream=True,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
        except Exception as exc:
            raise self._translate(exc) from exc

    def plan(self, system: str, user: str, tools: list[dict], model: str, *,
             timeout_s: float = 20.0) -> list[dict]:
        """OpenAI-style tool calls. The calls are returned, not run."""
        client = self._client(max_retries=1, timeout=timeout_s)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=1024,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice="required",
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        calls = []
        for call in response.choices[0].message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = {}
            calls.append({"name": call.function.name, "args": args if isinstance(args, dict) else {}})
        return calls


PROVIDERS = {p.name: p for p in (GeminiProvider, GroqProvider, NvidiaProvider)}

# Per provider: the pip package to install, and the module that proves it is.
SDKS = {"gemini": ("google-genai", "google.genai"),
        "groq": ("groq", "groq"),
        "nvidia": ("openai", "openai")}


def _require_sdk(provider: str) -> None:
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
    package, module = SDKS[provider]
    if not have(module):
        raise SystemExit(f"The {package} SDK is not installed. Run `pip install {package}`.")


def plan_tools(system: str, user: str, tools: list[dict],
               provider: str = DEFAULT_PROVIDER, model: str = "", *,
               timeout_s: float = 20.0) -> list[dict]:
    """Ask a provider which data tools to call. Returns [{name, args}], runs nothing."""
    _require_sdk(provider)
    backend = PROVIDERS[provider]()
    return backend.plan(system, user, tools, model or backend.default_model, timeout_s=timeout_s)


def quick_complete(system: str, user: str, provider: str = DEFAULT_PROVIDER,
                   model: str = "", *, max_tokens: int = 512, timeout_s: float = 15.0) -> str:
    """A short, low-retry completion: translation, or a JSON tool plan."""
    _require_sdk(provider)
    backend = PROVIDERS[provider]()
    return backend.complete(system, user, model or backend.default_model,
                            max_tokens=max_tokens, attempts=2, timeout_s=timeout_s)


def generate(filled: str, hits: list[tuple[Chunk, float]],
             provider: str = DEFAULT_PROVIDER, model: str = "") -> str:
    """One call, grounded in the retrieved sources and nothing else."""
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
    backend = PROVIDERS[provider]()

    package, module = SDKS[provider]
    if not have(module):
        raise SystemExit(
            f"The {package} SDK is not installed. Run `pip install {package}`, "
            "or re-run with --no-llm to see the retrieved sources only.")

    user = f"SOURCES\n=======\n{format_sources(hits)}\n\nTASK\n====\n{filled}"
    return backend.complete(SYSTEM, user, model or backend.default_model)


def generate_stream(filled: str, hits: list[tuple[Chunk, float]],
                    provider: str = DEFAULT_PROVIDER, model: str = ""):
    """generate(), yielded in pieces. Same prompt, same rules, same sources.

    The guard below is deliberately a copy of the one in generate() rather than
    a helper factored out of it. generate() is the path the CLI has always used,
    and leaving its body untouched is worth more than saving six lines here.
    """
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}")
    backend = PROVIDERS[provider]()

    package, module = SDKS[provider]
    if not have(module):
        raise SystemExit(
            f"The {package} SDK is not installed. Run `pip install {package}`, "
            "or re-run with --no-llm to see the retrieved sources only.")

    user = f"SOURCES\n=======\n{format_sources(hits)}\n\nTASK\n====\n{filled}"
    return backend.stream(SYSTEM, user, model or backend.default_model)


# ---------------------------------------------------------------------------
# Detection records
# ---------------------------------------------------------------------------

DETECTION_FIELDS = ["label", "confidence", "timestamp", "latitude", "longitude",
                    "depth_m", "sensor", "platform", "notes"]

QUERY_EXPANSION = {
    "explain": "risk standoff distance immediate actions do not approach notify authority",
    "anomaly": "unidentified underwater object protocol unclassified hold separation",
    "report": "report contents position depth time sensor notify authority sequence",
}


def build_detection(args: argparse.Namespace) -> dict:
    """Merge a JSON detection record with any command-line overrides."""
    record = json.loads(Path(args.detection).read_text(encoding="utf-8")) if args.detection else {}
    for name in DETECTION_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            record[name] = value
    if args.describe:
        record["visual_description"] = args.describe
    return record


def render_detection(record: dict) -> str:
    """Render for the prompt. The raw embedding is summarised, never printed."""
    if not record:
        return "(no structured detection data provided)"
    lines = []
    for key, value in record.items():
        if key == "embedding" and isinstance(value, list):
            value = f"<{len(value)}-dimensional detection embedding>"
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def detection_query(record: dict, task: str) -> str:
    parts = [str(v) for k, v in record.items()
             if k in {"label", "visual_description", "notes"} and v]
    base = " ".join(parts) or "unidentified underwater object"
    return f"{base} {QUERY_EXPANSION.get(task, '')}".strip()


# ---------------------------------------------------------------------------
# Benchmark: is the approximate index actually returning the right chunks?
# ---------------------------------------------------------------------------

BENCH_QUERIES = [
    "what is the disposal procedure for unexploded ordnance",
    "how far do I stay from a moored contact mine",
    "who do I report a suspected mine to in Indian waters",
    "the classifier could not identify this object, what do I do",
    "is derelict fishing net a hazard to a towed sensor",
    "how do I tell a manufactured object from a rock on side scan",
    "the wreck is shallower than the chart says",
    "should I mark the position with a buoy",
    "can I recover the object to the deck for a closer look",
    "what goes in the report to the authority",
]


def bench(args: argparse.Namespace) -> None:
    retriever = Retriever.load()
    if retriever.mode != "dense" or not isinstance(retriever.index, FaissHNSW):
        raise SystemExit("bench compares a FAISS HNSW index against exact search. "
                         "Rebuild with: python3 rag.py index")
    exact = retriever.exact_twin()
    k = args.k

    print(f"corpus: {len(retriever.chunks)} chunks | dim: {retriever.embedder.dim} | "
          f"M={HNSW_M} efConstruction={HNSW_EF_CONSTRUCTION} | k={k} | "
          f"{len(BENCH_QUERIES)} queries\n")
    print(f"{'efSearch':>9}  {'recall@k':>9}  {'top-1 agree':>12}")
    print(f"{'-' * 9}  {'-' * 9}  {'-' * 12}")

    for ef in args.ef:
        retriever.index.index.hnsw.efSearch = ef
        recalls, top1 = [], 0
        for query in BENCH_QUERIES:
            # Encode once and hand the same vector to both indexes. With a
            # remote embedder that halves the API calls; it also makes this a
            # clean comparison of the indexes rather than of the embeddings.
            qv = retriever.encode_query(query)
            truth = [i for i, _ in exact.index.search(qv, k)]
            got = [i for i, _ in retriever.index.search(qv, k)]
            recalls.append(len(set(truth) & set(got)) / len(truth) if truth else 1.0)
            if truth and got and truth[0] == got[0]:
                top1 += 1
        mean = sum(recalls) / len(recalls)
        print(f"{ef:>9}  {mean:>9.3f}  {top1:>7}/{len(BENCH_QUERIES):<4}")

    print(f"\nDefault efSearch is {HNSW_EF_SEARCH}. Recall below 1.000 means the "
          f"approximate\nindex missed a chunk that exact search returns. On a "
          f"safety corpus, raise\nefSearch until recall is 1.000 and keep it there.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def require_catalog() -> Catalog:
    catalog = Catalog.load()
    if catalog is None:
        raise SystemExit("No catalog indexed. Run: python3 rag.py index")
    return catalog


def print_hits(hits: list[tuple[Chunk, float]]) -> None:
    print("\nSOURCES")
    for i, (chunk, score) in enumerate(hits, start=1):
        meta = chunk.meta
        flag = "  [UNVERIFIED SEED CONTENT]" if meta.get("status") == "PLACEHOLDER" else ""
        print(f"  [S{i}] {score:.3f}  {meta.get('title')} -- {meta.get('section') or 'intro'}"
              f"  ({meta.get('authority', 'unknown authority')}){flag}")


def run_query(query: str, filled: str, args: argparse.Namespace,
              matches_block: str = "") -> None:
    retriever = Retriever.load(ef_search=args.ef_search)
    hits = retriever.search(query, k=args.k, per_doc=args.per_doc)
    if not hits:
        raise SystemExit("Nothing in the knowledge base matched that query.")
    if args.no_llm:
        if matches_block:
            print(matches_block + "\n")
        print(format_sources(hits))
        return
    print(generate(filled, hits, args.provider, args.model))
    print_hits(hits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def query_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-k", type=int, default=DEFAULT_K, help="chunks to retrieve")
        p.add_argument("--per-doc", type=int, default=2,
                       help="max chunks kept from any one document")
        p.add_argument("--ef-search", type=int, default=HNSW_EF_SEARCH,
                       help="HNSW candidate list size; higher is more accurate and slower")
        p.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER,
                       help=f"LLM backend (default {DEFAULT_PROVIDER}; "
                            f"override with DEEPECHO_PROVIDER)")
        p.add_argument("--model", default="", help="override the provider's default model")
        p.add_argument("--no-llm", action="store_true",
                       help="show retrieved sources without calling the model")

    def detection_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--detection", help="path to a detection JSON record")
        for name in DETECTION_FIELDS:
            p.add_argument(f"--{name.replace('_', '-')}", dest=name,
                           type=float if name == "confidence" else str)
        p.add_argument("--describe", help="free-text description of the object")
        p.add_argument("--top", type=int, default=3,
                       help="nearest known objects to consider")

    p_index = sub.add_parser("index", help="build the vector index from kb/")
    p_index.add_argument("--kb", default=str(KB_DIR))
    p_index.add_argument("--dim", type=int, default=DEFAULT_DIM,
                         help="random-projection only")
    p_index.add_argument("--metric", choices=["cosine", "pearson"], default="cosine",
                         help="pearson centres vectors before normalising")
    p_index.add_argument(
        "--embedder",
        choices=["auto", "tfidf-dense", "gemini", "random-projection", "sentence-transformers"],
        default="auto",
        help="auto prefers sentence-transformers, else tfidf-dense; gemini calls the API")
    p_index.add_argument("--ann", choices=["auto", "hnsw", "exact"], default="auto")
    p_index.add_argument("--hnsw-m", type=int, default=HNSW_M)
    p_index.add_argument("--ef-construction", type=int, default=HNSW_EF_CONSTRUCTION)

    p_bench = sub.add_parser("bench", help="HNSW recall against exact search")
    p_bench.add_argument("-k", type=int, default=DEFAULT_K)
    p_bench.add_argument("--ef", type=int, nargs="+", default=[8, 16, 32, 64, 128])

    p_search = sub.add_parser("search", help="retrieval only, no generation")
    p_search.add_argument("query")
    query_args(p_search)

    p_ask = sub.add_parser("ask", help="answer an operator question")
    p_ask.add_argument("question")
    query_args(p_ask)

    for name, help_text in [("explain", "explain a classified detection"),
                            ("anomaly", "protocol for an unclassified object"),
                            ("report", "generate an incident report")]:
        p = sub.add_parser(name, help=help_text)
        detection_args(p)
        query_args(p)

    p_match = sub.add_parser("match", help="nearest known objects, no generation")
    detection_args(p_match)

    args = parser.parse_args()

    if args.command == "index":
        retriever = Retriever.build(
            Path(args.kb), dim=args.dim, center=args.metric == "pearson",
            embedder_kind=args.embedder, ann=args.ann,
            m=args.hnsw_m, ef_construction=args.ef_construction)
        retriever.save()
        catalog = Catalog.build()
        if catalog is not None:
            catalog.save()
            print(f"Catalog: {len(catalog.entries)} known object classes, "
                  f"matching in {catalog.space} space")
        docs = {c.meta["doc_id"] for c in retriever.chunks}
        placeholders = sorted({c.meta["doc_id"] for c in retriever.chunks
                               if c.meta.get("status") == "PLACEHOLDER"})
        detail = (f"{retriever.embedder.kind} d={retriever.embedder.dim}, {retriever.index.kind}"
                  if retriever.mode == "dense" else "sparse tf-idf, exact")
        print(f"Indexed {len(retriever.chunks)} chunks from {len(docs)} documents "
              f"({detail}, metric={args.metric})")
        if placeholders:
            print(f"WARNING: {len(placeholders)} of {len(docs)} documents are PLACEHOLDER "
                  f"seed content, not authoritative sources: {', '.join(placeholders)}")
        return

    if args.command == "bench":
        bench(args)
        return

    if args.command == "search":
        retriever = Retriever.load(ef_search=args.ef_search)
        hits = retriever.search(args.query, k=args.k, per_doc=args.per_doc)
        print(format_sources(hits) if hits else "No match.")
        return

    if args.command == "ask":
        run_query(args.question, TASKS["ask"].format(question=args.question), args)
        return

    record = build_detection(args)
    if not record:
        raise SystemExit("Provide a detection: --detection FILE, or --label / --describe.")

    if args.command == "match":
        catalog = require_catalog()
        print(format_matches(catalog.match(record, top=args.top), catalog.space))
        return

    query = detection_query(record, args.command)
    if args.command == "anomaly":
        catalog = Catalog.load()
        if catalog is None:
            matches_block = ("NEAREST KNOWN OBJECTS: unavailable. No catalog is indexed, "
                             "so no comparison was made. Do not speculate about identity.")
        else:
            matches_block = format_matches(catalog.match(record, top=args.top), catalog.space)
        filled = TASKS["anomaly"].format(detection=render_detection(record),
                                         matches=matches_block)
        run_query(query, filled, args, matches_block)
        return

    run_query(query, TASKS[args.command].format(detection=render_detection(record)), args)


if __name__ == "__main__":
    main()
