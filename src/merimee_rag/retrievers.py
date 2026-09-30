"""Retrievers : BM25 (lexical), dense (embeddings), hybride (RRF), Solr (optionnel).

Tous exposent `search(query, k) -> list[Hit]` au niveau *chunk*.
"""
from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .corpus import Chunk


@dataclass
class Hit:
    chunk_id: str
    ref: str
    score: float
    rank: int


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


# ----------------------------------------------------------------------------- BM25
FR_STOPWORDS = set("""
a à au aux avec ce ces c ça dans de des du d elle elles en et eux il ils je j l la le les leur leurs lui ma mais
me même mes moi mon ne n nos notre nous on ou où par pas pour qu que qui s sa se ses son sur ta te tes toi ton tu
un une vos votre vous y est sont été était étaient fut furent être a ont avait avaient cette cet quel quelle quels
quelles quand comment pourquoi combien dont lors sous entre vers chez depuis puis ainsi aussi très plus
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


class FrenchAnalyzer:
    """minuscules → stopwords → suppression des accents → stemming Snowball."""

    def __init__(self) -> None:
        import snowballstemmer

        self.stemmer = snowballstemmer.stemmer("french")
        self.stop = {_strip_accents(w) for w in FR_STOPWORDS}

    def __call__(self, text: str) -> list[str]:
        text = text.lower().replace("’", "'")
        text = re.sub(r"\b[cdjlmnst]'", " ", text)  # élisions : l'église -> église
        toks = _TOKEN.findall(_strip_accents(text))
        toks = [t for t in toks if t not in self.stop and (len(t) > 1 or t.isdigit())]
        return self.stemmer.stemWords(toks)


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        from rank_bm25 import BM25Okapi

        self.analyzer = FrenchAnalyzer()
        self.ids = [c.chunk_id for c in chunks]
        self.refs = [c.ref for c in chunks]
        self.bm25 = BM25Okapi([self.analyzer(c.text) for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.analyzer(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        top = [i for i in _topk(scores, k) if scores[i] > 0]  # un score nul = aucun mot en commun
        return [Hit(self.ids[i], self.refs[i], float(scores[i]), r) for r, i in enumerate(top)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):  # le stemmer Snowball n'est pas picklable
        d = self.__dict__.copy()
        d.pop("analyzer", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.analyzer = FrenchAnalyzer()


# ----------------------------------------------------------------------------- dense
Encoder = Callable[[list[str], bool], np.ndarray]  # (textes, is_query) -> matrice normalisée


def sentence_transformer_encoder(model_name: str, batch_size: int = 64) -> Encoder:
    """Encodeur sentence-transformers. Gère les préfixes E5 (« query: » / « passage: »)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    is_e5 = "e5" in model_name.lower()

    def encode(texts: list[str], is_query: bool) -> np.ndarray:
        if is_e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=len(texts) > 1000, convert_to_numpy=True).astype(np.float32)

    return encode


class DenseRetriever:
    """Recherche exacte par produit scalaire (numpy). ~27k chunks × 384 dims ≈ 40 Mo : pas besoin de FAISS."""

    name = "dense"

    def __init__(self, ids: list[str], refs: list[str], matrix: np.ndarray, encoder: Encoder) -> None:
        self.ids, self.refs, self.matrix, self.encoder = ids, refs, matrix, encoder

    @classmethod
    def build(cls, chunks: Sequence[Chunk], encoder: Encoder) -> "DenseRetriever":
        mat = encoder([c.text for c in chunks], False)
        return cls([c.chunk_id for c in chunks], [c.ref for c in chunks], mat, encoder)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.encoder([query], True)[0]
        scores = self.matrix @ q
        top = _topk(scores, k)
        return [Hit(self.ids[i], self.refs[i], float(scores[i]), r) for r, i in enumerate(top)]

    def save(self, dir_: Path) -> None:
        dir_.mkdir(parents=True, exist_ok=True)
        np.save(dir_ / "embeddings.npy", self.matrix)
        (dir_ / "ids.json").write_text(json.dumps({"ids": self.ids, "refs": self.refs}), encoding="utf-8")

    @classmethod
    def load(cls, dir_: Path, encoder: Encoder) -> "DenseRetriever":
        meta = json.loads((dir_ / "ids.json").read_text(encoding="utf-8"))
        return cls(meta["ids"], meta["refs"], np.load(dir_ / "embeddings.npy"), encoder)


# ----------------------------------------------------------------------------- hybride
class HybridRetriever:
    """Reciprocal Rank Fusion : score(d) = Σ_r 1 / (k + rang_r(d)).

    RRF ne dépend que des rangs, donc pas besoin de normaliser des scores BM25 et
    cosinus qui vivent sur des échelles incomparables.
    """

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, candidates: int = 100) -> None:
        self.retrievers, self.rrf_k, self.candidates = list(retrievers), rrf_k, candidates

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[str, float] = {}
        refs: dict[str, str] = {}
        for r in self.retrievers:
            for h in r.search(query, self.candidates):
                fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (self.rrf_k + h.rank + 1)
                refs[h.chunk_id] = h.ref
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(cid, refs[cid], s, i) for i, (cid, s) in enumerate(ranked)]


# ----------------------------------------------------------------------------- Solr
class SolrRetriever:
    """Interroge l'index Solr du projet de cours (edismax). Niveau *notice* : chunk_id = ref.

    Le texte des passages est ensuite récupéré dans le corpus local via la ref.
    """

    name = "solr"

    def __init__(self, url: str, qf: str = "titre^3 commune^2 denomination^2 historique",
                 ref_field: str = "ref") -> None:
        self.url, self.qf, self.ref_field = url.rstrip("/"), qf, ref_field

    def search(self, query: str, k: int = 10) -> list[Hit]:
        import requests

        params = {"q": query, "defType": "edismax", "qf": self.qf, "rows": k,
                  "fl": f"{self.ref_field},score", "wt": "json", "q.op": "OR"}
        docs = requests.get(f"{self.url}/select", params=params, timeout=30).json()["response"]["docs"]
        out = []
        for i, d in enumerate(docs):
            ref = d[self.ref_field]
            ref = ref[0] if isinstance(ref, list) else ref
            out.append(Hit(f"{ref}#0", ref, float(d.get("score", 0)), i))
        return out


# ----------------------------------------------------------------------------- utils
def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def dedupe_refs(hits: Sequence[Hit], k: int | None = None) -> list[str]:
    """Passe du niveau chunk au niveau notice (première occurrence = meilleur rang)."""
    seen, out = set(), []
    for h in hits:
        if h.ref not in seen:
            seen.add(h.ref)
            out.append(h.ref)
            if k and len(out) >= k:
                break
    return out
