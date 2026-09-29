"""Construction / chargement des index sur disque."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .corpus import Chunk, Notice, build_chunks, load_corpus
from .retrievers import (BM25Retriever, DenseRetriever, Encoder, HybridRetriever, Retriever,
                         SolrRetriever, sentence_transformer_encoder)

MERIMEE_CSV_URL = "https://ministere-culture.s3.sbg.io.cloud.ovh.net/POP/merimee.csv"


def download(cfg: Config, url: str = MERIMEE_CSV_URL) -> Path:
    import requests

    cfg.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.raw_csv.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} Mo", end="", flush=True)
    print()
    tmp.replace(cfg.raw_csv)
    return cfg.raw_csv


def chunks_path(cfg: Config) -> Path:
    return cfg.index_dir / "chunks.jsonl"


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(l)) for l in f if l.strip()]


def build_indexes(cfg: Config, encoder: Encoder | None = None, dense: bool = True) -> dict:
    notices = load_corpus(cfg.corpus_path)
    chunks = build_chunks(notices, cfg.chunk_words, cfg.chunk_overlap)
    save_chunks(chunks, chunks_path(cfg))
    print(f"  {len(notices)} notices → {len(chunks)} chunks")
    print("  index BM25…")
    BM25Retriever(chunks).save(cfg.index_dir / "bm25.pkl")
    if dense:
        print(f"  embeddings ({cfg.embed_model})… (CPU : compter ~10-20 min)")
        enc = encoder or sentence_transformer_encoder(cfg.embed_model, cfg.embed_batch_size)
        DenseRetriever.build(chunks, enc).save(cfg.index_dir / "dense")
    meta = {"n_notices": len(notices), "n_chunks": len(chunks), "embed_model": cfg.embed_model,
            "chunk_words": cfg.chunk_words, "chunk_overlap": cfg.chunk_overlap}
    (cfg.index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class System:
    """Tout ce qu'il faut pour interroger : notices, chunks, retrievers."""

    def __init__(self, cfg: Config, encoder: Encoder | None = None, load_dense: bool = True) -> None:
        self.cfg = cfg
        self.notices: list[Notice] = load_corpus(cfg.corpus_path)
        self.chunks: list[Chunk] = load_chunks(chunks_path(cfg))
        self.retrievers: dict[str, Retriever] = {"bm25": BM25Retriever.load(cfg.index_dir / "bm25.pkl")}
        dense_dir = cfg.index_dir / "dense"
        if load_dense and dense_dir.exists():
            enc = encoder or sentence_transformer_encoder(cfg.embed_model, cfg.embed_batch_size)
            self.retrievers["dense"] = DenseRetriever.load(dense_dir, enc)
            self.retrievers["hybrid"] = HybridRetriever(
                [self.retrievers["bm25"], self.retrievers["dense"]], cfg.rrf_k, cfg.candidates)
        if cfg.solr_url:
            self.retrievers["solr"] = SolrRetriever(cfg.solr_url)
