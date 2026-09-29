"""Point d'entrée : `merimee-rag <commande>`.

  download   télécharge le CSV Mérimée (~95 Mo)
  inspect    affiche les colonnes détectées du CSV
  prepare    filtre les notices avec historique → data/corpus.jsonl
  index      chunks + index BM25 + embeddings
  ask        pose une question au RAG
"""
from __future__ import annotations

import argparse
import csv
import sys

from .config import Config


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # accents dans la console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="merimee-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("inspect")
    pp = sub.add_parser("prepare")
    pp.add_argument("--input", default=None, help="CSV Mérimée brut ou JSON nettoyé (défaut : data/raw/monuments.json, "
                    "sinon data/monuments.json.gz livré avec le dépôt, sinon data/raw/merimee.csv)")
    pi = sub.add_parser("index")
    pi.add_argument("--no-dense", action="store_true", help="BM25 seulement (rapide)")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid", "solr", "none"])
    pa.add_argument("--model", default=None)
    args = p.parse_args(argv)
    cfg = Config()

    if args.cmd == "download":
        from .store import download

        print(f"Téléchargement → {cfg.raw_csv}")
        download(cfg)

    elif args.cmd == "inspect":
        from .corpus import detect_encoding, resolve_columns, sniff_delimiter

        enc = detect_encoding(cfg.raw_csv)
        delim = sniff_delimiter(cfg.raw_csv, enc)
        with open(cfg.raw_csv, encoding=enc, newline="") as f:
            cols = next(csv.reader(f, delimiter=delim))
        print(f"encodage={enc}  séparateur={delim!r}  {len(cols)} colonnes")
        for c in cols:
            print("  -", c)
        print("\nMapping retenu :")
        for k, v in resolve_columns(cols).items():
            print(f"  {k:13s} ← {v}")

    elif args.cmd == "prepare":
        from pathlib import Path

        from .corpus import iter_notices, save_corpus

        bundled = cfg.raw_json.parent.parent / "monuments.json.gz"  # corpus livré avec le dépôt
        candidates = [cfg.raw_json, bundled, cfg.raw_csv]
        src = Path(args.input) if args.input else next((c for c in candidates if c.exists()), cfg.raw_csv)
        print(f"Source : {src}")
        n = save_corpus(iter_notices(src, cfg.min_historique_chars), cfg.corpus_path)
        print(f"{n} notices avec historique (≥ {cfg.min_historique_chars} caractères) → {cfg.corpus_path}")

    elif args.cmd == "index":
        from .store import build_indexes

        print(build_indexes(cfg, dense=not args.no_dense))

    elif args.cmd == "ask":
        from .llm import DiskCache, OllamaLLM
        from .pipeline import RAGPipeline
        from .store import System

        system = System(cfg, load_dense=args.retriever in ("dense", "hybrid"))
        llm = OllamaLLM(args.model or cfg.gen_model, cfg.ollama_url, cfg.temperature, DiskCache(cfg.llm_cache))
        retr = None if args.retriever == "none" else system.retrievers[args.retriever]
        rag = RAGPipeline(retr, system.chunks, system.notices, llm, cfg.top_k_context)
        res = rag.answer(args.question)
        print("\n" + res.answer + "\n")
        for c in res.contexts:
            mark = "✓" if c.n in res.cited else " "
            print(f" {mark} [{c.n}] {c.title} — {c.url}")
        print(f"\n({res.latency_s:.1f} s)")


if __name__ == "__main__":
    main()
