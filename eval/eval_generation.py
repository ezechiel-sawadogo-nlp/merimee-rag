"""Évaluation de bout en bout de la génération, pour plusieurs configurations :

  no-rag       LLM seul (baseline : que « sait » le modèle sans la base ?)
  rag-bm25     RAG avec BM25
  rag-dense    RAG avec embeddings
  rag-hybrid   RAG avec fusion RRF

Métriques (questions répondables) :
  correctness        exactitude vs réponse de référence (LLM-juge : 1 / 0.5 / 0)
  grounded_gold      part des affirmations soutenues par la notice source → mesure d'hallucination
                     comparable entre toutes les configs, y compris no-rag
  faithful_ctx       part des affirmations soutenues par les extraits fournis (RAG seulement)
  citation_hit       la notice source fait partie des extraits cités
  citation_valid     les numéros cités existent (pas de [7] quand on a fourni 5 extraits)
  false_abstention   le système refuse alors que la réponse était dans la base
Questions hors corpus :
  correct_abstention le système refuse de répondre, et seulement ça (comportement attendu)
  partial            (toutes questions) refus + affirmations mélangés : jugé comme une réponse,
                     jamais compté comme une abstention réussie

Analyse d'erreurs : chaque échec est attribué au retrieval (notice source non
récupérée) ou à la génération (notice récupérée mais réponse fausse).

Usage : python -m eval.eval_generation --limit 50 [--configs no-rag rag-hybrid]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from merimee_rag.config import Config
from merimee_rag.llm import LLM, DiskCache, OllamaLLM
from merimee_rag.pipeline import RAGPipeline

from .judge import judge_correctness, judge_faithfulness
from .metrics import bootstrap_ci

ALL_CONFIGS = ["no-rag", "rag-bm25", "rag-dense", "rag-hybrid"]


def score_one(rag: RAGPipeline, judge: LLM, q: dict, gold_texts: dict[str, str]) -> dict:
    res = rag.answer(q["question"])
    rec = {"config": rag.name, "id": q["id"], "type": q["type"], "question": q["question"],
           "ambiguous": bool(q.get("ambiguous", False)),
           "reference": q.get("answer"), "answer": res.answer, "status": res.status,
           "abstained": res.abstained, "partial": float(res.status == "partial"),
           "retrieved_refs": res.retrieved_refs, "cited_refs": res.cited_refs,
           # une réponse sortie du cache n'a pas de latence significative
           "latency_s": None if getattr(rag.llm, "last_from_cache", False) else res.latency_s}

    if q["type"] == "out_of_corpus":
        # sans RAG, la consigne ne demande pas de s'abstenir : la mesure n'aurait pas de sens
        rec["correct_abstention"] = float(res.abstained) if rag.retriever is not None else None
        return rec

    gold = set(q["gold_refs"])
    rec["false_abstention"] = float(res.abstained)
    is_rag = rag.retriever is not None
    if is_rag:
        rec["gold_retrieved"] = float(bool(gold & set(res.retrieved_refs)))
        rec["citation_hit"] = float(bool(gold & set(res.cited_refs)))
        n_ctx = len(res.contexts)
        rec["citation_valid"] = (float(bool(res.cited) and all(1 <= n <= n_ctx for n in res.cited))
                                 if not res.abstained else None)

    if res.abstained:
        rec["correctness"], rec["correctness_note"] = 0.0, "abstention"
        rec["grounded_gold"] = rec["faithful_ctx"] = None
    else:
        rec["correctness"], rec["correctness_note"] = judge_correctness(judge, q["question"], q["answer"], res.answer)
        source = "\n\n".join(gold_texts[r] for r in gold if r in gold_texts)
        rec["grounded_gold"], rec["claims_gold"] = judge_faithfulness(judge, source, res.answer)
        if is_rag:
            rec["faithful_ctx"], _ = judge_faithfulness(judge, rag.format_context(res.contexts), res.answer)

    if is_rag:
        if rec["correctness"] is not None and rec["correctness"] >= 1.0:
            rec["outcome"] = "correct"
        elif not rec["gold_retrieved"]:
            rec["outcome"] = "échec retrieval"
        elif res.abstained:
            rec["outcome"] = "abstention à tort"
        else:
            rec["outcome"] = "échec génération"
    return rec


METRICS = ["correctness", "grounded_gold", "faithful_ctx", "citation_hit", "citation_valid",
           "false_abstention", "correct_abstention", "partial", "latency_s"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (sous-ensemble, config). Le sous-ensemble « propres » retire les questions
    répondables sous-spécifiées (les questions hors corpus sont gardées dans les deux)."""
    subsets = [("toutes", df)]
    if "ambiguous" in df and df["ambiguous"].fillna(False).astype(bool).any():
        subsets.append(("propres", df[~df["ambiguous"].fillna(False).astype(bool)]))
    rows = []
    for subset, d in subsets:
        for cfg_name, g in d.groupby("config", sort=False):
            row = {"subset": subset, "config": cfg_name,
                   "n_answerable": int((g["type"] != "out_of_corpus").sum()),
                   "n_ooc": int((g["type"] == "out_of_corpus").sum())}
            for m in METRICS:
                if m in g:
                    vals = pd.to_numeric(g[m], errors="coerce").dropna().tolist()
                    mean, lo, hi = bootstrap_ci(vals) if vals else (float("nan"),) * 3
                    row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=cfg.eval_dir / "testset.jsonl")
    ap.add_argument("--configs", nargs="+", default=ALL_CONFIGS)
    ap.add_argument("--limit", type=int, default=50, help="nb de questions répondables (échantillon)")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    from merimee_rag.store import System

    system = System(cfg, load_dense=any(c in args.configs for c in ("rag-dense", "rag-hybrid")))
    cache = DiskCache(cfg.llm_cache)
    gen = OllamaLLM(args.gen_model or cfg.gen_model, cfg.ollama_url, cfg.temperature, cache)
    judge = OllamaLLM(args.judge_model or cfg.judge_model, cfg.ollama_url, 0.0, cache)
    gold_texts = {n.ref: f"{n.header()}\n{n.historique}" for n in system.notices}

    with open(args.testset, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    answerable = [x for x in items if x.get("gold_refs")]
    random.Random(cfg.seed).shuffle(answerable)
    questions = answerable[: args.limit] + [x for x in items if x["type"] == "out_of_corpus"]
    print(f"{len(questions)} questions × {len(args.configs)} configs — générateur {gen.model}, juge {judge.model}")

    records = []
    for c in args.configs:
        retr = None if c == "no-rag" else system.retrievers[c.removeprefix("rag-")]
        rag = RAGPipeline(retr, system.chunks, system.notices, gen, cfg.top_k_context)
        for i, q in enumerate(questions, 1):
            records.append(score_one(rag, judge, q, gold_texts))
            print(f"\r  {c:11s} {i}/{len(questions)}", end="", flush=True)
        print()

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_dir / "generation_per_query.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    df = pd.DataFrame(records)
    summ = summarize(df)
    summ.to_csv(cfg.results_dir / "generation_summary.csv", index=False)
    if "outcome" in df:
        errs = df.dropna(subset=["outcome"]).pivot_table(index="config", columns="outcome", values="id",
                                                          aggfunc="count", fill_value=0)
        errs.to_csv(cfg.results_dir / "generation_errors.csv")
        print("\nAnalyse d'erreurs :\n" + errs.to_string())
    cols = [c for c in ["subset", "config", "correctness", "grounded_gold", "faithful_ctx", "citation_hit",
                        "false_abstention", "correct_abstention"] if c in summ]
    print("\n" + summ[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
