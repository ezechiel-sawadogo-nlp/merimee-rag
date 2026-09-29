"""Recalcule tous les résumés à partir des résultats par question, SANS rappeler aucun LLM.

Utile quand on change une règle d'analyse après coup :
  - signalement des questions sous-spécifiées (eval/ambiguity.py) ;
  - détection des refus (merimee_rag.pipeline.answer_status) ;
  - abstention hors corpus non mesurée pour « no-rag » (sa consigne ne la demande pas).

Usage : python -m eval.rescore   (puis python -m eval.report)
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from merimee_rag.config import Config
from merimee_rag.corpus import load_corpus
from merimee_rag.pipeline import answer_status

from . import eval_generation, eval_retrieval
from .ambiguity import is_ambiguous


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    R, E = cfg.results_dir, cfg.eval_dir
    notices = {n.ref: n for n in load_corpus(cfg.corpus_path)}

    # 1. Jeu de test : signalement des questions sous-spécifiées
    items = [json.loads(l) for l in open(E / "testset.jsonl", encoding="utf-8") if l.strip()]
    for it in items:
        if it.get("gold_refs"):
            nt = notices[it["gold_refs"][0]]
            it["ambiguous"] = is_ambiguous(it["question"], nt.title, nt.commune)
    with open(E / "testset.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    amb = {it["id"] for it in items if it.get("ambiguous")}
    print(f"{len(amb)} questions sous-spécifiées sur {sum(1 for i in items if i.get('gold_refs'))}")

    # 2. Retrieval
    if (R / "retrieval_per_query.csv").exists():
        df = pd.read_csv(R / "retrieval_per_query.csv")
        df["ambiguous"] = df["id"].isin(amb)
        df.to_csv(R / "retrieval_per_query.csv", index=False)
        s = eval_retrieval.summarize(df)
        s.to_csv(R / "retrieval_summary.csv", index=False)
        eval_retrieval.pairwise(df).to_csv(R / "retrieval_pairwise.csv", index=False)
        print(s[["subset", "retriever", "n", "hit@1", "hit@5", "mrr@10"]].round(3).to_string(index=False))

    # 3. Génération : statut de chaque réponse recalculé avec la détection de refus actuelle
    gp = R / "generation_per_query.jsonl"
    if gp.exists():
        recs = [json.loads(l) for l in open(gp, encoding="utf-8") if l.strip()]
        changed = 0
        for r in recs:
            r["ambiguous"] = r["id"] in amb
            status = answer_status(r["answer"])
            if status != r.get("status"):
                changed += 1
            r["status"], r["abstained"] = status, status == "abstention"
            r["partial"] = float(status == "partial")
            is_rag = r["config"] != "no-rag"
            if r["type"] == "out_of_corpus":
                r["correct_abstention"] = float(r["abstained"]) if is_rag else None
                continue
            r["false_abstention"] = float(r["abstained"])
            if r["abstained"]:
                r["correctness"], r["grounded_gold"], r["faithful_ctx"] = 0.0, None, None
            if is_rag:
                if r.get("correctness") is not None and r["correctness"] >= 1.0:
                    r["outcome"] = "correct"
                elif not r.get("gold_retrieved"):
                    r["outcome"] = "échec retrieval"
                elif r["abstained"]:
                    r["outcome"] = "abstention à tort"
                else:
                    r["outcome"] = "échec génération"
        with open(gp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"{changed} réponses dont le statut change avec la détection de refus actuelle")
        g = pd.DataFrame(recs)
        s = eval_generation.summarize(g)
        s.to_csv(R / "generation_summary.csv", index=False)
        errs = (g.dropna(subset=["outcome"]).pivot_table(index="config", columns="outcome", values="id",
                                                          aggfunc="count", fill_value=0))
        errs.to_csv(R / "generation_errors.csv")
        cols = [c for c in ["subset", "config", "correctness", "grounded_gold", "citation_hit",
                            "correct_abstention"] if c in s]
        print(s[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
