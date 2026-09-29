"""Évaluation du retrieval au niveau *notice* : Hit@k, Recall@k, MRR@10, nDCG@10.

Chaque retriever renvoie des chunks ; on les ramène aux notices (dédoublonnage par
meilleur rang) avant de calculer les métriques. IC à 95 % par bootstrap et tests
appariés entre retrievers.

Usage : python -m eval.eval_retrieval [--retrievers bm25 dense hybrid solr]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

from merimee_rag.config import Config
from merimee_rag.retrievers import Retriever, dedupe_refs

from .metrics import bootstrap_ci, hit_at_k, ndcg_at_k, paired_bootstrap_pvalue, reciprocal_rank, recall_at_k

KS = (1, 3, 5, 10, 20)


def load_testset(path: Path, answerable_only: bool = True) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    return [x for x in items if x.get("gold_refs")] if answerable_only else items


def evaluate(retrievers: dict[str, Retriever], testset: list[dict], depth: int = 50) -> pd.DataFrame:
    rows = []
    for name, r in retrievers.items():
        t0 = time.perf_counter()
        for q in testset:
            gold = set(q["gold_refs"])
            ranked = dedupe_refs(r.search(q["question"], depth), k=max(KS))
            row = {"retriever": name, "id": q["id"], "lexical_overlap": q.get("lexical_overlap"),
                   "ambiguous": bool(q.get("ambiguous", False)),
                   "mrr@10": reciprocal_rank(ranked, gold, 10), "ndcg@10": ndcg_at_k(ranked, gold, 10)}
            for k in KS:
                row[f"hit@{k}"] = hit_at_k(ranked, gold, k)
                row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
            first = next((i for i, x in enumerate(ranked, 1) if x in gold), None)
            row["gold_rank"] = first
            rows.append(row)
        dt = (time.perf_counter() - t0) / max(1, len(testset))
        print(f"  {name:8s} {dt * 1000:7.1f} ms/requête")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (sous-ensemble, retriever). « propres » = sans les questions sous-spécifiées."""
    metrics = ["hit@1", "hit@5", "recall@10", "mrr@10", "ndcg@10"]
    subsets = [("toutes", df)]
    if "ambiguous" in df and df["ambiguous"].any():
        subsets.append(("propres", df[~df["ambiguous"].astype(bool)]))
    out = []
    for subset, d in subsets:
        for name, g in d.groupby("retriever", sort=False):
            row = {"subset": subset, "retriever": name, "n": len(g)}
            for m in metrics:
                mean, lo, hi = bootstrap_ci(g[m].tolist())
                row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
            out.append(row)
    return pd.DataFrame(out)


def pairwise(df: pd.DataFrame, metric: str = "mrr@10") -> pd.DataFrame:
    rows = []
    subsets = [("toutes", df)]
    if "ambiguous" in df and df["ambiguous"].any():
        subsets.append(("propres", df[~df["ambiguous"].astype(bool)]))
    for subset, d in subsets:
        rows += _pairwise(d, metric, subset)
    return pd.DataFrame(rows)


def _pairwise(df: pd.DataFrame, metric: str, subset: str) -> list[dict]:
    piv = df.pivot(index="id", columns="retriever", values=metric)
    rows = []
    for a, b in itertools.combinations(piv.columns, 2):
        rows.append({"subset": subset, "A": a, "B": b, "metric": metric, "delta(A-B)": piv[a].mean() - piv[b].mean(),
                     "p_value": paired_bootstrap_pvalue(piv[a].values, piv[b].values)})
    return rows


def by_overlap(df: pd.DataFrame, metric: str = "mrr@10") -> pd.DataFrame:
    """Performance selon le recouvrement lexical question/notice (biais pro-BM25 ?)."""
    if df["lexical_overlap"].isna().all():
        return pd.DataFrame()
    d = df.copy()
    d["overlap_bin"] = pd.qcut(d["lexical_overlap"].rank(method="first"), 3, labels=["faible", "moyen", "fort"])
    return d.pivot_table(index="overlap_bin", columns="retriever", values=metric, aggfunc="mean", observed=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=cfg.eval_dir / "testset.jsonl")
    ap.add_argument("--retrievers", nargs="+", default=None)
    args = ap.parse_args()

    from merimee_rag.store import System

    system = System(cfg)
    names = args.retrievers or list(system.retrievers)
    testset = load_testset(args.testset)
    print(f"{len(testset)} questions, retrievers : {names}")
    df = evaluate({n: system.retrievers[n] for n in names}, testset)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "retrieval_per_query.csv", index=False)
    summ = summarize(df)
    summ.to_csv(cfg.results_dir / "retrieval_summary.csv", index=False)
    pw = pairwise(df)
    pw.to_csv(cfg.results_dir / "retrieval_pairwise.csv", index=False)
    ov = by_overlap(df)
    if not ov.empty:
        ov.to_csv(cfg.results_dir / "retrieval_by_overlap.csv")

    cols = ["subset", "retriever", "n", "hit@1", "hit@5", "recall@10", "mrr@10", "ndcg@10"]
    print("\n" + summ[cols].round(3).to_string(index=False))
    print("\n" + pw.round(4).to_string(index=False))
    if not ov.empty:
        print("\nMRR@10 par recouvrement lexical :\n" + ov.round(3).to_string())


if __name__ == "__main__":
    main()
