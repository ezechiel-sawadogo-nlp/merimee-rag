"""Compare le RAG fixe (recherche hybride + 5 extraits) et l'agent sur eval/agent_testset.jsonl.

Mesures par type de question : exactitude (notation automatique ; juge LLM pour lookup),
et pour l'agent : bon choix d'outil, nombre d'appels, appels en erreur, latence, étapes épuisées,
plus un diagnostic de chaque échec (outil, arguments, synthèse…).

Par défaut les deux systèmes utilisent le MÊME modèle (qwen2.5:7b) : seule l'architecture change.

Usage : python -m eval.eval_agent [--limit-per-type 5] [--systems rag agent]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from merimee_rag.agent import Agent
from merimee_rag.config import Config
from merimee_rag.llm import DiskCache, OllamaLLM
from merimee_rag.pipeline import RAGPipeline

from .agent_scoring import diagnose, extra_filters, gold_seen, score, tool_ok
from .judge import judge_correctness
from .metrics import bootstrap_ci

TYPES = ["count", "list", "argmax", "multistep", "lookup", "ooc"]


def run(questions: list[dict], systems: dict, judge, titles: dict[str, str], log=print) -> list[dict]:
    recs = []
    for sys_name, runner in systems.items():
        for i, q in enumerate(questions, 1):
            res = runner.answer(q["question"])
            rec = {"system": sys_name, "id": q["id"], "type": q["type"], "question": q["question"],
                   "answer": res.answer, "status": res.status, "latency_s": res.latency_s}
            if sys_name == "agent":
                d = res.to_dict()
                rec.update(steps=d["steps"], n_tool_calls=len(d["steps"]), hit_max_steps=d["hit_max_steps"],
                           n_tool_errors=sum(s["error"] for s in d["steps"]),
                           tools=[s["tool"] for s in d["steps"]],
                           unsupported_refs=d["unsupported_refs"], n_unsupported_refs=len(d["unsupported_refs"]),
                           extra_filters=extra_filters(q, d["steps"]))
                rec["has_extra_filters"] = bool(rec["extra_filters"])
            else:
                rec.update(retrieved_refs=getattr(res, "retrieved_refs", []))
            s = score(q, res.answer, res.abstained, titles)
            if q["type"] == "lookup" and not res.abstained:
                s["score"], s["judge_note"] = judge_correctness(judge, q["question"], q["gold"], res.answer)
            rec.update(s)
            if sys_name == "agent":
                rec["tool_ok"] = tool_ok(q, rec["steps"])
                rec["gold_in_tools"] = gold_seen(q, rec["steps"])
                rec["diagnostic"] = diagnose(q, rec)
            recs.append(rec)
            log(f"\r  {sys_name:6s} {i}/{len(questions)}", end="", flush=True)
        log("")
    return recs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sys_name, t), g in df.groupby(["system", "type"], sort=False):
        m, lo, hi = bootstrap_ci(pd.to_numeric(g["score"], errors="coerce").dropna().tolist())
        row = {"system": sys_name, "type": t, "n": len(g), "score": m, "score_lo": lo, "score_hi": hi,
               "latency_s": g["latency_s"].mean()}
        if sys_name == "agent":
            row.update(tool_ok=pd.to_numeric(g["tool_ok"], errors="coerce").mean(),
                       n_tool_calls=g["n_tool_calls"].mean(),
                       tool_error_rate=g["n_tool_errors"].sum() / max(1, g["n_tool_calls"].sum()),
                       max_steps_rate=g["hit_max_steps"].mean(),
                       invented_citation_rate=(g["n_unsupported_refs"] > 0).mean(),
                       extra_filter_rate=g["has_extra_filters"].mean())
        rows.append(row)
    for sys_name, g in df.groupby("system", sort=False):
        m, lo, hi = bootstrap_ci(pd.to_numeric(g["score"], errors="coerce").dropna().tolist())
        rows.append({"system": sys_name, "type": "TOTAL", "n": len(g), "score": m, "score_lo": lo,
                     "score_hi": hi, "latency_s": g["latency_s"].mean(),
                     **({"n_tool_calls": g["n_tool_calls"].mean(),
                         "tool_ok": pd.to_numeric(g["tool_ok"], errors="coerce").mean()}
                        if sys_name == "agent" else {})})
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=cfg.eval_dir / "agent_testset.jsonl")
    ap.add_argument("--systems", nargs="+", default=["rag", "agent"])
    ap.add_argument("--limit-per-type", type=int, default=None)
    ap.add_argument("--model", default=None, help="modèle des deux systèmes (défaut : MERIMEE_AGENT_MODEL)")
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    from merimee_rag.store import System, build_toolbox

    system = System(cfg)
    toolbox = build_toolbox(system)
    cache = DiskCache(cfg.llm_cache)
    model = args.model or cfg.agent_model
    llm = OllamaLLM(model, cfg.ollama_url, cfg.temperature, cache)
    judge = OllamaLLM(args.judge_model or cfg.judge_model, cfg.ollama_url, 0.0, cache)
    runners = {"rag": RAGPipeline(system.retrievers["hybrid"], system.chunks, system.notices, llm,
                                  cfg.top_k_context),
               "agent": Agent(llm, toolbox, cfg.agent_max_steps)}
    runners = {k: v for k, v in runners.items() if k in args.systems}

    questions = [json.loads(l) for l in open(args.testset, encoding="utf-8") if l.strip()]
    if args.limit_per_type:
        per = defaultdict(int)
        questions = [q for q in questions if (per.__setitem__(q["type"], per[q["type"]] + 1)
                                              or per[q["type"]] <= args.limit_per_type)]
    titles = {r.ref: r.titre for r in toolbox.store.records}
    print(f"{len(questions)} questions × {list(runners)} — modèle {model}, juge {judge.model}")
    recs = run(questions, runners, judge, titles)

    R = cfg.results_dir
    R.mkdir(parents=True, exist_ok=True)
    with open(R / "agent_per_query.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    df = pd.DataFrame(recs)
    summ = summarize(df)
    summ.to_csv(R / "agent_summary.csv", index=False)
    if "agent" in runners:
        diag = df[df.system == "agent"].pivot_table(index="type", columns="diagnostic", values="id",
                                                    aggfunc="count", fill_value=0)
        diag.to_csv(R / "agent_errors.csv")
        print("\nDiagnostic de l'agent :\n" + diag.to_string())
    piv = summ.pivot(index="type", columns="system", values="score").reindex(TYPES + ["TOTAL"])
    print("\nExactitude par type :\n" + piv.round(3).to_string())


if __name__ == "__main__":
    main()
