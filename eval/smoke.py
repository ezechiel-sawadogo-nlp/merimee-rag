"""Test de non-régression rapide : quelques questions dont on connaît la bonne réponse.

À lancer après chaque modification des consignes ou du modèle (~1-3 min) :
    python -m eval.smoke
    python -m eval.smoke --retriever bm25 --model qwen2.5:7b

Chaque cas vérifie un mot-clé attendu dans la réponse (ou une abstention), ET l'absence de
mots-clés piégeux (ex. répondre avec le château de l'Ardoise pour Chambord).
"""
from __future__ import annotations

import argparse
import re
import sys

from merimee_rag.config import Config
from merimee_rag.llm import DiskCache, OllamaLLM
from merimee_rag.pipeline import RAGPipeline

# (question, motif attendu | "ABSTENTION", motif interdit | None) — faits vérifiés dans les notices.
CASES = [
    ("Qui a fait construire le château de Chambord ?", r"Fran[cç]ois", r"Dussier|Fesset|Ardoise"),
    ("Quel architecte a dessiné la flèche du clocher de l'église d'Avirey-Lingey ?", r"Lorillard", None),
    ("De quoi le moulin à eau de Brienne-la-Vieille a-t-il été équipé ?", r"turbine", None),
    ("Pour qui le château d'Écouen a-t-il été construit ?", r"Montmorency", None),
    ("Qui est l'architecte du château d'Écouen ?", r"Bullant", None),
    ("Quand la mosquée de Missiri, à Fréjus, a-t-elle été construite ?", r"19[23]0", None),
    ("Quand la Grande Mosquée de Djenné a-t-elle été reconstruite ?", "ABSTENTION", r"16e"),
    ("Quelle dynastie a construit l'Alhambra de Grenade ?", "ABSTENTION", None),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-cache", action="store_true", help="force de nouvelles générations")
    args = ap.parse_args()

    from merimee_rag.store import System

    system = System(cfg, load_dense=args.retriever in ("dense", "hybrid"))
    llm = OllamaLLM(args.model or cfg.gen_model, cfg.ollama_url, cfg.temperature,
                    None if args.no_cache else DiskCache(cfg.llm_cache))
    rag = RAGPipeline(system.retrievers[args.retriever], system.chunks, system.notices, llm, cfg.top_k_context)

    ok = 0
    for q, expected, forbidden in CASES:
        res = rag.answer(q)
        if expected == "ABSTENTION":
            passed = res.abstained
        else:
            passed = bool(re.search(expected, res.answer, re.I)) and not res.abstained
        if forbidden and re.search(forbidden, res.answer, re.I):
            passed = False
        ok += passed
        print(f"{'✅' if passed else '❌'} {q}\n   → {res.answer}  ({res.status}, {res.latency_s:.0f} s)")
        if not passed:  # pour comprendre l'échec : extraits fournis + sortie brute du modèle
            for c in res.contexts:
                print(f"     [{c.n}] {c.title}")
            print("     sortie brute : " + (res.raw or res.answer).replace("\n", " | "))
        print()
    print(f"{ok}/{len(CASES)} cas réussis — {llm.model}, retriever {args.retriever}")


if __name__ == "__main__":
    main()
