"""Construit le jeu de test : questions synthétiques ancrées dans une notice + questions hors corpus.

Pour chaque notice échantillonnée, le LLM produit (question, réponse, extrait).
Contrôle qualité automatique : l'« extrait » doit se retrouver (quasi) mot pour mot
dans l'historique, sinon la paire est rejetée (le LLM a inventé).

Biais connu : une question générée à partir d'un texte tend à en reprendre le
vocabulaire, ce qui avantage BM25. On demande donc explicitement de reformuler, et
on mesure le recouvrement lexical question/historique (`lexical_overlap`) pour
pouvoir analyser les résultats par tranche.

Usage :
  python -m eval.build_testset --n 150
  python -m eval.build_testset --n 150 --manual eval/manual_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from merimee_rag.config import Config
from merimee_rag.corpus import Notice, load_corpus
from merimee_rag.llm import DiskCache, LLM, OllamaLLM, parse_json
from merimee_rag.retrievers import FrenchAnalyzer

from .ambiguity import is_ambiguous

PROMPT = """Voici une notice de la base Mérimée (monuments historiques français).

MONUMENT : {header}
HISTORIQUE : {hist}

Rédige UNE question factuelle dont la réponse se trouve dans l'historique, puis sa réponse.
Contraintes :
- La question doit nommer le monument ET la commune, pour être non ambiguë parmi 40 000 monuments.
- Porte sur un fait précis (date, commanditaire, architecte, événement, transformation…).
- Reformule avec tes propres mots : ne recopie pas les phrases de l'historique.
- La réponse fait au plus une phrase.
- "extrait" = la phrase de l'historique, copiée mot pour mot, qui justifie la réponse.

Réponds uniquement en JSON : {{"question": "...", "reponse": "...", "extrait": "..."}}"""


def sample_notices(notices: list[Notice], n: int, seed: int, min_chars: int = 300) -> list[Notice]:
    """Échantillon stratifié par région (répartition proportionnelle, au moins 1 par région)."""
    rng = random.Random(seed)
    pool = [x for x in notices if len(x.historique) >= min_chars]
    by_region: dict[str, list[Notice]] = defaultdict(list)
    for x in pool:
        by_region[x.region or "?"].append(x)
    out: list[Notice] = []
    for reg, items in sorted(by_region.items()):
        rng.shuffle(items)
        quota = max(1, round(n * len(items) / len(pool)))
        out.extend(items[:quota])
    rng.shuffle(out)
    return out[:n]


def extract_found(extract: str, text: str, threshold: float = 0.85) -> bool:
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
        return re.sub(r"\W+", " ", s).strip()

    e, t = norm(extract), norm(text)
    if not e or len(e) < 15:
        return False
    if e in t:
        return True
    # tolérance aux petites retouches : on aligne sur le plus long bloc commun
    m = SequenceMatcher(None, t, e, autojunk=False).find_longest_match(0, len(t), 0, len(e))
    start = max(0, m.a - m.b)
    return SequenceMatcher(None, t[start:start + len(e)], e).ratio() >= threshold


def lexical_overlap(question: str, text: str, analyzer: FrenchAnalyzer) -> float:
    q = set(analyzer(question))
    return len(q & set(analyzer(text))) / len(q) if q else 0.0


def generate(notices: list[Notice], llm: LLM, verbose: bool = True) -> list[dict]:
    analyzer = FrenchAnalyzer()
    items, rejected = [], 0
    for i, nt in enumerate(notices):
        prompt = PROMPT.format(header=nt.header(), hist=nt.historique[:2500])
        data = parse_json(llm.chat([{"role": "user", "content": prompt}], json_mode=True))
        ok = (isinstance(data, dict) and data.get("question") and data.get("reponse")
              and extract_found(str(data.get("extrait", "")), nt.historique))
        if not ok:
            rejected += 1
            continue
        items.append({
            "id": f"syn-{nt.ref}", "type": "synthetic", "question": data["question"].strip(),
            "answer": data["reponse"].strip(), "evidence": data["extrait"].strip(),
            "gold_refs": [nt.ref], "region": nt.region,
            "lexical_overlap": round(lexical_overlap(data["question"], nt.historique, analyzer), 3),
            "ambiguous": is_ambiguous(data["question"], nt.title, nt.commune),
        })
        if verbose:
            print(f"\r  {i + 1}/{len(notices)}  gardées={len(items)}  rejetées={rejected}", end="", flush=True)
    if verbose:
        print()
    return items


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="nb de notices échantillonnées")
    ap.add_argument("--model", default=None, help="modèle Ollama générateur de questions (défaut : juge)")
    ap.add_argument("--manual", type=Path, default=None, help="JSONL de questions écrites à la main")
    ap.add_argument("--out", type=Path, default=cfg.eval_dir / "testset.jsonl")
    args = ap.parse_args()

    notices = load_corpus(cfg.corpus_path)
    llm = OllamaLLM(args.model or cfg.judge_model, cfg.ollama_url, 0.0, DiskCache(cfg.llm_cache))
    print(f"Génération de questions sur {args.n} notices avec {llm.model}…")
    items = generate(sample_notices(notices, args.n, cfg.seed), llm)

    with open(cfg.eval_dir / "out_of_corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                items.append({**q, "type": "out_of_corpus", "answer": None, "gold_refs": []})
    if args.manual and args.manual.exists():
        with open(args.manual, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append({"type": "manual", **json.loads(line)})

    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for it in items:
        counts[it["type"]] += 1
    n_amb = sum(1 for it in items if it.get("ambiguous"))
    print(f"{len(items)} questions → {args.out}  {dict(counts)}  dont {n_amb} sous-spécifiées (signalées)")


if __name__ == "__main__":
    main()
