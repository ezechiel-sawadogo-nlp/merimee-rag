"""Jeu de questions pour comparer RAG fixe et agent, avec des réponses EXACTES calculées sur les données.

Types (là où un RAG « 5 passages » est structurellement désavantagé, plus des témoins) :
  count      « Combien d'églises classées dans l'Aube ? »                → nombre exact
  list       « Quels monuments protégés compte la commune de X (dép.) ? » → ensemble de références
  argmax     « Quel département de la région R compte le plus de … ? »    → une valeur
  multistep  « Qui est l'architecte de X, et combien de monuments lui sont attribués ? » → nom + nombre
  lookup     question factuelle sur l'historique d'une notice (reprise de eval/testset.jsonl) → juge LLM
  ooc        question hors corpus → abstention attendue

Usage : python -m eval.agent_questions   → eval/agent_testset.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from merimee_rag.config import Config, metadata_path
from merimee_rag.metadata import MetadataStore, norm

PLURAL = {"église": "églises", "château": "châteaux", "chapelle": "chapelles", "maison": "maisons",
          "manoir": "manoirs", "abbaye": "abbayes", "pont": "ponts", "moulin": "moulins", "croix": "croix",
          "fontaine": "fontaines", "prieuré": "prieurés", "hôtel": "hôtels", "immeuble": "immeubles"}
FEM = {"église", "chapelle", "maison", "abbaye", "croix", "fontaine"}


def _de(word: str) -> str:
    """« de églises » → « d'églises »."""
    return ("d'" if norm(word)[:1] in "aeiouyh" else "de ") + word


def _adj(prot: str, denom: str) -> str:
    base = {"classé": "classé", "inscrit": "inscrit"}[prot]
    return base + ("es" if denom in FEM else "s")


def _author_name(a: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", a).strip()


def build(store: MetadataStore, testset: Path | None, seed: int = 7, sizes: dict | None = None) -> list[dict]:
    sizes = {"count": 20, "list": 12, "argmax": 10, "multistep": 10, "lookup": 15, "ooc": 5, **(sizes or {})}
    rng = random.Random(seed)
    items: list[dict] = []
    deps = sorted({r.departement for r in store.records if r.departement})
    regions = sorted({r.region for r in store.records if r.region})

    # --- count ----------------------------------------------------------------
    seen = set()
    tries = 0
    while sum(i["type"] == "count" for i in items) < sizes["count"] and tries < 5000:
        tries += 1
        d, dep, prot = rng.choice(list(PLURAL)), rng.choice(deps), rng.choice(["classé", "inscrit"])
        if (d, dep, prot) in seen:
            continue
        seen.add((d, dep, prot))
        n = len(store.match(denomination=d, departement=dep, protection=prot))
        if not 2 <= n <= 80:
            continue
        items.append({"type": "count", "question": f"Combien {_de(PLURAL[d])} {_adj(prot, d)} au titre des "
                      f"monuments historiques y a-t-il dans le département « {dep} » ?",
                      "gold": n, "filters": {"denomination": d, "departement": dep, "protection": prot},
                      "expected_tool": "count"})

    # --- list -------------------------------------------------------------------
    by_commune = Counter((c, r.departement) for r in store.records for c in r.commune)
    cands = sorted(k for k, v in by_commune.items() if 2 <= v <= 5)
    for commune, dep in rng.sample(cands, sizes["list"]):
        refs = sorted(r.ref for r in store.match(commune=commune, departement=dep))
        items.append({"type": "list", "question": f"Quels monuments protégés compte la commune de {commune} "
                      f"({dep}) ? Donne leurs références.", "gold": refs,
                      "gold_titles": [store.by_ref[r].titre for r in refs],
                      "filters": {"commune": commune, "departement": dep}, "expected_tool": "filter_notices"})

    # --- argmax -----------------------------------------------------------------
    seen = set()
    tries = 0
    while sum(i["type"] == "argmax" for i in items) < sizes["argmax"] and tries < 5000:
        tries += 1
        reg, d, prot = rng.choice(regions), rng.choice(list(PLURAL)), rng.choice(["classé", "inscrit"])
        if (reg, d, prot) in seen:
            continue
        seen.add((reg, d, prot))
        groups = store.group_counts(store.match(region=reg, denomination=d, protection=prot), "departement")
        if len(groups) < 3 or groups[0][1] < 5 or groups[0][1] == groups[1][1]:
            continue  # il faut un gagnant net
        items.append({"type": "argmax", "question": f"Quel département de la région {reg} compte le plus "
                      f"{_de(PLURAL[d])} {_adj(prot, d)} au titre des monuments historiques, et combien ?",
                      "gold": groups[0][0], "gold_count": groups[0][1],
                      "filters": {"region": reg, "denomination": d, "protection": prot, "group_by": "departement"},
                      "expected_tool": "count"})

    # --- multistep --------------------------------------------------------------
    author_counts = Counter(norm(_author_name(a)) for r in store.records for a in r.auteurs)
    title_counts = Counter(norm(r.titre) for r in store.records)
    pool = [r for r in store.records if len(r.auteurs) == 1 and title_counts[norm(r.titre)] == 1
            and 3 <= author_counts[norm(_author_name(r.auteurs[0]))] <= 60
            and len(norm(_author_name(r.auteurs[0])).split()) >= 2 and r.commune]
    for r in rng.sample(pool, sizes["multistep"]):
        name = _author_name(r.auteurs[0])
        n = len(store.match(auteur=name))
        items.append({"type": "multistep", "question": f"Qui est l'auteur (architecte) du monument « {r.titre} » "
                      f"à {r.commune[0]}, et combien de monuments de la base Mérimée lui sont attribués au total ?",
                      "gold": name, "gold_count": n, "gold_ref": r.ref,
                      "filters": {"auteur": name}, "expected_tool": "count"})

    # --- lookup (témoin : le terrain du RAG) --------------------------------------
    if testset and testset.exists():
        qs = [json.loads(l) for l in open(testset, encoding="utf-8") if l.strip()]
        clean = [q for q in qs if q.get("type") == "synthetic" and not q.get("ambiguous")]
        for q in rng.sample(clean, min(sizes["lookup"], len(clean))):
            items.append({"type": "lookup", "question": q["question"], "gold": q["answer"],
                          "gold_refs": q["gold_refs"], "expected_tool": "search"})
        ooc = [q for q in qs if q.get("type") == "out_of_corpus"]
        for q in rng.sample(ooc, min(sizes["ooc"], len(ooc))):
            items.append({"type": "ooc", "question": q["question"], "gold": None, "expected_tool": None})

    for i, it in enumerate(items):
        it["id"] = f"{it['type']}-{i:03d}"
    return items


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    store = MetadataStore.load(metadata_path(cfg))
    items = build(store, cfg.eval_dir / "testset.jsonl")
    out = cfg.eval_dir / "agent_testset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} questions → {out}  {dict(Counter(i['type'] for i in items))}")


if __name__ == "__main__":
    main()
