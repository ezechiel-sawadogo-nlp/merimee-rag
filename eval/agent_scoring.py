"""Notation automatique des réponses aux questions agent (sans LLM, sauf le type lookup).

Et diagnostic des erreurs de l'agent : la bonne information était-elle dans les résultats
de ses outils ? Si oui, l'échec vient de la synthèse ; sinon, du choix d'outil ou des arguments.
"""
from __future__ import annotations

import re
from typing import Any

from merimee_rag.metadata import norm

_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:[   ]\d{3})+|\d+)(?![\w]|[.,]\d)")
REF = re.compile(r"\b(?:PA|EA|IA|AP)[0-9A-Z]{8}\b")


def numbers(text: str) -> set[int]:
    """Nombres « de comptage » du texte : ignore 12e, 1er, 16ème, et les nombres collés à des lettres."""
    out = set()
    for m in _NUM.finditer(text or ""):
        tail = text[m.end():m.end() + 3].lower()
        if tail.startswith(("e ", "e.", "e,", "ème", "eme", "er ", "è")) or tail in ("e",):
            continue
        out.add(int(re.sub(r"\D", "", m.group(1))))
    return out


def mentions(answer: str, name: str) -> bool:
    """Tous les mots (≥ 3 lettres) du nom apparaissent dans la réponse, dans n'importe quel ordre."""
    a = set(norm(answer).split())
    words = [w for w in norm(name).split() if len(w) >= 3] or norm(name).split()
    return bool(words) and all(w in a for w in words)


def refs_in(answer: str, titles_by_ref: dict[str, str]) -> set[str]:
    """Références citées, ou notices dont le titre complet est recopié dans la réponse."""
    found = set(REF.findall(answer or ""))
    na = f" {norm(answer)} "
    for ref, title in titles_by_ref.items():
        nt = norm(title)
        if len(nt) >= 8 and f" {nt} " in na:
            found.add(ref)
    return found


def score(q: dict, answer: str, abstained: bool, titles_by_ref: dict[str, str] | None = None) -> dict:
    """Score dans [0, 1] + détails. Pour lookup, le score est fixé plus tard par le juge."""
    t = q["type"]
    if t == "ooc":
        return {"score": float(abstained)}
    if abstained:
        return {"score": 0.0, "detail": "abstention"}
    if t == "count":
        return {"score": float(q["gold"] in numbers(answer))}
    if t == "argmax":
        name_ok = mentions(answer, q["gold"])
        count_ok = q["gold_count"] in numbers(answer)
        return {"score": 1.0 if name_ok and count_ok else 0.5 if name_ok else 0.0,
                "name_ok": name_ok, "count_ok": count_ok}
    if t == "multistep":
        name_ok = mentions(answer, q["gold"])
        count_ok = q["gold_count"] in numbers(answer)
        return {"score": (name_ok + count_ok) / 2, "name_ok": name_ok, "count_ok": count_ok}
    if t == "list":
        gold = set(q["gold"])
        # Titres recopiés : on ne crédite que ceux des notices attendues. (Comparer à tous les titres de la
        # base ajoutait de fausses références : « Église Saint-Valentin » existe dans plusieurs communes.)
        got = refs_in(answer, dict(zip(q["gold"], q["gold_titles"])))
        tp = len(gold & got)
        p = tp / len(got) if got else 0.0
        r = tp / len(gold)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"score": f1, "precision": p, "recall": r}
    return {"score": None}  # lookup : juge


# ------------------------------------------------------------------ diagnostic agent
def _results(steps: list[dict]) -> list[dict]:
    return [s["result"] for s in steps if isinstance(s.get("result"), dict)]


def gold_seen(q: dict, steps: list[dict]) -> bool | None:
    """La réponse attendue apparaît-elle dans les résultats d'outils ? (None : non applicable)"""
    res = _results(steps)
    t = q["type"]
    if t == "count":
        return any(r.get("total") == q["gold"] for r in res)
    if t == "argmax":
        return any((r.get("groupes") or [{}])[0].get("valeur") == q["gold"] for r in res)
    if t == "multistep":
        return any(r.get("total") == q["gold_count"] for r in res)
    if t == "list":
        seen = {n.get("ref") for r in res for n in r.get("notices", [])}
        return set(q["gold"]) <= seen
    if t == "lookup":
        seen = {x.get("ref") for r in res for x in r.get("resultats", [])} | {r.get("ref") for r in res}
        return bool(set(q.get("gold_refs", [])) & seen)
    return None


def tool_ok(q: dict, steps: list[dict]) -> bool | None:
    """L'outil attendu a-t-il été appelé (avec group_by pour argmax) ?"""
    exp = q.get("expected_tool")
    if not exp:
        return None
    calls = [s for s in steps if s["tool"] == exp]
    if q["type"] == "argmax":
        return any(s["args"].get("group_by") for s in calls)
    if q["type"] == "list":  # count ne donne pas la liste ; filter_notices attendu
        return bool(calls)
    return bool(calls)


def extra_filters(q: dict, steps: list[dict]) -> list[str]:
    """Filtres ajoutés par l'agent sans que la question les demande (ex. annee_min inventée)."""
    wanted = set((q.get("filters") or {}).keys()) | {"group_by"}
    if not q.get("filters"):
        return []
    # le modèle envoie souvent tous les champs, vides (« auteur": null ») : seuls les filtres renseignés comptent
    used = {k for s in steps if s["tool"] in ("count", "filter_notices")
            for k, v in s["args"].items() if v not in (None, "", [], {})}
    return sorted(used - wanted)


def diagnose(q: dict, rec: dict[str, Any]) -> str:
    if rec["score"] is not None and rec["score"] >= 1.0:
        return "correct"
    steps = rec.get("steps", [])
    if q["type"] == "ooc":
        return "n'a pas refusé"
    if not steps:
        return "aucun outil appelé"
    if rec.get("hit_max_steps"):
        return "étapes épuisées"
    if gold_seen(q, steps):
        return "synthèse (bonne donnée, mauvaise réponse)"
    if tool_ok(q, steps) is False:
        return "mauvais outil"
    return "mauvais arguments (donnée jamais obtenue)"
