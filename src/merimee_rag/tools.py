"""Les outils de l'agent : définitions JSON (format « tools » d'Ollama) + exécution.

Chaque outil renvoie un dict JSON-sérialisable. Les erreurs de l'agent (filtre inconnu,
argument invalide) ne lèvent pas d'exception : elles sont renvoyées au LLM sous forme
{"erreur": ...} avec des suggestions, pour qu'il puisse se corriger à l'étape suivante.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from .corpus import Chunk
from .metadata import FILTER_FIELDS, GROUP_FIELDS, FilterError, MetadataStore
from .retrievers import Retriever

MAX_LIST = 20      # notices renvoyées au plus par filter_notices
MAX_GROUPS = 15    # groupes renvoyés au plus par count
MAX_TEXT = 2500    # caractères d'historique renvoyés par get_notice

_FILTER_PROPS: dict[str, dict] = {
    "commune": {"type": "string", "description": "Nom exact de la commune, ex. « Chambord »"},
    "departement": {"type": "string", "description": "Nom du département en toutes lettres, ex. « Loir-et-Cher »"},
    "region": {"type": "string", "description": "Nom de la région, ex. « Centre-Val de Loire »"},
    "denomination": {"type": "string", "description": "Type d'édifice au singulier : église, château, chapelle, "
                     "maison, pont, moulin, abbaye, croix, manoir, fontaine…"},
    "domaine": {"type": "string", "description": "Domaine : religieuse, domestique, militaire, funéraire, industrielle…"},
    "siecle": {"type": "integer", "description": "Siècle de construction, en nombre (12 pour le 12e siècle)"},
    "protection": {"type": "string", "enum": ["classé", "inscrit"],
                   "description": "Niveau de protection au titre des Monuments historiques"},
    "statut": {"type": "string", "enum": ["publique", "privée", "mixte"], "description": "Propriétaire"},
    "auteur": {"type": "string", "description": "Nom (ou partie du nom) d'un architecte ou auteur"},
    "annee_min": {"type": "integer", "description": "Année de protection minimale — UNIQUEMENT si la question "
                  "porte sur la date de protection (ex. « protégés depuis 1990 »)"},
    "annee_max": {"type": "integer", "description": "Année de protection maximale — même règle que annee_min"},
    "mot_cle": {"type": "string", "description": "Mot à chercher dans le titre ou l'historique, ex. « roman »"},
}
assert set(_FILTER_PROPS) == set(FILTER_FIELDS)

TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Recherche en texte libre dans les historiques des notices. À utiliser pour une question "
                       "sur l'histoire d'un monument précis (qui l'a construit, quand, que s'est-il passé).",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "La question ou des mots-clés, avec le nom du monument et la commune"},
            "method": {"type": "string", "enum": ["hybrid", "bm25", "dense"],
                       "description": "hybrid par défaut ; bm25 pour des noms propres rares"},
            "k": {"type": "integer", "description": "Nombre de passages (1 à 10, défaut 5)"}}}}},
    {"type": "function", "function": {
        "name": "filter_notices",
        "description": "Liste les notices qui satisfont des critères structurés (commune, département, type "
                       "d'édifice, siècle, protection…). Renvoie le total et au plus 20 notices.",
        "parameters": {"type": "object", "properties": _FILTER_PROPS}}},
    {"type": "function", "function": {
        "name": "count",
        "description": "Compte les notices qui satisfont des critères, et peut les regrouper (group_by) pour "
                       "répondre à « combien », « quel département a le plus de… ».",
        "parameters": {"type": "object", "properties": {
            **_FILTER_PROPS,
            "group_by": {"type": "string", "enum": list(GROUP_FIELDS),
                         "description": "Regrouper le décompte par ce champ (facultatif)"}}}}},
    {"type": "function", "function": {
        "name": "get_notice",
        "description": "Renvoie la notice complète (historique, auteurs, dates) à partir de sa référence, "
                       "ex. PA00098405.",
        "parameters": {"type": "object", "required": ["ref"], "properties": {
            "ref": {"type": "string", "description": "Référence Mérimée, ex. PA00098405"}}}}},
]
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SPECS]


def _coerce_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(float(str(v)))))
    except (TypeError, ValueError):
        return default


class Toolbox:
    def __init__(self, store: MetadataStore, retrievers: dict[str, Retriever] | None = None,
                 chunks: Sequence[Chunk] = ()) -> None:
        self.store = store
        self.retrievers = retrievers or {}
        self.chunks = {c.chunk_id: c for c in chunks}
        self._fns: dict[str, Callable[..., dict]] = {
            "search": self.search, "filter_notices": self.filter_notices,
            "count": self.count, "get_notice": self.get_notice,
        }

    @property
    def specs(self) -> list[dict]:
        return [t for t in TOOL_SPECS if t["function"]["name"] in self._fns
                and (t["function"]["name"] != "search" or self.retrievers)]

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, arguments: dict | str | None) -> dict:
        """Exécute un outil ; ne lève jamais : les erreurs sont renvoyées au LLM."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return {"erreur": f"arguments non JSON : {arguments[:200]}"}
        args = {k: v for k, v in (arguments or {}).items() if v not in (None, "")}
        fn = self._fns.get(name)
        if fn is None:
            return {"erreur": f"outil inconnu « {name} ». Outils disponibles : {TOOL_NAMES}"}
        try:
            return fn(**args)
        except FilterError as e:
            return {"erreur": str(e)}
        except TypeError as e:  # argument inattendu
            return {"erreur": f"arguments invalides pour {name} : {e}"}

    # ------------------------------------------------------------------ outils
    def search(self, query: str, method: str = "hybrid", k: int = 5) -> dict:
        if method not in self.retrievers:
            method = "hybrid" if "hybrid" in self.retrievers else next(iter(self.retrievers))
        k = _coerce_int(k, 5, 1, 10)
        hits = self.retrievers[method].search(str(query), k)
        out = []
        for h in hits:
            ch = self.chunks.get(h.chunk_id)
            rec = self.store.by_ref.get(h.ref)
            if ch is None:
                continue
            out.append({"ref": h.ref, "titre": rec.titre if rec else "",
                        "commune": ", ".join(rec.commune) if rec else "",
                        "extrait": ch.text.split("\n", 1)[-1][:700]})
        return {"methode": method, "resultats": out}

    def filter_notices(self, **filters: Any) -> dict:
        recs = self.store.match(**filters)
        return {"total": len(recs), "affichees": min(len(recs), MAX_LIST),
                "notices": [r.summary() for r in recs[:MAX_LIST]]}

    def count(self, group_by: str | None = None, **filters: Any) -> dict:
        recs = self.store.match(**filters)
        res: dict[str, Any] = {"total": len(recs), "filtres": filters}
        if group_by:
            groups = self.store.group_counts(recs, group_by)
            res["group_by"] = group_by
            res["groupes"] = [{"valeur": v, "nombre": n} for v, n in groups[:MAX_GROUPS]]
            res["nb_groupes"] = len(groups)
        return res

    def get_notice(self, ref: str) -> dict:
        ref = str(ref).strip().strip("[]").upper()
        r = self.store.by_ref.get(ref)
        if r is None:
            return {"erreur": f"référence inconnue : {ref}"}
        hist = r.historique
        return {**r.summary(), "region": r.region, "auteurs": r.auteurs, "statut": r.statut,
                "historique": hist[:MAX_TEXT] + ("…" if len(hist) > MAX_TEXT else ""), "url": r.url}
