"""Détection des questions « sous-spécifiées » du jeu de test.

Une question générée qui ne mentionne ni la commune ni aucun mot distinctif du nom du
monument (« Quel était le rôle de ce fort au 17e siècle ? ») n'a pas de réponse unique
parmi 23 000 notices : elle pénalise tous les systèmes et surtout le retrieval dense.
Le générateur (qwen2.5:7b) en a produit ~18 % malgré la consigne ; on les signale
(champ `ambiguous`) et on rapporte les métriques avec et sans elles.

Heuristique volontairement simple et vérifiable : on cherche dans la question au moins
un mot non générique du nom de la commune ou du titre de la notice.
"""
from __future__ import annotations

import re
import unicodedata

GENERIC = set("""
eglise chapelle chateau maison immeuble hotel ancien ancienne pont tour fort manoir domaine abbaye prieure
croix calvaire moulin porte maisons particulier dit dite de du des la le les l d et en sur sous saint sainte
notre dame batiment ensemble site vestiges restes monument edifice villa ferme grange halle fontaine lavoir
parc jardin jardins cathedrale basilique temple couvent college hopital mairie theatre gare usine rue place quai
""".split())


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def _distinctive(text: str) -> set[str]:
    return {w for w in _norm(text).split() if len(w) > 2 and w not in GENERIC and not w.isdigit()}


def is_ambiguous(question: str, title: str, commune: str) -> bool:
    words = set(_norm(question).split())
    return not (words & (_distinctive(title) | _distinctive(commune.replace(";", " "))))
