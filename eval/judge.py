"""LLM-juge : exactitude (vs réponse de référence) et fidélité (affirmations soutenues par un texte).

Choix de conception :
- Le juge est un modèle *différent* du générateur par défaut (qwen2.5:7b vs 3b)
  pour limiter le biais d'auto-préférence.
- La fidélité est décomposée en affirmations atomiques (à la RAGAS) : plus robuste
  qu'une note globale 1-5, et interprétable (on voit quelle affirmation est inventée).
"""
from __future__ import annotations

from merimee_rag.llm import LLM, parse_json

CORRECTNESS = """Tu évalues la réponse d'un système de questions-réponses sur le patrimoine.

Question : {question}
Réponse de référence : {reference}
Réponse du système : {answer}

La réponse du système est-elle correcte par rapport à la référence ?
- "correct" : contient l'information essentielle de la référence, sans contradiction.
- "partiel" : en partie juste, incomplète ou imprécise (ex. siècle au lieu de l'année exacte).
- "incorrect" : fausse, contradictoire, hors sujet, ou refus de répondre.
Des informations supplémentaires ne sont pas pénalisées ici si elles ne contredisent pas la référence.

Réponds en JSON : {{"verdict": "correct|partiel|incorrect", "justification": "une phrase"}}"""

FAITHFULNESS = """Tu vérifies si une réponse est fidèle à un texte source.

TEXTE SOURCE :
{source}

RÉPONSE :
{answer}

1. Découpe la réponse en affirmations factuelles atomiques (ignore les références [1], [2]…).
2. Pour chacune, indique si elle est explicitement soutenue par le texte source (true) ou non (false).
   Une affirmation plausible mais absente du texte est false.

Réponds en JSON : {{"affirmations": [{{"texte": "...", "soutenue": true}}]}}"""

VERDICT_SCORE = {"correct": 1.0, "partiel": 0.5, "incorrect": 0.0}


def judge_correctness(llm: LLM, question: str, reference: str, answer: str) -> tuple[float | None, str]:
    out = parse_json(llm.chat([{"role": "user", "content": CORRECTNESS.format(
        question=question, reference=reference, answer=answer)}], json_mode=True))
    if not isinstance(out, dict):
        return None, "juge: JSON invalide"
    verdict = str(out.get("verdict", "")).strip().lower()
    return VERDICT_SCORE.get(verdict), str(out.get("justification", ""))


def judge_faithfulness(llm: LLM, source: str, answer: str) -> tuple[float | None, list[dict]]:
    """Proportion d'affirmations soutenues. None si aucune affirmation (ex. abstention)."""
    out = parse_json(llm.chat([{"role": "user", "content": FAITHFULNESS.format(
        source=source[:6000], answer=answer)}], json_mode=True))
    claims = out.get("affirmations", []) if isinstance(out, dict) else []
    claims = [c for c in claims if isinstance(c, dict) and "soutenue" in c]
    if not claims:
        return None, []
    sup = [c["soutenue"] is True or str(c["soutenue"]).lower() == "true" for c in claims]
    return sum(sup) / len(sup), claims
