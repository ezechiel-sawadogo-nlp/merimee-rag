"""Mode agent : le LLM choisit lui-même ses outils (tool calling Ollama), en boucle.

Boucle : le modèle reçoit la question + la liste des outils → il appelle un ou plusieurs
outils → on exécute et on lui renvoie les résultats → … jusqu'à ce qu'il réponde sans
appeler d'outil (ou qu'on atteigne max_steps, auquel cas on lui demande de conclure).

Chaque étape est tracée (outil, arguments, résultat, erreur, durée) : c'est ce qui permet
d'évaluer le choix des outils et d'analyser les erreurs de l'agent.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .config import ABSTAIN
from .pipeline import answer_status
from .tools import Toolbox

SYSTEM_AGENT = f"""Tu es un assistant expert de la base Mérimée (24 824 monuments historiques français protégés).
Tu réponds en français, uniquement à partir des résultats de tes outils.

Choisis l'outil adapté :
- « combien de… » → count ;
- « quel département / quelle région / quelle commune a le plus (ou le moins) de… » → UN SEUL appel à count
  avec le filtre englobant (ex. region=…) et group_by (ex. group_by="departement") : le résultat classe tous
  les groupes, le premier est le plus grand ;
- lister des monuments selon des critères (commune, département, type, siècle, protection) → filter_notices ;
- histoire d'un monument précis (qui l'a construit, quand, pourquoi, de quoi il est fait) → search avec le nom
  du monument et sa commune, puis get_notice si l'extrait ne suffit pas ;
- « combien de monuments sont attribués à tel architecte » → trouve d'abord son nom exact (champ « auteurs »
  des résultats de search ou get_notice), puis count avec auteur="Nom Prénom" ;
- détails d'une notice dont tu connais la référence → get_notice.

Règles sur les arguments :
- n'utilise QUE les filtres que la question demande ; n'invente ni dates, ni bornes, ni critères ;
- « monuments protégés » ne veut pas dire protection="inscrit" : toute la base est protégée ; n'utilise
  protection que si la question dit « classé » ou « inscrit » ;
- une seule valeur par filtre (pas de liste) ;
- si un outil renvoie une erreur, lis les valeurs proposées et corrige tes arguments.

Réponse finale : courte ; reprends les nombres exacts renvoyés par les outils. Cite entre crochets les
références des notices (format PA suivi de 8 caractères) UNIQUEMENT si elles figurent dans les résultats
de tes outils ; un comptage n'a pas besoin de référence.
Appelle toujours au moins un outil avant de conclure que l'information est absente.
Si les outils ne permettent pas de répondre, réponds exactement : « {ABSTAIN} »"""

NUDGE = ("Tu n'as appelé aucun outil. Cherche d'abord dans la base (search pour un monument précis, "
         "count ou filter_notices pour des critères) avant de conclure.")

FINALIZE = ("Tu as atteint le nombre maximal d'étapes. Réponds maintenant à la question avec les informations "
            "déjà obtenues, sans appeler d'outil.")

# références Mérimée (PA00098405…) nues, ou tout identifiant entre crochets ([TEST1001])
REF_RE = re.compile(r"\b(?:PA|EA|IA|AP)[0-9A-Z]{8}\b|(?<=\[)[A-Z]{2,4}[0-9][0-9A-Z]{3,9}(?=\])")


class ToolLLM(Protocol):
    model: str

    def chat_message(self, messages: list[dict], tools: list[dict] | None = None, **opts: Any) -> dict: ...


@dataclass
class Step:
    tool: str
    args: dict
    result: dict
    error: bool
    latency_s: float


@dataclass
class AgentAnswer:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    status: str = "answer"
    hit_max_steps: bool = False
    latency_s: float = 0.0
    n_llm_calls: int = 0

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def cited_refs(self) -> list[str]:
        return list(dict.fromkeys(REF_RE.findall(self.answer)))

    @property
    def abstained(self) -> bool:
        return self.status == "abstention"

    @property
    def seen_refs(self) -> set[str]:
        """Références apparues dans les résultats d'outils (les seules que l'agent peut citer légitimement)."""
        found: set[str] = set()

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                if isinstance(x.get("ref"), str):
                    found.add(x["ref"])
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, str):
                found.update(REF_RE.findall(x))

        for s in self.steps:
            walk(s.result)
        return found

    @property
    def unsupported_refs(self) -> list[str]:
        """Références citées qui ne viennent d'aucun outil : citations inventées."""
        seen = self.seen_refs
        return [r for r in self.cited_refs if r not in seen]

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "status": self.status,
                "hit_max_steps": self.hit_max_steps, "latency_s": self.latency_s,
                "n_llm_calls": self.n_llm_calls, "cited_refs": self.cited_refs,
                "unsupported_refs": self.unsupported_refs,
                "steps": [{"tool": s.tool, "args": s.args, "error": s.error, "latency_s": s.latency_s,
                           "result": _shorten(s.result)} for s in self.steps]}


def _shorten(result: dict, n: int = 1500) -> dict | str:
    txt = json.dumps(result, ensure_ascii=False)
    return result if len(txt) <= n else txt[:n] + "…"


_CALL_IN_TEXT = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"(\w+)\"[^{}]*\"arguments\"\s*:\s*(\{.*?\})\s*\}", re.S)


def extract_tool_calls(msg: dict) -> list[tuple[str, Any]]:
    """Appels d'outils du message. Repli : certains modèles écrivent l'appel en JSON dans le texte
    (parfois entre balises <tool_call>) au lieu d'utiliser le champ tool_calls."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        if fn.get("name"):
            calls.append((fn["name"], fn.get("arguments") or {}))
    if not calls and msg.get("content"):
        for name, args in _CALL_IN_TEXT.findall(msg["content"]):
            try:
                calls.append((name, json.loads(args)))
            except json.JSONDecodeError:
                continue
    return calls


class Agent:
    def __init__(self, llm: ToolLLM, toolbox: Toolbox, max_steps: int = 6) -> None:
        self.llm, self.toolbox, self.max_steps = llm, toolbox, max_steps

    @property
    def name(self) -> str:
        return "agent"

    def answer(self, question: str, on_step: Callable[[int, Step], None] | None = None) -> AgentAnswer:
        t0 = time.perf_counter()
        messages: list[dict] = [{"role": "system", "content": SYSTEM_AGENT},
                                {"role": "user", "content": question}]
        res = AgentAnswer(question=question, answer="")
        nudged = False
        for _ in range(self.max_steps):
            msg = self.llm.chat_message(messages, tools=self.toolbox.specs)
            res.n_llm_calls += 1
            calls = extract_tool_calls(msg)
            if not calls:
                text = (msg.get("content") or "").strip()
                if not res.steps and not nudged and answer_status(text or ABSTAIN) == "abstention":
                    # abandon sans avoir rien cherché : on relance une fois (cas fréquent sur les questions factuelles)
                    nudged = True
                    messages += [{"role": "assistant", "content": text}, {"role": "user", "content": NUDGE}]
                    continue
                res.answer = text
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]})
            for name, args in calls:
                ts = time.perf_counter()
                out = self.toolbox.call(name, args)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                if isinstance(args, dict):
                    args = {k: v for k, v in args.items() if v not in (None, "", [], {})}
                res.steps.append(Step(name, args, out, "erreur" in out, time.perf_counter() - ts))
                if on_step:
                    on_step(len(res.steps), res.steps[-1])
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(out, ensure_ascii=False)})
        else:
            res.hit_max_steps = True
            messages.append({"role": "user", "content": FINALIZE})
            msg = self.llm.chat_message(messages, tools=None)
            res.n_llm_calls += 1
            res.answer = (msg.get("content") or "").strip()
        if not res.answer:
            res.answer = ABSTAIN
        res.status = answer_status(res.answer)
        res.latency_s = time.perf_counter() - t0
        return res
