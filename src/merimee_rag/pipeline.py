"""Pipeline RAG : retrieval → construction du contexte → génération citée."""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

from .config import ABSTAIN
from .corpus import Chunk, Notice
from .llm import LLM
from .retrievers import Retriever

# Consignes courtes : un modèle de 3B suit mal une longue liste de règles (v1 : 6 règles →
# il répondait sur le château de l'Ardoise à une question sur Chambord). On lui fait plutôt
# faire deux étapes explicites : choisir l'extrait qui parle du monument, puis répondre.
SYSTEM_RAG = """Tu réponds à des questions sur les monuments historiques français, en français,
uniquement à partir des extraits numérotés fournis. Tu n'utilises aucune autre connaissance."""

USER_RAG = """Question : {question}

Extraits de la base Mérimée :

{context}

Question : {question}

Travaille en deux étapes.
Étape 1 : quel extrait parle du monument demandé par la question (même monument, même lieu) ?
Un monument qui ne fait que mentionner ce nom, ou qui l'imite, ne compte pas. Si aucun ne convient, écris « aucun ».
Étape 2 : réponds en 1 à 3 phrases, uniquement avec ce que dit cet extrait, en citant son numéro entre crochets.

Réponds exactement dans ce format, sur deux lignes :
Extrait : le numéro choisi entre crochets, par exemple [2], ou bien le mot aucun
Réponse : ta réponse"""

SYSTEM_NO_RAG = """Tu es un assistant spécialisé dans le patrimoine protégé au titre des Monuments historiques.
Réponds en français, de façon concise (1 à 4 phrases)."""


@dataclass
class Context:
    n: int
    chunk_id: str
    ref: str
    title: str
    url: str
    text: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    contexts: list[Context] = field(default_factory=list)
    retrieved_refs: list[str] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    raw: str = ""  # sortie brute du LLM (avant extraction de la réponse)
    status: str = "answer"  # "answer" | "abstention" | "partial" (refuse ET affirme des faits)
    latency_s: float = 0.0

    @property
    def abstained(self) -> bool:
        return self.status == "abstention"

    @property
    def cited_refs(self) -> list[str]:
        by_n = {c.n: c.ref for c in self.contexts}
        return list(dict.fromkeys(by_n[n] for n in self.cited if n in by_n))


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s)


_ABSTAIN_NORM = _norm(ABSTAIN).rstrip(".")
_ABSTAIN_PATTERNS = [_ABSTAIN_NORM, "je ne trouve pas", "ne permettent pas de repondre",
                     "ne contiennent pas", "aucune information", "pas d'information",
                     "n'est pas mentionne", "ne mentionne pas", "ne sont pas mentionnes",
                     "aucun extrait", "ne parle pas de"]


def is_abstention(answer: str) -> bool:
    """Vrai si la réponse contient une formule de refus (même accompagnée d'autre chose)."""
    a = _norm(answer)
    return any(p in a for p in _ABSTAIN_PATTERNS)


def answer_status(answer: str, max_extra_words: int = 6) -> str:
    """Classe la réponse : "answer", "abstention" pure, ou "partial" (refus + affirmations).

    Un refus accompagné de faits (« X date du 16e siècle. Je ne trouve pas… ») n'est pas une
    abstention : il faut juger les faits affirmés, sinon on récompense une hallucination.
    """
    if not is_abstention(answer):
        return "answer"
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    rest = " ".join(s for s in sentences if not is_abstention(s))
    rest = re.sub(r"\[\d+(?:\s*[,;]\s*\d+)*\]", " ", rest)
    return "partial" if len(re.findall(r"\w+", rest)) > max_extra_words else "abstention"


_EXTRAIT = re.compile(r"(?<![\w'’])extraits?\s*:\s*(.*)$", re.I | re.M)
_REPONSE = re.compile(r"(?<![\w'’])r[ée]ponse\s*:\s*(.*)", re.I | re.S)


def parse_two_step(raw: str) -> tuple[str, list[int] | None]:
    """Extrait (réponse, extraits choisis) de la sortie « Extrait : … / Réponse : … ».

    Renvoie (ABSTAIN, []) si le modèle a choisi « aucun » ; (sortie brute, None) si le format
    n'est pas respecté (elle est alors jugée telle quelle).
    """
    text = raw.replace("*", "")  # tolère le gras Markdown (**Réponse** :)
    m_ex, m_rep = _EXTRAIT.search(text), _REPONSE.search(text)
    if not m_ex and not m_rep:
        return raw.strip(), None
    ex_line = m_ex.group(1) if m_ex else ""
    chosen = [int(x) for x in re.findall(r"\d+", ex_line)]
    # « aucun » ne compte que s'il n'y a aucun numéro : les petits modèles recopient parfois le
    # gabarit (« Extrait : [2] ou [3] ou aucun ») tout en ayant bien choisi un extrait.
    if not chosen and "aucun" in _norm(ex_line):
        return ABSTAIN, []
    answer = m_rep.group(1).strip() if m_rep else text.strip()
    if chosen and not parse_citations(answer) and not is_abstention(answer):
        answer = f"{answer} " + "".join(f"[{n}]" for n in chosen)  # citation oubliée : on la remet
    return answer, chosen


def parse_citations(answer: str) -> list[int]:
    nums = []
    for grp in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", answer):
        nums.extend(int(x) for x in re.split(r"[,;]\s*", grp))
    return list(dict.fromkeys(nums))


class RAGPipeline:
    def __init__(self, retriever: Retriever | None, chunks: Sequence[Chunk], notices: Sequence[Notice],
                 llm: LLM, top_k: int = 5) -> None:
        self.retriever, self.llm, self.top_k = retriever, llm, top_k
        self.chunks = {c.chunk_id: c for c in chunks}
        self.notices = {n.ref: n for n in notices}

    @property
    def name(self) -> str:
        return f"rag-{self.retriever.name}" if self.retriever else "no-rag"

    def retrieve(self, question: str) -> list[Context]:
        assert self.retriever is not None
        hits = self.retriever.search(question, self.top_k)
        ctxs = []
        for i, h in enumerate(hits, 1):
            ch = self.chunks.get(h.chunk_id) or self.chunks.get(f"{h.ref}#0")
            nt = self.notices.get(h.ref)
            if ch is None or nt is None:
                continue
            ctxs.append(Context(i, ch.chunk_id, h.ref, nt.title or nt.denomination, nt.url, ch.text))
        return ctxs

    @staticmethod
    def format_context(ctxs: Sequence[Context]) -> str:
        return "\n\n".join(f"[{c.n}] (réf. {c.ref})\n{c.text}" for c in ctxs)

    def answer(self, question: str) -> RAGAnswer:
        t0 = time.perf_counter()
        if self.retriever is None:
            msgs = [{"role": "system", "content": SYSTEM_NO_RAG}, {"role": "user", "content": question}]
            out = self.llm.chat(msgs).strip()
            return RAGAnswer(question, out, status=answer_status(out), latency_s=time.perf_counter() - t0)

        ctxs = self.retrieve(question)
        user = USER_RAG.format(question=question, context=self.format_context(ctxs))
        msgs = [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]
        raw = self.llm.chat(msgs).strip()
        out, _ = parse_two_step(raw)
        return RAGAnswer(
            question=question, answer=out, raw=raw, contexts=ctxs,
            retrieved_refs=list(dict.fromkeys(c.ref for c in ctxs)),
            cited=parse_citations(out), status=answer_status(out),
            latency_s=time.perf_counter() - t0,
        )
