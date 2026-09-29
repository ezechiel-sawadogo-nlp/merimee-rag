"""Tests sur des notices *fictives* (tests/fixtures), avec encodeur et LLM factices :
aucun téléchargement, aucun modèle, aucun Ollama requis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merimee_rag.config import ABSTAIN, Config
from merimee_rag.corpus import (chunk_notice, iter_csv_notices, load_corpus, parse_coords,
                                resolve_columns, save_corpus)
from merimee_rag.pipeline import RAGPipeline, is_abstention, parse_citations
from merimee_rag.retrievers import BM25Retriever, DenseRetriever, FrenchAnalyzer, HybridRetriever, dedupe_refs
from merimee_rag.store import System, build_indexes

FIX = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------ factices
def fake_encoder(texts: list[str], is_query: bool) -> np.ndarray:
    """Sac de mots haché (dimension 256), normalisé : assez pour des tests déterministes."""
    an = FrenchAnalyzer()
    m = np.zeros((len(texts), 256), np.float32)
    for i, t in enumerate(texts):
        for tok in an(t):
            m[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class FakeLLM:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, json_mode=False, **opts):
        self.calls += 1
        content = messages[-1]["content"]
        if "Réponse de référence" in content:
            return json.dumps({"verdict": "correct", "justification": "ok"})
        if "TEXTE SOURCE" in content:
            return json.dumps({"affirmations": [{"texte": "x", "soutenue": True},
                                                {"texte": "y", "soutenue": False}]})
        if "HISTORIQUE :" in content:
            return json.dumps({"question": "Quand fut fondée la chapelle Saint-Exemple de Villetest ?",
                               "reponse": "En 1452.",
                               "extrait": "La chapelle fut fondée en 1452 par le seigneur de Villetest."})
        if "Extraits de la base Mérimée" in content:
            if "Djenné" in content:
                return ABSTAIN
            return "Elle fut fondée en 1452 [1]."
        return "Réponse sans source."


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.raw_csv = FIX / "mini_pop.csv"
    c.corpus_path = tmp_path / "corpus.jsonl"
    c.index_dir = tmp_path / "index"
    c.results_dir = tmp_path / "results"
    c.llm_cache = tmp_path / "cache.jsonl"
    c.chunk_words, c.chunk_overlap = 25, 8
    c.solr_url = None
    return c


@pytest.fixture()
def system(cfg: Config) -> System:
    save_corpus(iter_csv_notices(cfg.raw_csv, cfg.min_historique_chars), cfg.corpus_path)
    build_indexes(cfg, encoder=fake_encoder)
    return System(cfg, encoder=fake_encoder)


# ------------------------------------------------------------------ corpus
def test_resolve_pop_codes_and_long_labels():
    m = resolve_columns(["REF", "TICO", "HIST", "COM", "DPT"])
    assert m["ref"] == "REF" and m["historique"] == "HIST" and m["title"] == "TICO"
    m2 = resolve_columns(["Reference", "Titre éditorial de la notice", "Commune forme éditoriale",
                          "Historique", "Coordonnées au format WGS84"])
    assert m2["title"] == "Titre éditorial de la notice" and m2["coords"].startswith("Coord")
    with pytest.raises(ValueError):
        resolve_columns(["foo", "bar"])


def test_filter_dedupe_and_formats(cfg):
    notices = list(iter_csv_notices(cfg.raw_csv, 80))
    refs = [n.ref for n in notices]
    assert refs == ["TEST0001", "TEST0002", "TEST0003", "TEST0005"]  # court + doublon exclus
    assert notices[0].title == "Chapelle Saint-Exemple"
    other = list(iter_csv_notices(FIX / "mini_labels.csv", 40))
    assert other[0].commune == "Exempleville" and other[0].lat == pytest.approx(48.85)


def test_parse_coords_order():
    assert parse_coords("48.5, 2.3") == (48.5, 2.3)
    assert parse_coords("2.35, 48.85") == (48.85, 2.35)
    assert parse_coords("") == (None, None)


def test_chunking_covers_text_and_keeps_header(cfg):
    nt = [n for n in iter_csv_notices(cfg.raw_csv) if n.ref == "TEST0005"][0]
    chunks = chunk_notice(nt, chunk_words=25, overlap=8)
    assert len(chunks) >= 2
    assert all(c.text.startswith("Château de Testmont (Testmont") for c in chunks)
    body = " ".join(c.text.split("\n", 1)[1] for c in chunks)
    for sent in ("1640", "1670", "bien national", "bibliothèque"):
        assert sent in body
    assert [c.chunk_id for c in chunks] == [f"TEST0005#{i}" for i in range(len(chunks))]


def test_chunking_long_sentence_does_not_loop():
    from merimee_rag.corpus import Notice

    nt = Notice("X", "T", "", "", "", "", "", "", "", "", "", "mot " * 500 + ". Fin.", "", None, None)
    assert len(chunk_notice(nt, 50, 10)) >= 1


# ------------------------------------------------------------------ retrieval
def test_french_analyzer():
    toks = FrenchAnalyzer()("L'église fut reconstruite en 1760")
    assert "1760" in toks and "l" not in toks and any(t.startswith("eglis") for t in toks)


def test_retrievers_find_the_right_notice(system):
    q = "Quel ingénieur a dirigé la construction du pont de Pontest ?"
    for name in ("bm25", "dense", "hybrid"):
        hits = system.retrievers[name].search(q, 5)
        assert dedupe_refs(hits)[0] == "TEST0003", name


def test_hybrid_rrf_math():
    from merimee_rag.retrievers import Hit

    class R:
        def __init__(self, name, order): self.name, self.order = name, order
        def search(self, q, k): return [Hit(c, c, 1.0, i) for i, c in enumerate(self.order[:k])]

    h = HybridRetriever([R("a", ["x", "y", "z"]), R("b", ["y", "x", "w"])], rrf_k=60).search("q", 4)
    assert {h[0].chunk_id, h[1].chunk_id} == {"x", "y"} and h[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert [x.chunk_id for x in h[2:]] == ["z", "w"] or [x.chunk_id for x in h[2:]] == ["w", "z"]


def test_index_roundtrip(system, cfg):
    bm = BM25Retriever.load(cfg.index_dir / "bm25.pkl")
    assert bm.search("moulin meunier", 1)[0].ref == "TEST0002"
    d = DenseRetriever.load(cfg.index_dir / "dense", fake_encoder)
    assert d.matrix.shape[0] == len(system.chunks)


# ------------------------------------------------------------------ génération
def test_answer_status_partial():
    from merimee_rag.pipeline import answer_status

    djenne = ("[1] La mosquée a été construite entre 1920 et 1930, mais elle est une réplique de la mosquée "
              "de Djenné au Soudan (actuellement au Mali), qui date du 16e siècle. Je ne trouve pas cette "
              "information dans la base Mérimée pour la date de sa reconstruction spécifique.")
    assert answer_status(djenne) == "partial"
    assert answer_status(ABSTAIN) == "abstention"
    assert answer_status("Je ne trouve pas cette information [1].") == "abstention"
    assert answer_status("Elle fut fondée en 1452 [1].") == "answer"


def test_citations_and_abstention_parsing():
    assert parse_citations("A [1]. B [2, 3]. C [1].") == [1, 2, 3]
    assert is_abstention(ABSTAIN) and is_abstention("Désolé, je ne trouve pas l'information.")
    assert not is_abstention("Elle fut fondée en 1452 [1].")


def test_rag_answer(system):
    llm = FakeLLM()
    rag = RAGPipeline(system.retrievers["hybrid"], system.chunks, system.notices, llm, top_k=3)
    res = rag.answer("Quand la chapelle de Villetest a-t-elle été fondée ?")
    assert res.contexts[0].ref == "TEST0001" and res.cited == [1] and res.cited_refs == ["TEST0001"]
    assert res.contexts[0].url.endswith("/TEST0001")
    assert RAGPipeline(None, system.chunks, system.notices, llm).answer("q").contexts == []


# ------------------------------------------------------------------ évaluation
def test_metrics():
    from eval.metrics import bootstrap_ci, ndcg_at_k, paired_bootstrap_pvalue, reciprocal_rank

    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], {"a"}) == 1.0
    m, lo, hi = bootstrap_ci([1, 0, 1, 1])
    assert m == 0.75 and lo <= m <= hi
    assert paired_bootstrap_pvalue([1] * 30, [0] * 30) < 0.01
    assert paired_bootstrap_pvalue([1, 0] * 15, [1, 0] * 15) == 1.0


def test_extract_check():
    from eval.build_testset import extract_found

    t = "Le clocher fut reconstruit en 1760 après un incendie. Des peintures furent découvertes."
    assert extract_found("Le clocher fut reconstruit en 1760 après un incendie.", t)
    assert extract_found("Le clocher fut reconstruit en 1760 apres un incendie", t)  # accent perdu
    assert extract_found("Le clocher a été reconstruit en 1760 après un incendie.", t)  # retouche légère
    assert not extract_found("Le clocher fut détruit par la foudre en 1890.", t)


def test_end_to_end_eval(system, cfg, monkeypatch):
    from eval import build_testset, eval_generation, eval_retrieval, report

    llm = FakeLLM()
    items = build_testset.generate(load_corpus(cfg.corpus_path)[:1], llm, verbose=False)
    assert items and items[0]["gold_refs"] == ["TEST0001"]
    items.append({"id": "ooc-01", "type": "out_of_corpus", "question": "Grande Mosquée de Djenné ?",
                  "answer": None, "gold_refs": []})

    df = eval_retrieval.evaluate({k: system.retrievers[k] for k in ("bm25", "dense", "hybrid")}, items[:1])
    summ = eval_retrieval.summarize(df)
    assert set(summ["retriever"]) == {"bm25", "dense", "hybrid"}
    assert (summ["hit@5"] == 1.0).all()

    gold_texts = {n.ref: n.historique for n in system.notices}
    recs = []
    for name in ("no-rag", "rag-bm25", "rag-hybrid"):
        retr = None if name == "no-rag" else system.retrievers[name.removeprefix("rag-")]
        rag = RAGPipeline(retr, system.chunks, system.notices, llm, 3)
        recs += [eval_generation.score_one(rag, llm, q, gold_texts) for q in items]
    g = pd.DataFrame(recs)
    rag_h = g[(g.config == "rag-hybrid") & (g.type == "synthetic")].iloc[0]
    assert rag_h.correctness == 1.0 and rag_h.grounded_gold == 0.5 and rag_h.citation_hit == 1.0
    assert rag_h.outcome == "correct"
    ooc = g[(g.config == "rag-hybrid") & (g.type == "out_of_corpus")].iloc[0]
    assert ooc.correct_abstention == 1.0

    cfg.results_dir.mkdir(parents=True)
    eval_retrieval.summarize(df).to_csv(cfg.results_dir / "retrieval_summary.csv", index=False)
    eval_generation.summarize(g).to_csv(cfg.results_dir / "generation_summary.csv", index=False)
    monkeypatch.setattr(report, "Config", lambda: cfg)
    report.main()
    text = (cfg.results_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "## Retrieval" in text and "## Génération" in text and "rag-hybrid" in text
    assert (cfg.results_dir / "generation.png").stat().st_size > 5000


def test_json_loader():
    from merimee_rag.corpus import iter_notices

    ns = list(iter_notices(FIX / "mini.json", 80))
    assert [n.ref for n in ns] == ["TEST0201"]
    n = ns[0]
    assert n.commune == "Tourville, Autreville" and n.siecle == "14e, 17e siècles"
    assert n.protection == "classé en 1911 (partiellement)" and n.lat == pytest.approx(45.1)
    assert "Auteur(s) : Pierre Inventé" in n.header()


def test_parse_two_step():
    from merimee_rag.pipeline import parse_two_step

    assert parse_two_step("Extrait : [2]\nRéponse : François Ier, en 1519.") == ("François Ier, en 1519. [2]", [2])
    assert parse_two_step("**Extrait** : aucun\n**Réponse** : rien") == (ABSTAIN, [])
    assert parse_two_step("Extrait : [1] et [2]\nRéponse : Lorillard [1].") == ("Lorillard [1].", [1, 2])
    assert parse_two_step("Réponse libre [1].") == ("Réponse libre [1].", None)
    # cas réel (qwen2.5:3b, Écouen) : le gabarit recopié ne doit pas faire croire à une abstention
    raw = ("Extrait : [2] ou [3] ou aucun\nRéponse : Le château d'Écouen a été construit pour le "
           "connétable Anne de Montmorency.")
    assert parse_two_step(raw) == ("Le château d'Écouen a été construit pour le connétable Anne de "
                                   "Montmorency. [2][3]", [2, 3])


def test_two_step_answer_in_pipeline(system):
    class TwoStep(FakeLLM):
        def chat(self, messages, json_mode=False, **opts):
            return "Extrait : [1]\nRéponse : Elle fut fondée en 1452."

    res = RAGPipeline(system.retrievers["bm25"], system.chunks, system.notices, TwoStep(), 3).answer(
        "Quand la chapelle de Villetest a-t-elle été fondée ?")
    assert res.answer == "Elle fut fondée en 1452. [1]" and res.cited == [1] and res.status == "answer"
    assert res.raw.startswith("Extrait")


def test_ollama_retries_on_500(monkeypatch):
    import requests

    from merimee_rag.llm import OllamaLLM

    calls = []

    class R:
        def __init__(self, code): self.status_code, self.text = code, "boom"
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ok"}}

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        return R(500) if len(calls) < 3 else R(200)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert OllamaLLM("m").chat([{"role": "user", "content": "x"}]) == "ok" and len(calls) == 3

    calls.clear()
    monkeypatch.setattr(requests, "post", lambda *a, **k: (calls.append(1), R(500))[1])
    with pytest.raises(RuntimeError, match="ne répond plus"):
        OllamaLLM("m").chat([{"role": "user", "content": "y"}])
    assert len(calls) == 5


def test_ambiguity_heuristic():
    from eval.ambiguity import is_ambiguous

    assert is_ambiguous("Quel était le rôle principal de ce fort au 17e siècle ?", "Fort l'Union", "Le Robert")
    assert not is_ambiguous("Qui a construit le château de Chambord ?", "Domaine de Chambord", "Chambord;Thoury")
    assert not is_ambiguous("Quand l'église de Villetest a-t-elle été fondée ?", "Chapelle", "Villetest")


def test_rescore_end_to_end(system, cfg, monkeypatch):
    """rescore recalcule statuts et sous-ensembles à partir des fichiers par question."""
    import json as _json

    from eval import report, rescore

    cfg.eval_dir = cfg.results_dir.parent / "evalx"
    cfg.eval_dir.mkdir()
    cfg.results_dir.mkdir()
    qs = [{"id": "q1", "type": "synthetic", "question": "Quand fut fondée la chapelle de Villetest ?",
           "answer": "1452", "gold_refs": ["TEST0001"]},
          {"id": "q2", "type": "synthetic", "question": "Quand fut-elle fondée ?", "answer": "1452",
           "gold_refs": ["TEST0001"]},
          {"id": "ooc", "type": "out_of_corpus", "question": "Big Ben ?", "answer": None, "gold_refs": []}]
    (cfg.eval_dir / "testset.jsonl").write_text("\n".join(_json.dumps(q) for q in qs), encoding="utf-8")
    recs = []
    for c in ("no-rag", "rag-bm25"):
        for q in qs:
            r = {"config": c, "id": q["id"], "type": q["type"], "answer": "Elle date de 1452 [1].",
                 "status": "answer", "correctness": 1.0, "grounded_gold": 1.0, "gold_retrieved": 1.0}
            if q["id"] == "ooc":
                r["answer"] = "Le monument demandé n'est pas mentionné dans cet extrait. [1]"
            recs.append(r)
    (cfg.results_dir / "generation_per_query.jsonl").write_text(
        "\n".join(_json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    monkeypatch.setattr(rescore, "Config", lambda: cfg)
    monkeypatch.setattr(report, "Config", lambda: cfg)
    rescore.main()
    t = [_json.loads(l) for l in (cfg.eval_dir / "testset.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x.get("ambiguous") for x in t] == [False, True, None]
    s = pd.read_csv(cfg.results_dir / "generation_summary.csv")
    rag_all = s[(s.subset == "toutes") & (s.config == "rag-bm25")].iloc[0]
    assert rag_all.correct_abstention == 1.0          # refus reformulé désormais reconnu
    assert pd.isna(s[(s.subset == "toutes") & (s.config == "no-rag")].iloc[0].correct_abstention)
    assert s[(s.subset == "propres") & (s.config == "rag-bm25")].iloc[0].n_answerable == 1
    report.main()
    assert "bien posées" in (cfg.results_dir / "REPORT.md").read_text(encoding="utf-8")


def test_json_gz_loader(tmp_path):
    import gzip
    import shutil

    from merimee_rag.corpus import iter_notices

    gz = tmp_path / "mini.json.gz"
    with open(FIX / "mini.json", "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    assert [n.ref for n in iter_notices(gz, 80)] == ["TEST0201"]
