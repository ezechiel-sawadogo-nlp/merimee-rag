"""Outils et boucle d'agent, sur des notices fictives (tests/fixtures/mini_meta.json), avec un faux LLM."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merimee_rag.agent import Agent, extract_tool_calls
from merimee_rag.config import ABSTAIN
from merimee_rag.metadata import MetadataStore
from merimee_rag.tools import TOOL_NAMES, TOOL_SPECS, Toolbox

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tb() -> Toolbox:
    return Toolbox(MetadataStore.load(FIX / "mini_meta.json"))


# ------------------------------------------------------------------ outils
def test_count_filters_and_normalization(tb):
    assert tb.call("count", {"denomination": "église"})["total"] == 3
    assert tb.call("count", {"denomination": "églises", "departement": "aube"})["total"] == 2
    assert tb.call("count", {"departement": "Cote d'Or", "siecle": "12e"})["total"] == 1
    assert tb.call("count", {"protection": "classé"})["total"] == 3
    assert tb.call("count", {"denomination": "croix"})["total"] == 1          # « croix de chemin »
    assert tb.call("count", {"commune": "Pontest"})["total"] == 2              # commune multiple « A;B »
    assert tb.call("count", {"mot_cle": "roman"})["total"] == 2               # « romane »
    assert tb.call("count", {"auteur": "fictif"})["total"] == 2
    assert tb.call("count", {"annee_min": 1900, "annee_max": 1930})["total"] == 3


def test_count_group_by(tb):
    out = tb.call("count", {"denomination": "église", "group_by": "region"})
    assert out["groupes"][0] == {"valeur": "Grand Est", "nombre": 2} and out["nb_groupes"] == 2


def test_filter_notices_and_get_notice(tb):
    out = tb.call("filter_notices", {"commune": "Riviertest"})
    assert out["total"] == 2 and {n["ref"] for n in out["notices"]} == {"TEST1004", "TEST1005"}
    n = tb.call("get_notice", {"ref": "[test1003]"})
    assert n["titre"] == "Château de Testmont" and "Jean Fictif" in n["historique"]
    assert "erreur" in tb.call("get_notice", {"ref": "PA99999999"})


def test_errors_are_returned_with_suggestions(tb):
    e = tb.call("count", {"departement": "Grand Est"})["erreur"]
    assert "region" in e and "Grand Est" in e                   # région passée comme département
    assert "Aube" in tb.call("count", {"departement": "Aubbe"})["erreur"]
    assert "inconnu" in tb.call("count", {"couleur": "rouge"})["erreur"]
    assert "outil inconnu" in tb.call("delete_all", {})["erreur"]
    assert tb.call("count", '{"denomination": "moulin"}')["total"] == 1    # arguments en chaîne JSON
    assert "non JSON" in tb.call("count", "{pas du json")["erreur"]


def test_specs_are_valid_and_search_hidden_without_retrievers(tb):
    names = [s["function"]["name"] for s in tb.specs]
    assert "search" not in names and set(names) == set(TOOL_NAMES) - {"search"}
    for s in tb.specs:
        assert s["type"] == "function" and s["function"]["parameters"]["type"] == "object"


def test_search_tool_with_retriever():
    from merimee_rag.corpus import Chunk
    from merimee_rag.retrievers import BM25Retriever

    chunks = [Chunk("TEST1002#0", "TEST1002", 0, "Église Notre-Dame-Imaginaire (Pontest)\nReconstruite par Jean Fictif."),
              Chunk("TEST1004#0", "TEST1004", 0, "Moulin du Gué (Riviertest)\nMoulin à eau."),
              # 3e et 4e passages : avec 2 documents seulement, l'IDF de BM25 vaut 0 pour tout mot présent une fois
              Chunk("TEST1003#0", "TEST1003", 0, "Château de Testmont (Testmont)\nBâti pour la famille."),
              Chunk("TEST1005#0", "TEST1005", 0, "Église Saint-Exemple (Riviertest)\nBelle église.")]
    t = Toolbox(MetadataStore.load(FIX / "mini_meta.json"), {"bm25": BM25Retriever(chunks)}, chunks)
    out = t.call("search", {"query": "qui a reconstruit l'église de Pontest", "method": "dense", "k": "3"})
    assert out["methode"] == "bm25" and out["resultats"][0]["ref"] == "TEST1002"
    assert "Reconstruite" in out["resultats"][0]["extrait"]
    assert out["resultats"][0]["auteurs"] == ["Jean Fictif (architecte)"]      # le nom de l'auteur est visible
    # cas réels (v1) : search(commune=…) et search(auteur=…) faisaient échouer l'appel
    out = t.call("search", {"query": "moulin", "commune": "Riviertest", "auteur": "x"})
    assert [r["ref"] for r in out["resultats"]] == ["TEST1004"] and "ignorés" in out["note"]
    assert t.call("search", {"query": "moulin", "commune": "Pontest"})["resultats"] == []


# ------------------------------------------------------------------ boucle d'agent
class ScriptedLLM:
    """Faux LLM : rejoue une suite de messages et garde ce qu'il a reçu."""
    model = "fake"

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat_message(self, messages, tools=None, **opts):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        return self.script.pop(0)


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_agent_loop_multi_step(tb):
    llm = ScriptedLLM([
        call("count", departement="Grand Est", denomination="église"),        # erreur → se corrige
        call("count", region="Grand Est", denomination="église"),
        {"role": "assistant", "content": "Il y a 2 églises protégées en Grand Est [TEST1001] [TEST1002]."},
    ])
    res = Agent(llm, tb, max_steps=5).answer("Combien d'églises en Grand Est ?")
    assert res.tools_used == ["count", "count"] and res.steps[0].error and not res.steps[1].error
    assert res.steps[1].result["total"] == 2 and res.cited_refs == ["TEST1001", "TEST1002"]
    assert res.status == "answer" and not res.hit_max_steps and res.n_llm_calls == 3
    # le résultat d'outil a bien été renvoyé au modèle, avec le nom de l'outil
    last_msgs = llm.seen[-1][0]
    assert last_msgs[-1]["role"] == "tool" and last_msgs[-1]["tool_name"] == "count"
    assert json.loads(last_msgs[-1]["content"])["total"] == 2
    assert json.dumps(res.to_dict(), ensure_ascii=False)


def test_agent_max_steps_forces_final_answer(tb):
    llm = ScriptedLLM([call("count", denomination="église")] * 2 + [{"role": "assistant", "content": "3 églises."}])
    res = Agent(llm, tb, max_steps=2).answer("Combien d'églises ?")
    assert res.hit_max_steps and res.answer == "3 églises." and llm.seen[-1][1] is None  # appel final sans outils


def test_agent_empty_answer_becomes_abstention(tb):
    llm = ScriptedLLM([{"role": "assistant", "content": ""}, {"role": "assistant", "content": ""}])
    res = Agent(llm, tb).answer("?")
    assert res.answer == ABSTAIN and res.abstained and res.n_llm_calls == 2  # une relance, puis abandon


def test_agent_is_nudged_to_use_a_tool_before_abstaining(tb):
    """Cas réel (v1) : « Je ne trouve pas… » sans aucun appel d'outil sur des questions factuelles."""
    llm = ScriptedLLM([{"role": "assistant", "content": ABSTAIN},
                       call("filter_notices", commune="Riviertest"),
                       {"role": "assistant", "content": "Le Moulin du Gué [TEST1004]."}])
    res = Agent(llm, tb).answer("Quels monuments à Riviertest ?")
    assert res.tools_used == ["filter_notices"] and not res.abstained
    assert "aucun outil" in llm.seen[1][0][-1]["content"]


def test_tool_call_written_in_text_is_recovered():
    msg = {"content": '<tool_call>{"name": "count", "arguments": {"denomination": "église"}}</tool_call>'}
    assert extract_tool_calls(msg) == [("count", {"denomination": "église"})]
    assert extract_tool_calls({"content": "Réponse normale."}) == []


# ------------------------------------------------------------------ notation et évaluation
def test_numbers_ignore_ordinals():
    from eval.agent_scoring import numbers

    assert numbers("Il y a 12 églises du 12e siècle, 1 234 notices et le 16ème ; bâti en 1519.") == {12, 1234, 1519}
    assert numbers("Aucune.") == set()


def test_score_types():
    from eval.agent_scoring import score

    assert score({"type": "count", "gold": 11}, "On compte 11 églises classées.", False)["score"] == 1.0
    assert score({"type": "count", "gold": 11}, "Il y en a 12.", False)["score"] == 0.0
    s = score({"type": "argmax", "gold": "Indre-et-Loire", "gold_count": 40},
              "C'est l'Indre-et-Loire, avec 40 châteaux.", False)
    assert s["score"] == 1.0
    s = score({"type": "multistep", "gold": "Bossu Jean", "gold_count": 5}, "L'architecte est Jean Bossu (3 œuvres).", False)
    assert s["score"] == 0.5 and s["name_ok"] and not s["count_ok"]
    q = {"type": "list", "gold": ["TEST1004", "TEST1005"], "gold_titles": ["Moulin du Gué", "Église Saint-Exemple"]}
    s = score(q, "Le Moulin du Gué [TEST1004] et le château X [PA00000001].", False)
    assert s["recall"] == 0.5 and s["precision"] == 0.5
    assert score(q, "Moulin du Gué et Église Saint-Exemple.", False)["score"] == 1.0   # titres recopiés
    assert score({"type": "ooc", "gold": None}, ABSTAIN, True)["score"] == 1.0
    assert score({"type": "count", "gold": 3}, ABSTAIN, True)["score"] == 0.0


def test_eval_agent_end_to_end(tb, tmp_path):
    """Boucle d'évaluation complète avec faux LLM scripté (agent) et faux RAG."""
    import pandas as pd

    from eval.eval_agent import run, summarize
    from merimee_rag.agent import AgentAnswer

    class FakeRAG:
        def answer(self, q):
            return AgentAnswer(q, "Je ne sais pas combien, peut-être 7.", status="answer")

    class Judge:
        model = "j"

        def chat(self, messages, json_mode=False, **o):
            return json.dumps({"verdict": "correct"})

    qs = [{"id": "c1", "type": "count", "question": "Combien d'églises dans l'Aube ?", "gold": 2,
           "expected_tool": "count"},
          {"id": "a1", "type": "argmax", "question": "Quelle région a le plus d'églises ?", "gold": "Grand Est",
           "gold_count": 2, "expected_tool": "count"}]
    agent = Agent(ScriptedLLM([
        call("count", denomination="église", departement="Aube"),
        {"role": "assistant", "content": "Il y a 2 églises."},
        call("count", denomination="église"),                      # oublie group_by
        {"role": "assistant", "content": "Je pense que c'est la Bourgogne."},
    ]), tb)
    recs = run(qs, {"rag": FakeRAG(), "agent": agent}, Judge(), {}, log=lambda *a, **k: None)
    by = {(r["system"], r["id"]): r for r in recs}
    assert by[("agent", "c1")]["score"] == 1.0 and by[("agent", "c1")]["diagnostic"] == "correct"
    assert by[("agent", "a1")]["tool_ok"] is False and by[("agent", "a1")]["diagnostic"] == "mauvais outil"
    assert by[("rag", "c1")]["score"] == 0.0
    s = summarize(pd.DataFrame(recs))
    tot = s[(s.system == "agent") & (s.type == "TOTAL")].iloc[0]
    assert tot.score == 0.5 and tot.n_tool_calls == 1.0


# ------------------------------------------------------------------ régressions observées avec qwen2.5:7b
def test_list_or_mongo_style_value_points_to_group_by(tb):
    """Cas réel : departement={"$in": [...]} au lieu de region + group_by."""
    e = tb.call("count", {"denomination": "château", "departement": {"$in": ["Aube", "Marne"]}})["erreur"]
    assert "une seule valeur" in e and "group_by" in e
    assert "group_by" in tb.call("count", {"departement": ["Aube", "Marne"]})["erreur"]


def test_invented_citation_detected(tb):
    llm = ScriptedLLM([call("count", denomination="église", departement="Aube"),
                       {"role": "assistant", "content": "2 églises [PA00098405]."}])
    res = Agent(llm, tb).answer("Combien d'églises dans l'Aube ?")
    assert res.unsupported_refs == ["PA00098405"]          # aucun outil ne l'a renvoyée
    llm = ScriptedLLM([call("filter_notices", commune="Riviertest"),
                       {"role": "assistant", "content": "Le moulin [TEST1004]."}])
    assert Agent(llm, tb).answer("?").unsupported_refs == []


def test_extra_filters_and_progress_callback(tb):
    from eval.agent_scoring import extra_filters

    q = {"type": "argmax", "filters": {"region": "Grand Est", "denomination": "église", "group_by": "departement"}}
    steps = [{"tool": "count", "args": {"region": "Grand Est", "denomination": "église", "annee_min": 1837}}]
    assert extra_filters(q, steps) == ["annee_min"]
    seen = []
    llm = ScriptedLLM([call("count", denomination="église"), {"role": "assistant", "content": "3."}])
    Agent(llm, tb).answer("?", on_step=lambda i, s: seen.append((i, s.tool)))
    assert seen == [(1, "count")]


def test_list_scoring_ignores_homonymous_titles():
    """Bug réel : « Église Saint-Valentin » existe ailleurs ; seule la notice attendue doit être créditée."""
    from eval.agent_scoring import extra_filters, score

    q = {"type": "list", "gold": ["PA00132863", "PA70000116"],
         "gold_titles": ["Maison-forte Vers le Village", "Église Saint-Valentin"]}
    ans = "- PA00132863 : Maison-forte Vers le Village - PA70000116 : Église Saint-Valentin"
    homonymes = {"PA00000001": "Église Saint-Valentin", "PA00000002": "Maison-forte Vers le Village"}
    assert score(q, ans, False, homonymes)["score"] == 1.0
    # champs envoyés vides par le modèle : ce ne sont pas des filtres inventés
    steps = [{"tool": "count", "args": {"departement": "Aube", "auteur": None, "mot_cle": ""}}]
    assert extra_filters({"filters": {"departement": "Aube"}}, steps) == []


# ------------------------------------------------------------------ régressions v2 (analyse d'erreurs)
def _search_toolbox():
    from merimee_rag.corpus import Chunk
    from merimee_rag.retrievers import BM25Retriever

    chunks = [Chunk("TEST1002#0", "TEST1002", 0, "Église Notre-Dame-Imaginaire (Pontest)\nReconstruite par Jean Fictif."),
              Chunk("TEST1004#0", "TEST1004", 0, "Moulin du Gué (Riviertest)\nMoulin à eau."),
              Chunk("TEST1003#0", "TEST1003", 0, "Château de Testmont (Testmont)\nBâti pour la famille."),
              Chunk("TEST1005#0", "TEST1005", 0, "Église Saint-Exemple (Riviertest)\nBelle église.")]
    return Toolbox(MetadataStore.load(FIX / "mini_meta.json"), {"bm25": BM25Retriever(chunks)}, chunks)


def test_search_without_query_rebuilds_it():
    # v2 : search(commune=…, mot_cle=…) sans query → 92 % d'appels en erreur sur les questions factuelles
    t = _search_toolbox()
    out = t.call("search", {"commune": "Riviertest", "mot_cle": "moulin", "annee_min": 1780})
    assert "erreur" not in out and out["resultats"][0]["ref"] == "TEST1004"
    assert "query absente" in out["note"]
    assert "erreur" in t.call("search", {})


def test_search_unknown_place_is_added_to_query_not_an_error():
    # v2 : search(query=…, commune="Girolata") → « commune inconnue » (hameau, pas une commune)
    out = _search_toolbox().call("search", {"query": "moulin", "commune": "Hameau-Imaginaire"})
    assert "erreur" not in out and out["resultats"] and "ajouté à la requête" in out["note"]


def test_search_spec_exposes_only_query_method_k():
    spec = next(t for t in TOOL_SPECS if t["function"]["name"] == "search")
    assert set(spec["function"]["parameters"]["properties"]) == {"query", "method", "k"}


def test_protection_tous_is_not_a_filter(tb):
    from eval.agent_scoring import extra_filters

    # v2 : « monuments protégés » → protection="classé" inventé dans 11 listes sur 12
    total = tb.call("count", {})["total"]
    for v in ("tous", "protégé", "Tous"):
        assert tb.call("count", {"protection": v})["total"] == total
    q = {"type": "list", "filters": {"commune": "Riviertest"}}
    assert extra_filters(q, [{"tool": "filter_notices", "args": {"commune": "Riviertest", "protection": "tous"}}]) == []


def test_homonymous_commune_warning(tb):
    out = tb.call("filter_notices", {"commune": "Homotest"})
    assert "attention" in out and "Aube" in out["attention"] and "Cher" in out["attention"]
    assert "attention" not in tb.call("filter_notices", {"commune": "Homotest", "departement": "Aube"})
    assert "attention" not in tb.call("count", {"commune": "Riviertest"})
