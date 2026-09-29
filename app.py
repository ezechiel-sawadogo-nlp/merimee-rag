"""Démo Streamlit : `streamlit run app.py`"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from merimee_rag.config import Config
from merimee_rag.llm import DiskCache, OllamaLLM
from merimee_rag.pipeline import RAGPipeline
from merimee_rag.store import System

st.set_page_config(page_title="Mérimée RAG", page_icon="🏛️", layout="wide")
cfg = Config()

EXAMPLES = ["Qui a fait construire le château de Chambord ?",
            "Quel architecte a dessiné la flèche du clocher de l'église d'Avirey-Lingey ?",
            "Quand la Grande Mosquée de Djenné a-t-elle été reconstruite ?"]


@st.cache_resource(show_spinner="Chargement des index…")
def load_system() -> System:
    return System(cfg)


system = load_system()
by_ref = {n.ref: n for n in system.notices}

# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    st.header("Réglages")
    names = [*system.retrievers, "none"]
    retr_name = st.radio("Retriever", names, index=names.index("hybrid") if "hybrid" in names else 0,
                         help="bm25 = mots-clés · dense = sens (embeddings) · hybrid = fusion des deux · "
                              "none = le LLM répond sans la base (pour comparer)")
    model = st.text_input("Modèle Ollama", cfg.gen_model)
    top_k = st.slider("Extraits fournis au LLM", 1, 10, cfg.top_k_context)
    st.caption(f"{len(system.notices):,} notices · {len(system.chunks):,} passages indexés".replace(",", " "))

# ------------------------------------------------------------------ question
st.title("🏛️ Interroger la base Mérimée")
st.caption("Monuments historiques protégés — réponses générées par un LLM local, citant les notices sources.")


def _view(df: pd.DataFrame):
    """Cadre la carte sur les points : zoom déduit de l'étendue, borné entre la France entière et le village."""
    import pydeck as pdk

    span = max(df["lat"].max() - df["lat"].min(), (df["lon"].max() - df["lon"].min()) * 0.7, 0.05)
    zoom = max(1.0, min(9.0, math.log2(360 / span) - 1.3))
    return pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=zoom)


def _map(df: pd.DataFrame):
    """Points de taille fixe à l'écran (pixels) : lisibles quel que soit le zoom."""
    import pydeck as pdk

    # Attention : pydeck transforme toute chaîne en expression (radius_units="pixels" devient
    # "@@=pixels" et est ignoré). On fixe donc la taille à l'écran avec des *nombres* :
    # rayon de 1 m, forcé à N pixels minimum → taille constante quel que soit le zoom.
    layers = [pdk.Layer("ScatterplotLayer", part, get_position="[lon, lat]", get_fill_color="color",
                        get_radius=1, radius_min_pixels=px, radius_max_pixels=px, stroked=True,
                        get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True)
              for part, px in ((df[~df["cited"]], 6), (df[df["cited"]], 9))  # cités dessinés par-dessus
              if len(part)]
    return pdk.Deck(layers=layers, initial_view_state=_view(df), tooltip={"text": "{label}\n{lieu}"})


def _use_example(text: str) -> None:
    st.session_state["question"] = text  # remplit vraiment le champ, au lieu d'un simple placeholder


st.text_input("Question", key="question", placeholder="Pose ta question sur un monument…")
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    c.button(ex, on_click=_use_example, args=(ex,))

q = st.session_state.get("question", "").strip()

# ------------------------------------------------------------------ réponse
if q:
    llm = OllamaLLM(model, cfg.ollama_url, cfg.temperature, DiskCache(cfg.llm_cache))
    rag = RAGPipeline(None if retr_name == "none" else system.retrievers[retr_name],
                      system.chunks, system.notices, llm, top_k)
    with st.spinner("Recherche et génération…"):
        try:
            res = rag.answer(q)
        except Exception as e:  # Ollama éteint, modèle absent…
            st.error(f"{e}\n\nVérifie qu'Ollama tourne (`ollama serve`) et que le modèle est installé "
                     f"(`ollama pull {model}`).")
            st.stop()

    st.markdown("### Réponse")
    st.markdown(f"> **{q}**")
    if res.abstained:
        st.info(res.answer)
    else:
        st.markdown(res.answer)
    if res.status == "partial":
        st.warning("Réponse hybride : le modèle dit ne pas trouver l'information mais affirme quand même "
                   "des faits. Vérifie-les dans les sources.")
    if retr_name == "none":
        st.warning("Mode sans base : cette réponse vient de la mémoire du modèle et n'est vérifiable dans "
                   "aucune source.")
    if res.raw and res.raw != res.answer:
        with st.expander("Raisonnement du modèle (sortie brute)"):
            st.text(res.raw)
    timing = "réponse en cache" if llm.last_from_cache else f"{res.latency_s:.1f} s"
    st.caption(f"{timing} · {retr_name} · {model}")

    if res.contexts:
        # Plusieurs passages peuvent venir de la même notice : on les regroupe.
        groups: dict[str, list] = {}
        for c in res.contexts:
            groups.setdefault(c.ref, []).append(c)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Sources")
            for ref, ctxs in groups.items():
                nums = [c.n for c in ctxs]
                is_cited = any(n in res.cited for n in nums)
                label = " ".join(f"[{n}]" for n in nums)
                badge = "✅ citée" if is_cited else "non citée"
                with st.expander(f"{label} {ctxs[0].title} — {badge}", expanded=is_cited):
                    header = ctxs[0].text.split("\n", 1)[0]
                    st.markdown(f"**{header}**  \n[Notice {ref} sur POP ↗]({ctxs[0].url})")
                    for c in ctxs:
                        body = c.text.split("\n", 1)[-1]
                        st.markdown(f"**[{c.n}]** {body}")
        with right:
            pts = []
            for ref, ctxs in groups.items():
                n = by_ref.get(ref)
                if n and n.lat is not None and n.lon is not None:
                    cited = any(c.n in res.cited for c in ctxs)
                    pts.append({"lat": n.lat, "lon": n.lon, "cited": cited,
                                "label": " ".join(f"[{c.n}]" for c in ctxs) + f" {ctxs[0].title}",
                                "lieu": ", ".join(x for x in (n.commune, n.departement) if x),
                                "color": [227, 73, 72, 230] if cited else [138, 137, 132, 200]})
            if pts:
                st.markdown("#### Localisation")
                st.pydeck_chart(_map(pd.DataFrame(pts)))
                missing = len(groups) - len(pts)
                note = f" · {missing} source(s) sans coordonnées" if missing else ""
                st.caption(f"🔴 notice citée · ⚪ récupérée mais non citée · survole un point pour son nom{note}")
            else:
                st.caption("Aucune des sources n'est géolocalisée.")
