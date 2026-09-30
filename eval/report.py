"""Assemble results/REPORT.md + figures à partir des CSV produits par les évaluations.

Usage : python -m eval.report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from merimee_rag.config import Config  # noqa: E402

# Une couleur par *système*, identique dans toutes les figures (la couleur suit l'entité).
COLORS = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a", "solr": "#4a3aa7",
          "no-rag": "#8a8984", "rag-bm25": "#2a78d6", "rag-dense": "#eb6834", "rag-hybrid": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"


def grouped_bars(summary: pd.DataFrame, key: str, metrics: list[str], labels: dict[str, str],
                 title: str, out: Path) -> None:
    summary = summary.dropna(axis=1, how="all")
    metrics = [m for m in metrics if m in summary]
    systems = summary[key].tolist()
    n = len(systems)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(metrics), 3.6), dpi=160)
    for i, (_, row) in enumerate(summary.iterrows()):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(metrics))]
        vals = [row[m] for m in metrics]
        err = [[row[m] - row.get(f"{m}_lo", row[m]) for m in metrics],
               [row.get(f"{m}_hi", row[m]) - row[m] for m in metrics]]
        ax.bar(xs, vals, width * 0.92, color=COLORS.get(row[key], "#888"), label=row[key],
               edgecolor="white", linewidth=1)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(range(len(metrics)), [labels.get(m, m) for m in metrics], color=INK2, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK2, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=8, ncol=n, loc="upper left", bbox_to_anchor=(0, -0.12), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, key: str, metrics: list[str], labels: dict[str, str]) -> str:
    metrics = [m for m in metrics if m in df and df[m].notna().any()]
    head = f"| {key} | " + " | ".join(labels.get(m, m) for m in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for m in metrics:
            v = r[m]
            if pd.isna(v):
                cells.append("–")
            elif m == "latency_s":
                cells.append(f"{v:.1f} s")
            else:
                cells.append(f"{v:.3f} <sub>[{r[f'{m}_lo']:.2f}–{r[f'{m}_hi']:.2f}]</sub>")
        lines.append(f"| **{r[key]}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


RET_LABELS = {"hit@1": "Hit@1", "hit@5": "Hit@5", "recall@10": "Recall@10", "mrr@10": "MRR@10", "ndcg@10": "nDCG@10"}
GEN_LABELS = {"correctness": "Exactitude", "grounded_gold": "Ancrage (notice)", "faithful_ctx": "Fidélité (extraits)",
              "citation_hit": "Citation juste", "false_abstention": "Abstention à tort",
              "correct_abstention": "Abstention hors corpus", "partial": "Réponse hybride (refus + faits)",
              "latency_s": "Latence"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = Config()
    R = cfg.results_dir
    parts = ["# Résultats — RAG Mérimée\n"]
    meta_p = cfg.index_dir / "meta.json"
    if meta_p.exists():
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        fr = lambda x: f"{x:,}".replace(",", " ")  # noqa: E731
        parts.append(f"Corpus : **{fr(m['n_notices'])} notices** avec historique → **{fr(m['n_chunks'])} chunks** "
                     f"({m['chunk_words']} mots, recouvrement {m['chunk_overlap']}). Embeddings : `{m['embed_model']}`.\n")
    parts.append("Moyennes avec intervalle de confiance à 95 % (bootstrap, 2 000 tirages) entre crochets.\n")

    if (R / "retrieval_summary.csv").exists():
        s_all = pd.read_csv(R / "retrieval_summary.csv")
        main_sub = "propres" if "subset" in s_all and (s_all["subset"] == "propres").any() else None
        s = s_all[s_all["subset"] == main_sub] if main_sub else s_all
        grouped_bars(s, "retriever", list(RET_LABELS), RET_LABELS, "Retrieval (niveau notice)", R / "retrieval.png")
        parts += ["## Retrieval\n"]
        if main_sub:
            n_all = int(s_all[s_all["subset"] == "toutes"]["n"].iloc[0])
            parts.append(f"**{int(s['n'].iloc[0])} questions bien posées** (sur {n_all} générées : les questions "
                         "sous-spécifiées, qui ne nomment ni le monument ni la commune, sont écartées — "
                         "voir `eval/ambiguity.py`).\n")
        else:
            parts.append(f"{int(s['n'].iloc[0])} questions répondables.\n")
        parts += ["![retrieval](retrieval.png)\n", md_table(s, "retriever", list(RET_LABELS), RET_LABELS), ""]
        if main_sub:
            t = s_all[s_all["subset"] == "toutes"]
            parts += [f"\n<details><summary>Sur les {n_all} questions, y compris sous-spécifiées</summary>\n",
                      md_table(t, "retriever", list(RET_LABELS), RET_LABELS), "\n</details>\n"]
        if (R / "retrieval_pairwise.csv").exists():
            pw = pd.read_csv(R / "retrieval_pairwise.csv")
            has_sub = "subset" in pw
            parts += ["\n**Comparaisons appariées (MRR@10, bootstrap apparié)**\n",
                      "| questions | A | B | Δ (A−B) | p |", "|---|---|---|---|---|"]
            parts += [f"| {r.subset if has_sub else 'toutes'} | {r.A} | {r.B} | {r['delta(A-B)']:+.3f} | {r.p_value:.3f} |"
                      for _, r in pw.iterrows()]
            parts.append("")
        if (R / "retrieval_by_overlap.csv").exists():
            ov = pd.read_csv(R / "retrieval_by_overlap.csv", index_col=0)
            parts += ["\n**MRR@10 selon le recouvrement lexical question/notice** "
                      "(contrôle du biais des questions synthétiques en faveur de BM25)\n",
                      "| recouvrement | " + " | ".join(ov.columns) + " |", "|---|" + "---|" * len(ov.columns)]
            parts += [f"| {i} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for i, r in ov.iterrows()]
            parts.append("")

    if (R / "generation_summary.csv").exists():
        g_all = pd.read_csv(R / "generation_summary.csv")
        main_sub = "propres" if "subset" in g_all and (g_all["subset"] == "propres").any() else None
        g = g_all[g_all["subset"] == main_sub] if main_sub else g_all
        plot_m = ["correctness", "grounded_gold", "faithful_ctx", "citation_hit", "correct_abstention"]
        grouped_bars(g, "config", plot_m, GEN_LABELS, "Génération (LLM-juge)", R / "generation.png")
        parts += ["## Génération\n",
                  f"{int(g['n_answerable'].iloc[0])} questions répondables"
                  f"{' bien posées' if main_sub else ''} + {int(g['n_ooc'].iloc[0])} hors corpus. "
                  "« Abstention hors corpus » n'est pas mesurée pour *no-rag*, dont la consigne ne demande pas "
                  "de refuser.\n",
                  "![generation](generation.png)\n", md_table(g, "config", list(GEN_LABELS), GEN_LABELS), ""]
        if main_sub:
            t = g_all[g_all["subset"] == "toutes"]
            parts += ["\n<details><summary>Y compris les questions sous-spécifiées</summary>\n",
                      md_table(t, "config", list(GEN_LABELS), GEN_LABELS), "\n</details>\n"]
        if (R / "generation_errors.csv").exists():
            e = pd.read_csv(R / "generation_errors.csv", index_col=0)
            parts += ["\n**Analyse d'erreurs** (toutes questions répondables : d'où vient l'échec ?)\n",
                      "| config | " + " | ".join(e.columns) + " |", "|---|" + "---|" * len(e.columns)]
            parts += [f"| {i} | " + " | ".join(str(int(v)) for v in r) + " |" for i, r in e.iterrows()]
            parts.append("")

    if (R / "agent_summary.csv").exists():
        a = pd.read_csv(R / "agent_summary.csv")
        types = [t for t in ["count", "list", "argmax", "multistep", "lookup", "ooc", "TOTAL"] if t in set(a["type"])]
        labels = {"count": "Comptage", "list": "Liste", "argmax": "Maximum", "multistep": "Multi-étapes",
                  "lookup": "Fait (historique)", "ooc": "Hors corpus", "TOTAL": "Total"}
        wide = []
        for sys_name, g in a.groupby("system", sort=False):
            row = {"system": {"rag": "rag-hybrid", "agent": "agent"}.get(sys_name, sys_name)}
            for _, r in g.iterrows():
                row[r["type"]], row[f"{r['type']}_lo"], row[f"{r['type']}_hi"] = r["score"], r["score_lo"], r["score_hi"]
            wide.append(row)
        wide = pd.DataFrame(wide)
        COLORS.setdefault("agent", "#4a3aa7")
        grouped_bars(wide, "system", types, labels, "RAG fixe vs agent (même modèle)", R / "agent.png")
        parts += ["## RAG fixe vs agent\n",
                  f"{int(a[a.type == 'TOTAL']['n'].iloc[0])} questions. Même modèle pour les deux systèmes ; "
                  "notation automatique sur des réponses calculées à partir des données (juge LLM pour "
                  "« Fait »).\n", "![agent](agent.png)\n", md_table(wide, "system", types, labels), ""]
        ag = a[a.system == "agent"].set_index("type")
        if not ag.empty and "tool_ok" in ag:
            parts += ["\n**Comportement de l'agent**\n",
                      "| type | bon outil | appels d'outils | appels en erreur | filtres inventés | "
                      "citations inventées | étapes épuisées | latence |",
                      "|---|---|---|---|---|---|---|---|"]
            for t in types:
                if t in ag.index and t != "TOTAL":
                    r = ag.loc[t]
                    fmt = lambda v, p=True: "–" if pd.isna(v) else (f"{v:.0%}" if p else f"{v:.1f}")  # noqa: E731
                    parts.append(f"| {labels[t]} | {fmt(r.get('tool_ok'))} | {fmt(r.get('n_tool_calls'), False)} | "
                                 f"{fmt(r.get('tool_error_rate'))} | {fmt(r.get('extra_filter_rate'))} | "
                                 f"{fmt(r.get('invented_citation_rate'))} | {fmt(r.get('max_steps_rate'))} | "
                                 f"{r['latency_s']:.1f} s |")
            parts.append("")
        v1p = R / "agent_v1_summary.csv"
        if v1p.exists():
            v1 = pd.read_csv(v1p)
            v1 = v1[v1.system == "agent"].set_index("type")["score"]
            v2 = a[a.system == "agent"].set_index("type")["score"]
            rag = a[a.system == "rag"].set_index("type")["score"]
            parts += ["\n**Itération guidée par l'analyse d'erreurs** (agent v1 → v2, même modèle, mêmes questions)\n",
                      "| type | RAG fixe | agent v1 | agent v2 |", "|---|---|---|---|"]
            parts += [f"| {labels[t]} | {rag.get(t, float('nan')):.2f} | {v1.get(t, float('nan')):.2f} | "
                      f"**{v2.get(t, float('nan')):.2f}** |" for t in types]
            parts.append("")
        if (R / "agent_errors.csv").exists():
            e = pd.read_csv(R / "agent_errors.csv", index_col=0)
            parts += ["\n**Diagnostic de chaque réponse de l'agent**\n",
                      "| type | " + " | ".join(e.columns) + " |", "|---|" + "---|" * len(e.columns)]
            parts += [f"| {labels.get(i, i)} | " + " | ".join(str(int(v)) for v in r) + " |" for i, r in e.iterrows()]
            parts.append("")

    (R / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"→ {R / 'REPORT.md'}")


if __name__ == "__main__":
    main()
