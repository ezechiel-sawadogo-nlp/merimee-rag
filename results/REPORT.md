# Résultats — RAG Mérimée

Corpus : **23 442 notices** avec historique → **26 687 chunks** (180 mots, recouvrement 40). Embeddings : `intfloat/multilingual-e5-small`.

Moyennes avec intervalle de confiance à 95 % (bootstrap, 2 000 tirages) entre crochets.

## Retrieval

**115 questions bien posées** (sur 141 générées : les questions sous-spécifiées, qui ne nomment ni le monument ni la commune, sont écartées — voir `eval/ambiguity.py`).

![retrieval](retrieval.png)

| retriever | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| **bm25** | 0.852 <sub>[0.78–0.91]</sub> | 0.965 <sub>[0.93–0.99]</sub> | 0.965 <sub>[0.93–0.99]</sub> | 0.903 <sub>[0.86–0.94]</sub> | 0.919 <sub>[0.88–0.96]</sub> |
| **dense** | 0.843 <sub>[0.77–0.90]</sub> | 0.922 <sub>[0.87–0.97]</sub> | 0.957 <sub>[0.91–0.99]</sub> | 0.882 <sub>[0.83–0.93]</sub> | 0.900 <sub>[0.85–0.94]</sub> |
| **hybrid** | 0.896 <sub>[0.83–0.95]</sub> | 0.965 <sub>[0.93–0.99]</sub> | 0.974 <sub>[0.94–1.00]</sub> | 0.925 <sub>[0.88–0.96]</sub> | 0.937 <sub>[0.90–0.97]</sub> |


<details><summary>Sur les 141 questions, y compris sous-spécifiées</summary>

| retriever | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| **bm25** | 0.816 <sub>[0.75–0.88]</sub> | 0.922 <sub>[0.87–0.96]</sub> | 0.929 <sub>[0.89–0.97]</sub> | 0.861 <sub>[0.81–0.91]</sub> | 0.878 <sub>[0.83–0.92]</sub> |
| **dense** | 0.730 <sub>[0.65–0.80]</sub> | 0.816 <sub>[0.75–0.87]</sub> | 0.872 <sub>[0.82–0.92]</sub> | 0.773 <sub>[0.71–0.83]</sub> | 0.797 <sub>[0.74–0.85]</sub> |
| **hybrid** | 0.794 <sub>[0.73–0.86]</sub> | 0.894 <sub>[0.84–0.94]</sub> | 0.922 <sub>[0.87–0.96]</sub> | 0.839 <sub>[0.78–0.89]</sub> | 0.859 <sub>[0.81–0.91]</sub> |

</details>


**Comparaisons appariées (MRR@10, bootstrap apparié)**

| questions | A | B | Δ (A−B) | p |
|---|---|---|---|---|
| toutes | bm25 | dense | +0.088 | 0.002 |
| toutes | bm25 | hybrid | +0.023 | 0.235 |
| toutes | dense | hybrid | -0.065 | 0.001 |
| propres | bm25 | dense | +0.021 | 0.390 |
| propres | bm25 | hybrid | -0.022 | 0.138 |
| propres | dense | hybrid | -0.043 | 0.028 |


**MRR@10 selon le recouvrement lexical question/notice** (contrôle du biais des questions synthétiques en faveur de BM25)

| recouvrement | bm25 | dense | hybrid |
|---|---|---|---|
| faible | 0.816 | 0.837 | 0.841 |
| moyen | 0.877 | 0.794 | 0.873 |
| fort | 0.893 | 0.688 | 0.805 |

## Génération

40 questions répondables bien posées + 15 hors corpus. « Abstention hors corpus » n'est pas mesurée pour *no-rag*, dont la consigne ne demande pas de refuser.

![generation](generation.png)

| config | Exactitude | Ancrage (notice) | Fidélité (extraits) | Citation juste | Abstention à tort | Abstention hors corpus | Réponse hybride (refus + faits) | Latence |
|---|---|---|---|---|---|---|---|---|
| **no-rag** | 0.362 <sub>[0.26–0.46]</sub> | 0.680 <sub>[0.59–0.76]</sub> | – | – | 0.000 <sub>[0.00–0.00]</sub> | – | 0.000 <sub>[0.00–0.00]</sub> | – |
| **rag-bm25** | 0.863 <sub>[0.78–0.94]</sub> | 0.928 <sub>[0.86–0.98]</sub> | 0.913 <sub>[0.84–0.97]</sub> | 0.950 <sub>[0.88–1.00]</sub> | 0.025 <sub>[0.00–0.07]</sub> | 0.867 <sub>[0.67–1.00]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 1.5 s |
| **rag-dense** | 0.825 <sub>[0.72–0.91]</sub> | 0.863 <sub>[0.78–0.94]</sub> | 0.885 <sub>[0.80–0.96]</sub> | 0.900 <sub>[0.80–0.97]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 0.933 <sub>[0.80–1.00]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 1.6 s |
| **rag-hybrid** | 0.863 <sub>[0.79–0.94]</sub> | 0.918 <sub>[0.85–0.97]</sub> | 0.850 <sub>[0.74–0.94]</sub> | 0.950 <sub>[0.88–1.00]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 0.600 <sub>[0.33–0.87]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 1.5 s |


<details><summary>Y compris les questions sous-spécifiées</summary>

| config | Exactitude | Ancrage (notice) | Fidélité (extraits) | Citation juste | Abstention à tort | Abstention hors corpus | Réponse hybride (refus + faits) | Latence |
|---|---|---|---|---|---|---|---|---|
| **no-rag** | 0.350 <sub>[0.26–0.45]</sub> | 0.652 <sub>[0.56–0.73]</sub> | – | – | 0.020 <sub>[0.00–0.06]</sub> | – | 0.000 <sub>[0.00–0.00]</sub> | – |
| **rag-bm25** | 0.790 <sub>[0.69–0.87]</sub> | 0.866 <sub>[0.78–0.94]</sub> | 0.897 <sub>[0.82–0.96]</sub> | 0.880 <sub>[0.78–0.96]</sub> | 0.020 <sub>[0.00–0.06]</sub> | 0.867 <sub>[0.67–1.00]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 1.5 s |
| **rag-dense** | 0.750 <sub>[0.65–0.84]</sub> | 0.796 <sub>[0.70–0.89]</sub> | 0.838 <sub>[0.75–0.92]</sub> | 0.760 <sub>[0.64–0.88]</sub> | 0.020 <sub>[0.00–0.06]</sub> | 0.933 <sub>[0.80–1.00]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 1.7 s |
| **rag-hybrid** | 0.780 <sub>[0.69–0.87]</sub> | 0.845 <sub>[0.75–0.93]</sub> | 0.835 <sub>[0.74–0.92]</sub> | 0.840 <sub>[0.74–0.94]</sub> | 0.020 <sub>[0.00–0.06]</sub> | 0.600 <sub>[0.33–0.87]</sub> | 0.000 <sub>[0.00–0.00]</sub> | 1.5 s |

</details>


**Analyse d'erreurs** (toutes questions répondables : d'où vient l'échec ?)

| config | abstention à tort | correct | échec génération | échec retrieval |
|---|---|---|---|---|
| rag-bm25 | 1 | 33 | 12 | 4 |
| rag-dense | 0 | 30 | 12 | 8 |
| rag-hybrid | 0 | 33 | 11 | 6 |
