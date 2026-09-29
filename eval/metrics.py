"""Métriques de retrieval + intervalles de confiance par bootstrap."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return float(any(r in gold for r in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], 1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Moyenne et IC à (1-alpha) par bootstrap percentile."""
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_pvalue(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0) -> float:
    """p-valeur bilatérale (bootstrap apparié) pour H0 : mean(a) == mean(b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    centered = d - d.mean()
    boots = centered[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float((np.abs(boots) >= abs(d.mean())).mean())
