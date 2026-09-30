"""Configuration centrale (chemins, modèles, hyperparamètres).

Tout est surchargeable par variables d'environnement pour éviter de toucher au code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- Données -----------------------------------------------------------
    raw_csv: Path = field(default_factory=lambda: Path(_env("MERIMEE_RAW_CSV", str(ROOT / "data" / "raw" / "merimee.csv"))))
    raw_json: Path = field(default_factory=lambda: Path(_env("MERIMEE_RAW_JSON", str(ROOT / "data" / "raw" / "monuments.json"))))
    corpus_path: Path = field(default_factory=lambda: Path(_env("MERIMEE_CORPUS", str(ROOT / "data" / "corpus.jsonl"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("MERIMEE_INDEX_DIR", str(ROOT / "data" / "index"))))
    bundled_json: Path = field(default_factory=lambda: ROOT / "data" / "monuments.json.gz")
    min_historique_chars: int = int(_env("MERIMEE_MIN_HIST", "80"))

    # --- Chunking ------------------------------------------------------------
    chunk_words: int = int(_env("MERIMEE_CHUNK_WORDS", "180"))
    chunk_overlap: int = int(_env("MERIMEE_CHUNK_OVERLAP", "40"))

    # --- Retrieval -----------------------------------------------------------
    embed_model: str = _env("MERIMEE_EMBED_MODEL", "intfloat/multilingual-e5-small")
    embed_batch_size: int = int(_env("MERIMEE_EMBED_BATCH", "64"))
    rrf_k: int = 60
    candidates: int = 100  # nb de candidats par retriever avant fusion

    # --- Génération (Ollama) ---------------------------------------------------
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    gen_model: str = _env("MERIMEE_GEN_MODEL", "qwen2.5:3b")
    judge_model: str = _env("MERIMEE_JUDGE_MODEL", "qwen2.5:7b")
    agent_model: str = _env("MERIMEE_AGENT_MODEL", "qwen2.5:7b")
    agent_max_steps: int = int(_env("MERIMEE_AGENT_MAX_STEPS", "6"))
    top_k_context: int = int(_env("MERIMEE_TOPK_CTX", "5"))
    temperature: float = 0.0
    llm_cache: Path = field(default_factory=lambda: Path(_env("MERIMEE_LLM_CACHE", str(ROOT / "data" / "llm_cache.jsonl"))))

    # --- Solr (optionnel : index du cours) --------------------------------------
    solr_url: str | None = os.environ.get("MERIMEE_SOLR_URL")  # ex. http://localhost:8983/solr/merimee

    # --- Évaluation -----------------------------------------------------------
    results_dir: Path = field(default_factory=lambda: ROOT / "results")
    eval_dir: Path = field(default_factory=lambda: ROOT / "eval")
    seed: int = 42


def metadata_path(cfg: "Config") -> Path:
    """Base complète pour les filtres/comptages de l'agent : JSON brut s'il existe, sinon celui du dépôt."""
    return cfg.raw_json if cfg.raw_json.exists() else cfg.bundled_json


ABSTAIN = "Je ne trouve pas cette information dans la base Mérimée."
