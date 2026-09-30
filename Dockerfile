# Image de l'application : démo Streamlit + CLI (RAG et agent). Ollama tourne dans son propre conteneur.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    MERIMEE_CORPUS=/data/corpus.jsonl \
    MERIMEE_INDEX_DIR=/data/index \
    MERIMEE_LLM_CACHE=/data/llm_cache.jsonl \
    OLLAMA_URL=http://ollama:11434

WORKDIR /app

# PyTorch en version CPU : l'image pèse ~2 Go au lieu de ~6 (les embeddings tournent très bien sur CPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dépendances d'abord (couche mise en cache tant que pyproject.toml ne change pas)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY app.py ./
COPY eval ./eval
COPY data/monuments.json.gz ./data/monuments.json.gz
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
