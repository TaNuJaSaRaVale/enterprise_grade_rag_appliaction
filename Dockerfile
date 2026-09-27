# Enterprise Agentic RAG — FastAPI backend (private :8000) + Streamlit UI (public :7860)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# uv: fast dependency installs (pinned to the version used locally)
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

# Hugging Face Spaces runs containers as uid 1000 — files must belong to that user
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Dependencies first: this slow layer stays cached when only code changes
COPY requirements-app.txt .
RUN uv pip install --system --no-cache -r requirements-app.txt

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# Bake the reranker model into the image so the first request doesn't download it
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/tmp/flashrank')"

COPY --chown=user app ./app
COPY --chown=user ui ./ui
COPY --chown=user start.sh .

EXPOSE 7860
CMD ["./start.sh"]
