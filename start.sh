#!/usr/bin/env bash
# Runs the private API and the public UI in one container.
set -euo pipefail

# API: localhost only — the only way in is through the UI
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# UI: the single public port (Hugging Face Spaces expects 7860)
streamlit run ui/app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &

# If either process dies, exit so the platform restarts the container
# instead of serving a half-working app
wait -n
exit $?
