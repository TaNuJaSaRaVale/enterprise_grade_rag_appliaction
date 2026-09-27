# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

import threading
import time
import uuid
from collections import deque
from pydantic import BaseModel, Field
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")

# Global limit protects the free-tier LLM quotas on a public demo. Global (not per-IP):
# in the deployed container every request arrives from the Streamlit UI on localhost.
RATE_LIMIT_PER_MIN = 20
_recent_requests: deque = deque()
_rate_lock = threading.Lock()  # sync endpoints run in a threadpool


def _rate_limited() -> bool:
    with _rate_lock:
        now = time.monotonic()
        while _recent_requests and now - _recent_requests[0] >= 60:
            _recent_requests.popleft()
        if len(_recent_requests) >= RATE_LIMIT_PER_MIN:
            return True
        _recent_requests.append(now)
        return False


# Thresholds on the cross-encoder (0-1) score of the best chunk, calibrated on this corpus:
# chunks that answered golden questions scored 0.98-0.99; unrelated chunks scored ~0.0.
CONFIDENCE_HIGH = 0.7
CONFIDENCE_MEDIUM = 0.3


def _confidence(final_output: dict) -> dict:
    """How well the documentation supports this answer — derived from retrieval, not the LLM."""
    if final_output.get("current_query") == "CONVERSATIONAL":
        return {"level": "n/a", "top_score": None, "reason": "Conversational reply — no documents needed."}
    scores = final_output.get("retrieval_scores") or []
    if not scores:
        return {"level": "Not in docs", "top_score": None,
                "reason": "No relevant documentation found — any guidance is general knowledge."}
    top = max(scores)
    if top >= CONFIDENCE_HIGH:
        level, reason = "High", "A retrieved passage closely matches the question."
    elif top >= CONFIDENCE_MEDIUM:
        level, reason = "Medium", "Retrieved passages are related but may not fully answer the question."
    else:
        level, reason = "Low", "Only weakly related passages were found — verify before relying on this."
    return {"level": level, "top_score": round(top, 3), "reason": reason}


def _source_files(documents: list[str]) -> list[str]:
    """Unique file names from '[n] Source: <file>\\n<text>' documents, in rank order."""
    files = []
    for doc in documents:
        first_line = doc.split("\n", 1)[0]
        name = first_line.split("Source:", 1)[-1].strip() if "Source:" in first_line else None
        if name and name not in files:
            files.append(name)
    return files


@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    # Bounded input: huge prompts would burn LLM quota
    q: str = Field(min_length=1, max_length=1000)
    # No shared default — a missing thread_id gets its own fresh memory thread
    thread_id: Optional[str] = Field(default=None, max_length=100)


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health")
def health():
    """Liveness check for the container platform. Deliberately makes no LLM/DB calls."""
    return {"status": "ok"}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id or str(uuid.uuid4())

    if _rate_limited():
        logfire.warning("⏳ Global rate limit hit — request rejected")
        return JSONResponse(status_code=429, content={
            "question": q,
            "answer": "The demo is receiving a lot of requests right now. Please try again in a minute.",
            "thought_process": ["Rate limited"],
            "status": "rate_limited",
            "sources": []
        })

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "retrieval_scores": [],  # reset per turn — MemorySaver would otherwise carry the last turn's scores
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "source_files": [],
                "confidence": {"level": "n/a", "top_score": None, "reason": "Handled by guardrails."},
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        documents = final_output.get("documents", [])

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": documents,
            "source_files": _source_files(documents),
            "confidence": _confidence(final_output),
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return JSONResponse(status_code=500, content={
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        })