import json
import os
import sys
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables explicitly from the root directory
env_path = os.path.join(REPO_ROOT, ".env")
load_dotenv(dotenv_path=env_path)

# Two ways to reach the RAG backend:
#   BACKEND_URL set   → HTTP to the FastAPI service (Docker image, local two-process setup)
#   BACKEND_URL unset → call the same query() in-process (single-process hosts, e.g. Streamlit Cloud)
BACKEND_URL = os.getenv("BACKEND_URL")


@st.cache_resource(show_spinner="Starting the RAG engine (first load takes ~1 min)...")
def _inprocess_backend():
    """Import the FastAPI module once per server and run its startup (NeMo init) once."""
    # Repo root must come FIRST: this file is ui/app.py, so a bare `import app` from ui/
    # would import this UI file instead of the backend package app/
    if sys.path[0] != REPO_ROOT:
        sys.path.insert(0, REPO_ROOT)
    import app.main as backend
    backend.initialize_rails()
    return backend


def call_backend(prompt: str, session_id: str) -> dict:
    if BACKEND_URL:
        response = requests.post(f"{BACKEND_URL}/query", json={"q": prompt, "thread_id": session_id}, timeout=120)
        return response.json()
    backend = _inprocess_backend()
    result = backend.query(backend.QueryRequest(q=prompt, thread_id=session_id))
    # query() returns a dict, or a JSONResponse for errors / rate limits
    return json.loads(result.body) if hasattr(result, "body") else result


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    


# --- PAGE CONFIG ---
APP_NAME = "KubeRAG"
st.set_page_config(
    page_title=f"{APP_NAME} · Enterprise Docs Assistant",
    page_icon="☸️",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "☸️"
USER_AVATAR = "👤"

# --- KUBERNETES THEME ---
# CSS instead of .streamlit/config.toml: the Docker image copies only app/ and ui/, so a
# root config file would not reach it. Colours use transparency so they read in light AND dark mode.
K8S_BLUE = "#326CE5"
st.markdown(f"""
<style>
.k8s-hero {{
    background: linear-gradient(135deg, {K8S_BLUE} 0%, #1A4FB8 100%);
    color: #FFFFFF; border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 1rem;
}}
.k8s-hero .helm {{ font-size: 2.6rem; line-height: 1; }}
.k8s-hero h1 {{ color: #FFFFFF; font-size: 1.7rem; margin: 0; padding: 0; }}
.k8s-hero p {{ color: rgba(255,255,255,0.88); margin: 0.2rem 0 0 0; font-size: 0.95rem; }}
.k8s-card {{
    border: 1px solid rgba(50,108,229,0.35); background: rgba(50,108,229,0.07);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 1rem;
}}
.k8s-chip {{
    display: inline-block; border: 1px solid rgba(50,108,229,0.45); background: rgba(50,108,229,0.10);
    border-radius: 999px; padding: 0.1rem 0.6rem; margin: 0.15rem 0.2rem 0.15rem 0; font-size: 0.8rem;
}}
[data-testid="stSidebar"] {{ border-right: 3px solid {K8S_BLUE}; }}
[data-testid="stSidebar"] .stButton button {{ text-align: left; justify-content: flex-start; }}
[data-testid="stSidebar"] .stButton button[kind="primary"] {{ background: {K8S_BLUE}; border-color: {K8S_BLUE}; }}
[data-testid="stChatInput"] {{ border: 1px solid {K8S_BLUE}; border-radius: 12px; }}
@media (max-width: 640px) {{
    .k8s-hero {{ padding: 0.8rem 1rem; }}
    .k8s-hero h1 {{ font-size: 1.3rem; }}
    .k8s-hero .helm {{ font-size: 2rem; }}
}}
</style>
""", unsafe_allow_html=True)

# Questions the current index answers well, plus one that shows the honest "not in docs" path
EXAMPLE_QUESTIONS = [
    ("📈 HPA replica range", "In the nginx HPA example, what replica range is used?"),
    ("🏛️ etcd & API server", "What roles do etcd and kube-apiserver play in the Kubernetes control plane?"),
    ("📬 Work queue Jobs", "How does a Job process a parallel work queue?"),
    ("🔍 Not in the docs", "How do I set up Istio mTLS?"),
]
KNOWLEDGE_BASE_TOPICS = ["Autoscaling (HPA/VPA)", "Jobs & CronJobs", "Cluster architecture",
                         "Monitoring Jobs", "Work queues", "Intel 5-level paging"]

def render_confidence(confidence, source_files):
    """Confidence badge (from reranker scores, not the LLM) + the files the answer came from."""
    level = (confidence or {}).get("level")
    if level and level != "n/a":
        score = confidence.get("top_score")
        text = f"**Confidence: {level}**" + (f" (relevance {score:.2f})" if score is not None else "") + f" — {confidence.get('reason', '')}"
        {"High": st.success, "Medium": st.info, "Low": st.warning}.get(level, st.error)(text)
    if source_files:
        st.caption("📚 Sources: " + ", ".join(source_files))


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR ---
with st.sidebar:
    st.title(f"☸️ {APP_NAME}")
    st.caption("Agentic RAG for Kubernetes & infrastructure docs")

    st.markdown("**What's in the knowledge base**")
    st.markdown("".join(f'<span class="k8s-chip">{t}</span>' for t in KNOWLEDGE_BASE_TOPICS),
                unsafe_allow_html=True)

    st.markdown("**Try asking**")
    for label, question in EXAMPLE_QUESTIONS:
        if st.button(label, help=question, width="stretch"):
            st.session_state.pending_prompt = question

    st.markdown("---")
    if st.button("🗑️ New conversation", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"🔭 Tracing: {LOGFIRE_STATUS}")

# --- MAIN CHAT ---
st.markdown(f"""
<div class="k8s-hero">
  <div class="helm">☸️</div>
  <div>
    <h1>{APP_NAME}</h1>
    <p>Ask questions about your Kubernetes and infrastructure documentation. Every answer cites its
    sources and shows how well the docs support it.</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
<div class="k8s-card">
<b>How it works:</b> your question passes guardrails, is rewritten into a search query, matched
against the indexed docs and reranked; the answer is written only from the passages found.
The confidence badge under each answer comes from the retrieval scores, not from the model's own opinion.
Pick a question on the left or type your own below.
</div>
""", unsafe_allow_html=True)


# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_confidence(message.get("confidence"), message.get("source_files"))

# Chat Input
# A sidebar example button sets pending_prompt; it is handled exactly like typed input
typed = st.chat_input("Ask about your Kubernetes documentation...")
if prompt := (typed or st.session_state.pop("pending_prompt", None)):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        data = call_backend(prompt, st.session_state.session_id)
                    
                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")
                    
                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                    
                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                # Create a preview title for each chunk
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i+1}: {preview}"):
                                    st.info(source)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")
            
            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            
            answer_placeholder.markdown(full_answer)
            render_confidence(data.get("confidence"), data.get("source_files"))
            st.session_state.messages.append({
                "role": "assistant", "content": full_answer,
                "confidence": data.get("confidence"), "source_files": data.get("source_files"),
            })
            logfire.info("✅ Chat cycle completed successfully.")