import json
import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


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
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

SCORECARD_PATH = os.path.join(os.path.dirname(__file__), "eval_scorecard.json")


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
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # --- EVAL SCORECARD (measured offline with Ragas; see evals/) ---
    st.markdown("---")
    st.subheader("📊 Measured Quality")
    try:
        with open(SCORECARD_PATH) as f:
            card = json.load(f)
        st.caption(f"Baseline eval · {card['goldens_scored']}/{card['goldens_total']} golden questions scored · "
                   f"judge: {card['judge_model']}")
        cols = st.columns(2)
        cols[0].metric("Source hit rate", f"{card['source_hit_rate']:.2f}")
        cols[1].metric("Avg latency", f"{card['avg_latency_s']:.1f}s")
        labels = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy",
                  "context_recall": "Context recall", "context_precision": "Context precision"}
        cols = st.columns(2)
        for i, (key, label) in enumerate(labels.items()):
            m = card["metrics"].get(key, {})
            if m.get("mean") is not None:
                cols[i % 2].metric(label, f"{m['mean']:.2f}", help=f"Averaged over {m['n']} questions (N/A excluded)")
    except FileNotFoundError:
        st.caption("Scorecard not generated yet (python -m evals.export_scorecard).")

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")


# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_confidence(message.get("confidence"), message.get("source_files"))

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
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
                        # Get backend URL from env, or default to local if not set
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(url, json=payload, timeout=60)
                        data = response.json()
                    
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