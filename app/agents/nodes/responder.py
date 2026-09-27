import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status

MAX_CONTEXT_CHARS = 25000
MAX_CHARS_PER_DOC = 10000  # some ingested chunks are whole documents (60k+ chars)
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS_PER_MESSAGE = 800

TECHNICAL_SYSTEM_PROMPT = """You are a Senior Technical Architect answering questions for an enterprise IT team
(Kubernetes, Intel hardware, enterprise networking).

Rules:
1. Ground your answer in the numbered DOCUMENTATION excerpts. Cite them inline as [1], [2] right after
   the statements they support. Never cite a number that is not in the excerpts.
2. If the excerpts only partly cover the question, answer the covered part and say clearly what the
   documentation does not cover. You may add widely known general guidance, but label it
   "General guidance (not from the docs):".
3. If the excerpts contain nothing relevant, say so in one sentence first.
4. Never invent commands, flags, config keys, version numbers, or product specs.
5. Format: start with a 1-3 sentence direct answer, then steps / commands / details as needed.
   Use Markdown with fenced code blocks for commands and YAML. Be concise — no filler, no
   restating the question, no long preambles. Aim for under 350 words unless the question
   genuinely needs more."""

NO_CONTEXT_SYSTEM_PROMPT = """You are a Senior Technical Architect for an enterprise IT team
(Kubernetes, Intel hardware, enterprise networking).

The internal knowledge base returned NO relevant documentation for this question.
- Begin with exactly: "I couldn't find this in the internal documentation."
- Then, if it is a Kubernetes / Intel / networking question, give a brief answer under the heading
  "General guidance (not from the docs):" — keep it short and accurate, and do not invent
  company-specific details, versions, or internal procedures.
- Suggest how the user could rephrase or what document would need to be ingested."""

CONVERSATIONAL_SYSTEM_PROMPT = """You are a friendly, professional Enterprise IT Assistant specialising in
Kubernetes, Intel hardware, and enterprise networking. Reply to the user's latest message using the
conversation history. Keep it short and natural. Do not make up technical facts — if the user asks
something technical, invite them to ask it directly so you can search the documentation."""


def _format_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages[-MAX_HISTORY_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > MAX_HISTORY_CHARS_PER_MESSAGE:
            content = content[:MAX_HISTORY_CHARS_PER_MESSAGE] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(no previous messages)"


def trim_documents(documents: list[str]) -> list[str]:
    """
    The context-size rule for the LLM. Public so evals/metrics.py judges exactly what
    the LLM was shown — change limits here and the eval follows automatically.
    """
    # Trim oversized docs instead of dropping them — skipping a doc larger than the
    # budget used to leave the LLM with an empty context for whole-PDF chunks.
    context, used = [], 0
    for doc in documents:
        budget = min(MAX_CHARS_PER_DOC, MAX_CONTEXT_CHARS - used)
        if budget <= 0:
            logfire.warning("Context budget exhausted — remaining documents dropped.")
            break
        if len(doc) > budget:
            logfire.warning(f"Document trimmed from {len(doc)} to {budget} chars to fit LLM token limits.")
            doc = doc[:budget] + "\n[...truncated]"
        context.append(doc)
        used += len(doc)
    return context


def _build_context(documents: list[str]) -> str:
    return "\n\n---\n\n".join(trim_documents(documents))


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]
    history_str = _format_history(state["messages"][:-1])
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        system_prompt = CONVERSATIONAL_SYSTEM_PROMPT
        user_prompt = f"CONVERSATION HISTORY:\n{history_str}\n\nLATEST MESSAGE:\n{user_msg}"
    elif state.get("documents"):
        logfire.info("Generating technical RAG response.")
        system_prompt = TECHNICAL_SYSTEM_PROMPT
        user_prompt = (
            f"DOCUMENTATION:\n{_build_context(state['documents'])}\n\n"
            f"CONVERSATION HISTORY:\n{history_str}\n\n"
            f"QUESTION:\n{user_msg}"
        )
    else:
        logfire.info("No relevant context retrieved — generating transparent fallback response.")
        system_prompt = NO_CONTEXT_SYSTEM_PROMPT
        user_prompt = f"CONVERSATION HISTORY:\n{history_str}\n\nQUESTION:\n{user_msg}"

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
