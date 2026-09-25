from typing import Literal

import logfire
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.gateway import get_langchain_llm

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

# Only the recent turns matter for intent + query rewriting; older ones add noise and tokens
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS_PER_MESSAGE = 500


class PlannerDecision(BaseModel):
    """Structured planner output. JSON schema output replaces free text, which
    gpt-oss occasionally duplicated (e.g. 'query...query...')."""
    intent: Literal["conversational", "technical"] = Field(
        description="'conversational' for greetings/small talk or questions answerable from the "
                    "conversation history alone; 'technical' when documentation must be searched."
    )
    search_query: str = Field(
        default="",
        description="For 'technical': a standalone, keyword-rich search query. Empty otherwise."
    )


planner_llm = llm.with_structured_output(PlannerDecision, method="json_schema")


def _format_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages[-MAX_HISTORY_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > MAX_HISTORY_CHARS_PER_MESSAGE:
            content = content[:MAX_HISTORY_CHARS_PER_MESSAGE] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "(no previous messages)"


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the recent conversation,
    and rewrites technical questions into standalone search queries.
    """
    history = _format_history(state["messages"][:-1])
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""You are the planner for an Enterprise IT assistant whose knowledge base covers
Kubernetes, Intel hardware, and enterprise networking.

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE:
"{user_message}"

Decide the intent:
- "conversational": greetings, thanks, small talk, or questions answerable purely from the
  conversation history above (e.g. "what did I just ask?").
- "technical": anything that needs the technical documentation.

For "technical", write search_query as a single standalone search query:
- Resolve pronouns and follow-ups using the history ("how do I do that on Intel NICs?" ->
  "configure SR-IOV virtual functions on Intel NICs").
- Keep the key technical terms, product names, and commands from the user's message.
- One line, no explanations, no quotes, 5-15 words.
"""

    with logfire.span("🧠 Planner Decision"):
        try:
            decision: PlannerDecision = planner_llm.invoke(prompt)
        except Exception as e:
            # Never fail the request on a planner hiccup — search with the raw message instead
            logfire.warning(f"Planner failed, falling back to raw user message: {e}")
            decision = PlannerDecision(intent="technical", search_query=user_message)

        search_query = decision.search_query.strip() or user_message
        logfire.info(f"Intent identified: {decision.intent} | query: {search_query}")

    if decision.intent == "conversational":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": search_query,
        "status": f"Technical research needed. Searching for: {search_query}",
        "plan": ["Intent: Technical", f"Search Term: {search_query}"]
    }
