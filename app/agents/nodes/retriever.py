import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]


    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        # Drop exact-duplicate chunks (same text ingested from overlapping sources)
        seen, unique_results = set(), []
        for doc in raw_results:
            key = doc["content"].strip()
            if key and key not in seen:
                seen.add(key)
                unique_results.append(doc)

        with logfire.span("⚖️ Semantic Reranking"):
            reranked = rerank_documents(query, unique_results, top_n=5)
            logfire.info(f"Reranking complete. Kept {len(reranked)} relevant chunks.")

        # Numbered + labelled so the responder can cite [1], [2]... and the API
        # response shows where each chunk came from
        formatted_docs = [
            f"[{i}] Source: {doc.get('source', 'Unknown')}\n{doc['content']}"
            for i, doc in enumerate(reranked, start=1)
        ]

    status = "Found technical context." if formatted_docs else "No relevant documentation found."
    return {
        "documents": formatted_docs,
        # rerank_score is absent only if reranking failed and fell back to Qdrant order
        "retrieval_scores": [round(doc.get("rerank_score", 0.0), 4) for doc in reranked],
        "status": status,
        "plan": state["plan"] + ["Context Retrieved" if formatted_docs else "Context: None relevant"]
    }
