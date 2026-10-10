import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

CANDIDATES_PER_QUERY = 15
TOP_N = 5


def _search_and_rerank(query: str) -> list[dict]:
    """Vector search + cross-encoder rerank for ONE query."""
    logfire.info(f"Searching Qdrant for: {query}")
    raw_results = search_enterprise_knowledge(query, limit=CANDIDATES_PER_QUERY)
    logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

    # Drop exact-duplicate chunks (same text ingested from overlapping sources)
    seen, unique_results = set(), []
    for doc in raw_results:
        key = doc["content"].strip()
        if key and key not in seen:
            seen.add(key)
            unique_results.append(doc)

    with logfire.span("⚖️ Semantic Reranking"):
        reranked = rerank_documents(query, unique_results, top_n=TOP_N)
        logfire.info(f"Reranking complete. Kept {len(reranked)} relevant chunks.")
    return reranked


def _merge(per_query: list[list[dict]]) -> list[dict]:
    """
    Combine the reranked chunks of several queries into one top-TOP_N list.
    A chunk found by more than one query keeps its highest score. Each query's best
    chunk is taken first, so one part of a multi-part question cannot fill every slot.
    """
    score = lambda doc: doc.get("rerank_score", 0.0)

    best: dict[str, dict] = {}
    for docs in per_query:
        for doc in docs:
            key = doc["content"].strip()
            if key not in best or score(doc) > score(best[key]):
                best[key] = doc

    picked: list[str] = []
    for docs in per_query:
        if docs and docs[0]["content"].strip() not in picked:
            picked.append(docs[0]["content"].strip())
    for key in sorted(best, key=lambda k: score(best[k]), reverse=True):
        if key not in picked:
            picked.append(key)

    return sorted((best[key] for key in picked[:TOP_N]), key=score, reverse=True)


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    Multi-part questions are searched once per sub-query: a chunk that answers only one
    part scores below the rerank cutoff against the combined question.
    """
    query = state["current_query"]

    # .get: threads checkpointed before sub_queries existed don't have the key
    queries = [query]
    for sub_query in state.get("sub_queries") or []:
        if sub_query.lower() not in (q.lower() for q in queries):
            queries.append(sub_query)

    with logfire.span("🔍 Knowledge Retrieval"):
        reranked = _merge([_search_and_rerank(q) for q in queries])
        if len(queries) > 1:
            logfire.info(f"Merged {len(queries)} queries into {len(reranked)} chunks.")

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
