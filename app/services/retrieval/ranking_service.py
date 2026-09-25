import time
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None

# MiniLM-L-12 ranks noticeably better than the default TinyBERT on this corpus
# (e.g. 0.99 vs 0.56 on the Intel paging doc) at ~1-2s CPU cost for 15 passages.
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# Cross-encoder scores are sigmoid (0-1). Chunks below this are unrelated to the
# question — sending them to the LLM only invites hallucinated answers.
MIN_RERANK_SCORE = 0.1


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank runs a local quantized ONNX cross-encoder (RERANK_MODEL).
    """
    global _ranker
    if _ranker is None:
        logfire.info(f"🧠 Initializing FlashRank Model ({RERANK_MODEL}) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(model_name=RERANK_MODEL, cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker(model_name=RERANK_MODEL)
    return _ranker


def rerank_documents(
    query: str,
    documents: list[dict],
    top_n: int = 5,
    min_score: float = MIN_RERANK_SCORE,
) -> list[dict]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.

    Takes Qdrant results ({'content', 'source', 'score'}) and returns the top_n
    that clear min_score, each with an added 'rerank_score'. May return [] when
    nothing in the knowledge base is relevant — the responder handles that.

    Why FlashRank?
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()

        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc["content"]}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        # Results are returned sorted by highest semantic score first
        reranked_docs = []
        for res in results:
            score = float(res["score"])
            if score < min_score or len(reranked_docs) >= top_n:
                break
            reranked_docs.append({**documents[res["id"]], "rerank_score": score})

        duration = time.time() - start_time
        top_score = float(results[0]["score"]) if results else "N/A"
        logfire.info(
            f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score} | "
            f"kept {len(reranked_docs)}/{len(documents)} (min_score={min_score})"
        )

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]