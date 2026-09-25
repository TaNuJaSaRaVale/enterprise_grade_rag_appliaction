import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Production gateway config — saved in the Portkey dashboard as "my_gateway_config"
# and attached as the DEFAULT config of PORTKEY_API_KEY. Do not send a config from
# code: the workspace blocks inline configs (400 inline_config_blocked) and the key
# forbids overriding its default (400 "Cannot override default config").
# Keep the dashboard config in sync with this shape:
#   - Fallback: primary @rag/<GROQ_MODEL> → @brag/<GROQ_FAST_MODEL> on failure
#   - Cache: simple mode
#   - Retry: 2 attempts on 429 / 503 before triggering the fallback target
#
#   {
#     "strategy": {"mode": "fallback"},
#     "cache": {"mode": "simple"},
#     "retry": {"attempts": 2, "on_status_codes": [429, 503]},
#     "targets": [
#       {"override_params": {"model": "@rag/openai/gpt-oss-120b"}},
#       {"override_params": {"model": "@brag/openai/gpt-oss-20b"}}
#     ]
#   }
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"