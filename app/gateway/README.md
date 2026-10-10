# `app/gateway/` — the LLM gateway (Portkey)

Every answer-path LLM call (planner and responder) goes through Portkey instead of calling Groq directly. Portkey is a proxy that adds fallback, retries and caching without any of that logic living in this code base.

## Files

| File | Role |
|---|---|
| `client.py` | The two clients and the cache-status helper |
| `__init__.py` | Re-exports `portkey_client`, `get_langchain_llm`, `extract_cache_status` |

## What `client.py` provides

| Name | Used by | What it is |
|---|---|---|
| `portkey_client` | `agents/nodes/responder.py` | Native Portkey client. Used for the final answer because it exposes response headers. |
| `get_langchain_llm(feature)` | `agents/nodes/planner.py` | A LangChain `ChatOpenAI` pointed at the Portkey gateway URL, model `@rag/openai/gpt-oss-120b`, temperature 0. The `feature` value is sent as Portkey metadata so calls can be filtered in the dashboard. |
| `extract_cache_status(response)` | responder | Reads the `x-portkey-cache-status` header, trying several attribute paths. Returns `"MISS"` if it finds nothing. |

## The gateway config lives in the Portkey dashboard

The config is saved in the dashboard as `my_gateway_config` and attached as the **default config of the API key**. This is its shape:

```json
{
  "strategy": {"mode": "fallback"},
  "cache": {"mode": "simple"},
  "retry": {"attempts": 2, "on_status_codes": [429, 503]},
  "targets": [
    {"override_params": {"model": "@rag/openai/gpt-oss-120b"}},
    {"override_params": {"model": "@brag/openai/gpt-oss-20b"}}
  ]
}
```

| Part | Effect |
|---|---|
| Fallback | Try `gpt-oss-120b` on the primary Groq key (`@rag`); on failure use `gpt-oss-20b` on the second key (`@brag`) |
| Retry | Two attempts on 429 (rate limit) or 503 before falling back |
| Cache | "simple" mode: an identical request returns the stored response |

**Do not send a config from code.** The workspace blocks inline configs (`400 inline_config_blocked`) and the key forbids overriding its default (`400 "Cannot override default config"`).

**Changing a model takes two edits:** `app/config.py` and the dashboard config. The code comment in `client.py` is only a copy of the dashboard config for reference.

## Why `ChatOpenAI` and not `ChatGroq`

`ChatGroq` is hard-wired to Groq's API and cannot be routed through a proxy. Portkey exposes an OpenAI-compatible endpoint, and `ChatOpenAI` accepts a `base_url` and extra headers. The models are still Groq models; Portkey sits in the middle. The `@rag/<model>` name format is Portkey's, which is why Groq's own client would not understand it.

## Who does not use the gateway

| Caller | Goes to | Why |
|---|---|---|
| Guardrails (`app/guardrails/rails.py`) | Groq directly (`ChatGroq`) | NeMo needs an LLM object, but intent matching is embeddings-only, so it makes no LLM call in practice. |
| Eval judge (`evals/judge.py`) | Groq directly | The key's locked default config would route every judge call to `gpt-oss-120b`, and the gateway cache could return a stale verdict. |

## What changed while building this, and why

| Change | Why |
|---|---|
| Direct Groq calls → Portkey gateway | Free-tier rate limits are hit often. Fallback to a second key and model, plus retries, keeps the demo answering. |
| Inline config in code → default config on the API key | The workspace rejects inline configs, and the key rejects overrides. Leaving the config out of the request is the only form that works. |
| Planner uses `ChatOpenAI` via the gateway | LangChain's structured output is needed for the planner, and `ChatGroq` cannot go through a proxy. |
| Responder uses the native Portkey client | Only the native client exposes the cache-status header, which the UI shows as "Cache: Hit ⚡". |
| Models updated to `gpt-oss-120b` / `gpt-oss-20b` | Groq retired the Llama models the project started with. |
