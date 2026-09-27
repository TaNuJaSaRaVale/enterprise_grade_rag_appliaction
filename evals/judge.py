"""
The judge LLM for evaluation — one Qwen-on-Groq connection, two thin adapters:

    QwenDeepEvalJudge        → the interface DeepEval expects (DeepEvalBaseLLM)
    get_ragas_llm()          → the interface Ragas 0.4 metrics expect (Instructor-based)
    get_ragas_embeddings()   → local embeddings for Ragas Answer Relevancy (no API quota)

Why Qwen and not the app's gpt-oss: a judge from a different model family reduces
self-grading bias. Why direct Groq and not Portkey: the Portkey API key has a locked
default config that routes every call to gpt-oss, and its cache could return stale verdicts.
"""
import asyncio
import os
import threading
import time
from collections import deque
from functools import lru_cache

# Disable library telemetry BEFORE importing ragas/deepeval (deepeval reads this at import).
# Ragas' analytics calls were timing out on this network, adding ~40s to EVERY score
# (measured: ascore 40.7s → 1.0s). setdefault keeps any value already set in the shell.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

JUDGE_MODEL = "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # already cached as the app's fallback

# OpenAI SDK retries 429 / 5xx with exponential backoff — the safety net behind the limiter
MAX_RETRIES = 6
TIMEOUT_S = 60

# Groq free tier for this model/key: 8,000 tokens per minute (read from x-ratelimit-* headers).
# Stay below it with a margin because token counts are estimated.
TOKENS_PER_MINUTE = 7000
EXPECTED_OUTPUT_TOKENS = 600  # judge verdicts observed at 45–170 tokens; generous margin


class TokenBudget:
    """
    Sliding-window rate limiter: before a call, wait until the tokens spent in the
    last 60s plus this call's estimate fit under the per-minute budget.
    Thread-safe (DeepEval may call from threads) with separate sync/async waits
    (time.sleep inside async code would freeze every other coroutine).
    Per-process only — run one eval at a time per key.
    """

    def __init__(self, tokens_per_minute: int, window_s: float = 60.0):
        self.limit = tokens_per_minute
        self.window_s = window_s
        self._calls = deque()  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _try_reserve(self, tokens: int):
        """Reserve if it fits → (0, entry); otherwise → (seconds to wait, None)."""
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0][0] >= self.window_s:
                self._calls.popleft()
            used = sum(entry[1] for entry in self._calls)
            # An oversized single call can never "fit" — let it through once the window is empty
            if used + tokens <= self.limit or not self._calls:
                entry = [now, tokens]  # list, so settle() can correct it in place
                self._calls.append(entry)
                return 0.0, entry
            return self.window_s - (now - self._calls[0][0]) + 0.05, None

    def settle(self, entry, actual_tokens):
        """Replace the pessimistic estimate with real usage so the window stays accurate."""
        if entry is not None and actual_tokens:
            with self._lock:
                entry[1] = actual_tokens

    def acquire_sync(self, tokens: int):
        while True:
            wait, entry = self._try_reserve(tokens)
            if entry is not None:
                return entry
            time.sleep(wait)

    async def acquire_async(self, tokens: int):
        while True:
            wait, entry = self._try_reserve(tokens)
            if entry is not None:
                return entry
            await asyncio.sleep(wait)


_budget = TokenBudget(TOKENS_PER_MINUTE)


def _estimate_tokens(request: dict) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in request.get("messages", []))
    return chars // 4 + EXPECTED_OUTPUT_TOKENS


def _api_key() -> str:
    # Dedicated judge key so eval traffic never eats the app's Groq rate limit
    key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_FALLBACK_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set JUDGE_GROQ in .env for the eval judge.")
    return key


@lru_cache(maxsize=1)
def _sync_client():
    from openai import OpenAI
    client = OpenAI(api_key=_api_key(), base_url=GROQ_BASE_URL, max_retries=MAX_RETRIES, timeout=TIMEOUT_S)
    create = client.chat.completions.create

    def paced_create(*args, **kwargs):  # every DeepEval judge call passes through the budget
        entry = _budget.acquire_sync(_estimate_tokens(kwargs))
        response = create(*args, **kwargs)
        _budget.settle(entry, getattr(getattr(response, "usage", None), "total_tokens", None))
        return response

    client.chat.completions.create = paced_create
    return client


@lru_cache(maxsize=1)
def _async_client():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=_api_key(), base_url=GROQ_BASE_URL, max_retries=MAX_RETRIES, timeout=TIMEOUT_S)
    create = client.chat.completions.create

    async def paced_create(*args, **kwargs):  # every Ragas (and async DeepEval) call passes through
        entry = await _budget.acquire_async(_estimate_tokens(kwargs))
        response = await create(*args, **kwargs)
        _budget.settle(entry, getattr(getattr(response, "usage", None), "total_tokens", None))
        return response

    client.chat.completions.create = paced_create
    return client


class QwenDeepEvalJudge(DeepEvalBaseLLM):
    """
    DeepEval adapter. DeepEval calls generate(prompt, schema=PydanticModel) when it
    wants structured output (see deepeval/models/base_model.py generate_with_schema).
    Returning a schema *instance* lets DeepEval skip its own fragile JSON extraction.
    """

    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        super().__init__(model)

    def load_model(self, *args, **kwargs):
        return _sync_client()

    def get_model_name(self, *args, **kwargs) -> str:
        return f"{self.model_name} (Groq)"

    def _request(self, prompt: str, schema) -> dict:
        request = {"model": self.model_name, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}
        if schema is not None:
            request["response_format"] = {"type": "json_object"}  # JSON mode
        return request

    def generate(self, prompt: str, schema=None, **kwargs):
        response = _sync_client().chat.completions.create(**self._request(prompt, schema))
        content = response.choices[0].message.content or ""
        return schema.model_validate_json(content) if schema is not None else content

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        response = await _async_client().chat.completions.create(**self._request(prompt, schema))
        content = response.choices[0].message.content or ""
        return schema.model_validate_json(content) if schema is not None else content


def get_ragas_llm():
    """Ragas 0.4 metrics take an Instructor-based LLM; Groq is OpenAI-compatible."""
    from ragas.llms import llm_factory
    return llm_factory(JUDGE_MODEL, provider="openai", client=_async_client(), temperature=0)


@lru_cache(maxsize=1)
def get_ragas_embeddings():
    """Local embeddings for Answer Relevancy — free, no network, no quota."""
    from ragas.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
