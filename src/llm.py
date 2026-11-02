"""
LLM access through LlamaIndex: one `OpenAILike` LLM object per model in config.MODELS.

Groq and OpenAI both speak the OpenAI API, so LlamaIndex's `OpenAILike` class talks to
both; only api_base, model name and API key differ (looked up in the registry).
`OpenAILike` (not LlamaIndex's `OpenAI` class) because it accepts any model name,
including ones newer than the installed LlamaIndex version knows about.
"""

import sys
import time

import openai
from llama_index.llms.openai_like import OpenAILike

try:
    from src.config import DEFAULT_MODEL, LLM_TEMPERATURE, MODELS, get_api_key
except ImportError:
    from config import DEFAULT_MODEL, LLM_TEMPERATURE, MODELS, get_api_key

# Friendly errors are returned as text (so the UI never crashes) and start with this
# prefix, so rag.py / app.py can tell an error apart from a real answer.
ERROR_PREFIX = "⚠️ "

# Generous limit: reasoning models sometimes think for a while before answering.
LLM_TIMEOUT_SECONDS = 120

OPENAI_API_BASE = "https://api.openai.com/v1"  # used when a model's base_url is None
CONTEXT_WINDOW = 128_000  # tokens; OpenAILike needs it to know how much context fits

# One LlamaIndex LLM per model, created on first use and then reused.
_llms: dict[str, OpenAILike] = {}


def get_llm(model_name: str = DEFAULT_MODEL) -> OpenAILike:
    """Build (or reuse) the LlamaIndex LLM for a model in the registry."""
    if model_name not in _llms:
        cfg = MODELS[model_name]
        # Some models (e.g. OpenAI reasoning models) only accept their default temperature, 1.
        temperature = LLM_TEMPERATURE if cfg.get("supports_temperature", True) else 1.0
        _llms[model_name] = OpenAILike(
            model=cfg["model"],
            api_base=cfg["base_url"] or OPENAI_API_BASE,
            api_key=get_api_key(model_name),
            temperature=temperature,
            is_chat_model=True,  # use the chat endpoint, like every modern model
            context_window=CONTEXT_WINDOW,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=1,  # fail fast so the user isn't left waiting on a dead server
        )
    return _llms[model_name]


def check_model(model_name: str) -> str | None:
    """Return a friendly error if the model can't be used (unknown, or key missing), else None."""
    if model_name not in MODELS:
        return f"{ERROR_PREFIX}Unknown model '{model_name}'. Check DEFAULT_MODEL in .env."
    if get_api_key(model_name) is None:
        key_env = MODELS[model_name]["api_key_env"]
        return f"{ERROR_PREFIX}{model_name}: no API key — add {key_env} to your .env file."
    return None


def available_models() -> list[str]:
    """Models that can be used right now, i.e. whose API key is in .env."""
    return [name for name in MODELS if get_api_key(name)]


def friendly_error(model_name: str, error: Exception) -> str:
    """Turn an API exception into a short message a user can act on."""
    key_env = MODELS[model_name]["api_key_env"]

    # Order matters: APITimeoutError is a subclass of APIConnectionError.
    if isinstance(error, openai.APITimeoutError):
        msg = f"The model took longer than {LLM_TIMEOUT_SECONDS}s to answer. Try again or pick another model."
    elif isinstance(error, openai.APIConnectionError):
        msg = "Couldn't reach the model's server. Check your internet connection."
    elif isinstance(error, openai.AuthenticationError):
        msg = f"The API key was rejected. Check {key_env} in your .env file."
    elif isinstance(error, openai.NotFoundError):
        model = MODELS[model_name]["model"]
        msg = f"Model '{model}' wasn't found, or your account can't use it."
    elif isinstance(error, openai.RateLimitError):
        msg = "Rate limit or quota reached for this provider. Wait a minute or pick another model."
    else:
        msg = f"The model returned an error: {error}"
    return f"{ERROR_PREFIX}{model_name}: {msg}"


def call_llm_timed(prompt: str, model_name: str = DEFAULT_MODEL) -> tuple[str, float]:
    """Send a prompt and return (answer text, response time in seconds).

    Never raises: problems come back as a friendly message starting with ERROR_PREFIX.
    """
    problem = check_model(model_name)
    if problem:
        return problem, 0.0

    start = time.perf_counter()
    try:
        response = get_llm(model_name).complete(prompt)
    except Exception as error:  # any failure -> message for the UI, not a crash
        elapsed = round(time.perf_counter() - start, 2)
        print(f"LLM error after {elapsed}s: {type(error).__name__}: {error}")
        return friendly_error(model_name, error), elapsed
    elapsed = round(time.perf_counter() - start, 2)

    # Reasoning models (GPT-6 Luna, gpt-oss) keep their "thinking" out of the message
    # content, so `response.text` holds only the final answer — that's all we return.
    answer = (response.text or "").strip()
    if not answer:
        # Happens if the model spent its whole output budget on reasoning.
        return f"{ERROR_PREFIX}{model_name}: the model returned an empty answer. Try again.", elapsed

    print(f"LLM {model_name}: answered in {elapsed}s")
    return answer, elapsed


def call_llm(prompt: str, model_name: str = DEFAULT_MODEL) -> str:
    """Send a prompt and return just the answer text (see call_llm_timed for the time)."""
    answer, _ = call_llm_timed(prompt, model_name)
    return answer


if __name__ == "__main__":
    # Quick test: `python src/llm.py` tries every usable model,
    # or `python src/llm.py "GPT-6 Luna (OpenAI)"` tries just one.
    print("=== Available models ===")
    usable = available_models()
    for name in MODELS:
        print(f"[{'x' if name in usable else ' '}] {name}")

    test_prompt = "In one short sentence: what does a Certificate of Quality for a lab product confirm?"
    to_test = [sys.argv[1]] if len(sys.argv) > 1 else usable
    print(f"\n=== Test prompt ===\n{test_prompt}")
    for name in to_test:
        answer, seconds = call_llm_timed(test_prompt, name)
        print(f"\n--- {name} ({seconds}s) ---\n{answer}")

    print("\n=== Error handling (should print friendly messages, not crash) ===")
    print(call_llm("hi", "not-a-real-model"))
    for name in MODELS:
        if name not in usable:
            print(call_llm("hi", name))
