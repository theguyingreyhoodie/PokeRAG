"""Pluggable LLM backends. Each `stream()` yields text chunks.

- ollama:    local and free (default)
- openai:    any OpenAI-compatible endpoint (Groq's free tier by default)
- anthropic: Claude, via the official SDK
- none:      skip generation; the app returns the retrieved context and analysis only
"""

import json

import httpx

from . import config


class LLMError(RuntimeError):
    pass


def stream_ollama(system, messages):
    body = {"model": config.OLLAMA_MODEL, "stream": True,
            "messages": [{"role": "system", "content": system}] + messages,
            "options": {"temperature": 0.2, "num_ctx": 8192}}
    try:
        with httpx.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=body, timeout=120) as r:
            if r.status_code != 200:
                raise LLMError(f"Ollama returned {r.status_code}: {r.read().decode()[:200]}")
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if text := chunk.get("message", {}).get("content"):
                        yield text
    except httpx.ConnectError as e:
        raise LLMError(f"Can't reach Ollama at {config.OLLAMA_URL}. Is it running? (`brew services start ollama`)") from e


def stream_openai(system, messages):
    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set (for Groq, create a free key at console.groq.com).")
    body = {"model": config.OPENAI_MODEL, "stream": True, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}] + messages}
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    with httpx.stream("POST", f"{config.OPENAI_BASE_URL}/chat/completions", json=body,
                      headers=headers, timeout=120) as r:
        if r.status_code != 200:
            raise LLMError(f"LLM API returned {r.status_code}: {r.read().decode()[:200]}")
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            if text := delta.get("content"):
                yield text


def stream_anthropic(system, messages):
    import anthropic

    client = anthropic.Anthropic()
    try:
        # Server-side fallback re-runs the request on another model if a safety classifier declines it.
        with client.beta.messages.stream(
            model=config.ANTHROPIC_MODEL,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"effort": "low"},  # grounded Q&A over provided context
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n(The model declined to answer this one.)"
        elif final.stop_reason == "max_tokens":
            yield "\n\n(Answer was cut off at the token limit.)"
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic authentication failed. Set ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Anthropic rate limit hit; try again shortly.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Can't reach the Anthropic API.") from e


PROVIDERS = {"ollama": stream_ollama, "openai": stream_openai, "anthropic": stream_anthropic}


def stream(system, messages):
    fn = PROVIDERS.get(config.LLM_PROVIDER)
    if fn is None:
        raise LLMError(f"Unknown or disabled LLM_PROVIDER '{config.LLM_PROVIDER}'.")
    return fn(system, messages)


def describe():
    model = {"ollama": config.OLLAMA_MODEL, "openai": config.OPENAI_MODEL,
             "anthropic": config.ANTHROPIC_MODEL}.get(config.LLM_PROVIDER)
    return {"provider": config.LLM_PROVIDER, "model": model}
