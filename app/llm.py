# One place that talks to the language model.
# Switches between Nebius, NVIDIA, and Grok using .env.
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load keys and settings from .env file with override=True
load_dotenv(override=True)

_client = None
_model = None
_provider = None

def _get_client():
    global _client, _model, _provider
    # Reload .env with override=True so changes to MODEL or PROVIDER take effect immediately
    load_dotenv(override=True)

    provider = os.getenv("PROVIDER", "nebius").lower().strip()
    model = os.getenv("MODEL", "").strip()

    # Reuse cached client if provider and model have not changed
    if _client is not None and _provider == provider and _model == model:
        return _client, _model

    _provider = provider

    if provider == "nebius":
        base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
        api_key = os.getenv("NEBIUS_API_KEY", "").strip()
        default_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        key_name = "NEBIUS_API_KEY"
    elif provider == "grok":
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        api_key = os.getenv("XAI_API_KEY", "").strip()
        default_model = "grok-beta"
        key_name = "XAI_API_KEY"
    else:
        # Default is NVIDIA
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        default_model = "nvidia/llama-3.1-nemotron-70b-instruct"
        key_name = "NVIDIA_API_KEY"

    _model = model or default_model

    if not api_key:
        raise ValueError(f"Missing API key: please set {key_name} in your .env file.")

    _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client, _model

def ask(prompt: str, temperature: float = 0.0) -> str:
    client, model = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=800,
    )

    msg = response.choices[0].message
    content = msg.content

    # If content is empty, check reasoning fields (used by some reasoning models)
    if not content:
        content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)

    if content:
        content = content.strip()

    if not content:
        finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        raise RuntimeError(
            f"Model '{model}' returned an empty reply (finish_reason='{finish_reason}'). "
            "Never returning empty string."
        )

    return content

if __name__ == "__main__":
    client, model = _get_client()
    provider = os.getenv("PROVIDER", "nebius")
    print(f"Provider: {provider}")
    print(f"Model:    {model}")
    print("Testing ask()...")
    res = ask("Answer in 3 words: What is the capital of France?")
    print(f"Result:   {res}")
