"""Small wrapper around the Groq chat API."""
from groq import Groq

import config

_client = None


def chat(prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.0,
         model: str | None = None) -> str:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    resp = _client.chat.completions.create(
        model=model or config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()