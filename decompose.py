import json
import re
import config
from llm import chat

PROMPT = """Split the finance question into independent sub-questions, each asking for ONE thing.
If the question already asks only one thing, return it unchanged.
Return ONLY a JSON list of strings (max 3). No explanation.

Question: What is a mutual fund, and what is the latest capital gains tax on equity funds in India?
["What is a mutual fund?", "What is the latest capital gains tax on equity mutual funds in India?"]
Question: What is the current RBI repo rate?
["What is the current RBI repo rate?"]
Question: {question}
"""


def parse_list(text: str, fallback: str) -> list[str]:
    try:
        m = re.search(r"\[.*\]", text, re.S)
        items = json.loads(m.group(0)) if m else []
        items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
        return items[:3] or [fallback]
    except Exception:
        return [fallback]
    
# words that usually mean "more than one question"; without them no LLM call is needed
CONNECTORS = re.compile(r"\band\b|,|;|&|\bvs\b|\bversus\b|\bcompare\b", re.I)


def decompose(question: str) -> list[str]:
    if not CONNECTORS.search(question):          # single-intent question: skip the LLM call
        return [question]
    reply = chat(PROMPT.format(question=question), model=getattr(config, "HELPER_MODEL", None))
    return parse_list(reply, question)