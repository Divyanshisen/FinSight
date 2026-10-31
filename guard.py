"""Safety guard: advice and prediction questions are refused before any retrieval (no tokens, no web call)."""
import re

PATTERNS = [
    r"\bshould i (buy|sell|invest|hold|put)\b",
    r"\bis it (a )?(good|right|safe) (time|idea) to (buy|sell|invest)\b",
    r"\b(will|going to) (the )?(nifty|sensex|market|markets|stock|share|price|gold|rupee)\b.{0,40}\b(go up|go down|rise|fall|crash|increase|decrease|cross|double|hit|reach)\b",
    r"\bguarantee[sd]?\b.{0,20}\b\d+\s*(%|percent|per cent)",
    r"\b(which|what)\b.{0,25}\b(stock|share|mutual fund|fund|scheme|crypto)s?\b.{0,40}\b(will (give|deliver|earn|double|beat)|to buy|should i)\b",
    r"\b\d+\s*(%|percent|per cent)\s+returns?\b.{0,30}\bnext\b",
]
_RX = [re.compile(p, re.I) for p in PATTERNS]

REPLY = ("I can't predict market movements or tell you what to buy or sell. I can explain how an instrument works "
         "or share verified facts such as rates and rules. For personal advice, please speak to a SEBI-registered investment adviser.")


def guard_reply(question: str):
    """Returns the refusal text for advice or prediction questions, else None."""
    return REPLY if any(rx.search(question) for rx in _RX) else None


if __name__ == "__main__":
    refuse = ["Which mutual fund will give 20 percent returns next year?", "Should I buy HDFC Bank shares now?",
              "Will the Nifty go up next month?", "Is it a good time to invest in gold?",
              "Which stock should I buy for next year?", "Any fund with guaranteed 15% returns?"]
    allow = ["What is a mutual fund?", "What is the current RBI repo rate?", "What are the current CRR and SLR?",
             "What is the long term capital gains tax rate on listed equity shares in India?",
             "How has India's GDP growth rate changed over the last 5 years?", "SIP of 5000 per month at 12% for 10 years",
             "What is the Reliance share price today?", "Which mutual fund category is best for long term?",
             "What is the difference between ELSS and PPF?", "What returns did equity funds give last year?",
             "Is there any guaranteed return scheme backed by the government?"]
    bad = 0
    for q in refuse:
        ok = guard_reply(q) is not None
        bad += not ok
        print("OK  " if ok else "FAIL", "refuse |", q)
    for q in allow:
        ok = guard_reply(q) is None
        bad += not ok
        print("OK  " if ok else "FAIL", "allow  |", q)
    print("failures:", bad)