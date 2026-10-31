"""S4: calculator tool. Runs only local code (no LLM calls, no tokens)."""
import json
import re

UNITS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
         "cr": 1e7, "k": 1e3, "thousand": 1e3, "million": 1e6, "billion": 1e9}
U = r"(lakhs?|lacs?|crores?|cr|k|thousand|million|billion)"
RATE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", re.I)
TENURE = re.compile(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)\b", re.I)
AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*" + U + r"?\b", re.I)
FROM_TO = re.compile(r"(?:from|between)\s+(\d[\d,]*(?:\.\d+)?)\s*" + U + r"?\s*(?:to|and)\s+"
                     r"(\d[\d,]*(?:\.\d+)?)\s*" + U + r"?", re.I)
GROWTH_WORDS = re.compile(r"growth|grew|change|increase|decrease|rise|fell|fall|gain|loss", re.I)
NOTE = "Illustrative calculation from the numbers in your question. Not financial advice."


def _clean(q: str) -> str:
    q = q.replace("₹", " ")
    return re.sub(r"\brs\.?\s*", " ", q, flags=re.I)


def _value(num: str, unit: str | None) -> float:
    return float(num.replace(",", "")) * UNITS.get((unit or "").lower(), 1)


def _amounts(text: str) -> list[float]:
    return [_value(n, u) for n, u in AMOUNT.findall(text)]


def _years(q: str):
    m = TENURE.search(q)
    if not m:
        return None
    n = float(m.group(1))
    return n / 12 if m.group(2).lower().startswith("mo") else n


def _strip(q: str) -> str:
    return TENURE.sub(" ", RATE.sub(" ", q))


def inr(x: float) -> str:
    s = str(abs(round(x)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head) + "," + tail
    return ("-" if x < 0 else "") + "₹" + s


def _chart(kind, title, x, values, name):
    spec = {"type": kind, "title": title, "x": x,
            "series": [{"name": name, "values": values}], "source": "Calculated by code"}
    return "\n\n```chart\n" + json.dumps(spec, ensure_ascii=False) + "\n```"


def _emi(q):
    rate, years, amts = RATE.search(q), _years(q), _amounts(_strip(q))
    if not (rate and years and amts):
        return None
    P, n, r = amts[0], round(years * 12), float(rate.group(1)) / 1200
    if n <= 0 or P <= 0:
        return None
    emi = P / n if r == 0 else P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total = emi * n
    text = (f"**EMI: {inr(emi)} per month** for {n} months.\n"
            f"Principal {inr(P)}, total interest {inr(total - P)}, total payment {inr(total)}.\n"
            f"Formula: EMI = P × r × (1+r)^n / ((1+r)^n − 1), with r = annual rate / 12 / 100. "
            f"Actual bank EMI can differ slightly (fees, rounding).\n{NOTE}")
    return text + _chart("bar", "EMI breakup (₹)", ["Principal", "Total interest", "Total payment"],
                         [round(P), round(total - P), round(total)], "Amount (₹)")


def _sip(q):
    rate, years, amts = RATE.search(q), _years(q), _amounts(_strip(q))
    if not (rate and years and amts) or years > 60:
        return None
    P, n, r = amts[0], round(years * 12), float(rate.group(1)) / 1200
    if n <= 0 or P <= 0:
        return None
    bal, yearly = 0.0, []
    for m in range(1, n + 1):
        bal = (bal + P) * (1 + r)             # deposit at the start of each month
        if m % 12 == 0:
            yearly.append(round(bal))
    if n % 12:
        yearly.append(round(bal))
    invested = P * n
    text = (f"**Estimated SIP value: {inr(bal)}** after {years:g} years.\n"
            f"Invested {inr(invested)}, estimated gain {inr(bal - invested)}.\n"
            f"Assumes {rate.group(1)}% a year, compounded monthly, with the instalment at the start of each month. "
            f"Mutual fund returns are not guaranteed; the rate is your assumption.\n{NOTE}")
    if len(yearly) >= 3:
        text += _chart("line", "SIP value by year (₹)", [f"Year {i}" for i in range(1, len(yearly) + 1)],
                       yearly, "Estimated value (₹)")
    return text


def _cagr(q):
    years, amts = _years(q), _amounts(_strip(q))
    if not years or len(amts) < 2 or amts[0] <= 0 or amts[1] <= 0:
        return None
    start, end = amts[0], amts[1]
    cagr = ((end / start) ** (1 / years) - 1) * 100
    return (f"**CAGR: {cagr:.2f}% a year** (from {start:,.2f} to {end:,.2f} over {years:g} years).\n"
            f"Formula: CAGR = (end / start)^(1 / years) − 1.\n{NOTE}")


def _growth(q):
    m = FROM_TO.search(q)
    if not m or not GROWTH_WORDS.search(q):
        return None
    n1, u1, n2, u2 = m.groups()
    looks_like_year = lambda s, u: u is None and "," not in s and s.isdigit() and 1900 <= int(s) <= 2100
    if looks_like_year(n1, u1) and looks_like_year(n2, u2):
        return None                            # "from 2013 to 2014" are years, not values
    start, end = _value(n1, u1), _value(n2, u2)
    if start == 0:
        return None
    pct = (end - start) / abs(start) * 100
    return (f"**Change: {pct:+.2f}%** (from {start:,.2f} to {end:,.2f}).\n"
            f"Formula: (new − old) / old × 100.\n{NOTE}")


def try_calculate(question: str):
    """Returns the answer text (with an optional chart block) or None if this is not a calculator question."""
    q = _clean(question)
    order = []
    if re.search(r"\bcagr\b", q, re.I):
        order.append(_cagr)
    if re.search(r"\bemi\b", q, re.I):
        order.append(_emi)
    if re.search(r"\bsip\b", q, re.I):
        order.append(_sip)
    order.append(_growth)
    for fn in order:
        try:
            out = fn(q)
        except Exception:
            out = None
        if out:
            return out
    return None


if __name__ == "__main__":
    import sys
    print(try_calculate(" ".join(sys.argv[1:])) or "Not a calculator question.")