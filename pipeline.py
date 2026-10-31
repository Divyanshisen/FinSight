"""Poora CRAG flow (paper Algorithm 1) + question decomposition + calculator + baseline RAG."""
import re
import config
from retriever import retrieve
from evaluator import score_pairs
from trigger import decide, CORRECT, INCORRECT, AMBIGUOUS
from refine import refine_documents
from web_search import web_knowledge
from generator import generate
from decompose import decompose
from calculator import try_calculate
from verifier import verify
from guard import guard_reply

NOT_ENOUGH = "could not find enough reliable information"
CHART_BLOCK = re.compile(r"\n*```chart\s*\{.*?\}\s*```", re.S)

# Trend / multi-year sawalon me ek chunk kaafi nahi hota, isliye web bhi chalao
TREND_PATTERN = re.compile(
    r"over the (last|past)|last \d+ years|past \d+ years|trend|how has|how have|"
    r"changed|history|historical|from (19|20)\d\d|since (19|20)\d\d|year[- ]on[- ]year",
    re.I,
)

FINANCIAL_METRICS = {
    "revenue growth": re.compile(
        r"\brevenue\s+growth\b|\bgrowth\s+in\s+revenue\b|\brevenue\s+grew\b",
        re.I,
    ),
    "adjusted operating margin": re.compile(
        r"\badjusted\s+(?:operating\s+)?margin\b",
        re.I,
    ),
}


def _metric_values(text: str) -> dict[str, set[float]]:
    """Extract values only when explicitly linked to a metric."""
    text = re.sub(r"\s+", " ", text)
    result = {name: set() for name in FINANCIAL_METRICS}

    patterns = {
        "revenue growth": re.compile(
            r"(?:(\d+(?:\.\d+)?)\s*%\s*revenue\s+growth"
            r"|revenue\s+growth[^%\n]{0,30}?(\d+(?:\.\d+)?)\s*%)",
            re.I,
        ),
        "adjusted operating margin": re.compile(
            r"(?:(\d+(?:\.\d+)?)\s*%\s*adjusted\s+operating\s+margin"
            r"|adjusted\s+operating\s+margin[^%\n]{0,30}?(\d+(?:\.\d+)?)\s*%)",
            re.I,
        ),
    }

    for name, pattern in patterns.items():
        for match in pattern.finditer(text):
            raw = next((g for g in match.groups() if g is not None), None)
            if raw is not None:
                result[name].add(float(raw))

    return result


def _financial_metric_check(
    answer: str, knowledge: list[dict]
) -> list[str]:
    """Validate each claimed metric against explicitly labelled source values."""
    answer_values = _metric_values(answer)
    source_values = {name: set() for name in FINANCIAL_METRICS}

    for item in knowledge:
        extracted = _metric_values(item.get("text", ""))
        for name in FINANCIAL_METRICS:
            source_values[name].update(extracted[name])

    problems = []
    for name, claimed_values in answer_values.items():
        supported_values = source_values[name]

        if claimed_values and not supported_values:
            problems.append(f"{name}: source value not established")
        else:
            for value in claimed_values - supported_values:
                problems.append(f"{name}: unsupported value {value:g}%")

    return problems


import json

NOT_VERIFIED_MSG = (
    "I could not verify the financial figures against "
    "their corresponding metrics in the retrieved sources."
)


def _number_strings(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,]*(?:\.\d+)?", text))


def _to_floats(text: str) -> set[float]:
    values = set()
    for n in _number_strings(text):
        try:
            values.add(float(n.replace(",", "")))
        except ValueError:
            pass
    return values


def _chart_is_supported(chart_text: str, source_floats: set[float]) -> bool:
    """A chart is kept only if every series value appears in the sources."""
    match = re.search(r"\{.*\}", chart_text, re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
        values = [v for s in data.get("series", []) for v in s.get("values", [])]
    except Exception:
        return False
    if not values:
        return False
    return all(
        any(abs(float(v) - s) < 1e-9 for s in source_floats) for v in values
    )


def _clean_answer(answer: str, knowledge: list):
    """Verify the answer and remove only the unsupported parts."""
    source_text = " ".join(item.get("text", "") for item in knowledge)
    source_floats = _to_floats(source_text)

    generic = verify(answer, knowledge)
    metric_problems = _financial_metric_check(answer, knowledge)

    if generic.get("status") == "refused":
        return answer, generic

        # Problems that name no number cannot be localised,
    # so fall back to strict mode.
    if any(not _number_strings(p) for p in metric_problems):
        return NOT_VERIFIED_MSG, {
            **generic,
            "status": "check",
            "unverified": metric_problems,
        }

    flagged = set()
    for item in (generic.get("unverified", []) or []):
        flagged |= _number_strings(str(item))

    for problem in metric_problems:
        flagged |= _number_strings(problem)

    # Re-check each flagged number: it stays "bad" only if
    # it is not in the sources.
    bad_numbers = set()

    for n in flagged:
        try:
            value = float(n.replace(",", ""))
        except ValueError:
            continue

        if not any(abs(value - s) < 1e-9 for s in source_floats):
            bad_numbers.add(n)
    charts = [m.group(0) for m in CHART_BLOCK.finditer(answer)]
    text_only = CHART_BLOCK.sub("", answer)

    # Remove only the lines that contain an unverified number.
    kept_lines = []
    for line in text_only.splitlines():
        has_bad = any(
            re.search(rf"(?<![\d.,]){re.escape(n)}(?!\d)", line)
            for n in bad_numbers
        )
        if not has_bad:
            kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()

    if bad_numbers and len(re.sub(r"[\W_]+", "", cleaned)) < 30:
        return NOT_VERIFIED_MSG, {**generic, "status": "check",
                                  "unverified": sorted(bad_numbers)}

    # Keep a chart only when all its values are found in the sources.
    good_charts = [c for c in charts if _chart_is_supported(c, source_floats)]
    if good_charts and not bad_numbers:
        cleaned = cleaned + "\n\n" + "\n\n".join(good_charts)

    status = generic.get("status", "verified")
    if bad_numbers:
        status = "partial"
        cleaned += ("\n\nNote: some figures could not be verified "
                    "in the sources and were removed.")

    return cleaned, {**generic, "status": status,
                     "unverified": sorted(bad_numbers),
                     "chart_kept": bool(good_charts and not bad_numbers)}


def _is_trend_question(question: str) -> bool:
    return bool(TREND_PATTERN.search(question))


def _crag_single(question: str) -> dict:
    docs = retrieve(question, config.TOP_K)                          # 1. retrieve
    scores = score_pairs(question, [d["text"] for d in docs])        # 2. evaluate
    action = decide(scores)                                          # 3. trigger

    # Trend sawal: PDF relevant ho tab bhi multi-year numbers ke liye web chahiye
    if action == CORRECT and _is_trend_question(question):
        action = AMBIGUOUS

    k_in, k_ex, query = [], [], None
    if action in (CORRECT, AMBIGUOUS):
        k_in = refine_documents(question, docs, scores)              # 4. refinement
    if action in (INCORRECT, AMBIGUOUS):
        query, k_ex = web_knowledge(question)
    if action == INCORRECT and not k_ex:           # web gave nothing: keep local evidence
        k_in = refine_documents(question, docs, scores)

    if action == INCORRECT and k_ex:
        action = AMBIGUOUS                        # 5. web search
    return {"question": question, "action": action, "scores": scores,
            "docs": docs, "k_in": k_in, "k_ex": k_ex, "web_query": query}


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        if it["text"] not in seen:
            seen.add(it["text"])
            out.append(it)
    return out


def _no_search_result(answer: str, action: str, question: str, sources: list, verification: dict) -> dict:
    """Result shape for answers that need no retrieval (guard refusal, calculator)."""
    return {"answer": answer, "action": action, "scores": [], "docs": [], "k_in": [], "k_ex": [],
            "knowledge": [], "web_query": None, "sub_questions": [question], "parts": [],
            "sources": sources, "verification": verification}



_PRIVATE = re.compile(
    r"\b(what|show|tell|check|give|fetch|display|read)\b.{0,25}\b(my|mine)\b.{0,25}\b(balance|account number|transactions?|history|statement|passbook|portfolio value|password|otp|pin)\b"
    r"|\bmy\b.{0,15}\b(balance|transaction history|account number)\b"
    r"|\b(personal|private)\b.{0,15}\b(bank account|account balance|balance)\b", re.I)

def crag(question: str) -> dict:
    # Advice / prediction questions
    refusal = guard_reply(question)


    if refusal:
        return _no_search_result(
            refusal, "REFUSED", question, [],
            {"status": "refused", "checked": 0,
             "unverified": [], "cited": False}
        )

    # Personal / private account questions: refuse before any retrieval
    if _PRIVATE.search(question):
        return _no_search_result(
            "I cannot access your personal bank or account information. "
            "Please check your bank's official app or website.",
            "REFUSED", question, [],
            {"status": "refused", "checked": 0, "unverified": [], "cited": False}
        )

    # Calculator
    calc = try_calculate(question)
    if calc:
        return _no_search_result(
            calc, "CALCULATOR", question,
            ["Calculated locally from the numbers in your question"],
            {"status": "calculated", "checked": 0,
             "unverified": [], "cited": False}
        )
    subs = decompose(question)                                       # compound sawal todo
    parts = [_crag_single(q) for q in subs]                          # har part pe CRAG

    k_in = _dedupe([k for p in parts for k in p["k_in"]])
    k_ex = _dedupe([k for p in parts for k in p["k_ex"]])
    knowledge = k_in + k_ex
    for item in knowledge:
        print("\nSOURCE:", item.get("source", "unknown"))
        print(item.get("text", "")[:3000])
    actions = {p["action"] for p in parts}
    action = actions.pop() if len(actions) == 1 else AMBIGUOUS       # mixed -> AMBIGUOUS

    answer = generate(question, knowledge)         

    print("RAW GENERATED ANSWER:", answer)                 # 6. generate

    # Sufficiency fallback: PDF "relevant" tha par jawab nahi mila -> web se dobara koshish
    if not k_ex and NOT_ENOUGH in answer.lower() and not any(p["web_query"] for p in parts):
        for p in parts:
            query, ex = web_knowledge(p["question"])
            p["web_query"], p["k_ex"], p["action"] = query, ex, AMBIGUOUS
        k_ex = _dedupe([k for p in parts for k in p["k_ex"]])
        if k_ex:
            knowledge = k_in + k_ex
            action = AMBIGUOUS
            answer = generate(question, knowledge)
            print("RAW GENERATED ANSWER:", answer)

    answer, verification = _clean_answer(answer, knowledge)          # 7. verify, chart only if numbers check out

    return {"answer": answer, "action": action, "verification": verification,
            "scores": [s for p in parts for s in p["scores"]],
            "docs": [d for p in parts for d in p["docs"]],
            "k_in": k_in, "k_ex": k_ex, "knowledge": knowledge,
            "web_query": " | ".join(p["web_query"] for p in parts if p["web_query"]) or None,
            "sub_questions": subs, "parts": parts,
            "sources": sorted({k["source"] for k in knowledge})}


def baseline_rag(question: str, k: int = 5) -> dict:
    """Normal RAG: docs seedha LLM ko, koi correction nahi (comparison ke liye)."""
    docs = retrieve(question, k, mode="dense")
    answer = generate(question, docs)
    return {"answer": answer, "sources": sorted({d["source"] for d in docs})}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the current repo rate?"
    r = crag(q)
    print("SUB-QUESTIONS:", r["sub_questions"])
    for p in r["parts"]:
        print(f'  - {p["action"]:9} max={max(p["scores"], default=0):.2f} | {p["question"]}')
    print("OVERALL ACTION:", r["action"])
    if r["web_query"]:
        print("WEB QUERY:", r["web_query"])
    print("ANSWER:", r["answer"])
    print("VERIFICATION:", r["verification"]["status"])
    print("SOURCES:", r["sources"])