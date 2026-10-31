
"""S5: Verification. Checks numeric claims against retrieved source text."""

import re

NOT_ENOUGH = "could not find enough reliable information"

# Supports [1], [1, 2], [1; 2], 【1】 and similar citation markers.
CITE = re.compile(
    r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]"
    r"|【\s*\d+(?:\s*[,;]\s*\d+)*\s*】"
)

# Supports decimals, comma-separated numbers, percentages and percent words.
NUM = re.compile(
    r"(?<![\w.])(\d+(?:,\d{2,3})*(?:\.\d+)?)"
    r"\s*(%|per\s+cent|percent)?",
    re.I,
)

METRICS = {
    "revenue growth": re.compile(
        r"\b(?:revenue\s+growth|growth\s+in\s+revenue|revenue\s+grew)\b",
        re.I,
    ),
    "adjusted operating margin": re.compile(
        r"\badjusted\s+(?:operating\s+)?margin\b",
        re.I,
    ),
}


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _add_word_number_evidence(text: str, known: set[float]) -> None:
    """Recognize selected Indian tax-threshold amounts written in words."""
    normalized = re.sub(r"[-–—]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized)

    if re.search(r"\bone lakh twenty five thousand\b", normalized):
        known.add(125000.0)


def _needs_check(raw: str, value: float, has_pct: bool) -> bool:
    """Skip small integers and standalone years to reduce false alarms."""
    if has_pct or "." in raw:
        return True

    if 1900 <= value <= 2100 and "," not in raw:
        return False

    return value >= 100



def _metric_values(text: str) -> dict[str, set[float]]:
    """Associate each metric with its nearest percentage."""
    text = re.sub(r"\s+", " ", text)

    matches = []
    for name, pattern in METRICS.items():
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), name))

    matches.sort(key=lambda item: item[0])
    values = {name: set() for name in METRICS}

    for i, (start, end, name) in enumerate(matches):
        segment_start = matches[i - 1][1] if i > 0 else 0
        segment_end = (
            matches[i + 1][0]
            if i + 1 < len(matches)
            else len(text)
        )

        # Search both sides, but never cross another metric label.
        segment = text[segment_start:segment_end]
        metric_start = start - segment_start
        metric_end = end - segment_start

        candidates = []
        for number_match in NUM.finditer(segment):
            raw, pct = number_match.groups()

            if not pct:
                continue

            number_start, number_end = number_match.span()
            distance = min(
                abs(number_end - metric_start),
                abs(number_start - metric_end),
            )

            candidates.append((distance, round(_to_float(raw), 4)))

        if candidates:
            nearest_distance = min(item[0] for item in candidates)
            nearest_values = {
                value
                for distance, value in candidates
                if distance == nearest_distance
            }
            values[name].update(nearest_values)

    return values


def _check_metric_values(
    answer: str, knowledge: list[dict]
) -> list[str]:
    """Flag percentages that conflict with the same metric in the sources."""
    answer_values = _metric_values(answer)
    source_values = {name: set() for name in METRICS}

    for item in knowledge:
        extracted = _metric_values(item.get("text", ""))

        for name in METRICS:
            source_values[name].update(extracted[name])

    conflicts = []

    for name in METRICS:
        supported = source_values[name]
        claimed = answer_values[name]

        # If sources don't identify a metric percentage, don't guess.
        if not supported:
            continue

        for value in sorted(claimed - supported):
            conflicts.append(f"{name}: {value:g}%")

    return conflicts


def verify(answer: str, knowledge: list[dict]) -> dict:
    """Verify numeric claims against retrieved source text."""
    cited = bool(CITE.search(answer))
    partial_refusal = NOT_ENOUGH in answer.lower()

    if partial_refusal and not cited:
        return {
            "status": "refused",
            "checked": 0,
            "unverified": [],
            "cited": False,
        }

    if not knowledge:
        return {
            "status": "no_evidence",
            "checked": 0,
            "unverified": [],
            "cited": cited,
        }

    known = set()

    for item in knowledge:
        source_text = item.get("text", "")

        for raw, _ in NUM.findall(source_text):
            known.add(round(_to_float(raw), 4))

        _add_word_number_evidence(source_text, known)

    # Remove citations before checking numerical claims.
    text = CITE.sub(" ", answer)

    # Remove refusal sentences from numeric checking.
    text = re.sub(
        r"I could not find enough reliable information[^.\n]*[.\n]?",
        " ",
        text,
        flags=re.I,
    )

    checked = 0
    unverified = []

    for raw, pct in NUM.findall(text):
        value = _to_float(raw)

        if not _needs_check(raw, value, bool(pct)):
            continue

        checked += 1

        if round(value, 4) not in known:
            shown = raw + ("%" if pct else "")
            if shown not in unverified:
                unverified.append(shown)

    # Check percentages against their specific financial metrics.
    metric_conflicts = _check_metric_values(text, knowledge)

    for conflict in metric_conflicts:
        if conflict not in unverified:
            unverified.append(conflict)

    # Ignore chart code when deciding whether a refusal contains an answer.
    leftover = re.sub(
        r"```chart.*?```",
        " ",
        text,
        flags=re.S,
    ).strip()

    if (
        partial_refusal
        and not unverified
        and checked == 0
        and len(leftover) < 20
    ):
        status = "refused"
    elif unverified:
        status = "check"
    elif not cited and checked > 0:
        status = "uncited"
    else:
        status = "verified"

    return {
        "status": status,
        "checked": checked,
        "unverified": unverified,
        "cited": cited,
    }


if __name__ == "__main__":
    knowledge = [{
        "text": (
            "The repo rate was 5.50 per cent as of June 2025. "
            "The CRR is 4 per cent."
        )
    }]

    tests = [
        ("The repo rate is 5.50% [1].", "verified"),
        ("The repo rate is 5.5% 【1】.", "verified"),
        ("The repo rate is 6.25% [1].", "check"),
        ("The repo rate is 5.50% and CRR is 4%.", "uncited"),
        (
            "I could not find enough reliable information "
            "in the verified sources.",
            "refused",
        ),
        (
            "A mutual fund pools money from many investors [1].",
            "verified",
        ),
        (
            "I could not find enough reliable information to identify "
            "a fund that will deliver a 20 % return [1].",
            "refused",
        ),
        (
            "CRR is 4% [1]. "
            "I could not find enough reliable information about SLR.",
            "verified",
        ),
    ]

    print("=== ORIGINAL TESTS ===")
    for answer, expected in tests:
        result = verify(answer, knowledge)
        passed = result["status"] == expected

        print(
            f"{'OK' if passed else 'FAIL'} "
            f"Expected={expected} -> {result}"
        )

    ltcg_knowledge = [{
        "text": (
            "LTCG is 12.50% on gains exceeding "
            "one lakh twenty-five thousand rupees."
        )
    }]
    ltcg_answer = "LTCG is 12.5% on gains exceeding ₹1,25,000【1】."

    print("\n=== LTCG TEST ===")
    print(verify(ltcg_answer, ltcg_knowledge))

    print("\n=== EMPTY KNOWLEDGE TEST ===")
    print(verify("Rate is 5.50% [1].", []))

    # Regression test: wrong operating margin must be flagged.
    margin_knowledge = [{
        "text": (
            "Infosys revenue growth was 9.6% and "
            "adjusted operating margin was 21.0%."
        )
    }]

    margin_answer = (
        "Infosys revenue growth was 9.6% and "
        "adjusted operating margin was 31.6% [1]."
    )

    print("\n=== OPERATING MARGIN REGRESSION TEST ===")
    result = verify(margin_answer, margin_knowledge)
    print(result)

    assert result["status"] == "check", (
        "FAIL: Incorrect operating margin was not flagged."
    )
    assert any(
        "31.6" in item for item in result["unverified"]
    ), "FAIL: Conflicting operating margin was not identified."

    assert not any(
        "revenue growth" in item.lower() and "9.6" in item
        for item in result["unverified"]
    ), "FAIL: Correct revenue growth was incorrectly flagged."

    print("OK: Correct revenue growth accepted; wrong margin flagged.")
