import config

CORRECT, INCORRECT, AMBIGUOUS = "CORRECT", "INCORRECT", "AMBIGUOUS"


def decide(scores: list[float], upper: float = config.UPPER_THRESHOLD, lower: float = config.LOWER_THRESHOLD) -> str:
    if not scores:
        return INCORRECT
    if max(scores) > upper:      # kam se kam ek doc upper se upar
        return CORRECT
    if max(scores) < lower:      # saare docs lower se neeche
        return INCORRECT
    return AMBIGUOUS