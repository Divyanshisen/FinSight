import re
import config
from evaluator import score_pairs


def split_strips(text: str, n: int = config.SENTENCES_PER_STRIP) -> list[str]:
    """Decompose: split a document into small strips (n sentences each)."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return [" ".join(sents[i:i + n]) for i in range(0, len(sents), n)]


def select_strips(question: str, strips: list[dict], min_keep: int = 0) -> list[dict]:
    """Select relevant strips, preserving matching official RBI rate panels."""
    if not strips:
        return []

    q = question.lower()
    scores = score_pairs(question, [s["text"] for s in strips])
    ranked = sorted(range(len(strips)), key=lambda i: scores[i], reverse=True)
    
        # list-type questions ("types of ...", "categories", "steps") need more strips, otherwise some items are lost
    is_list_q = bool(re.search(r"\b(types?|kinds?|categories|classif\w*|list|steps|components|features)\b", q))
    top_k = config.STRIP_TOP_K + (3 if is_list_q else 0)
    thr = config.STRIP_THRESHOLD * (0.5 if is_list_q else 1)
    keep = [i for i in ranked if scores[i] >= thr][:top_k]

    # Targeted exception: preserve the matching RBI homepage rate panel.
    # Do not lower the relevance threshold for other sources.
    for i, strip in enumerate(strips):
        source = strip.get("source", "").lower()
        text = " ".join(strip.get("text", "").lower().split())

        is_rbi_home = (
            "rbi.org.in/home.aspx" in source
            or "rbi.org.in/home" in source
        )

        if not is_rbi_home:
            continue

        wants_reserve_ratio = any(
            term in q for term in ("slr", "crr", "reserve ratio", "reserve ratios")
        )
        wants_policy_rate = any(
            term in q for term in (
                "repo rate", "policy rate", "reverse repo",
                "standing deposit facility", "marginal standing facility",
                "bank rate"
            )
        )

        matches_reserve_panel = (
            "reserve" in text
            and ("slr" in text or "crr" in text)
        )
        matches_policy_panel = (
            "policy repo rate" in text
            or ("standing deposit facility" in text and "repo" in text)
        )

        if (
            (wants_reserve_ratio and matches_reserve_panel)
            or (wants_policy_rate and matches_policy_panel)
        ):
            if i not in keep:
                keep.append(i)

    if not keep and min_keep:
        keep = ranked[:min_keep]

    # Preserve original document order and avoid duplicates.
    return [strips[i] for i in sorted(set(keep))]


def refine_documents(question: str, docs: list[dict], doc_scores: list[float]) -> list[dict]:
    """Refine the retrieved documents -> k_in (internal knowledge)."""
    useful = [d for d, s in zip(docs, doc_scores) if s >= config.LOWER_THRESHOLD] or docs
    strips = []
    for d in useful:
        if d.get("image"):
            # chart/visual description: keep it whole, splitting would lose its meaning
            strips.append({"text": d["text"], "source": d["source"], "image": d["image"]})
        else:
            strips += [{"text": st, "source": d["source"], "image": ""} for st in split_strips(d["text"])]
    return select_strips(question, strips)