"""Retrieval Evaluator (paper Section 4.2).
The paper fine-tunes T5-large. Here a lightweight pretrained cross-encoder is used: it reads a
(question, document) pair together and returns a relevance score between 0 and 1.
"""
import math

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.EVALUATOR_MODEL)
    return _model


def score_pairs(question: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    raw = _get_model().predict([(question, t) for t in texts])
    raw = [float(x) for x in raw]
    # some versions already apply a sigmoid, others return raw logits; handle both
    if any(x < 0 or x > 1 for x in raw):
        raw = [1 / (1 + math.exp(-x)) for x in raw]
    return raw