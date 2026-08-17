"""Compute a simple retrieval confidence score from tool contexts."""

from __future__ import annotations

import re


_SCORE_RE = re.compile(r"Score:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def compute_retrieval_confidence(contexts: list[str]) -> float:
    """Return mean retrieval score across contexts, clamped to [0, 1].

    Contexts are formatted tool outputs that may include a ``Score:`` line.
    Empty contexts yield ``0.0``.
    """
    if not contexts:
        return 0.0

    scores: list[float] = []
    for context in contexts:
        match = _SCORE_RE.search(context)
        if match:
            scores.append(float(match.group(1)))

    if not scores:
        # Presence of retrieved text without explicit scores → mid confidence
        return 0.5 if any(c.strip() for c in contexts) else 0.0

    mean = sum(scores) / len(scores)
    return max(0.0, min(1.0, mean))
