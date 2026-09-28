"""
String similarity functions.

We deliberately avoid a hard dependency on rapidfuzz / python-Levenshtein
so the pipeline runs anywhere with just numpy/pandas/scikit-learn
installed (see requirements.txt). If rapidfuzz IS available, we use it
for speed since blocking can generate many pairs; otherwise we fall back
to the pure-Python implementations below automatically.
"""
from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised when rapidfuzz absent
    _HAVE_RAPIDFUZZ = False


def levenshtein(a, b):
    if _HAVE_RAPIDFUZZ:
        return _RFLevenshtein.distance(a, b)
    return _levenshtein_pure(a, b)


def _levenshtein_pure(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def levenshtein_ratio(a, b):
    """1.0 == identical, 0.0 == totally different. Normalized by max length."""
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def token_sort_ratio(tokens_a, tokens_b):
    """Levenshtein ratio after sorting tokens -- neutralizes word-order swaps."""
    a = " ".join(sorted(tokens_a))
    b = " ".join(sorted(tokens_b))
    return levenshtein_ratio(a, b)


def prefix_match(a, b, k=3):
    if not a or not b:
        return 0.0
    return 1.0 if a[:k] == b[:k] else 0.0
