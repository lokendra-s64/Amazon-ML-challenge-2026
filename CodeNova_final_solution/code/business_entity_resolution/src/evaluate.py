"""
F_0.5 macro-average scoring, matching the challenge's evaluation exactly:

  F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed PER Source-1 entity,
  then averaged across all Source-1 entities (singletons included -- an
  entity with no true matches scores 1.0 if predicted empty, 0.0 if any
  match is (wrongly) predicted for it).

Use this on your own held-out validation split -- there is no ground
truth for the real test set.
"""
from __future__ import annotations


def f_half_score(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = 0.25
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def entity_f_half(predicted, truth):
    if not truth:
        # Singleton: correct iff predicted is also empty.
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_half_score(precision, recall)


def macro_f_half(predicted_by_entity, truth_by_entity):
    """Macro-average across the union of entities appearing in either dict."""
    all_ids = set(predicted_by_entity) | set(truth_by_entity)
    if not all_ids:
        return 0.0
    total = 0.0
    for eid in all_ids:
        pred = predicted_by_entity.get(eid, set())
        truth = truth_by_entity.get(eid, set())
        total += entity_f_half(pred, truth)
    return total / len(all_ids)
