"""Benchmark scoring per CONTRACTS.md §2.

R_w = Σ w·caught / Σ w over violation-bearing tasks. A violation is caught iff
the model emits a violation whose type matches (critical classes must also
match severity). Each emitted violation maps to at most one oracle violation;
unmatched = false positive. Unparseable output = parse miss.
"""

from __future__ import annotations

from .schema import SEVERITY_WEIGHTS, Severity, Verdict, ViolationType
from .verify import violation_keys


def severity_weight(severity: Severity) -> float:
    return SEVERITY_WEIGHTS[severity]


def score_predictions(expected: Verdict, predicted: Verdict) -> dict[str, float]:
    """Score one task: returns caught-weight, total-weight, fp count, parsed."""
    parsed = predicted is not None
    if not parsed:
        return {"caught": 0.0, "total": 0.0, "fp": 0.0, "parsed": 0.0}
    exp_keys = violation_keys(expected)
    pred_keys = violation_keys(predicted)
    total = sum(severity_weight(s) for _, s in exp_keys)
    caught = sum(severity_weight(s) for t, s in exp_keys if (t, s) in pred_keys)
    fp = sum(1 for t, s in pred_keys if (t, s) not in exp_keys)
    return {"caught": caught, "total": total, "fp": fp, "parsed": 1.0}


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-task dicts into benchmark metrics."""
    n = len(results)
    if n == 0:
        return {}
    parsed = sum(r["parsed"] for r in results) / n
    total_w = sum(r["total"] for r in results)
    caught_w = sum(r["caught"] for r in results)
    fp = sum(r["fp"] for r in results)
    n_violation_tasks = sum(1 for r in results if r["total"] > 0)
    return {
        "n_tasks": float(n),
        "parse_rate": parsed,
        "severity_weighted_recall": caught_w / total_w if total_w else 1.0,
        "n_violation_tasks": float(n_violation_tasks),
        "false_positives_per_task": fp / n,
    }


def class_recall(expected: list[Verdict], predicted: list[Verdict],
                 cls: ViolationType) -> float:
    """Per-class recall: fraction of tasks with `cls` caught by the model."""
    denom = 0
    caught = 0
    for e, p in zip(expected, predicted):
        if cls in {v.type for v in e.violations}:
            denom += 1
            if p is not None and cls in {v.type for v in p.violations}:
                caught += 1
    return caught / denom if denom else 1.0