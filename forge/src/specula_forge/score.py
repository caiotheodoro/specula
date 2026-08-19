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


def _severity_counts(exp_keys, pred_keys, severity: Severity) -> tuple[float, float]:
    total = sum(1.0 for _, s in exp_keys if s == severity)
    caught = sum(1.0 for t, s in exp_keys if s == severity and (t, s) in pred_keys)
    return caught, total


def score_predictions(expected: Verdict, predicted: Verdict | None) -> dict[str, float]:
    """Score one task: returns caught-weight, total-weight, fp count, parsed."""
    exp_keys = violation_keys(expected)
    pred_keys = violation_keys(predicted) if predicted is not None else set()
    crit_c, crit_t = _severity_counts(exp_keys, pred_keys, Severity.CRITICAL)
    high_c, high_t = _severity_counts(exp_keys, pred_keys, Severity.HIGH)
    parsed = predicted is not None
    total = sum(severity_weight(s) for _, s in exp_keys)
    if not parsed:
        return {"caught": 0.0, "total": total, "fp": 0.0, "parsed": 0.0,
                "citation_hits": 0.0, "citation_total": 0.0,
                "verdict_correct": 0.0, "emitted": 0.0,
                "critical_caught": 0.0, "critical_total": crit_t,
                "high_caught": 0.0, "high_total": high_t}
    caught = sum(severity_weight(s) for t, s in exp_keys if (t, s) in pred_keys)
    fp = sum(1 for t, s in pred_keys if (t, s) not in exp_keys)
    exp_cfr = {(v.type, v.severity): v.cfr for v in expected.violations}
    pred_cfr = {(v.type, v.severity): v.cfr for v in predicted.violations}
    citation_total = 0.0
    citation_hits = 0.0
    for key in exp_keys:
        if key in pred_keys:
            citation_total += 1.0
            if exp_cfr.get(key) == pred_cfr.get(key):
                citation_hits += 1.0
    verdict_correct = 1.0 if predicted.verdict == expected.verdict else 0.0
    return {"caught": caught, "total": total, "fp": fp, "parsed": 1.0,
            "citation_hits": citation_hits, "citation_total": citation_total,
            "verdict_correct": verdict_correct,
            "emitted": float(len(pred_keys)),
            "critical_caught": crit_c, "critical_total": crit_t,
            "high_caught": high_c, "high_total": high_t}


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
    cit_hits = sum(r["citation_hits"] for r in results)
    cit_total = sum(r["citation_total"] for r in results)
    emitted = sum(r.get("emitted", r["fp"] + r["citation_total"]) for r in results)
    matched = cit_total
    crit_c = sum(r.get("critical_caught", 0.0) for r in results)
    crit_t = sum(r.get("critical_total", 0.0) for r in results)
    high_c = sum(r.get("high_caught", 0.0) for r in results)
    high_t = sum(r.get("high_total", 0.0) for r in results)
    return {
        "n_tasks": float(n),
        "parse_rate": parsed,
        "severity_weighted_recall": caught_w / total_w if total_w else 1.0,
        "n_violation_tasks": float(n_violation_tasks),
        "false_positives_per_task": fp / n,
        "citation_exact_match": cit_hits / cit_total if cit_total else 1.0,
        "verdict_accuracy": sum(r.get("verdict_correct", 0.0) for r in results) / n,
        "precision": matched / emitted if emitted else 1.0,
        "critical_recall": crit_c / crit_t if crit_t else 1.0,
        "high_recall": high_c / high_t if high_t else 1.0,
    }




def reward(expected: Verdict, predicted: Verdict | None,
           fp_penalty: float = 0.3, verdict_bonus: float = 0.2,
           unparseable_reward: float = -1.0) -> float:
    """Scalar RLVR reward: severity-weighted recall minus FP penalty.

    Outcome-verifier only (no PRM). Unparseable output scores -1.0. A clean
    expected PASS has no recall ratio, so it is scored on verdict correctness.
    Citation exact-match is tracked in score_predictions, not this reward.
    """
    r = score_predictions(expected, predicted)
    if not r["parsed"]:
        return unparseable_reward
    if r["total"] == 0.0:
        return (1.0 if r["verdict_correct"] else -1.0) - fp_penalty * r["fp"]
    recall = r["caught"] / r["total"]
    bonus = verdict_bonus if r["verdict_correct"] else -verdict_bonus
    return recall - fp_penalty * r["fp"] + bonus


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