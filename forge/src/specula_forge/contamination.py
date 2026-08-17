"""Contamination monitoring per CONTRACTS.md §5.

Task signature = SHA-256 over sorted (field, value) pairs of the label's
ground-truth fields. A leak probe fires when any eval signature matches the
train set; must fire 1.0 on intentionally leaked sets and 0.0 on clean sets.
"""

from __future__ import annotations

from .schema import Task


def build_train_index(tasks: list[Task]) -> dict[str, str]:
    return {t.signature: t.task_id for t in tasks}


def probe(train_index: dict[str, str], eval_tasks: list[Task]) -> dict[str, list[str]]:
    """Return eval task_ids whose signature matches the train index."""
    leaked: list[str] = []
    for t in eval_tasks:
        if t.signature in train_index:
            leaked.append(t.task_id)
    return {"leaked": leaked, "n_leaked": len(leaked), "n_eval": len(eval_tasks)}


def split_overlap(train: list[Task], eval_tasks: list[Task]) -> int:
    """Count eval tasks sharing a signature with train (should be 0)."""
    index = build_train_index(train)
    return len(probe(index, eval_tasks)["leaked"])