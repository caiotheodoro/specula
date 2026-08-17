"""Benchmark eval: run a model adapter against the golden task set and score.

Usage:
  python -m specula_model.benchmark_eval --adapter-path adapters/champion \
      --tasks-file data/benchmark.jsonl [--model deepseek-v4-flash] [--concurrency 8]

Scores per CONTRACTS.md §2 via forge's scorer. Writes results.jsonl + report.

specula_forge must be on PYTHONPATH (pytest sets ../forge/src via
pyproject; otherwise PYTHONPATH=src:../forge/src).
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .schema import SYSTEM_PROMPT, VerdictOut, parse

SCORE_MODEL = None  # set by --score-model to use a frontier API for scoring


def _eval_payload(task) -> dict:
    """What the model sees: system prompt + real PNG pixels (not a hash)."""
    from specula_forge.generate import render_png
    return {"system": SYSTEM_PROMPT, "image": render_png(task.label)}


def _predict_one(task, model: str) -> str:
    payload = _eval_payload(task)
    if model == "oracle-mock":
        return task.expected.model_dump_json()
    raise NotImplementedError(
        "wire a real provider adapter (local vLLM/MLX or frontier API) in P3 "
        f"(system={len(payload['system'])} chars, image={len(payload['image'])} bytes)")


def run_benchmark(tasks_jsonl: Path, model: str, concurrency: int,
                  out: Path) -> list[dict]:
    from specula_forge.score import score_predictions
    from specula_forge.schema import Task

    tasks = [Task.model_validate_json(l) for l in
             tasks_jsonl.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["task_id"])
    remaining = [t for t in tasks if t.task_id not in done_ids]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        predictions = list(ex.map(
            lambda t: parse(_predict_one(t, model)),
            remaining))
    results = []
    with open(out, "a") as f:
        for t, p in zip(remaining, predictions):
            exp = VerdictOut(verdict=t.expected.verdict, violations=[
                {"type": v.type.value, "severity": v.severity.value,
                 "cfr": v.cfr, "observed": v.observed, "expected": v.expected,
                 "correction": v.correction} for v in t.expected.violations])
            row = {"task_id": t.task_id, "expected": exp.model_dump(),
                   "predicted": p.model_dump() if p else None,
                   "score": score_predictions(t.expected, _to_verdict(p))}
            f.write(json.dumps(row) + "\n")
            results.append(row["score"])
    all_scores = []
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "score" in rec:
            all_scores.append(rec["score"])
    print(json.dumps({"model": model, "summary": _summarize(all_scores or results)},
                     indent=2))
    return all_scores or results


def _to_verdict(p: VerdictOut | None):
    if p is None:
        return None
    from specula_forge.schema import Verdict, Violation
    return Verdict(verdict=p.verdict, violations=[
        Violation(type=v.type, severity=v.severity, cfr=v.cfr,
                  observed=v.observed, expected=v.expected,
                  correction=v.correction) for v in p.violations])


def _summarize(results: list[dict]) -> dict:
    from specula_forge.score import summarize
    return summarize(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--model", default="oracle-mock")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter path (ignored until a real adapter is wired)")
    args = ap.parse_args()
    run_benchmark(Path(args.tasks_file), args.model, args.concurrency,
                  Path(args.out))


if __name__ == "__main__":
    main()
