"""Benchmark eval: run a model adapter against the golden task set and score.

Usage:
  python -m labelforge_model.benchmark_eval --adapter-path adapters/champion \
      --tasks-file data/benchmark.jsonl [--model deepseek-v4-flash] [--concurrency 8]

Scores per CONTRACTS.md §2 via forge's scorer. Writes results.jsonl + report.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .schema import VerdictOut, parse

SCORE_MODEL = None  # set by --score-model to use a frontier API for scoring


def _predict_one(prompt: str, model: str) -> str:
    if model == "oracle-mock":
        return prompt
    raise NotImplementedError(
        "wire a real provider adapter (local vLLM/MLX or frontier API) in P3")


def run_benchmark(tasks_jsonl: Path, model: str, concurrency: int,
                  out: Path) -> list[dict]:
    from labelforge_forge.score import score_predictions
    from labelforge_forge.schema import Task

    tasks = [Task.model_validate_json(l) for l in
             tasks_jsonl.read_text().splitlines() if l.strip()]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        predictions = list(ex.map(
            lambda t: parse(_predict_one(json.dumps({
                "image": t.image_bytes_sha256, "text": "review label"}), model)),
            tasks))
    results = []
    with open(out, "w") as f:
        for t, p in zip(tasks, predictions):
            exp = VerdictOut(verdict=t.expected.verdict, violations=[
                {"type": v.type.value, "severity": v.severity.value,
                 "cfr": v.cfr, "observed": v.observed, "expected": v.expected,
                 "correction": v.correction} for v in t.expected.violations])
            row = {"task_id": t.task_id, "expected": exp.model_dump(),
                   "predicted": p.model_dump() if p else None,
                   "score": score_predictions(t.expected, _to_verdict(p))}
            f.write(json.dumps(row) + "\n")
            results.append(row["score"])
    print(json.dumps({"model": model, "summary": _summarize(results)},
                     indent=2))


def _to_verdict(p: VerdictOut | None):
    if p is None:
        return None
    from labelforge_forge.schema import Verdict, Violation
    return Verdict(verdict=p.verdict, violations=[
        Violation(type=v.type, severity=v.severity, cfr=v.cfr,
                  observed=v.observed, expected=v.expected,
                  correction=v.correction) for v in p.violations])


def _summarize(results: list[dict]) -> dict:
    from labelforge_forge.score import summarize
    return summarize(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--model", default="oracle-mock")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()
    run_benchmark(Path(args.tasks_file), args.model, args.concurrency,
                  Path(args.out))


if __name__ == "__main__":
    main()