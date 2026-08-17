# Plumb eval harness

Deterministic golden harness. Core metrics are oracle-scored
(`forge/src/plumb_forge/score.py`) — no LLM judges for verdicts.

- `forge/tests/` — oracle, generator, scoring, contamination unit tests
  (`make validate`).
- `model/src/plumb_model/benchmark_eval.py` — head-to-head runner
  (concurrent, JSONL-resumable).
- Residual prose (correction quality) may use a pairwise LLM judge per
  CONTRACTS §7, never for core verdicts.