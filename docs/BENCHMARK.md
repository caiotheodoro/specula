# Specula — Benchmark Report

_Template. Populated after P6 (head-to-head vs frontier)._

## Headline (held-out benchmark, seed 777, zero contamination)

| Model | Severity-w. recall | CRITICAL recall | HIGH recall | Precision | Parse |
|---|---|---|---|---|---|
| **Specula (Qwen3.8-27B QLoRA + RLVR)** | — | — | — | — | — |
| Qwen3.8-2.4T-A95B (frontier, zero-shot) | — | — | — | — | — |
| DeepSeek v4-flash (frontier, zero-shot) | — | — | — | — | — |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — | — |

## Run book

```sh
cd forge && uv run python -m specula_forge.cli pilot --seed 777 --n 1000 --out data/benchmark.jsonl
cd model && uv run python -m specula_model.benchmark_eval --tasks-file ../forge/data/benchmark.jsonl \
  --model specula --adapter-path adapters/champion
```

## Contamination

- Train/benchmark signature overlap = 0 (SHA-256 field-level).
- Leak probe: fires on leaked sets, 0 false-fires on clean (ROC).