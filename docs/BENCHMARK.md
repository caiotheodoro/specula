# Specula — Benchmark Report

Held-out seed **777**, n=1000, 734 FLAG. Oracle-scored (`specula_forge.score`).
Adapters served via Modal L4 batch eval (`cloud/modal_eval.py`), not a
frontier HTTP API. Precision `*` is vacuous: the model emitted **zero**
taxonomy-valid violation objects, so caught/emitted is 1.0 with empty
numerator and denominator.

## Headline (held-out benchmark, seed 777, zero contamination)

| Model | Severity-w. recall | CRITICAL recall | HIGH recall | Precision | Parse |
|---|---|---|---|---|---|
| **Specula (Qwen3.8-27B QLoRA + RLVR)** | 0.000 | 0.000 | 0.000 | 1.000* | 0.998 |
| VL SFT baseline (`sft-final`, **val** n=196) | 0.000 | 0.000 | 0.000 | 1.000* | 0.786 |
| Qwen3.8-2.4T-A95B (frontier, zero-shot) | — | — | — | — | not run (no API key) |
| DeepSeek v4-flash (frontier, zero-shot) | — | — | — | — | not run (no API key) |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — | not run |

RLVR on seed 777 (ap-NzsgAhks5GebScnminKlZT): 997 PASS, 1 FLAG, 2 unparseable.
The FLAG used a non-taxonomy `type` string, so it scored as no catch.
Verdict accuracy 0.267 equals the 266/1000 expected-PASS rate of always
predicting PASS. Citation exact-match 1.0 is also vacuous (no matched keys).

SFT on seed-7 **val** (ap-9hYSRHlAxpADZBCh5Hdolv): parse 0.786, 100 PASS / 54
FLAG / 42 truncated. FLAG text is allergen-like prose (`Allergen Declaration`,
`critical`) that `to_forge_verdict` drops because CONTRACTS ids are
`ALLERGEN` / `CRITICAL`. Oracle recall stays 0. A seed-777 SFT slice was
aborted at 73/100 after a Modal client heartbeat drop; not used above.

## Run book

```sh
# RLVR champion, seed 777
modal run cloud/modal_eval.py --tasks forge/data/benchmark.jsonl \
  --adapter /checkpoints/rlvr-final --name rlvr-bench

# SFT baseline (held-out val from seed 7)
modal run cloud/modal_eval.py --tasks forge/data/val.jsonl \
  --adapter /checkpoints/sft-final --name sft-val

# Local HTTP (needs a GPU): make serve, then
# SPECULA_LLM_BASE_URL=http://127.0.0.1:8000/v1
cd model && uv run python -m specula_model.benchmark_eval \
  --tasks-file ../forge/data/benchmark.jsonl --model specula
```

Smoke (2 val tasks, ap-gyPoLSdf6NozH4MfyaDatI) and RLVR val n=196
(ap-pJlbPpvMVE0CpabUbvHPt1) also scored recall 0.000 / parse 1.0.

## Contamination

- Train/benchmark signature overlap = 0 (SHA-256 field-level).
- Leak probe: fires on leaked sets, 0 false-fires on clean (ROC).
