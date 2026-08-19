# Specula — Benchmark Report

Held-out seed **777**, n=1000, 734 FLAG. Oracle-scored (`specula_forge.score`).
Precision `*` is vacuous: **zero** taxonomy-valid violation objects
(`ALLERGEN`, `CRITICAL`, …), so caught/emitted is 1.0 with an empty
denominator. Do not treat that as skill.

## Headline (held-out benchmark, seed 777, zero contamination)

| Model | Severity-w. recall | CRITICAL recall | HIGH recall | Precision | Parse |
|---|---|---|---|---|---|
| **Specula (Qwen3.8-27B QLoRA + RLVR)** | 0.000 | 0.000 | 0.000 | 1.000* | 0.998 |
| VL SFT baseline (`sft-final`, **val** n=196) | 0.000 | 0.000 | 0.000 | 1.000* | 0.786 |
| GPT-5.6 Luna (zero-shot, vision) | 0.000 | 0.000 | 0.000 | 1.000* | 0.989 |
| Qwen3.8-2.4T-A95B (frontier, zero-shot) | — | — | — | — | not run |
| DeepSeek v4-flash (frontier, zero-shot) | — | — | — | — | **text-only API** |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — | not run |

### GPT-5.6 Luna (`gpt-5.6-luna`, n=1000)

Parse 0.989, verdict accuracy **0.726** (988 FLAG / 1 PASS / 11 unparseable).
Oracle recall stays 0.000: 5239 emitted violation objects used free-text
types (`incorrect_percent_daily_values`, `calorie_rounding`, …) and mixed
severities (`high`/`major`/`critical`), none of which match CONTRACTS ids.
Luna only accepts default temperature (1); `temperature: 0` is HTTP 400.
Eval: `forge/data/eval-luna.jsonl` (gitignored), ~64 min, concurrency 6.

### DeepSeek v4-flash

`https://api.deepseek.com/chat/completions` with `deepseek-v4-flash` accepts
text (`Hi.`) and rejects `image_url` with HTTP 400
`unknown variant image_url, expected text`. No label-PNG score is possible
on the hosted API; OCR-then-text is a different task and was not run.

### Specula RLVR / SFT (unchanged)

RLVR (ap-NzsgAhks5GebScnminKlZT): 997 PASS, 1 FLAG, 2 unparseable. Verdict
accuracy 0.267 is the 266/1000 expected-PASS rate of always predicting PASS.

SFT val n=196 (ap-9hYSRHlAxpADZBCh5Hdolv): parse 0.786, 100 PASS / 54 FLAG /
42 truncated. Same taxonomy-id miss as Luna.

## Run book

```sh
modal run cloud/modal_eval.py --tasks forge/data/benchmark.jsonl \
  --adapter /checkpoints/rlvr-final --name rlvr-bench

# GPT-5.6 Luna (needs OPENAI_API_KEY; do not commit keys)
OPENAI_BASE_URL=https://api.openai.com/v1 \
  PYTHONPATH=src:../forge/src python -m specula_model.benchmark_eval \
  --tasks-file ../forge/data/benchmark.jsonl --model gpt-5.6-luna \
  --concurrency 6 --out ../forge/data/eval-luna.jsonl
```

## Contamination

- Train/benchmark signature overlap = 0 (SHA-256 field-level).
- Leak probe: fires on leaked sets, 0 false-fires on clean (ROC).
