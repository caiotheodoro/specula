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
| GPT-4o (Bluesminds, n=20 parsed) | 0.000 | 0.000 | 0.000 | 1.000* | 1.000 |
| GPT-5-mini (Bluesminds, n=13) | 0.000 | 0.000 | 0.000 | 1.000* | 0.846 |
| GPT-5.2 Chat (Bluesminds, n=10 parsed) | 0.000 | 0.000 | 0.000 | 1.000* | 1.000 |
| Llama 3.2 11B Vision (Bluesminds, n=16) | 0.000 | 0.000 | 0.000 | 1.000* | 1.000 |
| Nemotron Nano VL 8B (Bluesminds, n=24) | 0.000 | 0.000 | 0.000 | 1.000* | 1.000 |
| Nemotron Nano 12B VL (Bluesminds, n=5) | 0.000 | 0.000 | 0.000 | 1.000* | 1.000 |
| Qwen3.8-2.4T-A95B (frontier, zero-shot) | — | — | — | — | not in catalog |
| DeepSeek v4-flash (frontier, zero-shot) | — | — | — | — | **text-only API** |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — | not run |

### GPT-5.6 Luna (`gpt-5.6-luna`, n=1000)

Parse 0.989, verdict accuracy **0.726** (988 FLAG / 1 PASS / 11 unparseable).
Oracle recall stays 0.000: 5239 emitted violation objects used free-text
types (`incorrect_percent_daily_values`, `calorie_rounding`, …) and mixed
severities (`high`/`major`/`critical`), none of which match CONTRACTS ids.
Luna only accepts default temperature (1); `temperature: 0` is HTTP 400.
Eval: `forge/data/eval-luna.jsonl` (gitignored), ~64 min, concurrency 6.


### Bluesminds (`https://api.bluesminds.com/v1`)

Catalog lists 137 ids. Many return HTTP 410 (end of life), 404, timeout, or
`No connected db`. Qwen3.5-397B/122B are EOL (2026-07). No Qwen3.8-27B or
Qwen3.8-2.4T id. Llama 3.2 90B VL, Gemma 4 31B, and Kimi K2.5 time out
(504). Llama 4 Maverick, Phi-4 MM, Mistral Large 3, Step 3.5 Flash are
EOL. GPT-4o-mini / GPT-5.5: `No connected db`.

Vision-live (image accepted, including a red 8x8 color check where
noted): GPT-4o, GPT-5-mini, GPT-5.2 Chat (flaky), Llama 3.2 11B VL
(color: red), Nemotron Nano 12B VL, Nemotron Nano VL 8B (color: red),
Nemotron Omni 30B. Concurrent evals hit HTTP 429/503; those errors are
skipped (not counted as parse misses). Parsed-only slices:

- **gpt-4o** n=20: 18 FLAG / 2 PASS, verdict acc 0.70, recall 0.000
  (prose types like `Added Sugars Declaration`).
- **gpt-5-mini** n=13: parse 0.846, 11 FLAG / 2 unparsed, verdict acc
  0.69, recall 0.000.
- **gpt-5.2-chat** n=10: all FLAG, verdict acc 0.80, recall 0.000.
- **meta/llama-3.2-11b-vision-instruct** n=16: 15 PASS / 1 FLAG, verdict
  acc 0.125, recall 0.000.
- **nvidia/llama-3.1-nemotron-nano-vl-8b-v1** n=24: all PASS, verdict
  acc 0.29, recall 0.000.
- **nvidia/nemotron-nano-12b-v2-vl** n=5: 3 PASS / 2 FLAG, verdict acc
  0.40, recall 0.000.
- Omni 30B: 0 parsed rows (503 worker limit).

These n are too small to rank against Luna n=1000. They show the same
CONTRACTS-id miss as Luna/SFT, and that Bluesminds is currently
rate-limited for a full seed-777 pass.

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

# Bluesminds VL probe (use BLUESMINDS_API_KEY; concurrency 1)
OPENAI_BASE_URL=https://api.bluesminds.com/v1 \
  SPECULA_LLM_API_KEY="$BLUESMINDS_API_KEY" OPENAI_API_KEY="$BLUESMINDS_API_KEY" \
  PYTHONPATH=src:../forge/src python -m specula_model.benchmark_eval \
  --tasks-file ../forge/data/benchmark.jsonl --model gpt-4o \
  --concurrency 1 --limit 20 --out ../forge/data/eval-bm-gpt-4o.jsonl
```

## Contamination

- Train/benchmark signature overlap = 0 (SHA-256 field-level).
- Leak probe: fires on leaked sets, 0 false-fires on clean (ROC).
